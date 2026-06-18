"""
Iterative within-round search loop.

Each call to run_within_round_search() implements one full round:

  1. Seed: call seeder.seed(ctx) to produce the initial candidate batch.
  2. Screen each candidate (predicate synthesis fast-path or LLM → sandbox → encode).
     Hard failures go to registry.hard_rejected and are permanently banned.
  3. CV-score every valid candidate on the search set.
  4. Call evolver.evolve(ctx) to propose the next iteration's candidates.
  5. Repeat steps 2-4 until the search budget is exhausted or stagnation triggers.
  6. Select winner by highest complexity/dependency-adjusted CV score.
  7. Validate winner on the fixed held-out validation set.
  8. Register soft-rejected (CV-scored, non-winner) candidates in the registry.

Predicate synthesis fast path
------------------------------
If a CandidateFactor has compute_code already set (e.g. from RandomSeeder),
the LLM synthesiser is skipped and the code goes directly to sandbox validation.

Stagnation detection
---------------------
Controlled by disc_cfg.stagnation_patience (0 = disabled).
If the best adjusted score does not improve by at least stagnation_epsilon for
stagnation_patience consecutive iterations, the round terminates early.

Data-split helper
-----------------
split_participants() is called once before round 1 in pipeline.py.
The same search/validation split is reused across all rounds.
"""

import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.evaluation import compute_nmi, compute_correlation
from src.analysis.factor_encoder import encode_discrete_factor, encode_continuous_factor, encode_factor
from src.analysis.model_comparison import (
    CVScore,
    MultiOutcomeCVScore,
    build_extended_formula,
    evaluate_on_held_out,
    replace_formula_outcome,
    score_candidate_cv,
)
from src.discovery.contrast_searcher import generate_level_vs_rest_contrasts
from src.discovery.factor_registry import CandidateFactor, DiscoveredFactor, FactorRegistry
from src.discovery.llm_client import LLMClient
from src.discovery.predicate_synthesizer import synthesize_predicate
from src.discovery.sandbox import run_predicate
from src.discovery.strategies.base import (
    EvolutionStrategy,
    ScoredCandidate,
    SearchContext,
    SeedingStrategy,
)
from src.utils.config import BenchmarkConfig


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    round_num: int
    winner: Optional[ScoredCandidate]
    winner_val_series: Optional[pd.Series]
    all_scored: List[ScoredCandidate]
    hard_rejected_in_round: List[CandidateFactor]
    validation_improvement: Optional[float]           # primary outcome (backward compat)
    accepted: bool
    validation_improvements: Optional[Dict[str, float]] = None  # per-outcome


# ---------------------------------------------------------------------------
# Participant split
# ---------------------------------------------------------------------------

def split_participants(
    df: pd.DataFrame,
    validation_fraction: float,
    seed: int,
    participant_col: str = "participant_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into a search set and a fixed validation set by participant_id.
    Called once before round 1; reused across all rounds.
    """
    pids = sorted(df[participant_col].unique())
    rng = np.random.RandomState(seed)
    shuffled = list(pids)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * validation_fraction))
    val_pids    = set(shuffled[:n_val])
    search_pids = set(shuffled[n_val:])
    return (
        df[df[participant_col].isin(search_pids)].reset_index(drop=True),
        df[df[participant_col].isin(val_pids)].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_factor_column(
    candidate: CandidateFactor,
    df: pd.DataFrame,
    config: BenchmarkConfig,
    min_level_count: int = 1,
) -> Optional[pd.Series]:
    sandbox = run_predicate(
        predicate_code=candidate.compute_code,
        df=df,
        factor_type=candidate.factor_type,
        window_width=candidate.window_width,
        timeout_seconds=config.discovery.sandbox_timeout_seconds,
        backend=config.discovery.sandbox_backend,
        depends_on=candidate.depends_on,
    )
    if not sandbox.success:
        return None
    series, is_valid, _ = encode_factor(
        raw_values=sandbox.values,
        df_index=df.index,
        declared_levels=candidate.levels,
        min_level_count=min_level_count,
        factor_class=candidate.factor_class,
    )
    return series if is_valid else None


def _check_decomposition(
    candidate_series: pd.Series,
    df: pd.DataFrame,
    discovered_names: List[str],
) -> Tuple[str, Optional[List[str]]]:
    available = [n for n in discovered_names if n in df.columns]
    if len(available) < 1:
        return "none", None

    for fi_name in available:
        sub = pd.DataFrame({"_z": candidate_series, "_fi": df[fi_name]}).dropna()
        if sub.empty or sub["_fi"].nunique() < 2:
            continue
        cell_z = sub.groupby("_fi")["_z"].nunique()
        if not cell_z.le(1).all():
            continue
        n_cells = cell_z.shape[0]
        n_z = sub["_z"].nunique()
        if n_cells == n_z:
            return "bijection", [fi_name]
        else:
            return "coarsening", [fi_name]

    if len(available) < 2:
        return "none", None
    for fi_name, fj_name in combinations(available, 2):
        sub = pd.DataFrame({
            "_z": candidate_series, "_fi": df[fi_name], "_fj": df[fj_name]
        }).dropna()
        if sub.empty or sub[["_fi", "_fj"]].drop_duplicates().shape[0] < 2:
            continue
        cell_z = sub.groupby(["_fi", "_fj"])["_z"].nunique()
        if not cell_z.le(1).all():
            continue
        n_cells = cell_z.shape[0]
        n_z = sub["_z"].nunique()
        if n_cells == n_z:
            return "bijection", [fi_name, fj_name]
        else:
            return "coarsening", [fi_name, fj_name]
    return "none", None


def _n_params(candidate: CandidateFactor) -> int:
    if candidate.factor_class == "continuous":
        return 1
    return max(1, len(candidate.levels) - 1)


def _collect_reference_series(
    df: pd.DataFrame,
    base_factors: list,
    registry: FactorRegistry,
    factor_class: str,
) -> List[pd.Series]:
    """
    Return series for all known factors of the same class as the candidate.
    Covers observable base factors and already-discovered factors.
    """
    series_list: List[pd.Series] = []
    for bf in base_factors:
        col_class = "continuous" if bf.dtype == "continuous" else "discrete"
        if col_class == factor_class and bf.name in df.columns:
            series_list.append(df[bf.name])
    for disc in registry.discovered:
        if disc.candidate.factor_class == factor_class and disc.column_name in df.columns:
            series_list.append(df[disc.column_name])
    return series_list


def _compute_novelty_score(
    series: pd.Series,
    factor_class: str,
    reference_series: List[pd.Series],
) -> float:
    """
    Novelty of `series` relative to the reference pool: 1 − max_similarity.
    Similarity is NMI for discrete factors, |Spearman ρ| for continuous.
    Returns 1.0 when the pool is empty (nothing to compare against).
    """
    if not reference_series:
        return 1.0
    if factor_class == "discrete":
        max_sim = max(compute_nmi(series, ref) for ref in reference_series)
    else:
        max_sim = max(compute_correlation(series, ref) for ref in reference_series)
    return float(1.0 - max_sim)


def _compute_adjusted_score(
    mean_ll: float,
    se_ll: float,
    candidate: CandidateFactor,
    stability_weight: float,
    complexity_exponent: float,
    depends_on_exponent: float,
    novelty_score: float = 0.0,
    novelty_weight: float = 0.0,
) -> float:
    raw    = mean_ll - stability_weight * se_ll
    n_par  = _n_params(candidate)
    n_deps = max(1, len(candidate.depends_on))
    return raw / (n_par ** complexity_exponent * n_deps ** depends_on_exponent) \
           + novelty_weight * novelty_score


def adjusted_score(
    sc: ScoredCandidate,
    stability_weight: float,
    complexity_exponent: float,
    depends_on_exponent: float,
) -> float:
    """Return the pre-computed adjusted score (kept for backward compat)."""
    return sc.adjusted_score


def _select_winner(all_scored: List[ScoredCandidate]) -> Optional[ScoredCandidate]:
    if not all_scored:
        return None
    return max(all_scored, key=lambda sc: sc.adjusted_score)


def _score_precomputed_candidate(
    candidate: CandidateFactor,
    series: pd.Series,
    search_df: pd.DataFrame,
    config: BenchmarkConfig,
    registry: FactorRegistry,
) -> Tuple[Optional[ScoredCandidate], Optional[str]]:
    """
    Score an already-computed candidate series with the standard CV objective.

    Ordinary synthesized candidates and automatically derived level contrasts
    use this same path, so contrasts are judged against the same baseline as
    their parent factors.
    """
    disc_cfg = config.discovery
    ref_series = _collect_reference_series(
        search_df, config.base_factors, registry, candidate.factor_class
    )
    novelty = _compute_novelty_score(series, candidate.factor_class, ref_series)

    col_name = candidate.name
    search_with_col = search_df.copy()
    search_with_col[col_name] = series

    formula_null = registry.get_current_formula()
    formula_alt = build_extended_formula(
        formula_null,
        col_name,
        factor_class=candidate.factor_class,
    )

    outcome_defs = config.outcome_variable_defs
    per_outcome_cvs: Dict[str, CVScore] = {}
    current_lhs = formula_null.split("~")[0].strip()
    for od in outcome_defs:
        lhs = current_lhs if od.name in current_lhs else od.name
        f_null = replace_formula_outcome(formula_null, lhs)
        f_alt = replace_formula_outcome(formula_alt, lhs)
        try:
            cv_o = score_candidate_cv(
                df=search_with_col,
                formula_null=f_null,
                formula_alt=f_alt,
                participant_col="participant_id",
                n_folds=disc_cfg.cv_n_folds,
                random_state=config.seed,
                spec=config.model_specs[od.name],
            )
        except Exception as exc:
            return None, f"cv_error ({od.name}): {exc}"
        per_outcome_cvs[od.name] = cv_o

    multi_cv = MultiOutcomeCVScore(per_outcome=per_outcome_cvs)
    primary_cv = per_outcome_cvs[outcome_defs[0].name]
    adj = _compute_adjusted_score(
        multi_cv.joint_mean,
        primary_cv.se_ll_improvement,
        candidate,
        disc_cfg.stability_weight,
        disc_cfg.complexity_exponent,
        disc_cfg.depends_on_exponent,
        novelty_score=novelty,
        novelty_weight=disc_cfg.novelty_weight,
    )
    return ScoredCandidate(
        candidate=candidate,
        cv_score_mean=multi_cv.joint_mean,
        cv_score_se=primary_cv.se_ll_improvement,
        adjusted_score=adj,
        cv_score=primary_cv,
        multi_cv_score=multi_cv,
        column_values=series,
        novelty_score=novelty,
    ), None


def _format_score(sc: ScoredCandidate, config: BenchmarkConfig) -> str:
    outcome_defs = config.outcome_variable_defs
    disc_cfg = config.discovery
    novelty_str = f" novelty={sc.novelty_score:.3f}" if disc_cfg.novelty_weight > 0.0 else ""
    if len(outcome_defs) > 1 and sc.multi_cv_score is not None:
        outcome_parts = " ".join(
            f"{od.name}={sc.multi_cv_score.per_outcome[od.name].mean_ll_improvement:.4f}"
            for od in outcome_defs
        )
        return f"✓ joint_mean={sc.cv_score_mean:.4f} [{outcome_parts}]{novelty_str}"
    return (
        f"✓ cv_mean={sc.cv_score.mean_ll_improvement:.4f} "
        f"±se={sc.cv_score.se_ll_improvement:.4f} "
        f"(n={sc.cv_score.n_participants})"
        f"{novelty_str}"
    )


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(
    registry: FactorRegistry,
    config: BenchmarkConfig,
    round_num: int,
    iteration: int,
    scored: List[ScoredCandidate],
    all_scored: List[ScoredCandidate],
    task_context: str,
    observable_descriptions: Optional[dict],
) -> SearchContext:
    disc_cfg = config.discovery
    n_to_generate = (
        disc_cfg.seeding_strategy.n_candidates if iteration == 0
        else disc_cfg.evolution_strategy.n_candidates
    )
    observable_factors_meta = []
    for bf in config.base_factors:
        entry: dict = {"name": bf.name, "dtype": bf.dtype, "levels": bf.levels}
        if observable_descriptions and bf.name in observable_descriptions:
            entry["description"] = observable_descriptions[bf.name]
        observable_factors_meta.append(entry)

    return SearchContext(
        task_context=task_context,
        observable_factors=observable_factors_meta,
        discovered_factors=registry.discovered,
        hard_rejected=registry.hard_rejected,
        scored_candidates=scored,
        all_scored_candidates=all_scored,
        round_num=round_num,
        iteration=iteration,
        allowed_factor_types=disc_cfg.allowed_factor_types,
        allowed_factor_classes=disc_cfg.allowed_factor_classes,
        max_window_width=disc_cfg.max_window_width,
        n_to_generate=n_to_generate,
        top_k=disc_cfg.evolution_strategy.top_k,
    )


def _build_obs_desc_str(
    observable_cols: List[str],
    descriptions: Optional[Dict[str, str]],
) -> str:
    lines = []
    for name in observable_cols:
        if name in ("participant_id", "block_index", "trial_index"):
            continue
        desc = (descriptions or {}).get(name, "")
        lines.append(f"  {name}{': ' + desc if desc else ''}")
    return "\n".join(lines) if lines else "  (none)"


# ---------------------------------------------------------------------------
# Main round search
# ---------------------------------------------------------------------------

def run_within_round_search(
    search_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    config: BenchmarkConfig,
    llm: LLMClient,
    registry: FactorRegistry,
    round_num: int,
    observable_cols: List[str],
    seeder: SeedingStrategy,
    evolver: EvolutionStrategy,
    task_context: str = "",
    observable_descriptions: Optional[dict] = None,
) -> RoundResult:
    disc_cfg = config.discovery
    llm_cfg  = config.llm
    stat_cfg = config.statistical

    all_scored:             List[ScoredCandidate] = []
    hard_rejected_in_round: List[CandidateFactor] = []
    evaluated_names:        set                   = set()

    stagnation_enabled = disc_cfg.stagnation_patience > 0
    best_adj_score     = float("-inf")
    stagnation_count   = 0

    # ----------------------------------------------------------------
    # Initial seeding
    # ----------------------------------------------------------------
    ctx = _build_context(registry, config, round_num, 0, [], [], task_context, observable_descriptions)
    candidates = seeder.seed(ctx)
    print(f"  Initial batch: {len(candidates)} candidate(s) proposed")

    # ----------------------------------------------------------------
    # Search iterations
    # ----------------------------------------------------------------
    for iteration in range(1, disc_cfg.max_search_iterations + 1):
        print(f"\n  [Iter {iteration}/{disc_cfg.max_search_iterations}] "
              f"evaluating {len(candidates)} candidate(s)")
        newly_scored: List[ScoredCandidate] = []

        for candidate in candidates:
            if registry.is_duplicate(candidate):
                print(f"    [skip] {candidate.name} — duplicate")
                continue
            if candidate.name in evaluated_names:
                print(f"    [skip] {candidate.name} — already scored this round")
                continue

            print(f"    → {candidate.name} ({candidate.factor_type}, {candidate.factor_class})", end=" ", flush=True)

            # --- predicate synthesis (fast path if code already set) ---
            if candidate.compute_code is not None:
                candidate.predicate_status = "valid"
                compute_code = candidate.compute_code
            else:
                compute_code = synthesize_predicate(
                    llm=llm,
                    candidate=candidate,
                    working_df=search_df,
                    discovered=registry.discovered,
                    max_retries=disc_cfg.max_synthesis_retries,
                    temperature=llm_cfg.predicate_temperature,
                    timeout_seconds=disc_cfg.sandbox_timeout_seconds,
                    backend=disc_cfg.sandbox_backend,
                    max_tokens=llm_cfg.max_tokens_predicate,
                    observable_factor_descriptions=_build_obs_desc_str(
                        observable_cols, observable_descriptions
                    ),
                )
            if compute_code is None:
                reason = f"synthesis_failed ({candidate.predicate_status})"
                registry.hard_reject(candidate, reason)
                hard_rejected_in_round.append(candidate)
                print(f"✗ {reason}")
                continue

            # --- sandbox ---
            sandbox = run_predicate(
                predicate_code=compute_code,
                df=search_df,
                factor_type=candidate.factor_type,
                window_width=candidate.window_width,
                timeout_seconds=disc_cfg.sandbox_timeout_seconds,
                backend=disc_cfg.sandbox_backend,
                depends_on=candidate.depends_on,
            )
            if not sandbox.success:
                reason = f"sandbox_{sandbox.error_type}"
                registry.hard_reject(candidate, reason)
                hard_rejected_in_round.append(candidate)
                print(f"✗ {reason}")
                continue

            # --- encoding ---
            series, is_valid, enc_reason = encode_factor(
                raw_values=sandbox.values,
                df_index=search_df.index,
                declared_levels=candidate.levels,
                min_level_count=stat_cfg.min_level_count,
                factor_class=candidate.factor_class,
            )
            if not is_valid:
                reason = f"encoding: {enc_reason}"
                registry.hard_reject(candidate, reason)
                hard_rejected_in_round.append(candidate)
                print(f"✗ {reason}")
                continue

            # --- decomposition check ---
            if disc_cfg.decomposition_check_enabled and registry.discovered and candidate.factor_class == "discrete":
                discovered_names_in_df = [
                    f.column_name for f in registry.discovered if f.column_name in search_df.columns
                ]
                decomp_result, decomp_factors = _check_decomposition(
                    candidate_series=series,
                    df=search_df,
                    discovered_names=discovered_names_in_df,
                )
                if decomp_result == "bijection":
                    reason = f"exact_interaction_relabeling:{'+'.join(decomp_factors or [])}"
                    registry.hard_reject(candidate, reason)
                    hard_rejected_in_round.append(candidate)
                    if decomp_factors and len(decomp_factors) == 2:
                        registry.queue_pending_interaction(decomp_factors[0], decomp_factors[1])
                    print(f"✗ decomposable (bijection with {decomp_factors}) → queued for interaction search")
                    continue
                elif decomp_result == "coarsening":
                    candidate.coarsening_of = decomp_factors
                    print(f"~ coarsening of {decomp_factors} (proceeding to CV scoring)", end=" ", flush=True)

            # --- CV scoring (all outcomes) ---
            sc, score_error = _score_precomputed_candidate(
                candidate, series, search_df, config, registry
            )
            if sc is None:
                reason = score_error or "cv_error"
                registry.hard_reject(candidate, reason)
                hard_rejected_in_round.append(candidate)
                print(f"✗ {reason}")
                continue

            evaluated_names.add(candidate.name)
            newly_scored.append(sc)
            print(_format_score(sc, config))

            # --- automated contrast search ---
            if getattr(disc_cfg, "run_contrast_search", True):
                contrast_pairs = generate_level_vs_rest_contrasts(
                    candidate,
                    series,
                    max_contrasts=getattr(disc_cfg, "max_contrasts_per_candidate", 8),
                )
                for contrast, raw_contrast_series in contrast_pairs:
                    if registry.is_duplicate(contrast) or contrast.name in evaluated_names:
                        continue

                    contrast_series, contrast_valid, contrast_reason = encode_factor(
                        raw_values=list(raw_contrast_series),
                        df_index=search_df.index,
                        declared_levels=contrast.levels,
                        min_level_count=stat_cfg.min_level_count,
                        factor_class=contrast.factor_class,
                    )
                    print(f"    ↳ contrast {contrast.name}", end=" ", flush=True)
                    if not contrast_valid:
                        reason = f"encoding: {contrast_reason}"
                        registry.hard_reject(contrast, reason)
                        hard_rejected_in_round.append(contrast)
                        print(f"✗ {reason}")
                        continue

                    contrast_sc, contrast_error = _score_precomputed_candidate(
                        contrast, contrast_series, search_df, config, registry
                    )
                    if contrast_sc is None:
                        reason = contrast_error or "cv_error"
                        registry.hard_reject(contrast, reason)
                        hard_rejected_in_round.append(contrast)
                        print(f"✗ {reason}")
                        continue

                    evaluated_names.add(contrast.name)
                    newly_scored.append(contrast_sc)
                    print(_format_score(contrast_sc, config))

        all_scored.extend(newly_scored)

        # --- stagnation check ---
        if stagnation_enabled and all_scored:
            current_best = max(sc.adjusted_score for sc in all_scored)
            if current_best - best_adj_score > disc_cfg.stagnation_epsilon:
                best_adj_score = current_best
                stagnation_count = 0
            else:
                stagnation_count += 1
            if stagnation_count >= disc_cfg.stagnation_patience:
                print(f"  Early stop: no improvement for {stagnation_count} iteration(s).")
                break

        # --- evolution ---
        if iteration < disc_cfg.max_search_iterations:
            ctx = _build_context(
                registry, config, round_num, iteration,
                newly_scored, all_scored, task_context, observable_descriptions,
            )
            if not all_scored:
                ctx_seed = _build_context(
                    registry, config, round_num, 0, [], [], task_context, observable_descriptions
                )
                candidates = seeder.seed(ctx_seed)
                print(f"  No valid candidates — re-seeding: {len(candidates)} candidate(s)")
            else:
                candidates = evolver.evolve(ctx)
                print(f"  Evolution proposed {len(candidates)} candidate(s)")

    # ----------------------------------------------------------------
    # Select winner
    # ----------------------------------------------------------------
    if not all_scored:
        print(f"\n  Round {round_num}: no valid candidates scored.")
        return RoundResult(
            round_num=round_num, winner=None, winner_val_series=None,
            all_scored=[], hard_rejected_in_round=hard_rejected_in_round,
            validation_improvement=None, accepted=False,
        )

    outcome_defs = config.outcome_variable_defs
    # In multi-outcome mode, only consider candidates that improved ALL outcomes on CV.
    if len(outcome_defs) > 1:
        eligible = [
            sc for sc in all_scored
            if sc.multi_cv_score is not None and sc.multi_cv_score.joint_min > 0
        ]
        winner = _select_winner(eligible if eligible else all_scored)
    else:
        winner = _select_winner(all_scored)

    print(
        f"\n  Winner: {winner.candidate.name}"
        f"  (cv_mean={winner.cv_score_mean:.4f},"
        f" se={winner.cv_score_se:.4f},"
        f" n_params={_n_params(winner.candidate)},"
        f" n_deps={len(winner.candidate.depends_on)},"
        f" adj_score={winner.adjusted_score:.4f})"
    )

    for sc in all_scored:
        if sc is not winner:
            registry.add_evaluated(sc.candidate, sc.cv_score_mean, sc.cv_score_se)

    # ----------------------------------------------------------------
    # Validate winner
    # ----------------------------------------------------------------
    val_series = compute_factor_column(winner.candidate, validation_df, config)
    if val_series is None:
        print("  ✗ Validation: could not compute winner column on validation set.")
        registry.add_evaluated(winner.candidate, winner.cv_score_mean, winner.cv_score_se)
        return RoundResult(
            round_num=round_num, winner=winner, winner_val_series=None,
            all_scored=all_scored, hard_rejected_in_round=hard_rejected_in_round,
            validation_improvement=None, accepted=False,
        )

    col_name = winner.candidate.name
    search_with_winner = search_df.copy()
    search_with_winner[col_name] = winner.column_values

    val_with_winner = validation_df.copy()
    val_with_winner[col_name] = val_series

    formula_null_primary = registry.get_current_formula()
    formula_alt_primary  = build_extended_formula(formula_null_primary, col_name, factor_class=winner.candidate.factor_class)

    # Validate on every outcome; all must meet the threshold.
    val_improvements: Dict[str, float] = {}
    val_accepted = True
    current_lhs_primary = formula_null_primary.split("~")[0].strip()
    for od in outcome_defs:
        lhs = current_lhs_primary if od.name in current_lhs_primary else od.name
        f_null = replace_formula_outcome(formula_null_primary, lhs)
        f_alt  = replace_formula_outcome(formula_alt_primary,  lhs)
        impr = evaluate_on_held_out(
            df_train=search_with_winner,
            df_test=val_with_winner,
            formula_null=f_null,
            formula_alt=f_alt,
            spec=config.model_specs[od.name],
        )
        val_improvements[od.name] = impr
        if not (math.isfinite(impr) and impr >= disc_cfg.min_validation_improvement):
            val_accepted = False

    accepted = val_accepted
    primary_improvement = val_improvements.get(outcome_defs[0].name, float("-inf"))

    if len(outcome_defs) > 1:
        impr_str = ", ".join(f"{k}={v:.4f}" for k, v in val_improvements.items())
        status = "ACCEPTED" if accepted else f"REJECTED ({impr_str})"
        print(f"  Validation improvements: {impr_str} → {status}")
    else:
        status = "ACCEPTED" if accepted else f"REJECTED (improvement={primary_improvement:.4f} < threshold={disc_cfg.min_validation_improvement})"
        print(f"  Validation improvement: {primary_improvement:.4f} → {status}")

    if not accepted:
        registry.add_evaluated(winner.candidate, winner.cv_score_mean, winner.cv_score_se)

    return RoundResult(
        round_num=round_num, winner=winner, winner_val_series=val_series,
        all_scored=all_scored, hard_rejected_in_round=hard_rejected_in_round,
        validation_improvement=primary_improvement,
        validation_improvements=val_improvements,
        accepted=accepted,
    )
