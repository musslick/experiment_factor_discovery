"""
Iterative within-round search loop.

Each call to run_within_round_search() implements one full round:

  1. Generate an initial batch of candidates.
  2. Screen each candidate (synthesize predicate → sandbox → encode).
     Hard failures go to registry.hard_rejected and are permanently banned.
  3. CV-score every valid candidate on the search set (marginal improvement
     over the current null formula).
  4. Feed CV scores back to the LLM for refinement (top-k highlighted).
  5. Repeat steps 2-4 until the search budget is exhausted.
  6. Select the winner by highest (mean − λ_se × se) / n_params^exponent CV score,
     where n_params = n_levels − 1.  This normalises by model complexity so that
     compound multi-level factors must earn each additional parameter they add.
  7. Validate the winner once on the fixed held-out validation set.
     Accept if the mean per-participant LL improvement exceeds the threshold.
  8. Register soft-rejected (CV-scored, non-winner) candidates in the registry
     so they can be revisited in later rounds.

Between rounds the LLM starts from scratch.  Only factors that failed to
compile (hard_rejected) are permanently off-limits.  Factors that compiled
but scored weakly in a previous round are free to be re-proposed; the updated
null formula (which now contains the accepted factors) gives them a fair chance
to show marginal value they could not demonstrate before.

Data-split helper
-----------------
split_participants() is called once before round 1 in pipeline.py.  The same
search/validation split is reused in every subsequent round.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.factor_encoder import encode_factor
from src.analysis.model_comparison import (
    CVScore,
    build_extended_formula,
    evaluate_on_held_out,
    score_candidate_cv,
)
from src.discovery.candidate_generator import generate_candidates, refine_candidates
from src.discovery.factor_registry import CandidateFactor, DiscoveredFactor, FactorRegistry
from src.discovery.llm_client import LLMClient
from src.discovery.predicate_synthesizer import synthesize_predicate
from src.discovery.sandbox import run_predicate
from src.utils.config import BenchmarkConfig


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    """A candidate that passed all screening checks and received a CV score."""
    candidate: CandidateFactor
    cv_score: CVScore
    column_values: pd.Series   # aligned to search_df index


@dataclass
class RoundResult:
    round_num: int
    winner: Optional[ScoredCandidate]
    winner_val_series: Optional[pd.Series]  # winner column aligned to validation_df
    all_scored: List[ScoredCandidate]
    hard_rejected_in_round: List[CandidateFactor]
    validation_improvement: Optional[float]  # None if winner was not validated
    accepted: bool


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

    The split is done once before round 1 and reused across all rounds so
    the validation set is never re-randomised mid-experiment.
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
    """
    Run the candidate's predicate on df and return an encoded Series.
    Uses min_level_count=1 by default (no count guard needed outside of
    the main screening path).  Returns None on any failure.

    Public so pipeline.py can recompute values on the full dataset for
    evaluation after accepting a winner on just the search set.
    """
    sandbox = run_predicate(
        predicate_code=candidate.compute_code,
        df=df,
        factor_type=candidate.factor_type,
        timeout_seconds=config.discovery.sandbox_timeout_seconds,
        backend=config.discovery.sandbox_backend,
    )
    if not sandbox.success:
        return None
    series, is_valid, _ = encode_factor(
        raw_values=sandbox.values,
        df_index=df.index,
        declared_levels=candidate.levels,
        min_level_count=min_level_count,
    )
    return series if is_valid else None


def adjusted_score(
    sc: ScoredCandidate,
    stability_weight: float,
    complexity_exponent: float,
    depends_on_exponent: float,
) -> float:
    """
    Complexity-adjusted CV score used for winner selection and cross-round seeding:

        score = (mean_ll − λ_se × se) / (n_params^e_c × n_deps^e_d)

    n_params = max(1, n_levels − 1) — penalises multi-level factors.
    n_deps   = max(1, len(depends_on)) — penalises factors that combine many
               input variables, catching binary interaction factors that evade
               the level-count penalty.
    """
    raw    = sc.cv_score.mean_ll_improvement - stability_weight * sc.cv_score.se_ll_improvement
    n_par  = max(1, len(sc.candidate.levels) - 1)
    n_deps = max(1, len(sc.candidate.depends_on))
    return raw / (n_par ** complexity_exponent * n_deps ** depends_on_exponent)


def _select_winner(
    all_scored: List[ScoredCandidate],
    stability_weight: float,
    complexity_exponent: float,
    depends_on_exponent: float,
) -> Optional[ScoredCandidate]:
    if not all_scored:
        return None
    return max(
        all_scored,
        key=lambda sc: adjusted_score(sc, stability_weight, complexity_exponent, depends_on_exponent),
    )


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
) -> RoundResult:
    """
    Run one full round of the iterative search.

    Parameters
    ----------
    search_df       : DataFrame restricted to search participants; already
                      contains columns for all previously discovered factors.
    validation_df   : Fixed held-out DataFrame; updated with discovered
                      factor columns by pipeline.py after each accepted round.
    config          : Full benchmark configuration.
    llm             : Configured LLMClient.
    registry        : Current FactorRegistry (mutated in place: hard_reject /
                      add_evaluated are called during the round).
    round_num       : 1-based round index (used for logging and CandidateFactor).
    observable_cols : Base observable column names (not including discovered
                      factors, which are passed via registry.discovered).

    Returns
    -------
    RoundResult — winner (if any), validation outcome, and full audit trail.
    """
    disc_cfg = config.discovery
    llm_cfg  = config.llm
    stat_cfg = config.statistical

    all_scored:             List[ScoredCandidate]   = []
    hard_rejected_in_round: List[CandidateFactor]   = []
    evaluated_names:        set                      = set()  # block exact re-evals within round

    # ----------------------------------------------------------------
    # Initial generation
    # ----------------------------------------------------------------
    candidates = generate_candidates(
        llm=llm,
        observable_factors=observable_cols,
        discovered_so_far=registry.discovered,
        rejected_so_far=registry.hard_rejected,
        round_num=round_num,
        max_candidates=disc_cfg.max_candidates_per_round,
        temperature=llm_cfg.candidate_temperature,
        max_tokens=llm_cfg.max_tokens_candidate,
    )
    print(f"  Initial batch: {len(candidates)} candidate(s) proposed")

    # ----------------------------------------------------------------
    # Search iterations
    # ----------------------------------------------------------------
    for iteration in range(1, disc_cfg.max_search_iterations + 1):
        print(f"\n  [Iter {iteration}/{disc_cfg.max_search_iterations}] "
              f"evaluating {len(candidates)} candidate(s)")
        newly_scored: List[ScoredCandidate] = []

        for candidate in candidates:
            # --- duplicate / already-evaluated guard ---
            if registry.is_duplicate(candidate):
                print(f"    [skip] {candidate.name} — duplicate")
                continue
            if candidate.name in evaluated_names:
                print(f"    [skip] {candidate.name} — already scored this round")
                continue

            print(f"    → {candidate.name} ({candidate.factor_type})", end=" ", flush=True)

            # --- predicate synthesis (hard-reject on failure) ---
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
            )
            if compute_code is None:
                reason = f"synthesis_failed ({candidate.predicate_status})"
                registry.hard_reject(candidate, reason)
                hard_rejected_in_round.append(candidate)
                print(f"✗ {reason}")
                continue

            # --- sandbox execution (hard-reject on failure) ---
            sandbox = run_predicate(
                predicate_code=compute_code,
                df=search_df,
                factor_type=candidate.factor_type,
                timeout_seconds=disc_cfg.sandbox_timeout_seconds,
                backend=disc_cfg.sandbox_backend,
            )
            if not sandbox.success:
                reason = f"sandbox_{sandbox.error_type}"
                registry.hard_reject(candidate, reason)
                hard_rejected_in_round.append(candidate)
                print(f"✗ {reason}")
                continue

            # --- factor encoding (hard-reject on failure) ---
            series, is_valid, enc_reason = encode_factor(
                raw_values=sandbox.values,
                df_index=search_df.index,
                declared_levels=candidate.levels,
                min_level_count=stat_cfg.min_level_count,
            )
            if not is_valid:
                reason = f"encoding: {enc_reason}"
                registry.hard_reject(candidate, reason)
                hard_rejected_in_round.append(candidate)
                print(f"✗ {reason}")
                continue

            # --- CV scoring (soft outcome — no hard reject) ---
            col_name = candidate.name
            search_with_col = search_df.copy()
            search_with_col[col_name] = series

            formula_null = registry.get_current_formula()
            formula_alt  = build_extended_formula(formula_null, col_name)

            try:
                cv_score = score_candidate_cv(
                    df=search_with_col,
                    formula_null=formula_null,
                    formula_alt=formula_alt,
                    participant_col="participant_id",
                    n_folds=disc_cfg.cv_n_folds,
                    random_state=config.seed,
                )
            except Exception as exc:
                reason = f"cv_error: {exc}"
                registry.hard_reject(candidate, reason)
                hard_rejected_in_round.append(candidate)
                print(f"✗ {reason}")
                continue

            evaluated_names.add(col_name)
            sc = ScoredCandidate(candidate=candidate, cv_score=cv_score, column_values=series)
            newly_scored.append(sc)
            print(
                f"✓ cv_mean={cv_score.mean_ll_improvement:.4f} "
                f"±se={cv_score.se_ll_improvement:.4f} "
                f"(n={cv_score.n_participants})"
            )

        all_scored.extend(newly_scored)

        # --- refinement (all iterations except the last) ---
        if iteration < disc_cfg.max_search_iterations:
            if not all_scored:
                print("  No valid candidates yet — generating a fresh batch.")
                candidates = generate_candidates(
                    llm=llm,
                    observable_factors=observable_cols,
                    discovered_so_far=registry.discovered,
                    rejected_so_far=registry.hard_rejected,
                    round_num=round_num,
                    max_candidates=disc_cfg.candidates_per_refinement,
                    temperature=llm_cfg.candidate_temperature,
                    max_tokens=llm_cfg.max_tokens_candidate,
                )
            else:
                candidates = refine_candidates(
                    llm=llm,
                    scored_candidates=all_scored,
                    hard_rejected=registry.hard_rejected,
                    top_k=disc_cfg.refinement_top_k,
                    observable_factors=observable_cols,
                    discovered_so_far=registry.discovered,
                    round_num=round_num,
                    iteration_num=iteration,
                    n_to_generate=disc_cfg.candidates_per_refinement,
                    temperature=llm_cfg.candidate_temperature,
                    max_tokens=llm_cfg.max_tokens_candidate,
                )
                print(f"  Refinement proposed {len(candidates)} candidate(s)")

    # ----------------------------------------------------------------
    # Select winner
    # ----------------------------------------------------------------
    if not all_scored:
        print(f"\n  Round {round_num}: no valid candidates scored.")
        return RoundResult(
            round_num=round_num,
            winner=None,
            winner_val_series=None,
            all_scored=[],
            hard_rejected_in_round=hard_rejected_in_round,
            validation_improvement=None,
            accepted=False,
        )

    winner = _select_winner(
        all_scored,
        disc_cfg.stability_weight,
        disc_cfg.complexity_exponent,
        disc_cfg.depends_on_exponent,
    )
    adj = adjusted_score(
        winner,
        disc_cfg.stability_weight,
        disc_cfg.complexity_exponent,
        disc_cfg.depends_on_exponent,
    )
    print(
        f"\n  Winner: {winner.candidate.name}"
        f"  (cv_mean={winner.cv_score.mean_ll_improvement:.4f},"
        f" se={winner.cv_score.se_ll_improvement:.4f},"
        f" n_levels={len(winner.candidate.levels)},"
        f" n_deps={len(winner.candidate.depends_on)},"
        f" adj_score={adj:.4f})"
    )

    # Record non-winners as evaluated (soft, not hard)
    for sc in all_scored:
        if sc is not winner:
            registry.add_evaluated(
                sc.candidate,
                sc.cv_score.mean_ll_improvement,
                sc.cv_score.se_ll_improvement,
            )

    # ----------------------------------------------------------------
    # Validate winner on held-out set
    # ----------------------------------------------------------------
    val_series = compute_factor_column(winner.candidate, validation_df, config)
    if val_series is None:
        print("  ✗ Validation: could not compute winner column on validation set.")
        registry.add_evaluated(
            winner.candidate,
            winner.cv_score.mean_ll_improvement,
            winner.cv_score.se_ll_improvement,
        )
        return RoundResult(
            round_num=round_num,
            winner=winner,
            winner_val_series=None,
            all_scored=all_scored,
            hard_rejected_in_round=hard_rejected_in_round,
            validation_improvement=None,
            accepted=False,
        )

    col_name = winner.candidate.name
    search_with_winner = search_df.copy()
    search_with_winner[col_name] = winner.column_values

    val_with_winner = validation_df.copy()
    val_with_winner[col_name] = val_series

    formula_null = registry.get_current_formula()
    formula_alt  = build_extended_formula(formula_null, col_name)

    improvement = evaluate_on_held_out(
        df_train=search_with_winner,
        df_test=val_with_winner,
        formula_null=formula_null,
        formula_alt=formula_alt,
    )

    accepted = math.isfinite(improvement) and improvement >= disc_cfg.min_validation_improvement
    status   = "ACCEPTED" if accepted else f"REJECTED (improvement={improvement:.4f} < threshold={disc_cfg.min_validation_improvement})"
    print(f"  Validation improvement: {improvement:.4f} → {status}")

    if not accepted:
        registry.add_evaluated(
            winner.candidate,
            winner.cv_score.mean_ll_improvement,
            winner.cv_score.se_ll_improvement,
        )

    return RoundResult(
        round_num=round_num,
        winner=winner,
        winner_val_series=val_series,
        all_scored=all_scored,
        hard_rejected_in_round=hard_rejected_in_round,
        validation_improvement=improvement,
        accepted=accepted,
    )
