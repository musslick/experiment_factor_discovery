"""
Discovery pipeline orchestrator.

Runs N rounds of iterative within-round search:
  1. Generate, screen, and CV-score candidate factors on the search set.
  2. Refine candidates based on CV scores (repeated for max_search_iterations).
  3. Select the winner (highest complexity/dependency-adjusted CV score).
  4. Validate the winner once on the held-out validation set.
  5. If accepted, register the factor and advance the baseline formula.
     If rejected, continue to the next round without updating the model.
  6. Carry the top-k non-winners forward as seeds for the next round,
     skipping re-synthesis since their compute_code is already validated.

Participant split is done once before round 1; the same search/validation
division is used in every subsequent round.

Evaluation note
---------------
DiscoveredFactor.column_values is stored as a Series indexed to the full
observable_df (all participants) so that evaluation.py can align it with
full_df by pandas index without mismatch.  After accepting a winner whose
column was computed on the search set, the column is recomputed on the full
dataset before being stored in the registry.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.analysis.model_comparison import build_extended_formula
from src.discovery.effect_searcher import EffectSearchResult, run_effect_search
from src.discovery.factor_registry import DiscoveredFactor, FactorRegistry
from src.discovery.llm_client import LLMClient
from src.discovery.strategies import build_seeding_strategy, build_evolution_strategy
from src.discovery.within_round_search import (
    RoundResult,
    compute_factor_column,
    run_within_round_search,
    split_participants,
)
from src.utils.config import BenchmarkConfig


def _save_round_log(result: RoundResult, output_dir: str) -> None:
    """Serialise a RoundResult to JSON (column Series values are omitted)."""
    log: dict = {
        "round":                    result.round_num,
        "accepted":                 result.accepted,
        "validation_improvement":   result.validation_improvement,
        "validation_improvements":  result.validation_improvements,
        "winner":                   None,
        "all_scored":               [],
        "hard_rejected":            [],
    }

    if result.winner is not None:
        w = result.winner
        n_part = w.cv_score.n_participants if w.cv_score is not None else None
        log["winner"] = {
            "name":             w.candidate.name,
            "factor_type":      w.candidate.factor_type,
            "levels":           w.candidate.levels,
            "depends_on":       w.candidate.depends_on,
            "description":      w.candidate.description,
            "contrast_of":      w.candidate.contrast_of,
            "contrast_positive_levels": w.candidate.contrast_positive_levels,
            "cv_score_mean":    w.cv_score_mean,
            "cv_score_se":      w.cv_score_se,
            "n_participants":   n_part,
            "novelty_score":    w.novelty_score,
            "predicate_status": w.candidate.predicate_status,
            "sweetpea_code":    w.candidate.sweetpea_code,
            "compute_code":     w.candidate.compute_code,
        }

    for sc in result.all_scored:
        n_part = sc.cv_score.n_participants if sc.cv_score is not None else None
        log["all_scored"].append({
            "name":             sc.candidate.name,
            "factor_type":      sc.candidate.factor_type,
            "levels":           sc.candidate.levels,
            "depends_on":       sc.candidate.depends_on,
            "description":      sc.candidate.description,
            "contrast_of":      sc.candidate.contrast_of,
            "contrast_positive_levels": sc.candidate.contrast_positive_levels,
            "cv_score_mean":    sc.cv_score_mean,
            "cv_score_se":      sc.cv_score_se,
            "n_participants":   n_part,
            "novelty_score":    sc.novelty_score,
            "predicate_status": sc.candidate.predicate_status,
            "sweetpea_code":    sc.candidate.sweetpea_code,
            "compute_code":     sc.candidate.compute_code,
        })

    for c in result.hard_rejected_in_round:
        log["hard_rejected"].append({
            "name":             c.name,
            "factor_type":      c.factor_type,
            "levels":           c.levels,
            "contrast_of":      c.contrast_of,
            "contrast_positive_levels": c.contrast_positive_levels,
            "rejection_reason": c.rejection_reason,
            "predicate_status": c.predicate_status,
            "sweetpea_code":    c.sweetpea_code,
            "compute_code":     c.compute_code,
        })

    log_path = Path(output_dir) / f"round_{result.round_num:02d}_candidates.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Round {result.round_num} log → {log_path}")


def _save_effect_log(result: EffectSearchResult, output_dir: str) -> None:
    """Serialise an EffectSearchResult to JSON."""
    log: dict = {
        "round":            result.round_num,
        "n_accepted":       len(result.accepted_effects),
        "accepted_effects": [],
        "all_tested":       [],
    }
    for e in result.accepted_effects:
        log["accepted_effects"].append({
            "term":                   e.term,
            "factor_names":           e.factor_names,
            "cv_score_mean":          e.cv_score_mean,
            "cv_score_se":            e.cv_score_se,
            "n_participants":         e.n_participants,
            "validation_improvement": e.validation_improvement,
            "source":                 e.source,
            "llm_rationale":          e.llm_rationale,
        })
    for t in result.all_tested:
        log["all_tested"].append({
            "factor_names":    t.factor_names,
            "term":            t.term,
            "formula_hash":    t.formula_hash,
            "cv_score_mean":   t.cv_score_mean,
            "cv_score_se":     t.cv_score_se,
            "outcome":         t.outcome,
        })
    log_path = Path(output_dir) / f"round_{result.round_num:02d}_effects.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  Round {result.round_num} effect log → {log_path}")


def run_discovery_pipeline(
    observable_df: pd.DataFrame,
    config: BenchmarkConfig,
    llm: LLMClient,
    registry: FactorRegistry,
    output_dir: str,
) -> FactorRegistry:
    """
    Run the full multi-round iterative discovery pipeline.

    Parameters
    ----------
    observable_df : DataFrame with only the observable columns (no hidden factors).
                    Its index is used as the reference for all column_values
                    stored in DiscoveredFactor (evaluation alignment).
    config        : Full benchmark configuration.
    llm           : Configured LLMClient.
    registry      : FactorRegistry (starts empty for a fresh run).
    output_dir    : Directory where per-round JSON logs are written.

    Returns
    -------
    Updated FactorRegistry containing all accepted discovered factors.
    """
    disc_cfg = config.discovery

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build search strategies (once; reused across all rounds)
    # ------------------------------------------------------------------
    seeder  = build_seeding_strategy(config, llm)
    evolver = build_evolution_strategy(config, llm)
    print(f"\n  Seeding strategy  : {disc_cfg.seeding_strategy.type}")
    print(f"  Evolution strategy: {disc_cfg.evolution_strategy.type}")

    # ------------------------------------------------------------------
    # One-time participant split (fixed across all rounds)
    # ------------------------------------------------------------------
    search_df, validation_df = split_participants(
        observable_df,
        validation_fraction=disc_cfg.validation_fraction,
        seed=config.seed,
    )
    n_search = search_df["participant_id"].nunique()
    n_val    = validation_df["participant_id"].nunique()
    print(f"\n  Participant split: {n_search} search / {n_val} validation")

    # Register outcome variable definitions for multi-outcome support.
    registry.set_outcome_variable_defs(config.outcome_variable_defs)

    # full_working_df tracks all participants with discovered factor columns
    # appended after each accepted round.  column_values stored in the registry
    # are aligned to this DataFrame's index so that evaluation.py can join them
    # with full_df (ground-truth) without index mismatch.
    full_working_df = observable_df.copy()

    observable_cols = list(observable_df.columns)

    # Build per-benchmark context strings injected into every LLM prompt
    task_context = config.task_context.strip()
    observable_descriptions: dict = {}
    for bf in config.base_factors:
        if bf.dtype == "categorical" and bf.levels:
            observable_descriptions[bf.name] = " | ".join(f'"{lv}"' for lv in bf.levels)
        elif bf.dtype == "continuous":
            observable_descriptions[bf.name] = "float (continuous)"
    # outcome variable(s)
    for od in config.outcome_variable_defs:
        observable_descriptions[od.name] = "0 | 1" if od.type == "binary" else "continuous (float)"

    for round_num in range(1, disc_cfg.n_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  Round {round_num} / {disc_cfg.n_rounds}")
        print(f"  Baseline formula : {registry.get_current_formula()}")
        print(f"  Discovered so far: {registry.get_discovered_column_names()}")
        print(f"{'='*60}")

        result = run_within_round_search(
            search_df=search_df,
            validation_df=validation_df,
            config=config,
            llm=llm,
            registry=registry,
            round_num=round_num,
            observable_cols=observable_cols,
            seeder=seeder,
            evolver=evolver,
            task_context=task_context,
            observable_descriptions=observable_descriptions,
        )

        _save_round_log(result, output_dir)

        new_factor_name: Optional[str] = None

        if result.accepted:
            winner   = result.winner
            col_name = winner.candidate.name

            # Advance the search and validation DataFrames
            search_df[col_name] = winner.column_values
            if result.winner_val_series is not None:
                validation_df[col_name] = result.winner_val_series

            # Recompute on all participants so column_values aligns with full_df
            # for evaluation.  full_working_df already carries previously
            # discovered factor columns needed by the predicate.
            full_series = compute_factor_column(winner.candidate, full_working_df, config)
            if full_series is not None:
                full_working_df[col_name] = full_series
            col_values_for_registry = full_series if full_series is not None else winner.column_values

            formula_alt = build_extended_formula(registry.get_current_formula(), col_name, factor_class=winner.candidate.factor_class)
            discovered  = DiscoveredFactor(
                candidate=winner.candidate,
                column_name=col_name,
                column_values=col_values_for_registry,
                lrt_statistic=0.0,
                lrt_pvalue=1.0,
                lrt_dof=0,
                formula_with=formula_alt,
                validation_improvement=result.validation_improvement,
                validation_improvements=result.validation_improvements,
                novelty_score=result.winner.novelty_score,
            )
            registry.register(discovered)
            new_factor_name = col_name
            print(f"\n  ✓ '{col_name}' added to model. "
                  f"New formula: {registry.get_current_formula()}")
        else:
            print(f"\n  Round {round_num}: no factor accepted — baseline unchanged.")

        # ------------------------------------------------------------------
        # Phase 2: effect search (interaction terms)
        # ------------------------------------------------------------------
        if config.discovery.run_effect_search:
            effect_result = run_effect_search(
                search_df=search_df,
                validation_df=validation_df,
                config=config,
                llm=llm,
                registry=registry,
                round_num=round_num,
                new_factor_name=new_factor_name,
            )
            _save_effect_log(effect_result, output_dir)

    print(f"\n{'='*60}")
    print(f"  Discovery complete.")
    print(f"  Discovered factors: {registry.get_discovered_column_names()}")
    print(f"{'='*60}\n")
    return registry
