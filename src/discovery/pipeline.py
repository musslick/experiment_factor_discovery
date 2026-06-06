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
from src.discovery.factor_registry import DiscoveredFactor, FactorRegistry
from src.discovery.llm_client import LLMClient
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
        "round":                  result.round_num,
        "accepted":               result.accepted,
        "validation_improvement": result.validation_improvement,
        "winner":                 None,
        "all_scored":             [],
        "hard_rejected":          [],
    }

    if result.winner is not None:
        log["winner"] = {
            "name":           result.winner.candidate.name,
            "factor_type":    result.winner.candidate.factor_type,
            "levels":         result.winner.candidate.levels,
            "depends_on":     result.winner.candidate.depends_on,
            "description":    result.winner.candidate.description,
            "cv_score_mean":  result.winner.cv_score.mean_ll_improvement,
            "cv_score_se":    result.winner.cv_score.se_ll_improvement,
            "n_participants": result.winner.cv_score.n_participants,
        }

    for sc in result.all_scored:
        log["all_scored"].append({
            "name":           sc.candidate.name,
            "factor_type":    sc.candidate.factor_type,
            "levels":         sc.candidate.levels,
            "depends_on":     sc.candidate.depends_on,
            "description":    sc.candidate.description,
            "cv_score_mean":  sc.cv_score.mean_ll_improvement,
            "cv_score_se":    sc.cv_score.se_ll_improvement,
            "n_participants": sc.cv_score.n_participants,
        })

    for c in result.hard_rejected_in_round:
        log["hard_rejected"].append({
            "name":             c.name,
            "factor_type":      c.factor_type,
            "levels":           c.levels,
            "rejection_reason": c.rejection_reason,
        })

    log_path = Path(output_dir) / f"round_{result.round_num:02d}_candidates.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Round {result.round_num} log → {log_path}")



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

    # full_working_df tracks all participants with discovered factor columns
    # appended after each accepted round.  column_values stored in the registry
    # are aligned to this DataFrame's index so that evaluation.py can join them
    # with full_df (ground-truth) without index mismatch.
    full_working_df = observable_df.copy()

    observable_cols = list(observable_df.columns)

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
        )

        _save_round_log(result, output_dir)

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

            formula_alt = build_extended_formula(registry.get_current_formula(), col_name)
            discovered  = DiscoveredFactor(
                candidate=winner.candidate,
                column_name=col_name,
                column_values=col_values_for_registry,
                lrt_statistic=0.0,
                lrt_pvalue=1.0,
                lrt_dof=0,
                formula_with=formula_alt,
                validation_improvement=result.validation_improvement,
            )
            registry.register(discovered)
            print(f"\n  ✓ '{col_name}' added to model. "
                  f"New formula: {registry.get_current_formula()}")
        else:
            print(f"\n  Round {round_num}: no factor accepted — baseline unchanged.")

    print(f"\n{'='*60}")
    print(f"  Discovery complete.")
    print(f"  Discovered factors: {registry.get_discovered_column_names()}")
    print(f"{'='*60}\n")
    return registry
