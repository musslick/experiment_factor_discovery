"""
Loader for empirical (real-world) datasets.

Returns (full_df, input_df):
  full_df  — participant_id + base_factors + hidden_factors + extra_columns + outcome_variable.
             Hidden factor columns are present for ground-truth evaluation.
  input_df — full_df minus the hidden factor columns (never shown to the pipeline).
"""

from typing import Tuple

import pandas as pd

from src.utils.config import BenchmarkConfig


def load_empirical_data(cfg: BenchmarkConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load an empirical CSV and split into (full_df, input_df).

    full_df  — all relevant columns including hidden factor columns.
    input_df — full_df minus the hidden factor columns.

    The participant ID column is renamed to "participant_id" regardless of its
    original name in the CSV, to match the convention expected by the pipeline.
    """
    ds = cfg.dataset
    raw = pd.read_csv(ds.path)

    pid_col      = ds.participant_id_column
    base_names   = [bf.name   for bf in ds.base_factors]
    hidden_cols  = [hf.column for hf in ds.hidden_factors]   # original CSV column names
    hidden_names = [hf.name   for hf in ds.hidden_factors]   # model-facing names

    keep = [pid_col] + base_names + hidden_cols + ds.extra_columns + [ds.outcome_variable]

    # Deduplicate while preserving order
    seen: set = set()
    keep_dedup = []
    for c in keep:
        if c not in seen:
            keep_dedup.append(c)
            seen.add(c)

    missing = [c for c in keep_dedup if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Empirical dataset '{ds.path}' is missing expected columns: {missing}.\n"
            f"Available columns: {list(raw.columns)}"
        )

    full_df = raw[keep_dedup].copy()

    # Rename participant ID column to pipeline convention
    if pid_col != "participant_id":
        full_df = full_df.rename(columns={pid_col: "participant_id"})

    # Rename hidden factor CSV columns to their model-facing names if they differ
    rename_map = {hf.column: hf.name for hf in ds.hidden_factors if hf.column != hf.name}
    if rename_map:
        full_df = full_df.rename(columns=rename_map)

    # Add trial_index (sequential within each participant, preserving original row order)
    # so the sandbox harness can sort trials correctly — synthetic data has this column
    # by construction, but empirical CSVs typically use a different name.
    if "trial_index" not in full_df.columns:
        full_df["trial_index"] = (
            full_df.groupby("participant_id", sort=False).cumcount()
        )

    input_df = full_df.drop(columns=hidden_names, errors="ignore")

    return full_df, input_df
