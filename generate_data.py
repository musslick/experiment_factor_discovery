"""
Phase 1 data generation script.

Usage:
    python generate_data.py [--config config/stroop_benchmark.yaml]

Outputs:
    data/ground_truth/stroop_full.csv  – all columns including hidden factors
    data/input/stroop_input.csv        – observable columns only (discovery input)
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.data_generation.sweetpea_builder import build_stroop_dataset
from src.data_generation.stroop_model import sample_accuracy


def generate(config_path: str) -> None:
    cfg = load_config(config_path)
    dg  = cfg.data_generation

    print(f"Generating data: {dg.n_participants} participants × "
          f"{dg.n_blocks_per_participant} blocks × 18 trials "
          f"= {dg.n_participants * dg.n_blocks_per_participant * 18} total trials")

    # Build trial sequences (includes hidden derived factor columns)
    df_full = build_stroop_dataset(
        n_participants=dg.n_participants,
        n_blocks_per_participant=dg.n_blocks_per_participant,
        seed=cfg.seed,
    )

    # Sample binary accuracy from the logistic model
    df_full = sample_accuracy(df_full, dg.logistic_model, seed=cfg.seed)

    # Save ground-truth dataset (all columns)
    Path("data/ground_truth").mkdir(parents=True, exist_ok=True)
    gt_path = "data/ground_truth/stroop_full.csv"
    df_full.to_csv(gt_path, index=False)
    print(f"Saved ground truth → {gt_path}  ({len(df_full)} rows, {df_full.shape[1]} cols)")

    # Save masked dataset (strip hidden factor columns)
    observable_cols = ["participant_id", "trial_index", "task", "color", "word", "correct"]
    df_input = df_full[observable_cols]
    Path("data/input").mkdir(parents=True, exist_ok=True)
    input_path = "data/input/stroop_input.csv"
    df_input.to_csv(input_path, index=False)
    print(f"Saved discovery input → {input_path}  ({len(df_input)} rows, {df_input.shape[1]} cols)")

    # Quick sanity summary
    mean_acc      = df_full["correct"].mean()
    con_acc       = df_full.loc[df_full["congruency"] == "congruent",   "correct"].mean()
    inc_acc       = df_full.loc[df_full["congruency"] == "incongruent", "correct"].mean()
    rep_acc       = df_full.loc[df_full["task_transition"] == "repeat", "correct"].mean()
    swi_acc       = df_full.loc[df_full["task_transition"] == "switch", "correct"].mean()
    nan_tt        = df_full["task_transition"].isna().sum()

    print(f"\nSanity check:")
    print(f"  Overall accuracy:          {mean_acc:.3f}")
    print(f"  Congruent accuracy:        {con_acc:.3f}")
    print(f"  Incongruent accuracy:      {inc_acc:.3f}  (Δ = {con_acc - inc_acc:+.3f})")
    print(f"  Task-repeat accuracy:      {rep_acc:.3f}")
    print(f"  Task-switch accuracy:      {swi_acc:.3f}  (Δ = {rep_acc - swi_acc:+.3f})")
    print(f"  NaN task_transition rows:  {nan_tt} "
          f"({100*nan_tt/len(df_full):.1f}% — one per block start)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/stroop_benchmark.yaml")
    args = parser.parse_args()
    generate(args.config)
