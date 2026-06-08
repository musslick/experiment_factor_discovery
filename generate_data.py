"""
Standalone data-generation script.

Usage:
    python generate_data.py --config config/stroop_benchmark.yaml
    python generate_data.py --config config/run.yaml   # generates all listed benchmarks
"""

import argparse
import yaml
from pathlib import Path
from typing import List

import pandas as pd

from src.data_generation import get_data_generator
from src.utils.config import load_config


def _resolve_configs(config_path: str):
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    if "benchmarks" in raw:
        base = Path(config_path).parent
        paths = [str(base.parent / p) if not Path(p).is_absolute() else p
                 for p in raw["benchmarks"]]
        shared = {k: v for k, v in raw.items() if k != "benchmarks"}
        return paths, shared
    return [config_path], {}


def generate_one(config_path: str, shared_defaults: dict = None) -> None:
    cfg       = load_config(config_path, defaults=shared_defaults or None)
    dg        = cfg.data_generation
    generator = get_data_generator(cfg.benchmark_type)

    n_trials_per = dg.n_blocks_per_participant  # generators use this as their block count
    print(f"\n[{cfg.name}]  type={cfg.benchmark_type}  "
          f"{dg.n_participants} participants × {n_trials_per} blocks  "
          f"seed={cfg.seed}")

    full_df, input_df = generator.generate(cfg)

    name = cfg.name
    gt_path    = Path(f"data/ground_truth/{name}_full.csv")
    input_path = Path(f"data/input/{name}_input.csv")

    gt_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(gt_path, index=False)
    input_df.to_csv(input_path, index=False)

    print(f"  Ground truth saved  → {gt_path}   ({len(full_df)} rows, {full_df.shape[1]} cols)")
    print(f"  Discovery input saved → {input_path}  ({len(input_df)} rows, {input_df.shape[1]} cols)")

    # Quick sanity: outcome rate
    ov = dg.outcome_variable
    if ov in full_df.columns:
        print(f"  Mean {ov}: {full_df[ov].mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark data")
    parser.add_argument("--config", default="config/run.yaml")
    args = parser.parse_args()

    paths, shared = _resolve_configs(args.config)
    for p in paths:
        generate_one(p, shared_defaults=shared)


if __name__ == "__main__":
    main()
