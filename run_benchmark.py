"""
End-to-end benchmark runner for the factor discovery pipeline.

Usage:
    python run_benchmark.py [--config config/stroop_benchmark.yaml]
                            [--regenerate]

Steps:
    1. Generate (or load) synthetic Stroop data
    2. Run the multi-round LLM discovery pipeline
    3. Evaluate discovered factors against the ground truth
    4. Save an evaluation report to results/<run_timestamp>/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.analysis.evaluation import match_factors_bijection
from src.data_generation.stroop_model import sample_accuracy
from src.data_generation.sweetpea_builder import build_stroop_dataset
from src.discovery.factor_registry import FactorRegistry
from src.discovery.llm_client import LLMClient
from src.discovery.pipeline import run_discovery_pipeline
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Factor discovery benchmark")
    parser.add_argument("--config", default="config/stroop_benchmark.yaml",
                        help="Path to benchmark config YAML")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration of synthetic data even if files exist")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Create a timestamped output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = Path(cfg.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output directory: {run_dir}")

    # ------------------------------------------------------------------
    # Step 1: Data generation (skipped if files already exist)
    # ------------------------------------------------------------------
    full_path  = Path("data/ground_truth/stroop_full.csv")
    input_path = Path("data/input/stroop_input.csv")

    dg = cfg.data_generation

    # Check whether existing files are stale (wrong participant count or missing)
    needs_regen = args.regenerate or not full_path.exists()
    if not needs_regen and full_path.exists():
        existing_n = pd.read_csv(full_path, usecols=["participant_id"])["participant_id"].nunique()
        if existing_n != dg.n_participants:
            print(f"\n  Config specifies {dg.n_participants} participants but existing data "
                  f"has {existing_n} — regenerating.")
            needs_regen = True

    if needs_regen:
        print("\nGenerating synthetic data …")
        df_full = build_stroop_dataset(
            n_participants=dg.n_participants,
            n_blocks_per_participant=dg.n_blocks_per_participant,
            seed=cfg.seed,
        )
        df_full = sample_accuracy(df_full, dg.logistic_model, seed=cfg.seed)

        full_path.parent.mkdir(parents=True, exist_ok=True)
        df_full.to_csv(full_path, index=False)

        observable = ["participant_id", "trial_index", "task", "color", "word", "correct"]
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df_full[observable].to_csv(input_path, index=False)

        print(f"  Ground truth → {full_path}")
        print(f"  Discovery input → {input_path}")
    else:
        print(f"\nLoading existing data from {full_path} "
              f"({dg.n_participants} participants)")
        df_full = pd.read_csv(full_path)

    df_obs = pd.read_csv(input_path)

    # ------------------------------------------------------------------
    # Step 2: Discovery pipeline
    # ------------------------------------------------------------------
    print("\nRunning discovery pipeline …")
    llm = LLMClient(model=cfg.llm.model)

    # Baseline includes all observable design factors from the input CSV
    # (everything except the bookkeeping columns and the outcome).
    _non_factor = {"participant_id", "trial_index", "correct"}
    _factor_cols = [c for c in df_obs.columns if c not in _non_factor]
    baseline_formula = "correct ~ " + " + ".join(f"C({c})" for c in _factor_cols)
    print(f"  Baseline formula: {baseline_formula}")

    registry = FactorRegistry(baseline_formula=baseline_formula)
    registry = run_discovery_pipeline(df_obs, cfg, llm, registry, str(run_dir))

    # ------------------------------------------------------------------
    # Step 3: Evaluation
    # ------------------------------------------------------------------
    print("\nEvaluating results …")
    report = match_factors_bijection(
        ground_truth_factors=cfg.evaluation.ground_truth_factors,
        discovered_factors=registry.discovered,
        full_df=df_full,
        threshold=cfg.evaluation.bijection_threshold,
    )

    print(f"\n{'='*50}")
    print(f"  Precision : {report.precision:.3f}")
    print(f"  Recall    : {report.recall:.3f}")
    print(f"  F1        : {report.f1:.3f}")
    print(f"  Matched   : {[(p.ground_truth_name, p.discovered_name, f'{p.agreement_rate:.3f}') for p in report.matched_pairs]}")
    if report.unmatched_ground_truth:
        print(f"  Missed GT : {report.unmatched_ground_truth}")
    if report.unmatched_discovered:
        print(f"  False pos : {report.unmatched_discovered}")
    print(f"{'='*50}\n")

    # ------------------------------------------------------------------
    # Step 4: Save report
    # ------------------------------------------------------------------
    report_dict = {
        "config":          args.config,
        "timestamp":       timestamp,
        "precision":       report.precision,
        "recall":          report.recall,
        "f1":              report.f1,
        "n_ground_truth":  report.n_ground_truth,
        "n_discovered":    report.n_discovered,
        "matched_pairs":   [
            {"ground_truth": p.ground_truth_name,
             "discovered":   p.discovered_name,
             "agreement":    round(p.agreement_rate, 4)}
            for p in report.matched_pairs
        ],
        "unmatched_ground_truth": report.unmatched_ground_truth,
        "unmatched_discovered":   report.unmatched_discovered,
        "discovered_factors": [
            {
                "name":                   f.column_name,
                "validation_improvement": f.validation_improvement,
                "formula":                f.formula_with,
            }
            for f in registry.discovered
        ],
    }
    report_path = run_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
