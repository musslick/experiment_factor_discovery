"""
produce_results_empirical.py

Runs empirical benchmarks (to evaluate known hidden factor recovery) and
discovery runs (to find novel factors), then collects and saves aggregated
results to a single JSON file.

Usage:
    python produce_results_empirical.py --config config/produce_empirical.yaml
    python produce_results_empirical.py --config config/produce_empirical.yaml \\
        --datasets stroop_congruency janker \\
        --n-benchmark-runs 5 --n-discovery-runs 3 \\
        --base-seed 0 --output-dir results/aggregated \\
        --output-name empirical_results.json
    python produce_results_empirical.py --config config/produce_empirical.yaml --skip-benchmark
    python produce_results_empirical.py --config config/produce_empirical.yaml --skip-discovery
    python produce_results_empirical.py --resume results/aggregated/empirical_results.json
"""

import argparse
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from run_benchmark import run_single_benchmark
from run_discovery import run_single_discovery
from src.data_generation.empirical_loader import load_empirical_data
from src.discovery.factor_registry import CandidateFactor
from src.discovery.within_round_search import compute_factor_column
from src.utils.config import BenchmarkConfig, load_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_CONFIGS: Dict[str, str] = {
    "stroop_congruency":      "config/empirical_stroop_congruency.yaml",
    "janker":                 "config/empirical_janker_et_al.yaml",
    "digit_task_switching":   "config/empirical_digit_task_switiching.yaml",
    "flanker_congruency":     "config/empirical_flanker_congruency.yaml",
    "grange_n2_task":         "config/empirical_grange_rr_n-2_task.yaml",
    "hirsch_dual_task":       "config/empirical_hirsch_et_al_dual_task.yaml",
    "brooks_prospect_theory": "config/empirical_prospect_theory_brooks_et_al.yaml",
    "strittmatter":           "config/empirical_strittmatter_et_al.yaml",
    "weber_with_rewards":     "config/empirical_weber_et_al_with_rewards.yaml",
    "weber_without_rewards":  "config/empirical_weber_et_al_without_rewards.yaml",
}

OUTCOME_DISPLAY: Dict[str, str] = {
    "latency": "RT (ms)",
    "accuracy": "Accuracy (%)",
    "correct": "Accuracy (%)",
    "chose_left": "P(chose left)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_produce_config(config_path: str) -> dict:
    """Load the produce_empirical.yaml top-level config."""
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    # Strip leading/trailing document markers if any
    if raw is None:
        raw = {}
    return raw


def _display_name(cfg: BenchmarkConfig) -> str:
    """Build a display name like 'Stroop (N=466)'."""
    n = cfg.dataset.n_participants if cfg.dataset and cfg.dataset.n_participants else "?"
    # Use the benchmark name as a human-readable base
    name_map = {
        "stroop_congruency_empirical": "Stroop",
        "stroop_categorization_empirical": "Stroop Categorization",
    }
    base = name_map.get(cfg.name, cfg.name.replace("_", " ").title())
    return f"{base} (N={n})"


def _ground_truth_factors(cfg: BenchmarkConfig) -> List[dict]:
    """Extract ground-truth factor metadata from the config."""
    factors = []
    for gtf in cfg.evaluation.ground_truth_factors:
        factors.append({
            "name": gtf.name,
            "type": gtf.type,
            "factor_class": gtf.factor_class,
            "n_levels": len(gtf.levels),
            "levels": list(gtf.levels),
        })
    return factors


def _collect_benchmark_run(run_dir: Path, cfg: BenchmarkConfig, seed: int) -> Optional[dict]:
    """
    Read evaluation_report.json from run_dir / cfg.name / and return a
    standardised benchmark-run dict.  Returns None if the file is missing.
    """
    report_path = run_dir / cfg.name / "evaluation_report.json"
    if not report_path.exists():
        print(f"  [WARN] evaluation_report.json not found at {report_path}")
        return None

    with open(report_path) as fh:
        report = json.load(fh)

    # Support both old flat layout and the newer nested factor_evaluation layout
    fe = report.get("factor_evaluation", report)

    precision = fe.get("precision", report.get("precision", 0.0))
    recall = fe.get("recall", report.get("recall", 0.0))
    f1 = fe.get("f1", report.get("f1", 0.0))
    n_ground_truth = fe.get("n_ground_truth", 0)
    n_discovered = fe.get("n_discovered", 0)

    matched_pairs = [
        {
            "ground_truth": p.get("ground_truth", p.get("ground_truth_name", "")),
            "discovered": p.get("discovered", p.get("discovered_name", "")),
            "agreement": p.get("agreement", p.get("agreement_rate", 0.0)),
        }
        for p in fe.get("matched_pairs", [])
    ]

    # Level-recovery per matched factor
    level_recovery: Dict[str, dict] = {}
    gt_map = {gtf.name: gtf for gtf in cfg.evaluation.ground_truth_factors}
    for pair in matched_pairs:
        gt_name = pair["ground_truth"]
        gtf = gt_map.get(gt_name)
        if gtf is None:
            continue
        n_levels = len(gtf.levels)
        # We don't have per-level breakdown in the report — use agreement as a proxy:
        # if agreement == 1.0 all levels recovered, else 0
        n_recovered = n_levels if pair["agreement"] >= 0.99 else 0
        level_recall = n_recovered / n_levels if n_levels > 0 else 0.0
        level_recovery[gt_name] = {
            "n_levels": n_levels,
            "n_recovered": n_recovered,
            "level_recall": level_recall,
        }

    # Continuous correlation per matched factor (not in report currently → empty)
    continuous_corr: Dict[str, float] = {}

    unmatched_gt = fe.get("unmatched_ground_truth", [])
    unmatched_disc = fe.get("unmatched_discovered", [])

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_ground_truth": n_ground_truth,
        "n_discovered": n_discovered,
        "matched_pairs": matched_pairs,
        "level_recovery_per_factor": level_recovery,
        "continuous_correlation_per_factor": continuous_corr,
        "unmatched_ground_truth": unmatched_gt,
        "unmatched_discovered": unmatched_disc,
    }


def _infer_factor_type(compute_code: str) -> str:
    """
    Infer whether a factor is 'window' or 'within_trial' from its compute_code.
    A window factor's function takes a list ('window' or 'w') as its parameter
    and accesses elements by index (e.g. window[0], w[-1]).
    """
    if not compute_code:
        return "within_trial"
    # Check function signature for 'window' or 'w' as parameter
    sig_match = re.search(r"def\s+compute_factor\s*\(\s*(\w+)", compute_code)
    if sig_match:
        param = sig_match.group(1)
        if param in ("window", "w"):
            return "window"
    return "within_trial"


def _infer_window_width(compute_code: str) -> int:
    """
    Infer the window width from compute_code by finding the largest absolute
    numeric index used (e.g. window[2] → width 3, w[-3] → width 3).
    Defaults to 2.
    """
    if not compute_code:
        return 2
    indices = re.findall(r"\[(-?\d+)\]", compute_code)
    max_abs = 2
    for idx_str in indices:
        abs_idx = abs(int(idx_str))
        if abs_idx > max_abs:
            max_abs = abs_idx
    return max_abs if max_abs >= 2 else 2


def _outcome_label(cfg: BenchmarkConfig) -> str:
    """Determine the display label for the outcome variable."""
    outcome_var = cfg.dataset.outcome_variable if cfg.dataset else "outcome"
    null_formula = cfg.dataset.null_formula if cfg.dataset else ""

    if null_formula and "np.log(" in null_formula:
        base_label = OUTCOME_DISPLAY.get(outcome_var, outcome_var)
        return f"log {base_label}" if base_label != outcome_var else f"log {outcome_var}"

    return OUTCOME_DISPLAY.get(outcome_var, outcome_var)


def _compute_participant_level_data(
    factor_info: dict,
    cfg: BenchmarkConfig,
    outcome_label: str,
) -> Optional[dict]:
    """
    For a discovered factor, compute per-participant means grouped by factor level.

    Returns a dict with keys:
        outcome_label, levels, group_means, group_sems, participant_means
    or None if computation fails.
    """
    compute_code = factor_info.get("compute_code", "")
    if not compute_code:
        return None

    # Use load_empirical_data so that trial_index is added (required by the sandbox
    # harness) and all columns (including hidden factors promoted to base in
    # discovery mode) are available for the compute_code.
    try:
        full_df, _ = load_empirical_data(cfg)
    except Exception as exc:
        print(f"    [WARN] Could not load dataset for {cfg.name}: {exc}")
        return None

    outcome_var = cfg.dataset.outcome_variable
    if outcome_var not in full_df.columns:
        print(f"    [WARN] outcome variable '{outcome_var}' not in dataset")
        return None

    factor_type = factor_info.get("factor_type", _infer_factor_type(compute_code))
    window_width = factor_info.get("window_width", _infer_window_width(compute_code))
    levels_declared = factor_info.get("levels", [])
    # Empty levels list indicates a continuous factor
    factor_class = "continuous" if not levels_declared else "discrete"

    candidate = CandidateFactor(
        name=factor_info.get("name", "discovered_factor"),
        description="",
        factor_type=factor_type,
        levels=list(levels_declared),
        depends_on=list(full_df.columns),  # all columns → no payload filtering in sandbox
        factor_class=factor_class,
        window_width=window_width,
        compute_code=compute_code,
    )

    try:
        factor_series = compute_factor_column(candidate, full_df, cfg, min_level_count=1)
    except Exception as exc:
        print(f"    [WARN] compute_factor_column failed for {candidate.name}: {exc}")
        return None

    if factor_series is None:
        print(f"    [WARN] compute_factor_column returned None for {candidate.name}")
        return None

    df = full_df.copy()
    df["_factor_col"] = factor_series

    mask = df["_factor_col"].notna() & df[outcome_var].notna()
    df = df[mask]

    if df.empty:
        return None

    unique_levels = sorted(df["_factor_col"].unique().tolist(), key=str)

    participant_means: Dict[str, List[float]] = {str(lv): [] for lv in unique_levels}
    grouped = df.groupby(["participant_id", "_factor_col"])[outcome_var].mean()

    for (pid, lv), val in grouped.items():
        key = str(lv)
        if key in participant_means:
            participant_means[key].append(float(val))

    group_means = []
    group_sems = []
    for lv in unique_levels:
        key = str(lv)
        vals = participant_means[key]
        if len(vals) == 0:
            group_means.append(float("nan"))
            group_sems.append(float("nan"))
        else:
            arr = np.array(vals)
            group_means.append(float(np.mean(arr)))
            sem = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            group_sems.append(sem)

    return {
        "outcome_label": outcome_label,
        "levels": [str(lv) for lv in unique_levels],
        "group_means": group_means,
        "group_sems": group_sems,
        "participant_means": participant_means,
    }


def _collect_discovery_run(
    run_dir: Path,
    cfg: BenchmarkConfig,
    seed: int,
    outcome_label: str,
) -> Optional[dict]:
    """
    Read discovery_results.yaml from run_dir / cfg.name and return a
    standardised discovery-run dict.  Returns None if the file is missing.
    """
    results_path = run_dir / cfg.name / "discovery_results.yaml"
    if not results_path.exists():
        print(f"  [WARN] discovery_results.yaml not found at {results_path}")
        return None

    with open(results_path) as fh:
        results = yaml.safe_load(fh)

    # Build effects list
    effects = results.get("effects", [])

    # Build discovered factors with participant-level data
    discovered_factors = []
    for f in results.get("discovered_factors", []):
        factor_name = f.get("name", "")
        levels = f.get("levels", [])
        compute_code = f.get("compute_code", "")
        sweetpea_code = f.get("sweetpea_code", "")
        validation_improvement = float(f.get("validation_improvement", 0.0))
        proposer = f.get("proposer", "llm")

        # Retrieve llm_name / llm_interpretation from the effects list (main effects)
        llm_name = ""
        llm_interpretation = ""
        for eff in effects:
            if eff.get("type") == "main" and eff.get("factor") == factor_name:
                llm_name = eff.get("llm_name", "")
                llm_interpretation = eff.get("llm_interpretation", "")
                break

        # Compute participant-level data
        factor_info = {
            "name": factor_name,
            "levels": levels,
            "compute_code": compute_code,
        }
        pld = _compute_participant_level_data(factor_info, cfg, outcome_label)

        discovered_factors.append({
            "name": factor_name,
            "levels": levels,
            "compute_code": compute_code,
            "sweetpea_code": sweetpea_code,
            "validation_improvement": validation_improvement,
            "proposer": proposer,
            "llm_name": llm_name,
            "llm_interpretation": llm_interpretation,
            "participant_level_data": pld,
        })

    # Find best factor name (highest validation_improvement)
    best_factor_name = ""
    if discovered_factors:
        best_idx = max(
            range(len(discovered_factors)),
            key=lambda i: discovered_factors[i]["validation_improvement"],
        )
        best_factor_name = discovered_factors[best_idx]["name"]

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "discovered_factors": discovered_factors,
        "effects": effects,
        "best_factor_name": best_factor_name,
    }


def _save_results(output_path: Path, results: dict) -> None:
    """Save the results dict as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[INFO] Results saved → {output_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run empirical benchmarks and discovery, then aggregate results."
    )
    parser.add_argument(
        "--config",
        default="config/produce_empirical.yaml",
        help="Path to produce_empirical.yaml",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=None,
        help="Override dataset list (default: from config)",
    )
    parser.add_argument(
        "--n-benchmark-runs",
        type=int,
        default=None,
        help="Number of benchmark runs per dataset (default: from config)",
    )
    parser.add_argument(
        "--n-discovery-runs",
        type=int,
        default=None,
        help="Number of discovery runs per dataset (default: from config or 3)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Base random seed (default: from config or 0)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory in which to save the output JSON (default: from config)",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Filename for the output JSON (default: from config or empirical_results.json)",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmark runs (only run discovery)",
    )
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Skip discovery runs (only run benchmarks)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing results JSON to resume from (skips already-completed runs)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load produce config
    # ------------------------------------------------------------------
    prod_cfg = _load_produce_config(args.config)

    datasets = args.datasets or prod_cfg.get("datasets", list(DATASET_CONFIGS.keys()))
    n_benchmark_runs = args.n_benchmark_runs or prod_cfg.get("n_benchmark_runs", 10)
    n_discovery_runs = args.n_discovery_runs or prod_cfg.get("n_discovery_runs", 3)
    base_seed = args.base_seed if args.base_seed is not None else prod_cfg.get("base_seed", 0)
    output_dir = Path(args.output_dir or prod_cfg.get("output_dir", "results/empirical"))
    output_name = args.output_name or prod_cfg.get("output_name", "empirical_results.json")
    output_path = output_dir / output_name

    # Copy the produce config into the output directory for reproducibility.
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, output_dir / Path(args.config).name)

    # Shared defaults for each phase, drawn from the produce config.
    # benchmark_shared: merged into each dataset config for run_single_benchmark calls.
    # discovery_shared: merged into each dataset config for run_single_discovery calls.
    bm_defaults: Optional[dict] = prod_cfg.get("benchmark_shared") or None
    disc_defaults: Optional[dict] = prod_cfg.get("discovery_shared") or None

    # ------------------------------------------------------------------
    # Resume: load existing results
    # ------------------------------------------------------------------
    if args.resume and Path(args.resume).exists():
        print(f"[INFO] Resuming from {args.resume}")
        with open(args.resume) as fh:
            aggregated = json.load(fh)
    else:
        aggregated = {
            "generated_at": datetime.now().isoformat(),
            "n_benchmark_runs": n_benchmark_runs,
            "n_discovery_runs": n_discovery_runs,
            "base_seed": base_seed,
            "datasets": {},
        }

    aggregated["generated_at"] = datetime.now().isoformat()
    aggregated["n_benchmark_runs"] = n_benchmark_runs
    aggregated["n_discovery_runs"] = n_discovery_runs
    aggregated["base_seed"] = base_seed

    # ------------------------------------------------------------------
    # Iterate over datasets
    # ------------------------------------------------------------------
    for ds_name in datasets:
        if ds_name not in DATASET_CONFIGS:
            print(f"[WARN] Unknown dataset '{ds_name}' — skipping.")
            continue

        config_path = DATASET_CONFIGS[ds_name]
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}  (config: {config_path})")
        print(f"{'='*70}")

        cfg = load_config(config_path, defaults=bm_defaults)

        display_name = _display_name(cfg)
        n_participants = cfg.dataset.n_participants if cfg.dataset else None
        gt_factors = _ground_truth_factors(cfg)
        out_label = _outcome_label(cfg)

        # Initialise dataset entry if not already present
        if ds_name not in aggregated["datasets"]:
            aggregated["datasets"][ds_name] = {
                "config_path": config_path,
                "display_name": display_name,
                "n_participants": n_participants,
                "ground_truth_factors": gt_factors,
                "benchmark_runs": [],
                "discovery_runs": [],
                "best_discovery_run_idx": None,
            }
        ds_entry = aggregated["datasets"][ds_name]

        # Update display fields in case they changed
        ds_entry["config_path"] = config_path
        ds_entry["display_name"] = display_name
        ds_entry["n_participants"] = n_participants
        ds_entry["ground_truth_factors"] = gt_factors

        # ------------------------------------------------------------------
        # Benchmark runs
        # ------------------------------------------------------------------
        if not args.skip_benchmark:
            existing_bm_seeds = {r["seed"] for r in ds_entry.get("benchmark_runs", [])}
            for run_idx in range(n_benchmark_runs):
                seed = base_seed + run_idx

                if seed in existing_bm_seeds:
                    print(f"\n  [SKIP] Benchmark run seed={seed} already completed.")
                    continue

                print(f"\n  Benchmark run {run_idx + 1}/{n_benchmark_runs}  (seed={seed})")

                # Build a seeded config
                seeded_cfg = load_config(config_path, defaults=bm_defaults)
                seeded_cfg.seed = seed

                # Create a unique run directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = Path(seeded_cfg.output_dir) / f"run_{timestamp}"
                run_dir.mkdir(parents=True, exist_ok=True)

                try:
                    run_single_benchmark(seeded_cfg, run_dir, regenerate=False)
                except Exception as exc:
                    print(f"  [ERROR] Benchmark run failed: {exc}")
                    continue

                bm_result = _collect_benchmark_run(run_dir, seeded_cfg, seed)
                if bm_result is not None:
                    ds_entry["benchmark_runs"].append(bm_result)
                    print(f"  Collected benchmark result: P={bm_result['precision']:.3f}  "
                          f"R={bm_result['recall']:.3f}  F1={bm_result['f1']:.3f}")

                # Save after each run
                _save_results(output_path, aggregated)

        # ------------------------------------------------------------------
        # Discovery runs
        # ------------------------------------------------------------------
        if not args.skip_discovery:
            existing_disc_seeds = {r["seed"] for r in ds_entry.get("discovery_runs", [])}
            for run_idx in range(n_discovery_runs):
                seed = base_seed + run_idx

                if seed in existing_disc_seeds:
                    print(f"\n  [SKIP] Discovery run seed={seed} already completed.")
                    continue

                print(f"\n  Discovery run {run_idx + 1}/{n_discovery_runs}  (seed={seed})")

                # Build a seeded config
                seeded_cfg = load_config(config_path, defaults=disc_defaults)
                seeded_cfg.seed = seed

                # Create a unique run directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_dir = Path(seeded_cfg.output_dir) / f"discovery_{timestamp}"
                run_dir.mkdir(parents=True, exist_ok=True)

                try:
                    run_single_discovery(seeded_cfg, run_dir)
                except Exception as exc:
                    print(f"  [ERROR] Discovery run failed: {exc}")
                    continue

                disc_result = _collect_discovery_run(run_dir, seeded_cfg, seed, out_label)
                if disc_result is not None:
                    ds_entry["discovery_runs"].append(disc_result)
                    n_found = len(disc_result["discovered_factors"])
                    print(f"  Collected discovery result: {n_found} factors discovered.")

                # Save after each run
                _save_results(output_path, aggregated)

        # ------------------------------------------------------------------
        # Determine best discovery run
        # ------------------------------------------------------------------
        disc_runs = ds_entry.get("discovery_runs", [])
        if disc_runs:
            best_idx = max(
                range(len(disc_runs)),
                key=lambda i: sum(
                    f.get("validation_improvement", 0.0)
                    for f in disc_runs[i].get("discovered_factors", [])
                ),
            )
            ds_entry["best_discovery_run_idx"] = best_idx
        else:
            ds_entry["best_discovery_run_idx"] = None

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    _save_results(output_path, aggregated)
    print("\n[DONE] All runs complete.")


if __name__ == "__main__":
    main()
