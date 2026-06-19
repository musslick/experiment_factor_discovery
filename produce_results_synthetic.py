"""
produce_results_synthetic.py

Runs each of 3 synthetic benchmarks N times (with different seeds), collects
results from evaluation_report.json and round_*_candidates.json files, and
writes a single aggregated JSON.

Usage:
    python produce_results_synthetic.py \
        --config config/produce_synthetic.yaml \
        [--benchmarks stroop_simon rdk prospect_theory] \
        [--n-runs N] \
        [--base-seed S] \
        [--output-dir DIR] \
        [--output-name NAME] \
        [--regenerate] \
        [--resume PATH]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from run_benchmark import run_single_benchmark
from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS: Dict[str, str] = {
    "stroop_simon": "config/synthetic_stroop_simon_benchmark.yaml",
    "rdk": "config/synthetic_rdk_task_switching_benchmark.yaml",
    "prospect_theory": "config/synthetic_prospect_theory_benchmark.yaml",
}

BENCHMARK_DISPLAY: Dict[str, str] = {
    "stroop_simon": "Stroop-Simon",
    "rdk": "RDK Task-Switching",
    "prospect_theory": "Prospect Theory",
}

_TYPE_MAP = {"transition": "window"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_type(t: str) -> str:
    """Map 'transition' -> 'window'; leave everything else unchanged."""
    return _TYPE_MAP.get(t, t)


def _gt_factor_dicts(cfg) -> List[Dict[str, Any]]:
    """Return ground-truth factor list in the output JSON schema format."""
    result = []
    for gtf in cfg.evaluation.ground_truth_factors:
        result.append({
            "name": gtf.name,
            "type": _normalise_type(gtf.type),
            "factor_class": gtf.factor_class,
            "n_levels": len(gtf.levels),
            "levels": list(gtf.levels),
        })
    return result


def _load_round_logs(report_dir: Path) -> List[Dict[str, Any]]:
    """
    Read all round_*_candidates.json files from *report_dir* and return a
    sorted list of round-log dicts.
    """
    files = sorted(report_dir.glob("round_*_candidates.json"))
    logs = []
    cumulative_calls = 0
    for fpath in files:
        try:
            with open(fpath) as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"    Warning: could not read {fpath}: {exc}")
            continue

        all_scored = data.get("all_scored", [])
        hard_rejected = data.get("hard_rejected", [])
        n_scored = len(all_scored)
        n_hard_rejected = len(hard_rejected)
        n_candidates_this_round = n_scored + n_hard_rejected
        cumulative_calls += n_candidates_this_round

        winner = data.get("winner") or {}
        proposer_counts: Dict[str, int] = {}
        for sc in all_scored:
            p = sc.get("proposer", "llm")
            proposer_counts[p] = proposer_counts.get(p, 0) + 1
        logs.append({
            "round": data.get("round", len(logs) + 1),
            "accepted": bool(data.get("accepted", False)),
            "n_scored": n_scored,
            "n_hard_rejected": n_hard_rejected,
            "winner_cv_mean": float(winner.get("cv_score_mean", 0.0)) if winner else 0.0,
            "winner_proposer": winner.get("proposer", "llm") if winner else None,
            "validation_improvement": float(data.get("validation_improvement", 0.0)),
            "cumulative_synthesis_calls": cumulative_calls,
            "n_llm_scored": proposer_counts.get("llm", 0),
            "n_random_seeder_scored": proposer_counts.get("random_seeder", 0),
            "n_random_lookup_seeder_scored": proposer_counts.get("random_lookup_seeder", 0),
        })
    return logs


def _extract_run_result(
    report_dir: Path,
    cfg,
    seed: int,
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Parse evaluation_report.json and round logs from *report_dir* and return
    the run-result dict matching the output schema.
    """
    report_path = report_dir / "evaluation_report.json"
    with open(report_path) as fh:
        report = json.load(fh)

    fe = report["factor_evaluation"]

    precision = float(fe.get("precision", 0.0))
    recall = float(fe.get("recall", 0.0))
    f1 = float(fe.get("f1", 0.0))
    n_ground_truth = int(fe.get("n_ground_truth", 0))
    n_discovered = int(fe.get("n_discovered", 0))
    matched_pairs = fe.get("matched_pairs", [])
    unmatched_gt = fe.get("unmatched_ground_truth", [])
    unmatched_disc = fe.get("unmatched_discovered", [])

    # Build lookup: gt_name -> agreement from matched_pairs
    gt_agreement: Dict[str, float] = {}
    for mp in matched_pairs:
        gt_agreement[mp["ground_truth"]] = float(mp.get("agreement", 0.0))

    # Level recovery and continuous correlation
    level_recovery_per_factor: Dict[str, Dict[str, Any]] = {}
    continuous_correlation_per_factor: Dict[str, float] = {}

    gt_factors_by_name = {gtf.name: gtf for gtf in cfg.evaluation.ground_truth_factors}

    level_recovery_raw = fe.get("level_recovery")

    if level_recovery_raw is not None:
        # Group entries by ground_truth factor name
        levels_list = level_recovery_raw.get("levels", [])
        grouped: Dict[str, List[dict]] = {}
        for entry in levels_list:
            gt_name = entry.get("ground_truth", "")
            grouped.setdefault(gt_name, []).append(entry)

        for gt_name, entries in grouped.items():
            gtf = gt_factors_by_name.get(gt_name)
            factor_class = gtf.factor_class if gtf else "discrete"
            if factor_class == "discrete":
                n_total = len(entries)
                n_recovered = sum(1 for e in entries if e.get("recovered", False))
                level_recall = n_recovered / n_total if n_total > 0 else 0.0
                level_recovery_per_factor[gt_name] = {
                    "n_levels": n_total,
                    "n_recovered": n_recovered,
                    "level_recall": level_recall,
                }
            # Continuous factors from level_recovery go into continuous_correlation
            else:
                # Use agreement from matched_pairs as correlation proxy
                continuous_correlation_per_factor[gt_name] = gt_agreement.get(gt_name, 0.0)
    else:
        # Fallback: use matched_pairs agreement
        for gtf in cfg.evaluation.ground_truth_factors:
            if gtf.factor_class == "discrete":
                n_levels = len(gtf.levels)
                agreement = gt_agreement.get(gtf.name, 0.0)
                # agreement as level_recall proxy; n_recovered estimated from agreement
                n_recovered = round(agreement * n_levels) if n_levels > 0 else 0
                level_recovery_per_factor[gtf.name] = {
                    "n_levels": n_levels,
                    "n_recovered": n_recovered,
                    "level_recall": agreement,
                }
            else:
                continuous_correlation_per_factor[gtf.name] = gt_agreement.get(gtf.name, 0.0)

    # Round logs
    round_logs = _load_round_logs(report_dir)
    n_rounds_run = len(round_logs)
    n_synthesis_calls = round_logs[-1]["cumulative_synthesis_calls"] if round_logs else 0

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_ground_truth": n_ground_truth,
        "n_discovered": n_discovered,
        "matched_pairs": matched_pairs,
        "level_recovery_per_factor": level_recovery_per_factor,
        "continuous_correlation_per_factor": continuous_correlation_per_factor,
        "unmatched_ground_truth": unmatched_gt,
        "unmatched_discovered": unmatched_disc,
        "n_synthesis_calls": n_synthesis_calls,
        "n_rounds_run": n_rounds_run,
        "round_logs": round_logs,
    }


def _already_done(output: Dict[str, Any], benchmark_key: str, seed: int) -> bool:
    """Return True if this benchmark+seed is already recorded in *output*."""
    runs = output.get("benchmarks", {}).get(benchmark_key, {}).get("runs", [])
    return any(r["seed"] == seed for r in runs)


def _save_output(output: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run synthetic benchmarks multiple times and aggregate results."
    )
    parser.add_argument(
        "--config",
        default="config/produce_synthetic.yaml",
        help="YAML config with benchmarks, n_runs, base_seed, output_dir, output_name",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(BENCHMARK_CONFIGS.keys()),
        help="Override: which benchmarks to run",
    )
    parser.add_argument("--n-runs", type=int, help="Override: number of runs per benchmark")
    parser.add_argument("--base-seed", type=int, help="Override: base seed (seed = base_seed + run_idx)")
    parser.add_argument("--output-dir", help="Override: output directory")
    parser.add_argument("--output-name", help="Override: output JSON filename")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of synthetic data on each run",
    )
    parser.add_argument(
        "--resume",
        metavar="PATH",
        help="Load existing aggregated JSON and skip already-completed seeds",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load YAML config (if it exists), then apply CLI overrides
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    yaml_cfg: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as fh:
            yaml_cfg = yaml.safe_load(fh) or {}
    else:
        print(f"Config file {config_path} not found; using defaults / CLI args only.")

    benchmarks: List[str] = args.benchmarks or yaml_cfg.get("benchmarks", list(BENCHMARK_CONFIGS.keys()))
    n_runs: int = args.n_runs if args.n_runs is not None else int(yaml_cfg.get("n_runs", 3))
    base_seed: int = args.base_seed if args.base_seed is not None else int(yaml_cfg.get("base_seed", 0))
    output_dir: str = args.output_dir or yaml_cfg.get("output_dir", "results/synthetic_aggregated")
    output_name: str = args.output_name or yaml_cfg.get("output_name", "aggregated_results.json")

    output_path = Path(output_dir) / output_name

    # Extract shared defaults for load_config(): everything in the produce YAML
    # except the produce-specific top-level keys.
    _PRODUCE_KEYS = {"benchmarks", "n_runs", "base_seed", "output_dir", "output_name"}
    shared_defaults: Dict[str, Any] = {k: v for k, v in yaml_cfg.items() if k not in _PRODUCE_KEYS} or None

    # ------------------------------------------------------------------
    # Resume from existing file
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": n_runs,
        "base_seed": base_seed,
        "benchmarks": {},
    }
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            with open(resume_path) as fh:
                output = json.load(fh)
            print(f"Resuming from {resume_path}")
        else:
            print(f"Resume file {resume_path} not found; starting fresh.")

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    for bm_key in benchmarks:
        config_yaml_path = BENCHMARK_CONFIGS[bm_key]
        display_name = BENCHMARK_DISPLAY[bm_key]

        # Load config once to get ground-truth factors and benchmark name
        cfg_template = load_config(config_yaml_path, defaults=shared_defaults)
        bm_name = cfg_template.name  # e.g. "stroop_simon_factor_discovery"

        # Initialise benchmark entry in output if not present
        if bm_key not in output["benchmarks"]:
            output["benchmarks"][bm_key] = {
                "config_path": config_yaml_path,
                "display_name": display_name,
                "ground_truth_factors": _gt_factor_dicts(cfg_template),
                "runs": [],
            }

        for run_idx in range(n_runs):
            seed = base_seed + run_idx

            if _already_done(output, bm_key, seed):
                print(
                    f"[{display_name}] Run {run_idx + 1}/{n_runs} (seed={seed}) — already done, skipping."
                )
                continue

            print(f"\n[{display_name}] Run {run_idx + 1}/{n_runs} (seed={seed}) ...")

            # Build a fresh cfg with the correct seed
            cfg = load_config(config_yaml_path, defaults=shared_defaults)
            cfg.seed = seed

            # Each run gets its own subdirectory so logs don't collide
            run_tag = f"{bm_key}_seed{seed}"
            run_dir = Path(output_dir) / "runs" / run_tag
            run_dir.mkdir(parents=True, exist_ok=True)

            try:
                run_single_benchmark(cfg, run_dir, regenerate=args.regenerate)
            except Exception as exc:
                print(f"  ERROR running {display_name} seed={seed}: {exc}", file=sys.stderr)
                # Save progress so far and continue
                _save_output(output, output_path)
                continue

            # Collect results
            report_dir = run_dir / bm_name
            try:
                run_result = _extract_run_result(report_dir, cfg, seed, run_dir)
            except Exception as exc:
                print(
                    f"  ERROR parsing results for {display_name} seed={seed}: {exc}",
                    file=sys.stderr,
                )
                _save_output(output, output_path)
                continue

            output["benchmarks"][bm_key]["runs"].append(run_result)

            # Update timestamp and save immediately (crash recovery)
            output["generated_at"] = datetime.now(timezone.utc).isoformat()
            _save_output(output, output_path)
            print(f"  Saved intermediate results to {output_path}")

    # Final save
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    _save_output(output, output_path)
    print(f"\nAggregated results written to {output_path}")


if __name__ == "__main__":
    main()
