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
import csv
import json
import re
import shutil
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
_LLM_MODEL_KEYS = {"id", "provider", "model", "ollama_base_url"}
_LLM_PROVIDERS = {"anthropic", "ollama"}
_MODEL_SUMMARY_FIELDS = (
    "benchmark",
    "model_id",
    "provider",
    "model",
    "expected",
    "completed",
    "failed",
    "mean_precision",
    "mean_recall",
    "mean_f1",
)


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
        validation_improvement = data.get("validation_improvement", 0.0)
        if validation_improvement is not None:
            validation_improvement = float(validation_improvement)
        logs.append({
            "round": data.get("round", len(logs) + 1),
            "accepted": bool(data.get("accepted", False)),
            "n_scored": n_scored,
            "n_hard_rejected": n_hard_rejected,
            "winner_cv_mean": float(winner.get("cv_score_mean", 0.0)) if winner else 0.0,
            "winner_proposer": winner.get("proposer", "llm") if winner else None,
            "validation_improvement": validation_improvement,
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


def _parse_llm_models(raw_models: Any) -> List[Dict[str, str]]:
    """Validate the optional model matrix from the produce config."""
    if raw_models is None:
        return []
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("llm_models must be a non-empty list")

    models: List[Dict[str, str]] = []
    seen_ids = set()
    for index, raw in enumerate(raw_models):
        if not isinstance(raw, dict):
            raise ValueError(f"llm_models[{index}] must be a mapping")

        unknown = set(raw) - _LLM_MODEL_KEYS
        if unknown:
            raise ValueError(
                f"llm_models[{index}] has unsupported keys: {', '.join(sorted(unknown))}"
            )

        model_id = raw.get("id")
        provider = raw.get("provider")
        model = raw.get("model")
        if not isinstance(model_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id
        ):
            raise ValueError(
                f"llm_models[{index}].id must contain only letters, numbers, '.', '_' or '-'"
            )
        if model_id in seen_ids:
            raise ValueError(f"Duplicate llm_models id: {model_id}")
        if provider not in _LLM_PROVIDERS:
            raise ValueError(
                f"llm_models[{index}].provider must be one of: "
                f"{', '.join(sorted(_LLM_PROVIDERS))}"
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"llm_models[{index}].model must be a non-empty string")

        parsed = {"id": model_id, "provider": provider, "model": model}
        base_url = raw.get("ollama_base_url")
        if base_url is not None:
            if not isinstance(base_url, str) or not base_url.strip():
                raise ValueError(
                    f"llm_models[{index}].ollama_base_url must be a non-empty string"
                )
            parsed["ollama_base_url"] = base_url

        seen_ids.add(model_id)
        models.append(parsed)

    return models


def _apply_llm_model(cfg, model_spec: Optional[Dict[str, str]]) -> None:
    if model_spec is None:
        return
    cfg.llm.provider = model_spec["provider"]
    cfg.llm.model = model_spec["model"]
    if "ollama_base_url" in model_spec:
        cfg.llm.ollama_base_url = model_spec["ollama_base_url"]


def _ensure_model_entry(
    benchmark_output: Dict[str, Any],
    model_spec: Dict[str, str],
) -> Dict[str, Any]:
    models = benchmark_output.setdefault("models", {})
    model_id = model_spec["id"]
    expected = {k: v for k, v in model_spec.items() if k != "id"}
    existing = models.get(model_id)
    if existing is None:
        existing = {**expected, "runs": [], "failures": []}
        models[model_id] = existing
    else:
        actual = {
            key: existing[key]
            for key in _LLM_MODEL_KEYS - {"id"}
            if key in existing
        }
        if actual != expected:
            raise ValueError(
                f"Cannot resume model '{model_id}': stored configuration {actual} "
                f"does not match {expected}"
            )
        existing.setdefault("runs", [])
        existing.setdefault("failures", [])
    return existing


def _record_model_failure(
    model_output: Dict[str, Any],
    seed: int,
    stage: str,
    error: Exception,
    run_dir: Path,
) -> None:
    failure = {
        "seed": seed,
        "stage": stage,
        "error": str(error),
        "run_dir": str(run_dir),
    }
    model_output["failures"] = [
        existing
        for existing in model_output.get("failures", [])
        if existing.get("seed") != seed
    ] + [failure]


def _clear_model_failure(model_output: Dict[str, Any], seed: int) -> None:
    model_output["failures"] = [
        failure
        for failure in model_output.get("failures", [])
        if failure.get("seed") != seed
    ]


def _already_done(
    output: Dict[str, Any],
    benchmark_key: str,
    seed: int,
    model_spec: Optional[Dict[str, str]] = None,
) -> bool:
    """Return True if this benchmark, model and seed are already recorded."""
    benchmark_output = output.get("benchmarks", {}).get(benchmark_key, {})
    if model_spec is None:
        runs = benchmark_output.get("runs", [])
    else:
        runs = (
            benchmark_output.get("models", {})
            .get(model_spec["id"], {})
            .get("runs", [])
        )
    return any(r["seed"] == seed for r in runs)


def _save_output(output: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))


def _write_model_summary(
    output: Dict[str, Any], output_path: Path, expected_runs: int
) -> Path:
    rows = []
    for benchmark_key, benchmark in output.get("benchmarks", {}).items():
        for model_id, model_entry in benchmark.get("models", {}).items():
            runs = model_entry.get("runs", [])

            def mean(metric: str) -> Optional[float]:
                if not runs:
                    return None
                return round(sum(float(run[metric]) for run in runs) / len(runs), 6)

            rows.append({
                "benchmark": benchmark_key,
                "model_id": model_id,
                "provider": model_entry.get("provider", ""),
                "model": model_entry.get("model", ""),
                "expected": expected_runs,
                "completed": len(runs),
                "failed": len(model_entry.get("failures", [])),
                "mean_precision": mean("precision"),
                "mean_recall": mean("recall"),
                "mean_f1": mean("f1"),
            })

    summary_path = output_path.with_name(f"{output_path.stem}_model_summary.csv")
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MODEL_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nModel summary written to {summary_path}")
    print("  " + " | ".join(_MODEL_SUMMARY_FIELDS))
    for row in rows:
        values = ("" if row[key] is None else str(row[key]) for key in _MODEL_SUMMARY_FIELDS)
        print("  " + " | ".join(values))
    return summary_path


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
        help="YAML config with benchmarks, run grid, output path, and optional llm_models",
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
    llm_models = _parse_llm_models(yaml_cfg.get("llm_models"))

    output_path = Path(output_dir) / output_name

    # Extract shared defaults for load_config(): everything in the produce YAML
    # except the produce-specific top-level keys.
    _PRODUCE_KEYS = {
        "benchmarks", "n_runs", "base_seed", "output_dir", "output_name", "llm_models",
    }
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

    stored_benchmarks = output.get("benchmarks", {})
    if stored_benchmarks:
        has_legacy_results = any(
            "runs" in benchmark for benchmark in stored_benchmarks.values()
        )
        has_matrix_results = any(
            "models" in benchmark for benchmark in stored_benchmarks.values()
        )
        if llm_models:
            if has_legacy_results:
                raise ValueError(
                    "Cannot resume single-model results with llm_models in the config"
                )
            if output.get("llm_models") != llm_models:
                raise ValueError("Cannot resume with a different llm_models matrix")
            for key, requested in (("n_runs", n_runs), ("base_seed", base_seed)):
                if output.get(key) != requested:
                    raise ValueError(f"Cannot resume with a different {key}")
        elif has_matrix_results:
            raise ValueError(
                "Cannot resume multi-model results without llm_models in the config"
            )

    if llm_models:
        output["llm_models"] = llm_models

    # Copy only after resume validation so rejected resumes leave prior artifacts intact.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, Path(output_dir) / Path(args.config).name)

    invocation_had_failures = False

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
            benchmark_output = {
                "config_path": config_yaml_path,
                "display_name": display_name,
                "ground_truth_factors": _gt_factor_dicts(cfg_template),
            }
            benchmark_output["models" if llm_models else "runs"] = {} if llm_models else []
            output["benchmarks"][bm_key] = benchmark_output
        benchmark_output = output["benchmarks"][bm_key]
        if llm_models:
            if "runs" in benchmark_output:
                raise ValueError(
                    "Cannot resume single-model results with llm_models in the config"
                )
            for entry in benchmark_output.setdefault("models", {}).values():
                entry.setdefault("failures", [])
        elif "models" in benchmark_output:
            raise ValueError(
                "Cannot resume multi-model results without llm_models in the config"
            )
        else:
            benchmark_output.setdefault("runs", [])

        model_matrix: List[Optional[Dict[str, str]]] = llm_models or [None]
        for model_spec in model_matrix:
            if model_spec is None:
                model_output = benchmark_output
                run_label = display_name
            else:
                model_output = _ensure_model_entry(benchmark_output, model_spec)
                run_label = f"{display_name} / {model_spec['id']}"

            for run_idx in range(n_runs):
                seed = base_seed + run_idx

                if _already_done(output, bm_key, seed, model_spec):
                    print(
                        f"[{run_label}] Run {run_idx + 1}/{n_runs} "
                        f"(seed={seed}) — already done, skipping."
                    )
                    continue

                print(f"\n[{run_label}] Run {run_idx + 1}/{n_runs} (seed={seed}) ...")

                # Build a fresh cfg with the correct seed and model.
                cfg = load_config(config_yaml_path, defaults=shared_defaults)
                cfg.seed = seed
                _apply_llm_model(cfg, model_spec)

                # Each run gets its own subdirectory so logs don't collide.
                run_tag = f"{bm_key}_seed{seed}"
                run_root = Path(output_dir) / "runs"
                if model_spec is not None:
                    run_root /= model_spec["id"]
                run_dir = run_root / run_tag
                run_dir.mkdir(parents=True, exist_ok=True)

                try:
                    run_single_benchmark(cfg, run_dir, regenerate=args.regenerate)
                except Exception as exc:
                    print(f"  ERROR running {run_label} seed={seed}: {exc}", file=sys.stderr)
                    if model_spec is not None:
                        _record_model_failure(
                            model_output, seed, "benchmark_execution", exc, run_dir
                        )
                        invocation_had_failures = True
                    # Save progress so far and continue
                    _save_output(output, output_path)
                    continue

                # Collect results
                report_dir = run_dir / bm_name
                try:
                    run_result = _extract_run_result(report_dir, cfg, seed, run_dir)
                except Exception as exc:
                    print(
                        f"  ERROR parsing results for {run_label} seed={seed}: {exc}",
                        file=sys.stderr,
                    )
                    if model_spec is not None:
                        _record_model_failure(
                            model_output, seed, "result_parsing", exc, run_dir
                        )
                        invocation_had_failures = True
                    _save_output(output, output_path)
                    continue

                if model_spec is not None:
                    _clear_model_failure(model_output, seed)
                model_output["runs"].append(run_result)

                # Update timestamp and save immediately (crash recovery)
                output["generated_at"] = datetime.now(timezone.utc).isoformat()
                _save_output(output, output_path)
                print(f"  Saved intermediate results to {output_path}")

    # Final save
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    _save_output(output, output_path)
    print(f"\nAggregated results written to {output_path}")
    if llm_models:
        _write_model_summary(output, output_path, n_runs)
    if invocation_had_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
