"""
End-to-end benchmark runner for the factor discovery pipeline.

Usage:
    # Single benchmark (synthetic or empirical)
    python run_benchmark.py --config config/empirical_stroop_congruency.yaml

    # Multi-benchmark run config (synthetic and/or empirical can be mixed)
    python run_benchmark.py --config config/benchmark.yaml

    # Force regeneration of synthetic data
    python run_benchmark.py --config config/benchmark.yaml --regenerate

Supported modes (set via benchmark.mode in each config file):
    synthetic_benchmark — generate or load synthetic data, evaluate against ground truth
    empirical_benchmark — load a real CSV, strip hidden factors, evaluate recovery

Steps per benchmark:
    1. Load or generate data
    2. Run the multi-round LLM discovery pipeline
    3. Evaluate discovered factors against the ground truth
    4. Save an evaluation report to results/<run_timestamp>/
"""

import argparse
import dataclasses
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.analysis.evaluation import match_factors_bijection
from src.analysis.model_comparison import compute_final_model_statistics
from src.analysis.plotting import plot_all_effects
from src.data_generation import get_data_generator, load_empirical_data
from src.discovery.factor_registry import FactorRegistry
from src.discovery.llm_client import LLMClient
from src.discovery.pipeline import run_discovery_pipeline
from src.utils.config import BenchmarkConfig, load_config, load_run_config


def _build_baseline_formula(cfg: BenchmarkConfig) -> str:
    """
    Build the null formula from base_factors.

    Honours dataset.null_formula when set (empirical configs).
    Factors marked include_in_formula=False are excluded from the regression
    baseline to avoid perfect collinearity.
    """
    if cfg.dataset and cfg.dataset.null_formula:
        return cfg.dataset.null_formula
    outcome = cfg.outcome_variable
    terms = []
    for bf in cfg.base_factors:
        if not bf.include_in_formula:
            continue
        if bf.dtype == "categorical":
            terms.append(f"C({bf.name})")
        else:
            terms.append(bf.name)
    return f"{outcome} ~ " + " + ".join(terms) if terms else f"{outcome} ~ 1"


def _load_data(cfg: BenchmarkConfig, regenerate: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (full_df, input_df) for the given config.

    full_df  — all relevant columns including ground-truth hidden factor columns
               (used for evaluation only; never passed to the pipeline).
    input_df — observable columns only (passed to the discovery pipeline).
    """
    if cfg.mode == "empirical_benchmark":
        print(f"\nLoading empirical data from {cfg.dataset.path} …")
        full_df, input_df = load_empirical_data(cfg)
        n_part = full_df["participant_id"].nunique()
        print(f"  {len(full_df):,} rows, {n_part} participants, "
              f"{len(input_df.columns)} observable columns")
        return full_df, input_df

    # synthetic_benchmark: generate or load from disk
    name = cfg.name
    full_path  = Path(f"data/ground_truth/{name}_full.csv")
    input_path = Path(f"data/input/{name}_input.csv")
    dg_cfg     = cfg.data_generation
    generator  = get_data_generator(cfg.benchmark_type)

    needs_regen = regenerate or not full_path.exists()
    if not needs_regen and full_path.exists():
        existing_n = pd.read_csv(full_path, usecols=["participant_id"])["participant_id"].nunique()
        if existing_n != dg_cfg.n_participants:
            print(f"\n  Config specifies {dg_cfg.n_participants} participants but existing data "
                  f"has {existing_n} — regenerating.")
            needs_regen = True

    if needs_regen:
        print("\nGenerating synthetic data …")
        full_df, _input_df = generator.generate(cfg)

        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(full_path, index=False)

        input_path.parent.mkdir(parents=True, exist_ok=True)
        _input_df.to_csv(input_path, index=False)

        print(f"  Ground truth → {full_path}  ({len(full_df)} rows)")
        print(f"  Discovery input → {input_path}  ({len(_input_df)} rows, {len(_input_df.columns)} cols)")
    else:
        print(f"\nLoading existing data from {full_path} ({dg_cfg.n_participants} participants)")
        full_df = pd.read_csv(full_path)

    # Always read input from CSV to guarantee consistent dtypes / column order
    input_df = pd.read_csv(input_path)
    return full_df, input_df


def run_single_benchmark(
    cfg: BenchmarkConfig,
    run_dir: Path,
    regenerate: bool,
) -> None:
    if cfg.dataset is None and cfg.data_generation is None:
        raise ValueError(f"Config '{cfg.name}' has neither a 'dataset' nor a 'data_generation' section.")

    print(f"\n{'#'*60}")
    print(f"  Benchmark : {cfg.name}  (mode={cfg.mode})")
    print(f"{'#'*60}")

    # ------------------------------------------------------------------
    # Step 1: Data loading / generation
    # ------------------------------------------------------------------
    full_df, input_df = _load_data(cfg, regenerate)

    # ------------------------------------------------------------------
    # Step 2: Discovery pipeline
    # ------------------------------------------------------------------
    print("\nRunning discovery pipeline …")
    llm = LLMClient(model=cfg.llm.model)

    baseline_formula = _build_baseline_formula(cfg)
    print(f"  Baseline formula: {baseline_formula}")

    bm_dir = run_dir / cfg.name
    bm_dir.mkdir(parents=True, exist_ok=True)

    config_path = bm_dir / "benchmark_config.yaml"
    config_path.write_text(yaml.dump(dataclasses.asdict(cfg), default_flow_style=False, sort_keys=False))
    print(f"  Config snapshot → {config_path}")

    registry = FactorRegistry(baseline_formula=baseline_formula)
    registry = run_discovery_pipeline(input_df, cfg, llm, registry, str(bm_dir))

    # ------------------------------------------------------------------
    # Step 3: Final model statistics (LRT on full dataset)
    # ------------------------------------------------------------------
    print("\nComputing final model statistics …")

    # Build analysis_df once: input_df + all discovered factor columns.
    analysis_df = input_df.copy()
    for _f in registry.discovered:
        if _f.column_name not in analysis_df.columns:
            analysis_df[_f.column_name] = _f.column_values

    final_stats = compute_final_model_statistics(
        analysis_df, baseline_formula,
        registry.discovered, registry.discovered_effects,
    )
    # Strip the identifier keys so they don't duplicate fields already present
    # in the report dicts when the stats are spread in.
    factor_stats_by_name = {
        s["name"]: {k: v for k, v in s.items() if k != "name"}
        for s in final_stats["factors"]
    }
    effect_stats_by_term = {
        s["term"]: {k: v for k, v in s.items() if k != "term"}
        for s in final_stats["interactions"]
    }

    # ------------------------------------------------------------------
    # Step 3b: Effect plots
    # ------------------------------------------------------------------
    print("\nGenerating effect plots …")
    factor_class_lookup = {
        bf.name: ("discrete" if bf.dtype == "categorical" else "continuous")
        for bf in cfg.base_factors
    }
    for _f in registry.discovered:
        factor_class_lookup[_f.column_name] = _f.candidate.factor_class
    plot_all_effects(
        df=analysis_df,
        discovered_factors=registry.discovered,
        discovered_effects=registry.discovered_effects,
        factor_class_lookup=factor_class_lookup,
        outcome_col=cfg.outcome_variable,
        participant_col="participant_id",
        output_dir=bm_dir,
        factor_stats_by_name=factor_stats_by_name,
        effect_stats_by_term=effect_stats_by_term,
    )

    # ------------------------------------------------------------------
    # Step 4: Evaluation
    # ------------------------------------------------------------------
    print("\nEvaluating results …")
    ev_cfg = cfg.evaluation
    report = match_factors_bijection(
        ground_truth_factors=ev_cfg.ground_truth_factors,
        discovered_factors=registry.discovered,
        full_df=full_df,
        threshold=ev_cfg.bijection_threshold,
        continuous_threshold=ev_cfg.continuous_correlation_threshold,
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
    print(f"{'='*50}")

    if registry.discovered_effects:
        print(f"\n  Discovered interaction effects ({len(registry.discovered_effects)}):")
        for e in registry.discovered_effects:
            print(f"    {e.term}  (round {e.round_num}, cv={e.cv_score_mean:.4f}, val={e.validation_improvement:.4f})")
    else:
        print("\n  No interaction effects discovered.")

    # ------------------------------------------------------------------
    # Step 5: Save report
    # ------------------------------------------------------------------
    gt_interactions = ev_cfg.ground_truth_interactions
    discovered_interaction_keys = {tuple(sorted(e.factor_names)) for e in registry.discovered_effects}
    gt_interaction_results = [
        {"factors": gti.factors, "matched": tuple(sorted(gti.factors)) in discovered_interaction_keys}
        for gti in gt_interactions
    ]
    n_gt_int = len(gt_interactions)
    n_matched_int = sum(1 for r in gt_interaction_results if r["matched"])
    interaction_recall = n_matched_int / n_gt_int if n_gt_int > 0 else None

    report_dict = {
        "config":            str(cfg.name),
        "mode":              cfg.mode,
        "factor_evaluation": {
            "precision":              report.precision,
            "recall":                 report.recall,
            "f1":                     report.f1,
            "n_ground_truth":         report.n_ground_truth,
            "n_discovered":           report.n_discovered,
            "matched_pairs": [
                {"ground_truth": p.ground_truth_name,
                 "discovered":   p.discovered_name,
                 "agreement":    round(p.agreement_rate, 4)}
                for p in report.matched_pairs
            ],
            "unmatched_ground_truth": report.unmatched_ground_truth,
            "unmatched_discovered":   report.unmatched_discovered,
            "discovered_factors": [
                {"name":                   f.column_name,
                 "validation_improvement": f.validation_improvement,
                 "formula":                f.formula_with,
                 "sweetpea_code":          f.candidate.sweetpea_code,
                 "compute_code":           f.candidate.compute_code,
                 **factor_stats_by_name.get(f.column_name, {})}
                for f in registry.discovered
            ],
        },
        "interaction_evaluation": {
            "discovered": [
                {"term": e.term, "factor_names": e.factor_names,
                 "round": e.round_num, "cv_improvement": e.cv_score_mean,
                 "validation_improvement": e.validation_improvement,
                 "source": e.source, "llm_rationale": e.llm_rationale,
                 **effect_stats_by_term.get(e.term, {})}
                for e in registry.discovered_effects
            ],
            "ground_truth_interactions": gt_interaction_results,
            "interaction_recall":        interaction_recall,
        },
        "precision": report.precision,
        "recall":    report.recall,
        "f1":        report.f1,
    }

    report_path = bm_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))
    print(f"\nReport saved → {report_path}\n")


def _resolve_configs(config_path: str):
    """
    Return (benchmark_paths, shared_defaults).

    If config_path is a benchmark.yaml (contains 'benchmarks'), resolve relative
    paths and extract shared defaults from the remaining keys.
    If it is a single benchmark YAML, return ([config_path], {}).
    """
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    if "benchmarks" in raw:
        base = Path(config_path).parent
        paths = [
            str(base.parent / p) if not Path(p).is_absolute() else p
            for p in raw["benchmarks"]
        ]
        shared = {k: v for k, v in raw.items() if k != "benchmarks"}
        return paths, shared
    return [config_path], {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Factor discovery benchmark runner")
    parser.add_argument("--config", default="config/benchmark.yaml",
                        help="Path to benchmark.yaml (multi-benchmark) or a single benchmark YAML")
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regeneration of synthetic data")
    args = parser.parse_args()

    benchmark_paths, shared_defaults = _resolve_configs(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_cfg = load_config(benchmark_paths[0], defaults=shared_defaults or None)
    run_dir   = Path(first_cfg.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output directory: {run_dir}")

    for bm_path in benchmark_paths:
        cfg = load_config(bm_path, defaults=shared_defaults or None)
        run_single_benchmark(cfg, run_dir, regenerate=args.regenerate)


if __name__ == "__main__":
    main()
