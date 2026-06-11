"""
Novel factor discovery runner.

Loads an empirical dataset with ALL known factors visible (base + hidden),
runs the multi-round LLM discovery pipeline to find genuinely new factors,
then asks the LLM to name and interpret every discovered effect.

Usage:
    # Single dataset config
    python run_discovery.py --config config/novel_stroop_congruency.yaml

    # Multi-dataset discovery config
    python run_discovery.py --config config/discovery.yaml

Output per dataset:
    results/discovery_<timestamp>/<name>/discovery_results.yaml

The output YAML contains:
    - discovered_factors   : name, levels, sweetpea_code, compute_code
    - existing_factors_in_final_model : base_factors present in the final formula
    - final_model          : the final regression formula
    - effects              : each new main / interaction effect with llm_name + llm_interpretation
"""

import copy
import argparse
import json
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.analysis.model_comparison import compute_final_model_statistics
from src.analysis.plotting import plot_all_effects
from src.data_generation import load_empirical_data
from src.discovery.factor_registry import DiscoveredEffect, DiscoveredFactor, FactorRegistry
from src.discovery.llm_client import LLMClient, make_llm_client
from src.discovery.pipeline import run_discovery_pipeline
from src.utils.config import BaseFactor, BenchmarkConfig, load_config

_PROMPT_DIR = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Baseline formula
# ---------------------------------------------------------------------------

def _build_discovery_baseline_formula(cfg: BenchmarkConfig) -> str:
    """
    Build the starting formula for novel discovery.

    Honours dataset.full_formula when set; otherwise auto-builds from all
    base_factors (which already include the formerly-hidden known factors in
    novel_discovery configs).
    """
    if cfg.dataset and cfg.dataset.full_formula:
        return cfg.dataset.full_formula
    outcome = cfg.outcome_variable
    terms = [
        f"C({bf.name})" if bf.dtype == "categorical" else bf.name
        for bf in cfg.base_factors
        if bf.include_in_formula
    ]
    return f"{outcome} ~ " + " + ".join(terms) if terms else f"{outcome} ~ 1"


# ---------------------------------------------------------------------------
# Effect naming
# ---------------------------------------------------------------------------

def _load_prompt(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8")


def _parse_naming_response(raw: str) -> List[dict]:
    """Extract a JSON array from the LLM response; return [] on failure."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        bracket = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket:
            text = bracket.group(0)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return []


def _name_and_interpret_effects(
    registry: FactorRegistry,
    cfg: BenchmarkConfig,
    llm: LLMClient,
) -> List[dict]:
    """
    Call the LLM to assign a human-readable name and interpretation to each
    newly discovered main effect and interaction effect.

    Returns a list of dicts with keys: term, llm_name, llm_interpretation.
    """
    if not registry.discovered and not registry.discovered_effects:
        return []

    effects_data = []

    for f in registry.discovered:
        term = (
            f"C({f.column_name})"
            if f.candidate.factor_class == "discrete"
            else f.column_name
        )
        effects_data.append({
            "term": term,
            "type": "main",
            "factor_name": f.column_name,
            "levels": f.candidate.levels,
            "description": f.candidate.description,
            "cv_improvement": round(f.validation_improvement or 0.0, 4),
            "validation_improvement": round(f.validation_improvement or 0.0, 4),
        })

    for e in registry.discovered_effects:
        effects_data.append({
            "term": e.term,
            "type": "interaction",
            "factor_names": e.factor_names,
            "cv_improvement": round(e.cv_score_mean, 4),
            "validation_improvement": round(e.validation_improvement, 4),
        })

    system = _load_prompt("effect_naming_system.txt")
    user = (
        _load_prompt("effect_naming_user.txt")
        .replace("<<task_context>>", cfg.task_context.strip())
        .replace("<<final_formula>>", registry.get_current_formula())
        .replace("<<effects_json>>", json.dumps(effects_data, indent=2))
    )

    raw = llm.complete(system=system, user=user, max_tokens=2000, temperature=0.3)
    named = _parse_naming_response(raw)

    # Index by term for fast lookup
    named_by_term = {item.get("term", ""): item for item in named if isinstance(item, dict)}
    return named_by_term


# ---------------------------------------------------------------------------
# Output YAML
# ---------------------------------------------------------------------------

def _build_output(
    cfg: BenchmarkConfig,
    registry: FactorRegistry,
    baseline_formula: str,
    named_by_term: dict,
    final_stats: dict,
) -> dict:
    """Assemble the discovery_results.yaml content as a plain dict."""

    # Strip the identifier keys ("name"/"term") so they don't duplicate fields
    # already present in the output dicts when the stats are spread in.
    factor_stats_by_name = {
        s["name"]: {k: v for k, v in s.items() if k != "name"}
        for s in final_stats.get("factors", [])
    }
    effect_stats_by_term = {
        s["term"]: {k: v for k, v in s.items() if k != "term"}
        for s in final_stats.get("interactions", [])
    }

    discovered_factors = []
    for f in registry.discovered:
        discovered_factors.append({
            "name": f.column_name,
            "levels": f.candidate.levels,
            "sweetpea_code": f.candidate.sweetpea_code or "",
            "compute_code": f.candidate.compute_code or "",
            "validation_improvement": round(f.validation_improvement or 0.0, 4),
        })

    existing_factors = [bf.name for bf in cfg.base_factors]

    effects = []
    for f in registry.discovered:
        term = (
            f"C({f.column_name})"
            if f.candidate.factor_class == "discrete"
            else f.column_name
        )
        naming = named_by_term.get(term, {})
        effects.append({
            "term": term,
            "type": "main",
            "factor": f.column_name,
            "validation_improvement": round(f.validation_improvement or 0.0, 4),
            "llm_name": naming.get("name", ""),
            "llm_interpretation": naming.get("interpretation", ""),
            **factor_stats_by_name.get(f.column_name, {}),
        })

    for e in registry.discovered_effects:
        naming = named_by_term.get(e.term, {})
        effects.append({
            "term": e.term,
            "type": "interaction",
            "factors": e.factor_names,
            "cv_improvement": round(e.cv_score_mean, 4),
            "validation_improvement": round(e.validation_improvement, 4),
            "llm_name": naming.get("name", ""),
            "llm_interpretation": naming.get("interpretation", ""),
            **effect_stats_by_term.get(e.term, {}),
        })

    return {
        "name": cfg.name,
        "dataset": cfg.dataset.path if cfg.dataset else "",
        "baseline_formula": baseline_formula,
        "final_model": {"formula": registry.get_current_formula()},
        "existing_factors_in_final_model": existing_factors,
        "discovered_factors": discovered_factors,
        "effects": effects,
    }


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_single_discovery(
    cfg: BenchmarkConfig,
    run_dir: Path,
) -> None:
    if cfg.dataset is None:
        raise ValueError(
            f"Config '{cfg.name}' has no 'dataset' section. "
            "run_discovery.py requires an empirical dataset config. "
            "Synthetic benchmark configs can only be used with run_benchmark.py."
        )

    print(f"\n{'#'*60}")
    print(f"  Discovery : {cfg.name}")
    print(f"{'#'*60}")

    # ------------------------------------------------------------------
    # Step 1: Build an augmented config that promotes hidden_factors into
    #         base_factors so the pipeline can see all known factors in its
    #         observable_descriptions (and the LLM knows their levels).
    #         With hidden_factors cleared, load_empirical_data will not strip
    #         any columns — input_df == full_df.
    # ------------------------------------------------------------------
    aug_cfg = copy.deepcopy(cfg)
    for hf in aug_cfg.dataset.hidden_factors:
        dtype = "categorical" if hf.factor_class == "discrete" else "continuous"
        aug_cfg.dataset.base_factors.append(
            BaseFactor(name=hf.name, dtype=dtype, levels=list(hf.levels))
        )
    aug_cfg.dataset.hidden_factors = []

    print(f"\nLoading dataset from {cfg.dataset.path} …")
    _, input_df = load_empirical_data(aug_cfg)
    n_part = input_df["participant_id"].nunique()
    print(f"  {len(input_df):,} rows, {n_part} participants, "
          f"{len(input_df.columns)} observable columns")

    # ------------------------------------------------------------------
    # Step 2: Discovery pipeline
    # ------------------------------------------------------------------
    print("\nRunning discovery pipeline …")
    llm = make_llm_client(aug_cfg.llm)

    baseline_formula = _build_discovery_baseline_formula(aug_cfg)
    print(f"  Baseline formula: {baseline_formula}")

    ds_dir = run_dir / cfg.name
    ds_dir.mkdir(parents=True, exist_ok=True)

    registry = FactorRegistry(baseline_formula=baseline_formula)
    registry = run_discovery_pipeline(input_df, aug_cfg, llm, registry, str(ds_dir))

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
        spec=aug_cfg.model_spec,
    )

    # ------------------------------------------------------------------
    # Step 4: Name and interpret discovered effects
    # ------------------------------------------------------------------
    print("\nNaming and interpreting discovered effects …")
    named_by_term = _name_and_interpret_effects(registry, aug_cfg, llm)

    # ------------------------------------------------------------------
    # Step 4b: Effect plots (after naming so LLM titles are available)
    # ------------------------------------------------------------------
    print("\nGenerating effect plots …")
    factor_class_lookup = {
        bf.name: ("discrete" if bf.dtype == "categorical" else "continuous")
        for bf in aug_cfg.base_factors
    }
    for _f in registry.discovered:
        factor_class_lookup[_f.column_name] = _f.candidate.factor_class

    factor_stats_by_name_plots = {
        s["name"]: {k: v for k, v in s.items() if k != "name"}
        for s in final_stats.get("factors", [])
    }
    effect_stats_by_term_plots = {
        s["term"]: {k: v for k, v in s.items() if k != "term"}
        for s in final_stats.get("interactions", [])
    }

    # Build LLM name lookup: factor column_name → llm_name, effect term → llm_name
    llm_name_lookup = {}
    for _f in registry.discovered:
        term = f"C({_f.column_name})" if _f.candidate.factor_class == "discrete" else _f.column_name
        llm_name = named_by_term.get(term, {}).get("name", "")
        if llm_name:
            llm_name_lookup[_f.column_name] = llm_name
    for _e in registry.discovered_effects:
        llm_name = named_by_term.get(_e.term, {}).get("name", "")
        if llm_name:
            llm_name_lookup[_e.term] = llm_name

    plot_all_effects(
        df=analysis_df,
        discovered_factors=registry.discovered,
        discovered_effects=registry.discovered_effects,
        factor_class_lookup=factor_class_lookup,
        outcome_col=aug_cfg.outcome_variable,
        participant_col="participant_id",
        output_dir=ds_dir,
        factor_stats_by_name=factor_stats_by_name_plots,
        effect_stats_by_term=effect_stats_by_term_plots,
        llm_name_lookup=llm_name_lookup,
    )

    # ------------------------------------------------------------------
    # Step 5: Save discovery_results.yaml
    # ------------------------------------------------------------------
    output = _build_output(aug_cfg, registry, baseline_formula, named_by_term, final_stats)
    result_path = ds_dir / "discovery_results.yaml"
    result_path.write_text(yaml.dump(output, default_flow_style=False, allow_unicode=True,
                                     sort_keys=False))
    print(f"\nDiscovery results saved → {result_path}\n")

    if registry.discovered:
        print(f"  Discovered factors ({len(registry.discovered)}):")
        for f in registry.discovered:
            print(f"    {f.column_name}  (val={f.validation_improvement:.4f})")
    else:
        print("  No new factors discovered.")

    if registry.discovered_effects:
        print(f"\n  Discovered interaction effects ({len(registry.discovered_effects)}):")
        for e in registry.discovered_effects:
            print(f"    {e.term}  (round {e.round_num}, val={e.validation_improvement:.4f})")


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _resolve_discovery_configs(config_path: str):
    """
    Return (dataset_paths, shared_defaults).

    If config_path is a discovery.yaml (contains 'datasets'), resolve relative
    paths and extract shared defaults from the remaining keys.
    If it is a single dataset config YAML, return ([config_path], {}).
    """
    with open(config_path) as fh:
        raw = yaml.safe_load(fh)
    if "datasets" in raw:
        base = Path(config_path).parent
        paths = [
            str(base.parent / p) if not Path(p).is_absolute() else p
            for p in raw["datasets"]
        ]
        shared = {k: v for k, v in raw.items() if k != "datasets"}
        return paths, shared
    return [config_path], {}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Novel factor discovery runner")
    parser.add_argument("--config", default="config/discovery.yaml",
                        help="Path to discovery.yaml (multi-dataset) or a single novel_discovery YAML")
    args = parser.parse_args()

    dataset_paths, shared_defaults = _resolve_discovery_configs(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_cfg = load_config(dataset_paths[0], defaults=shared_defaults or None)
    run_dir   = Path(first_cfg.output_dir) / f"discovery_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Discovery output directory: {run_dir}")

    for ds_path in dataset_paths:
        cfg = load_config(ds_path, defaults=shared_defaults or None)
        run_single_discovery(cfg, run_dir)


if __name__ == "__main__":
    main()
