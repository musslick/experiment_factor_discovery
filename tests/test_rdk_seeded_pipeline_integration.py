"""
Integration regression for the deterministic seeded RDK discovery path.

This intentionally disables LLM evolution, contrast search, and effect search.
It verifies the headline result for the template-seeded path: random-template
seeding plus the statistical pipeline recovers the RDK hidden factors without
any model calls.
"""

import pytest

from run_benchmark import _build_baseline_formula
from src.analysis.evaluation import match_factors_bijection
from src.data_generation import get_data_generator
from src.discovery.factor_registry import FactorRegistry
from src.discovery.pipeline import run_discovery_pipeline
from src.utils.config import load_config, load_run_config


pytestmark = pytest.mark.integration


def test_seeded_rdk_pipeline_recovers_hidden_factors(tmp_path):
    run_cfg = load_run_config("config/benchmark.yaml")
    cfg = load_config(
        "config/synthetic_rdk_task_switching_benchmark.yaml",
        defaults=run_cfg.shared,
    )
    cfg.discovery.n_rounds = 4
    cfg.discovery.max_search_iterations = 1
    cfg.discovery.run_contrast_search = False
    cfg.discovery.run_effect_search = False

    assert cfg.discovery.seeding_strategy.type == "random"
    assert cfg.discovery.seeding_strategy.n_candidates == 50

    full_df, input_df = get_data_generator(cfg.benchmark_type).generate(cfg)
    registry = FactorRegistry(baseline_formula=_build_baseline_formula(cfg))
    registry = run_discovery_pipeline(
        input_df,
        cfg,
        llm=None,
        registry=registry,
        output_dir=str(tmp_path),
    )

    report = match_factors_bijection(
        ground_truth_factors=cfg.evaluation.ground_truth_factors,
        discovered_factors=registry.discovered,
        full_df=full_df,
        threshold=cfg.evaluation.bijection_threshold,
        continuous_threshold=cfg.evaluation.continuous_correlation_threshold,
    )

    expected_gt = {gt.name for gt in cfg.evaluation.ground_truth_factors}
    matched_gt = {pair.ground_truth_name for pair in report.matched_pairs}

    assert report.recall == 1.0
    assert matched_gt == expected_gt
    for pair in report.matched_pairs:
        assert pair.agreement_rate == pytest.approx(1.0)
