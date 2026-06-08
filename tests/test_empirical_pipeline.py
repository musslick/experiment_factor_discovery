"""
Integration test: run the discovery pipeline on the real Stroop dataset and
verify that congruency is recovered.

Requires a valid ANTHROPIC_API_KEY environment variable and makes real LLM calls.
Guard with pytest.mark.integration so it is skipped in CI unless explicitly
requested:

    pytest tests/test_empirical_pipeline.py -m integration -v
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
EMPIRICAL_CFG = str(PROJECT_ROOT / "config" / "empirical_stroop_congruency.yaml")


@pytest.mark.integration
def test_empirical_benchmark_recovers_congruency(tmp_path):
    """
    Run 1 discovery round on the empirical Stroop dataset and assert that at
    least one discovered factor achieves bijection ≥ 0.90 with the ground-truth
    congruency column.
    """
    from src.analysis.evaluation import match_factors_bijection
    from src.data_generation import load_empirical_data
    from src.discovery.factor_registry import FactorRegistry
    from src.discovery.llm_client import LLMClient
    from src.discovery.pipeline import run_discovery_pipeline
    from src.utils.config import load_config
    from run_benchmark import _build_baseline_formula

    # Load config and override n_rounds to 1 for speed
    cfg = load_config(EMPIRICAL_CFG)
    cfg.discovery.n_rounds = 1

    full_df, input_df = load_empirical_data(cfg)

    llm      = LLMClient(model=cfg.llm.model)
    baseline = _build_baseline_formula(cfg)
    registry = FactorRegistry(baseline_formula=baseline)
    registry = run_discovery_pipeline(input_df, cfg, llm, registry, str(tmp_path))

    report = match_factors_bijection(
        ground_truth_factors=cfg.evaluation.ground_truth_factors,
        discovered_factors=registry.discovered,
        full_df=full_df,
        threshold=0.90,
        continuous_threshold=0.70,
    )

    assert report.recall >= 0.5, (
        f"Expected to recover at least one of the hidden factors "
        f"(recall={report.recall:.3f}). Discovered: "
        f"{[f.column_name for f in registry.discovered]}"
    )
