"""
Unit tests for empirical config parsing and data loading.
No LLM calls are made in this file.
"""

import copy
from pathlib import Path

import pytest

from run_benchmark import _build_baseline_formula
from src.data_generation.empirical_loader import load_empirical_data
from src.utils.config import BenchmarkConfig, DatasetConfig, HiddenFactor, load_config

# Resolve paths relative to the project root
PROJECT_ROOT  = Path(__file__).parent.parent
EMPIRICAL_CFG = str(PROJECT_ROOT / "config" / "empirical_stroop_congruency.yaml")
SYNTHETIC_CFG = str(PROJECT_ROOT / "config" / "synthetic_stroop_benchmark.yaml")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestParseEmpiricalConfig:
    def test_mode(self):
        cfg = load_config(EMPIRICAL_CFG)
        assert cfg.mode == "empirical_benchmark"

    def test_dataset_path(self):
        cfg = load_config(EMPIRICAL_CFG)
        assert cfg.dataset.path == "data/empirical/stroop_data_with_congruency.csv"

    def test_base_factors(self):
        cfg = load_config(EMPIRICAL_CFG)
        assert len(cfg.base_factors) == 2
        names = [bf.name for bf in cfg.base_factors]
        assert "color" in names and "word" in names

    def test_hidden_factors(self):
        cfg = load_config(EMPIRICAL_CFG)
        assert len(cfg.dataset.hidden_factors) == 1
        hf = cfg.dataset.hidden_factors[0]
        assert hf.name == "congruency"
        assert hf.column == "congruency"
        assert hf.levels == ["congruent", "incongruent"]

    def test_ground_truth_auto_built(self):
        """EvaluationConfig.ground_truth_factors is auto-built from hidden_factors."""
        cfg = load_config(EMPIRICAL_CFG)
        assert len(cfg.evaluation.ground_truth_factors) == 1
        gtf = cfg.evaluation.ground_truth_factors[0]
        assert gtf.name == "congruency"
        assert gtf.levels == ["congruent", "incongruent"]

    def test_extra_columns(self):
        cfg = load_config(EMPIRICAL_CFG)
        assert "trialnum" in cfg.dataset.extra_columns
        assert "blocknum" in cfg.dataset.extra_columns

    def test_data_generation_is_none(self):
        cfg = load_config(EMPIRICAL_CFG)
        assert cfg.data_generation is None


class TestEmpiricalConfigDualUse:
    """
    The empirical dataset config is used by both run_benchmark.py (benchmarking)
    and run_discovery.py (novel discovery).  Verify the config and the in-memory
    augmentation that run_discovery.py applies.
    """

    def test_hidden_factors_promoted_for_discovery(self):
        """run_discovery.py promotes hidden_factors into base_factors in memory."""
        import copy
        from src.utils.config import BaseFactor

        cfg = load_config(EMPIRICAL_CFG)
        aug = copy.deepcopy(cfg)
        for hf in aug.dataset.hidden_factors:
            dtype = "categorical" if hf.factor_class == "discrete" else "continuous"
            aug.dataset.base_factors.append(BaseFactor(name=hf.name, dtype=dtype, levels=list(hf.levels)))
        aug.dataset.hidden_factors = []

        names = [bf.name for bf in aug.base_factors]
        assert "congruency" in names   # promoted
        assert aug.dataset.hidden_factors == []

    def test_augmented_input_df_has_all_columns(self):
        """After promotion, load_empirical_data returns input_df with hidden columns visible."""
        import copy
        from src.utils.config import BaseFactor

        cfg = load_config(EMPIRICAL_CFG)
        aug = copy.deepcopy(cfg)
        for hf in aug.dataset.hidden_factors:
            dtype = "categorical" if hf.factor_class == "discrete" else "continuous"
            aug.dataset.base_factors.append(BaseFactor(name=hf.name, dtype=dtype, levels=list(hf.levels)))
        aug.dataset.hidden_factors = []

        _, input_df = load_empirical_data(aug)
        assert "congruency" in input_df.columns

    def test_discovery_baseline_formula_includes_hidden(self):
        """The discovery baseline auto-builds from all base_factors (including promoted ones)."""
        import copy
        from src.utils.config import BaseFactor
        from run_discovery import _build_discovery_baseline_formula

        cfg = load_config(EMPIRICAL_CFG)
        aug = copy.deepcopy(cfg)
        for hf in aug.dataset.hidden_factors:
            dtype = "categorical" if hf.factor_class == "discrete" else "continuous"
            aug.dataset.base_factors.append(BaseFactor(name=hf.name, dtype=dtype, levels=list(hf.levels)))
        aug.dataset.hidden_factors = []

        formula = _build_discovery_baseline_formula(aug)
        assert "C(congruency)" in formula


class TestSyntheticBackwardsCompat:
    def test_mode_default(self):
        cfg = load_config(SYNTHETIC_CFG)
        assert cfg.mode == "synthetic_benchmark"

    def test_data_generation_populated(self):
        cfg = load_config(SYNTHETIC_CFG)
        assert cfg.data_generation is not None
        assert cfg.data_generation.n_participants > 0

    def test_dataset_is_none(self):
        cfg = load_config(SYNTHETIC_CFG)
        assert cfg.dataset is None

    def test_base_factors_property(self):
        """cfg.base_factors resolves through data_generation."""
        cfg = load_config(SYNTHETIC_CFG)
        assert len(cfg.base_factors) > 0


# ---------------------------------------------------------------------------
# Baseline formula
# ---------------------------------------------------------------------------

class TestBaselineFormula:
    def test_auto_built_from_base_factors(self):
        cfg = load_config(EMPIRICAL_CFG)
        formula = _build_baseline_formula(cfg)
        assert formula == "accuracy ~ C(color) + C(word)"

    def test_null_formula_override(self):
        cfg = load_config(EMPIRICAL_CFG)
        cfg.dataset.null_formula = "accuracy ~ 1"
        formula = _build_baseline_formula(cfg)
        assert formula == "accuracy ~ 1"

    def test_synthetic_formula(self):
        cfg = load_config(SYNTHETIC_CFG)
        formula = _build_baseline_formula(cfg)
        assert formula.startswith("correct ~")


# ---------------------------------------------------------------------------
# Empirical data loading
# ---------------------------------------------------------------------------

class TestEmpiricalDataLoading:
    @pytest.fixture(scope="class")
    def loaded(self):
        cfg = load_config(EMPIRICAL_CFG)
        return load_empirical_data(cfg)

    def test_full_df_has_hidden_column(self, loaded):
        full_df, _ = loaded
        assert "congruency" in full_df.columns

    def test_input_df_lacks_hidden_column(self, loaded):
        _, input_df = loaded
        assert "congruency" not in input_df.columns

    def test_participant_count(self, loaded):
        full_df, _ = loaded
        assert full_df["participant_id"].nunique() == 466

    def test_base_factor_columns_present(self, loaded):
        full_df, input_df = loaded
        for name in ("color", "word"):
            assert name in full_df.columns
            assert name in input_df.columns

    def test_outcome_column_present(self, loaded):
        full_df, input_df = loaded
        assert "accuracy" in full_df.columns
        assert "accuracy" in input_df.columns

    def test_extra_columns_present(self, loaded):
        full_df, input_df = loaded
        for col in ("trialnum", "blocknum", "blockcode"):
            assert col in full_df.columns
            assert col in input_df.columns

    def test_participant_id_renamed(self, loaded):
        full_df, input_df = loaded
        assert "participant_id" in full_df.columns
        assert "subjectid" not in full_df.columns


class TestHiddenFactorColumnRename:
    def test_column_renamed_to_name(self, tmp_path):
        """When hf.column != hf.name, load_empirical_data renames the column."""
        import pandas as pd
        from src.utils.config import BaseFactor, HiddenFactor, DatasetConfig, BenchmarkConfig, DiscoveryConfig, LLMConfig, StatisticalConfig, EvaluationConfig

        # Build a tiny CSV where the hidden column has a different CSV name
        df = pd.DataFrame({
            "pid":         [1, 1, 2, 2],
            "color":       ["red", "blue", "red", "blue"],
            "word":        ["red", "blue", "blue", "red"],
            "values.cong": ["congruent", "congruent", "incongruent", "incongruent"],
            "accuracy":    [1, 1, 0, 0],
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        dataset = DatasetConfig(
            path=str(csv_path),
            participant_id_column="pid",
            outcome_variable="accuracy",
            task_context="",
            base_factors=[
                BaseFactor(name="color", dtype="categorical", levels=["red", "blue"]),
                BaseFactor(name="word",  dtype="categorical", levels=["red", "blue"]),
            ],
            hidden_factors=[
                HiddenFactor(name="congruency", column="values.cong",
                             type="within_trial", levels=["congruent", "incongruent"]),
            ],
            extra_columns=[],
        )
        # Minimal BenchmarkConfig
        cfg = BenchmarkConfig(
            name="test", benchmark_type="empirical", seed=0, output_dir=".",
            mode="empirical_benchmark",
            discovery=DiscoveryConfig(n_rounds=1, max_synthesis_retries=1,
                                       sandbox_timeout_seconds=5, sandbox_backend="subprocess"),
            llm=LLMConfig(model="x", max_tokens_candidate=10, max_tokens_predicate=10,
                          candidate_temperature=0.5, predicate_temperature=0.2),
            statistical=StatisticalConfig(min_level_count=1),
            evaluation=EvaluationConfig(ground_truth_factors=[], bijection_threshold=0.95),
            dataset=dataset,
        )

        full_df, input_df = load_empirical_data(cfg)
        assert "congruency" in full_df.columns      # renamed to model name
        assert "values.cong" not in full_df.columns
        assert "congruency" not in input_df.columns
