"""
Unit tests for the novelty score feature (novelty_epoch).

TestComputeNMI
    Validates compute_nmi across key cases: identical series, independent
    series, unequal cardinalities, all-NaN input.

TestComputeNoveltyScore
    Validates _compute_novelty_score: empty pool, identical reference,
    orthogonal reference, continuous path.

TestComputeAdjustedScoreNovelty
    Validates that novelty_weight=0.0 is a no-op (backward compat) and
    that a positive weight shifts the score by novelty_weight * novelty_score.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.evaluation import compute_nmi
from src.discovery.within_round_search import (
    _build_obs_desc_str,
    _compute_adjusted_score,
    _compute_novelty_score,
)
from src.discovery.factor_registry import CandidateFactor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(n_levels: int = 2, factor_class: str = "discrete") -> CandidateFactor:
    return CandidateFactor(
        name="test_factor",
        description="",
        factor_type="within_trial",
        factor_class=factor_class,
        levels=[str(i) for i in range(n_levels)],
        depends_on=[],
    )


# ---------------------------------------------------------------------------
# TestObservableDescription
# ---------------------------------------------------------------------------

class TestObservableDescription:
    def test_execution_metadata_is_not_prompted_as_observable_factor(self):
        desc = _build_obs_desc_str(
            ["participant_id", "block_index", "trial_index", "task"],
            {"task": "motion | color | orientation"},
        )

        assert "task: motion | color | orientation" in desc
        assert "participant_id" not in desc
        assert "block_index" not in desc
        assert "trial_index" not in desc


# ---------------------------------------------------------------------------
# TestComputeNMI
# ---------------------------------------------------------------------------

class TestComputeNMI:
    def test_identical_series_returns_one(self):
        s = pd.Series(["A", "B", "A", "B", "A"])
        assert compute_nmi(s, s) == pytest.approx(1.0)

    def test_independent_series_near_zero(self):
        rng = np.random.RandomState(0)
        a = pd.Series(rng.choice(["X", "Y"], size=200))
        b = pd.Series(rng.choice(["P", "Q"], size=200))
        assert compute_nmi(a, b) < 0.1

    def test_unequal_cardinality_returns_in_range(self):
        a = pd.Series(["A", "B", "A", "B", "A", "B"])
        b = pd.Series(["X", "Y", "Z", "X", "Y", "Z"])
        result = compute_nmi(a, b)
        assert 0.0 <= result <= 1.0

    def test_all_nan_returns_zero(self):
        a = pd.Series([np.nan, np.nan, np.nan])
        b = pd.Series([np.nan, np.nan, np.nan])
        assert compute_nmi(a, b) == 0.0

    def test_one_nan_series_returns_zero(self):
        a = pd.Series([np.nan, np.nan])
        b = pd.Series(["A", "B"])
        assert compute_nmi(a, b) == 0.0

    def test_deterministic_perfect_match_with_relabeling(self):
        a = pd.Series(["A", "B", "A", "B"])
        b = pd.Series(["X", "Y", "X", "Y"])
        assert compute_nmi(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestComputeNoveltyScore
# ---------------------------------------------------------------------------

class TestComputeNoveltyScore:
    def test_empty_pool_returns_one(self):
        s = pd.Series(["A", "B", "A"])
        assert _compute_novelty_score(s, "discrete", []) == pytest.approx(1.0)

    def test_identical_reference_returns_zero(self):
        s = pd.Series(["A", "B", "A", "B", "A"])
        assert _compute_novelty_score(s, "discrete", [s]) == pytest.approx(0.0)

    def test_orthogonal_reference_near_one(self):
        rng = np.random.RandomState(1)
        cand = pd.Series(rng.choice(["A", "B"], size=300))
        ref  = pd.Series(rng.choice(["X", "Y"], size=300))
        score = _compute_novelty_score(cand, "discrete", [ref])
        assert score > 0.9

    def test_continuous_path_uses_spearman(self):
        # Perfectly correlated → similarity 1.0 → novelty 0.0
        s = pd.Series(np.linspace(0, 1, 50))
        assert _compute_novelty_score(s, "continuous", [s]) == pytest.approx(0.0, abs=1e-6)

    def test_continuous_anticorrelated_still_zero_novelty(self):
        # |Spearman| = 1 for perfect anti-correlation → novelty = 0
        s = pd.Series(np.linspace(0, 1, 50))
        neg = pd.Series(np.linspace(1, 0, 50))
        assert _compute_novelty_score(s, "continuous", [neg]) == pytest.approx(0.0, abs=1e-6)

    def test_multiple_references_takes_max_similarity(self):
        # One independent ref, one identical ref → max similarity = 1 → novelty = 0
        s = pd.Series(["A", "B", "A", "B", "A"])
        rng = np.random.RandomState(2)
        noise = pd.Series(rng.choice(["X", "Y"], size=5))
        assert _compute_novelty_score(s, "discrete", [noise, s]) == pytest.approx(0.0)

    def test_novelty_clipped_to_unit_interval(self):
        s = pd.Series(["A", "B", "A"])
        ref = pd.Series(["X", "Y", "X"])
        score = _compute_novelty_score(s, "discrete", [ref])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# TestComputeAdjustedScoreNovelty
# ---------------------------------------------------------------------------

class TestComputeAdjustedScoreNovelty:
    def _base_score(self, candidate):
        return _compute_adjusted_score(
            mean_ll=0.05,
            se_ll=0.01,
            candidate=candidate,
            stability_weight=1.0,
            complexity_exponent=1.0,
            depends_on_exponent=0.5,
        )

    def test_zero_weight_is_noop(self):
        c = _make_candidate(n_levels=2)
        base = self._base_score(c)
        with_novelty = _compute_adjusted_score(
            mean_ll=0.05,
            se_ll=0.01,
            candidate=c,
            stability_weight=1.0,
            complexity_exponent=1.0,
            depends_on_exponent=0.5,
            novelty_score=1.0,
            novelty_weight=0.0,
        )
        assert with_novelty == pytest.approx(base)

    def test_positive_weight_increases_score(self):
        c = _make_candidate(n_levels=2)
        base = self._base_score(c)
        boosted = _compute_adjusted_score(
            mean_ll=0.05,
            se_ll=0.01,
            candidate=c,
            stability_weight=1.0,
            complexity_exponent=1.0,
            depends_on_exponent=0.5,
            novelty_score=1.0,
            novelty_weight=0.05,
        )
        assert boosted == pytest.approx(base + 0.05)

    def test_zero_novelty_score_adds_nothing(self):
        c = _make_candidate(n_levels=2)
        base = self._base_score(c)
        same = _compute_adjusted_score(
            mean_ll=0.05,
            se_ll=0.01,
            candidate=c,
            stability_weight=1.0,
            complexity_exponent=1.0,
            depends_on_exponent=0.5,
            novelty_score=0.0,
            novelty_weight=0.05,
        )
        assert same == pytest.approx(base)

    def test_bonus_scales_with_novelty_score(self):
        c = _make_candidate(n_levels=2)
        half = _compute_adjusted_score(
            mean_ll=0.05, se_ll=0.01, candidate=c,
            stability_weight=1.0, complexity_exponent=1.0, depends_on_exponent=0.5,
            novelty_score=0.5, novelty_weight=0.04,
        )
        full = _compute_adjusted_score(
            mean_ll=0.05, se_ll=0.01, candidate=c,
            stability_weight=1.0, complexity_exponent=1.0, depends_on_exponent=0.5,
            novelty_score=1.0, novelty_weight=0.04,
        )
        assert full - half == pytest.approx(0.02)
