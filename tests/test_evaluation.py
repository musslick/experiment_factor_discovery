"""
Functional unit tests for Phase 5: evaluation.py.

Test classes
------------
TestComputeAgreement
    Validates the bijection-based agreement metric for all important cases:
    perfect match, swapped labels, random noise, NaN exclusion, cardinality mismatch.

TestBijectionMatching
    Validates match_factors_bijection end-to-end: correct precision / recall / F1,
    label-invariant matching, false positives, threshold enforcement, edge cases.

TestOracleEvaluation
    Uses the generated ground-truth CSV to verify that feeding the true factor
    columns back as "discovered" factors yields P = R = F1 = 1.0, even when the
    level names are relabelled. Skipped automatically if the data file is absent.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.evaluation import compute_agreement, match_factors_bijection, EvaluationReport
from src.discovery.factor_registry import CandidateFactor, DiscoveredFactor
from src.utils.config import GroundTruthFactor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_discovered(
    name: str,
    factor_type: str,
    levels: list,
    values: pd.Series,
) -> DiscoveredFactor:
    candidate = CandidateFactor(
        name=name, description="test", factor_type=factor_type,
        levels=levels, depends_on=[],
    )
    return DiscoveredFactor(
        candidate=candidate, column_name=name, column_values=values,
        lrt_statistic=10.0, lrt_pvalue=1e-5, lrt_dof=1,
        formula_with=f"correct ~ C({name})",
    )


def _gt(name: str, factor_type: str, levels: list) -> GroundTruthFactor:
    return GroundTruthFactor(name=name, type=factor_type, levels=levels)


# ---------------------------------------------------------------------------
# Shared synthetic data (module-scoped for speed)
# ---------------------------------------------------------------------------

N = 200
SEED = 0

@pytest.fixture(scope="module")
def base_df() -> pd.DataFrame:
    """Small Stroop-like DataFrame with ground-truth factor columns."""
    rng = np.random.default_rng(SEED)
    color = rng.choice(["red", "blue", "green"], N)
    word  = rng.choice(["red", "blue", "green"], N)
    task  = rng.choice(["color_naming", "word_reading"], N)

    congruency = np.where(color == word, "congruent", "incongruent")

    task_transition = np.empty(N, dtype=object)
    for i in range(N):
        if i % 18 == 0:            # simulate block boundaries
            task_transition[i] = np.nan
        else:
            task_transition[i] = "repeat" if task[i] == task[i - 1] else "switch"

    return pd.DataFrame({
        "color":           color,
        "word":            word,
        "task":            task,
        "congruency":      congruency,
        "task_transition": task_transition,
    })


# ---------------------------------------------------------------------------
# TestComputeAgreement
# ---------------------------------------------------------------------------

class TestComputeAgreement:

    def test_identical_series_is_1(self, base_df):
        s = pd.Series(base_df["congruency"].values)
        assert compute_agreement(s, s) == pytest.approx(1.0)

    def test_perfect_match_different_labels(self, base_df):
        """
        Same partition, different label names:
        "congruent" → "A", "incongruent" → "B".
        The bijection should find the right mapping and return 1.0.
        """
        gt   = pd.Series(base_df["congruency"].values)
        disc = gt.map({"congruent": "A", "incongruent": "B"})
        assert compute_agreement(gt, disc) == pytest.approx(1.0)

    def test_inverted_binary_label_still_matches(self, base_df):
        """Swapping both labels: bijection {A→incongruent, B→congruent} gives 1.0."""
        gt   = pd.Series(base_df["congruency"].values)
        disc = gt.map({"congruent": "B", "incongruent": "A"})
        assert compute_agreement(gt, disc) == pytest.approx(1.0)

    def test_random_noise_is_near_chance(self):
        rng  = np.random.default_rng(99)
        gt   = pd.Series(rng.choice(["congruent", "incongruent"], N))
        disc = pd.Series(rng.choice(["X", "Y"], N))
        agr  = compute_agreement(gt, disc)
        # Random: expected ~0.5; allow generous margin (0.35–0.65)
        assert 0.35 <= agr <= 0.65

    def test_nan_rows_excluded(self, base_df):
        """NaN in the transition factor should be ignored, not treated as a level."""
        gt   = base_df["task_transition"]        # has NaN at block starts
        disc = gt.map({"repeat": "same", "switch": "different"})  # also has NaN
        agr  = compute_agreement(gt, disc)
        assert agr == pytest.approx(1.0)

    def test_different_cardinality_returns_zero(self, base_df):
        """GT binary, discovered 3-level — no bijection possible."""
        gt   = pd.Series(base_df["congruency"].values)
        disc = pd.Series(np.random.choice(["A", "B", "C"], N))
        assert compute_agreement(gt, disc) == pytest.approx(0.0)

    def test_all_nan_returns_zero(self):
        gt   = pd.Series([np.nan] * 10)
        disc = pd.Series([np.nan] * 10)
        assert compute_agreement(gt, disc) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestBijectionMatching
# ---------------------------------------------------------------------------

class TestBijectionMatching:

    # ---- single factor: perfect match ----

    def test_perfect_match_same_labels(self, base_df):
        gt_ser  = base_df["congruency"]
        disc    = _make_discovered("congruency", "within_trial",
                                   ["congruent", "incongruent"], gt_ser.copy())
        report  = match_factors_bijection(
            [_gt("congruency", "within_trial", ["congruent", "incongruent"])],
            [disc], base_df, threshold=0.95,
        )
        assert report.precision == pytest.approx(1.0)
        assert report.recall    == pytest.approx(1.0)
        assert report.f1        == pytest.approx(1.0)
        assert len(report.matched_pairs) == 1

    def test_perfect_match_different_labels(self, base_df):
        """
        LLM named the levels 'same' / 'different' instead of 'congruent' /
        'incongruent'.  The bijection should still find a perfect match.
        """
        gt_ser  = base_df["congruency"]
        disc_ser = gt_ser.map({"congruent": "same", "incongruent": "different"})
        disc    = _make_discovered("color_match", "within_trial",
                                   ["same", "different"], disc_ser)
        report  = match_factors_bijection(
            [_gt("congruency", "within_trial", ["congruent", "incongruent"])],
            [disc], base_df, threshold=0.95,
        )
        assert report.precision == pytest.approx(1.0)
        assert report.recall    == pytest.approx(1.0)
        assert len(report.matched_pairs) == 1
        assert report.matched_pairs[0].agreement_rate == pytest.approx(1.0)

    # ---- false negative: GT factor not discovered ----

    def test_false_negative(self, base_df):
        rng      = np.random.default_rng(42)
        noise    = pd.Series(rng.choice(["X", "Y"], N))
        disc     = _make_discovered("noise", "within_trial", ["X", "Y"], noise)
        report   = match_factors_bijection(
            [_gt("congruency", "within_trial", ["congruent", "incongruent"])],
            [disc], base_df, threshold=0.95,
        )
        assert report.recall    == pytest.approx(0.0)
        assert "congruency" in report.unmatched_ground_truth
        assert "noise"      in report.unmatched_discovered

    # ---- false positive: spurious discovered factor ----

    def test_false_positive_lowers_precision(self, base_df):
        """1 GT factor, 2 discovered (1 correct + 1 noise) → precision = 0.5."""
        gt_ser   = base_df["congruency"]
        correct  = _make_discovered("congruency", "within_trial",
                                    ["congruent", "incongruent"], gt_ser.copy())
        rng      = np.random.default_rng(7)
        noise_s  = pd.Series(rng.choice(["A", "B"], N))
        spurious = _make_discovered("noise", "within_trial", ["A", "B"], noise_s)

        report = match_factors_bijection(
            [_gt("congruency", "within_trial", ["congruent", "incongruent"])],
            [correct, spurious], base_df, threshold=0.95,
        )
        assert report.precision == pytest.approx(0.5)
        assert report.recall    == pytest.approx(1.0)
        assert "noise" in report.unmatched_discovered

    # ---- two GT factors: both discovered ----

    def test_two_factors_both_discovered(self, base_df):
        """Both congruency and task_transition discovered → P = R = F1 = 1."""
        con_disc  = _make_discovered(
            "congruency", "within_trial", ["congruent", "incongruent"],
            base_df["congruency"].copy(),
        )
        tt_disc   = _make_discovered(
            "task_transition", "transition", ["repeat", "switch"],
            base_df["task_transition"].copy(),
        )
        report = match_factors_bijection(
            [_gt("congruency", "within_trial", ["congruent", "incongruent"]),
             _gt("task_transition", "transition", ["repeat", "switch"])],
            [con_disc, tt_disc], base_df, threshold=0.95,
        )
        assert report.precision == pytest.approx(1.0)
        assert report.recall    == pytest.approx(1.0)
        assert report.f1        == pytest.approx(1.0)
        assert len(report.matched_pairs) == 2

    # ---- two GT factors: only one discovered ----

    def test_two_factors_one_discovered(self, base_df):
        """Only congruency found → recall = 0.5, precision = 1.0, F1 = 0.667."""
        con_disc = _make_discovered(
            "congruency", "within_trial", ["congruent", "incongruent"],
            base_df["congruency"].copy(),
        )
        report = match_factors_bijection(
            [_gt("congruency", "within_trial", ["congruent", "incongruent"]),
             _gt("task_transition", "transition", ["repeat", "switch"])],
            [con_disc], base_df, threshold=0.95,
        )
        assert report.precision == pytest.approx(1.0)
        assert report.recall    == pytest.approx(0.5)
        assert report.f1        == pytest.approx(2 * 1.0 * 0.5 / 1.5)
        assert "task_transition" in report.unmatched_ground_truth

    # ---- threshold enforcement ----

    def test_threshold_rejects_weak_match(self, base_df):
        """
        Introduce 10 % noise into the discovered column so agreement ≈ 0.90.
        At threshold = 0.95 this should NOT match; at threshold = 0.85 it should.
        """
        rng     = np.random.default_rng(3)
        gt_ser  = base_df["congruency"]
        noisy   = gt_ser.copy().map({"congruent": "same", "incongruent": "different"})
        flip    = rng.choice(noisy.index, size=int(0.10 * N), replace=False)
        noisy[flip] = noisy[flip].map({"same": "different", "different": "same"})

        disc = _make_discovered("approx_match", "within_trial", ["same", "different"], noisy)
        gt   = [_gt("congruency", "within_trial", ["congruent", "incongruent"])]

        report_strict = match_factors_bijection(gt, [disc], base_df, threshold=0.95)
        report_lenient = match_factors_bijection(gt, [disc], base_df, threshold=0.85)

        assert len(report_strict.matched_pairs)  == 0, "Should not match at 0.95"
        assert len(report_lenient.matched_pairs) == 1, "Should match at 0.85"

    # ---- edge cases ----

    def test_empty_discovered(self, base_df):
        report = match_factors_bijection(
            [_gt("congruency", "within_trial", ["congruent", "incongruent"])],
            [], base_df, threshold=0.95,
        )
        assert report.recall    == pytest.approx(0.0)
        assert report.precision == pytest.approx(1.0)  # no false positives
        assert report.f1        == pytest.approx(0.0)
        assert "congruency" in report.unmatched_ground_truth

    def test_empty_ground_truth(self, base_df):
        disc = _make_discovered("noise", "within_trial", ["A", "B"],
                                pd.Series(["A"] * N))
        report = match_factors_bijection([], [disc], base_df, threshold=0.95)
        assert report.precision == pytest.approx(0.0)
        assert report.f1        == pytest.approx(0.0)
        assert "noise" in report.unmatched_discovered

    def test_both_empty(self, base_df):
        report = match_factors_bijection([], [], base_df)
        assert report.precision == pytest.approx(1.0)
        assert report.recall    == pytest.approx(1.0)
        assert report.f1        == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestOracleEvaluation
# ---------------------------------------------------------------------------

FULL_CSV = "data/ground_truth/stroop_factor_discovery_full.csv"


@pytest.fixture(scope="module")
def full_df_gt():
    """Load the generated ground-truth CSV; skip the class if it doesn't exist."""
    try:
        return pd.read_csv(FULL_CSV)
    except FileNotFoundError:
        pytest.skip(f"{FULL_CSV} not found — run `python generate_data.py --config config/synthetic_stroop_benchmark.yaml` first")


class TestOracleEvaluation:
    """
    Oracle tests: feed the true GT factor columns back as 'discovered' factors
    and verify that P = R = F1 = 1.0 — with and without level relabelling.
    These tests validate the full pipeline path without any LLM calls.
    """

    def _make_oracle_discovered(self, full_df, col_name, factor_type, levels):
        """Wrap a GT column as a DiscoveredFactor."""
        return _make_discovered(col_name, factor_type, levels,
                                full_df[col_name].copy())

    def test_perfect_oracle_congruency(self, full_df_gt):
        disc   = self._make_oracle_discovered(
            full_df_gt, "congruency", "within_trial", ["congruent", "incongruent"]
        )
        report = match_factors_bijection(
            [_gt("congruency", "within_trial", ["congruent", "incongruent"])],
            [disc], full_df_gt, threshold=0.95,
        )
        assert report.f1 == pytest.approx(1.0)
        assert report.matched_pairs[0].agreement_rate == pytest.approx(1.0)

    def test_perfect_oracle_task_transition(self, full_df_gt):
        disc   = self._make_oracle_discovered(
            full_df_gt, "task_transition", "transition", ["repeat", "switch"]
        )
        report = match_factors_bijection(
            [_gt("task_transition", "transition", ["repeat", "switch"])],
            [disc], full_df_gt, threshold=0.95,
        )
        assert report.f1 == pytest.approx(1.0)

    def test_oracle_both_factors_at_once(self, full_df_gt):
        discs = [
            self._make_oracle_discovered(
                full_df_gt, "congruency", "within_trial", ["congruent", "incongruent"]
            ),
            self._make_oracle_discovered(
                full_df_gt, "task_transition", "transition", ["repeat", "switch"]
            ),
        ]
        report = match_factors_bijection(
            [_gt("congruency",      "within_trial", ["congruent", "incongruent"]),
             _gt("task_transition", "transition",   ["repeat", "switch"])],
            discs, full_df_gt, threshold=0.95,
        )
        assert report.precision == pytest.approx(1.0)
        assert report.recall    == pytest.approx(1.0)
        assert report.f1        == pytest.approx(1.0)
        assert len(report.matched_pairs) == 2

    def test_oracle_relabelled_levels_still_match(self, full_df_gt):
        """
        Relabel the discovered congruency column (congruent → 'C', incongruent → 'I').
        The bijection must still find the correct mapping and return agreement = 1.0.
        """
        relabelled = (full_df_gt["congruency"]
                      .map({"congruent": "C", "incongruent": "I"}))
        disc   = _make_discovered(
            "color_match", "within_trial", ["C", "I"], relabelled
        )
        report = match_factors_bijection(
            [_gt("congruency", "within_trial", ["congruent", "incongruent"])],
            [disc], full_df_gt, threshold=0.95,
        )
        assert report.f1 == pytest.approx(1.0)
        assert report.matched_pairs[0].agreement_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestComputeCorrelation
# ---------------------------------------------------------------------------

class TestComputeCorrelation:
    """Tests for the new compute_correlation function."""

    def test_import(self):
        from src.analysis.evaluation import compute_correlation
        assert callable(compute_correlation)

    def test_perfect_positive_correlation(self):
        from src.analysis.evaluation import compute_correlation
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert compute_correlation(s, s) == pytest.approx(1.0)

    def test_perfect_negative_correlation_returns_abs(self):
        from src.analysis.evaluation import compute_correlation
        s  = pd.Series([float(i) for i in range(10)])
        s2 = pd.Series([float(9 - i) for i in range(10)])
        # Spearman ρ = -1.0; abs should give 1.0
        assert compute_correlation(s, s2) == pytest.approx(1.0)

    def test_independent_series_near_zero(self):
        from src.analysis.evaluation import compute_correlation
        rng = np.random.default_rng(77)
        s1 = pd.Series(rng.normal(0, 1, 200))
        s2 = pd.Series(rng.normal(0, 1, 200))
        rho = compute_correlation(s1, s2)
        assert rho < 0.3  # random noise should be near zero

    def test_nan_rows_excluded(self):
        from src.analysis.evaluation import compute_correlation
        s1 = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        s2 = pd.Series([1.0, 2.0, 3.0, 4.0,    5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        rho = compute_correlation(s1, s2)
        assert rho > 0.9  # should still be high after dropping the NaN row

    def test_fewer_than_3_valid_rows_returns_zero(self):
        from src.analysis.evaluation import compute_correlation
        s1 = pd.Series([1.0, 2.0])
        s2 = pd.Series([4.0, 5.0])
        assert compute_correlation(s1, s2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestContinuousFactorMatching
# ---------------------------------------------------------------------------

class TestContinuousFactorMatching:
    """Tests for match_factors_bijection with continuous factors."""

    def _make_continuous_discovered(self, name, values):
        from src.discovery.factor_registry import CandidateFactor, DiscoveredFactor
        candidate = CandidateFactor(
            name=name, description="test", factor_type="within_trial",
            factor_class="continuous", levels=[], depends_on=[],
        )
        return DiscoveredFactor(
            candidate=candidate, column_name=name, column_values=values,
            lrt_statistic=0.0, lrt_pvalue=1.0, lrt_dof=0,
            formula_with=f"correct ~ {name}",
        )

    def _gt_continuous(self, name):
        return GroundTruthFactor(name=name, type="within_trial",
                                 levels=[], factor_class="continuous")

    def test_continuous_perfect_match(self):
        n = 100
        rng = np.random.default_rng(1)
        vals = pd.Series(rng.normal(0, 1, n))
        df = pd.DataFrame({"gt_cont": vals})

        disc = self._make_continuous_discovered("found_cont", vals.copy())
        gt   = self._gt_continuous("gt_cont")

        report = match_factors_bijection([gt], [disc], df,
                                          continuous_threshold=0.7)
        assert report.precision == pytest.approx(1.0)
        assert report.recall    == pytest.approx(1.0)
        assert report.matched_pairs[0].correlation == pytest.approx(1.0, abs=0.01)

    def test_cross_class_never_matches(self, base_df):
        """A discrete GT factor should never match a continuous discovered factor."""
        rng  = np.random.default_rng(5)
        vals = pd.Series(rng.normal(0, 1, N))
        disc_cont = self._make_continuous_discovered("cont_noise", vals)
        gt_disc   = _gt("congruency", "within_trial", ["congruent", "incongruent"])

        report = match_factors_bijection([gt_disc], [disc_cont], base_df,
                                          threshold=0.95, continuous_threshold=0.7)
        assert report.precision == pytest.approx(0.0)
        assert report.recall    == pytest.approx(0.0)
        assert "congruency"  in report.unmatched_ground_truth
        assert "cont_noise"  in report.unmatched_discovered

    def test_weak_correlation_below_threshold_not_matched(self):
        n = 200
        rng = np.random.default_rng(42)
        gt_vals   = pd.Series(rng.normal(0, 1, n))
        disc_vals = pd.Series(rng.normal(0, 1, n))  # independent
        df = pd.DataFrame({"gt_cont": gt_vals})

        disc = self._make_continuous_discovered("weak_cont", disc_vals)
        gt   = self._gt_continuous("gt_cont")

        report = match_factors_bijection([gt], [disc], df, continuous_threshold=0.7)
        assert len(report.matched_pairs) == 0
