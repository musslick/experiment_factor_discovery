"""
Functional unit tests for Phase 2: factor_encoder and model_comparison.

Synthetic data is generated with numpy directly (no SweetPea) for speed.
The data-generating process mirrors the ground-truth logistic model so that
the LRT tests reflect realistic discovery scenarios.

Test classes
------------
TestEncodeFactory      – encode_factor: correct encoding, NaN handling,
                          and all validation failure modes.
TestBuildFormula       – build_extended_formula: formula string construction.
TestLRT                – compare_models_lrt:
                          (a) a true within-trial factor is detected,
                          (b) a true transition factor is detected after
                              conditioning on an already-known factor,
                          (c) a pure-noise factor is correctly NOT detected,
                          (d) perfect separation is flagged.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.factor_encoder import encode_factor
from src.analysis.model_comparison import compare_models_lrt, build_extended_formula, LRTResult


# ---------------------------------------------------------------------------
# Shared synthetic dataset (module-scoped for speed)
# ---------------------------------------------------------------------------

N_TRIALS      = 3000   # enough for reliable power on both effects
SEED          = 7
INTERCEPT     = 0.5
BETA_CON      = 0.8    # congruency effect
BETA_TASK_REP = 0.4    # task-transition effect


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    """
    Generates a synthetic Stroop-like dataset directly from numpy without
    invoking SweetPea, keeping the fixture fast (< 1 s).

    Includes:
        task, color, word   – observable factors
        congruency          – within-trial hidden factor  (color == word)
        task_transition     – transition hidden factor    (NaN for first trial
                              of each 18-trial 'block')
        correct             – binary outcome from the logistic model
    """
    rng = np.random.default_rng(SEED)
    tasks  = ["color_naming", "word_reading"]
    colors = ["red", "blue", "green"]
    words  = ["red", "blue", "green"]

    task  = rng.choice(tasks,  N_TRIALS)
    color = rng.choice(colors, N_TRIALS)
    word  = rng.choice(words,  N_TRIALS)

    # Within-trial factor
    congruency = np.where(color == word, "congruent", "incongruent")

    # Transition factor — simulate 18-trial blocks; first trial of each block = NaN
    block_size = 18
    task_transition = np.empty(N_TRIALS, dtype=object)
    for i in range(N_TRIALS):
        if i % block_size == 0:
            task_transition[i] = np.nan
        else:
            task_transition[i] = "repeat" if task[i] == task[i - 1] else "switch"

    # Logistic model: NaN transition → 0 contribution
    logit = (
        INTERCEPT
        + BETA_CON      * (congruency == "congruent").astype(float)
        + BETA_TASK_REP * (task_transition == "repeat").astype(float)  # NaN → 0
    )
    correct = rng.binomial(1, _sigmoid(logit)).astype(int)

    return pd.DataFrame({
        "task":            task,
        "color":           color,
        "word":            word,
        "congruency":      congruency,
        "task_transition": task_transition,
        "correct":         correct,
    })


# ---------------------------------------------------------------------------
# TestEncodeFactory
# ---------------------------------------------------------------------------

class TestEncodeFactory:
    """Tests for encode_factor."""

    def _make_index(self, n=20):
        return pd.RangeIndex(n)

    def test_valid_within_trial_factor(self):
        """All values match declared levels → is_valid=True, no NaN."""
        values = ["congruent", "incongruent"] * 10
        s, ok, reason = encode_factor(values, self._make_index(20),
                                      ["congruent", "incongruent"], min_level_count=5)
        assert ok, reason
        assert s.isna().sum() == 0
        assert set(s.unique()) == {"congruent", "incongruent"}

    def test_none_and_empty_string_become_nan(self):
        """None and '' (SweetPea's placeholder) are converted to NaN."""
        values = [None, "", "repeat", "switch"] * 5
        s, ok, reason = encode_factor(values, self._make_index(20),
                                      ["repeat", "switch"], min_level_count=3)
        assert ok, reason
        assert s.isna().sum() == 10        # 5 None + 5 '' → NaN
        assert set(s.dropna().unique()) == {"repeat", "switch"}

    def test_length_mismatch_is_invalid(self):
        """Wrong-length list → is_valid=False."""
        values = ["congruent"] * 5
        _, ok, reason = encode_factor(values, self._make_index(20),
                                      ["congruent", "incongruent"])
        assert not ok
        assert "length mismatch" in reason.lower()

    def test_unexpected_level_is_invalid(self):
        """Values not in declared_levels → is_valid=False."""
        values = ["congruent", "typo"] * 10
        _, ok, reason = encode_factor(values, self._make_index(20),
                                      ["congruent", "incongruent"])
        assert not ok
        assert "unexpected" in reason.lower()

    def test_insufficient_level_count_is_invalid(self):
        """A level appearing fewer than min_level_count times → is_valid=False."""
        # Only 1 occurrence of "incongruent", min is 5
        values = ["congruent"] * 19 + ["incongruent"]
        _, ok, reason = encode_factor(values, self._make_index(20),
                                      ["congruent", "incongruent"], min_level_count=5)
        assert not ok
        assert "incongruent" in reason

    def test_exactly_at_min_count_is_valid(self):
        """Exactly min_level_count occurrences of each level → is_valid=True."""
        values = ["repeat"] * 5 + ["switch"] * 5
        _, ok, reason = encode_factor(values, self._make_index(10),
                                      ["repeat", "switch"], min_level_count=5)
        assert ok, reason

    def test_index_alignment(self):
        """Series index matches the provided df_index."""
        idx = pd.Index([10, 20, 30, 40, 50, 60])
        values = ["repeat", "switch"] * 3
        s, _, _ = encode_factor(values, idx, ["repeat", "switch"], min_level_count=1)
        assert list(s.index) == list(idx)


# ---------------------------------------------------------------------------
# TestBuildFormula
# ---------------------------------------------------------------------------

class TestBuildFormula:

    def test_from_intercept_only(self):
        f = build_extended_formula("correct ~ 1", "congruency")
        assert f == "correct ~ C(congruency)"

    def test_from_existing_term(self):
        f = build_extended_formula("correct ~ C(congruency)", "task_transition")
        assert f == "correct ~ C(congruency) + C(task_transition)"

    def test_lhs_preserved(self):
        f = build_extended_formula("accuracy ~ 1", "noise")
        assert f.startswith("accuracy ~")

    def test_new_factor_wrapped_in_C(self):
        f = build_extended_formula("correct ~ 1", "my_factor")
        assert "C(my_factor)" in f


# ---------------------------------------------------------------------------
# TestLRT
# ---------------------------------------------------------------------------

class TestLRT:
    """
    Tests for compare_models_lrt.  Uses the module-scoped synthetic_df fixture.
    All four scenarios (true within-trial factor, true transition factor,
    noise factor, and separation) are covered.
    """

    ALPHA = 0.05

    # --- (a) True within-trial factor: congruency ---

    def test_true_within_trial_factor_is_significant(self, synthetic_df):
        """
        LRT of 'correct ~ 1' vs 'correct ~ C(congruency)' should be
        highly significant (β_con = 0.8).
        """
        result = compare_models_lrt(
            synthetic_df,
            formula_null="correct ~ 1",
            formula_alt="correct ~ C(congruency)",
        )
        assert isinstance(result, LRTResult)
        assert result.pvalue < 1e-6, (
            f"Congruency effect not detected: p = {result.pvalue:.4e}"
        )
        assert result.llf_alt > result.llf_null, "Alt model must fit better"
        assert result.dof == 1
        assert result.n_obs == N_TRIALS   # no NaN in congruency

    # --- (b) True transition factor: task_transition given congruency ---

    def test_true_transition_factor_is_significant_given_known(self, synthetic_df):
        """
        LRT of 'correct ~ C(congruency)' vs
               'correct ~ C(congruency) + C(task_transition)'
        should be significant (β_task_rep = 0.4).

        Only non-NaN task_transition rows are used (shared mask).
        """
        result = compare_models_lrt(
            synthetic_df,
            formula_null="correct ~ C(congruency)",
            formula_alt="correct ~ C(congruency) + C(task_transition)",
        )
        assert result.pvalue < self.ALPHA, (
            f"Task-transition effect not detected: p = {result.pvalue:.4f}"
        )
        assert result.dof == 1
        # Shared rows exclude block-start NaN trials
        n_valid = synthetic_df["task_transition"].notna().sum()
        assert result.n_obs == n_valid

    # --- (c) Pure noise factor: must NOT be significant ---

    def test_noise_factor_is_not_significant(self, synthetic_df):
        """
        A randomly generated binary column uncorrelated with correct should
        not pass the LRT at α = 0.05.  Tested with a fixed seed.
        """
        rng = np.random.default_rng(99)
        df = synthetic_df.copy()
        df["noise"] = rng.choice(["A", "B"], len(df))

        result = compare_models_lrt(
            df,
            formula_null="correct ~ 1",
            formula_alt="correct ~ C(noise)",
        )
        assert result.pvalue > self.ALPHA, (
            f"Noise factor incorrectly flagged as significant: p = {result.pvalue:.4f}"
        )

    # --- (d) Perfect separation is flagged ---

    def test_separation_is_detected(self, synthetic_df):
        """
        A column that perfectly predicts the outcome should trigger the
        separation flag (converged=False, separation_detected=True).
        The p-value in this case is not reliable and callers should reject
        the candidate.
        """
        df = synthetic_df.copy()
        # Perfect predictor: same label as the outcome
        df["perfect"] = np.where(df["correct"] == 1, "yes", "no")

        result = compare_models_lrt(
            df,
            formula_null="correct ~ 1",
            formula_alt="correct ~ C(perfect)",
        )
        assert result.separation_detected, (
            "Expected separation_detected=True for a perfect predictor"
        )

    # --- Additional invariants ---

    def test_result_fields_are_finite(self, synthetic_df):
        """All numeric result fields should be finite for a well-behaved model."""
        result = compare_models_lrt(
            synthetic_df,
            formula_null="correct ~ 1",
            formula_alt="correct ~ C(congruency)",
        )
        assert np.isfinite(result.statistic)
        assert np.isfinite(result.pvalue)
        assert np.isfinite(result.llf_null)
        assert np.isfinite(result.llf_alt)

    def test_shared_nan_mask_excludes_transition_nans(self, synthetic_df):
        """
        When the alt formula includes task_transition, the shared mask
        drops block-start NaN rows for BOTH models, so n_obs is the same
        as the number of non-NaN task_transition rows.
        """
        result_with_trans = compare_models_lrt(
            synthetic_df,
            formula_null="correct ~ C(congruency)",
            formula_alt="correct ~ C(congruency) + C(task_transition)",
        )
        result_without_trans = compare_models_lrt(
            synthetic_df,
            formula_null="correct ~ 1",
            formula_alt="correct ~ C(congruency)",
        )
        n_valid_trans = int(synthetic_df["task_transition"].notna().sum())
        assert result_with_trans.n_obs == n_valid_trans
        assert result_without_trans.n_obs == N_TRIALS  # congruency has no NaN
