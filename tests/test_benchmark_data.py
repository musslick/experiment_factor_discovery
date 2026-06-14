"""
Functionality tests for the three new benchmark data generators:
  Stroop–Simon, RDK task-switching, Prospect theory.

Each benchmark is tested for:
  1. Structural integrity — correct shape, columns, dtypes, NaN placement.
  2. Factor validity     — levels are within declared sets, derived factors
                           are consistent with the base factor values.
  3. Statistical validity — logistic model produces correct directional effects.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.utils.config import load_config, LogisticModelConfig, ModelTerm
from src.data_generation import get_data_generator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

N_PARTICIPANTS = 5
N_BLOCKS       = 4    # small for speed
SEED           = 99


def _z_test_one_sided(n1, k1, n2, k2):
    """H0: p1 <= p2.  Returns one-sided p-value."""
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    return float(stats.norm.sf((p1 - p2) / se)) if se > 0 else 1.0


def _load_small(bm_type, config_path, n_participants=N_PARTICIPANTS, n_blocks=N_BLOCKS):
    cfg = load_config(config_path)
    cfg.data_generation.n_participants = n_participants
    cfg.data_generation.n_blocks_per_participant = n_blocks
    cfg.seed = SEED
    gen = get_data_generator(bm_type)
    return gen.generate(cfg)


# ===========================================================================
# Stroop–Simon
# ===========================================================================

@pytest.fixture(scope="module")
def stroop_simon_dfs():
    return _load_small("stroop_simon", "config/synthetic_stroop_simon_benchmark.yaml")


class TestStroopSimonStructure:
    TRIALS_PER_BLOCK = 27

    def test_shape(self, stroop_simon_dfs):
        full_df, input_df = stroop_simon_dfs
        expected_rows = N_PARTICIPANTS * N_BLOCKS * self.TRIALS_PER_BLOCK
        assert len(full_df) == expected_rows
        assert len(input_df) == expected_rows

    def test_observable_columns(self, stroop_simon_dfs):
        _, input_df = stroop_simon_dfs
        expected = {"participant_id", "block_index", "trial_index", "word", "color",
                    "stimulus_location", "correct_response", "correct"}
        assert expected.issubset(set(input_df.columns))

    def test_hidden_columns_not_in_input(self, stroop_simon_dfs):
        full_df, input_df = stroop_simon_dfs
        hidden = {"word_color_congruency", "location_response_congruency",
                  "congruency_previous_trial", "response_transition"}
        assert hidden.issubset(set(full_df.columns))
        assert not hidden.intersection(set(input_df.columns))

    def test_correct_response_derived_from_color(self, stroop_simon_dfs):
        full_df, _ = stroop_simon_dfs
        mapping = {"red": "left", "blue": "middle", "green": "right"}
        assert (full_df["correct_response"] == full_df["color"].map(mapping)).all()

    def test_word_color_congruency(self, stroop_simon_dfs):
        full_df, _ = stroop_simon_dfs
        con = full_df[full_df["word_color_congruency"] == "congruent"]
        assert (con["word"] == con["color"]).all()
        inc = full_df[full_df["word_color_congruency"] == "incongruent"]
        assert (inc["word"] != inc["color"]).all()

    def test_location_response_congruency(self, stroop_simon_dfs):
        full_df, _ = stroop_simon_dfs
        con = full_df[full_df["location_response_congruency"] == "congruent"]
        assert (con["stimulus_location"] == con["correct_response"]).all()
        inc = full_df[full_df["location_response_congruency"] == "incongruent"]
        assert (inc["stimulus_location"] != inc["correct_response"]).all()

    def test_transition_nan_at_block_starts(self, stroop_simon_dfs):
        full_df, _ = stroop_simon_dfs
        expected_nan = N_PARTICIPANTS * N_BLOCKS
        for col in ["congruency_previous_trial", "response_transition"]:
            assert full_df[col].isna().sum() == expected_nan, \
                f"{col}: expected {expected_nan} NaN, got {full_df[col].isna().sum()}"

    def test_outcome_binary(self, stroop_simon_dfs):
        full_df, _ = stroop_simon_dfs
        assert full_df["correct"].isin([0, 1]).all()

    def test_factor_levels(self, stroop_simon_dfs):
        full_df, _ = stroop_simon_dfs
        assert set(full_df["word"].unique())                     <= {"red", "blue", "green"}
        assert set(full_df["color"].unique())                    <= {"red", "blue", "green"}
        assert set(full_df["stimulus_location"].unique())        <= {"left", "middle", "right"}
        assert set(full_df["correct_response"].unique())         <= {"left", "middle", "right"}
        assert set(full_df["word_color_congruency"].unique())    <= {"congruent", "incongruent"}
        assert set(full_df["location_response_congruency"].unique()) <= {"congruent", "incongruent"}


class TestStroopSimonStatistics:
    def test_word_color_congruency_effect(self, stroop_simon_dfs):
        full_df, _ = stroop_simon_dfs
        con = full_df[full_df["word_color_congruency"] == "congruent"]["correct"]
        inc = full_df[full_df["word_color_congruency"] == "incongruent"]["correct"]
        assert con.mean() > inc.mean(), \
            f"congruent ({con.mean():.3f}) not > incongruent ({inc.mean():.3f})"

    def test_location_response_congruency_effect(self, stroop_simon_dfs):
        full_df, _ = stroop_simon_dfs
        con = full_df[full_df["location_response_congruency"] == "congruent"]["correct"]
        inc = full_df[full_df["location_response_congruency"] == "incongruent"]["correct"]
        assert con.mean() > inc.mean(), \
            f"lrc congruent ({con.mean():.3f}) not > incongruent ({inc.mean():.3f})"


# ===========================================================================
# RDK task-switching
# ===========================================================================

@pytest.fixture(scope="module")
def rdk_dfs():
    return _load_small("rdk_task_switching", "config/synthetic_rdk_task_switching_benchmark.yaml")


class TestRDKStructure:
    TRIALS_PER_BLOCK = 24

    def test_shape(self, rdk_dfs):
        full_df, input_df = rdk_dfs
        expected = N_PARTICIPANTS * N_BLOCKS * self.TRIALS_PER_BLOCK
        assert len(full_df) == expected
        assert len(input_df) == expected

    def test_observable_columns(self, rdk_dfs):
        _, input_df = rdk_dfs
        expected = {"participant_id", "block_index", "trial_index", "task", "motion",
                    "color", "orientation", "motion_coherence", "color_coherence",
                    "orientation_coherence", "correct_response", "correct"}
        assert expected.issubset(set(input_df.columns))

    def test_hidden_columns_not_in_input(self, rdk_dfs):
        full_df, input_df = rdk_dfs
        hidden = {"task_transition", "n2_task_inhibition",
                  "current_stimulus_difficulty", "past_stimulus_difficulty"}
        assert hidden.issubset(set(full_df.columns))
        assert not hidden.intersection(set(input_df.columns))

    def test_correct_response_derived_from_task(self, rdk_dfs):
        full_df, _ = rdk_dfs
        motion_left = full_df[(full_df["task"] == "motion") &
                               (full_df["motion"] == "up")]
        assert (motion_left["correct_response"] == "left").all()

        color_left = full_df[(full_df["task"] == "color") &
                              (full_df["color"] == "blue")]
        assert (color_left["correct_response"] == "left").all()

    def test_coherence_in_range(self, rdk_dfs):
        full_df, _ = rdk_dfs
        for col in ["motion_coherence", "color_coherence", "orientation_coherence"]:
            assert full_df[col].between(0.0, 1.0).all(), f"{col} out of [0,1]"

    def test_current_stimulus_difficulty(self, rdk_dfs):
        full_df, _ = rdk_dfs
        motion = full_df[full_df["task"] == "motion"]
        expected = 1 - motion["motion_coherence"]
        pd.testing.assert_series_equal(
            motion["current_stimulus_difficulty"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_difficulty_in_range(self, rdk_dfs):
        full_df, _ = rdk_dfs
        assert full_df["current_stimulus_difficulty"].between(0.0, 1.0).all()
        valid_past = full_df["past_stimulus_difficulty"].dropna()
        assert valid_past.between(0.0, 1.0).all()

    def test_past_stimulus_difficulty_is_block_local_lag(self, rdk_dfs):
        full_df, _ = rdk_dfs
        expected = full_df.groupby(
            ["participant_id", "block_index"],
            sort=False,
        )["current_stimulus_difficulty"].shift(1)

        pd.testing.assert_series_equal(
            full_df["past_stimulus_difficulty"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_n2_task_inhibition_levels(self, rdk_dfs):
        full_df, _ = rdk_dfs
        valid = full_df["n2_task_inhibition"].dropna()
        assert set(valid.unique()) <= {"aba_return", "cba_nonreturn", "other"}

    def test_task_transition_nan_at_block_starts(self, rdk_dfs):
        full_df, _ = rdk_dfs
        expected_nan = N_PARTICIPANTS * N_BLOCKS
        assert full_df["task_transition"].isna().sum() == expected_nan

    def test_outcome_binary(self, rdk_dfs):
        full_df, _ = rdk_dfs
        assert full_df["correct"].isin([0, 1]).all()


class TestRDKStatistics:
    def test_task_switch_cost(self, rdk_dfs):
        full_df, _ = rdk_dfs
        valid = full_df.dropna(subset=["task_transition"])
        rep = valid[valid["task_transition"] == "repeat"]["correct"]
        swi = valid[valid["task_transition"] == "switch"]["correct"]
        assert rep.mean() > swi.mean(), \
            f"repeat ({rep.mean():.3f}) not > switch ({swi.mean():.3f})"

    def test_difficulty_effect(self, rdk_dfs):
        full_df, _ = rdk_dfs
        easy = full_df[full_df["current_stimulus_difficulty"] < 0.5]["correct"]
        hard = full_df[full_df["current_stimulus_difficulty"] >= 0.5]["correct"]
        assert easy.mean() >= hard.mean(), \
            f"easy ({easy.mean():.3f}) not >= hard ({hard.mean():.3f})"


# ===========================================================================
# Prospect theory
# ===========================================================================

@pytest.fixture(scope="module")
def prospect_dfs():
    # Use more trials for stability; for prospect theory n_blocks = n_trials
    return _load_small("prospect_theory", "config/synthetic_prospect_theory_benchmark.yaml",
                       n_participants=N_PARTICIPANTS, n_blocks=50)


class TestProspectTheoryStructure:
    def test_shape(self, prospect_dfs):
        full_df, input_df = prospect_dfs
        assert len(full_df) == N_PARTICIPANTS * 50
        assert len(input_df) == N_PARTICIPANTS * 50

    def test_observable_columns(self, prospect_dfs):
        _, input_df = prospect_dfs
        expected = {"participant_id", "trial_index",
                    "left_gain", "left_loss", "left_gain_probability",
                    "right_gain", "right_loss", "right_gain_probability",
                    "chose_left"}
        assert expected.issubset(set(input_df.columns))

    def test_hidden_columns_not_in_input(self, prospect_dfs):
        full_df, input_df = prospect_dfs
        hidden = {"expected_value_difference", "gain_difference",
                  "loss_difference", "probability_difference",
                  "dominance_relation", "previous_expected_value_difference",
                  "value_difference_transition"}
        assert hidden.issubset(set(full_df.columns))
        assert not hidden.intersection(set(input_df.columns))

    def test_gain_probability_in_range(self, prospect_dfs):
        full_df, _ = prospect_dfs
        assert full_df["left_gain_probability"].between(0.0, 1.0).all()
        assert full_df["right_gain_probability"].between(0.0, 1.0).all()

    def test_expected_value_formula(self, prospect_dfs):
        full_df, _ = prospect_dfs
        lev = (full_df["left_gain_probability"] * full_df["left_gain"]
               - (1 - full_df["left_gain_probability"]) * full_df["left_loss"])
        pd.testing.assert_series_equal(
            full_df["left_expected_value"].reset_index(drop=True),
            lev.reset_index(drop=True),
            check_names=False, rtol=1e-5,
        )

    def test_ev_difference_formula(self, prospect_dfs):
        full_df, _ = prospect_dfs
        expected = full_df["left_expected_value"] - full_df["right_expected_value"]
        pd.testing.assert_series_equal(
            full_df["expected_value_difference"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False, rtol=1e-5,
        )

    def test_dominance_relation_levels(self, prospect_dfs):
        full_df, _ = prospect_dfs
        assert set(full_df["dominance_relation"].unique()) <= {
            "left_dominates", "right_dominates", "no_dominance"
        }

    def test_value_difference_transition_levels(self, prospect_dfs):
        full_df, _ = prospect_dfs
        valid = full_df["value_difference_transition"].dropna()
        assert set(valid.unique()) <= {"repeat", "switch"}

    def test_first_trial_transition_nan(self, prospect_dfs):
        full_df, _ = prospect_dfs
        first_trials = full_df.groupby("participant_id").head(1)
        assert first_trials["previous_expected_value_difference"].isna().all()
        assert first_trials["value_difference_transition"].isna().all()

    def test_outcome_binary(self, prospect_dfs):
        full_df, _ = prospect_dfs
        assert full_df["chose_left"].isin([0, 1]).all()


class TestProspectTheoryStatistics:
    def test_ev_difference_drives_choice(self, prospect_dfs):
        """Higher EV for left gamble should predict more left choices."""
        full_df, _ = prospect_dfs
        left_better  = full_df[full_df["expected_value_difference"] > 20]["chose_left"]
        right_better = full_df[full_df["expected_value_difference"] < -20]["chose_left"]
        if len(left_better) >= 5 and len(right_better) >= 5:
            assert left_better.mean() > right_better.mean(), \
                f"left_better ({left_better.mean():.3f}) not > right_better ({right_better.mean():.3f})"
