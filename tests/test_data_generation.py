"""
Functional unit tests for Phase 1 data generation.

Test 1 – Structural integrity:
    Verifies the generated DataFrame has the correct shape, columns, dtypes,
    and that NaN appears only where expected (transition-factor block starts).

Test 2 – Statistical validity:
    Verifies that the logistic model produces the expected directional effects:
      - congruent accuracy > incongruent accuracy
      - task-repeat accuracy > task-switch accuracy
    Uses a one-sided z-test on proportions so the test is robust even with a
    small number of participants.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.data_generation.sweetpea_builder import build_stroop_dataset
from src.data_generation.stroop_model import sample_accuracy
from src.utils.config import LogisticModelConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_PARTICIPANTS        = 5   # small for speed
N_BLOCKS              = 4   # 4 × 18 = 72 trials per participant
TRIALS_PER_BLOCK      = 18  # 2 tasks × 3 colors × 3 words
SEED                  = 0

MODEL = LogisticModelConfig(
    intercept=0.5,
    congruent=0.8,
    task_repeat=0.4,
)

@pytest.fixture(scope="module")
def full_df():
    """Build and return the full dataset (with hidden factors + correct column)."""
    df = build_stroop_dataset(N_PARTICIPANTS, N_BLOCKS, seed=SEED)
    df = sample_accuracy(df, MODEL, seed=SEED)
    return df


# ---------------------------------------------------------------------------
# Test 1: Structural integrity
# ---------------------------------------------------------------------------

class TestStructure:
    EXPECTED_COLS = {
        "participant_id", "trial_index", "task", "color", "word",
        "congruency", "task_transition", "correct",
    }

    def test_row_count(self, full_df):
        expected = N_PARTICIPANTS * N_BLOCKS * TRIALS_PER_BLOCK
        assert len(full_df) == expected, (
            f"Expected {expected} rows, got {len(full_df)}"
        )

    def test_columns_present(self, full_df):
        missing = self.EXPECTED_COLS - set(full_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_participant_ids(self, full_df):
        pids = sorted(full_df["participant_id"].unique())
        assert pids == list(range(N_PARTICIPANTS))

    def test_trials_per_participant(self, full_df):
        counts = full_df.groupby("participant_id").size()
        expected = N_BLOCKS * TRIALS_PER_BLOCK
        assert (counts == expected).all(), (
            f"Not all participants have {expected} trials: {counts.to_dict()}"
        )

    def test_no_nan_in_non_transition_columns(self, full_df):
        for col in ["task", "color", "word", "congruency", "correct", "participant_id", "trial_index"]:
            n_nan = full_df[col].isna().sum()
            assert n_nan == 0, f"Unexpected NaN in column '{col}': {n_nan} NaN values"

    def test_task_transition_nan_only_at_block_starts(self, full_df):
        """
        NaN in task_transition should occur exactly once per block
        (the first trial of each block has no predecessor).
        """
        expected_nan = N_PARTICIPANTS * N_BLOCKS
        actual_nan   = full_df["task_transition"].isna().sum()
        assert actual_nan == expected_nan, (
            f"Expected {expected_nan} NaN task_transition values "
            f"(one per block start), got {actual_nan}"
        )

    def test_task_transition_nan_at_trial_index_zero_within_each_block(self, full_df):
        """Each participant's trial sequence should have NaN at multiples of TRIALS_PER_BLOCK."""
        for pid, grp in full_df.groupby("participant_id"):
            grp = grp.sort_values("trial_index").reset_index(drop=True)
            block_starts = grp.index[grp.index % TRIALS_PER_BLOCK == 0]
            assert grp.loc[block_starts, "task_transition"].isna().all(), (
                f"Participant {pid}: expected NaN at block-start positions"
            )
            non_starts = grp.index[grp.index % TRIALS_PER_BLOCK != 0]
            assert grp.loc[non_starts, "task_transition"].notna().all(), (
                f"Participant {pid}: unexpected NaN at non-block-start positions"
            )

    def test_factor_levels(self, full_df):
        assert set(full_df["task"].unique())        <= {"color_naming", "word_reading"}
        assert set(full_df["color"].unique())       <= {"red", "blue", "green"}
        assert set(full_df["word"].unique())        <= {"red", "blue", "green"}
        assert set(full_df["congruency"].unique())  <= {"congruent", "incongruent"}
        valid_tt = {"repeat", "switch"}
        actual_tt = set(full_df["task_transition"].dropna().unique())
        assert actual_tt <= valid_tt
        assert set(full_df["correct"].unique())     <= {0, 1}

    def test_crossing_balance(self, full_df):
        """
        Each (task, color, word) combination should appear equally often
        across the full dataset (perfectly balanced crossing).
        """
        counts = full_df.groupby(["task", "color", "word"]).size()
        assert counts.nunique() == 1, (
            f"Crossing not balanced — unique counts: {counts.unique()}"
        )

    def test_congruency_correctness(self, full_df):
        """Congruent trials must have color == word."""
        con_rows = full_df[full_df["congruency"] == "congruent"]
        assert (con_rows["color"] == con_rows["word"]).all()

        inc_rows = full_df[full_df["congruency"] == "incongruent"]
        assert (inc_rows["color"] != inc_rows["word"]).all()


# ---------------------------------------------------------------------------
# Test 2: Statistical validity of the logistic model
# ---------------------------------------------------------------------------

class TestStatisticalValidity:
    """
    Checks that the sampled accuracy reflects the ground-truth effect directions.
    Tests use a one-sided proportion z-test (alpha = 0.05) so that false
    failures are unlikely even with the small N used for speed.
    """

    @staticmethod
    def _proportion_z_test(n1, k1, n2, k2):
        """
        One-sided z-test: H0: p1 <= p2, H1: p1 > p2.
        Returns p-value.
        """
        p1, p2 = k1 / n1, k2 / n2
        p_pool = (k1 + k2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        if se == 0:
            return 1.0
        z = (p1 - p2) / se
        return stats.norm.sf(z)  # one-sided p-value

    def test_congruency_effect(self, full_df):
        """
        Congruent accuracy should be significantly higher than incongruent.
        With β_con = 0.8 and ~30k trials this should be extremely significant;
        even with the small test fixture (5 × 4 × 18 = 360 trials) it should hold.
        """
        con = full_df[full_df["congruency"] == "congruent"]["correct"]
        inc = full_df[full_df["congruency"] == "incongruent"]["correct"]

        # Directional check (always required)
        assert con.mean() > inc.mean(), (
            f"Congruent accuracy ({con.mean():.3f}) not greater than "
            f"incongruent ({inc.mean():.3f})"
        )

        # Statistical significance (one-sided, α = 0.05)
        p = self._proportion_z_test(len(con), con.sum(), len(inc), inc.sum())
        assert p < 0.05, (
            f"Congruency effect not significant (p = {p:.4f}); "
            f"con_acc={con.mean():.3f}, inc_acc={inc.mean():.3f}"
        )

    def test_task_transition_effect(self, full_df):
        """
        Task-repeat accuracy should be higher than task-switch accuracy.
        Only non-NaN task_transition rows are compared.
        With β_task_rep = 0.4 the effect should be detectable even in the
        small test fixture.
        """
        valid = full_df.dropna(subset=["task_transition"])
        rep   = valid[valid["task_transition"] == "repeat"]["correct"]
        swi   = valid[valid["task_transition"] == "switch"]["correct"]

        assert rep.mean() > swi.mean(), (
            f"Task-repeat accuracy ({rep.mean():.3f}) not greater than "
            f"task-switch ({swi.mean():.3f})"
        )

        p = self._proportion_z_test(len(rep), rep.sum(), len(swi), swi.sum())
        assert p < 0.05, (
            f"Task-transition effect not significant (p = {p:.4f}); "
            f"rep_acc={rep.mean():.3f}, swi_acc={swi.mean():.3f}"
        )
