"""
Functional unit tests for Phase 1 data generation (Stroop benchmark).

Test 1 – Structural integrity:
    Verifies the generated DataFrame has the correct shape, columns, dtypes,
    and that NaN appears only where expected (transition-factor block starts).

Test 2 – Statistical validity:
    Verifies that the logistic model produces the expected directional effects:
      - congruent accuracy > incongruent accuracy
      - task-repeat accuracy > task-switch accuracy
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.data_generation.sweetpea_builder import build_stroop_dataset
from src.data_generation.stroop_model import sample_accuracy
from src.utils.config import LogisticModelConfig, ModelTerm


N_PARTICIPANTS   = 5
N_BLOCKS         = 4    # 4 × 18 = 72 trials per participant
TRIALS_PER_BLOCK = 18   # 2 tasks × 3 colors × 3 words
SEED             = 0

MODEL = LogisticModelConfig(
    intercept=0.5,
    terms=[
        ModelTerm(factor="congruency",      level="congruent", coefficient=0.8),
        ModelTerm(factor="task_transition", level="repeat",    coefficient=0.4),
    ],
)


@pytest.fixture(scope="module")
def full_df():
    df = build_stroop_dataset(N_PARTICIPANTS, N_BLOCKS, seed=SEED)
    df = sample_accuracy(df, MODEL, seed=SEED)
    return df


class TestStructure:
    EXPECTED_COLS = {
        "participant_id", "trial_index", "task", "color", "word",
        "congruency", "task_transition", "correct",
    }

    def test_row_count(self, full_df):
        expected = N_PARTICIPANTS * N_BLOCKS * TRIALS_PER_BLOCK
        assert len(full_df) == expected

    def test_columns_present(self, full_df):
        assert not (self.EXPECTED_COLS - set(full_df.columns))

    def test_participant_ids(self, full_df):
        assert sorted(full_df["participant_id"].unique()) == list(range(N_PARTICIPANTS))

    def test_trials_per_participant(self, full_df):
        counts = full_df.groupby("participant_id").size()
        assert (counts == N_BLOCKS * TRIALS_PER_BLOCK).all()

    def test_no_nan_in_non_transition_columns(self, full_df):
        for col in ["task", "color", "word", "congruency", "correct", "participant_id", "trial_index"]:
            assert full_df[col].isna().sum() == 0, f"Unexpected NaN in {col}"

    def test_task_transition_nan_count(self, full_df):
        expected_nan = N_PARTICIPANTS * N_BLOCKS
        assert full_df["task_transition"].isna().sum() == expected_nan

    def test_task_transition_nan_at_block_starts(self, full_df):
        for pid, grp in full_df.groupby("participant_id"):
            grp = grp.sort_values("trial_index").reset_index(drop=True)
            block_starts = grp.index[grp.index % TRIALS_PER_BLOCK == 0]
            assert grp.loc[block_starts, "task_transition"].isna().all()
            non_starts = grp.index[grp.index % TRIALS_PER_BLOCK != 0]
            assert grp.loc[non_starts, "task_transition"].notna().all()

    def test_factor_levels(self, full_df):
        assert set(full_df["task"].unique())       <= {"color_naming", "word_reading"}
        assert set(full_df["color"].unique())      <= {"red", "blue", "green"}
        assert set(full_df["word"].unique())       <= {"red", "blue", "green"}
        assert set(full_df["congruency"].unique()) <= {"congruent", "incongruent"}
        assert set(full_df["task_transition"].dropna().unique()) <= {"repeat", "switch"}
        assert set(full_df["correct"].unique())    <= {0, 1}

    def test_crossing_balance(self, full_df):
        counts = full_df.groupby(["task", "color", "word"]).size()
        assert counts.nunique() == 1

    def test_congruency_correctness(self, full_df):
        con = full_df[full_df["congruency"] == "congruent"]
        assert (con["color"] == con["word"]).all()
        inc = full_df[full_df["congruency"] == "incongruent"]
        assert (inc["color"] != inc["word"]).all()


class TestStatisticalValidity:
    @staticmethod
    def _z_test(n1, k1, n2, k2):
        p1, p2  = k1 / n1, k2 / n2
        p_pool  = (k1 + k2) / (n1 + n2)
        se      = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        return stats.norm.sf((p1 - p2) / se) if se > 0 else 1.0

    def test_congruency_effect(self, full_df):
        con = full_df[full_df["congruency"] == "congruent"]["correct"]
        inc = full_df[full_df["congruency"] == "incongruent"]["correct"]
        assert con.mean() > inc.mean()
        assert self._z_test(len(con), con.sum(), len(inc), inc.sum()) < 0.05

    def test_task_transition_effect(self, full_df):
        valid = full_df.dropna(subset=["task_transition"])
        rep   = valid[valid["task_transition"] == "repeat"]["correct"]
        swi   = valid[valid["task_transition"] == "switch"]["correct"]
        assert rep.mean() > swi.mean()
        assert self._z_test(len(rep), rep.sum(), len(swi), swi.sum()) < 0.05
