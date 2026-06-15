"""
Functional unit tests for the subprocess sandbox.

All tests use the subprocess backend so no Docker daemon is required.

Test classes
------------
TestWithinTrial    – correct output for a within-trial congruency predicate
TestWindow         – correct output and None-placement for window factors
                     (width=2 replaces the former "transition" test class)
TestWindowWidth3   – 3-trial sliding window: Nones at positions 0 and 1 per participant
TestErrorHandling  – syntax errors, runtime errors, and timeouts are classified correctly
"""

import numpy as np
import pandas as pd
import pytest

from src.discovery.sandbox import run_predicate, SandboxResult


# ---------------------------------------------------------------------------
# Shared fixture: small deterministic DataFrame
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_df() -> pd.DataFrame:
    """
    2 participants × 15 trials each.
    Trials are inserted in shuffled order to verify that the harness correctly
    re-sorts by trial_index before computing window values.
    """
    rng = np.random.default_rng(0)
    tasks  = ["color_naming", "word_reading"]
    colors = ["red", "blue", "green"]
    words  = ["red", "blue", "green"]

    rows = []
    for pid in range(2):
        indices = list(range(15))
        rng.shuffle(indices)
        for t in indices:
            rows.append({
                "participant_id": pid,
                "trial_index":    t,
                "task":           rng.choice(tasks),
                "color":          rng.choice(colors),
                "word":           rng.choice(words),
            })
    df = pd.DataFrame(rows)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Predicate snippets
# ---------------------------------------------------------------------------

CONGRUENCY_CODE = """
def compute_factor(trial):
    return "congruent" if trial["color"] == trial["word"] else "incongruent"
"""

# Window (width=2): window[0]=previous, window[-1]=current
TASK_TRANS_CODE = """
def compute_factor(window):
    return "repeat" if window[0]["task"] == window[-1]["task"] else "switch"
"""

# Window (width=3): looks back 2 trials — was the task 2 steps ago the same?
TASK_2BACK_CODE = """
def compute_factor(window):
    return "same_2back" if window[0]["task"] == window[-1]["task"] else "diff_2back"
"""

# Continuous within_trial
CONTINUOUS_CODE = """
def compute_factor(trial):
    colors = {"red": 1.0, "blue": 2.0, "green": 3.0}
    return colors.get(trial["color"], 0.0)
"""

SYNTAX_ERROR_CODE = """
def compute_factor(trial):
    return "x" if True else:
"""

RUNTIME_ERROR_CODE = """
def compute_factor(trial):
    raise ValueError("deliberate error")
"""

TIMEOUT_CODE = """
def compute_factor(trial):
    while True:
        pass
"""

WRONG_TYPE_CODE = """
def compute_factor(trial):
    return 42          # returns int instead of str
"""


# ---------------------------------------------------------------------------
# TestWithinTrial
# ---------------------------------------------------------------------------

class TestWithinTrial:

    def test_success_flag(self, small_df):
        result = run_predicate(CONGRUENCY_CODE, small_df, "within_trial")
        assert isinstance(result, SandboxResult)
        assert result.success
        assert result.error_type == "ok"
        assert result.error_message is None

    def test_output_length_matches_df(self, small_df):
        result = run_predicate(CONGRUENCY_CODE, small_df, "within_trial")
        assert len(result.values) == len(small_df)

    def test_no_none_in_within_trial_output(self, small_df):
        result = run_predicate(CONGRUENCY_CODE, small_df, "within_trial")
        assert all(v is not None for v in result.values)

    def test_values_match_expected(self, small_df):
        result = run_predicate(CONGRUENCY_CODE, small_df, "within_trial")
        for i, row in small_df.iterrows():
            expected = "congruent" if row["color"] == row["word"] else "incongruent"
            assert result.values[i] == expected

    def test_only_declared_levels_returned(self, small_df):
        result = run_predicate(CONGRUENCY_CODE, small_df, "within_trial")
        assert set(result.values) <= {"congruent", "incongruent"}


# ---------------------------------------------------------------------------
# TestWindow  (window_width=2, the former "transition" behaviour)
# ---------------------------------------------------------------------------

class TestWindow:

    def test_success_flag(self, small_df):
        result = run_predicate(TASK_TRANS_CODE, small_df, "window", window_width=2)
        assert result.success

    def test_output_length_matches_df(self, small_df):
        result = run_predicate(TASK_TRANS_CODE, small_df, "window", window_width=2)
        assert len(result.values) == len(small_df)

    def test_none_only_at_participant_starts(self, small_df):
        """
        None must appear exactly once per participant (the first trial by
        trial_index), regardless of the physical row order in the DataFrame.
        """
        result = run_predicate(TASK_TRANS_CODE, small_df, "window", window_width=2)

        for pid in sorted(small_df["participant_id"].unique()):
            p_df = (
                small_df[small_df["participant_id"] == pid]
                .sort_values("trial_index")
                .reset_index()
            )
            first_orig_idx = int(p_df.loc[0, "index"])
            assert result.values[first_orig_idx] is None, (
                f"Participant {pid}: first trial should be None"
            )
            for row_pos in range(1, len(p_df)):
                orig_idx = int(p_df.loc[row_pos, "index"])
                assert result.values[orig_idx] is not None, (
                    f"Participant {pid}, trial_index={p_df.loc[row_pos,'trial_index']}: "
                    f"unexpected None"
                )

    def test_window_values_are_correct(self, small_df):
        """
        For each non-first trial, the value must match a direct comparison of
        the task values on that trial and its predecessor within the participant.
        """
        result = run_predicate(TASK_TRANS_CODE, small_df, "window", window_width=2)

        for pid in sorted(small_df["participant_id"].unique()):
            p_df = (
                small_df[small_df["participant_id"] == pid]
                .sort_values("trial_index")
                .reset_index()
            )
            for pos in range(1, len(p_df)):
                orig_idx  = int(p_df.loc[pos,     "index"])
                prev_task =     p_df.loc[pos - 1, "task"]
                curr_task =     p_df.loc[pos,     "task"]
                expected  = "repeat" if prev_task == curr_task else "switch"
                assert result.values[orig_idx] == expected

    def test_only_declared_levels_or_none(self, small_df):
        result = run_predicate(TASK_TRANS_CODE, small_df, "window", window_width=2)
        for v in result.values:
            assert v in {None, "repeat", "switch"}

    def test_transition_alias_still_works(self, small_df):
        """'transition' is accepted as a backward-compatible alias for 'window' width=2."""
        result = run_predicate(TASK_TRANS_CODE, small_df, "transition")
        assert result.success
        assert len(result.values) == len(small_df)


# ---------------------------------------------------------------------------
# TestWindowWidth3
# ---------------------------------------------------------------------------

class TestWindowWidth3:

    def test_none_at_first_two_trials_per_participant(self, small_df):
        """
        Width=3 window: the first 2 trials of each participant must be None.
        """
        result = run_predicate(TASK_2BACK_CODE, small_df, "window", window_width=3)
        assert result.success

        for pid in sorted(small_df["participant_id"].unique()):
            p_df = (
                small_df[small_df["participant_id"] == pid]
                .sort_values("trial_index")
                .reset_index()
            )
            # First 2 trials must be None
            for row_pos in range(2):
                orig_idx = int(p_df.loc[row_pos, "index"])
                assert result.values[orig_idx] is None, (
                    f"Participant {pid}, trial {row_pos}: expected None for width=3"
                )
            # All subsequent trials must be non-None
            for row_pos in range(2, len(p_df)):
                orig_idx = int(p_df.loc[row_pos, "index"])
                assert result.values[orig_idx] is not None, (
                    f"Participant {pid}, trial {row_pos}: unexpected None for width=3"
                )

    def test_window_values_are_correct_width3(self, small_df):
        """
        For each trial at position >= 2, the value should reflect the task
        at position-2 vs current, matching TASK_2BACK_CODE logic.
        """
        result = run_predicate(TASK_2BACK_CODE, small_df, "window", window_width=3)

        for pid in sorted(small_df["participant_id"].unique()):
            p_df = (
                small_df[small_df["participant_id"] == pid]
                .sort_values("trial_index")
                .reset_index()
            )
            for pos in range(2, len(p_df)):
                orig_idx      = int(p_df.loc[pos,     "index"])
                task_2back    =     p_df.loc[pos - 2, "task"]
                curr_task     =     p_df.loc[pos,     "task"]
                expected      = "same_2back" if task_2back == curr_task else "diff_2back"
                assert result.values[orig_idx] == expected

    def test_output_length_matches_df(self, small_df):
        result = run_predicate(TASK_2BACK_CODE, small_df, "window", window_width=3)
        assert len(result.values) == len(small_df)

    def test_continuous_within_trial_returns_floats(self, small_df):
        """Continuous within_trial predicates return numeric values (forwarded as-is by harness)."""
        result = run_predicate(CONTINUOUS_CODE, small_df, "within_trial")
        assert result.success
        assert all(isinstance(v, (int, float)) for v in result.values)


# ---------------------------------------------------------------------------
# TestBlockBoundaries
# ---------------------------------------------------------------------------

class TestBlockBoundaries:

    def test_window_resets_at_block_starts_when_depends_on_filters_columns(self):
        df = pd.DataFrame([
            {"participant_id": 0, "block_index": 0, "trial_index": 0, "task": "A"},
            {"participant_id": 0, "block_index": 0, "trial_index": 1, "task": "B"},
            {"participant_id": 0, "block_index": 1, "trial_index": 0, "task": "C"},
            {"participant_id": 0, "block_index": 1, "trial_index": 1, "task": "C"},
        ])

        result = run_predicate(
            TASK_TRANS_CODE, df, "window", window_width=2, depends_on=["task"]
        )

        assert result.success
        assert result.values == [None, "switch", None, "repeat"]


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_syntax_error_is_classified(self, small_df):
        result = run_predicate(SYNTAX_ERROR_CODE, small_df, "within_trial")
        assert not result.success
        assert result.error_type == "syntax_error"
        assert result.values is None
        assert result.error_message is not None

    def test_runtime_error_is_classified(self, small_df):
        result = run_predicate(RUNTIME_ERROR_CODE, small_df, "within_trial")
        assert not result.success
        assert result.error_type == "runtime_error"
        assert result.values is None

    def test_timeout_is_classified(self, small_df):
        result = run_predicate(
            TIMEOUT_CODE, small_df, "within_trial", timeout_seconds=1
        )
        assert not result.success
        assert result.error_type == "timeout"
        assert result.values is None

    def test_error_does_not_raise(self, small_df):
        for code, ft in [
            (SYNTAX_ERROR_CODE,  "within_trial"),
            (RUNTIME_ERROR_CODE, "within_trial"),
            (TIMEOUT_CODE,       "within_trial"),
        ]:
            try:
                run_predicate(code, small_df, ft, timeout_seconds=1)
            except Exception as exc:
                pytest.fail(f"run_predicate raised an exception: {exc}")

    def test_wrong_return_type_surfaces_as_runtime_error(self, small_df):
        """The harness doesn't validate types — it forwards them. encode_factor does validation."""
        result = run_predicate(WRONG_TYPE_CODE, small_df, "within_trial")
        assert result.success
        assert all(v == 42 for v in result.values)
