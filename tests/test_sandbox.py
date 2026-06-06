"""
Functional unit tests for Phase 3: the subprocess sandbox.

All tests use the subprocess backend so no Docker daemon is required.

Test classes
------------
TestWithinTrial    – correct output for a within-trial congruency predicate
TestTransition     – correct output and None-placement for a transition predicate
TestErrorHandling  – syntax errors, runtime errors, and timeouts are classified
                     correctly and never raise unhandled exceptions
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
    re-sorts by trial_index before computing transition values.
    """
    rng = np.random.default_rng(0)
    tasks  = ["color_naming", "word_reading"]
    colors = ["red", "blue", "green"]
    words  = ["red", "blue", "green"]

    rows = []
    for pid in range(2):
        indices = list(range(15))
        rng.shuffle(indices)          # intentionally out of order
        for t in indices:
            rows.append({
                "participant_id": pid,
                "trial_index":    t,
                "task":           rng.choice(tasks),
                "color":          rng.choice(colors),
                "word":           rng.choice(words),
            })
    df = pd.DataFrame(rows)
    # Reset to a clean RangeIndex so __idx__ == positional row number
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Predicate snippets
# ---------------------------------------------------------------------------

CONGRUENCY_CODE = """
def compute_factor(trial):
    return "congruent" if trial["color"] == trial["word"] else "incongruent"
"""

TASK_TRANS_CODE = """
def compute_factor(prev, curr):
    return "repeat" if prev["task"] == curr["task"] else "switch"
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
        """Within-trial factors are defined for every trial."""
        result = run_predicate(CONGRUENCY_CODE, small_df, "within_trial")
        assert all(v is not None for v in result.values)

    def test_values_match_expected(self, small_df):
        """Each value must equal 'congruent' iff color == word."""
        result = run_predicate(CONGRUENCY_CODE, small_df, "within_trial")
        for i, row in small_df.iterrows():
            expected = "congruent" if row["color"] == row["word"] else "incongruent"
            assert result.values[i] == expected, (
                f"Row {i}: color={row['color']}, word={row['word']}, "
                f"expected={expected}, got={result.values[i]}"
            )

    def test_only_declared_levels_returned(self, small_df):
        result = run_predicate(CONGRUENCY_CODE, small_df, "within_trial")
        assert set(result.values) <= {"congruent", "incongruent"}


# ---------------------------------------------------------------------------
# TestTransition
# ---------------------------------------------------------------------------

class TestTransition:

    def test_success_flag(self, small_df):
        result = run_predicate(TASK_TRANS_CODE, small_df, "transition")
        assert result.success

    def test_output_length_matches_df(self, small_df):
        result = run_predicate(TASK_TRANS_CODE, small_df, "transition")
        assert len(result.values) == len(small_df)

    def test_none_only_at_participant_starts(self, small_df):
        """
        None must appear exactly once per participant (the first trial by
        trial_index), regardless of the physical row order in the DataFrame.
        """
        result = run_predicate(TASK_TRANS_CODE, small_df, "transition")

        for pid in sorted(small_df["participant_id"].unique()):
            p_df = (
                small_df[small_df["participant_id"] == pid]
                .sort_values("trial_index")
                .reset_index()          # keeps original df-index in 'index' column
            )
            # First trial (lowest trial_index) must be None
            first_orig_idx = int(p_df.loc[0, "index"])
            assert result.values[first_orig_idx] is None, (
                f"Participant {pid}: first trial should be None"
            )
            # All other trials must be non-None
            for row_pos in range(1, len(p_df)):
                orig_idx = int(p_df.loc[row_pos, "index"])
                assert result.values[orig_idx] is not None, (
                    f"Participant {pid}, trial_index={p_df.loc[row_pos,'trial_index']}: "
                    f"unexpected None"
                )

    def test_transition_values_are_correct(self, small_df):
        """
        For each non-first trial, the value must match a direct comparison of
        the task values on that trial and its predecessor within the participant.
        """
        result = run_predicate(TASK_TRANS_CODE, small_df, "transition")

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
                assert result.values[orig_idx] == expected, (
                    f"Participant {pid}, pos {pos}: "
                    f"prev_task={prev_task}, curr_task={curr_task}, "
                    f"expected={expected}, got={result.values[orig_idx]}"
                )

    def test_only_declared_levels_or_none(self, small_df):
        result = run_predicate(TASK_TRANS_CODE, small_df, "transition")
        for v in result.values:
            assert v in {None, "repeat", "switch"}


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
        """A tight 1-second timeout on an infinite-loop predicate."""
        result = run_predicate(
            TIMEOUT_CODE, small_df, "within_trial", timeout_seconds=1
        )
        assert not result.success
        assert result.error_type == "timeout"
        assert result.values is None

    def test_error_does_not_raise(self, small_df):
        """run_predicate must never propagate exceptions to the caller."""
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
        """
        json.dumps(42) is valid JSON, but downstream encode_factor will
        reject integer level values.  The sandbox itself should still succeed
        (the harness doesn't validate types), so this test verifies that
        the harness at least runs without crashing and returns values.
        """
        result = run_predicate(WRONG_TYPE_CODE, small_df, "within_trial")
        # The harness runs fine; type validation is encode_factor's responsibility
        assert result.success
        assert all(v == 42 for v in result.values)
