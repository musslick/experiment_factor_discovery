"""
Generates synthetic RDK task-switching trial sequences using SweetPea.

Observable (crossing) factors:
  task × motion × color × orientation → 24 trials/block.
Continuous base factors (sampled independently per trial via ContinuousFactor):
  motion_coherence, color_coherence, orientation_coherence.
correct_response is derived from (task, motion, color, orientation) and is
observable but excluded from the regression baseline (it is a deterministic
function of the other base factors).

Hidden derived factors:
  task_transition (Transition)          – categorical
  n2_task_inhibition (Window width=3)   – categorical (aba_return / cba_nonreturn / other)

Post-hoc continuous hidden factors (computed after synthesis):
  current_stimulus_difficulty  = 1 - coherence_of_current_task
  past_stimulus_difficulty     = 1 - coherence_of_previous_task
"""

import random
import numpy as np
import pandas as pd

from sweetpea import (
    Factor, DerivedLevel, WithinTrial, Transition, Window,
    ContinuousFactor, UniformDistribution,
    CrossBlock, synthesize_trials, RandomGen,
)

_PLACEHOLDER     = ""
TRIALS_PER_BLOCK = 24   # 3 tasks × 2 motions × 2 colors × 2 orientations


def _build_block() -> CrossBlock:
    task        = Factor("task",        ["motion", "color", "orientation"])
    motion      = Factor("motion",      ["up", "down"])
    color       = Factor("color",       ["blue", "red"])
    orientation = Factor("orientation", ["left", "right"])

    motion_coherence      = ContinuousFactor("motion_coherence",
                                             distribution=UniformDistribution(0.0, 1.0))
    color_coherence       = ContinuousFactor("color_coherence",
                                             distribution=UniformDistribution(0.0, 1.0))
    orientation_coherence = ContinuousFactor("orientation_coherence",
                                             distribution=UniformDistribution(0.0, 1.0))

    # correct_response dispatches on task value
    def cr_left(t, m, c, o):
        if t == "motion":      return m == "up"
        if t == "color":       return c == "blue"
        if t == "orientation": return o == "left"
        return False

    def cr_right(t, m, c, o):
        return not cr_left(t, m, c, o)

    correct_response = Factor("correct_response", [
        DerivedLevel("left",  WithinTrial(cr_left,  [task, motion, color, orientation])),
        DerivedLevel("right", WithinTrial(cr_right, [task, motion, color, orientation])),
    ])

    # task_transition (width=2): w[0]=current, w[-1]=previous
    def tt_rep(w): return w[0] == w[-1]
    def tt_swi(w): return w[0] != w[-1]

    task_transition = Factor("task_transition", [
        DerivedLevel("repeat", Transition(tt_rep, [task])),
        DerivedLevel("switch", Transition(tt_swi, [task])),
    ])

    # n2_task_inhibition (width=3): w[0]=current, w[-1]=1-back, w[-2]=2-back
    def ni_aba(w):   return w[0] == w[-2] and w[0] != w[-1]
    def ni_cba(w):   return w[0] != w[-2] and w[0] != w[-1] and w[-1] != w[-2]
    def ni_other(w): return not ni_aba(w) and not ni_cba(w)

    n2_task_inhibition = Factor("n2_task_inhibition", [
        DerivedLevel("aba_return",    Window(ni_aba,   [task], 3, 1)),
        DerivedLevel("cba_nonreturn", Window(ni_cba,   [task], 3, 1)),
        DerivedLevel("other",         Window(ni_other, [task], 3, 1)),
    ])

    return CrossBlock(
        design=[
            task, motion, color, orientation,
            motion_coherence, color_coherence, orientation_coherence,
            correct_response,
            task_transition,
            n2_task_inhibition,
        ],
        crossing=[task, motion, color, orientation],
        constraints=[],
    )


def _add_continuous_hidden_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute current_stimulus_difficulty and past_stimulus_difficulty post-hoc,
    within each participant's sequence.
    """
    df = df.copy()

    def _relevant_coherence(row):
        t = row["task"]
        if t == "motion":      return row["motion_coherence"]
        if t == "color":       return row["color_coherence"]
        if t == "orientation": return row["orientation_coherence"]
        return np.nan

    df["current_stimulus_difficulty"] = df.apply(_relevant_coherence, axis=1).rsub(1)

    group_cols = ["participant_id"]
    if "block_index" in df.columns:
        group_cols.append("block_index")

    past_vals = []
    for _, grp in df.groupby(group_cols, sort=False):
        past_vals.append(grp["current_stimulus_difficulty"].shift(1))
    df["past_stimulus_difficulty"] = pd.concat(past_vals).reindex(df.index)

    return df


def build_rdk_dataset(
    n_participants: int,
    n_blocks_per_participant: int,
    seed: int,
) -> pd.DataFrame:
    """
    Synthesise n_participants × n_blocks_per_participant blocks and assemble
    into a single DataFrame, then add continuous hidden factors.

    Transition/window factors are NaN at block-start positions.

    Returns
    -------
    pd.DataFrame with columns:
        participant_id, trial_index,
        task, motion, color, orientation,
        motion_coherence, color_coherence, orientation_coherence,
        correct_response,
        task_transition, n2_task_inhibition,
        current_stimulus_difficulty, past_stimulus_difficulty
    """
    random.seed(seed)
    np.random.seed(seed)

    block          = _build_block()
    n_total_blocks = n_participants * n_blocks_per_participant
    experiments    = synthesize_trials(block, samples=n_total_blocks, sampling_strategy=RandomGen)

    rows = []
    for p_idx in range(n_participants):
        trial_counter = 0
        for b_idx in range(n_blocks_per_participant):
            exp = experiments[p_idx * n_blocks_per_participant + b_idx]
            for t_idx in range(TRIALS_PER_BLOCK):
                tt_raw = exp["task_transition"][t_idx]
                n2_raw = exp["n2_task_inhibition"][t_idx]
                rows.append({
                    "participant_id":        p_idx,
                    "block_index":          b_idx,
                    "trial_index":           trial_counter,
                    "task":                  exp["task"][t_idx],
                    "motion":                exp["motion"][t_idx],
                    "color":                 exp["color"][t_idx],
                    "orientation":           exp["orientation"][t_idx],
                    "motion_coherence":      exp["motion_coherence"][t_idx],
                    "color_coherence":       exp["color_coherence"][t_idx],
                    "orientation_coherence": exp["orientation_coherence"][t_idx],
                    "correct_response":      exp["correct_response"][t_idx],
                    "task_transition":       np.nan if tt_raw == _PLACEHOLDER else tt_raw,
                    "n2_task_inhibition":    np.nan if n2_raw == _PLACEHOLDER else n2_raw,
                })
                trial_counter += 1

    df = pd.DataFrame(rows)
    df = _add_continuous_hidden_factors(df)
    return df
