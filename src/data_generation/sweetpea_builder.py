"""
Generates synthetic Stroop trial sequences using SweetPea.

All factors (observable + hidden derived) are defined in the CrossBlock design.
Only the observable factors (task, color, word) are placed in the crossing,
so SweetPea counterbalances over those while still computing and outputting
the hidden derived factors for every trial.

The hidden columns are stripped in generate_data.py to produce the discovery
challenge input file.
"""

import random
import numpy as np
import pandas as pd

from sweetpea import (
    Factor,
    DerivedLevel,
    WithinTrial,
    Transition,
    CrossBlock,
    synthesize_trials,
    RandomGen,
)

# Sentinel used by SweetPea for the first trial of each Transition factor
_TRANSITION_PLACEHOLDER = ""


def _build_block() -> CrossBlock:
    """
    Defines the full Stroop experimental design in SweetPea.

    Observable (crossing) factors: task, color, word  →  2 × 3 × 3 = 18 trials/block
    Hidden derived factors (design only, not counterbalanced):
        congruency     – within-trial: color == word
        task_transition – transition:  task repeated or switched from previous trial
    """
    task  = Factor("task",  ["color_naming", "word_reading"])
    color = Factor("color", ["red", "blue", "green"])
    word  = Factor("word",  ["red", "blue", "green"])

    # --- hidden within-trial factor: congruency ---
    def con(color, word): return color == word
    def inc(color, word): return color != word

    congruency = Factor("congruency", [
        DerivedLevel("congruent",   WithinTrial(con, [color, word])),
        DerivedLevel("incongruent", WithinTrial(inc, [color, word])),
    ])

    # --- hidden transition factor: task_transition ---
    # SweetPea Transition predicate receives one list per factor:
    #   task[0]  = previous trial value
    #   task[-1] = current trial value
    def task_rep(task): return task[0] == task[-1]
    def task_swi(task): return task[0] != task[-1]

    task_transition = Factor("task_transition", [
        DerivedLevel("repeat", Transition(task_rep, [task])),
        DerivedLevel("switch", Transition(task_swi, [task])),
    ])

    return CrossBlock(
        design=[task, color, word, congruency, task_transition],
        crossing=[task, color, word],
        constraints=[],
    )


def build_stroop_dataset(
    n_participants: int,
    n_blocks_per_participant: int,
    seed: int,
) -> pd.DataFrame:
    """
    Synthesizes n_participants × n_blocks_per_participant trial sequence blocks
    and assembles them into a single DataFrame.

    Each block contains 18 trials (one full crossing of task × color × word).
    Blocks are concatenated per participant to produce ~198 trials/participant.

    task_transition is NaN for the first trial of each block (no predecessor
    within that block). This is a deliberate boundary artefact: transition values
    across block boundaries are not modelled.

    Returns
    -------
    pd.DataFrame with columns:
        participant_id  : int
        trial_index     : int  (0-based, within participant)
        task            : str
        color           : str
        word            : str
        congruency      : str  ("congruent" | "incongruent")
        task_transition : str  ("repeat" | "switch" | NaN at block starts)
    """
    random.seed(seed)
    np.random.seed(seed)

    block = _build_block()
    trials_per_block = block.trials_per_sample()

    n_total_blocks = n_participants * n_blocks_per_participant
    experiments = synthesize_trials(block, samples=n_total_blocks, sampling_strategy=RandomGen)

    rows = []
    for p_idx in range(n_participants):
        trial_counter = 0
        for b_idx in range(n_blocks_per_participant):
            exp = experiments[p_idx * n_blocks_per_participant + b_idx]
            for t_idx in range(trials_per_block):
                tt_raw = exp["task_transition"][t_idx]
                rows.append({
                    "participant_id":  p_idx,
                    "trial_index":     trial_counter,
                    "task":            exp["task"][t_idx],
                    "color":           exp["color"][t_idx],
                    "word":            exp["word"][t_idx],
                    "congruency":      exp["congruency"][t_idx],
                    # Replace SweetPea's empty-string placeholder with NaN
                    "task_transition": np.nan if tt_raw == _TRANSITION_PLACEHOLDER else tt_raw,
                })
                trial_counter += 1

    return pd.DataFrame(rows)
