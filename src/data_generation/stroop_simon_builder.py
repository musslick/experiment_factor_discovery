"""
Generates synthetic Stroop–Simon trial sequences using SweetPea.

Observable (crossing) factors: word × color × stimulus_location → 27 trials/block.
correct_response is derived from color and is also observable.

Hidden derived factors (design only, not exposed to the discovery pipeline):
  word_color_congruency       – WithinTrial: word == color
  location_response_congruency – WithinTrial: stimulus_location == correct_response
  congruency_previous_trial   – Transition: previous trial's word_color_congruency
  response_transition         – Transition: correct_response repeated or switched
"""

import random
import numpy as np
import pandas as pd

from sweetpea import (
    Factor, DerivedLevel, WithinTrial, Transition,
    CrossBlock, synthesize_trials, RandomGen,
)

_PLACEHOLDER = ""
TRIALS_PER_BLOCK = 27   # 3 words × 3 colors × 3 locations


def _build_block() -> CrossBlock:
    word             = Factor("word",             ["red", "blue", "green"])
    color            = Factor("color",            ["red", "blue", "green"])
    stimulus_location = Factor("stimulus_location", ["left", "middle", "right"])

    # correct_response derived from color
    def cr_left(c):   return c == "red"
    def cr_middle(c): return c == "blue"
    def cr_right(c):  return c == "green"

    correct_response = Factor("correct_response", [
        DerivedLevel("left",   WithinTrial(cr_left,   [color])),
        DerivedLevel("middle", WithinTrial(cr_middle, [color])),
        DerivedLevel("right",  WithinTrial(cr_right,  [color])),
    ])

    # word_color_congruency
    def wcc_con(w, c): return w == c
    def wcc_inc(w, c): return w != c

    word_color_congruency = Factor("word_color_congruency", [
        DerivedLevel("congruent",   WithinTrial(wcc_con, [word, color])),
        DerivedLevel("incongruent", WithinTrial(wcc_inc, [word, color])),
    ])

    # location_response_congruency
    def lrc_con(sl, cr): return sl == cr
    def lrc_inc(sl, cr): return sl != cr

    location_response_congruency = Factor("location_response_congruency", [
        DerivedLevel("congruent",   WithinTrial(lrc_con, [stimulus_location, correct_response])),
        DerivedLevel("incongruent", WithinTrial(lrc_inc, [stimulus_location, correct_response])),
    ])

    # congruency_previous_trial: was the PREVIOUS trial's word_color_congruency congruent?
    # Window dict convention: w[0] = current, w[-1] = previous
    def cpt_con(w): return w[-1] == "congruent"
    def cpt_inc(w): return w[-1] == "incongruent"

    congruency_previous_trial = Factor("congruency_previous_trial", [
        DerivedLevel("congruent",   Transition(cpt_con, [word_color_congruency])),
        DerivedLevel("incongruent", Transition(cpt_inc, [word_color_congruency])),
    ])

    # response_transition: did correct_response repeat from previous trial?
    def rt_rep(w): return w[0] == w[-1]
    def rt_swi(w): return w[0] != w[-1]

    response_transition = Factor("response_transition", [
        DerivedLevel("repeat", Transition(rt_rep, [correct_response])),
        DerivedLevel("switch", Transition(rt_swi, [correct_response])),
    ])

    return CrossBlock(
        design=[
            word, color, stimulus_location,
            correct_response,
            word_color_congruency,
            location_response_congruency,
            congruency_previous_trial,
            response_transition,
        ],
        crossing=[word, color, stimulus_location],
        constraints=[],
    )


def build_stroop_simon_dataset(
    n_participants: int,
    n_blocks_per_participant: int,
    seed: int,
) -> pd.DataFrame:
    """
    Synthesise n_participants × n_blocks_per_participant blocks and assemble
    into a single DataFrame.

    Transition factors are NaN at the first trial of each block (no predecessor
    within that block).

    Returns
    -------
    pd.DataFrame with columns:
        participant_id, trial_index,
        word, color, stimulus_location, correct_response,
        word_color_congruency, location_response_congruency,
        congruency_previous_trial, response_transition
    """
    random.seed(seed)
    np.random.seed(seed)

    block           = _build_block()
    n_total_blocks  = n_participants * n_blocks_per_participant
    experiments     = synthesize_trials(block, samples=n_total_blocks, sampling_strategy=RandomGen)

    rows = []
    for p_idx in range(n_participants):
        trial_counter = 0
        for b_idx in range(n_blocks_per_participant):
            exp = experiments[p_idx * n_blocks_per_participant + b_idx]
            for t_idx in range(TRIALS_PER_BLOCK):
                cpt_raw = exp["congruency_previous_trial"][t_idx]
                rt_raw  = exp["response_transition"][t_idx]
                rows.append({
                    "participant_id":              p_idx,
                    "trial_index":                 trial_counter,
                    "word":                        exp["word"][t_idx],
                    "color":                       exp["color"][t_idx],
                    "stimulus_location":           exp["stimulus_location"][t_idx],
                    "correct_response":            exp["correct_response"][t_idx],
                    "word_color_congruency":       exp["word_color_congruency"][t_idx],
                    "location_response_congruency": exp["location_response_congruency"][t_idx],
                    "congruency_previous_trial":   np.nan if cpt_raw == _PLACEHOLDER else cpt_raw,
                    "response_transition":         np.nan if rt_raw  == _PLACEHOLDER else rt_raw,
                })
                trial_counter += 1

    return pd.DataFrame(rows)
