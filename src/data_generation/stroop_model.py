"""
Applies the ground-truth logistic model to the full Stroop dataset to sample
binary accuracy for each trial.

The model is:
    logit(P(correct)) = β0
        + β_con     × I(congruency == "congruent")
        + β_task    × I(task_transition == "repeat")
        + β_resp    × I(response_transition == "repeat")   [optional]
        + β_cs[key] × I(congruency_sequence == key)        [optional]

Trials with NaN transition values (first trial of each block) receive a 0
contribution from those terms, so P(correct) is determined by the intercept
and within-trial terms only.
"""

import numpy as np
import pandas as pd

from src.utils.config import LogisticModelConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_accuracy(
    df: pd.DataFrame,
    model: LogisticModelConfig,
    seed: int,
) -> pd.DataFrame:
    """
    Adds a binary 'correct' column to df by sampling from the logistic model.

    NaN transition values contribute 0 to the linear predictor (they are treated
    as if the corresponding factor has no effect for that trial).

    Parameters
    ----------
    df    : DataFrame from sweetpea_builder.build_stroop_dataset (includes all
            hidden factor columns)
    model : LogisticModelConfig with ground-truth coefficients
    seed  : random seed for reproducibility

    Returns
    -------
    df with an additional 'correct' column (int: 0 or 1).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    logit = np.full(len(df), model.intercept, dtype=float)

    # Within-trial term: congruency (always defined)
    logit += model.congruent * (df["congruency"] == "congruent").astype(float).values

    # Transition term: task_transition (NaN → 0 contribution via fillna)
    if model.task_repeat != 0.0 and "task_transition" in df.columns:
        task_rep = (df["task_transition"] == "repeat").astype(float).values
        logit += model.task_repeat * task_rep

    # Optional: response_transition
    if model.response_repeat != 0.0 and "response_transition" in df.columns:
        resp_rep = (df["response_transition"] == "repeat").astype(float).values
        logit += model.response_repeat * resp_rep

    # Optional: congruency_sequence (multi-level)
    if model.congruency_sequence and "congruency_sequence" in df.columns:
        for level, coef in model.congruency_sequence.items():
            if coef != 0.0:
                logit += coef * (df["congruency_sequence"] == level).astype(float).values

    p_correct = _sigmoid(logit)
    df["correct"] = rng.binomial(1, p_correct).astype(int)
    return df
