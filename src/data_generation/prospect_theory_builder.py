"""
Generates synthetic prospect-theory risky-choice trial sequences using SweetPea.

Since all base factors are continuous, a dummy categorical factor (1 level) is
used to satisfy SweetPea's CrossBlock API.  Each block contains exactly 1 trial,
so synthesising n_participants × n_trials blocks produces a flat independent
sequence for each participant.

Observable base factors (all continuous, sampled via SweetPea ContinuousFactor):
  left_gain, left_loss, left_gain_probability
  right_gain, right_loss, right_gain_probability

Hidden derived factors (computed post-hoc within each participant's sequence):
  left_expected_value, right_expected_value
  expected_value_difference, gain_difference, loss_difference, probability_difference
  dominance_relation                     (categorical)
  previous_expected_value_difference     (continuous, 1-trial lag)
  value_difference_transition            (categorical, sign-flip in EV advantage)

A correlation check prints a warning if any |r| among the continuous derived
factors exceeds CORR_WARN_THRESHOLD.
"""

import random
import numpy as np
import pandas as pd

from sweetpea import (
    Factor, ContinuousFactor, UniformDistribution,
    CrossBlock, synthesize_trials, RandomGen,
)

TRIALS_PER_BLOCK  = 1        # one independent trial per SweetPea "block"
CORR_WARN_THRESHOLD = 0.50   # warn when |r| between derived factors exceeds this


def _build_block(n_trials: int) -> CrossBlock:
    """
    Each participant's block contains n_trials independent trials.
    The dummy factor has n_trials levels so that SweetPea's CrossBlock
    produces exactly n_trials rows per synthesised sequence.
    """
    dummy = Factor("dummy", [str(i) for i in range(n_trials)])

    left_gain              = ContinuousFactor("left_gain",
                                              distribution=UniformDistribution(0.0, 100.0))
    left_loss              = ContinuousFactor("left_loss",
                                              distribution=UniformDistribution(0.0, 100.0))
    left_gain_probability  = ContinuousFactor("left_gain_probability",
                                              distribution=UniformDistribution(0.05, 0.95))
    right_gain             = ContinuousFactor("right_gain",
                                              distribution=UniformDistribution(0.0, 100.0))
    right_loss             = ContinuousFactor("right_loss",
                                              distribution=UniformDistribution(0.0, 100.0))
    right_gain_probability = ContinuousFactor("right_gain_probability",
                                              distribution=UniformDistribution(0.05, 0.95))

    return CrossBlock(
        design=[
            dummy,
            left_gain, left_loss, left_gain_probability,
            right_gain, right_loss, right_gain_probability,
        ],
        crossing=[dummy],
        constraints=[],
    )


def _add_hidden_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all hidden derived factors within each participant's sequence."""
    df = df.copy()

    p_L = df["left_gain_probability"]
    p_R = df["right_gain_probability"]
    g_L = df["left_gain"]
    g_R = df["right_gain"]
    l_L = df["left_loss"]
    l_R = df["right_loss"]

    df["left_expected_value"]  = p_L * g_L - (1 - p_L) * l_L
    df["right_expected_value"] = p_R * g_R - (1 - p_R) * l_R
    df["expected_value_difference"] = df["left_expected_value"] - df["right_expected_value"]
    df["gain_difference"]        = g_L - g_R
    df["loss_difference"]        = l_L - l_R
    df["probability_difference"] = p_L - p_R

    # dominance_relation
    def _dominance(row):
        l_adv = (row["left_gain"]  >= row["right_gain"] and
                 row["left_loss"]  <= row["right_loss"] and
                 row["left_gain_probability"] >= row["right_gain_probability"])
        r_adv = (row["right_gain"] >= row["left_gain"]  and
                 row["right_loss"] <= row["left_loss"]   and
                 row["right_gain_probability"] >= row["left_gain_probability"])
        l_strict = (row["left_gain"]  > row["right_gain"] or
                    row["left_loss"]  < row["right_loss"] or
                    row["left_gain_probability"] > row["right_gain_probability"])
        r_strict = (row["right_gain"] > row["left_gain"] or
                    row["right_loss"] < row["left_loss"]  or
                    row["right_gain_probability"] > row["left_gain_probability"])
        if l_adv and l_strict:
            return "left_dominates"
        if r_adv and r_strict:
            return "right_dominates"
        return "no_dominance"

    df["dominance_relation"] = df.apply(_dominance, axis=1)

    # Transition factors computed within each participant (no block boundaries)
    ev_diff_lag  = []
    vdt_vals     = []
    for pid, grp in df.groupby("participant_id", sort=False):
        ev_diff  = grp["expected_value_difference"]
        lag      = ev_diff.shift(1)
        ev_diff_lag.append(lag)

        # sign of current EV advantage
        sign_cur  = ev_diff.apply(lambda x: "left_better" if x > 0 else ("right_better" if x < 0 else "tie"))
        sign_prev = sign_cur.shift(1)

        def _vdt(cur, prev):
            if pd.isna(prev):
                return np.nan
            return "repeat" if cur == prev else "switch"

        vdt = pd.Series(
            [_vdt(c, p) for c, p in zip(sign_cur, sign_prev)],
            index=grp.index,
        )
        vdt_vals.append(vdt)

    df["previous_expected_value_difference"] = pd.concat(ev_diff_lag).reindex(df.index)
    df["value_difference_transition"]        = pd.concat(vdt_vals).reindex(df.index)

    return df


def _check_correlations(df: pd.DataFrame) -> None:
    """Print a warning if any pairwise |r| among continuous derived factors is high."""
    check_cols = [
        "expected_value_difference", "gain_difference",
        "loss_difference", "probability_difference",
    ]
    available = [c for c in check_cols if c in df.columns]
    if len(available) < 2:
        return
    corr = df[available].corr(method="pearson")
    for i, ci in enumerate(available):
        for j, cj in enumerate(available):
            if j <= i:
                continue
            r = corr.loc[ci, cj]
            if abs(r) > CORR_WARN_THRESHOLD:
                print(
                    f"  [prospect_theory] WARNING: |r({ci}, {cj})| = {abs(r):.3f} "
                    f"> {CORR_WARN_THRESHOLD} — consider adjusting sampling distributions."
                )


def build_prospect_theory_dataset(
    n_participants: int,
    n_blocks_per_participant: int,  # interpreted as n_trials_per_participant
    seed: int,
) -> pd.DataFrame:
    """
    Synthesise n_participants × n_blocks_per_participant independent trials
    and assemble into a flat per-participant sequence, then add all hidden
    derived factors.

    Because TRIALS_PER_BLOCK = 1, n_blocks_per_participant equals the number
    of trials per participant.

    Returns
    -------
    pd.DataFrame with columns:
        participant_id, trial_index,
        left_gain, left_loss, left_gain_probability,
        right_gain, right_loss, right_gain_probability,
        left_expected_value, right_expected_value,
        expected_value_difference, gain_difference,
        loss_difference, probability_difference,
        dominance_relation,
        previous_expected_value_difference,
        value_difference_transition
    """
    random.seed(seed)
    np.random.seed(seed)

    # Each experiment is one participant's full n_trials sequence.
    # synthesize_trials returns n_participants experiments, each with n_trials rows.
    n_trials    = n_blocks_per_participant
    block       = _build_block(n_trials)
    experiments = synthesize_trials(block, samples=n_participants, sampling_strategy=RandomGen)

    rows = []
    for p_idx in range(n_participants):
        exp = experiments[p_idx]
        for t_idx in range(n_trials):
            rows.append({
                "participant_id":        p_idx,
                "trial_index":           t_idx,
                "left_gain":             exp["left_gain"][t_idx],
                "left_loss":             exp["left_loss"][t_idx],
                "left_gain_probability": exp["left_gain_probability"][t_idx],
                "right_gain":            exp["right_gain"][t_idx],
                "right_loss":            exp["right_loss"][t_idx],
                "right_gain_probability": exp["right_gain_probability"][t_idx],
            })

    df = pd.DataFrame(rows)
    df = _add_hidden_factors(df)
    _check_correlations(df)
    return df
