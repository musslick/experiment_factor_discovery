"""
Nested logistic regression model comparison via Likelihood Ratio Test (LRT).

Design notes
------------
* Both models are always fit on the same set of rows — the intersection of
  non-NaN rows across all columns referenced by either formula.  This is
  essential for a valid LRT: if the null and alternative models were fit on
  different rows their log-likelihoods would not be comparable.

* Degrees of freedom are computed as df_model_alt − df_model_null, where
  statsmodels' df_model counts regressors excluding the intercept.  For a
  binary factor encoded via C() this is 1; for an L-level factor it is L−1.

* Perfect / quasi-perfect separation is flagged when the model fails to
  converge AND at least one |z-score| exceeds 10.  Callers should reject
  such candidates rather than trust the reported p-value.
"""

import re
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import chi2
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class LRTResult:
    statistic: float          # −2 × (llf_null − llf_alt)
    pvalue: float             # chi2 survival function
    dof: int                  # df_model_alt − df_model_null
    llf_null: float           # log-likelihood of null model
    llf_alt: float            # log-likelihood of alternative model
    n_obs: int                # number of rows used (shared valid mask)
    converged: bool           # whether the alternative model converged
    separation_detected: bool # True → treat result as unreliable
    formula_null: str
    formula_alt: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_columns(formula: str) -> set:
    """
    Return the set of DataFrame column names referenced in a patsy formula.

    Strips C() wrappers and ignores pure-numeric tokens and known patsy
    function names (C, I, Q, np).
    """
    # Remove C(...) wrappers, keeping the inner name
    cleaned = re.sub(r'\bC\((\w+)\)', r'\1', formula)
    # Collect all identifier-like tokens (start with letter or underscore)
    tokens = set(re.findall(r'\b([A-Za-z_]\w*)\b', cleaned))
    tokens -= {"C", "I", "Q", "np", "cr", "bs", "te"}
    return tokens


def _fit_logit_catching_separation(formula, data, **kwargs):
    """
    Fit a logit model, returning (result, separation_detected).

    statsmodels emits PerfectSeparationWarning when it detects perfect or
    quasi-perfect separation.  Catching that warning is more reliable than
    inspecting t-values after the fact, because coefficients and their standard
    errors can be NaN or Inf in the separation case.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = smf.logit(formula, data=data).fit(**kwargs)
    separation = any(
        issubclass(w.category, PerfectSeparationWarning) for w in caught
    )
    return result, separation


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_models_lrt(
    df: pd.DataFrame,
    formula_null: str,
    formula_alt: str,
) -> LRTResult:
    """
    Fit null and alternative logistic regression models on a shared set of
    valid rows, then compute a Likelihood Ratio Test.

    Parameters
    ----------
    df           : DataFrame containing all columns referenced by either formula.
    formula_null : patsy formula for the restricted (null) model,
                   e.g. ``"correct ~ 1"`` or ``"correct ~ C(congruency)"``.
    formula_alt  : patsy formula for the unrestricted (alternative) model,
                   e.g. ``"correct ~ C(congruency) + C(task_transition)"``.

    Returns
    -------
    LRTResult
    """
    # Identify all columns needed by either formula and compute shared mask
    all_cols = _extract_columns(formula_null) | _extract_columns(formula_alt)
    present  = [c for c in all_cols if c in df.columns]
    valid    = df[present].notna().all(axis=1)
    df_shared = df[valid].copy()

    # Fit both models on identical rows; catch separation warnings on the alt model
    fit_kwargs = dict(disp=0, method="newton", maxiter=200, warn_convergence=False)
    res_null, _            = _fit_logit_catching_separation(formula_null, df_shared, **fit_kwargs)
    res_alt,  sep_detected = _fit_logit_catching_separation(formula_alt,  df_shared, **fit_kwargs)

    # LRT
    stat = float(-2.0 * (res_null.llf - res_alt.llf))
    dof  = max(int(res_alt.df_model) - int(res_null.df_model), 1)
    pval = float(chi2.sf(stat, df=dof))

    return LRTResult(
        statistic=stat,
        pvalue=pval,
        dof=dof,
        llf_null=float(res_null.llf),
        llf_alt=float(res_alt.llf),
        n_obs=int(len(df_shared)),
        converged=bool(res_alt.mle_retvals.get("converged", False)),
        separation_detected=sep_detected,
        formula_null=formula_null,
        formula_alt=formula_alt,
    )


# ---------------------------------------------------------------------------
# CV scoring
# ---------------------------------------------------------------------------

@dataclass
class CVScore:
    mean_ll_improvement: float  # mean per-participant LL improvement (alt − null)
    se_ll_improvement: float    # SE across participants (std / sqrt(n_participants))
    n_participants: int         # number of participants contributing finite scores
    n_folds: int                # number of CV folds used


def _participant_ll_improvements(
    res_null,
    res_alt,
    df_test: pd.DataFrame,
    all_cols: set,
    participant_col: str,
    outcome_col: str = "correct",
) -> List[float]:
    """
    For each participant in df_test, compute summed LL improvement (alt − null)
    over that participant's valid (non-NaN) trials.  Returns a list of floats,
    one per participant that had at least one valid row.
    """
    test_present = [c for c in all_cols if c in df_test.columns]
    valid_mask = df_test[test_present].notna().all(axis=1)
    df_valid = df_test[valid_mask].copy()

    scores: List[float] = []
    for pid in sorted(df_valid[participant_col].unique()):
        p_df = df_valid[df_valid[participant_col] == pid]
        if p_df.empty:
            continue
        try:
            p_alt  = np.clip(np.asarray(res_alt.predict(p_df),  dtype=float), 1e-10, 1 - 1e-10)
            p_null = np.clip(np.asarray(res_null.predict(p_df), dtype=float), 1e-10, 1 - 1e-10)
            y = p_df[outcome_col].values.astype(float)
            ll_alt  = float(np.sum(y * np.log(p_alt)  + (1 - y) * np.log(1 - p_alt)))
            ll_null = float(np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null)))
            scores.append(ll_alt - ll_null)
        except Exception:
            continue
    return scores


def score_candidate_cv(
    df: pd.DataFrame,
    formula_null: str,
    formula_alt: str,
    participant_col: str = "participant_id",
    n_folds: int = 5,
    random_state: int = 42,
) -> CVScore:
    """
    Participant-wise k-fold CV scoring of a candidate factor.

    For each fold, null and alternative models are fit on the training
    participants and evaluated on the held-out participants.  The score
    for each held-out participant is the summed LL improvement over that
    participant's valid trials.  The overall CVScore reports the mean and
    SE of these per-participant scores, giving every participant equal weight
    regardless of trial count.

    Parameters
    ----------
    df           : DataFrame with both formula columns and the candidate column.
    formula_null : Baseline formula (already contains previously discovered factors).
    formula_alt  : Baseline + candidate factor column (C(candidate_name) appended).
    participant_col : Column identifying participants.
    n_folds      : Number of CV folds (each fold = one group of participants).
    random_state : Seed for participant shuffling.
    """
    pids = sorted(df[participant_col].unique())
    n_total = len(pids)

    rng = np.random.RandomState(random_state)
    pids_shuffled = list(pids)
    rng.shuffle(pids_shuffled)

    fold_groups = [list(g) for g in np.array_split(pids_shuffled, n_folds)]
    all_cols = _extract_columns(formula_null) | _extract_columns(formula_alt)
    fit_kwargs = dict(disp=0, method="newton", maxiter=200, warn_convergence=False)

    outcome_col = formula_null.split("~")[0].strip()
    per_participant_scores: List[float] = []

    for fold_pids in fold_groups:
        if not fold_pids:
            continue
        fold_set  = set(fold_pids)
        train_set = set(pids) - fold_set

        train_df = df[df[participant_col].isin(train_set)]
        test_df  = df[df[participant_col].isin(fold_set)]

        train_present = [c for c in all_cols if c in train_df.columns]
        train_valid   = train_df[train_present].notna().all(axis=1)
        df_train_fit  = train_df[train_valid].copy()

        if df_train_fit.empty:
            per_participant_scores.extend([-np.inf] * len(fold_pids))
            continue

        try:
            res_null_f, _            = _fit_logit_catching_separation(formula_null, df_train_fit, **fit_kwargs)
            res_alt_f,  sep_detected = _fit_logit_catching_separation(formula_alt,  df_train_fit, **fit_kwargs)
        except Exception:
            per_participant_scores.extend([-np.inf] * len(fold_pids))
            continue

        if sep_detected:
            per_participant_scores.extend([-np.inf] * len(fold_pids))
            continue

        fold_scores = _participant_ll_improvements(
            res_null_f, res_alt_f, test_df, all_cols, participant_col,
            outcome_col=outcome_col,
        )
        # Participants with no valid rows are silently dropped (no -inf added)
        per_participant_scores.extend(fold_scores)

    scores_arr    = np.array(per_participant_scores)
    finite_scores = scores_arr[np.isfinite(scores_arr)]

    if len(finite_scores) == 0:
        return CVScore(
            mean_ll_improvement=-np.inf,
            se_ll_improvement=np.inf,
            n_participants=0,
            n_folds=n_folds,
        )

    mean_score = float(np.mean(finite_scores))
    se_score   = (
        float(np.std(finite_scores, ddof=1) / np.sqrt(len(finite_scores)))
        if len(finite_scores) > 1 else 0.0
    )
    return CVScore(
        mean_ll_improvement=mean_score,
        se_ll_improvement=se_score,
        n_participants=int(len(finite_scores)),
        n_folds=n_folds,
    )


def evaluate_on_held_out(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    formula_null: str,
    formula_alt: str,
    participant_col: str = "participant_id",
) -> float:
    """
    Fit null and alt models on df_train; return mean per-participant LL
    improvement on df_test.  Consistent with the CV scoring metric.

    Returns -inf if fitting fails or no participants could be evaluated.
    """
    all_cols = _extract_columns(formula_null) | _extract_columns(formula_alt)

    train_present = [c for c in all_cols if c in df_train.columns]
    train_valid   = df_train[train_present].notna().all(axis=1)
    df_train_fit  = df_train[train_valid].copy()

    if df_train_fit.empty:
        return -np.inf

    fit_kwargs = dict(disp=0, method="newton", maxiter=200, warn_convergence=False)
    try:
        res_null, _            = _fit_logit_catching_separation(formula_null, df_train_fit, **fit_kwargs)
        res_alt,  sep_detected = _fit_logit_catching_separation(formula_alt,  df_train_fit, **fit_kwargs)
    except Exception:
        return -np.inf

    if sep_detected:
        return -np.inf

    outcome_col = formula_null.split("~")[0].strip()
    scores = _participant_ll_improvements(
        res_null, res_alt, df_test, all_cols, participant_col,
        outcome_col=outcome_col,
    )
    return float(np.mean(scores)) if scores else -np.inf


# ---------------------------------------------------------------------------
# Formula helpers
# ---------------------------------------------------------------------------

def build_extended_formula(
    current_null: str,
    new_factor_name: str,
    factor_class: str = "discrete",
) -> str:
    """
    Append a term for new_factor_name to the right-hand side of current_null.

    For discrete factors: appends ``C(new_factor_name)`` (treatment-coded categorical).
    For continuous factors: appends the bare column name (numeric predictor).

    Examples
    --------
    >>> build_extended_formula("correct ~ 1", "congruency")
    'correct ~ C(congruency)'

    >>> build_extended_formula("correct ~ C(congruency)", "task_transition")
    'correct ~ C(congruency) + C(task_transition)'

    >>> build_extended_formula("correct ~ 1", "rt_proxy", factor_class="continuous")
    'correct ~ rt_proxy'
    """
    lhs, rhs = current_null.split("~", 1)
    rhs = rhs.strip()
    new_term = new_factor_name if factor_class == "continuous" else f"C({new_factor_name})"
    if rhs == "1":
        return f"{lhs.strip()} ~ {new_term}"
    return f"{lhs.strip()} ~ {rhs} + {new_term}"
