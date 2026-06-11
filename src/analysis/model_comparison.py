"""
Nested model comparison via Likelihood Ratio Test (LRT).

Supports logistic regression (binary outcomes) and linear regression
(continuous outcomes), each optionally with random intercepts per
participant via statsmodels mixedlm (linear mixed effects only).

Design notes
------------
* Both models are always fit on the same set of rows — the intersection of
  non-NaN rows across all columns referenced by either formula.  This is
  essential for a valid LRT: if the null and alternative models were fit on
  different rows their log-likelihoods would not be comparable.

* Degrees of freedom are computed as the difference in fixed-effect
  parameter counts between null and alternative models.

* Perfect / quasi-perfect separation is flagged for logistic models when the
  model fails to converge AND at least one |z-score| exceeds 10.  Callers
  should reject such candidates rather than trust the reported p-value.

* Linear mixed effects models must be fit with reml=False (maximum likelihood)
  for LRT comparisons of models with different fixed effects to be valid.
  REML-based LRT is only valid for comparing random-effects structures.
"""

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning


# ---------------------------------------------------------------------------
# ModelSpec — encapsulates statistical modelling choices
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """
    Specifies the statistical model family and random-effects structure.

    family          : "logistic" for binary outcomes (logistic regression);
                      "linear" for continuous outcomes (OLS or linear mixed effects).
    mixed_effects   : if True, add a random intercept per participant (1|participant_col).
                      Only supported for family="linear"; logistic mixed effects
                      require pymer4 / R and raise NotImplementedError.
    participant_col : grouping column for the random intercept.
    """
    family: str = "logistic"
    mixed_effects: bool = False
    participant_col: str = "participant_id"


_DEFAULT_SPEC = ModelSpec(family="logistic", mixed_effects=False)


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

def _outcome_col_from_formula(formula: str) -> str:
    """Extract the raw column name from a formula LHS, stripping any function wrappers.

    "np.log(latency) ~ ..." → "latency"
    "latency ~ ..."         → "latency"
    """
    lhs = formula.split("~")[0].strip()
    m = re.search(r'\(([^()]+)\)\s*$', lhs)
    return m.group(1).strip() if m else lhs


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


def _fit_model(
    formula: str,
    data: pd.DataFrame,
    spec: ModelSpec,
    **logit_kwargs,
) -> Tuple:
    """
    Fit a model according to spec.  Returns (result, issue_detected).

    issue_detected meanings:
      logistic fixed:  True → perfect/quasi-perfect separation detected
      linear fixed:    always False (OLS is closed-form)
      linear mixed:    True → model did not converge
    """
    if spec.family == "logistic":
        if spec.mixed_effects:
            raise NotImplementedError(
                "Logistic mixed effects (GLMM) are not supported: statsmodels has no "
                "reliable frequentist GLMM implementation.  Set mixed_effects: false "
                "for binary outcomes, or use outcome_type: continuous for linear mixed effects."
            )
        return _fit_logit_catching_separation(formula, data, **logit_kwargs)

    # Linear family
    if spec.mixed_effects:
        # Random intercept per participant: (1|participant_col).
        # reml=False is required for LRT comparisons of nested fixed-effect models.
        # Fall back to OLS when mixedlm hits a singular design matrix (e.g. a
        # candidate factor that is perfectly collinear with an existing predictor).
        try:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = smf.mixedlm(
                    formula, data, groups=data[spec.participant_col]
                ).fit(reml=False, disp=False)
            issue = not bool(getattr(result, "converged", True))
            return result, issue
        except np.linalg.LinAlgError:
            result = smf.ols(formula, data).fit()
            return result, False

    # Linear fixed effects (OLS — closed-form, never fails to converge)
    result = smf.ols(formula, data).fit()
    return result, False


def _model_dof(result) -> int:
    """
    Return the number of fixed-effect parameters from a fitted model result.

    MixedLM exposes k_fe (includes intercept).
    OLS / Logit expose df_model (excludes intercept).
    Both give the same difference when comparing nested models.
    """
    if hasattr(result, "k_fe"):      # MixedLM
        return int(result.k_fe)
    return int(result.df_model)      # OLS / Logit


def _result_converged(result) -> bool:
    """Extract convergence status from any supported result type."""
    if hasattr(result, "mle_retvals") and isinstance(result.mle_retvals, dict):
        return bool(result.mle_retvals.get("converged", False))
    return bool(getattr(result, "converged", True))


def _resid_variance(result) -> float:
    """Residual variance σ² for linear models (used in Gaussian LL computation)."""
    for attr in ("scale", "mse_resid"):
        val = getattr(result, attr, None)
        if val is not None:
            return max(float(val), 1e-10)
    return 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare_models_lrt(
    df: pd.DataFrame,
    formula_null: str,
    formula_alt: str,
    spec: Optional[ModelSpec] = None,
) -> LRTResult:
    """
    Fit null and alternative models on a shared set of valid rows, then compute
    a Likelihood Ratio Test.

    Parameters
    ----------
    df           : DataFrame containing all columns referenced by either formula.
    formula_null : patsy formula for the restricted (null) model.
    formula_alt  : patsy formula for the unrestricted (alternative) model.
    spec         : ModelSpec controlling family and mixed-effects structure.
                   Defaults to logistic fixed effects for backward compatibility.

    Returns
    -------
    LRTResult
    """
    spec = spec or _DEFAULT_SPEC

    all_cols = _extract_columns(formula_null) | _extract_columns(formula_alt)
    present  = [c for c in all_cols if c in df.columns]
    valid    = df[present].notna().all(axis=1)
    df_shared = df[valid].copy()

    logit_kwargs = dict(disp=0, method="newton", maxiter=200, warn_convergence=False)
    kw = logit_kwargs if spec.family == "logistic" else {}

    res_null, _            = _fit_model(formula_null, df_shared, spec, **kw)
    res_alt,  sep_detected = _fit_model(formula_alt,  df_shared, spec, **kw)

    stat = float(-2.0 * (res_null.llf - res_alt.llf))
    dof  = max(_model_dof(res_alt) - _model_dof(res_null), 1)
    pval = float(chi2.sf(stat, df=dof))

    return LRTResult(
        statistic=stat,
        pvalue=pval,
        dof=dof,
        llf_null=float(res_null.llf),
        llf_alt=float(res_alt.llf),
        n_obs=int(len(df_shared)),
        converged=_result_converged(res_alt),
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
    outcome_lhs: str = "",
    family: str = "logistic",
) -> List[float]:
    """
    For each participant in df_test, compute summed LL improvement (alt − null)
    over that participant's valid (non-NaN) trials.  Returns a list of floats,
    one per participant that had at least one valid row.

    For logistic models: binary cross-entropy log-likelihood.
    For linear models:   Gaussian log-likelihood using σ² from the training fit.
      Held-out participants (not seen during training) get population-level
      predictions — i.e., fixed effects only, random intercept = 0.
    """
    test_present = [c for c in all_cols if c in df_test.columns]
    valid_mask = df_test[test_present].notna().all(axis=1)
    df_valid = df_test[valid_mask].copy()

    # Pre-compute σ² for linear models once (from training-set fit)
    if family == "linear":
        sigma2_alt  = _resid_variance(res_alt)
        sigma2_null = _resid_variance(res_null)

    scores: List[float] = []
    for pid in sorted(df_valid[participant_col].unique()):
        p_df = df_valid[df_valid[participant_col] == pid]
        if p_df.empty:
            continue
        try:
            if outcome_lhs and outcome_lhs != outcome_col:
                col_ns = {col: p_df[col].values for col in p_df.columns}
                y = np.asarray(eval(outcome_lhs, {"np": np}, col_ns), dtype=float)
            else:
                y = p_df[outcome_col].values.astype(float)
            if family == "logistic":
                p_alt  = np.clip(np.asarray(res_alt.predict(p_df),  dtype=float), 1e-10, 1 - 1e-10)
                p_null = np.clip(np.asarray(res_null.predict(p_df), dtype=float), 1e-10, 1 - 1e-10)
                ll_alt  = float(np.sum(y * np.log(p_alt)  + (1 - y) * np.log(1 - p_alt)))
                ll_null = float(np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null)))
            else:
                # Gaussian LL: −n/2·log(2πσ²) − RSS/(2σ²)
                yhat_alt  = np.asarray(res_alt.predict(p_df),  dtype=float)
                yhat_null = np.asarray(res_null.predict(p_df), dtype=float)
                n = len(y)
                ll_alt  = float(-0.5 * n * np.log(2 * np.pi * sigma2_alt)
                                - np.sum((y - yhat_alt) ** 2) / (2 * sigma2_alt))
                ll_null = float(-0.5 * n * np.log(2 * np.pi * sigma2_null)
                                - np.sum((y - yhat_null) ** 2) / (2 * sigma2_null))
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
    spec: Optional[ModelSpec] = None,
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
    spec = spec or _DEFAULT_SPEC

    pids = sorted(df[participant_col].unique())

    rng = np.random.RandomState(random_state)
    pids_shuffled = list(pids)
    rng.shuffle(pids_shuffled)

    fold_groups = [list(g) for g in np.array_split(pids_shuffled, n_folds)]
    all_cols = _extract_columns(formula_null) | _extract_columns(formula_alt)
    logit_kwargs = dict(disp=0, method="newton", maxiter=200, warn_convergence=False)
    kw = logit_kwargs if spec.family == "logistic" else {}

    outcome_col = _outcome_col_from_formula(formula_null)
    outcome_lhs = formula_null.split("~")[0].strip()
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
            res_null_f, _            = _fit_model(formula_null, df_train_fit, spec, **kw)
            res_alt_f,  sep_detected = _fit_model(formula_alt,  df_train_fit, spec, **kw)
        except Exception:
            per_participant_scores.extend([-np.inf] * len(fold_pids))
            continue

        if sep_detected:
            per_participant_scores.extend([-np.inf] * len(fold_pids))
            continue

        fold_scores = _participant_ll_improvements(
            res_null_f, res_alt_f, test_df, all_cols, participant_col,
            outcome_col=outcome_col,
            outcome_lhs=outcome_lhs,
            family=spec.family,
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


@dataclass
class MultiOutcomeCVScore:
    """CV scores for all outcomes; used when multiple outcome variables are specified."""
    per_outcome: Dict[str, CVScore]

    @property
    def joint_mean(self) -> float:
        """Mean per-outcome mean LL improvement — used for ranking candidates."""
        scores = [s.mean_ll_improvement for s in self.per_outcome.values()]
        return sum(scores) / len(scores)

    @property
    def joint_min(self) -> float:
        """Minimum per-outcome mean LL improvement — used as acceptance gate."""
        return min(s.mean_ll_improvement for s in self.per_outcome.values())


@dataclass
class MultiOutcomeImprovement:
    """Held-out validation improvements for all outcomes."""
    per_outcome: Dict[str, float]
    accepted: bool   # True iff every outcome met the validation threshold


def evaluate_on_held_out(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    formula_null: str,
    formula_alt: str,
    participant_col: str = "participant_id",
    spec: Optional[ModelSpec] = None,
) -> float:
    """
    Fit null and alt models on df_train; return mean per-participant LL
    improvement on df_test.  Consistent with the CV scoring metric.

    Returns -inf if fitting fails or no participants could be evaluated.
    """
    spec = spec or _DEFAULT_SPEC

    all_cols = _extract_columns(formula_null) | _extract_columns(formula_alt)

    train_present = [c for c in all_cols if c in df_train.columns]
    train_valid   = df_train[train_present].notna().all(axis=1)
    df_train_fit  = df_train[train_valid].copy()

    if df_train_fit.empty:
        return -np.inf

    logit_kwargs = dict(disp=0, method="newton", maxiter=200, warn_convergence=False)
    kw = logit_kwargs if spec.family == "logistic" else {}
    try:
        res_null, _            = _fit_model(formula_null, df_train_fit, spec, **kw)
        res_alt,  sep_detected = _fit_model(formula_alt,  df_train_fit, spec, **kw)
    except Exception:
        return -np.inf

    if sep_detected:
        return -np.inf

    outcome_col = _outcome_col_from_formula(formula_null)
    outcome_lhs = formula_null.split("~")[0].strip()
    scores = _participant_ll_improvements(
        res_null, res_alt, df_test, all_cols, participant_col,
        outcome_col=outcome_col,
        outcome_lhs=outcome_lhs,
        family=spec.family,
    )
    return float(np.mean(scores)) if scores else -np.inf


# ---------------------------------------------------------------------------
# Final model statistics
# ---------------------------------------------------------------------------

def compute_final_model_statistics(
    df: pd.DataFrame,
    baseline_formula: str,
    discovered_factors,
    discovered_effects,
    spec: Optional[ModelSpec] = None,
) -> dict:
    """
    Run a sequential LRT for every discovered factor and interaction on the
    full dataset (all participants, not just the held-out validation set).

    Factors and interactions are processed in discovery order — sorted by
    round_num, with factors preceding interactions within the same round.
    Each LRT compares the cumulative formula before the item against the
    formula that includes it (sequential / Type-I tests).

    Returns a dict:
        {
          "factors":      [{"name": ..., "lrt_statistic": ..., "lrt_pvalue": ...,
                            "lrt_dof": ..., "pseudo_r2_mcfadden": ...,
                            "n_obs": ..., "converged": ...}, ...],
          "interactions": [{"term": ..., same fields}, ...],
        }

    ``discovered_factors`` objects must expose: .column_name, .column_values,
    .formula_with, .candidate.round_num.
    ``discovered_effects`` objects must expose: .term, .formula_with, .round_num.
    """
    # Add every discovered factor column so the LRT formulae can reference them.
    analysis_df = df.copy()
    for f in discovered_factors:
        if f.column_name not in analysis_df.columns:
            analysis_df[f.column_name] = f.column_values

    # Merge and sort: factors before effects within the same round.
    events: list = []
    for f in discovered_factors:
        events.append((f.candidate.round_num, 0, "factor", f))
    for e in discovered_effects:
        events.append((e.round_num, 1, "effect", e))
    events.sort(key=lambda x: (x[0], x[1]))

    factor_stats: list = []
    interaction_stats: list = []
    prev_formula = baseline_formula

    spec = spec or _DEFAULT_SPEC

    for _round, _order, kind, item in events:
        try:
            lrt = compare_models_lrt(analysis_df, prev_formula, item.formula_with, spec=spec)
            # McFadden pseudo-R² for the marginal contribution of this item:
            #   1 − (llf_alt / llf_null), where both log-likelihoods are negative.
            if lrt.llf_null < 0:
                pseudo_r2: Optional[float] = round(
                    1.0 - (lrt.llf_alt / lrt.llf_null), 4
                )
            else:
                pseudo_r2 = None
            stats = {
                "lrt_statistic":      round(lrt.statistic, 4),
                "lrt_pvalue":         lrt.pvalue,
                "lrt_dof":            lrt.dof,
                "pseudo_r2_mcfadden": pseudo_r2,
                "n_obs":              lrt.n_obs,
                "converged":          lrt.converged,
            }
        except Exception as exc:
            stats = {"lrt_error": str(exc)}

        if kind == "factor":
            factor_stats.append({"name": item.column_name, **stats})
        else:
            interaction_stats.append({"term": item.term, **stats})

        prev_formula = item.formula_with

    return {"factors": factor_stats, "interactions": interaction_stats}


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


def replace_formula_outcome(formula: str, outcome_name: str) -> str:
    """
    Replace the LHS of *formula* with *outcome_name*.

    Used to derive per-outcome formulas from the primary formula when
    multiple outcome variables are specified.

    Examples
    --------
    >>> replace_formula_outcome("correct ~ C(congruency)", "rt")
    'rt ~ C(congruency)'
    >>> replace_formula_outcome("correct ~ 1", "accuracy")
    'accuracy ~ 1'
    """
    _, rhs = formula.split("~", 1)
    return f"{outcome_name} ~ {rhs.strip()}"
