"""
Evaluation of discovery quality: matches discovered factors to ground-truth
factors using a bijection-based comparison that is robust to level-name
differences.

A discovered factor D matches ground-truth factor G if there exists a
bijection φ between their level sets such that for ≥ threshold of the
applicable (non-NaN) trials, φ(D(t)) = G(t).

The Hungarian algorithm (scipy.optimize.linear_sum_assignment) finds the
maximum-weight bijection when multiple discovered and GT factors must be
matched simultaneously.
"""

from dataclasses import dataclass
from itertools import permutations
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from src.discovery.factor_registry import DiscoveredFactor
from src.utils.config import GroundTruthFactor


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MatchedPair:
    ground_truth_name: str
    discovered_name: str
    agreement_rate: float        # bijection agreement for discrete; |Spearman ρ| for continuous
    correlation: Optional[float] = None   # set for continuous matches, None for discrete


@dataclass
class EvaluationReport:
    matched_pairs: List[MatchedPair]
    unmatched_ground_truth: List[str]   # false negatives (GT not recovered)
    unmatched_discovered: List[str]     # false positives (spurious discoveries)
    precision: float
    recall: float
    f1: float
    n_ground_truth: int
    n_discovered: int


# ---------------------------------------------------------------------------
# Core agreement metric
# ---------------------------------------------------------------------------

def compute_agreement(gt_series: pd.Series, disc_series: pd.Series) -> float:
    """
    Maximum agreement rate between gt_series and disc_series over all
    bijections between their respective level sets.

    Rows where either series is NaN are excluded (handles transition-factor
    block-start positions without any special casing).

    Returns 0.0 when:
      - no valid rows remain after dropping NaN, or
      - the two series have different numbers of unique levels (no bijection
        is possible when cardinalities differ).
    """
    combined = pd.DataFrame({"gt": gt_series, "disc": disc_series}).dropna()
    if len(combined) == 0:
        return 0.0

    gt_vals    = combined["gt"]
    disc_vals  = combined["disc"]
    disc_levels = list(disc_vals.unique())
    gt_levels   = list(gt_vals.unique())

    if len(disc_levels) != len(gt_levels):
        return 0.0   # bijection requires equal cardinality

    best = 0.0
    for perm in permutations(gt_levels):
        mapping    = dict(zip(disc_levels, perm))
        agreement  = float((disc_vals.map(mapping) == gt_vals).mean())
        best       = max(best, agreement)
    return best


def compute_correlation(gt_series: pd.Series, disc_series: pd.Series) -> float:
    """
    Absolute Spearman rank correlation between two continuous series,
    computed on rows where both are non-NaN.

    Returns 0.0 if fewer than 3 valid rows remain.
    """
    combined = pd.DataFrame({"gt": gt_series, "disc": disc_series}).dropna()
    combined["gt"]   = pd.to_numeric(combined["gt"],   errors="coerce")
    combined["disc"] = pd.to_numeric(combined["disc"], errors="coerce")
    combined = combined.dropna()
    if len(combined) < 3:
        return 0.0
    rho, _ = spearmanr(combined["gt"], combined["disc"])
    if not np.isfinite(rho):
        return 0.0
    return float(abs(rho))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_factors_bijection(
    ground_truth_factors: List[GroundTruthFactor],
    discovered_factors: List[DiscoveredFactor],
    full_df: pd.DataFrame,
    threshold: float = 0.95,
    continuous_threshold: float = 0.7,
) -> EvaluationReport:
    """
    Match discovered factors to ground-truth factors via maximum-weight bijection.

    Discrete factors are matched by bijection agreement (compute_agreement).
    Continuous factors are matched by absolute Spearman correlation (compute_correlation).
    Cross-class pairs (discrete GT vs continuous discovered, or vice versa) are
    assigned an agreement of 0 and never matched.

    Parameters
    ----------
    ground_truth_factors     : Ground-truth factor specs from the benchmark config.
    discovered_factors       : Factors accepted by the discovery pipeline.
    full_df                  : Complete dataset including ground-truth factor columns.
    threshold                : Minimum agreement rate for discrete matches (default 0.95).
    continuous_threshold     : Minimum |ρ| for continuous matches (default 0.7).

    Returns
    -------
    EvaluationReport with per-factor match details and aggregate P / R / F1.
    """
    n_gt   = len(ground_truth_factors)
    n_disc = len(discovered_factors)

    if n_gt == 0 and n_disc == 0:
        return EvaluationReport([], [], [], 1.0, 1.0, 1.0, 0, 0)
    if n_gt == 0:
        return EvaluationReport(
            matched_pairs=[], unmatched_ground_truth=[],
            unmatched_discovered=[f.column_name for f in discovered_factors],
            precision=0.0, recall=1.0, f1=0.0,
            n_ground_truth=0, n_discovered=n_disc,
        )
    if n_disc == 0:
        return EvaluationReport(
            matched_pairs=[],
            unmatched_ground_truth=[gt.name for gt in ground_truth_factors],
            unmatched_discovered=[],
            precision=1.0, recall=0.0, f1=0.0,
            n_ground_truth=n_gt, n_discovered=0,
        )

    # Build agreement matrix (n_gt × n_disc)
    agreement = np.zeros((n_gt, n_disc))
    for i, gt in enumerate(ground_truth_factors):
        gt_series = full_df[gt.name]
        gt_fc = getattr(gt, "factor_class", "discrete")
        for j, disc in enumerate(discovered_factors):
            disc_fc = disc.candidate.factor_class
            if gt_fc != disc_fc:
                agreement[i, j] = 0.0
            elif gt_fc == "continuous":
                agreement[i, j] = compute_correlation(gt_series, disc.column_values)
            else:
                agreement[i, j] = compute_agreement(gt_series, disc.column_values)

    # Maximum-weight bijection via Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-agreement)

    matched_gt:   set = set()
    matched_disc: set = set()
    matched_pairs: List[MatchedPair] = []

    for r, c in zip(row_ind, col_ind):
        gt_fc = getattr(ground_truth_factors[r], "factor_class", "discrete")
        thresh = continuous_threshold if gt_fc == "continuous" else threshold
        if agreement[r, c] >= thresh:
            corr = float(agreement[r, c]) if gt_fc == "continuous" else None
            matched_pairs.append(MatchedPair(
                ground_truth_name=ground_truth_factors[r].name,
                discovered_name=discovered_factors[c].column_name,
                agreement_rate=float(agreement[r, c]),
                correlation=corr,
            ))
            matched_gt.add(r)
            matched_disc.add(c)

    unmatched_gt   = [ground_truth_factors[i].name       for i in range(n_gt)   if i not in matched_gt]
    unmatched_disc = [discovered_factors[j].column_name  for j in range(n_disc) if j not in matched_disc]

    n_matched = len(matched_pairs)
    precision = n_matched / n_disc if n_disc > 0 else 0.0
    recall    = n_matched / n_gt   if n_gt   > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return EvaluationReport(
        matched_pairs=matched_pairs,
        unmatched_ground_truth=unmatched_gt,
        unmatched_discovered=unmatched_disc,
        precision=precision,
        recall=recall,
        f1=f1,
        n_ground_truth=n_gt,
        n_discovered=n_disc,
    )
