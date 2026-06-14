"""
Evaluation of discovery quality: matches discovered factors to ground-truth
factors using a bijection-based comparison that is robust to level-name
differences.

A discovered factor D matches ground-truth factor G if there exists a
bijection phi between their level sets such that for >= threshold of the
applicable (non-NaN) trials, phi(D(t)) = G(t).

The Hungarian algorithm (scipy.optimize.linear_sum_assignment) finds the
maximum-weight bijection when multiple discovered and GT factors must be
matched simultaneously.
"""

from dataclasses import dataclass
from itertools import combinations, permutations
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
    agreement_rate: float        # bijection agreement for discrete; |Spearman rho| for continuous
    correlation: Optional[float] = None   # set for continuous matches, None for discrete


@dataclass
class LevelRecoveryMatch:
    ground_truth_name: str
    ground_truth_level: str
    discovered_name: str
    discovered_subset: List[str]
    precision: float
    recall: float
    f1: float
    jaccard: float
    support: int


@dataclass
class GroundTruthLevelRecovery:
    ground_truth_name: str
    ground_truth_level: str
    recovered: bool
    best_match: Optional[LevelRecoveryMatch]


@dataclass
class LevelRecoveryReport:
    level_results: List[GroundTruthLevelRecovery]
    recovered_count: int
    total_count: int
    recall: float
    threshold: float


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
    level_recovery: Optional[LevelRecoveryReport] = None


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


def compute_nmi(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Normalized mutual information between two discrete series.
    Handles unequal level cardinalities and NaN rows (dropped before scoring).
    Returns 0.0 if fewer than 2 valid rows remain.
    """
    from sklearn.metrics import normalized_mutual_info_score
    combined = pd.DataFrame({"a": series_a, "b": series_b}).dropna()
    if len(combined) < 2:
        return 0.0
    return float(normalized_mutual_info_score(
        combined["a"].astype(str),
        combined["b"].astype(str),
        average_method="arithmetic",
    ))


def _ordered_levels(series: pd.Series, declared_levels: List[str]) -> List:
    observed = list(pd.Series(series).dropna().unique())
    levels = [lv for lv in declared_levels if lv in observed]
    levels.extend(lv for lv in observed if lv not in levels)
    return levels


def _level_subsets(levels: List, max_exhaustive_levels: int) -> List[tuple]:
    """Return non-empty, non-complete subsets to test as positive predictions."""
    levels = list(levels)
    n_levels = len(levels)
    if n_levels < 2:
        return []

    if n_levels <= max_exhaustive_levels:
        subsets = []
        for size in range(1, n_levels):
            subsets.extend(combinations(levels, size))
        return subsets

    singleton = [(lv,) for lv in levels]
    complements = [tuple(other for other in levels if other != lv) for lv in levels]
    return singleton + complements


def _score_level_subset(
    gt_series: pd.Series,
    gt_level,
    disc_series: pd.Series,
    disc_subset: tuple,
) -> Optional[tuple]:
    combined = pd.DataFrame({"gt": gt_series, "disc": disc_series}).dropna()
    if combined.empty:
        return None

    target = combined["gt"] == gt_level
    predicted = combined["disc"].isin(set(disc_subset))

    support = int(target.sum())
    if support == 0:
        return None

    tp = int((target & predicted).sum())
    fp = int((~target & predicted).sum())
    fn = int((target & ~predicted).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return precision, recall, f1, jaccard, support


def compute_level_recovery(
    ground_truth_factors: List[GroundTruthFactor],
    discovered_factors: List[DiscoveredFactor],
    full_df: pd.DataFrame,
    threshold: float = 0.95,
    max_exhaustive_levels: int = 8,
) -> LevelRecoveryReport:
    """
    Evaluate recovery of individual levels from discrete ground-truth factors.

    This is intentionally separate from full-factor bijection matching. A level
    is recovered when some subset of levels from an accepted discrete discovered
    factor isolates that ground-truth level with F1 >= threshold.
    """
    results: List[GroundTruthLevelRecovery] = []

    discrete_discovered = [
        disc for disc in discovered_factors
        if disc.candidate.factor_class == "discrete"
    ]

    for gt in ground_truth_factors:
        if getattr(gt, "factor_class", "discrete") != "discrete":
            continue
        if gt.name not in full_df.columns:
            continue

        gt_series = full_df[gt.name]
        gt_levels = _ordered_levels(gt_series, list(gt.levels))

        for gt_level in gt_levels:
            best_match: Optional[LevelRecoveryMatch] = None
            for disc in discrete_discovered:
                disc_series = disc.column_values
                disc_levels = _ordered_levels(disc_series, list(disc.candidate.levels))
                for subset in _level_subsets(disc_levels, max_exhaustive_levels):
                    scores = _score_level_subset(gt_series, gt_level, disc_series, subset)
                    if scores is None:
                        continue
                    precision, recall, f1, jaccard, support = scores
                    if best_match is not None and f1 <= best_match.f1:
                        continue
                    best_match = LevelRecoveryMatch(
                        ground_truth_name=gt.name,
                        ground_truth_level=str(gt_level),
                        discovered_name=disc.column_name,
                        discovered_subset=[str(lv) for lv in subset],
                        precision=float(precision),
                        recall=float(recall),
                        f1=float(f1),
                        jaccard=float(jaccard),
                        support=support,
                    )

            recovered = best_match is not None and best_match.f1 >= threshold
            results.append(GroundTruthLevelRecovery(
                ground_truth_name=gt.name,
                ground_truth_level=str(gt_level),
                recovered=recovered,
                best_match=best_match,
            ))

    recovered_count = sum(1 for r in results if r.recovered)
    total_count = len(results)
    recall = recovered_count / total_count if total_count else 1.0
    return LevelRecoveryReport(
        level_results=results,
        recovered_count=recovered_count,
        total_count=total_count,
        recall=float(recall),
        threshold=threshold,
    )


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
    continuous_threshold     : Minimum |rho| for continuous matches (default 0.7).

    Returns
    -------
    EvaluationReport with per-factor match details and aggregate P / R / F1.
    """
    n_gt   = len(ground_truth_factors)
    n_disc = len(discovered_factors)
    level_recovery = compute_level_recovery(
        ground_truth_factors=ground_truth_factors,
        discovered_factors=discovered_factors,
        full_df=full_df,
        threshold=threshold,
    )

    if n_gt == 0 and n_disc == 0:
        return EvaluationReport([], [], [], 1.0, 1.0, 1.0, 0, 0, level_recovery)
    if n_gt == 0:
        return EvaluationReport(
            matched_pairs=[], unmatched_ground_truth=[],
            unmatched_discovered=[f.column_name for f in discovered_factors],
            precision=0.0, recall=1.0, f1=0.0,
            n_ground_truth=0, n_discovered=n_disc,
            level_recovery=level_recovery,
        )
    if n_disc == 0:
        return EvaluationReport(
            matched_pairs=[],
            unmatched_ground_truth=[gt.name for gt in ground_truth_factors],
            unmatched_discovered=[],
            precision=1.0, recall=0.0, f1=0.0,
            n_ground_truth=n_gt, n_discovered=0,
            level_recovery=level_recovery,
        )

    # Build agreement matrix (n_gt x n_disc)
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
        level_recovery=level_recovery,
    )
