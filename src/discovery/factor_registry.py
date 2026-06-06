"""
Tracks candidate and discovered factors across discovery rounds.

CandidateFactor    – a factor proposed by the LLM, before statistical validation.
EvaluatedCandidate – a factor that passed screening and received a CV score but
                     was not selected as the round winner.  These are soft-rejected
                     and may be re-proposed in a later round.
DiscoveredFactor   – a candidate that passed CV scoring, was selected as winner,
                     and passed held-out validation.
FactorRegistry     – accumulates all three; maintains the evolving baseline formula.

Rejection taxonomy
------------------
hard_rejected : synthesis failure, sandbox crash, encoding failure, or exact
                duplicate of an already-discovered factor.  Permanently banned —
                shown to the LLM as off-limits.
evaluated_candidates : valid, CV-scored candidates that were not selected as
                winner in their round.  NOT permanently banned — a later round
                may revisit them once more factors are in the null model.
low_scoring_candidates (property) : subset of evaluated_candidates whose mean
                CV score is <= 0 (no marginal improvement detected).
"""

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CandidateFactor:
    name: str
    description: str
    factor_type: str          # "within_trial" | "transition"
    levels: List[str]
    depends_on: List[str]
    round_num: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    compute_code: Optional[str] = None    # compute_factor function (for sandbox)
    sweetpea_code: Optional[str] = None  # SweetPea Factor definition (archival)
    predicate_status: str = "pending"    # "pending"|"valid"|"syntax_error"|"runtime_error"|"timeout"|"synthesis_failed"
    lrt_pvalue: Optional[float] = None
    accepted: bool = False
    rejection_reason: Optional[str] = None


@dataclass
class EvaluatedCandidate:
    """A valid, CV-scored candidate that was not accepted in its round."""
    candidate: CandidateFactor
    cv_score_mean: float
    cv_score_se: float
    round_num: int


@dataclass
class DiscoveredFactor:
    candidate: CandidateFactor
    column_name: str
    column_values: pd.Series
    lrt_statistic: float        # kept for backward compatibility; set to 0.0 in new pipeline
    lrt_pvalue: float           # kept for backward compatibility; set to 1.0 in new pipeline
    lrt_dof: int                # kept for backward compatibility; set to 0 in new pipeline
    formula_with: str           # alternative model formula that detected this factor
    validation_improvement: Optional[float] = None  # mean per-participant LL gain on held-out set


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class FactorRegistry:
    """
    Accumulates discovered, hard-rejected, and softly-evaluated factors
    across discovery rounds and maintains the current baseline formula.
    """

    def __init__(self, baseline_formula: str = "correct ~ 1"):
        self.discovered: List[DiscoveredFactor] = []
        self.hard_rejected: List[CandidateFactor] = []
        self.evaluated_candidates: List[EvaluatedCandidate] = []
        self.baseline_formula: str = baseline_formula

    # --- mutation ---

    def register(self, factor: DiscoveredFactor) -> None:
        """Accept a factor: append to discovered list and advance the baseline formula."""
        self.discovered.append(factor)
        self.baseline_formula = factor.formula_with

    def hard_reject(self, candidate: CandidateFactor, reason: str) -> None:
        """
        Permanently ban a candidate (synthesis failure, sandbox crash, encoding
        failure, or duplicate).  Shown to the LLM as permanently off-limits.
        """
        candidate.accepted = False
        candidate.rejection_reason = reason
        self.hard_rejected.append(candidate)

    def reject(self, candidate: CandidateFactor, reason: str) -> None:
        """Backward-compatible alias for hard_reject."""
        self.hard_reject(candidate, reason)

    def add_evaluated(
        self,
        candidate: CandidateFactor,
        cv_score_mean: float,
        cv_score_se: float,
    ) -> None:
        """
        Record a valid, CV-scored candidate that was not selected as winner.
        These are soft-rejected and may re-appear in later rounds.
        """
        self.evaluated_candidates.append(
            EvaluatedCandidate(
                candidate=candidate,
                cv_score_mean=cv_score_mean,
                cv_score_se=cv_score_se,
                round_num=candidate.round_num,
            )
        )

    # --- queries ---

    @property
    def low_scoring_candidates(self) -> List[EvaluatedCandidate]:
        """Evaluated candidates whose mean CV score is <= 0."""
        return [e for e in self.evaluated_candidates if e.cv_score_mean <= 0.0]

    @property
    def rejected(self) -> List[CandidateFactor]:
        """Backward-compatible access to hard_rejected."""
        return self.hard_rejected

    def get_current_formula(self) -> str:
        return self.baseline_formula

    def get_discovered_column_names(self) -> List[str]:
        return [f.column_name for f in self.discovered]

    def is_duplicate(self, candidate: CandidateFactor) -> bool:
        """
        True if an equivalent candidate is already discovered or hard-rejected.
        Evaluated (soft-rejected) candidates are NOT considered duplicates so
        the LLM may revisit them in later rounds.
        """
        key = _canonical_key(candidate.name)
        seen = (
            {_canonical_key(r.name) for r in self.hard_rejected}
            | {_canonical_key(f.candidate.name) for f in self.discovered}
        )
        return key in seen


def _canonical_key(name: str) -> str:
    return name.lower().replace(" ", "_")
