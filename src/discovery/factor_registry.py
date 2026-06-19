"""
Tracks candidate and discovered factors across discovery rounds.

CandidateFactor    – a factor proposed by the LLM, before statistical validation.
EvaluatedCandidate – a factor that passed screening and received a CV score but
                     was not selected as the round winner.  These are soft-rejected
                     and may be re-proposed in a later round.
DiscoveredFactor   – a candidate that passed CV scoring, was selected as winner,
                     and passed held-out validation.
DiscoveredEffect   – an interaction term accepted by the effect search phase
                     (Phase 2 of each round).  Stored as a formula extension,
                     not as a new column.
TestedInteraction  – audit record for every interaction pair evaluated by the
                     effect search, keyed to the formula state at test time.
FactorRegistry     – accumulates all of the above; maintains the evolving
                     baseline formula.

Rejection taxonomy
------------------
hard_rejected : synthesis failure, sandbox crash, encoding failure, exact
                duplicate of an already-discovered factor, OR a candidate
                that is an exact relabeling of a Cartesian product of
                simpler discovered factors.  Permanently banned —
                shown to the LLM as off-limits.
evaluated_candidates : valid, CV-scored candidates that were not selected as
                winner in their round.  NOT permanently banned — a later round
                may revisit them once more factors are in the null model.
low_scoring_candidates (property) : subset of evaluated_candidates whose mean
                CV score is <= 0 (no marginal improvement detected).
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CandidateFactor:
    name: str
    description: str
    factor_type: str          # "within_trial" | "window"
    levels: List[str]
    depends_on: List[str]
    factor_class: str = "discrete"   # "discrete" | "continuous"
    window_width: int = 2            # only meaningful when factor_type == "window"
    window_stride: int = 1           # only meaningful when factor_type == "window"
    round_num: int = 0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    compute_code: Optional[str] = None    # compute_factor function (for sandbox)
    sweetpea_code: Optional[str] = None  # SweetPea Factor definition (archival)
    predicate_status: str = "pending"    # "pending"|"valid"|"syntax_error"|"runtime_error"|"timeout"|"synthesis_failed"
    lrt_pvalue: Optional[float] = None
    accepted: bool = False
    rejection_reason: Optional[str] = None
    coarsening_of: Optional[List[str]] = None  # set when candidate is a coarsening of a Cartesian product
    proposer: str = "llm"  # "llm" | "random_seeder" | "random_lookup_seeder"


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
    validation_improvement: Optional[float] = None  # primary-outcome held-out LL gain
    validation_improvements: Optional[Dict[str, float]] = None  # per-outcome held-out LL gains
    novelty_score: float = 0.0  # 1 - max_similarity to known factors at time of evaluation

    @property
    def is_continuous(self) -> bool:
        return self.candidate.factor_class == "continuous"


@dataclass
class DiscoveredEffect:
    """
    An interaction term accepted by the effect search phase (Phase 2).
    Stored as a formula extension; no new column is synthesized.
    """
    term: str                     # e.g. "C(congruency):C(previous_congruency)"
    factor_names: List[str]       # e.g. ["congruency", "previous_congruency"], always sorted
    effect_type: str              # "interaction"
    effect_order: int             # 2 for pairwise, 3 for triple
    cv_score_mean: float
    cv_score_se: float
    n_participants: int
    validation_improvement: float
    round_num: int
    formula_with: str             # cumulative formula after adding this term
    source: str                   # "effect_search" | "decomposition_check_referral"
    llm_rationale: Optional[str] = None
    validation_improvements: Optional[Dict[str, float]] = None  # per-outcome held-out LL gains


@dataclass
class TestedInteraction:
    """
    Audit record for every interaction pair evaluated by the effect search.
    Keyed to the formula state at test time so that stale rejections are not
    re-used once the model changes.
    """
    factor_names: List[str]       # always sorted
    term: str
    formula_hash: str             # hash of the formula at test time
    formula_snapshot: str         # the actual formula string (for debugging)
    cv_score_mean: Optional[float]
    cv_score_se: Optional[float]
    outcome: str                  # "accepted" | "below_cv_threshold" | "failed_validation" | "skipped"
    round_num: int


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
        # Decompose formula into primary outcome name and RHS so that
        # per-outcome formulas can be derived on demand (multi-outcome support).
        lhs, rhs = baseline_formula.split("~", 1)
        self._primary_outcome: str = lhs.strip()
        self._formula_rhs: str = rhs.strip()
        # Populated by pipeline.py via set_outcome_variable_defs(); each entry
        # must expose a .name attribute.  Empty list = single-outcome mode.
        self._outcome_variable_defs: list = []
        # Interaction effect tracking (Phase 2)
        self.discovered_effects: List[DiscoveredEffect] = []
        self.tested_interactions: List[TestedInteraction] = []
        self.pending_interactions: List[Tuple[str, str]] = []

    # --- multi-outcome setup ---

    def set_outcome_variable_defs(self, defs: list) -> None:
        """Register all outcome variable definitions for multi-outcome mode."""
        self._outcome_variable_defs = list(defs)
        # Do NOT overwrite _primary_outcome here: it was parsed from the baseline
        # formula LHS in __init__ and may contain a transform (e.g. "np.log(latency)").
        # Replacing it with defs[0].name would silently drop that transform.

    # --- formula properties ---

    @property
    def baseline_formula(self) -> str:
        """Primary outcome's current formula (backward-compat accessor)."""
        return f"{self._primary_outcome} ~ {self._formula_rhs}"

    @property
    def baseline_formulas(self) -> Dict[str, str]:
        """Per-outcome current formulas.  Single-entry dict in single-outcome mode."""
        if not self._outcome_variable_defs:
            return {self._primary_outcome: self.baseline_formula}
        return {od.name: f"{od.name} ~ {self._formula_rhs}" for od in self._outcome_variable_defs}

    # --- mutation ---

    def register(self, factor: DiscoveredFactor) -> None:
        """Accept a factor: append to discovered list and advance all baseline formulas."""
        self.discovered.append(factor)
        _, rhs = factor.formula_with.split("~", 1)
        self._formula_rhs = rhs.strip()

    def register_effect(self, effect: DiscoveredEffect) -> None:
        """Accept an interaction effect and advance all baseline formulas."""
        self.discovered_effects.append(effect)
        _, rhs = effect.formula_with.split("~", 1)
        self._formula_rhs = rhs.strip()

    def hard_reject(self, candidate: CandidateFactor, reason: str) -> None:
        """
        Permanently ban a candidate (synthesis failure, sandbox crash, encoding
        failure, duplicate, or exact interaction relabeling).  Shown to the LLM
        as permanently off-limits.
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

    def queue_pending_interaction(self, f_i_name: str, f_j_name: str) -> None:
        """
        Queue an interaction pair discovered via the decomposition check.
        These become Tier 2 candidates in the next Phase 2 run.
        """
        pair: Tuple[str, str] = tuple(sorted([f_i_name, f_j_name]))  # type: ignore[assignment]
        if pair not in self.pending_interactions:
            self.pending_interactions.append(pair)

    def record_interaction_test(
        self,
        factor_names: List[str],
        term: str,
        formula_hash: str,
        formula_snapshot: str,
        cv_score_mean: Optional[float],
        cv_score_se: Optional[float],
        outcome: str,
        round_num: int,
    ) -> None:
        """Record an interaction evaluation (accepted or not)."""
        self.tested_interactions.append(
            TestedInteraction(
                factor_names=sorted(factor_names),
                term=term,
                formula_hash=formula_hash,
                formula_snapshot=formula_snapshot,
                cv_score_mean=cv_score_mean,
                cv_score_se=cv_score_se,
                outcome=outcome,
                round_num=round_num,
            )
        )

    # --- queries ---

    def current_formula_hash(self) -> str:
        return hashlib.md5(self.baseline_formula.encode()).hexdigest()[:12]

    def is_interaction_accepted(self, factor_names: List[str]) -> bool:
        key = tuple(sorted(factor_names))
        return any(tuple(sorted(e.factor_names)) == key for e in self.discovered_effects)

    def is_interaction_tested_under_current_formula(self, factor_names: List[str]) -> bool:
        """
        True only if this pair was evaluated under the CURRENT formula state.
        A stale rejection (from a different formula) does not count.
        """
        key = tuple(sorted(factor_names))
        current_hash = self.current_formula_hash()
        return any(
            tuple(sorted(t.factor_names)) == key and t.formula_hash == current_hash
            for t in self.tested_interactions
        )

    @property
    def low_scoring_candidates(self) -> List[EvaluatedCandidate]:
        """Evaluated candidates whose mean CV score is <= 0."""
        return [e for e in self.evaluated_candidates if e.cv_score_mean <= 0.0]

    @property
    def rejected(self) -> List[CandidateFactor]:
        """Backward-compatible access to hard_rejected."""
        return self.hard_rejected

    def get_discrete_discovered(self) -> List[DiscoveredFactor]:
        """Return only discrete discovered factors."""
        return [f for f in self.discovered if f.candidate.factor_class == "discrete"]

    def get_continuous_discovered(self) -> List[DiscoveredFactor]:
        """Return only continuous discovered factors."""
        return [f for f in self.discovered if f.candidate.factor_class == "continuous"]

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
