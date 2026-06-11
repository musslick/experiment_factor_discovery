"""
Abstract base classes and shared data structures for search strategies.

ScoredCandidate   – a candidate that passed screening and was CV-scored.
                    Exposes summary fields (cv_score_mean, cv_score_se,
                    adjusted_score) for strategies, plus internal fields
                    (cv_score, column_values) used by within_round_search.

SearchContext     – all information a strategy needs to propose candidates.
                    Passed to both seed() and evolve().

SeedingStrategy   – ABC for initial-population generators.
EvolutionStrategy – ABC for iteration-level refiners.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List

from src.discovery.factor_registry import CandidateFactor, DiscoveredFactor


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    """
    A candidate that passed all screening checks and received a CV score.

    Strategy-facing fields:
        candidate, cv_score_mean, cv_score_se, adjusted_score

    Internal fields set by within_round_search (ignored by strategies):
        cv_score      – the raw CVScore object
        column_values – pd.Series aligned to the search_df index
    """
    candidate: CandidateFactor
    cv_score_mean: float
    cv_score_se: float
    adjusted_score: float
    cv_score: Any = field(default=None, repr=False)
    column_values: Any = field(default=None, repr=False)
    novelty_score: float = 0.0
    multi_cv_score: Any = field(default=None, repr=False)   # MultiOutcomeCVScore | None


@dataclass
class SearchContext:
    """
    All information strategies need to propose candidates.
    Built by within_round_search._build_context() before each call.
    """

    # Task description — forwarded to LLM strategies
    task_context: str

    # Observable factor metadata (each entry: {name, dtype, levels, description})
    observable_factors: List[dict]

    # Already-discovered factors (available as trial dict keys in predicates)
    discovered_factors: List[DiscoveredFactor]

    # Hard-rejected candidates (permanently off-limits)
    hard_rejected: List[CandidateFactor]

    # Scored candidates from the current iteration (best → worst)
    scored_candidates: List[ScoredCandidate]

    # All scored candidates accumulated across the entire round so far
    all_scored_candidates: List[ScoredCandidate]

    # Round / iteration metadata
    round_num: int
    iteration: int   # 0 = seeding, 1+ = evolution iterations

    # Allowed factor space
    allowed_factor_types: List[str]   # e.g. ["within_trial", "window"]
    allowed_factor_classes: List[str] # e.g. ["discrete", "continuous"]
    max_window_width: int

    # Budget
    n_to_generate: int   # how many candidates to return
    top_k: int           # how many top scorers to focus on (evolution)


# ---------------------------------------------------------------------------
# Strategy interfaces
# ---------------------------------------------------------------------------

class SeedingStrategy(ABC):
    """Proposes the initial candidate batch for a new round (before any scoring)."""

    @abstractmethod
    def seed(self, context: SearchContext) -> List[CandidateFactor]:
        """
        Return up to context.n_to_generate CandidateFactor stubs.
        Must not return names already in {c.name for c in context.hard_rejected}.
        Candidates with compute_code already set bypass LLM predicate synthesis.
        """
        ...


class EvolutionStrategy(ABC):
    """Proposes the next candidate batch informed by scored results."""

    @abstractmethod
    def evolve(self, context: SearchContext) -> List[CandidateFactor]:
        """
        Return up to context.n_to_generate CandidateFactor stubs.
        Must not return names already in {c.name for c in context.hard_rejected}.
        Candidates with compute_code already set bypass LLM predicate synthesis.
        """
        ...
