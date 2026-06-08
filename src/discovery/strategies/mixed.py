"""
MixedSeeder  — runs multiple seeders with per-component quotas and merges results.
MixedEvolver — runs multiple evolvers with per-component quotas and merges results.
"""

from typing import List, Tuple

from src.discovery.factor_registry import CandidateFactor
from src.discovery.strategies.base import (
    EvolutionStrategy,
    SearchContext,
    SeedingStrategy,
)


def _merge_unique(
    batches: List[List[CandidateFactor]],
    banned_names: set,
    n_want: int,
) -> List[CandidateFactor]:
    seen = set(banned_names)
    merged = []
    for batch in batches:
        for c in batch:
            if c.name not in seen:
                seen.add(c.name)
                merged.append(c)
                if len(merged) >= n_want:
                    return merged
    return merged


class MixedSeeder(SeedingStrategy):
    """
    Runs each sub-seeder with its configured n_candidates quota.
    Merges results deduplicating by name.
    """

    def __init__(self, components: List[Tuple[SeedingStrategy, int]]) -> None:
        # components: list of (seeder, n_candidates)
        self._components = components

    def seed(self, context: SearchContext) -> List[CandidateFactor]:
        banned = {c.name for c in context.hard_rejected}
        batches = []
        for seeder, n_cand in self._components:
            sub_ctx = _with_n(context, n_cand)
            batches.append(seeder.seed(sub_ctx))
        return _merge_unique(batches, banned, context.n_to_generate)


class MixedEvolver(EvolutionStrategy):
    """
    Runs each sub-evolver with its configured n_candidates quota.
    Merges results deduplicating by name.
    """

    def __init__(self, components: List[Tuple[EvolutionStrategy, int]]) -> None:
        self._components = components

    def evolve(self, context: SearchContext) -> List[CandidateFactor]:
        banned = {c.name for c in context.hard_rejected}
        batches = []
        for evolver, n_cand in self._components:
            sub_ctx = _with_n(context, n_cand)
            batches.append(evolver.evolve(sub_ctx))
        return _merge_unique(batches, banned, context.n_to_generate)


def _with_n(ctx: SearchContext, n: int) -> SearchContext:
    """Return a shallow copy of ctx with n_to_generate replaced."""
    import dataclasses
    return dataclasses.replace(ctx, n_to_generate=n)
