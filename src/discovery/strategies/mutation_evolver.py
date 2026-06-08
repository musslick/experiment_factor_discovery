"""
MutationEvolver — pure structural mutations of the top-k scored candidates.
No LLM call. Sets compute_code=None on all offspring, forcing LLM re-synthesis
with the updated description.
"""

import copy
import random
from typing import List, Optional

from src.discovery.factor_registry import CandidateFactor
from src.discovery.strategies.base import EvolutionStrategy, SearchContext


_MUTATION_TYPES = [
    "type_scope",
    "width_adjust",
    "depends_expand",
    "depends_contract",
    "class_flip",
]


class MutationEvolver(EvolutionStrategy):
    """
    Generates next-iteration candidates by applying one structural mutation
    to each of the top-k scored parents:

    type_scope      — within_trial ↔ window (sets window_width=2 when switching)
    width_adjust    — window_width ± 1 (window factors only)
    depends_expand  — add one observable factor to depends_on
    depends_contract— remove one factor from depends_on (if |depends_on| > 1)
    class_flip      — discrete ↔ continuous (clears levels for continuous)

    Mutations that are not applicable for a given parent are skipped and a
    different mutation type is tried.  Offspring have compute_code=None so that
    the predicate synthesizer re-synthesises based on the updated description.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def evolve(self, context: SearchContext) -> List[CandidateFactor]:
        if not context.all_scored_candidates:
            return []

        all_observable_names = [f["name"] for f in context.observable_factors]
        banned = {c.name for c in context.hard_rejected}

        # Select parents: top-k by adjusted_score
        parents = sorted(
            context.all_scored_candidates,
            key=lambda sc: sc.adjusted_score,
            reverse=True,
        )[: context.top_k]

        offspring: List[CandidateFactor] = []
        for sc in parents:
            if len(offspring) >= context.n_to_generate:
                break
            mutated = self._mutate(
                sc.candidate,
                all_observable_names,
                context.allowed_factor_types,
                context.allowed_factor_classes,
                context.max_window_width,
                banned | {c.name for c in offspring},
            )
            if mutated is not None:
                offspring.append(mutated)

        return offspring[: context.n_to_generate]

    # ------------------------------------------------------------------

    def _mutate(
        self,
        parent: CandidateFactor,
        observable_names: List[str],
        allowed_types: List[str],
        allowed_classes: List[str],
        max_window_width: int,
        banned_names: set,
    ) -> Optional[CandidateFactor]:
        mutation_order = _MUTATION_TYPES[:]
        self._rng.shuffle(mutation_order)

        for mutation_type in mutation_order:
            result = self._apply_mutation(
                parent, mutation_type,
                observable_names, allowed_types, allowed_classes,
                max_window_width,
            )
            if result is None:
                continue
            if result.name in banned_names:
                continue
            return result
        return None

    def _apply_mutation(
        self,
        p: CandidateFactor,
        mutation_type: str,
        observable_names: List[str],
        allowed_types: List[str],
        allowed_classes: List[str],
        max_window_width: int,
    ) -> Optional[CandidateFactor]:
        c = copy.deepcopy(p)
        c.compute_code = None
        c.predicate_status = "pending"
        c.id = __import__("uuid").uuid4().hex[:8]

        if mutation_type == "type_scope":
            if p.factor_type == "within_trial" and "window" in allowed_types:
                c.factor_type = "window"
                c.window_width = 2
                c.name = f"{p.name}_w2"
                c.description = f"Window (width=2) version of: {p.description}"
            elif p.factor_type == "window" and "within_trial" in allowed_types:
                c.factor_type = "within_trial"
                c.window_width = 2
                c.name = f"{p.name}_wt"
                c.description = f"Within-trial version of: {p.description}"
            else:
                return None

        elif mutation_type == "width_adjust":
            if p.factor_type != "window":
                return None
            delta = self._rng.choice([-1, 1])
            new_width = p.window_width + delta
            if new_width < 2 or new_width > max_window_width:
                return None
            c.window_width = new_width
            c.name = f"{p.name}_w{new_width}"
            c.description = f"{p.description} (window width adjusted to {new_width})"

        elif mutation_type == "depends_expand":
            available = [n for n in observable_names if n not in p.depends_on]
            if not available:
                return None
            extra = self._rng.choice(available)
            c.depends_on = sorted(set(p.depends_on) | {extra})
            c.name = f"{p.name}_plus_{extra}"
            c.description = f"{p.description} (also conditioned on {extra})"

        elif mutation_type == "depends_contract":
            if len(p.depends_on) <= 1:
                return None
            to_remove = self._rng.choice(p.depends_on)
            c.depends_on = [d for d in p.depends_on if d != to_remove]
            c.name = f"{p.name}_drop_{to_remove}"
            c.description = f"{p.description} (excluding {to_remove})"

        elif mutation_type == "class_flip":
            if p.factor_class == "discrete" and "continuous" in allowed_classes:
                c.factor_class = "continuous"
                c.levels = []
                c.name = f"{p.name}_cont"
                c.description = f"Continuous version of: {p.description}"
            elif p.factor_class == "continuous" and "discrete" in allowed_classes:
                c.factor_class = "discrete"
                c.levels = ["high", "low"]
                c.name = f"{p.name}_disc"
                c.description = f"Discrete (high/low) version of: {p.description}"
            else:
                return None

        else:
            return None

        return c
