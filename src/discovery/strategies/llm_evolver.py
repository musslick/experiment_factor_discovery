"""
LLMEvolver — wraps the existing refine_candidates() function.
Default evolution strategy; produces identical behaviour to the pre-refactoring code.
"""

from typing import Dict, List, Optional

from src.discovery.candidate_generator import generate_candidates, refine_candidates
from src.discovery.factor_registry import CandidateFactor
from src.discovery.llm_client import LLMClient
from src.discovery.strategies.base import EvolutionStrategy, SearchContext


class LLMEvolver(EvolutionStrategy):
    """
    Generates next-iteration candidates by prompting the LLM with CV scores
    from the current round. Falls back to generate_candidates() when
    context.all_scored_candidates is empty.
    """

    def __init__(self, llm: LLMClient, llm_cfg, disc_cfg) -> None:
        self._llm = llm
        self._llm_cfg = llm_cfg
        self._disc_cfg = disc_cfg

    def evolve(self, context: SearchContext) -> List[CandidateFactor]:
        observable_names = [f["name"] for f in context.observable_factors]
        observable_descriptions: Optional[Dict[str, str]] = {
            f["name"]: f["description"]
            for f in context.observable_factors
            if f.get("description")
        } or None

        if not context.all_scored_candidates:
            # No scored candidates yet — generate fresh
            return generate_candidates(
                llm=self._llm,
                observable_factors=observable_names,
                discovered_so_far=context.discovered_factors,
                rejected_so_far=context.hard_rejected,
                round_num=context.round_num,
                max_candidates=context.n_to_generate,
                temperature=self._llm_cfg.candidate_temperature,
                max_tokens=self._llm_cfg.max_tokens_candidate,
                max_window_width=context.max_window_width,
                task_context=context.task_context,
                observable_descriptions=observable_descriptions,
            )

        return refine_candidates(
            llm=self._llm,
            scored_candidates=context.all_scored_candidates,
            hard_rejected=context.hard_rejected,
            top_k=context.top_k,
            observable_factors=observable_names,
            discovered_so_far=context.discovered_factors,
            round_num=context.round_num,
            iteration_num=context.iteration,
            n_to_generate=context.n_to_generate,
            temperature=self._llm_cfg.candidate_temperature,
            max_tokens=self._llm_cfg.max_tokens_candidate,
            max_window_width=context.max_window_width,
            task_context=context.task_context,
            observable_descriptions=observable_descriptions,
        )
