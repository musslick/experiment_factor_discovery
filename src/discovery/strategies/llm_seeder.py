"""
LLMSeeder — wraps the existing generate_candidates() function.
Default seeding strategy; produces identical behaviour to the pre-refactoring code.
"""

from typing import Dict, List, Optional

from src.discovery.candidate_generator import generate_candidates
from src.discovery.factor_registry import CandidateFactor
from src.discovery.llm_client import LLMClient
from src.discovery.strategies.base import SearchContext, SeedingStrategy


class LLMSeeder(SeedingStrategy):
    """
    Generates candidates by prompting the LLM with the task description,
    observable factors, and previously discovered/rejected factors.
    """

    def __init__(self, llm: LLMClient, llm_cfg, disc_cfg) -> None:
        self._llm = llm
        self._llm_cfg = llm_cfg
        self._disc_cfg = disc_cfg

    def seed(self, context: SearchContext) -> List[CandidateFactor]:
        observable_names = [f["name"] for f in context.observable_factors]
        observable_descriptions: Optional[Dict[str, str]] = {
            f["name"]: f["description"]
            for f in context.observable_factors
            if f.get("description")
        } or None

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
