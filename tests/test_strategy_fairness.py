"""Fairness and budget invariants for candidate search strategies."""

import json
from types import SimpleNamespace

import pytest

from src.discovery.candidate_generator import generate_candidates, refine_candidates
from src.discovery.factor_registry import CandidateFactor
from src.discovery.strategies import build_evolution_strategy, build_seeding_strategy
from src.discovery.strategies.base import SearchContext, ScoredCandidate
from src.discovery.strategies.llm_genetic_evolver import LLMGeneticEvolver
from src.discovery.strategies.mixed import MixedEvolver, MixedSeeder
from src.utils.config import EvolutionStrategyConfig, SeedingStrategyConfig


def _context(n_to_generate: int = 6) -> SearchContext:
    return SearchContext(
        task_context="test task",
        observable_factors=[
            {"name": "color", "dtype": "categorical", "levels": ["red", "blue"]},
            {"name": "word", "dtype": "categorical", "levels": ["red", "blue"]},
            {"name": "task", "dtype": "categorical", "levels": ["color", "word"]},
        ],
        discovered_factors=[],
        hard_rejected=[],
        scored_candidates=[],
        all_scored_candidates=[],
        round_num=1,
        iteration=0,
        allowed_factor_types=["within_trial", "window"],
        allowed_factor_classes=["discrete"],
        max_window_width=4,
        n_to_generate=n_to_generate,
        top_k=2,
    )


def _seeder_config(seed: int, strategy_type: str):
    return SimpleNamespace(
        seed=seed,
        discovery=SimpleNamespace(
            seeding_strategy=SeedingStrategyConfig(
                type=strategy_type,
                n_candidates=6,
            )
        ),
        llm=SimpleNamespace(),
    )


def _names(candidates):
    return [candidate.name for candidate in candidates]


@pytest.mark.parametrize("strategy_type", ["random", "random_lookup"])
def test_random_seeders_are_reproducible_from_benchmark_seed(strategy_type):
    context = _context()
    first = build_seeding_strategy(_seeder_config(11, strategy_type), llm=None)
    replay = build_seeding_strategy(_seeder_config(11, strategy_type), llm=None)
    different = build_seeding_strategy(_seeder_config(12, strategy_type), llm=None)

    first_batches = (_names(first.seed(context)), _names(first.seed(context)))
    replay_batches = (_names(replay.seed(context)), _names(replay.seed(context)))

    assert replay_batches == first_batches
    assert _names(different.seed(context)) != first_batches[0]
    if strategy_type == "random_lookup":
        assert first_batches[0] == [
            "rlookup_task_word_wt_9e8aa0",
            "rlookup_word_w4_d92ab2",
            "rlookup_word_color_wt_2dd52d",
            "rlookup_color_word_w2_adeb78",
            "rlookup_word_color_w3_9f147b",
            "rlookup_task_w4_d71162",
        ]


def test_mutation_evolver_is_reproducible_from_benchmark_seed():
    parent = CandidateFactor(
        name="base",
        description="base factor",
        factor_type="within_trial",
        factor_class="discrete",
        levels=["a", "b"],
        depends_on=["color"],
    )
    scored = ScoredCandidate(parent, 1.0, 0.1, 0.9)
    context = _context(n_to_generate=1)
    context.scored_candidates = [scored]
    context.all_scored_candidates = [scored]
    context.iteration = 1
    context.allowed_factor_classes = ["discrete", "continuous"]

    def config(seed):
        return SimpleNamespace(
            seed=seed,
            discovery=SimpleNamespace(
                evolution_strategy=EvolutionStrategyConfig(type="mutation")
            ),
            llm=SimpleNamespace(),
        )

    first = build_evolution_strategy(config(17), llm=None)
    replay = build_evolution_strategy(config(17), llm=None)
    first_batches = (_names(first.evolve(context)), _names(first.evolve(context)))
    replay_batches = (_names(replay.evolve(context)), _names(replay.evolve(context)))

    assert replay_batches == first_batches
    assert (
        _names(build_evolution_strategy(config(18), llm=None).evolve(context))
        != first_batches[0]
    )


class _StubLLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_genetic_system_prompt_renders_window_limit():
    llm = _StubLLM("[]")
    parent = CandidateFactor(
        name="parent",
        description="test",
        factor_type="within_trial",
        levels=["a", "b"],
        depends_on=["task"],
        predicate_status="valid",
    )
    context = _context(n_to_generate=1)
    context.iteration = 1
    context.all_scored_candidates = [ScoredCandidate(parent, 1.0, 0.1, 0.9)]
    evolver = LLMGeneticEvolver(
        llm=llm,
        llm_cfg=SimpleNamespace(max_tokens_candidate=100, candidate_temperature=0.0),
        disc_cfg=SimpleNamespace(),
        evolver_cfg=SimpleNamespace(),
    )

    evolver.evolve(context)

    assert "<<max_window_width>>" not in llm.calls[0]["system"]
    assert "maximum 4" in llm.calls[0]["system"]


def test_llm_candidate_overproduction_is_capped():
    response = json.dumps([
        {
            "name": f"candidate_{index}",
            "description": "test",
            "factor_type": "within_trial",
            "levels": ["a", "b"],
            "depends_on": ["task"],
        }
        for index in range(5)
    ])
    llm = _StubLLM(response)

    generated = generate_candidates(
        llm=llm,
        observable_factors=["task"],
        discovered_so_far=[],
        rejected_so_far=[],
        round_num=1,
        max_candidates=2,
        temperature=0.0,
    )
    refined = refine_candidates(
        llm=llm,
        scored_candidates=[],
        hard_rejected=[],
        top_k=1,
        observable_factors=["task"],
        discovered_so_far=[],
        round_num=1,
        iteration_num=1,
        n_to_generate=3,
        temperature=0.0,
    )

    assert _names(generated) == ["candidate_0", "candidate_1"]
    assert _names(refined) == ["candidate_0", "candidate_1", "candidate_2"]
    assert len(llm.calls) == 2
    for call in llm.calls:
        assert "<<max_window_width>>" not in call["system"]
        assert "between 2 and 5" in call["system"]


class _OverproducingStrategy:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.requested = []

    def _candidates(self):
        return [
            CandidateFactor(
                name=f"{self.prefix}_{index}",
                description="test",
                factor_type="within_trial",
                levels=["a", "b"],
                depends_on=["task"],
            )
            for index in range(5)
        ]

    def seed(self, context):
        self.requested.append(context.n_to_generate)
        return self._candidates()

    def evolve(self, context):
        self.requested.append(context.n_to_generate)
        return self._candidates()


@pytest.mark.parametrize(
    ("mixed_type", "method_name"),
    [(MixedSeeder, "seed"), (MixedEvolver, "evolve")],
)
def test_mixed_components_cannot_exceed_their_quotas(mixed_type, method_name):
    first = _OverproducingStrategy("first")
    second = _OverproducingStrategy("second")
    mixed = mixed_type([(first, 1), (second, 2)])

    candidates = getattr(mixed, method_name)(_context(n_to_generate=5))

    assert _names(candidates) == ["first_0", "second_0", "second_1"]
    assert first.requested == [1]
    assert second.requested == [2]
