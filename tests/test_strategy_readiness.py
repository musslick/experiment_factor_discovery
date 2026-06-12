from types import SimpleNamespace

import pandas as pd

from src.discovery.factor_registry import CandidateFactor
from src.discovery.sandbox import run_predicate
from src.discovery.strategies import build_seeding_strategy
from src.discovery.strategies.base import ScoredCandidate, SearchContext
from src.discovery.strategies.llm_genetic_evolver import _apply_diversity_guard
from src.discovery.strategies.random_seeder import (
    FactorTemplateLibrary,
    RandomLookupSeeder,
    RandomSeeder,
)
from src.data_generation import get_data_generator
from src.utils.config import load_config


def _context(iteration: int = 0) -> SearchContext:
    return SearchContext(
        task_context="",
        observable_factors=[
            {"name": "task", "dtype": "categorical", "levels": ["A", "B", "C"]},
            {"name": "color", "dtype": "categorical", "levels": ["red", "blue"]},
            {"name": "word", "dtype": "categorical", "levels": ["red", "blue"]},
        ],
        discovered_factors=[],
        hard_rejected=[],
        scored_candidates=[],
        all_scored_candidates=[],
        round_num=1,
        iteration=iteration,
        allowed_factor_types=["within_trial", "window"],
        allowed_factor_classes=["discrete"],
        max_window_width=3,
        n_to_generate=5,
        top_k=3,
    )


def _n2_template():
    candidates = FactorTemplateLibrary().enumerate(_context())
    return next(c for c in candidates if c.name == "n2_task_inhibition")


def _rdk_context() -> SearchContext:
    cfg = load_config("config/synthetic_rdk_task_switching_benchmark.yaml")
    return SearchContext(
        task_context="",
        observable_factors=[
            {"name": bf.name, "dtype": bf.dtype, "levels": bf.levels}
            for bf in cfg.base_factors
        ],
        discovered_factors=[],
        hard_rejected=[],
        scored_candidates=[],
        all_scored_candidates=[],
        round_num=1,
        iteration=0,
        allowed_factor_types=["within_trial", "window"],
        allowed_factor_classes=["discrete", "continuous"],
        max_window_width=4,
        n_to_generate=50,
        top_k=3,
    )


def _rdk_template(name: str):
    candidates = FactorTemplateLibrary().enumerate(_rdk_context())
    return next(c for c in candidates if c.name == name)


def test_task_n2_template_matches_aba_cba_other_patterns():
    candidate = _n2_template()
    df = pd.DataFrame(
        [
            {"participant_id": 1, "trial_index": i, "task": task}
            for i, task in enumerate(["A", "B", "A", "C", "C", "A", "B", "A"])
        ]
    )

    result = run_predicate(
        candidate.compute_code,
        df,
        candidate.factor_type,
        window_width=candidate.window_width,
        depends_on=candidate.depends_on,
    )

    assert result.success, result.error_message
    assert candidate.levels == ["aba_return", "cba_nonreturn", "other"]
    assert candidate.priority
    assert result.values == [
        None,
        None,
        "aba_return",
        "cba_nonreturn",
        "other",
        "other",
        "cba_nonreturn",
        "aba_return",
    ]


def test_random_seeder_prioritizes_task_n2_template():
    context = _context()
    context.n_to_generate = 1
    seeder_cfg = SimpleNamespace(seed_multiplier=1.0, template_bias="uniform")

    candidates = RandomSeeder(None, seeder_cfg, seed=42).seed(context)

    assert [c.name for c in candidates] == ["n2_task_inhibition"]


def test_random_seeder_prioritizes_rdk_carryover_template():
    context = _rdk_context()
    context.n_to_generate = 2
    seeder_cfg = SimpleNamespace(seed_multiplier=1.0, template_bias="uniform")

    candidates = RandomSeeder(None, seeder_cfg, seed=42).seed(context)

    assert [c.name for c in candidates] == [
        "n2_task_inhibition",
        "task_sel_difficulty_lag_w2",
    ]


def test_task_n2_template_matches_generated_rdk_hidden_factor():
    cfg = load_config("config/synthetic_rdk_task_switching_benchmark.yaml")
    cfg.data_generation.n_participants = 1
    cfg.data_generation.n_blocks_per_participant = 2
    cfg.seed = 123
    full_df, input_df = get_data_generator(cfg.benchmark_type).generate(cfg)
    candidate = _n2_template()

    result = run_predicate(
        candidate.compute_code,
        input_df,
        candidate.factor_type,
        window_width=candidate.window_width,
        depends_on=candidate.depends_on,
    )
    expected = [
        None if pd.isna(value) else value
        for value in full_df["n2_task_inhibition"].tolist()
    ]

    assert result.success, result.error_message
    assert result.values == expected


def test_task_selected_lag_difficulty_template_matches_generated_rdk_hidden_factor():
    cfg = load_config("config/synthetic_rdk_task_switching_benchmark.yaml")
    cfg.data_generation.n_participants = 1
    cfg.data_generation.n_blocks_per_participant = 2
    cfg.seed = 123
    full_df, input_df = get_data_generator(cfg.benchmark_type).generate(cfg)
    candidate = _rdk_template("task_sel_difficulty_lag_w2")

    result = run_predicate(
        candidate.compute_code,
        input_df,
        candidate.factor_type,
        window_width=candidate.window_width,
        depends_on=candidate.depends_on,
    )
    actual = pd.Series(result.values, dtype="float64")
    expected = full_df["past_stimulus_difficulty"].reset_index(drop=True)

    assert result.success, result.error_message
    assert candidate.priority
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_config_seed_controls_built_random_lookup_seeder():
    cfg = load_config("config/synthetic_stroop_benchmark.yaml")
    cfg.discovery.seeding_strategy.type = "random_lookup"
    cfg.discovery.seeding_strategy.n_candidates = 5
    cfg.discovery.seeding_strategy.max_table_size = 16

    first = build_seeding_strategy(cfg, llm=None).seed(_context(iteration=0))
    second = build_seeding_strategy(cfg, llm=None).seed(_context(iteration=0))

    cfg.seed += 1
    different_seed = build_seeding_strategy(cfg, llm=None).seed(_context(iteration=0))

    assert [c.name for c in first] == [c.name for c in second]
    assert [c.name for c in first] != [c.name for c in different_seed]


def test_random_lookup_seed_is_reproducible_by_context():
    seeder_cfg = SimpleNamespace(
        max_depends_on=2,
        max_output_levels=2,
        max_table_size=16,
        allow_window=True,
    )

    first = RandomLookupSeeder(None, seeder_cfg, seed=123).seed(_context(iteration=0))
    second = RandomLookupSeeder(None, seeder_cfg, seed=123).seed(_context(iteration=0))
    later_iteration = RandomLookupSeeder(None, seeder_cfg, seed=123).seed(_context(iteration=1))

    assert [c.name for c in first] == [c.name for c in second]
    assert [c.name for c in first] != [c.name for c in later_iteration]


def test_genetic_diversity_guard_drops_structural_duplicates():
    existing = CandidateFactor(
        name="existing_task_window",
        description="",
        factor_type="window",
        factor_class="discrete",
        window_width=2,
        levels=["repeat", "switch"],
        depends_on=["task"],
    )
    duplicate = CandidateFactor(
        name="duplicate_task_window",
        description="",
        factor_type="window",
        factor_class="discrete",
        window_width=2,
        levels=["same", "different"],
        depends_on=["task"],
    )
    novel = CandidateFactor(
        name="novel_color_word",
        description="",
        factor_type="within_trial",
        factor_class="discrete",
        levels=["match", "mismatch"],
        depends_on=["color", "word"],
    )
    population = [
        ScoredCandidate(
            candidate=existing,
            cv_score_mean=1.0,
            cv_score_se=0.1,
            adjusted_score=1.0,
        )
    ]

    kept = _apply_diversity_guard(
        offspring=[duplicate, novel],
        population=population,
        max_window=3,
        n_factors=3,
        threshold=0.1,
    )

    assert kept == [novel]


def test_genetic_diversity_guard_drops_same_shape_dependency_expansions():
    exact_n2 = CandidateFactor(
        name="n2_task_inhibition",
        description="",
        factor_type="window",
        factor_class="discrete",
        window_width=3,
        levels=["aba_return", "cba_nonreturn", "other"],
        depends_on=["task"],
    )
    overexpanded_n2 = CandidateFactor(
        name="n2_task_inhibition_with_coherence",
        description="",
        factor_type="window",
        factor_class="discrete",
        window_width=3,
        levels=["aba_return", "cba_nonreturn", "other"],
        depends_on=[
            "task",
            "motion_coherence",
            "color_coherence",
            "orientation_coherence",
        ],
    )
    genuine_transition = CandidateFactor(
        name="task_transition_w2",
        description="",
        factor_type="window",
        factor_class="discrete",
        window_width=2,
        levels=["repeat", "switch"],
        depends_on=["task"],
    )
    population = [
        ScoredCandidate(
            candidate=exact_n2,
            cv_score_mean=1.0,
            cv_score_se=0.1,
            adjusted_score=1.0,
        )
    ]

    kept = _apply_diversity_guard(
        offspring=[overexpanded_n2, genuine_transition],
        population=population,
        max_window=4,
        n_factors=4,
        threshold=0.1,
    )

    assert kept == [genuine_transition]
