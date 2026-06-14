import numpy as np
import pandas as pd
from src.discovery.contrast_searcher import generate_level_vs_rest_contrasts
from src.discovery.factor_registry import CandidateFactor, FactorRegistry
from src.discovery.sandbox import run_predicate
from src.discovery.strategies.base import EvolutionStrategy, SearchContext, SeedingStrategy
from src.discovery.within_round_search import run_within_round_search
from src.utils.config import (
    BaseFactor,
    BenchmarkConfig,
    DataGenerationConfig,
    DiscoveryConfig,
    EvaluationConfig,
    LLMConfig,
    LogisticModelConfig,
    StatisticalConfig,
)


class StaticSeeder(SeedingStrategy):
    def __init__(self, candidates):
        self._candidates = candidates

    def seed(self, context: SearchContext):
        return self._candidates


class EmptyEvolver(EvolutionStrategy):
    def evolve(self, context: SearchContext):
        return []


def _state_candidate() -> CandidateFactor:
    return CandidateFactor(
        name="state_three",
        description="Three-level state candidate.",
        factor_type="within_trial",
        factor_class="discrete",
        levels=["A", "B", "C"],
        depends_on=["state"],
        compute_code="""
def compute_factor(trial: dict) -> str:
    return trial['state']
""".strip(),
    )


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        name="contrast_test",
        benchmark_type="unit",
        seed=11,
        output_dir="results",
        discovery=DiscoveryConfig(
            n_rounds=1,
            max_synthesis_retries=0,
            sandbox_timeout_seconds=10,
            sandbox_backend="subprocess",
            max_search_iterations=1,
            stability_weight=0.0,
            complexity_exponent=1.0,
            depends_on_exponent=0.0,
            cv_n_folds=3,
            min_validation_improvement=0.0,
            run_contrast_search=True,
            max_contrasts_per_candidate=8,
            run_effect_search=False,
            allowed_factor_classes=["discrete"],
        ),
        llm=LLMConfig(
            model="none",
            max_tokens_candidate=1,
            max_tokens_predicate=1,
            candidate_temperature=0.0,
            predicate_temperature=0.0,
        ),
        statistical=StatisticalConfig(min_level_count=5, outcome_type="binary"),
        evaluation=EvaluationConfig(ground_truth_factors=[], bijection_threshold=0.95),
        data_generation=DataGenerationConfig(
            n_participants=0,
            n_blocks_per_participant=0,
            outcome_variable="correct",
            task_context="Synthetic contrast test.",
            base_factors=[BaseFactor(name="state", dtype="categorical", levels=["A", "B", "C"])],
            hidden_factors=[],
            logistic_model=LogisticModelConfig(intercept=0.0, terms=[]),
        ),
    )


def _contrast_df(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(60):
        states = np.array(["A", "B", "C"] * 6, dtype=object)
        rng.shuffle(states)
        for trial_index, state in enumerate(states):
            p_correct = 0.82 if state == "B" else 0.35
            rows.append({
                "participant_id": pid,
                "block_index": 0,
                "trial_index": trial_index,
                "state": state,
                "correct": int(rng.random() < p_correct),
            })
    return pd.DataFrame(rows)


def test_level_vs_rest_contrasts_are_recomputable_through_sandbox():
    parent = _state_candidate()
    df = pd.DataFrame({
        "participant_id": [1, 1, 1, 1],
        "trial_index": [0, 1, 2, 3],
        "state": ["A", "B", "C", "B"],
    })
    parent_series = pd.Series(["A", "B", "C", "B"])

    contrasts = generate_level_vs_rest_contrasts(parent, parent_series)

    assert [c.name for c, _ in contrasts] == [
        "state_three__a_vs_rest",
        "state_three__b_vs_rest",
        "state_three__c_vs_rest",
    ]
    b_contrast, b_series = contrasts[1]
    assert getattr(b_contrast, "contrast_of") == "state_three"
    assert getattr(b_contrast, "contrast_positive_levels") == ["B"]
    assert b_series.tolist() == ["not_b", "is_b", "not_b", "is_b"]

    sandbox = run_predicate(
        predicate_code=b_contrast.compute_code,
        df=df,
        factor_type=b_contrast.factor_type,
        depends_on=b_contrast.depends_on,
    )

    assert sandbox.success, sandbox.error_message
    assert sandbox.values == ["not_b", "is_b", "not_b", "is_b"]


def test_within_round_search_scores_contrasts_and_can_select_active_contrast():
    df = _contrast_df()
    search_df = df[df["participant_id"] < 45].reset_index(drop=True)
    validation_df = df[df["participant_id"] >= 45].reset_index(drop=True)
    candidate = _state_candidate()
    candidate.round_num = 1

    result = run_within_round_search(
        search_df=search_df,
        validation_df=validation_df,
        config=_config(),
        llm=None,
        registry=FactorRegistry(baseline_formula="correct ~ 1"),
        round_num=1,
        observable_cols=list(df.columns),
        seeder=StaticSeeder([candidate]),
        evolver=EmptyEvolver(),
    )

    assert result.accepted
    assert result.winner is not None
    assert result.winner.candidate.name == "state_three__b_vs_rest"
    assert getattr(result.winner.candidate, "contrast_of") == "state_three"
    assert result.validation_improvement is not None
    assert result.validation_improvement > 0.0

    scored_names = {sc.candidate.name for sc in result.all_scored}
    assert "state_three" in scored_names
    assert "state_three__a_vs_rest" in scored_names
    assert "state_three__b_vs_rest" in scored_names
    assert "state_three__c_vs_rest" in scored_names
    assert set(result.winner.column_values.dropna().unique()) == {"is_b", "not_b"}


def _window_contrast_df(seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    tasks = np.array(["motion", "color", "orientation"], dtype=object)
    for pid in range(70):
        prev = None
        two_back = None
        for trial_index in range(48):
            task = str(rng.choice(tasks))
            transition = None if prev is None else ("repeat" if task == prev else "switch")
            signature = None
            if two_back is not None:
                if two_back == prev == task:
                    signature = "aaa"
                elif two_back == prev:
                    signature = "aab"
                elif two_back == task:
                    signature = "aba"
                elif prev == task:
                    signature = "abb"
                else:
                    signature = "abc"
            p_correct = 0.45
            if transition == "repeat":
                p_correct += 0.08
            if signature == "aba":
                p_correct -= 0.22
            rows.append({
                "participant_id": pid,
                "block_index": 0,
                "trial_index": trial_index,
                "task": task,
                "task_transition_w2": transition,
                "correct": int(rng.random() < p_correct),
            })
            two_back = prev
            prev = task
    return pd.DataFrame(rows)


def _task_signature_candidate() -> CandidateFactor:
    return CandidateFactor(
        name="task_signature_w3",
        description="Full width-3 task equality signature.",
        factor_type="window",
        factor_class="discrete",
        window_width=3,
        levels=["aaa", "aab", "aba", "abb", "abc"],
        depends_on=["task"],
        compute_code="""
def compute_factor(window: list) -> str:
    two_back = window[0]['task']
    previous = window[-2]['task']
    current = window[-1]['task']
    if two_back == previous == current:
        return 'aaa'
    if two_back == previous:
        return 'aab'
    if two_back == current:
        return 'aba'
    if previous == current:
        return 'abb'
    return 'abc'
""".strip(),
    )


def test_window_signature_contrast_can_select_aba_after_transition_baseline():
    df = _window_contrast_df()
    search_df = df[df["participant_id"] < 52].reset_index(drop=True)
    validation_df = df[df["participant_id"] >= 52].reset_index(drop=True)
    candidate = _task_signature_candidate()
    candidate.round_num = 1
    config = _config()
    config.data_generation.base_factors = [
        BaseFactor(name="task", dtype="categorical", levels=["motion", "color", "orientation"]),
    ]
    config.seed = 19

    result = run_within_round_search(
        search_df=search_df,
        validation_df=validation_df,
        config=config,
        llm=None,
        registry=FactorRegistry(baseline_formula="correct ~ C(task_transition_w2)"),
        round_num=1,
        observable_cols=list(df.columns),
        seeder=StaticSeeder([candidate]),
        evolver=EmptyEvolver(),
    )

    assert result.accepted
    assert result.winner is not None
    assert result.winner.candidate.name == "task_signature_w3__aba_vs_rest"
    assert result.validation_improvement is not None
    assert result.validation_improvement > 0.0
