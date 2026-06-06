import yaml
from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


@dataclass
class LogisticModelConfig:
    intercept: float
    congruent: float
    task_repeat: float
    response_repeat: float = 0.0
    congruency_sequence: Dict[str, float] = field(default_factory=dict)


@dataclass
class DataGenerationConfig:
    n_participants: int
    n_blocks_per_participant: int
    hidden_factors: List[str]
    logistic_model: LogisticModelConfig


@dataclass
class DiscoveryConfig:
    n_rounds: int
    max_candidates_per_round: int   # candidates generated in the first iteration of each round
    max_synthesis_retries: int
    sandbox_timeout_seconds: int
    sandbox_backend: str
    docker_image: str = "python:3.9-slim"
    # iterative within-round search
    candidates_per_refinement: int = 8
    max_search_iterations: int = 3
    refinement_top_k: int = 3
    stability_weight: float = 1.0
    complexity_exponent: float = 1.0   # exponent on n_params=(n_levels−1) in winner score denominator
    depends_on_exponent: float = 0.5  # exponent on n_deps=len(depends_on) in winner score denominator
    cv_n_folds: int = 5
    validation_fraction: float = 0.20
    min_validation_improvement: float = 0.001


@dataclass
class LLMConfig:
    model: str
    max_tokens_candidate: int
    max_tokens_predicate: int
    candidate_temperature: float
    predicate_temperature: float


@dataclass
class StatisticalConfig:
    min_level_count: int


@dataclass
class GroundTruthFactor:
    name: str
    type: str   # "within_trial" | "transition"
    levels: List[str]


@dataclass
class EvaluationConfig:
    ground_truth_factors: List[GroundTruthFactor]
    bijection_threshold: float


@dataclass
class BenchmarkConfig:
    name: str
    seed: int
    output_dir: str
    data_generation: DataGenerationConfig
    discovery: DiscoveryConfig
    llm: LLMConfig
    statistical: StatisticalConfig
    evaluation: EvaluationConfig


def load_config(path: str) -> BenchmarkConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    bm = raw["benchmark"]

    dg = raw["data_generation"]
    lm = dg["logistic_model"]
    data_gen = DataGenerationConfig(
        n_participants=dg["n_participants"],
        n_blocks_per_participant=dg["n_blocks_per_participant"],
        hidden_factors=dg["hidden_factors"],
        logistic_model=LogisticModelConfig(
            intercept=lm["intercept"],
            congruent=lm["congruent"],
            task_repeat=lm["task_repeat"],
            response_repeat=lm.get("response_repeat", 0.0),
            congruency_sequence=lm.get("congruency_sequence") or {},
        ),
    )

    disc = raw["discovery"]
    discovery = DiscoveryConfig(
        n_rounds=disc["n_rounds"],
        max_candidates_per_round=disc["max_candidates_per_round"],
        max_synthesis_retries=disc["max_synthesis_retries"],
        sandbox_timeout_seconds=disc["sandbox_timeout_seconds"],
        sandbox_backend=disc["sandbox_backend"],
        docker_image=disc.get("docker_image", "python:3.9-slim"),
        candidates_per_refinement=disc.get("candidates_per_refinement", 8),
        max_search_iterations=disc.get("max_search_iterations", 3),
        refinement_top_k=disc.get("refinement_top_k", 3),
        stability_weight=disc.get("stability_weight", 1.0),
        complexity_exponent=disc.get("complexity_exponent", 1.0),
        depends_on_exponent=disc.get("depends_on_exponent", 0.5),
        cv_n_folds=disc.get("cv_n_folds", 5),
        validation_fraction=disc.get("validation_fraction", 0.20),
        min_validation_improvement=disc.get("min_validation_improvement", 0.001),
    )

    llm_raw = raw["llm"]
    llm = LLMConfig(
        model=llm_raw["model"],
        max_tokens_candidate=llm_raw["max_tokens_candidate"],
        max_tokens_predicate=llm_raw["max_tokens_predicate"],
        candidate_temperature=llm_raw["candidate_temperature"],
        predicate_temperature=llm_raw["predicate_temperature"],
    )

    stat = raw["statistical"]
    statistical = StatisticalConfig(
        min_level_count=stat["min_level_count"],
    )

    ev = raw["evaluation"]
    evaluation = EvaluationConfig(
        ground_truth_factors=[
            GroundTruthFactor(name=f["name"], type=f["type"], levels=f["levels"])
            for f in ev["ground_truth_factors"]
        ],
        bijection_threshold=ev["bijection_threshold"],
    )

    return BenchmarkConfig(
        name=bm["name"],
        seed=bm["seed"],
        output_dir=bm["output_dir"],
        data_generation=data_gen,
        discovery=discovery,
        llm=llm,
        statistical=statistical,
        evaluation=evaluation,
    )
