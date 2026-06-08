import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Strategy config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SeedingStrategyConfig:
    type: str = "llm"              # "llm" | "random" | "random_lookup" | "mixed"
    n_candidates: int = 5
    seed_multiplier: float = 1.0
    template_bias: str = "uniform"
    max_depends_on: int = 2
    max_output_levels: int = 2
    max_table_size: int = 64
    allow_window: bool = True
    components: List[dict] = field(default_factory=list)


@dataclass
class EvolutionStrategyConfig:
    type: str = "llm"              # "llm" | "llm_genetic" | "mutation" | "mixed"
    n_candidates: int = 5
    top_k: int = 3
    n_elite: int = 1
    n_tournament_parents: int = 3
    n_diversity_parents: int = 1
    operator_mix_mutation: float = 0.40
    operator_mix_crossover: float = 0.30
    operator_mix_repair: float = 0.20
    operator_mix_novel: float = 0.10
    diversity_threshold: float = 0.1
    components: List[dict] = field(default_factory=list)


@dataclass
class ModelTerm:
    factor: str
    coefficient: float
    level: Optional[str] = None  # None → continuous predictor; str → discrete indicator


@dataclass
class LogisticModelConfig:
    intercept: float
    terms: List[ModelTerm] = field(default_factory=list)


@dataclass
class BaseFactor:
    name: str
    dtype: str               # "categorical" | "continuous"
    levels: List[str] = field(default_factory=list)
    include_in_formula: bool = True


@dataclass
class HiddenFactor:
    """
    A factor present in the empirical CSV that is either withheld from the
    discovery pipeline (empirical_benchmark mode) or used as part of the known
    starting model (novel_discovery mode).
    """
    name: str               # factor name used in the model and evaluation
    column: str             # CSV column name (often equals name)
    type: str               # "within_trial" | "transition" | "window"
    levels: List[str]
    factor_class: str = "discrete"


@dataclass
class DataGenerationConfig:
    n_participants: int
    n_blocks_per_participant: int
    outcome_variable: str
    task_context: str
    base_factors: List[BaseFactor]
    hidden_factors: List[str]
    logistic_model: LogisticModelConfig


@dataclass
class DatasetConfig:
    """
    Dataset configuration for real (empirical) data.
    Used by empirical_benchmark and novel_discovery modes.
    """
    path: str
    participant_id_column: str           # column name for participant ID in the CSV
    outcome_variable: str
    task_context: str
    base_factors: List[BaseFactor]       # always visible to the pipeline
    hidden_factors: List[HiddenFactor]   # withheld in empirical_benchmark; empty in novel_discovery
    extra_columns: List[str]             # kept in the dataframe but not added to the model
    n_participants: Optional[int] = None
    n_trials_per_participant: Optional[int] = None
    null_formula: Optional[str] = None  # overrides the auto-built baseline for empirical_benchmark
    full_formula: Optional[str] = None  # overrides the auto-built baseline for novel_discovery


@dataclass
class DiscoveryConfig:
    n_rounds: int
    max_synthesis_retries: int
    sandbox_timeout_seconds: int
    sandbox_backend: str
    docker_image: str = "python:3.9-slim"
    max_search_iterations: int = 3
    stability_weight: float = 1.0
    complexity_exponent: float = 1.0
    depends_on_exponent: float = 0.5
    cv_n_folds: int = 5
    validation_fraction: float = 0.20
    min_validation_improvement: float = 0.001
    decomposition_check_enabled: bool = True
    decomposition_check_max_arity: int = 2
    run_effect_search: bool = True
    max_interaction_order: int = 2
    max_interactions_per_round: int = 1
    effect_search_full_pass: bool = False
    effect_search_full_pass_interval: Optional[int] = None
    effect_search_min_cv_improvement: float = 0.05
    effect_search_min_validation_improvement: float = 0.001
    llm_rank_interactions: bool = False
    max_tokens_interaction_ranking: int = 1000
    allowed_factor_types: List[str] = field(default_factory=lambda: ["within_trial", "window"])
    allowed_factor_classes: List[str] = field(default_factory=lambda: ["discrete"])
    max_window_width: int = 5
    stagnation_epsilon: float = 0.001
    stagnation_patience: int = 0
    seeding_strategy: SeedingStrategyConfig = field(default_factory=SeedingStrategyConfig)
    evolution_strategy: EvolutionStrategyConfig = field(default_factory=EvolutionStrategyConfig)


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
    type: str
    levels: List[str]
    factor_class: str = "discrete"


@dataclass
class GroundTruthInteraction:
    factors: List[str]


@dataclass
class EvaluationConfig:
    ground_truth_factors: List[GroundTruthFactor]
    bijection_threshold: float
    ground_truth_interactions: List[GroundTruthInteraction] = field(default_factory=list)
    continuous_correlation_threshold: float = 0.7


@dataclass
class BenchmarkConfig:
    name: str
    benchmark_type: str
    seed: int
    output_dir: str
    discovery: DiscoveryConfig
    llm: LLMConfig
    statistical: StatisticalConfig
    evaluation: EvaluationConfig
    mode: str = "synthetic_benchmark"   # "synthetic_benchmark" | "empirical_benchmark" | "novel_discovery"
    data_generation: Optional[DataGenerationConfig] = None  # synthetic_benchmark only
    dataset: Optional[DatasetConfig] = None                 # empirical_benchmark / novel_discovery

    # --- Convenience properties that resolve uniformly across all modes ---

    @property
    def base_factors(self) -> List[BaseFactor]:
        src = self.dataset if self.dataset is not None else self.data_generation
        return src.base_factors if src is not None else []

    @property
    def outcome_variable(self) -> str:
        src = self.dataset if self.dataset is not None else self.data_generation
        if src is None:
            raise ValueError("BenchmarkConfig has neither dataset nor data_generation.")
        return src.outcome_variable

    @property
    def task_context(self) -> str:
        src = self.dataset if self.dataset is not None else self.data_generation
        return src.task_context if src is not None else ""


@dataclass
class RunConfig:
    benchmarks: List[str]
    shared: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge two dicts; values in *override* take precedence."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _parse_logistic_model(lm: dict) -> LogisticModelConfig:
    terms = []
    for t in lm.get("terms", []):
        terms.append(ModelTerm(
            factor=t["factor"],
            coefficient=float(t["coefficient"]),
            level=t.get("level", None),
        ))
    return LogisticModelConfig(intercept=float(lm["intercept"]), terms=terms)


def _parse_base_factors(raw: list) -> List[BaseFactor]:
    result = []
    for f in raw:
        result.append(BaseFactor(
            name=f["name"],
            dtype=f.get("dtype", "categorical"),
            levels=list(f.get("levels", [])),
            include_in_formula=bool(f.get("include_in_formula", True)),
        ))
    return result


def _parse_seeding_strategy_config(raw: dict) -> SeedingStrategyConfig:
    return SeedingStrategyConfig(
        type=raw.get("type", "llm"),
        n_candidates=int(raw.get("n_candidates", 5)),
        seed_multiplier=float(raw.get("seed_multiplier", 1.0)),
        template_bias=raw.get("template_bias", "uniform"),
        max_depends_on=int(raw.get("max_depends_on", 2)),
        max_output_levels=int(raw.get("max_output_levels", 2)),
        max_table_size=int(raw.get("max_table_size", 64)),
        allow_window=bool(raw.get("allow_window", True)),
        components=list(raw.get("components", [])),
    )


def _parse_evolution_strategy_config(raw: dict) -> EvolutionStrategyConfig:
    om = raw.get("operator_mix", {})
    return EvolutionStrategyConfig(
        type=raw.get("type", "llm"),
        n_candidates=int(raw.get("n_candidates", 5)),
        top_k=int(raw.get("top_k", 3)),
        n_elite=int(raw.get("n_elite", 1)),
        n_tournament_parents=int(raw.get("n_tournament_parents", 3)),
        n_diversity_parents=int(raw.get("n_diversity_parents", 1)),
        operator_mix_mutation=float(om.get("mutation", raw.get("operator_mix_mutation", 0.40))),
        operator_mix_crossover=float(om.get("crossover", raw.get("operator_mix_crossover", 0.30))),
        operator_mix_repair=float(om.get("repair", raw.get("operator_mix_repair", 0.20))),
        operator_mix_novel=float(om.get("novel", raw.get("operator_mix_novel", 0.10))),
        diversity_threshold=float(raw.get("diversity_threshold", 0.1)),
        components=list(raw.get("components", [])),
    )


def _parse_dataset_config(raw: dict) -> DatasetConfig:
    hidden: List[HiddenFactor] = []
    for hf in raw.get("hidden_factors", []):
        if isinstance(hf, str):
            # Legacy shorthand: bare string column name
            hidden.append(HiddenFactor(name=hf, column=hf, type="within_trial", levels=[]))
        else:
            hidden.append(HiddenFactor(
                name=hf["name"],
                column=hf.get("column", hf["name"]),
                type=hf.get("type", "within_trial"),
                levels=list(hf.get("levels", [])),
                factor_class=hf.get("factor_class", "discrete"),
            ))
    return DatasetConfig(
        path=raw["path"],
        participant_id_column=raw.get("participant_id_column", "participant_id"),
        outcome_variable=raw["outcome_variable"],
        task_context=raw.get("task_context", ""),
        base_factors=_parse_base_factors(raw.get("base_factors", [])),
        hidden_factors=hidden,
        extra_columns=list(raw.get("extra_columns", [])),
        n_participants=raw.get("n_participants"),
        n_trials_per_participant=raw.get("n_trials_per_participant"),
        null_formula=raw.get("null_formula"),
        full_formula=raw.get("full_formula"),
    )


def load_config(path: str, defaults: dict = None) -> BenchmarkConfig:
    """
    Load a benchmark config YAML.

    If *defaults* is provided (e.g. shared params from benchmark.yaml), they are
    deep-merged with the file contents, with the file taking precedence.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if defaults:
        raw = _deep_merge(defaults, raw)
    return _parse_benchmark_dict(raw)


def _parse_benchmark_dict(raw: dict) -> BenchmarkConfig:
    bm      = raw["benchmark"]
    disc    = raw.get("discovery", {})
    llm_raw = raw.get("llm", {})
    stat    = raw.get("statistical", {})
    ev      = raw.get("evaluation", {})

    mode = bm.get("mode", "synthetic_benchmark")

    # --- Parse the data source based on mode ---
    data_gen: Optional[DataGenerationConfig] = None
    dataset_cfg: Optional[DatasetConfig] = None

    if mode == "synthetic_benchmark":
        dg = raw["data_generation"]
        data_gen = DataGenerationConfig(
            n_participants=dg["n_participants"],
            n_blocks_per_participant=dg["n_blocks_per_participant"],
            outcome_variable=dg.get("outcome_variable", "correct"),
            task_context=dg.get("task_context", ""),
            base_factors=_parse_base_factors(dg.get("base_factors", [])),
            hidden_factors=dg.get("hidden_factors", []),
            logistic_model=_parse_logistic_model(dg["logistic_model"]),
        )
    else:
        dataset_cfg = _parse_dataset_config(raw["dataset"])

    # Auto-build ground_truth_factors for empirical_benchmark from hidden_factors
    if mode == "empirical_benchmark" and dataset_cfg and not ev.get("ground_truth_factors"):
        ev = dict(ev)  # copy to avoid mutating caller's dict
        ev["ground_truth_factors"] = [
            {
                "name": hf.name,
                "type": hf.type,
                "levels": hf.levels,
                "factor_class": hf.factor_class,
            }
            for hf in dataset_cfg.hidden_factors
        ]

    # --- Discovery ---
    discovery = DiscoveryConfig(
        n_rounds=disc.get("n_rounds", 2),
        max_synthesis_retries=disc.get("max_synthesis_retries", 2),
        sandbox_timeout_seconds=disc.get("sandbox_timeout_seconds", 10),
        sandbox_backend=disc.get("sandbox_backend", "subprocess"),
        docker_image=disc.get("docker_image", "python:3.9-slim"),
        max_search_iterations=disc.get("max_search_iterations", 3),
        stability_weight=disc.get("stability_weight", 1.0),
        complexity_exponent=disc.get("complexity_exponent", 1.0),
        depends_on_exponent=disc.get("depends_on_exponent", 0.5),
        cv_n_folds=disc.get("cv_n_folds", 5),
        validation_fraction=disc.get("validation_fraction", 0.20),
        min_validation_improvement=disc.get("min_validation_improvement", 0.001),
        decomposition_check_enabled=disc.get("decomposition_check_enabled", True),
        decomposition_check_max_arity=disc.get("decomposition_check_max_arity", 2),
        run_effect_search=disc.get("run_effect_search", True),
        max_interaction_order=disc.get("max_interaction_order", 2),
        max_interactions_per_round=disc.get("max_interactions_per_round", 1),
        effect_search_full_pass=disc.get("effect_search_full_pass", False),
        effect_search_full_pass_interval=disc.get("effect_search_full_pass_interval", None),
        effect_search_min_cv_improvement=disc.get("effect_search_min_cv_improvement", 0.05),
        effect_search_min_validation_improvement=disc.get("effect_search_min_validation_improvement", 0.001),
        llm_rank_interactions=disc.get("llm_rank_interactions", False),
        max_tokens_interaction_ranking=disc.get("max_tokens_interaction_ranking", 1000),
        allowed_factor_types=disc.get("allowed_factor_types", ["within_trial", "window"]),
        allowed_factor_classes=disc.get("allowed_factor_classes", ["discrete"]),
        max_window_width=disc.get("max_window_width", 5),
        stagnation_epsilon=disc.get("stagnation_epsilon", 0.001),
        stagnation_patience=disc.get("stagnation_patience", 0),
        seeding_strategy=_parse_seeding_strategy_config(disc.get("seeding_strategy", {})),
        evolution_strategy=_parse_evolution_strategy_config(disc.get("evolution_strategy", {})),
    )

    llm = LLMConfig(
        model=llm_raw.get("model", "claude-sonnet-4-6"),
        max_tokens_candidate=llm_raw.get("max_tokens_candidate", 2000),
        max_tokens_predicate=llm_raw.get("max_tokens_predicate", 1000),
        candidate_temperature=llm_raw.get("candidate_temperature", 0.9),
        predicate_temperature=llm_raw.get("predicate_temperature", 0.2),
    )

    statistical = StatisticalConfig(min_level_count=stat.get("min_level_count", 5))

    evaluation = EvaluationConfig(
        ground_truth_factors=[
            GroundTruthFactor(
                name=f["name"],
                type=f["type"],
                levels=f["levels"],
                factor_class=f.get("factor_class", "discrete"),
            )
            for f in ev.get("ground_truth_factors", [])
        ],
        bijection_threshold=ev.get("bijection_threshold", 0.95),
        ground_truth_interactions=[
            GroundTruthInteraction(factors=sorted(i["factors"]))
            for i in ev.get("ground_truth_interactions", [])
        ],
        continuous_correlation_threshold=ev.get("continuous_correlation_threshold", 0.7),
    )

    return BenchmarkConfig(
        name=bm["name"],
        benchmark_type=bm.get("type", "stroop"),
        mode=mode,
        seed=bm.get("seed", 42),
        output_dir=bm.get("output_dir", "results"),
        discovery=discovery,
        llm=llm,
        statistical=statistical,
        evaluation=evaluation,
        data_generation=data_gen,
        dataset=dataset_cfg,
    )


def load_run_config(path: str) -> RunConfig:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    benchmarks = raw.get("benchmarks", [])
    shared = {k: v for k, v in raw.items() if k != "benchmarks"}
    return RunConfig(benchmarks=benchmarks, shared=shared)
