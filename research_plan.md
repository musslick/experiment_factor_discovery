# Research Plan: Automated Discovery of Experimental Variables via LLM-Driven Program Synthesis

## 1. Research Problem and Motivation

A central challenge in empirical science is the discovery of new experimental factors that explain observed behavior. Researchers often rely on domain expertise and intuition to propose derived variables — combinations of basic experimental factors that capture theoretically meaningful distinctions. For example, in the Stroop task (Stroop, 1935), the canonical experimental factors are the word stimulus, its ink color, and the task instruction (color naming vs. word reading). However, a rich literature has emerged by examining *derived* factors computed from these basics:

- **Congruency**: whether the word meaning matches its ink color (MacLeod, 1991)
- **Congruency sequence**: whether congruency repeats or changes across trials (Gratton et al., 1992)
- **Task transition**: whether the current task repeats or switches (Monsell, 2003)
- **Response transition**: whether the required motor response repeats or switches (Pashler & Baynes, 2001)

Each of these discoveries required a researcher to hypothesize a new derived factor, formalize it, compute it from existing data, and demonstrate its statistical relevance. This process is currently manual, slow, and dependent on domain expertise. The goal of this project is to automate it.

**Research Question**: Can a computational pipeline, driven by a large language model (LLM) and formal program synthesis, systematically discover derived experimental factors from behavioral data — recovering factors that are known to matter for behavior but are hidden from the pipeline?

---

## 2. Core Idea

We reduce the problem of discovering new derived experimental factors to **predicate function synthesis**: a derived factor is fully specified by a function that maps one or more existing factor values (within a trial, or across consecutive trials) to a discrete level label or continuous value. This representation directly maps onto the SweetPea declarative experiment design language (Musslick et al., 2020), which distinguishes:

- **Within-trial derived factors**: `DerivedLevel(label, WithinTrial(predicate, [factor1, factor2, ...]))`
- **Window derived factors**: `DerivedLevel(label, Window(predicate, [factor], window_size, stride))`

The synthesis problem becomes: find predicate functions that, when applied to an existing trial sequence, produce a factor that significantly improves cross-validated prediction of the behavioral outcome.

Factors can be:
- **Discrete**: the function returns a string level label (e.g., `"congruent"`, `"repeat"`)
- **Continuous**: the function returns a float (e.g., expected value difference, stimulus difficulty)

Within-trial factors depend only on the current trial's fields; window factors depend on a sliding window of recent trials and can capture sequential effects.

---

## 3. Approach Overview

The pipeline operates as follows:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Discovery Pipeline                           │
│                                                                      │
│  Observable data: base factors + outcome column(s)                   │
│        │                                                             │
│        ▼  (once, before round 1)                                     │
│  ┌─────────────────────┐                                             │
│  │  Participant split  │  80% search set / 20% held-out             │
│  │  + null formula     │  validation set                            │
│  └──────────┬──────────┘                                             │
│             │                                                        │
│             ▼  ─────────────── repeated for N rounds ───────────    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  PHASE 1: Factor Discovery (within-round iterative search)   │   │
│  │                                                              │   │
│  │  ┌─────────────────────────────────────────────┐            │   │
│  │  │  Seeding (one of: random / LLM / mixed)     │            │   │
│  │  │  → N candidate factor descriptions          │            │   │
│  │  └──────────────────┬──────────────────────────┘            │   │
│  │                     │  for each candidate:                  │   │
│  │                     ▼                                       │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  Predicate synthesis (LLM → compute_factor() code)  │   │   │
│  │  │  + sandboxed execution + factor encoding            │   │   │
│  │  │  + hard-rejection (bijection/coarsening check)      │   │   │
│  │  └──────────────────┬───────────────────────────────────┘  │   │
│  │                     ▼                                       │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │  CV scoring on search set (participant-wise 5-fold)   │  │   │
│  │  │  score = (mean − λ·SE) / complexity + novelty_weight  │  │   │
│  │  └──────────────────┬────────────────────────────────────┘  │   │
│  │                     │                                       │   │
│  │                     ↺  Evolution (LLM genetic / mutation)   │   │
│  │                     │  → refined candidates                 │   │
│  │                     │  (repeat up to max_search_iterations) │   │
│  │                     ▼                                       │   │
│  │  ┌───────────────────────────────────────────────────────┐  │   │
│  │  │  Validation on held-out set                           │  │   │
│  │  │  mean LL gain ≥ threshold → register factor           │  │   │
│  │  └───────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  PHASE 2: Effect Search (optional)                           │   │
│  │  CV-score and validate pairwise interaction terms            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│             │  registered factors → updated null formula            │
│             │  + context for next round's seeding                   │
│             ▼                                                        │
│  Evaluation: bijection matching vs. ground-truth factors             │
│  Precision / Recall / F1                                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Benchmark Design

### 4.1 Synthetic Benchmarks

Four synthetic benchmarks provide controlled discovery challenges with known ground-truth factors. All are generated via SweetPea, which produces full trial sequences including hidden derived-factor columns. The hidden columns are stripped to create the pipeline input; the full dataset is retained for evaluation.

#### Benchmark 1: Stroop Task (2 factors)

- **Observable factors**: `task` (color_naming/word_reading), `color` (red/blue/green), `word` (red/blue/green)
- **Hidden derived factors**:
  - `congruency` (within-trial, discrete): color == word → congruent/incongruent
  - `task_transition` (window, width=2, discrete): repeat/switch
- **Ground-truth logistic model**: intercept 0.5; congruency_congruent +0.8; task_transition_repeat +0.4
- **Data**: 100 participants × 11 blocks × 18 trials = 19,800 rows; outcome: accuracy

#### Benchmark 2: Stroop-Simon Task (4 factors)

- **Observable factors**: `word`, `color`, `stimulus_location` (left/middle/right)
- **Hidden derived factors**:
  - `word_color_congruency` (within-trial, discrete): 2 levels
  - `location_response_congruency` (within-trial, discrete): 2 levels
  - `congruency_previous_trial` (window, width=2, discrete): 2 levels
  - `response_transition` (window, width=2, discrete): repeat/switch
- **Ground-truth model**: coefficients 0.8, 0.6, 0.3, 0.2
- **Data**: 100 participants × 8 blocks × 27 trials = 21,600 rows; outcome: accuracy

#### Benchmark 3: RDK Task-Switching (4 factors, mixed types)

- **Observable factors**: `task` (motion/color/orientation), three stimulus dimensions (motion, color, orientation), three continuous coherence values (0–1), `correct_response`
- **Hidden derived factors**:
  - `task_transition` (window, width=2, discrete): repeat/switch
  - `current_stimulus_difficulty` (within-trial, continuous): 1 − coherence of current task's dimension
  - `past_stimulus_difficulty` (window, width=2, continuous): difficulty of previous trial
  - `n2_task_inhibition` (window, width=3, discrete): aba_return/cba_nonreturn/other
- **Ground-truth model**: z-scored continuous factors; coefficients −0.8, −0.2 (difficulties), −0.4 (n2 return), +0.5 (task repeat)
- **Data**: 100 participants × 8 blocks × 24 trials = 19,200 rows; outcome: accuracy

#### Benchmark 4: Prospect Theory Risky Choice (9 factors)

- **Observable factors**: 6 continuous gamble parameters (left/right gain, loss, probability)
- **Hidden derived factors** (9):
  - `left_expected_value`, `right_expected_value`, `expected_value_difference` (within-trial, continuous)
  - `gain_difference`, `loss_difference`, `probability_difference` (within-trial, continuous)
  - `dominance_relation` (within-trial, discrete): left_dominates/right_dominates/no_dominance
  - `previous_expected_value_difference`, `value_difference_transition` (window, width=2)
- **Ground-truth model**: 6-term logistic model with mixed continuous and discrete predictors
- **Data**: 100 participants × 200 trials = 20,000 rows; outcome: binary choice

### 4.2 Data Generation

```python
# Observable (regular) factors
task  = Factor("task",  ["color_naming", "word_reading"])
color = Factor("color", ["red", "blue", "green"])
word  = Factor("word",  ["red", "blue", "green"])

# Hidden derived factor: congruency (within-trial)
congruency = Factor("congruency", [
    DerivedLevel("congruent",   WithinTrial(lambda color, word: color == word,   [color, word])),
    DerivedLevel("incongruent", WithinTrial(lambda color, word: color != word,   [color, word])),
])

# Hidden derived factor: task_transition (window, width=2)
task_transition = Factor("task_transition", [
    DerivedLevel("repeat", Window(lambda w: w[0] == w[-1], [task], 2, 1)),
    DerivedLevel("switch", Window(lambda w: w[0] != w[-1], [task], 2, 1)),
])

# Crossing only over observable factors; derived factors ride along
block = CrossBlock(
    design   = [task, color, word, congruency, task_transition],
    crossing = [task, color, word],
    constraints = [],
)
experiments = synthesize_trials(block, samples=n_participants * n_blocks_per_participant,
                                sampling_strategy=RandomGen)
```

SweetPea outputs a dataset containing all factor columns. The full dataset is saved to `data/ground_truth/<name>_full.csv`; hidden columns are dropped to produce `data/input/<name>_input.csv`.

### 4.3 Empirical Benchmark

**Stroop Congruency (empirical)**: N = 466 participants, outcomes = log(latency) + accuracy.
- **Observable base factors**: `color` (red/blue/green/black), `word` (red/blue/green/black)
- **Known hidden factor**: `congruency` (within-trial, discrete)
- **Pipeline mode**: `empirical_benchmark` — the known factor column is withheld from the pipeline and used as ground truth for F1 evaluation
- **Multi-outcome**: joint improvement required across latency (continuous) and accuracy (binary)

---

## 5. Discovery Pipeline: Technical Design

### 5.1 Candidate Generation (Seeding)

Each round begins with seeding: generating an initial pool of candidate factor descriptions. Three strategies are available:

**Random seeder** (`type: random`): Combinatorial enumeration from templates. Samples factor type (within-trial / window), factor class (discrete / continuous), n_levels, and depends_on set from the observable factors. Bias can be `uniform` or `complexity` (upweights multi-level and window candidates).

**LLM seeder** (`type: llm`): A single LLM call proposes candidates as a JSON array. The system prompt describes both factor types, the predicate language, and constraints (no renamings of known factors, no re-proposals of previously rejected candidates). The user prompt provides the task description, observable factors, already-discovered factors, and CV feedback from prior rounds.

Output format (same for both strategies):
```json
[
  {
    "name": "congruency",
    "description": "Whether the ink color matches the word meaning within a trial",
    "factor_type": "within_trial",
    "factor_class": "discrete",
    "levels": ["congruent", "incongruent"],
    "depends_on": ["color", "word"]
  }
]
```

**Mixed seeder** (`type: mixed`): Interleaves random and LLM proposals within the same round.

### 5.2 Predicate Synthesis

For each candidate description, the LLM synthesizes a `compute_factor` Python function. The function signature depends on factor type and class:

```python
# Within-trial discrete: receives the current trial as a dict
def compute_factor(trial: dict) -> str:
    if trial["color"] == trial["word"]:
        return "congruent"
    return "incongruent"

# Window discrete: receives a list of dicts (oldest first, current last)
def compute_factor(window: list) -> str:
    if window[-2]["task"] == window[-1]["task"]:
        return "repeat"
    return "switch"

# Within-trial continuous
def compute_factor(trial: dict) -> float:
    return 1.0 - trial["motion_coherence"]

# Window continuous
def compute_factor(window: list) -> float:
    prev_coherence = window[-2].get("motion_coherence", 0.5)
    return 1.0 - prev_coherence
```

The LLM also produces a SweetPea `Factor` definition for each candidate, which is saved alongside the compute function for archival and experimental reuse.

**Self-correction loop**: Sandbox execution errors (syntax error, runtime error, wrong return type, invalid level name) are appended to the next LLM call. Up to `max_synthesis_retries` (default: 2) retries are allowed per candidate.

Allowed imports in synthesized code: `json`, `math`, `itertools`, `functools`, `re`, `collections`.

### 5.3 Sandboxed Code Execution

The synthesized `compute_factor` function is executed against the existing trial data in an isolated subprocess. No SweetPea re-synthesis is needed at discovery time.

**Execution harness**:
1. Serialize the DataFrame to JSON (trial dicts, grouped by participant and sorted by `trial_index`).
2. Apply `compute_factor` to every trial (within-trial) or to every sliding window (window factors), returning `None` for the first `window_width − 1` trials of each block.
3. Validate that all non-`None` values match the declared levels (discrete) or are finite floats (continuous).
4. Timeout: 10–30s (configurable; use larger values for big empirical datasets).

Two backends:
- **Subprocess** (default): Python subprocess with restricted imports. Adequate for research use.
- **Docker** (optional): `llm-sandbox` Docker container for stronger isolation.

### 5.4 Candidate Scoring

Valid candidates are scored by participant-wise cross-validated log-likelihood improvement on the search set.

**Formula extension**: The current null formula (which grows with each round) is extended with the candidate factor:
```
# Round 1 null:  "accuracy ~ C(task) + C(color) + C(word)"
# After adding congruency:
#               "accuracy ~ C(task) + C(color) + C(word) + C(congruency)"
```
Discrete factors are encoded with `C(name)` (treatment contrasts); continuous factors are added as raw numeric columns.

**Cross-validation** (`cv_n_folds = 5`, participant-wise splits on the search set):
```
For each fold:
    Fit null model on training participants
    Fit alternative model on training participants
    Compute per-trial log-likelihood improvement on held-out participants
    Average across held-out trials → per-participant LL improvement
Result: CVScore(mean_ll_improvement, std_error, n_participants)
```

**Novelty score**: Measures how different the candidate is from all already-discovered factors.
- Discrete: `novelty = 1 − max NMI(candidate, known_factor)` over all known factors
- Continuous: `novelty = 1 − max |Spearman ρ(candidate, known_factor)|`

**Adjusted score** (used for winner selection):
```
adjusted_score = (mean_ll_improvement − λ · std_error)
                 / (n_params^α · n_depends_on^β)
                 + novelty_weight · novelty_score
```
where `λ` (`stability_weight`) penalises noisy candidates, `α` (`complexity_exponent`) penalises many parameters, `β` (`depends_on_exponent`) penalises many input dependencies, and `novelty_weight` encourages representational diversity.

**Hard rejection**: Before scoring, candidates are checked for decomposition — if a candidate is a bijection or coarsening of an already-discovered factor it is rejected without scoring.

**Validation**: The round winner is tested once on the fixed held-out 20% validation set. It is accepted if:
```
mean per-participant LL gain on held-out set ≥ min_validation_improvement
```
Only accepted factors are registered and added to the null formula for subsequent rounds.

### 5.5 Iterative Refinement (Genetic Evolution)

After each scoring iteration, the evolution strategy proposes a new candidate batch derived from the current top-k parents. Two strategies are available:

**LLM genetic evolver** (`type: llm_genetic`): A single LLM call applies one of four operators to the top-k parents:
- **Mutation**: Modify one aspect of a parent (change a level, add/remove a dependency, adjust the window width)
- **Crossover**: Combine aspects of two parents into a new candidate
- **Repair**: Fix a parent that failed synthesis (given the error message)
- **Novel**: Propose an entirely new candidate inspired by the parents but distinct from them

The operator mix is configurable (`operator_mix`); default is mutation-only. A diversity threshold prevents re-proposing candidates too similar to the current pool.

**Mutation evolver** (`type: mutation`): Random perturbations of parent candidates (no LLM call).

The generate → score → evolve cycle repeats for up to `max_search_iterations` iterations per round. **Stagnation detection**: if the best score does not improve by more than `epsilon` for `patience` consecutive iterations, the within-round search terminates early.

### 5.6 Interaction / Effect Search (Phase 2)

After each round's main factor is registered, the pipeline optionally searches for interaction terms:

1. All pairwise products of discovered factor columns are enumerated.
2. Each interaction is CV-scored on the search set.
3. Interactions above `effect_search_min_cv_improvement` are validated on the held-out set.
4. Accepted interactions (up to `max_interactions_per_round` per round) are added to the null formula.
5. If `llm_rank_interactions` is enabled, an LLM call ranks candidates by plausibility before testing.

### 5.7 Multi-Outcome Support

When a dataset specifies multiple `outcome_variables` (e.g., accuracy + log-latency), cross-validated scoring is computed per outcome. A candidate is accepted only if it achieves the minimum validation improvement threshold on **every** specified outcome. Scores are ranked by mean improvement across outcomes.

---

## 6. Evaluation

### 6.1 Factor Matching

A discovered factor D **matches** ground-truth factor G if:
- **Discrete–discrete**: There exists a bijection φ between their level sets such that, for ≥ 95% of applicable (non-NaN) trials, `φ(D(t)) = G(t)`.
- **Continuous–continuous**: `|Spearman ρ(D, G)| ≥ 0.70` over applicable trials.

The matching algorithm:
1. Compute an agreement matrix: `agreement[i, j]` = bijection agreement rate (discrete) or |Spearman ρ| (continuous) between GT factor `i` and discovered factor `j`.
2. Find maximum-weight assignment via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
3. Accept matched pairs above the respective threshold.

### 6.2 Metrics

| Metric | Definition |
|--------|------------|
| **Precision** | \|accepted matches\| / \|discovered factors\| |
| **Recall** | \|accepted matches\| / \|ground-truth factors\| |
| **F1** | 2 × Precision × Recall / (Precision + Recall) |

Metrics are computed separately for main-effect factors and interaction effects. Additional per-factor diagnostics are recorded: CV score mean and SE, validation improvement, round discovered, synthesis retries, novelty score.

---

## 7. Project Structure

```
experimental_design_search/
├── config/
│   ├── benchmark.yaml                      # Multi-benchmark runner config
│   ├── discovery.yaml                      # Multi-dataset discovery runner config
│   ├── synthetic_stroop_benchmark.yaml
│   ├── synthetic_stroop_simon_benchmark.yaml
│   ├── synthetic_rdk_task_switching_benchmark.yaml
│   ├── synthetic_prospect_theory_benchmark.yaml
│   └── empirical_stroop_congruency.yaml
├── data/
│   ├── ground_truth/                       # Full datasets with hidden factor columns
│   ├── input/                              # Observable-only inputs (pipeline input)
│   └── empirical/                          # Real behavioral datasets
├── benchmarks/
│   ├── stroop.md
│   ├── stroop_simon.md
│   ├── rdk_task_switching.md
│   └── prospect_theory.md
├── src/
│   ├── analysis/
│   │   ├── evaluation.py                   # Bijection matching + P/R/F1
│   │   ├── factor_encoder.py               # Predicate output → pandas Series
│   │   ├── model_comparison.py             # CV scoring, formula building, multi-outcome
│   │   └── plotting.py                     # Effect plots for discovered factors
│   ├── data_generation/
│   │   ├── base.py                         # Logistic model + outcome sampling
│   │   ├── empirical_loader.py             # Load empirical CSV datasets
│   │   ├── sweetpea_builder.py             # SweetPea trial sequence synthesis
│   │   ├── stroop_model.py                 # Stroop task builder
│   │   ├── stroop_simon_builder.py         # Stroop-Simon task builder
│   │   ├── rdk_builder.py                  # RDK task-switching builder
│   │   └── prospect_theory_builder.py      # Prospect theory builder
│   ├── discovery/
│   │   ├── pipeline.py                     # Multi-round discovery orchestration
│   │   ├── within_round_search.py          # Iterative seeding → scoring → evolution loop
│   │   ├── effect_searcher.py              # Interaction term discovery (Phase 2)
│   │   ├── llm_client.py                   # Anthropic SDK wrapper with retry
│   │   ├── candidate_generator.py          # Candidate description generation
│   │   ├── predicate_synthesizer.py        # LLM → compute_factor() + self-correction
│   │   ├── sandbox.py                      # Subprocess/Docker sandboxed execution
│   │   ├── factor_registry.py              # Factor registry + formula management
│   │   └── strategies/
│   │       ├── base.py                     # SeedingStrategy + EvolutionStrategy ABCs
│   │       ├── random_seeder.py            # Combinatorial random seeder
│   │       ├── llm_seeder.py               # LLM-based seeder
│   │       ├── mixed.py                    # Mixed seeder
│   │       ├── mutation_evolver.py         # Random mutation evolver
│   │       ├── llm_evolver.py              # LLM-guided evolver
│   │       └── llm_genetic_evolver.py      # LLM genetic operator evolver
│   └── utils/
│       └── config.py                       # YAML loader + typed dataclasses
├── prompts/
│   ├── candidate_generation_system.txt
│   ├── candidate_generation_user.txt
│   ├── candidate_refinement_system.txt
│   ├── candidate_refinement_user.txt
│   ├── genetic_evolution_system.txt
│   ├── genetic_evolution_user.txt
│   ├── predicate_synthesis_system.txt
│   ├── predicate_synthesis_user.txt
│   ├── effect_ranking_system.txt
│   ├── effect_ranking_user.txt
│   ├── effect_naming_system.txt
│   └── effect_naming_user.txt
├── results/
│   ├── run_<timestamp>/                    # Benchmark run outputs
│   └── discovery_<timestamp>/             # Discovery run outputs
├── tests/
│   ├── test_data_generation.py
│   ├── test_model_comparison.py
│   ├── test_sandbox.py
│   ├── test_llm_integration.py
│   ├── test_evaluation.py
│   ├── test_benchmark_data.py
│   ├── test_empirical_config.py
│   ├── test_empirical_pipeline.py
│   └── test_novelty.py
├── generate_data.py                        # Data generation CLI
├── run_benchmark.py                        # Benchmarking mode CLI
├── run_discovery.py                        # Novel discovery mode CLI
├── run_llm_test.py                         # LLM integration smoke test
├── diagnose_synthesis.py                   # Predicate synthesis debugger
├── requirements.txt
├── research_plan.md                        # This document
└── paper_plan.md                           # Conference paper plan
```

---

## 8. Configuration

All parameters are specified in YAML config files. Individual benchmark configs can be run directly or aggregated into a top-level `benchmark.yaml` / `discovery.yaml` for batch runs.

Key configuration sections and parameters:

```yaml
benchmark:
  name: "stroop_benchmark"
  mode: "synthetic_benchmark"      # synthetic_benchmark | empirical_benchmark | novel_discovery
  seed: 42
  output_dir: "results"

dataset:                           # for empirical modes
  path: "data/empirical/my_data.csv"
  participant_id_column: "subject_id"
  outcome_variables:
    - {name: accuracy, type: binary}
    - {name: latency,  type: continuous}   # multi-outcome support
  task_context: |
    Describe the task for the LLM.
  base_factors:
    - {name: color, dtype: categorical, levels: [red, blue, green]}
  hidden_factors:
    - {name: congruency, type: within_trial, levels: [congruent, incongruent]}

data_generation:                   # for synthetic modes
  n_participants: 100
  n_blocks_per_participant: 11
  hidden_factors: [congruency, task_transition]
  logistic_model:
    intercept: 0.5
    congruent: 0.8
    task_repeat: 0.4

discovery:
  n_rounds: 5
  max_search_iterations: 3
  max_synthesis_retries: 2
  sandbox_timeout_seconds: 10      # increase for large datasets
  sandbox_backend: "subprocess"    # subprocess | docker
  cv_n_folds: 5
  validation_fraction: 0.20
  min_validation_improvement: 0.001
  # Scoring weights
  stability_weight: 1.0            # λ: penalise noisy candidates
  complexity_exponent: 0.0         # α: penalise many parameters
  depends_on_exponent: 0.0         # β: penalise many dependencies
  novelty_weight: 0.0              # bonus for representational diversity
  # Seeding strategy
  seeding_strategy:
    type: "random"                 # random | llm | mixed
    n_candidates: 50
    template_bias: "uniform"       # uniform | complexity
    allow_window: true
    max_depends_on: 3
    max_output_levels: 3
  # Evolution strategy
  evolution_strategy:
    type: "llm_genetic"            # llm_genetic | llm | mutation | mixed
    n_candidates: 5
    top_k: 3
    operator_mix:
      mutation: 1.0
      crossover: 0.0
      repair: 0.0
      novel: 0.0
  # Effect search (Phase 2)
  run_effect_search: true
  max_interaction_order: 2
  max_interactions_per_round: 1
  effect_search_min_cv_improvement: 0.05
  effect_search_min_validation_improvement: 0.001

llm:
  model: "claude-sonnet-4-6"
  candidate_temperature: 0.9
  predicate_temperature: 0.2

statistical:
  min_level_count: 5              # guards against near-constant factors
```

---

## 9. Software Dependencies

```
sweetpea>=0.35.0        # Experiment design DSL
anthropic>=0.20.0       # Claude API client
statsmodels>=0.14.0     # Logistic/linear regression + LRT
scipy>=1.11.0           # Hungarian algorithm, Spearman correlation
scikit-learn>=1.3.0     # Normalized mutual information
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0       # Effect plots
pyyaml>=6.0
patsy>=0.5.6            # Formula parsing/encoding
pytest>=7.0.0
python-dotenv>=1.0.0
```

**Environment setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 10. Expected Results and Evaluation Criteria

**Success criteria per benchmark complexity:**

| Benchmark | Expected F1 | Difficulty Drivers |
|---|---|---|
| Stroop (2 factors, discrete) | ~1.0 | Minimal — canonical Stroop factors are well-known to LLMs |
| Stroop-Simon (4 factors, discrete) | ~0.8–1.0 | Sequential factors require window synthesis |
| RDK Task-Switching (4 factors, mixed) | ~0.6–0.9 | Continuous factors harder; n2 inhibition non-obvious |
| Prospect Theory (9 factors, continuous) | ~0.4–0.7 | Large factor space; many correlated EV variants |

**Factor type difficulty ordering (predicted):**
Within-trial discrete > Window discrete > Within-trial continuous > Window continuous

**Partial success indicators:**
- Correct factor type (within-trial vs. window) for each discovered factor
- Agreement rate ≥ 0.95 on discrete bijection matching
- |Spearman ρ| ≥ 0.70 for continuous factor matching
- LRT p-values < 0.001 for all registered factors

**Failure modes to analyze:**
- Synthesis failures: syntax errors, wrong return type, invalid level names — tracked per candidate
- Statistical false negatives: correct predicate synthesized but validation gain below threshold
- Statistical false positives: spurious factors passing held-out validation
- Decomposition failures: rediscovering bijections of known factors (should be caught by hard rejection)

---

## 11. Key Design Decisions and Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Experiment design language | SweetPea | Declarative; maps directly to predicate synthesis; established in cognitive science |
| Predicate language | Python with restricted imports | Flexible enough for all factor types; sandboxable; LLMs code it reliably |
| Seeding | Random + LLM strategies | Random ensures coverage; LLM provides domain-relevant proposals |
| Scoring | Participant-wise CV log-likelihood | Avoids overfitting; marginal LL improvement is interpretable; complexity-adjusted winner selection controls spurious multi-level factors |
| Validation | Fixed held-out 20% set | Separates selection (CV on search set) from acceptance (held-out set) |
| Factor matching | Bijection via Hungarian algorithm | Handles arbitrary LLM level naming; exact matching for discrete, correlation for continuous |
| Novelty scoring | NMI (discrete) / Spearman ρ (continuous) | Promotes diversity in discovered factors; prevents rediscovering variants |
| Sandbox backend | Subprocess (default) | No Docker dependency for standard use; Docker available for stronger isolation |
| Multi-outcome | Joint acceptance across all outcomes | Ensures factors are genuinely predictive of all behavioral signatures, not just one |

---

## 12. References

- Gratton, G., Coles, M. G. H., & Donchin, E. (1992). Optimizing the use of information: Strategic control of activation of responses. *Journal of Experimental Psychology: General*, 121(4), 480–506.
- MacLeod, C. M. (1991). Half a century of research on the Stroop effect: An integrative review. *Psychological Bulletin*, 109(2), 163–203.
- Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*, 7(3), 134–140.
- Musslick, S., Cherkaev, A., Draut, B., Butt, A. S., Donnelly, P., Langlois, V., ... & Cohen, J. D. (2020). SweetPea: A standard language for factorial experimental design. *Behavior Research Methods*, 52, 2370–2395.
- Pashler, H., & Baynes, K. (2001). Attention and performance. In *The MIT Encyclopedia of Cognitive Sciences*. MIT Press.
- Stroop, J. R. (1935). Studies of interference in serial verbal reactions. *Journal of Experimental Psychology*, 18(6), 643–662.

---

## Resources

### SweetPea
Declarative language for factorial experiment design. Used for generating counterbalanced trial sequences and for formally expressing derived factor definitions.

- **Repository**: https://github.com/sweetpea-org/sweetpea-py
- **Documentation**: https://sweetpea-org.github.io/
- **Key classes**: `Factor`, `DerivedLevel`, `WithinTrial`, `Window`, `CrossBlock`, `synthesize_trials`, `RandomGen`

### Anthropic Python SDK
Used to call the Claude API for candidate generation, predicate synthesis, genetic evolution, and effect naming.

- **Repository**: https://github.com/anthropics/anthropic-sdk-python
- **API reference**: https://docs.anthropic.com/en/api/

### LLM Sandbox (`llm-sandbox`)
Optional Docker-based sandboxed execution for LLM-generated code.

- **Repository**: https://github.com/vndee/llm-sandbox
