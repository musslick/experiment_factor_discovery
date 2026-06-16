# Automated Discovery of Experimental Variables

A benchmarking pipeline that automatically recovers latent derived experimental factors from behavioral data using LLM-driven program synthesis.

Given a dataset annotated only with observable design factors (e.g. task, color, word) and behavioral outcomes (accuracy, RT), the pipeline proposes candidate derived factors, synthesizes Python predicate functions for each, validates them in a sandbox, and uses participant-wise cross-validated logistic/linear regression to determine which candidates genuinely improve prediction. Discovered factors are matched against hidden ground truth to compute precision, recall, and F1.

The pipeline includes **four synthetic benchmarks** (Stroop, Stroop-Simon, RDK task-switching, Prospect Theory) and supports real empirical datasets in both benchmark mode (evaluate recovery of known factors) and discovery mode (find genuinely new factors).

See `research_plan.md` for the full scientific background and design rationale. See `paper_plan.md` for the conference paper plan.

---

## Prerequisites

- Python 3.9 or later
- An Anthropic API key (set in `.env`)

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` (or create `.env` directly) and add your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick start

**1. Generate synthetic data**

```bash
python generate_data.py --config config/synthetic_stroop_benchmark.yaml
```

Produces:
- `data/ground_truth/stroop_full.csv` — all columns including hidden factors
- `data/input/stroop_input.csv` — observable columns only (pipeline input)

**2. Smoke-test the LLM integration**

```bash
python run_llm_test.py
```

Runs connectivity, candidate generation, and predicate synthesis checks and prints pass/fail for each.

**3. Run a benchmark**

```bash
# Single synthetic benchmark
python run_benchmark.py --config config/synthetic_stroop_benchmark.yaml

# All benchmarks in one run
python run_benchmark.py --config config/benchmark.yaml

# Empirical benchmark (evaluate recovery of known factor)
python run_benchmark.py --config config/empirical_stroop_congruency.yaml
```

**4. Run novel discovery on empirical data**

```bash
# Single dataset
python run_discovery.py --config config/empirical_stroop_congruency.yaml

# All datasets in one run
python run_discovery.py --config config/discovery.yaml
```

Regenerate synthetic data in the same run by passing `--regenerate`.

---

## How the pipeline works

Each round runs an iterative within-round search (Phase 1), followed by an optional interaction-effect search (Phase 2):

```
Observable data: base factors + outcome column(s)
        │
        ▼  (once, before round 1)
┌─────────────────────┐
│  Participant split  │  80% search set / 20% held-out validation set
│  + null formula     │
└──────────┬──────────┘
           │
           ▼  ──────────── repeated for N rounds ────────────
┌──────────────────────────────────────────────────────────────────┐
│  PHASE 1: Factor Discovery (within-round iterative search)       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Seeding (random / LLM / mixed)                          │   │
│  │  → N candidate factor descriptions                       │   │
│  └──────────────────────┬─────────────────────────────────  │   │
│                         │  for each candidate:              │   │
│                         ▼                                   │   │
│  ┌──────────────────────────────────────────────────────┐   │   │
│  │  Predicate synthesis  LLM → compute_factor() code    │   │   │
│  │  + sandboxed execution + factor encoding             │   │   │
│  │  + hard-rejection (bijection/coarsening check)       │   │   │
│  └──────────────────────┬─────────────────────────────── │   │   │
│                         ▼                                │   │   │
│  ┌──────────────────────────────────────────────────────┐│   │   │
│  │  CV scoring on search set (participant-wise 5-fold)  ││   │   │
│  │  score = (mean − λ·SE) / complexity + novelty_weight ││   │   │
│  └──────────────────────┬─────────────────────────────── │   │   │
│                         │                                │   │   │
│                         ↺  Evolution (LLM genetic /      │   │   │
│                         │  mutation) → refined candidates │   │   │
│                         │  (up to max_search_iterations)  │   │   │
│                         ▼                                │   │   │
│  ┌──────────────────────────────────────────────────────┐│   │   │
│  │  Validation on held-out set                          ││   │   │
│  │  mean LL gain ≥ threshold → register factor          ││   │   │
│  └──────────────────────────────────────────────────────┘│   │   │
│                                                          │   │   │
│  PHASE 2: Effect Search (optional)                       │   │   │
│  CV-score and validate pairwise interaction terms        │   │   │
└──────────────────────────────────────────────────────────┘   │   │
           │  registered factors → updated null formula         │
           │  + context for next round's seeding                │
           ▼                                                     │
┌─────────────────────┐                                         │
│  Evaluation         │  Bijection matching vs. hidden          │
│                     │  ground-truth factors                   │
│                     │  Precision / Recall / F1                │
└─────────────────────┘
```

**Predicate contract.** The LLM produces a function named `compute_factor`:
- Within-trial: `def compute_factor(trial: dict) -> str | float`
- Window: `def compute_factor(window: list) -> str | float` (list of trial dicts, oldest first)

**Factor matching.** A discovered factor matches a ground-truth factor when:
- Discrete: a bijection between level sets agrees on ≥ 95% of applicable trials (handles LLMs using different level names, e.g. `"same"/"different"` instead of `"congruent"/"incongruent"`)
- Continuous: |Spearman ρ| ≥ 0.70 over applicable trials

The Hungarian algorithm finds the optimal assignment when multiple factors must be matched simultaneously.

---

## Benchmarks

| Benchmark | Base Factors | Hidden Factors | Factor Types | Outcome |
|---|---|---|---|---|
| Stroop | 3 (task, color, word) | 2 (congruency, task_transition) | within-trial + window, discrete | accuracy |
| Stroop-Simon | 3 (word, color, location) | 4 (word-color congruency, location-response congruency, sequential congruency, response_transition) | within-trial + window, discrete | accuracy |
| RDK Task-Switching | 9 (task, stimuli, coherences) | 4 (task_transition, stimulus_difficulty, past_difficulty, n2_inhibition) | window + within-trial, discrete + continuous | accuracy |
| Prospect Theory | 6 continuous gamble params | 9 (expected values, differences, dominance, sequential EV) | within-trial + window, continuous + discrete | binary choice |
| Stroop (empirical) | 2 (color, word) | 1 (congruency) | within-trial, discrete | latency + accuracy |

---

## Project structure

```
.
├── config/
│   ├── benchmark.yaml                      # Multi-benchmark runner
│   ├── discovery.yaml                      # Multi-dataset discovery runner
│   ├── synthetic_stroop_benchmark.yaml
│   ├── synthetic_stroop_simon_benchmark.yaml
│   ├── synthetic_rdk_task_switching_benchmark.yaml
│   ├── synthetic_prospect_theory_benchmark.yaml
│   └── empirical_stroop_congruency.yaml
├── data/
│   ├── ground_truth/                       # Full datasets (created by generate_data.py)
│   ├── input/                              # Observable-only inputs (pipeline input)
│   └── empirical/                          # Real behavioral datasets
├── benchmarks/
│   ├── stroop.md
│   ├── stroop_simon.md
│   ├── rdk_task_switching.md
│   └── prospect_theory.md
├── prompts/
│   ├── candidate_generation_{system,user}.txt
│   ├── candidate_refinement_{system,user}.txt
│   ├── genetic_evolution_{system,user}.txt
│   ├── predicate_synthesis_{system,user}.txt
│   ├── effect_ranking_{system,user}.txt
│   └── effect_naming_{system,user}.txt
├── results/                                # Timestamped run outputs
├── src/
│   ├── analysis/
│   │   ├── evaluation.py                   # Bijection matching, P/R/F1
│   │   ├── factor_encoder.py               # Sandbox output → pandas Series + validation
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
│   │   ├── pipeline.py                     # Multi-round orchestration
│   │   ├── within_round_search.py          # Seeding → scoring → evolution loop
│   │   ├── effect_searcher.py              # Interaction term discovery (Phase 2)
│   │   ├── llm_client.py                   # Anthropic SDK wrapper with retry
│   │   ├── candidate_generator.py          # Candidate description generation
│   │   ├── predicate_synthesizer.py        # LLM → compute_factor() + self-correction
│   │   ├── sandbox.py                      # Subprocess/Docker sandboxed execution
│   │   ├── factor_registry.py              # Factor registry + formula management
│   │   └── strategies/                     # Seeding and evolution strategy plugins
│   │       ├── base.py
│   │       ├── random_seeder.py
│   │       ├── llm_seeder.py
│   │       ├── mixed.py
│   │       ├── mutation_evolver.py
│   │       ├── llm_evolver.py
│   │       └── llm_genetic_evolver.py
│   └── utils/
│       └── config.py                       # YAML loader + typed dataclasses
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
├── generate_data.py
├── run_benchmark.py
├── run_discovery.py
├── run_llm_test.py
├── diagnose_synthesis.py
├── research_plan.md
└── paper_plan.md
```

---

## Configuration

Key parameters (all live in the benchmark/dataset YAML configs):

| Section | Key parameters |
|---|---|
| `benchmark` | `name`, `mode` (synthetic_benchmark / empirical_benchmark / novel_discovery), `seed`, `output_dir` |
| `dataset` | `path`, `participant_id_column`, `outcome_variables` (list, multi-outcome support), `base_factors`, `hidden_factors`, `task_context`, `null_formula` |
| `data_generation` | `n_participants`, `n_blocks_per_participant`, `hidden_factors`, logistic model coefficients |
| `discovery` | `n_rounds`, `max_search_iterations`, `max_synthesis_retries`, `sandbox_backend` (subprocess / docker), `sandbox_timeout_seconds` |
| `discovery` (scoring) | `cv_n_folds`, `validation_fraction`, `min_validation_improvement`, `stability_weight` (λ), `complexity_exponent` (α), `depends_on_exponent` (β), `novelty_weight` |
| `discovery.seeding_strategy` | `type` (random / llm / mixed), `n_candidates`, `template_bias`, `allow_window`, `max_depends_on` |
| `discovery.evolution_strategy` | `type` (llm_genetic / llm / mutation / mixed), `n_candidates`, `top_k`, `operator_mix` |
| `discovery` (effect search) | `run_effect_search`, `max_interaction_order`, `max_interactions_per_round`, `effect_search_min_cv_improvement` |
| `llm` | `model` (default: `claude-sonnet-4-6`), `candidate_temperature`, `predicate_temperature` |
| `statistical` | `min_level_count` (guards against near-constant factors) |

---

## Using your own empirical data

There are two modes for running the pipeline on a real dataset:

| Script | Mode | Purpose |
|---|---|---|
| `run_benchmark.py` | `empirical_benchmark` | Evaluate factor recovery against a **known** ground truth — the dataset contains hidden factor columns that the pipeline must re-discover. |
| `run_discovery.py` | `novel_discovery` | Find genuinely **new** factors — the pipeline starts from all currently known factors and searches for what lies beyond them. |

### Step 1 — Prepare your CSV

The pipeline expects a flat trial-level CSV. Required columns:

| Column | Notes |
|---|---|
| Participant ID | Any name; specify via `participant_id_column` in the config. |
| Outcome column(s) | Binary integer (0/1) for accuracy; float for RT/latency. Specify via `outcome_variables`. |
| Base factor columns | One column per observable design factor. |

Rows do not need to be sorted; the pipeline groups by participant and uses row order within each group as trial order.

### Step 2 — Create the config YAML

#### Mode A: `empirical_benchmark` (known ground truth)

```yaml
benchmark:
  name: "my_experiment"
  mode: "empirical_benchmark"

dataset:
  path: "data/empirical/my_experiment.csv"
  participant_id_column: "subject_id"
  outcome_variables:
    - {name: accuracy, type: binary}
  task_context: |
    Describe the task here. This text is injected into every LLM prompt.
  base_factors:
    - {name: factor_a, dtype: categorical, levels: [level1, level2]}
    - {name: factor_b, dtype: categorical, levels: [level1, level2, level3]}
  hidden_factors:
    - name: "my_hidden_factor"
      type: "within_trial"
      levels: ["level_a", "level_b"]
  null_formula: "accuracy ~ C(factor_a) + C(factor_b)"

discovery:
  sandbox_timeout_seconds: 30
```

#### Mode B: `novel_discovery` (no known ground truth)

List all currently known factors under `hidden_factors` — they will be included in the starting model, and the pipeline searches for factors on top of them.

```yaml
benchmark:
  name: "my_experiment_discovery"
  mode: "novel_discovery"

dataset:
  path: "data/empirical/my_experiment.csv"
  participant_id_column: "subject_id"
  outcome_variables:
    - {name: accuracy, type: binary}
  task_context: |
    Describe the task here.
  base_factors:
    - {name: factor_a, dtype: categorical, levels: [level1, level2]}
  hidden_factors:
    - name: "known_factor"
      type: "within_trial"
      levels: ["level_a", "level_b"]
  null_formula: "accuracy ~ C(factor_a) + C(known_factor)"

discovery:
  sandbox_timeout_seconds: 30
```

### Step 3 — Run

```bash
# Benchmarking — evaluate factor recovery
python run_benchmark.py --config config/my_experiment.yaml

# Discovery — find new factors
python run_discovery.py --config config/my_experiment.yaml
```

**Benchmark output** (`results/run_<timestamp>/my_experiment/`):
```
├── benchmark_config.yaml       # snapshot of the resolved config
├── round_01_candidates.json    # all scored candidates + predicates
├── round_01_effects.json       # interaction effects found in round 1
└── evaluation_report.json      # precision / recall / F1 vs ground truth
```

**Discovery output** (`results/discovery_<timestamp>/my_experiment/`):
```
├── round_01_candidates.json
├── round_01_effects.json
└── discovery_results.yaml      # factors with SweetPea code + LLM interpretations
```

`discovery_results.yaml` contains the discovered factor names, level assignments, the `compute_factor` Python code, the corresponding SweetPea `Factor` definition, and an LLM-generated name and psychological interpretation for every discovered main effect and interaction.

### Step 4 — Add to a batch run (optional)

Register your config in `config/benchmark.yaml` (for benchmarking) or `config/discovery.yaml` (for discovery) to include it in a multi-dataset run:

```yaml
# config/benchmark.yaml
benchmarks:
  - config/synthetic_stroop_benchmark.yaml
  - config/my_experiment.yaml      # ← add here

benchmark:
  seed: 42
  output_dir: "results"
```

```bash
python run_benchmark.py --config config/benchmark.yaml
```

Shared defaults in the top-level config are deep-merged with individual dataset configs; per-dataset values take precedence.

---

## Testing

```bash
# Fast unit tests only (no API calls, ~20 s)
.venv/bin/python -m pytest -m "not integration" -v

# Full suite including LLM integration tests (~70 s, uses Anthropic API)
.venv/bin/python -m pytest -v
```

| Test file | API calls | Coverage |
|---|---|---|
| `test_data_generation.py` | No | SweetPea output, logistic model sampling |
| `test_model_comparison.py` | No | CV scoring, formula building, multi-outcome |
| `test_sandbox.py` | No | Subprocess execution, error handling |
| `test_llm_integration.py` | Yes | API calls, synthesis pipeline, end-to-end round |
| `test_evaluation.py` | No | Bijection matching, Hungarian algorithm, oracle tests |
| `test_benchmark_data.py` | No | Data generation for all 4 synthetic benchmarks |
| `test_empirical_config.py` | No | Empirical dataset config validation |
| `test_empirical_pipeline.py` | Yes | End-to-end empirical discovery |
| `test_novelty.py` | No | NMI and Spearman novelty score computation |
