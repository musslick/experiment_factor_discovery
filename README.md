# Automated Discovery of Experimental Variables

A benchmarking pipeline that automatically recovers latent derived experimental factors from behavioural data using LLM-driven program synthesis.

Given a dataset annotated only with basic observable design factors (e.g. task, colour, word) and a binary outcome (accuracy), the pipeline proposes candidate derived factors, synthesises Python predicate functions for each, validates them against the data, and uses nested logistic regression (likelihood ratio test) to determine which candidates genuinely improve prediction. Discovered factors are then matched against a hidden ground truth to compute precision, recall, and F1.

The initial benchmark is built around the **Stroop task**, where the hidden ground-truth factors are *congruency* (within-trial) and *task transition* (across consecutive trials).

See `research_plan.md` for the full scientific background and design rationale.

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
python generate_data.py --config config/stroop_benchmark.yaml
```

Produces:
- `data/ground_truth/stroop_full.csv` — all columns including hidden factors
- `data/input/stroop_input.csv` — observable columns only (pipeline input)

**2. Smoke-test the LLM integration**

```bash
python run_llm_test.py
```

Runs four checks (API connectivity, candidate generation, within-trial predicate synthesis, transition predicate synthesis) and prints pass/fail for each.

**3. Run the full benchmark**

```bash
python run_benchmark.py --config config/stroop_benchmark.yaml
```

Runs the multi-round discovery pipeline on the input data, evaluates against the ground truth, and writes a timestamped report to `results/run_<timestamp>/`.

---

## VS Code launch configurations

| Configuration | What it runs |
|---|---|
| Run Benchmark | `run_benchmark.py` end-to-end |
| Generate Data | `generate_data.py` |
| Test LLM (smoke test) | `run_llm_test.py` |
| Test LLM (pytest suite) | Full integration test suite |
| Python Debugger: Current File | Whichever file is open |

---

## Project structure

```
.
├── config/
│   └── stroop_benchmark.yaml        # All tunable parameters
├── data/
│   ├── ground_truth/                # Full dataset (created by generate_data.py)
│   └── input/                       # Observable-only dataset (pipeline input)
├── prompts/
│   ├── candidate_generation_system.txt
│   ├── candidate_generation_user.txt
│   ├── candidate_refinement_system.txt  # Iterative refinement based on CV scores
│   ├── candidate_refinement_user.txt
│   ├── effect_ranking_system.txt    # LLM ranking of interaction candidates
│   ├── effect_ranking_user.txt
│   ├── predicate_synthesis_system.txt
│   └── predicate_synthesis_user.txt
├── results/                         # Timestamped run outputs
├── src/
│   ├── analysis/
│   │   ├── evaluation.py            # Bijection matching, precision/recall/F1
│   │   ├── factor_encoder.py        # Sandbox output → pandas Series + validation
│   │   └── model_comparison.py      # Logistic CV scoring, formula building
│   ├── data_generation/
│   │   ├── stroop_model.py          # Ground-truth logistic model + accuracy sampling
│   │   └── sweetpea_builder.py      # SweetPea trial sequence synthesis
│   ├── discovery/
│   │   ├── candidate_generator.py   # LLM → structured factor proposals + refinement
│   │   ├── effect_searcher.py       # Interaction term discovery (Phase 2)
│   │   ├── factor_registry.py       # CandidateFactor, DiscoveredFactor, FactorRegistry
│   │   ├── llm_client.py            # Anthropic SDK wrapper with retry
│   │   ├── pipeline.py              # Multi-round discovery orchestration
│   │   ├── predicate_synthesizer.py # LLM → predicate code + self-correction loop
│   │   ├── sandbox.py               # Subprocess/Docker sandboxed code execution
│   │   └── within_round_search.py   # Iterative generate→score→refine loop per round
│   └── utils/
│       └── config.py                # YAML loader + typed dataclasses
├── tests/
│   ├── test_data_generation.py      # Phase 1: SweetPea output + logistic model
│   ├── test_model_comparison.py     # Phase 2: CV scoring, formula builder, factor encoder
│   ├── test_sandbox.py              # Phase 3: subprocess harness + error handling
│   ├── test_llm_integration.py      # Phase 4: API calls, synthesis, pipeline round
│   └── test_evaluation.py           # Phase 5: bijection matching, oracle tests
├── generate_data.py                 # Data generation CLI
├── run_benchmark.py                 # End-to-end benchmark CLI
├── run_llm_test.py                  # LLM smoke test script
├── research_plan.md                 # Full scientific design document
└── requirements.txt
```

---

## How the pipeline works

Each round runs an iterative within-round search (Phase 1), followed by an optional interaction-effect search (Phase 2):

```
Observable data: {task, color, word, correct}
        │
        ▼  (once, before round 1)
┌─────────────────────┐
│  Participant split   │  80% search set / 20% held-out validation set
└──────────┬──────────┘
           │
           ▼  ──────────── repeated for N rounds ────────────
┌──────────────────────────────────────────────────────────┐
│  PHASE 1: Factor Discovery (within-round search loop)    │
│                                                          │
│  ┌──────────────────┐                                    │
│  │ Candidate        │  LLM proposes derived factor       │
│  │ generator        │  candidates as JSON                │
│  └────────┬─────────┘                                    │
│           │  for each candidate:                         │
│           ▼                                              │
│  ┌──────────────────┐                                    │
│  │ Predicate        │  LLM writes compute_factor()       │
│  │ synthesizer      │  + SweetPea definition             │
│  │ (+ sandbox)      │  Self-correction on error          │
│  └────────┬─────────┘                                    │
│           ▼                                              │
│  ┌──────────────────┐                                    │
│  │ Factor encoder   │  Applies predicate → new column    │
│  │                  │  Validates level counts            │
│  └────────┬─────────┘                                    │
│           ▼                                              │
│  ┌──────────────────┐                                    │
│  │ CV scoring       │  Participant-wise 5-fold CV on     │
│  │ (search set)     │  search set; marginal LL           │
│  │                  │  improvement over null formula     │
│  └────────┬─────────┘                                    │
│           │  ↺ refine: LLM sees CV scores → proposes    │
│           │    improved candidates (up to K iterations)  │
│           ▼                                              │
│  ┌──────────────────┐                                    │
│  │ Winner selection │  Highest complexity-adjusted       │
│  │                  │  score: (mean−λ·SE) / complexity   │
│  └────────┬─────────┘                                    │
│           ▼                                              │
│  ┌──────────────────┐                                    │
│  │ Validation       │  Winner tested on held-out set     │
│  │ (held-out set)   │  mean LL gain ≥ threshold          │
│  │                  │  → register factor                 │
│  └──────────────────┘                                    │
│                                                          │
│  PHASE 2: Effect Search (optional)                       │
│  CV-score and validate pairwise interaction terms        │
│  among discovered factors on the same split              │
└──────────────────────────────────────────────────────────┘
           │  registered factors feed into next round's
           │  null formula and candidate generation context
           ▼
┌─────────────────────┐
│  Evaluation         │  Bijection matching vs hidden ground-truth factors
│                     │  Precision / Recall / F1
└─────────────────────┘
```

**Predicate contract.** The LLM produces a function named `compute_factor`:
- Within-trial: `def compute_factor(trial: dict) -> str`
- Transition: `def compute_factor(prev: dict, curr: dict) -> str`

The function is executed in an isolated subprocess (or Docker container) against the full dataset; no SweetPea re-synthesis is needed during discovery.

**Factor matching.** A discovered factor matches a ground-truth factor when there exists a bijection between their level sets that agrees on ≥ 95 % of applicable trials. This handles cases where the LLM uses different level names (e.g. `"same"/"different"` instead of `"congruent"/"incongruent"`). The Hungarian algorithm finds the optimal assignment when multiple factors must be matched simultaneously.

---

## Configuration

All parameters live in `config/stroop_benchmark.yaml`:

| Section | Key parameters |
|---|---|
| `data_generation` | `n_participants` (100), `n_blocks_per_participant` (11 × 18 = 198 trials/participant), `hidden_factors`, logistic model coefficients |
| `discovery` | `n_rounds`, `max_candidates_per_round`, `max_synthesis_retries`, `sandbox_backend`; iterative search: `candidates_per_refinement`, `max_search_iterations`, `refinement_top_k`; scoring: `cv_n_folds`, `validation_fraction`, `min_validation_improvement`, `stability_weight`, `complexity_exponent`, `depends_on_exponent` |
| `llm` | `model` (default `claude-sonnet-4-6`), temperatures for candidate vs predicate generation |
| `statistical` | `min_level_count` (guards against near-constant factors) |
| `evaluation` | `ground_truth_factors`, `bijection_threshold`, `ground_truth_interactions` |
| `discovery` (effect search) | `run_effect_search`, `max_interaction_order`, `max_interactions_per_round`, `effect_search_min_cv_improvement`, `effect_search_min_validation_improvement`, `llm_rank_interactions` |

The initial benchmark uses **2 hidden factors** (congruency + task transition). Extend to all 4 by adding `response_transition` and `congruency_sequence` to `hidden_factors` in the config and updating the logistic model coefficients.

**Sandbox backend.** Set `sandbox_backend: subprocess` for local execution (no extra dependencies). Set `sandbox_backend: docker` for stronger isolation via `llm-sandbox` (requires Docker daemon and `pip install llm-sandbox`).

---

## Using your own empirical data

There are two ways to run the pipeline on a real dataset:

| Script | Mode | Purpose |
|---|---|---|
| `run_benchmark.py` | `empirical_benchmark` | Evaluate factor recovery against a **known** ground truth — the dataset contains hidden factor columns that the pipeline must re-discover. |
| `run_discovery.py` | `novel_discovery` | Find genuinely **new** factors — the pipeline starts from all currently known factors and searches for whatever lies beyond them. |

Both use the same YAML config format. The difference is a single field (`mode`) and which factors you list as `hidden_factors`.

---

### Step 1 — Prepare your CSV

The pipeline expects a flat trial-level CSV. Required columns:

| Column | Notes |
|---|---|
| Participant ID | Any name; specify via `participant_id_column` in the config. |
| Outcome variable | Binary integer (0 / 1) or boolean. Specify via `outcome_variable`. |
| Base factor columns | One column per observable design factor (e.g. `color`, `word`). |

Optional but common:

| Column | Notes |
|---|---|
| Trial ordering | Any column that identifies trial order within a participant (e.g. `trialnum`). List it under `extra_columns` to keep it in the dataframe. The pipeline assigns its own `trial_index` from row order within each participant. |
| Hidden factor columns | Only needed for `empirical_benchmark` mode, where they are the ground truth the pipeline is evaluated against. |

Rows do **not** need to be sorted; the pipeline groups by participant and uses row order within each group as trial order.

---

### Step 2 — Create the dataset config YAML

Create a file in `config/`, e.g. `config/my_experiment.yaml`.

#### Mode A: `empirical_benchmark` (known ground truth)

Use this when your CSV already contains the hidden factor columns and you want to measure how well the pipeline recovers them.

```yaml
benchmark:
  name: "my_experiment"
  mode: "empirical_benchmark"

dataset:
  path: "data/empirical/my_experiment.csv"
  participant_id_column: "subject_id"   # column name for participant ID in the CSV
  outcome_variable: "accuracy"          # binary 0/1 column
  task_context: |
    Describe the task here. This text is injected into every LLM prompt, so
    be specific: what does the participant see, what response do they make,
    what does the outcome measure?
  base_factors:
    - {name: factor_a, dtype: categorical, levels: [level1, level2]}
    - {name: factor_b, dtype: categorical, levels: [level1, level2, level3]}
  hidden_factors:               # columns present in the CSV but withheld from the pipeline
    - name: "my_hidden_factor"  # model-facing name (used in formulas and evaluation)
      column: "my_hidden_factor"  # actual CSV column name (omit if same as name)
      type: "within_trial"      # "within_trial" or "window"
      levels: ["level_a", "level_b"]
  extra_columns: ["trialnum"]   # columns to keep in the dataframe but not model
  null_formula: "accuracy ~ C(factor_a) + C(factor_b)"   # auto-built if omitted
  full_formula: "accuracy ~ C(factor_a) + C(factor_b) + C(my_hidden_factor)"

discovery:
  sandbox_timeout_seconds: 30   # increase for large datasets (>50k rows)
  allowed_factor_classes: ["discrete"]
```

The `evaluation` section is optional: ground-truth factors are auto-populated from `hidden_factors` when omitted.

#### Mode B: `novel_discovery` (no known ground truth)

Use this when you want to discover factors beyond what is already known. List **all currently known factors** under `hidden_factors` — they will be included in the starting model (the `full_formula` baseline), and the pipeline searches for what lies on top of them.

```yaml
benchmark:
  name: "my_experiment_discovery"
  mode: "novel_discovery"

dataset:
  path: "data/empirical/my_experiment.csv"
  participant_id_column: "subject_id"
  outcome_variable: "accuracy"
  task_context: |
    Describe the task here.
  base_factors:
    - {name: factor_a, dtype: categorical, levels: [level1, level2]}
    - {name: factor_b, dtype: categorical, levels: [level1, level2, level3]}
  hidden_factors:               # already-known factors to include in the starting baseline
    - name: "known_factor"
      type: "within_trial"
      levels: ["level_a", "level_b"]
  extra_columns: ["trialnum"]
  null_formula: "accuracy ~ C(factor_a) + C(factor_b)"
  full_formula: "accuracy ~ C(factor_a) + C(factor_b) + C(known_factor)"

discovery:
  sandbox_timeout_seconds: 30
  allowed_factor_classes: ["discrete"]
```

The key difference from Mode A: in `novel_discovery` the hidden factors are **not** stripped from the data — they are visible to the pipeline as part of the starting model. The pipeline then searches for factors above and beyond them.

---

### Step 3 — Run

#### Benchmarking — evaluate factor recovery

```bash
python run_benchmark.py --config config/my_experiment.yaml
```

Results are written to `results/run_<timestamp>/my_experiment/`:

```
results/run_20260608_120000/
└── my_experiment/
    ├── benchmark_config.yaml       # snapshot of the resolved config
    ├── round_01_candidates.json    # all scored candidates + predicates
    ├── round_01_effects.json       # interaction effects found in round 1
    └── evaluation_report.json      # precision / recall / F1 vs ground truth
```

To regenerate synthetic data in the same run (not applicable to empirical mode), pass `--regenerate`.

#### Discovery — find new factors

```bash
python run_discovery.py --config config/my_experiment.yaml
```

Results are written to `results/discovery_<timestamp>/my_experiment/`:

```
results/discovery_20260608_120000/
└── my_experiment/
    ├── round_01_candidates.json
    ├── round_01_effects.json
    └── discovery_results.yaml      # discovered factors with SweetPea code + LLM interpretation
```

`discovery_results.yaml` contains the discovered factor names, level assignments, the `compute_factor` Python code for each, the corresponding SweetPea `Factor` definition, and an LLM-generated name and interpretation for every discovered main effect and interaction.

---

### Step 4 — Add to a multi-dataset run (optional)

To include your dataset in a batch run alongside others, register it in the relevant top-level config.

**For benchmarking** — add to `config/benchmark.yaml`:

```yaml
benchmarks:
  - config/synthetic_stroop_benchmark.yaml
  - config/my_experiment.yaml      # ← add here

# shared defaults (overridden by individual configs)
benchmark:
  seed: 42
  output_dir: "results"
discovery:
  n_rounds: 3
  ...
```

Then run all benchmarks in one session:

```bash
python run_benchmark.py --config config/benchmark.yaml
```

**For discovery** — add to `config/discovery.yaml`:

```yaml
datasets:
  - config/my_experiment.yaml      # ← add here

# shared defaults merged into every dataset config
benchmark:
  seed: 42
  output_dir: "results"
discovery:
  n_rounds: 3
  ...
```

Then run all discovery sessions in one session:

```bash
python run_discovery.py --config config/discovery.yaml
```

Shared defaults in `benchmark.yaml` / `discovery.yaml` are deep-merged with individual configs; per-dataset values take precedence.

---

## Testing

```bash
# Fast unit tests only (no API calls, ~20 s)
.venv/bin/python -m pytest -m "not integration" -v

# Full suite including LLM integration tests (~70 s, uses Anthropic API)
.venv/bin/python -m pytest -v
```

| Test file | Phase | API calls |
|---|---|---|
| `test_data_generation.py` | 1 | No |
| `test_model_comparison.py` | 2 | No |
| `test_sandbox.py` | 3 | No |
| `test_llm_integration.py` | 4 | Yes |
| `test_evaluation.py` | 5 | No |
