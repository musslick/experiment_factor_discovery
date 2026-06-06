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
│   └── stroop_benchmark.yaml      # All tunable parameters
├── data/
│   ├── ground_truth/              # Full dataset (created by generate_data.py)
│   └── input/                     # Observable-only dataset (pipeline input)
├── prompts/
│   ├── candidate_generation_system.txt
│   ├── candidate_generation_user.txt
│   ├── predicate_synthesis_system.txt
│   └── predicate_synthesis_user.txt
├── results/                       # Timestamped run outputs
├── src/
│   ├── analysis/
│   │   ├── evaluation.py          # Bijection matching, precision/recall/F1
│   │   ├── factor_encoder.py      # Sandbox output → pandas Series + validation
│   │   └── model_comparison.py    # Logistic LRT, formula building
│   ├── data_generation/
│   │   ├── stroop_model.py        # Ground-truth logistic model + accuracy sampling
│   │   └── sweetpea_builder.py    # SweetPea trial sequence synthesis
│   ├── discovery/
│   │   ├── candidate_generator.py # LLM → structured factor proposals
│   │   ├── factor_registry.py     # CandidateFactor, DiscoveredFactor, FactorRegistry
│   │   ├── llm_client.py          # Anthropic SDK wrapper with retry
│   │   ├── pipeline.py            # Multi-round discovery orchestration
│   │   ├── predicate_synthesizer.py # LLM → predicate code + self-correction loop
│   │   └── sandbox.py             # Subprocess/Docker sandboxed code execution
│   └── utils/
│       └── config.py              # YAML loader + typed dataclasses
├── tests/
│   ├── test_data_generation.py    # Phase 1: SweetPea output + logistic model
│   ├── test_model_comparison.py   # Phase 2: LRT, formula builder, factor encoder
│   ├── test_sandbox.py            # Phase 3: subprocess harness + error handling
│   ├── test_llm_integration.py    # Phase 4: API calls, synthesis, pipeline round
│   └── test_evaluation.py         # Phase 5: bijection matching, oracle tests
├── generate_data.py               # Phase 1 CLI
├── run_benchmark.py               # End-to-end benchmark CLI
├── run_llm_test.py                # LLM smoke test script
├── research_plan.md               # Full scientific design document
└── requirements.txt
```

---

## How the pipeline works

```
Observable data: {task, color, word, correct}
        │
        ▼
┌─────────────────────┐
│  Candidate generator │  LLM proposes derived factor candidates as JSON
│  (LLM call)         │  e.g. {name: "congruency", type: "within_trial", ...}
└──────────┬──────────┘
           │  for each candidate:
           ▼
┌─────────────────────┐
│  Predicate          │  LLM writes compute_factor() + SweetPea definition
│  synthesizer        │  Self-correction loop: sandbox validates, retries on error
│  (LLM call + sandbox│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Factor encoder     │  Applies predicate to dataset → new column
│                     │  Validates level counts (guards against separation)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Logistic LRT       │  null:  correct ~ [known factors]
│                     │  alt:   correct ~ [known factors] + C(candidate)
│                     │  Significant (p < α) → register factor
└──────────┬──────────┘
           │  repeat for N rounds; registered factors become available
           │  as context for the next round's candidate proposals
           ▼
┌─────────────────────┐
│  Evaluation         │  Bijection matching vs hidden ground-truth factors
│                     │  Precision / Recall / F1
└─────────────────────┘
```

**Predicate contract.** The LLM must produce a function named `compute_factor`:
- Within-trial: `def compute_factor(trial: dict) -> str`
- Transition: `def compute_factor(prev: dict, curr: dict) -> str`

The function is executed in an isolated subprocess (or Docker container) against the full dataset; no SweetPea re-synthesis is needed during discovery.

**Factor matching.** A discovered factor matches a ground-truth factor when there exists a bijection between their level sets that agrees on ≥ 95 % of applicable trials. This handles cases where the LLM uses different level names (e.g. `"same"/"different"` instead of `"congruent"/"incongruent"`). The Hungarian algorithm finds the optimal assignment when multiple factors must be matched simultaneously.

---

## Configuration

All parameters live in `config/stroop_benchmark.yaml`:

| Section | Key parameters |
|---|---|
| `data_generation` | `n_participants`, `n_blocks_per_participant`, `hidden_factors`, logistic model coefficients |
| `discovery` | `n_rounds`, `max_candidates_per_round`, `max_synthesis_retries`, `sandbox_backend` |
| `llm` | `model` (default `claude-sonnet-4-6`), temperatures for candidate vs predicate generation |
| `statistical` | `alpha` (LRT threshold), `min_level_count`, `separation_check` |
| `evaluation` | `ground_truth_factors`, `bijection_threshold` |

The initial benchmark uses **2 hidden factors** (congruency + task transition). Extend to all 4 by uncommenting the `response_transition` and `congruency_sequence` entries in the config and updating the logistic model coefficients.

**Sandbox backend.** Set `sandbox_backend: subprocess` for local execution (no extra dependencies). Set `sandbox_backend: docker` for stronger isolation via `llm-sandbox` (requires Docker daemon and `pip install llm-sandbox`).

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
