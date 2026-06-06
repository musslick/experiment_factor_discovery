# Research Plan: Automated Discovery of Experimental Variables via LLM-Driven Program Synthesis

## 1. Research Problem and Motivation

A central challenge in empirical science is the discovery of new experimental factors that explain observed behavior. Researchers often rely on domain expertise and intuition to propose derived variables—combinations of basic experimental factors that capture theoretically meaningful distinctions. For example, in the Stroop task (Stroop, 1935), the canonical experimental factors are the word stimulus, its ink color, and the task instruction (color naming vs. word reading). However, a rich literature has emerged by examining *derived* factors computed from these basics:

- **Congruency**: whether the word meaning matches its ink color (MacLeod, 1991)
- **Congruency sequence**: whether congruency repeats or changes across trials (Gratton et al., 1992)
- **Task transition**: whether the current task repeats or switches (Monsell, 2003)
- **Response transition**: whether the required motor response repeats or switches (Pashler & Baynes, 2001)

Each of these discoveries required a researcher to hypothesize a new derived factor, formalize it, compute it from existing data, and demonstrate its statistical relevance. This process is currently manual, slow, and dependent on domain expertise. The goal of this project is to automate it.

**Research Question**: Can a computational pipeline, driven by a large language model (LLM) and formal program synthesis, systematically discover derived experimental factors from behavioral data—recovering factors that are known to matter for behavior but are hidden from the pipeline?

---

## 2. Core Idea

We reduce the problem of discovering new derived experimental factors to **predicate function synthesis**: a derived factor is fully specified by a predicate that maps one or more existing factor values (within a trial, or across consecutive trials) to a discrete level label. This representation directly maps onto the SweetPea declarative experiment design language (Musslick et al., 2020), which distinguishes:

- **Within-trial derived factors**: `DerivedLevel(label, WithinTrial(predicate, [factor1, factor2, ...]))`
- **Transition derived factors**: `DerivedLevel(label, Transition(predicate, [factor]))`

The synthesis problem becomes: find predicates that, when applied to an existing trial sequence, produce a factor that significantly improves prediction of the behavioral outcome (accuracy).

---

## 3. Approach Overview

The pipeline operates as follows:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Discovery Pipeline                                  │
│                                                                             │
│  Observable data: {task, color, word, correct}                              │
│        │                                                                    │
│        ▼                                                                    │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐      │
│  │  LLM-based  │────▶│  LLM-based Code  │────▶│  Sandboxed Code     │      │
│  │  Candidate  │     │  Synthesis       │     │  Execution          │      │
│  │  Generation │     │  (Predicates)    │     │  (Docker/subprocess)│      │
│  └─────────────┘     └──────────────────┘     └──────────┬──────────┘      │
│                                                           │                  │
│                                                           ▼                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Statistical Analysis: Nested Logistic Regression + Likelihood Ratio  │ │
│  │  Test                                                                  │ │
│  │  null model:  correct ~ [known factors]                                │ │
│  │  alt model:   correct ~ [known factors] + [candidate factor]           │ │
│  │  LRT: p-value → significant? → register factor                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│        │                                                                    │
│        ▼                                                                    │
│  Discovered factors fed back as context for next round                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

The pipeline runs for a fixed number of rounds. Each round:
1. The LLM proposes candidate derived factors given the current known factors and the task context.
2. The LLM synthesizes Python predicate functions for each candidate.
3. Predicates are validated and executed in a sandbox, computing a new factor column for every trial.
4. A nested logistic regression comparison determines whether the candidate factor significantly improves prediction.
5. Significant factors are registered and become available as inputs to future candidates.

---

## 4. Benchmark Design

### 4.1 Synthetic Data Generation

We generate synthetic data from a known statistical model of the Stroop task. This creates a controlled "discovery challenge" where we can precisely measure what the pipeline recovers.

**Step 1: Full trial sequence generation (SweetPea, including hidden derived factors)**

All factors — both observable and hidden — are defined in the SweetPea design. The observable factors are placed in the crossing (fully counterbalanced); the hidden derived factors are included in the design only, so SweetPea computes their values for each trial without counterbalancing over them.

```python
# Observable (regular) factors
task  = Factor("task",  ["color_naming", "word_reading"])
color = Factor("color", ["red", "blue", "green"])
word  = Factor("word",  ["red", "blue", "green"])

# Hidden derived factor: congruency (within-trial)
def con(color, word):   return color == word
def inc(color, word):   return color != word
congruency = Factor("congruency", [
    DerivedLevel("congruent",   WithinTrial(con, [color, word])),
    DerivedLevel("incongruent", WithinTrial(inc, [color, word])),
])

# Hidden derived factor: task_transition (transition)
def task_rep(task): return task[0] == task[-1]
def task_swi(task): return task[0] != task[-1]
task_transition = Factor("task_transition", [
    DerivedLevel("repeat", Transition(task_rep, [task])),
    DerivedLevel("switch", Transition(task_swi, [task])),
])

# (Additional hidden factors defined analogously; omitted in 2-factor prototype)

# Crossing only over observable factors; derived factors ride along
block = fully_cross_block(
    design   = [task, color, word, congruency, task_transition],
    crossing = [task, color, word],
    constraints = [],
)
experiments = synthesize_trials(block, samples=n_participants,
                                sampling_strategy=UniformCombinatoricSamplingStrategy)
```

SweetPea outputs a dataset that already contains all factor columns, with transition-factor values correctly set to `NaN` for the first trial in each synthesized sequence. This yields a base block of 18 trials (2 × 3 × 3), repeated to reach approximately 200 trials per participant across 30 participants.

**Step 2: Dataset masking**

The full dataset (all columns) is saved as `data/ground_truth/stroop_full.csv`. The hidden derived-factor columns are then dropped to produce the discovery-challenge input:

```python
HIDDEN_COLUMNS = ["congruency", "task_transition"]  # extend for full benchmark
observable_df = full_df.drop(columns=HIDDEN_COLUMNS)
observable_df.to_csv("data/input/stroop_input.csv", index=False)
```

No post-hoc computation of derived factors is required; SweetPea handles all factor values during synthesis.

| Factor | Type | Definition in SweetPea |
|--------|------|------------------------|
| `congruency` | within-trial | `DerivedLevel` via `WithinTrial`; congruent iff `color == word` |
| `task_transition` | transition | `DerivedLevel` via `Transition`; repeat iff `task[0] == task[-1]` |
| `response_transition` | transition | Repeat iff response repeats (response = color or word depending on task) |
| `congruency_sequence` | transition | Four levels: `cc`, `ci`, `ic`, `ii` — congruency on prev and current trial |

**Step 3: Accuracy sampling**

Accuracy is sampled from a logistic model with known coefficients:

```
logit(P(correct)) = β₀
  + β_con     × I(congruency == "congruent")
  + β_task    × I(task_transition == "repeat")
  + β_resp    × I(response_transition == "repeat")
  + β_cs[key] × I(congruency_sequence == key)
```

Default coefficients (chosen to produce realistic effect sizes):

| Parameter | Value | Psychological Interpretation |
|-----------|-------|------------------------------|
| β₀ | 0.5 | ~62% baseline accuracy |
| β_con | 0.8 | Stroop congruency effect |
| β_task | 0.4 | Task switch cost |
| β_resp | 0.3 | Response repetition priming |
| β_cs[cc] | 0.2 | Post-congruent advantage (CSE) |
| β_cs[ci] | -0.3 | Post-congruent disadvantage on incongruent |
| β_cs[ic] | 0.1 | Post-incongruent on congruent |
| β_cs[ii] | 0.0 | Reference |

**Step 4: Masking**

The discovery pipeline receives only the observable columns: `{participant_id, trial_index, task, color, word, correct}`. All derived factor columns are stripped. The full dataset (with all factors) is retained as ground truth for evaluation.

### 4.2 Initial Benchmark Scope (Prototyping Phase)

For rapid prototyping, the initial benchmark uses only **2 hidden factors**:
- `congruency` (within-trial, 2-level)
- `task_transition` (transition, 2-level)

This covers both factor types, allows end-to-end validation, and keeps the statistical model simple. The ground-truth logistic model is correspondingly simplified (β_resp = 0, β_cs = 0). All 4 hidden factors will be used in subsequent experiments.

---

## 5. Discovery Pipeline: Technical Design

### 5.1 LLM-Based Candidate Generation

The LLM (default: `claude-sonnet-4-6`, configurable) is prompted with:
- **System prompt**: Explains the two types of derived factors (within-trial and transition), provides complete SweetPea examples showing how Boolean level predicates are written and assembled into a `Factor` definition, specifies the required JSON output schema, and lists constraints (no renamings, no re-proposals of rejected candidates).
- **User prompt**: Lists current observable factors, already-discovered factors with their descriptions, rejected candidates with reasons, and Stroop task context.

The LLM returns up to `max_candidates_per_round` (default: 8) candidate proposals as a JSON array:
```json
[
  {
    "name": "congruency",
    "description": "Whether the ink color matches the word meaning within a trial",
    "factor_type": "within_trial",
    "levels": ["congruent", "incongruent"],
    "depends_on": ["color", "word"],
    "rationale": "The Stroop effect is classically driven by color-word congruency"
  }
]
```

### 5.2 LLM-Based Predicate Synthesis

In SweetPea, each `DerivedLevel` is defined by a **Boolean predicate** (one per level) that returns `True` when the level applies and `False` otherwise. The pipeline therefore synthesizes predicates at the level granularity, not the factor granularity. For each candidate factor, a single LLM call produces three artifacts:

**(a) Boolean level predicates in SweetPea format** — one function per declared level, matching the SweetPea calling convention:

*Within-trial* predicates receive one positional argument per factor listed in `WithinTrial(fn, [factors])`:
```python
# SweetPea WithinTrial predicate signature: separate arg per factor
def congruent(color, word):    return color == word
def incongruent(color, word):  return color != word
```

*Transition* predicates receive one list argument per factor listed in `Transition(fn, [factors])`, where index `[0]` is the **previous** trial's value and index `[-1]` is the **current** trial's value:
```python
# SweetPea Transition predicate signature: one list per factor, [0]=prev, [-1]=curr
def task_repeat(task):  return task[0] == task[-1]
def task_switch(task):  return task[0] != task[-1]
```

**(b) Assembled SweetPea Factor definition** — combining the level predicates into a complete `Factor` declaration for archival:
```python
congruency = Factor("congruency", [
    DerivedLevel("congruent",   WithinTrial(congruent,   [color, word])),
    DerivedLevel("incongruent", WithinTrial(incongruent, [color, word])),
])

task_transition = Factor("task_transition", [
    DerivedLevel("repeat", Transition(task_repeat, [task])),
    DerivedLevel("switch", Transition(task_switch, [task])),
])
```

**(c) A compute function for post-hoc evaluation** — a string-returning function that applies the level predicates in order and returns the level name of the first predicate that evaluates to `True`. This is what the sandbox executes against the existing trial data:
```python
# Compute function (used for post-hoc evaluation, not SweetPea synthesis)
def compute_congruency(trial: dict) -> str:
    color, word = trial["color"], trial["word"]
    if congruent(color, word):   return "congruent"
    if incongruent(color, word): return "incongruent"
    raise ValueError("No level matched")

def compute_task_transition(prev: dict, curr: dict) -> str:
    task = [prev["task"], curr["task"]]  # [0]=prev, [-1]=curr
    if task_repeat(task): return "repeat"
    if task_switch(task): return "switch"
    raise ValueError("No level matched")
```

The prompts include worked examples of all three artifacts for both factor types, so the LLM can pattern-match reliably.

**Self-correction loop**: If the sandbox rejects the code (syntax error, runtime error, wrong return type, level name mismatch), the error is appended to the next LLM call. Up to `max_synthesis_retries` (default: 3) retries are allowed per candidate.

### 5.3 Sandboxed Code Execution and Factor Computation

Discovered factors are evaluated by **post-computing** their level values on the existing trial data — the candidate's compute function (artifact (c) from §5.2) is applied to each row (or each consecutive pair of rows for transition factors) to produce a new column. No SweetPea trial synthesis is re-run during discovery; that would require the SAT solver and could take up to 5 minutes. The sandbox only needs to execute plain Python, so a 10-second timeout is appropriate.

The SweetPea factor definition (artifact (b)) is saved alongside the compute function as a machine-readable record. It can be used in future work to generate new counterbalanced datasets that explicitly cross the discovered factor, but it plays no role in the current statistical evaluation.

**Sandbox execution harness:**
1. Serialize the DataFrame to JSON.
2. Write a harness script that imports the level predicates and compute function, then applies the compute function to every trial in participant order (grouping by `participant_id`, sorting by `trial_index`).
3. For transition factors: return `None` for `trial_index == 0` within each participant (no predecessor); all other rows receive a level string.
4. Run the harness with a 10-second timeout; capture stdout as a JSON-encoded list of level values aligned to the DataFrame row order.
5. Validate that all non-`null` values are strings matching the candidate's declared level names.

Two backends are supported (selectable via config):
- **Docker** (default): Uses `llm-sandbox` to execute code in a `python:3.9-slim` container, providing strong isolation from the host.
- **Subprocess** (fallback): Runs in a restricted subprocess with limited imports (`json`, `math`, `itertools`, `functools`, `re`). Adequate for research use when Docker is unavailable.

Any validation failure or timeout triggers the self-correction loop in §5.2.

### 5.4 Nested Logistic Regression Model Comparison

**Formula structure**: Uses `statsmodels.formula.api.logit` with `patsy` formula strings.

- Binary factors (2 levels): Encoded as float 0/1 column. One parameter added to the model.
- Multi-level factors (≥3 levels): Encoded with `C(factor_name)` treatment contrasts. `L-1` parameters added.

**Shared NaN mask**: Both null and alternative models are fit on the intersection of non-NaN rows across all columns referenced by either formula. This ensures the LRT degrees-of-freedom comparison is valid.

**LRT computation**:
```
statistic = −2 × (LLF_null − LLF_alt)
df        = n_params_alt − n_params_null
p-value   = chi2.sf(statistic, df=df)
```

**Separation detection**: If the alternative model fails to converge and any coefficient has `|z| > 10`, the candidate is rejected with `rejection_reason = "separation_detected"` rather than crashing.

**Formula progression across rounds:**
```
Start:           null = "correct ~ 1"
After round 1:   null = "correct ~ congruency"      (if congruency discovered)
After round 2:   null = "correct ~ congruency + task_transition"
```
Each newly discovered factor is added to the null model for all subsequent comparisons.

---

## 6. Evaluation

### 6.1 Factor Matching

A discovered factor D **matches** ground-truth factor G if:
- They are the same factor type (within-trial vs. transition), AND
- There exists a bijection φ between their level sets such that, for ≥95% of applicable trials (non-NaN rows), φ(D(t)) = G(t).

This allows the LLM to use arbitrary level names (e.g., "same"/"different" instead of "repeat"/"switch") while still receiving credit for the correct partition.

**Matching algorithm**:
1. Compute an agreement matrix: `agreement[i, j]` = maximum agreement rate over all bijections between GT factor `i` and discovered factor `j`.
2. Find maximum-weight bijection via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`).
3. Accept matched pairs with agreement rate ≥ 0.95 (configurable).

### 6.2 Metrics

| Metric | Definition |
|--------|------------|
| **Precision** | |accepted matches| / |discovered factors| |
| **Recall** | |accepted matches| / |ground truth factors| |
| **F1** | 2 × Precision × Recall / (Precision + Recall) |

Additional per-factor statistics are recorded:
- LRT p-value and statistic for each discovered factor
- Round in which each factor was discovered
- Number of synthesis retries required
- Whether any true factors were proposed but rejected (false negatives by type: statistical vs. synthesis failure)

---

## 7. Project Structure

```
experimental_design_search/
├── .venv/                             # Python virtual environment
├── config/
│   └── stroop_benchmark.yaml          # Full configuration (see §8)
├── data/
│   ├── ground_truth/
│   │   └── stroop_full.csv            # All factors including hidden ones
│   └── input/
│       └── stroop_input.csv           # Observable factors only
├── src/
│   ├── data_generation/
│   │   ├── sweetpea_builder.py        # SweetPea design (all factors) + trial sequence generation
│   │   └── stroop_model.py            # Accuracy sampling from ground-truth logistic model
│   ├── discovery/
│   │   ├── pipeline.py                # Discovery loop orchestration
│   │   ├── llm_client.py              # Anthropic SDK wrapper with retry
│   │   ├── candidate_generator.py     # LLM → structured factor proposals
│   │   ├── predicate_synthesizer.py   # LLM → Python predicate code
│   │   ├── sandbox.py                 # Subprocess/Docker sandboxed execution
│   │   └── factor_registry.py         # Tracks accepted/rejected factors, manages formula
│   ├── analysis/
│   │   ├── factor_encoder.py          # Predicate output → pandas Series
│   │   ├── model_comparison.py        # Logistic regression + LRT
│   │   └── evaluation.py              # Bijection matching + P/R/F1
│   └── utils/
│       ├── config.py                  # YAML loader + dataclass validation
│       └── logging_utils.py           # Structured JSON logging
├── prompts/
│   ├── candidate_generation_system.txt
│   ├── candidate_generation_user.txt
│   ├── predicate_synthesis_system.txt
│   └── predicate_synthesis_user.txt
├── results/
│   └── run_{timestamp}/
│       ├── discovered_factors.json
│       ├── round_{k}_candidates.json
│       └── evaluation_report.json
├── run_benchmark.py                   # CLI: python run_benchmark.py --config config/stroop_benchmark.yaml
├── requirements.txt
└── research_plan.md                   # This document
```

---

## 8. Configuration

All parameters are specified in `config/stroop_benchmark.yaml`:

```yaml
benchmark:
  name: "stroop_factor_discovery"
  seed: 42
  output_dir: "results"

data_generation:
  n_participants: 30
  n_trials_per_participant: 18        # one fully-crossed block (2×3×3)
  target_trials_per_participant: 198  # repeated to reach ~200 trials
  logistic_model:
    intercept: 0.5
    congruent: 0.8
    task_repeat: 0.4
    response_repeat: 0.3
    congruency_sequence: {cc: 0.2, ci: -0.3, ic: 0.1, ii: 0.0}

discovery:
  n_rounds: 4
  max_candidates_per_round: 8
  max_synthesis_retries: 3
  sandbox_timeout_seconds: 10
  sandbox_backend: "docker"           # "docker" | "subprocess"
  docker_image: "python:3.9-slim"

llm:
  model: "claude-sonnet-4-6"          # configurable; use claude-opus-4-8 for max capability
  max_tokens_candidate: 2000
  max_tokens_predicate: 1000
  candidate_temperature: 0.9
  predicate_temperature: 0.2

statistical:
  alpha: 0.05
  min_level_count: 5
  separation_check: true

evaluation:
  ground_truth_factors:
    - {name: congruency,           type: within_trial, levels: [congruent, incongruent]}
    - {name: task_transition,      type: transition,   levels: [repeat, switch]}
    # Uncomment to extend to full benchmark:
    # - {name: response_transition,  type: transition,   levels: [repeat, switch]}
    # - {name: congruency_sequence,  type: transition,   levels: [cc, ci, ic, ii]}
  bijection_threshold: 0.95
```

---

## 9. Software Dependencies

```
# requirements.txt
sweetpea>=0.35.0
anthropic>=0.20.0
statsmodels>=0.14.0
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
patsy>=0.5.6
llm-sandbox>=0.1.0        # Docker backend
```

**Environment setup:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Running the benchmark:**
```bash
python run_benchmark.py --config config/stroop_benchmark.yaml
```

---

## 11. Expected Results and Evaluation Criteria

**Success criteria (2-factor prototype):**
- Recall = 1.0: both `congruency` and `task_transition` discovered within 4 rounds
- Precision = 1.0: no false positives (only factors that genuinely improve model fit are registered)
- F1 = 1.0

**Partial success indicators:**
- Recall ≥ 0.5 (at least one of the two hidden factors discovered)
- Correct factor type (within-trial vs. transition) for each discovered factor
- LRT p-values < 0.001 for all registered factors

**Failure modes to analyze:**
- Predicate synthesis failures (syntax errors, wrong return types) — tracked per candidate
- Statistical false negatives (correct predicate synthesized but LRT not significant due to insufficient data)
- Statistical false positives (spurious factors passing LRT due to multiple comparisons)

**Future directions:**
- Extend to all 4 hidden factors (add `response_transition` and `congruency_sequence`)
- Test with other tasks (N-back, flanker, AX-CPT)
- Vary effect sizes and sample sizes to characterize statistical power
- Compare multiple LLMs (sonnet vs. opus vs. open-source) as the synthesis engine
- Investigate hierarchical discovery (discovered factors seeding higher-order factor proposals)

---

## 12. Key Design Decisions and Rationale

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Experiment design language | SweetPea | Declarative, maps directly to predicate synthesis; established in cognitive science |
| LLM for synthesis | Claude (Anthropic API) | Strong code generation; configurable per run |
| Sandbox | Docker via llm-sandbox | Security isolation for LLM-generated code; subprocess fallback for convenience |
| Statistical test | Logistic LRT | Appropriate for binary outcomes (accuracy); nested model comparison is principled and interpretable |
| Factor matching | Bijection via Hungarian algorithm | Handles arbitrary LLM level naming; exact matching with configurable tolerance |
| Trial generation | UniformCombinatoricSamplingStrategy | No Docker SAT server required; fully local and reproducible |
| Prototyping scope | 2 hidden factors | Enables rapid end-to-end validation before scaling to full complexity |

---

## References

- Gratton, G., Coles, M. G. H., & Donchin, E. (1992). Optimizing the use of information: Strategic control of activation of responses. *Journal of Experimental Psychology: General*, 121(4), 480–506.
- MacLeod, C. M. (1991). Half a century of research on the Stroop effect: An integrative review. *Psychological Bulletin*, 109(2), 163–203.
- Monsell, S. (2003). Task switching. *Trends in Cognitive Sciences*, 7(3), 134–140.
- Musslick, S., Cherkaev, A., Draut, B., Butt, A. S., Donnelly, P., Langlois, V., ... & Cohen, J. D. (2020). SweetPea: A standard language for factorial experimental design. *Behavior Research Methods*, 52, 2370–2395.
- Stroop, J. R. (1935). Studies of interference in serial verbal reactions. *Journal of Experimental Psychology*, 18(6), 643–662.

---

## Resources

Key tools used in this project, with pointers to documentation for implementation reference.

### SweetPea
Declarative language for factorial experiment design. Used for generating counterbalanced trial sequences and for formally expressing derived factor definitions.

- **Repository**: https://github.com/sweetpea-org/sweetpea-py
- **Documentation**: https://sweetpea-org.github.io/
- **API reference**: https://sweetpea-org.github.io/api/sweetpea.html
- **Factor & derivation guide**: https://sweetpea-org.github.io/guide/factorial_design.html

Key classes: `Factor`, `DerivedLevel`, `WithinTrial`, `Transition`, `fully_cross_block`, `synthesize_trials`, `UniformCombinatoricSamplingStrategy`.

### LLM Sandbox (`llm-sandbox`)
Docker-based sandboxed execution environment for running LLM-generated Python code safely. Used to execute synthesized predicate functions against the trial dataset.

- **Repository**: https://github.com/vndee/llm-sandbox
- **Documentation**: https://vndee.github.io/llm-sandbox/

Key usage pattern: `SandboxSession(lang="python")` as a context manager; `session.run(code, libraries=[...])` to execute code and capture stdout/stderr.

### Anthropic Python SDK
Used to call the Claude API for candidate factor generation and predicate code synthesis.

- **Repository**: https://github.com/anthropics/anthropic-sdk-python
- **API reference**: https://docs.anthropic.com/en/api/
