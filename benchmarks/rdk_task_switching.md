# RDK Task-Switching Benchmark

## Task description

On each trial, a random-dot kinematogram (RDK) stimulus is presented.  The stimulus
varies simultaneously along three dimensions: motion direction, colour, and orientation.
A task cue instructs the participant to classify one of the three dimensions:

| Task | Instruction |
|---|---|
| `motion` | Is the net motion direction **up** or **down**? |
| `color` | Is the dominant colour **blue** or **red**? |
| `orientation` | Is the dominant orientation **left** or **right**? |

The correct response is fully determined by the task cue and the relevant stimulus
feature.  The other two dimensions are irrelevant and must be ignored.

Motion, colour, and orientation coherence values (0–1) reflect how strongly the
stimulus is biased toward one level; higher coherence means easier discrimination.

Accuracy (correct / incorrect) is the binary outcome.

## Experimental design

Data are generated using SweetPea.  The **crossing** is over
`task × motion × color × orientation`
(3 × 2 × 2 × 2 = **24 trials per block**).

Coherence values (`motion_coherence`, `color_coherence`, `orientation_coherence`) are
sampled independently per trial from `Uniform(0, 1)` via SweetPea `ContinuousFactor`.

### Base factors (observable — exposed to the discovery pipeline)

| Factor | Type | Levels / Range |
|---|---|---|
| `task` | categorical | `motion`, `color`, `orientation` |
| `motion` | categorical | `up`, `down` |
| `color` | categorical | `blue`, `red` |
| `orientation` | categorical | `left`, `right` |
| `motion_coherence` | continuous | [0, 1] |
| `color_coherence` | continuous | [0, 1] |
| `orientation_coherence` | continuous | [0, 1] |
| `correct_response` | categorical | `left`, `right` |

`correct_response` is derived from `task` and the relevant stimulus feature.  It is
available in the trial dict so the LLM can use it when synthesising derived factors,
but it is **excluded from the regression baseline** because it is a deterministic
function of the other base factors (including it would create a rank-deficient design
matrix).

### Hidden derived factors (not exposed — ground truth for evaluation)

| Factor | Scope | Type | Definition |
|---|---|---|---|
| `task_transition` | transition (width=2) | discrete | Task repeated (`repeat`) or switched (`switch`) from the previous trial. |
| `current_stimulus_difficulty` | within-trial | continuous | `1 − coherence_of_current_task`. High values → harder. |
| `past_stimulus_difficulty` | window (width=2) | continuous | `1 − coherence_of_previous_task`.  NaN on the first trial of each block. |
| `n2_task_inhibition` | window (width=3) | discrete | `aba_return` if the current task equals the task 2-back (and differs from 1-back); `cba_nonreturn` if all three tasks differ; `other` otherwise. |

Implementation note: `past_stimulus_difficulty` is block-local, matching SweetPea
window semantics. Earlier generated RDK data shifted this value across an entire
participant sequence, which contradicted the benchmark specification above by carrying
difficulty across block boundaries.

#### `current_stimulus_difficulty` derivation

```
if task == "motion":      difficulty = 1 − motion_coherence
if task == "color":       difficulty = 1 − color_coherence
if task == "orientation": difficulty = 1 − orientation_coherence
```

#### `n2_task_inhibition` derivation

```
window = [task_t−2, task_t−1, task_t]

aba_return    if task_t == task_t−2  and  task_t != task_t−1
cba_nonreturn if task_t != task_t−2  and  task_t != task_t−1  and  task_t−1 != task_t−2
other         otherwise
```

The key comparison is `aba_return` vs `cba_nonreturn`: backward inhibition theory
predicts worse performance for `aba_return` because the current task was recently
inhibited.

## Ground-truth statistical model

```
logit(P(correct)) = 0.5
    + 0.5 × I(task_transition == "repeat")           [task-repetition benefit]
    − 0.8 × current_stimulus_difficulty              [difficulty effect; z-scored]
    − 0.2 × past_stimulus_difficulty                 [carry-over difficulty; z-scored]
    − 0.4 × I(n2_task_inhibition == "aba_return")    [backward inhibition]
```

Continuous predictors are z-scored (mean=0, sd=1) before the coefficient is applied.
NaN values contribute 0.

## Discovery challenge

The pipeline observes `task`, `motion`, `color`, `orientation`, the three coherence
factors, `correct_response`, and `correct`.  It must recover:

1. `task_transition` — a discrete transition factor (the classic task-switching cost).
2. `current_stimulus_difficulty` — a continuous within-trial factor that selectively reads the coherence value associated with the current task.
3. `past_stimulus_difficulty` — a continuous window factor (the difficulty carry-over).
4. `n2_task_inhibition` — a three-level window factor requiring a 3-trial history.

This benchmark tests the pipeline's ability to discover both discrete and continuous
hidden factors, and factors that require multi-trial windows (width 2 and 3).

## Config file

`config/synthetic_rdk_task_switching_benchmark.yaml`
