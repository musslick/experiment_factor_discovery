# Stroop Benchmark

## Task description

On each trial, participants see a colour word (e.g. "red", "blue", "green") printed in
coloured ink.  They must perform one of two tasks:

| Task | Instruction |
|---|---|
| `color_naming` | Report the **ink colour**; ignore the word meaning. |
| `word_reading` | Report the **word meaning**; ignore the ink colour. |

The motor response is fully determined by the task and the relevant stimulus attribute.
Accuracy (correct / incorrect) is the binary outcome.

## Experimental design

Data are generated using SweetPea.  The **crossing** is over
`task × color × word` (2 × 3 × 3 = **18 trials per block**).

### Base factors (observable — exposed to the discovery pipeline)

| Factor | Type | Levels |
|---|---|---|
| `task` | categorical | `color_naming`, `word_reading` |
| `color` | categorical | `red`, `blue`, `green` |
| `word` | categorical | `red`, `blue`, `green` |

### Hidden derived factors (not exposed — ground truth for evaluation)

| Factor | Scope | Levels | Definition |
|---|---|---|---|
| `congruency` | within-trial | `congruent`, `incongruent` | `color == word` |
| `task_transition` | transition (width=2) | `repeat`, `switch` | Task is the same as on the previous trial. |

## Ground-truth statistical model

```
logit(P(correct)) = 0.5
    + 0.8 × I(congruency == "congruent")
    + 0.4 × I(task_transition == "repeat")
```

NaN transition values (first trial of each block) contribute 0 to the linear predictor.

## Discovery challenge

The pipeline observes only `task`, `color`, `word`, and `correct`.  It must discover:

1. `congruency` — a within-trial factor reflecting whether the ink colour matches the word meaning.
2. `task_transition` — a transition factor reflecting whether the task repeated or switched from the previous trial.

Both factors carry unique predictive signal for accuracy; neither is recoverable by simply reading the observable columns.

## Config file

`config/synthetic_stroop_benchmark.yaml`
