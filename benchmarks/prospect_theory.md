# Prospect-Theory Risky Choice Benchmark

## Task description

On each trial, two monetary gambles are displayed side-by-side (left and right).
Each gamble has:
- a **gain amount** (the prize if the gain outcome occurs),
- a **loss amount** (the cost if the loss outcome occurs), and
- a **gain probability** (the probability of the gain outcome; the loss probability
  is implicit: `loss_prob = 1 − gain_prob`).

Participants choose which gamble they prefer (`chose_left = 1` for left, `0` for right).
All base factors are manipulable design variables; participant choices and received
outcomes are not modelled.

## Experimental design

All six base factors are continuous.  Data are generated using SweetPea
`ContinuousFactor` with `UniformDistribution`.  Because there are no categorical factors
to counterbalance, a dummy categorical factor with `n_trials` levels is used as the
crossing scaffold, yielding exactly `n_trials` independent trials per participant with
no block structure.

Each participant's trial sequence is fully interleaved (no blocks); transition factors
are computed within the full participant sequence using pandas shifts.

Base factor sampling ranges:

| Factor | Distribution |
|---|---|
| `left_gain`, `right_gain` | Uniform(0, 100) |
| `left_loss`, `right_loss` | Uniform(0, 100) |
| `left_gain_probability`, `right_gain_probability` | Uniform(0.05, 0.95) |

### Base factors (observable — exposed to the discovery pipeline)

| Factor | Type | Range |
|---|---|---|
| `left_gain` | continuous | [0, 100] |
| `left_loss` | continuous | [0, 100] |
| `left_gain_probability` | continuous | [0.05, 0.95] |
| `right_gain` | continuous | [0, 100] |
| `right_loss` | continuous | [0, 100] |
| `right_gain_probability` | continuous | [0.05, 0.95] |

### Hidden derived factors (not exposed — ground truth for evaluation)

| Factor | Scope | Type | Definition |
|---|---|---|---|
| `left_expected_value` | within-trial | continuous | `p_L × g_L − (1−p_L) × l_L` |
| `right_expected_value` | within-trial | continuous | `p_R × g_R − (1−p_R) × l_R` |
| `expected_value_difference` | within-trial | continuous | `left_EV − right_EV` |
| `gain_difference` | within-trial | continuous | `left_gain − right_gain` |
| `loss_difference` | within-trial | continuous | `left_loss − right_loss` |
| `probability_difference` | within-trial | continuous | `left_gain_prob − right_gain_prob` |
| `dominance_relation` | within-trial | discrete | `left_dominates`, `right_dominates`, or `no_dominance` (see below) |
| `previous_expected_value_difference` | window (width=2) | continuous | `expected_value_difference` on the previous trial.  NaN on trial 1. |
| `value_difference_transition` | window (width=2) | discrete | `repeat` if the objectively better side is the same as the previous trial; `switch` otherwise.  NaN on trial 1. |

#### `dominance_relation` derivation

The left gamble dominates when it has at least as high a gain, at least as low a loss,
and at least as high a gain probability as the right gamble, with at least one strict
advantage.  Right dominance is defined symmetrically.  Otherwise `no_dominance`.

#### `value_difference_transition` derivation

```
sign(t) = "left_better"  if expected_value_difference(t) > 0
          "right_better" if expected_value_difference(t) < 0
          "tie"          if expected_value_difference(t) == 0

value_difference_transition(t) = "repeat" if sign(t) == sign(t−1)
                                  "switch" otherwise
```

## Ground-truth statistical model

The ground-truth model drives `P(chose_left)`.  All continuous predictors are z-scored
(mean=0, sd=1) before the coefficient is applied so that effect sizes are comparable.

```
logit(P(chose_left)) = 0.0
    + 0.8 × z(expected_value_difference)              [EV advantage for left]
    − 0.4 × z(loss_difference)                        [loss aversion: higher left loss → fewer left choices]
    + 0.3 × z(probability_difference)                 [probability sensitivity]
    + 1.0 × I(dominance_relation == "left_dominates") [stochastic dominance pulls toward left]
    − 1.0 × I(dominance_relation == "right_dominates")[stochastic dominance pulls toward right]
    − 0.2 × I(value_difference_transition == "switch") [choice consistency penalty after side-switch]
```

### Collinearity note

`expected_value_difference`, `gain_difference`, `loss_difference`, and
`probability_difference` are not independent (EV is a weighted combination of the
other three).  The ground-truth model includes only `expected_value_difference` as the
primary EV term, plus `loss_difference` and `probability_difference` as independent
additions to capture loss aversion and probability weighting beyond expected value.

A post-generation correlation check warns when any pairwise `|r|` among the four
continuous derived factors exceeds 0.5.

## Discovery challenge

The pipeline observes only the six continuous base factor columns plus `chose_left`.
It must discover:

1. `expected_value_difference` — a continuous within-trial factor combining all six base inputs.
2. `gain_difference` — a simpler continuous within-trial factor.
3. `loss_difference` — a signed loss contrast, needed to model loss aversion independent of EV.
4. `probability_difference` — a probability sensitivity factor.
5. `dominance_relation` — a discrete within-trial factor encoding stochastic dominance.
6. `previous_expected_value_difference` — a continuous transition factor (EV on the prior trial).
7. `value_difference_transition` — a discrete transition factor encoding side-consistency.

This benchmark is qualitatively different from the accuracy benchmarks:
- The outcome is **choice**, not accuracy.
- All base factors are **continuous**; none are categorical.
- All hidden derived factors are **within-trial or window computed from continuous inputs**.
- The pipeline must enable `allowed_factor_classes: ["discrete", "continuous"]` to discover continuous factors.

## Config file

`config/prospect_theory_benchmark.yaml`
