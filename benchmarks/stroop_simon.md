# Stroop–Simon Benchmark

## Task description

On each trial, a colour word is presented at one of three screen locations.
Participants must respond only to the **ink colour** of the word, ignoring both the
word meaning and the stimulus location.

Response mapping (determined entirely by ink colour):

| Ink colour | Correct response |
|---|---|
| `red`   | `left`   |
| `blue`  | `middle` |
| `green` | `right`  |

Accuracy (correct / incorrect) is the binary outcome.

## Experimental design

Data are generated using SweetPea.  The **crossing** is over
`word × color × stimulus_location` (3 × 3 × 3 = **27 trials per block**).

### Base factors (observable — exposed to the discovery pipeline)

| Factor | Type | Levels |
|---|---|---|
| `word` | categorical | `red`, `blue`, `green` |
| `color` | categorical | `red`, `blue`, `green` |
| `stimulus_location` | categorical | `left`, `middle`, `right` |
| `correct_response` | categorical | `left`, `middle`, `right` |

`correct_response` is derived deterministically from `color` (red→left, blue→middle,
green→right) and is included in the observable input because it is a rule-level base
factor.

### Hidden derived factors (not exposed — ground truth for evaluation)

| Factor | Scope | Levels | Definition |
|---|---|---|---|
| `word_color_congruency` | within-trial | `congruent`, `incongruent` | `word == color` |
| `location_response_congruency` | within-trial | `congruent`, `incongruent` | `stimulus_location == correct_response` |
| `congruency_previous_trial` | transition (width=2) | `congruent`, `incongruent` | `word_color_congruency` on the previous trial |
| `response_transition` | transition (width=2) | `repeat`, `switch` | `correct_response` is the same as on the previous trial |

## Ground-truth statistical model

```
logit(P(correct)) = 0.5
    + 0.8 × I(word_color_congruency == "congruent")       [Stroop effect]
    + 0.6 × I(location_response_congruency == "congruent") [Simon effect]
    + 0.3 × I(congruency_previous_trial == "congruent")    [congruency sequence effect]
    + 0.2 × I(response_transition == "repeat")             [response repetition benefit]
```

NaN transition values (first trial of each block) contribute 0.

## Discovery challenge

The pipeline observes `word`, `color`, `stimulus_location`, `correct_response`, and
`correct`.  It must recover four hidden factors:

1. `word_color_congruency` — the classic Stroop congruency contrast (within-trial).
2. `location_response_congruency` — the Simon spatial compatibility contrast (within-trial).
3. `congruency_previous_trial` — the previous trial's Stroop congruency, needed to model congruency sequence effects (CSE).
4. `response_transition` — whether the motor response repeated, capturing response repetition / priming effects.

This benchmark is harder than the basic Stroop because (a) there are four targets instead
of two, (b) two are within-trial and two are transition factors, and (c) two factors share
the same level labels (`congruent` / `incongruent`), requiring the evaluator to use bijection
matching rather than name matching.

## Config file

`config/synthetic_stroop_simon_benchmark.yaml`
