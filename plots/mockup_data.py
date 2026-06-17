"""Mockup data for all AutoFactor paper figures.

All values are plausible given existing results but are simulated.
Replace with real results before final paper submission.
"""

import numpy as np

rng = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Panel A: Synthetic benchmark P / R / F1
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARKS = ['Stroop-Simon', 'RDK\nTask-Switching', 'Prospect\nTheory']
N_GT_FACTORS = [4, 4, 7]   # ground-truth factor counts (from evaluation configs)

# (mean, SE) over 5 seeds
BENCHMARK_PRF = {
    'Stroop-Simon': {
        'precision': (0.83, 0.08),
        'recall':    (0.70, 0.10),
        'f1':        (0.75, 0.08),
    },
    'RDK\nTask-Switching': {
        'precision': (0.78, 0.11),
        'recall':    (0.44, 0.09),
        'f1':        (0.56, 0.09),
    },
    'Prospect\nTheory': {
        'precision': (0.60, 0.14),
        'recall':    (0.24, 0.08),
        'f1':        (0.34, 0.09),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Panel B: Per-factor level/correlation recovery
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (display_name, benchmark, factor_scope, factor_class, n_levels,
#              recovery_mean, recovery_se)
# Discrete rows → level recall (proportion of levels matched at ≥95%)
# Continuous rows → mean |Spearman ρ| against best-matching discovered factor
# Rows are ordered: within-trial discrete first, then window discrete,
# then within-trial continuous, then window continuous — within each benchmark.

FACTOR_ROWS = [
    # ── Stroop-Simon ──────────────────────────────────────────────────────────
    ('word–color congruency',        'Stroop-Simon',        'within_trial', 'discrete',   2, 0.98, 0.02),
    ('location–response congruency', 'Stroop-Simon',        'within_trial', 'discrete',   2, 0.74, 0.13),
    ('response transition',          'Stroop-Simon',        'window',       'discrete',   2, 0.52, 0.15),
    ('congruency (prev. trial)',      'Stroop-Simon',        'window',       'discrete',   2, 0.20, 0.10),

    # ── RDK Task-Switching ────────────────────────────────────────────────────
    ('stimulus difficulty',          'RDK\nTask-Switching', 'within_trial', 'continuous', 0, 0.88, 0.06),
    ('task transition',              'RDK\nTask-Switching', 'window',       'discrete',   2, 0.62, 0.14),
    ('n2 task inhibition',           'RDK\nTask-Switching', 'window',       'discrete',   3, 0.27, 0.10),
    ('past stimulus difficulty',     'RDK\nTask-Switching', 'window',       'continuous', 0, 0.44, 0.13),

    # ── Prospect Theory ───────────────────────────────────────────────────────
    ('dominance relation',           'Prospect\nTheory',    'within_trial', 'discrete',   3, 0.52, 0.15),
    ('value diff. transition',       'Prospect\nTheory',    'window',       'discrete',   2, 0.32, 0.13),
    ('expected value difference',    'Prospect\nTheory',    'within_trial', 'continuous', 0, 0.72, 0.11),
    ('loss difference',              'Prospect\nTheory',    'within_trial', 'continuous', 0, 0.54, 0.13),
    ('probability difference',       'Prospect\nTheory',    'within_trial', 'continuous', 0, 0.48, 0.12),
    ('gain difference',              'Prospect\nTheory',    'within_trial', 'continuous', 0, 0.36, 0.12),
    ('prev. EV difference',          'Prospect\nTheory',    'window',       'continuous', 0, 0.20, 0.09),
]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Panel C: 2×2 factor-type recall matrix
# ─────────────────────────────────────────────────────────────────────────────
# Aggregated over all 3 benchmarks and 5 seeds.
# Rows: Within-trial / Window; Cols: Discrete / Continuous

FACTORTYPE_MATRIX = np.array([
    [0.75, 0.60],   # within-trial: discrete, continuous
    [0.38, 0.33],   # window:       discrete, continuous
])

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Empirical dataset recovery (P / R / F1)
# ─────────────────────────────────────────────────────────────────────────────

EMPIRICAL_DATASETS = ['Stroop\n(N=466)', 'Janker et al.\n(N=80)', 'Dataset 3\n(TBD)']
N_EMPIRICAL_GT = [1, 2, 2]

EMPIRICAL_PRF = {
    'Stroop\n(N=466)': {
        'precision': (0.75, 0.14),
        'recall':    (0.90, 0.10),
        'f1':        (0.80, 0.11),
    },
    'Janker et al.\n(N=80)': {
        'precision': (0.67, 0.16),
        'recall':    (0.60, 0.15),
        'f1':        (0.62, 0.14),
    },
    'Dataset 3\n(TBD)': {
        'precision': (0.70, 0.15),
        'recall':    (0.55, 0.14),
        'f1':        (0.60, 0.13),
    },
}

# Synthetic reference F1 (closest matching complexity benchmark)
EMPIRICAL_SYNTHETIC_REF = {
    'Stroop\n(N=466)':      0.75,   # ← Stroop-Simon F1 (used as reference)
    'Janker et al.\n(N=80)': 0.56,  # ← RDK F1
    'Dataset 3\n(TBD)':      0.56,
}

# Per-factor recovery on empirical benchmark datasets
EMPIRICAL_FACTOR_ROWS = [
    # (display_name, dataset, scope, class, n_levels, mean, se)
    ('congruency',             'Stroop\n(N=466)',       'within_trial', 'discrete', 2, 0.90, 0.10),
    ('task switch cost',       'Janker et al.\n(N=80)', 'window',       'discrete', 2, 0.68, 0.16),
    ('response repetition',    'Janker et al.\n(N=80)', 'window',       'discrete', 2, 0.52, 0.18),
    ('factor A (TBD)',         'Dataset 3\n(TBD)',      'within_trial', 'discrete', 2, 0.72, 0.15),
    ('factor B (TBD)',         'Dataset 3\n(TBD)',      'window',       'discrete', 2, 0.48, 0.17),
]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Novel factor discoveries on empirical datasets
# ─────────────────────────────────────────────────────────────────────────────

def _make_participant_means(group_means, n_participants, within_sd, between_sd, seed=0):
    """Simulate participant-level means for each condition."""
    r = np.random.default_rng(seed)
    participant_offsets = r.normal(0, between_sd, n_participants)
    result = []
    for gm in group_means:
        means = gm + participant_offsets + r.normal(0, within_sd, n_participants)
        result.append(means)
    return result


NOVEL_DISCOVERIES = [
    {
        'factor_name':      'Ink Color Repetition',
        'dataset':          'Stroop (N = 466)',
        'outcome_label':    'log RT (ms)',
        'levels':           ['repeat', 'switch'],
        'group_means':      [6.505, 6.600],
        'group_ci95':       [0.006, 0.006],
        'participant_data': _make_participant_means(
            [6.505, 6.600], n_participants=80,
            within_sd=0.025, between_sd=0.06, seed=1),
        'delta_ll':         13.2,
        'n':                466,
        'literature_ref':   'cf. Mayr et al., 2003',
        'code': (
            "def compute_factor(w):\n"
            "    a = w[0]['color']\n"
            "    b = w[1]['color']\n"
            "    if a == b:\n"
            "        return 'repeat'\n"
            "    return 'switch'"
        ),
    },
    {
        'factor_name':      'Response Repetition Benefit',
        'dataset':          'Janker et al. (N = 80)',
        'outcome_label':    'Accuracy (%)',
        'levels':           ['repeat', 'switch'],
        'group_means':      [83.5, 74.2],
        'group_ci95':       [1.8,  2.0],
        'participant_data': _make_participant_means(
            [83.5, 74.2], n_participants=60,
            within_sd=2.5, between_sd=5.0, seed=2),
        'delta_ll':         9.4,
        'n':                80,
        'literature_ref':   None,
        'code': (
            "def compute_factor(w):\n"
            "    a = w[0]['response']\n"
            "    b = w[1]['response']\n"
            "    if a == b:\n"
            "        return 'repeat'\n"
            "    return 'switch'"
        ),
    },
    {
        'factor_name':      'Task × Difficulty Interaction',
        'dataset':          'Dataset 3 (TBD)',
        'outcome_label':    'Accuracy (%)',
        'levels':           ['easy\nrepeat', 'hard\nrepeat',
                             'easy\nswitch', 'hard\nswitch'],
        'group_means':      [88.0, 72.0, 80.0, 62.0],
        'group_ci95':       [2.0,  2.5,  2.2,  2.8],
        'participant_data': _make_participant_means(
            [88.0, 72.0, 80.0, 62.0], n_participants=50,
            within_sd=3.5, between_sd=5.0, seed=3),
        'delta_ll':         7.1,
        'n':                None,
        'literature_ref':   None,
        'code': (
            "# interaction (effect search)\n"
            "C(task_transition)\n"
            "    : C(difficulty)"
        ),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Ablation study
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_CONDITIONS = [
    'Full system',
    'w/o LLM seeding',
    'w/o evolution',
    'w/o novelty bonus',
    '+ complexity penalty',
    'w/o interaction search',
]

# (mean F1, SE) per condition × benchmark
ABLATION_DATA = {
    'Stroop-Simon': [
        (0.75, 0.08),   # Full system
        (0.62, 0.10),   # w/o LLM seeding
        (0.55, 0.11),   # w/o evolution
        (0.70, 0.09),   # w/o novelty bonus
        (0.68, 0.09),   # + complexity penalty
        (0.65, 0.09),   # w/o interaction search
    ],
    'RDK\nTask-Switching': [
        (0.56, 0.09),
        (0.34, 0.11),
        (0.28, 0.10),
        (0.52, 0.10),
        (0.40, 0.10),
        (0.48, 0.10),
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Search efficiency curves
# ─────────────────────────────────────────────────────────────────────────────

def _efficiency_curve(n_calls, target_f1, shape, noise_sd=0.03, seed=0):
    """Simulate rolling-max F1 curve as a function of cumulative LLM calls."""
    r = np.random.default_rng(seed)
    x = np.arange(0, n_calls + 1)
    # logistic growth toward target_f1
    raw = target_f1 / (1 + np.exp(-shape * (x - n_calls * 0.35)))
    noise = r.normal(0, noise_sd, len(x))
    curve = np.clip(np.maximum.accumulate(raw + noise), 0, target_f1)
    return x, curve


N_SEEDS = 5
N_CALLS = 120

EFF_BENCHMARKS = ['Stroop-Simon', 'RDK\nTask-Switching', 'Prospect\nTheory']

EFFICIENCY_DATA = {}
for bench, (llm_shape, rnd_shape, llm_target, rnd_target) in zip(
    EFF_BENCHMARKS,
    [(0.08, 0.04, 0.75, 0.62),   # SS: LLM faster
     (0.07, 0.03, 0.56, 0.42),   # RDK
     (0.06, 0.025, 0.34, 0.22)], # PT
):
    llm_curves, rnd_curves = [], []
    for s in range(N_SEEDS):
        _, c = _efficiency_curve(N_CALLS, llm_target, llm_shape, seed=s)
        llm_curves.append(c)
        _, c = _efficiency_curve(N_CALLS, rnd_target, rnd_shape, seed=s + 100)
        rnd_curves.append(c)
    EFFICIENCY_DATA[bench] = {
        'x':          np.arange(0, N_CALLS + 1),
        'llm_mean':   np.mean(llm_curves, axis=0),
        'llm_se':     np.std(llm_curves, axis=0) / np.sqrt(N_SEEDS),
        'rnd_mean':   np.mean(rnd_curves, axis=0),
        'rnd_se':     np.std(rnd_curves, axis=0) / np.sqrt(N_SEEDS),
    }
