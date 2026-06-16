"""Figure 3 — Empirical Dataset Recovery.

Two-panel figure:
  A  P/R/F1 per empirical dataset, with synthetic reference line
  B  Per-factor recovery on known ground-truth factors
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
import style
from mockup_data import (
    EMPIRICAL_DATASETS, N_EMPIRICAL_GT,
    EMPIRICAL_PRF, EMPIRICAL_SYNTHETIC_REF,
    EMPIRICAL_FACTOR_ROWS,
)

style.apply_style()

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(style.W2, 3.8),
    gridspec_kw={'width_ratios': [2.8, 3.6], 'wspace': 0.42})


# ═══════════════════════════════════════════════════════════════════════════════
# Panel A — P / R / F1 per empirical dataset
# ═══════════════════════════════════════════════════════════════════════════════

metrics       = ['precision', 'recall', 'f1']
metric_labels = ['Precision', 'Recall', 'F1']
n_ds    = len(EMPIRICAL_DATASETS)
bar_w   = 0.22
x_centers = np.arange(n_ds) * 0.90

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    offset = (i - 1) * bar_w
    means = [EMPIRICAL_PRF[d][metric][0] for d in EMPIRICAL_DATASETS]
    ses   = [EMPIRICAL_PRF[d][metric][1] for d in EMPIRICAL_DATASETS]
    ax_a.bar(x_centers + offset, means, bar_w,
             color=style.METRIC_COLORS[metric], label=label,
             yerr=ses, error_kw=dict(lw=0.8, capsize=2.5, capthick=0.8),
             zorder=3)

# Synthetic reference lines (one per dataset)
for xi, ds in zip(x_centers, EMPIRICAL_DATASETS):
    ref = EMPIRICAL_SYNTHETIC_REF[ds]
    ax_a.plot([xi - 0.42, xi + 0.42], [ref, ref],
              color=style.REF_LINE_COLOR, lw=1.2, ls='--', zorder=4)

# Invisible proxy for reference line in legend
ref_line = plt.Line2D([0], [0], color=style.REF_LINE_COLOR,
                      lw=1.2, ls='--', label='Synthetic reference (F1)')

# Annotate N ground-truth factors
for xi, n in zip(x_centers, N_EMPIRICAL_GT):
    ax_a.text(xi, 1.04, f'{n} factor{"s" if n > 1 else ""}',
              ha='center', va='bottom',
              fontsize=style.FS_ANNOT, color='#555555')

ax_a.set_xticks(x_centers)
ax_a.set_xticklabels(EMPIRICAL_DATASETS, fontsize=style.FS_TICK)
ax_a.set_ylabel('Score', fontsize=style.FS_LABEL)
ax_a.set_ylim(0, 1.12)
ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

handles, labels = ax_a.get_legend_handles_labels()
ax_a.legend(handles + [ref_line], labels + ['Synthetic ref. (F1)'],
            loc='lower left', fontsize=style.FS_ANNOT,
            frameon=False, handlelength=1.2)

style.despine(ax_a)
style.panel_label(ax_a, 'A')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel B — Per-factor recovery on empirical known factors
# ═══════════════════════════════════════════════════════════════════════════════

n_factors = len(EMPIRICAL_FACTOR_ROWS)
y_positions = np.arange(n_factors)

for i, row in enumerate(EMPIRICAL_FACTOR_ROWS):
    name, ds, scope, fclass, n_lvl, mean, se = row
    color = style.FACTOR_COLORS[f'{scope}_{fclass}']
    is_continuous = (fclass == 'continuous')

    ax_b.barh(i, mean, height=0.55,
              color=color, alpha=0.90, zorder=3)

    if is_continuous:
        ax_b.barh(i, mean, height=0.55,
                  color='none', hatch=style.CONTINUOUS_HATCH,
                  edgecolor=color, linewidth=0.0, zorder=4)

    ax_b.errorbar(mean, i, xerr=se, fmt='none',
                  color='#444444', lw=0.8, capsize=2.5, capthick=0.8,
                  zorder=5)

ax_b.set_yticks(y_positions)
ax_b.set_yticklabels([r[0] for r in EMPIRICAL_FACTOR_ROWS], fontsize=6.5)
ax_b.invert_yaxis()

# Dataset section separators and labels
dataset_groups = []
prev_ds, start_i = EMPIRICAL_FACTOR_ROWS[0][1], 0
for i, row in enumerate(EMPIRICAL_FACTOR_ROWS[1:], start=1):
    if row[1] != prev_ds:
        dataset_groups.append((prev_ds, start_i, i - 1))
        prev_ds, start_i = row[1], i
dataset_groups.append((prev_ds, start_i, len(EMPIRICAL_FACTOR_ROWS) - 1))

for ds_name, start, end in dataset_groups:
    if start > 0:
        ax_b.axhline(start - 0.5, color='#BBBBBB', lw=0.8, zorder=0)
    ax_b.text(1.04, (start + end) / 2.0,
              ds_name.replace('\n', ' '),
              transform=style.blended(ax_b),
              ha='left', va='center',
              fontsize=style.FS_ANNOT, fontweight='bold', color='#444444',
              clip_on=False)

# Full-recovery threshold line
ax_b.axvline(0.95, color='#999999', lw=0.8, ls=':', zorder=1)
ax_b.text(0.95, -0.6, '0.95', ha='center', va='top',
          fontsize=style.FS_ANNOT, color='#999999')

ax_b.set_xlabel('Recovery score  (level recall  |  |Spearman ρ|)',
                fontsize=style.FS_LABEL)
ax_b.set_xlim(0, 1.0)
ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

style.despine(ax_b, left=True)
ax_b.spines['left'].set_visible(False)
ax_b.tick_params(left=False)
style.panel_label(ax_b, 'B', x=-0.04)

fig.text(0.5, 0.5, 'MOCKUP DATA', ha='center', va='center',
         fontsize=28, color='lightgray', alpha=0.35,
         rotation=30, transform=fig.transFigure, zorder=0)

out_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'figure3.png'))
fig.savefig(os.path.join(out_dir, 'figure3.pdf'))
print('Saved figure3.png / figure3.pdf')
plt.show()
