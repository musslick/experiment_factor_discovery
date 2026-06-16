"""Figure 2 — Synthetic Benchmark Recovery.

Three-panel figure:
  A  Overall P/R/F1 per benchmark (grouped bar chart)
  B  Per-factor level/correlation recovery (horizontal bars)
  C  2×2 factor-type recall matrix (heatmap)

Layout: A and C share the left column; B spans the full right column.
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
    BENCHMARKS, N_GT_FACTORS, BENCHMARK_PRF,
    FACTOR_ROWS, FACTORTYPE_MATRIX,
)

style.apply_style()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar_color(scope, fclass):
    return style.FACTOR_COLORS[f'{scope}_{fclass}']


# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(style.W2, 8.2))

outer = gridspec.GridSpec(1, 2, figure=fig,
                          width_ratios=[2.6, 4.2], wspace=0.42)

left = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=outer[0],
    height_ratios=[1.6, 1.6], hspace=0.65)

ax_a = fig.add_subplot(left[0])
ax_c = fig.add_subplot(left[1])
ax_b = fig.add_subplot(outer[1])


# ═══════════════════════════════════════════════════════════════════════════════
# Panel A — Benchmark P / R / F1
# ═══════════════════════════════════════════════════════════════════════════════

metrics = ['precision', 'recall', 'f1']
metric_labels = ['Precision', 'Recall', 'F1']
n_bench = len(BENCHMARKS)
n_metrics = len(metrics)
bar_w = 0.22
group_gap = 0.85
x_centers = np.arange(n_bench) * group_gap

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    offset = (i - 1) * bar_w
    means = [BENCHMARK_PRF[b][metric][0] for b in BENCHMARKS]
    ses   = [BENCHMARK_PRF[b][metric][1] for b in BENCHMARKS]
    ax_a.bar(x_centers + offset, means, bar_w,
             color=style.METRIC_COLORS[metric], label=label,
             yerr=ses, error_kw=dict(lw=0.8, capsize=2.5, capthick=0.8),
             zorder=3)

# Annotate number of ground-truth factors above each group
for xi, n in zip(x_centers, N_GT_FACTORS):
    ax_a.text(xi, 1.05, f'{n} factors', ha='center', va='bottom',
              fontsize=style.FS_ANNOT, color='#555555', clip_on=False)

ax_a.set_xticks(x_centers)
ax_a.set_xticklabels(BENCHMARKS, fontsize=style.FS_TICK)
ax_a.set_ylabel('Score', fontsize=style.FS_LABEL)
ax_a.set_ylim(0, 1.12)
ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax_a.legend(loc='upper right', fontsize=style.FS_ANNOT,
            frameon=False, ncol=1, handlelength=1.0)
style.despine(ax_a)
style.panel_label(ax_a, 'A')


# ═══════════════════════════════════════════════════════════════════════════════
# Panel C — 2×2 Factor-type recall matrix
# ═══════════════════════════════════════════════════════════════════════════════

im = ax_c.imshow(FACTORTYPE_MATRIX, aspect='auto',
                 cmap='Blues', vmin=0, vmax=1,
                 interpolation='nearest')

# Cell value annotations
for row in range(2):
    for col in range(2):
        val = FACTORTYPE_MATRIX[row, col]
        text_color = 'white' if val > 0.55 else '#333333'
        ax_c.text(col, row, f'{val:.2f}',
                  ha='center', va='center',
                  fontsize=style.FS_LABEL, color=text_color,
                  fontweight='bold')

ax_c.set_xticks([0, 1])
ax_c.set_xticklabels(['Discrete', 'Continuous'], fontsize=style.FS_TICK)
ax_c.set_yticks([0, 1])
ax_c.set_yticklabels(['Within-trial', 'Window'], fontsize=style.FS_TICK)
ax_c.set_xlabel('Factor class', fontsize=style.FS_LABEL, labelpad=4)
ax_c.set_ylabel('Factor scope', fontsize=style.FS_LABEL, labelpad=4)
ax_c.tick_params(length=0)
for sp in ax_c.spines.values():
    sp.set_visible(False)

# Colorbar
cbar = fig.colorbar(im, ax=ax_c, orientation='horizontal',
                    fraction=0.07, pad=0.22, shrink=0.85)
cbar.ax.tick_params(labelsize=style.FS_ANNOT)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_label('Recovery score', fontsize=style.FS_ANNOT, labelpad=3)

style.panel_label(ax_c, 'C', x=-0.16)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel B — Per-factor level / correlation recovery
# ═══════════════════════════════════════════════════════════════════════════════

n_factors = len(FACTOR_ROWS)
y_positions = np.arange(n_factors)

# Plot bars bottom-to-top so first row (word-color congruency) is at the top
# after inverting y-axis.
for i, row in enumerate(FACTOR_ROWS):
    name, bench, scope, fclass, n_lvl, mean, se = row
    y = i
    color = _bar_color(scope, fclass)
    is_continuous = (fclass == 'continuous')

    ax_b.barh(y, mean, height=0.58,
              color=color, alpha=0.90, zorder=3)

    if is_continuous:
        ax_b.barh(y, mean, height=0.58,
                  color='none', hatch=style.CONTINUOUS_HATCH,
                  edgecolor=color, linewidth=0.0, zorder=4)

    ax_b.errorbar(mean, y, xerr=se, fmt='none',
                  color='#444444', lw=0.8, capsize=2.5, capthick=0.8,
                  zorder=5)

# Y-axis: factor labels
ax_b.set_yticks(y_positions)
ax_b.set_yticklabels([r[0] for r in FACTOR_ROWS], fontsize=6.5)
ax_b.invert_yaxis()

# Benchmark section headers + separators
benchmark_groups = [
    ('Stroop-Simon',        0,  3),
    ('RDK Task-Switching',  4,  7),
    ('Prospect Theory',     8, 14),
]
sep_color = '#BBBBBB'
header_x = 1.06   # just outside the right edge of the plot in axis coords

for bench_name, start, end in benchmark_groups:
    mid_data = (start + end) / 2.0

    # Separator line above each group (except the very first)
    if start > 0:
        ax_b.axhline(start - 0.5, color=sep_color, lw=0.8, zorder=0,
                     xmin=0, xmax=1)

    # Benchmark label — placed to the right of the bars
    ax_b.text(header_x, mid_data,
              bench_name,
              transform=style.blended(ax_b),
              ha='left', va='center',
              fontsize=style.FS_ANNOT, fontweight='bold', color='#444444',
              rotation=0, clip_on=False)

ax_b.set_xlabel(
    'Recovery score  (level recall  |  |Spearman ρ|)',
    fontsize=style.FS_LABEL)
ax_b.set_xlim(0, 1.0)
ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_b.axvline(0, color='#999999', lw=0.6, zorder=0)

style.despine(ax_b, left=True)
ax_b.spines['left'].set_visible(False)
ax_b.tick_params(left=False)

# Legend for factor types
legend_handles = [
    mpatches.Patch(facecolor=style.FACTOR_COLORS['within_trial_discrete'],
                   label='Within-trial, discrete'),
    mpatches.Patch(facecolor=style.FACTOR_COLORS['window_discrete'],
                   label='Window, discrete'),
    mpatches.Patch(facecolor=style.FACTOR_COLORS['within_trial_continuous'],
                   hatch=style.CONTINUOUS_HATCH,
                   edgecolor=style.FACTOR_COLORS['within_trial_continuous'],
                   label='Within-trial, continuous (|ρ|)'),
    mpatches.Patch(facecolor=style.FACTOR_COLORS['window_continuous'],
                   hatch=style.CONTINUOUS_HATCH,
                   edgecolor=style.FACTOR_COLORS['window_continuous'],
                   label='Window, continuous (|ρ|)'),
]
ax_b.legend(handles=legend_handles, loc='lower right',
            fontsize=style.FS_ANNOT, frameon=False,
            handlelength=1.2, handleheight=0.9)

style.panel_label(ax_b, 'B', x=-0.04)


# ── Mockup watermark ──────────────────────────────────────────────────────────
fig.text(0.5, 0.5, 'MOCKUP DATA', ha='center', va='center',
         fontsize=28, color='lightgray', alpha=0.35,
         rotation=30, transform=fig.transFigure, zorder=0)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'figure2.png'))
fig.savefig(os.path.join(out_dir, 'figure2.pdf'))
print('Saved figure2.png / figure2.pdf')
plt.show()
