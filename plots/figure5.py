"""Figure 5 — Ablation Study.

Grouped horizontal bar chart. Rows = ablation conditions (full system on top).
Two bar groups per row: Stroop-Simon (medium) and RDK Task-Switching (hard).
Dashed reference lines at full-system F1 for each benchmark.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
import style
from mockup_data import ABLATION_CONDITIONS, ABLATION_DATA

style.apply_style()

benchmarks = list(ABLATION_DATA.keys())
n_cond   = len(ABLATION_CONDITIONS)
n_bench  = len(benchmarks)

bar_h    = 0.28
y_gap    = 0.85           # distance between condition centers
y_pos    = np.arange(n_cond) * y_gap

bench_colors  = ['#4393C3', '#2166AC']
bench_offsets = [+bar_h / 2, -bar_h / 2]

fig, ax = plt.subplots(figsize=(style.W1 + 1.0, 3.6))

for bi, (bench, color, offset) in enumerate(
        zip(benchmarks, bench_colors, bench_offsets)):
    data = ABLATION_DATA[bench]
    means = np.array([d[0] for d in data])
    ses   = np.array([d[1] for d in data])

    ax.barh(y_pos + offset, means, bar_h,
            color=color, alpha=0.88, zorder=3,
            label=bench.replace('\n', ' '))

    ax.errorbar(means, y_pos + offset, xerr=ses, fmt='none',
                color='#444444', lw=0.8, capsize=2.5, capthick=0.8,
                zorder=5)

    # Dashed reference at full-system F1
    ref_f1 = data[0][0]
    ax.axvline(ref_f1, color=color, lw=0.9, ls='--', alpha=0.6, zorder=2)

# Y-axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(ABLATION_CONDITIONS, fontsize=style.FS_TICK)
ax.invert_yaxis()

ax.set_xlabel('Factor Recovery (F1)', fontsize=style.FS_LABEL)
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

ax.legend(loc='lower right', fontsize=style.FS_ANNOT,
          frameon=False, handlelength=1.0)

# Subtle horizontal separators between conditions
for y in y_pos[1:] - y_gap / 2:
    ax.axhline(y, color='#E8E8E8', lw=0.6, zorder=0)

style.despine(ax, left=True)
ax.spines['left'].set_visible(False)
ax.tick_params(left=False)

fig.text(0.5, 0.5, 'MOCKUP DATA', ha='center', va='center',
         fontsize=28, color='lightgray', alpha=0.35,
         rotation=30, transform=fig.transFigure, zorder=0)

out_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'figure5.png'))
fig.savefig(os.path.join(out_dir, 'figure5.pdf'))
print('Saved figure5.png / figure5.pdf')
plt.show()
