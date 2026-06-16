"""Figure 6 — Search Efficiency Curves.

Line plot: cumulative LLM synthesis calls (x) vs. best F1 so far (y).
Two lines per benchmark: LLM seeding (solid) vs. random seeding (dashed).
Shaded ribbon = ±1 SE across 5 seeds.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import style
from mockup_data import EFFICIENCY_DATA, EFF_BENCHMARKS

style.apply_style()

BENCH_COLORS = {
    'Stroop-Simon':        '#4393C3',
    'RDK\nTask-Switching': '#2166AC',
    'Prospect\nTheory':    '#08519C',
}

fig, ax = plt.subplots(figsize=(style.W1 + 0.4, 3.0))

for bench in EFF_BENCHMARKS:
    d     = EFFICIENCY_DATA[bench]
    x     = d['x']
    color = BENCH_COLORS[bench]
    label = bench.replace('\n', ' ')

    # LLM seeding — solid
    ax.plot(x, d['llm_mean'], color=color, lw=1.4, ls='-',
            label=f'{label} (LLM)')
    ax.fill_between(x,
                    d['llm_mean'] - d['llm_se'],
                    d['llm_mean'] + d['llm_se'],
                    color=color, alpha=0.15)

    # Random seeding — dashed
    ax.plot(x, d['rnd_mean'], color=color, lw=1.2, ls='--',
            label=f'{label} (random)')
    ax.fill_between(x,
                    d['rnd_mean'] - d['rnd_se'],
                    d['rnd_mean'] + d['rnd_se'],
                    color=color, alpha=0.08)

ax.set_xlabel('Cumulative synthesis calls', fontsize=style.FS_LABEL)
ax.set_ylabel('Best F1 (rolling max)', fontsize=style.FS_LABEL)
ax.set_xlim(0, x[-1])
ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

# Custom legend: benchmark colors for solid/dashed
from matplotlib.lines import Line2D
legend_entries = []
for bench in EFF_BENCHMARKS:
    color = BENCH_COLORS[bench]
    label = bench.replace('\n', ' ')
    legend_entries.append(
        Line2D([0], [0], color=color, lw=1.4, ls='-', label=label))

legend_entries.append(
    Line2D([0], [0], color='#555555', lw=1.4, ls='-',  label='LLM seeding'))
legend_entries.append(
    Line2D([0], [0], color='#555555', lw=1.2, ls='--', label='Random seeding'))

ax.legend(handles=legend_entries, loc='lower right',
          fontsize=style.FS_ANNOT, frameon=False,
          handlelength=1.4, ncol=1)

style.despine(ax)

fig.text(0.5, 0.5, 'MOCKUP DATA', ha='center', va='center',
         fontsize=28, color='lightgray', alpha=0.35,
         rotation=30, transform=fig.transFigure, zorder=0)

out_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'figure6.png'))
fig.savefig(os.path.join(out_dir, 'figure6.pdf'))
print('Saved figure6.png / figure6.pdf')
plt.show()
