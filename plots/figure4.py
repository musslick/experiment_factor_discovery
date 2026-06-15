"""Figure 4 — Novel Factor Discovery on Empirical Datasets.

Vertical layout per discovery cell:
  Top (~62%):  effect plot — participant dots + group mean ± 95% CI
  Bottom (~38%): code inset — compute_factor() block only
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(__file__))
import style
from mockup_data import NOVEL_DISCOVERIES

style.apply_style()

N        = len(NOVEL_DISCOVERIES)
N_COLS   = 3
N_ROWS   = (N + N_COLS - 1) // N_COLS

fig = plt.figure(figsize=(style.W2, 5.5 * N_ROWS))

outer_gs = gridspec.GridSpec(N_ROWS, N_COLS, figure=fig,
                             hspace=0.55, wspace=0.38)

# ── Code block colors ─────────────────────────────────────────────────────────
CODE_BG  = '#F5F5F5'
CODE_EDG = '#DDDDDD'


def plot_discovery(outer_cell, disc):
    """Render one discovery: effect plot (top) + code inset (bottom)."""

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_cell,
        height_ratios=[3.5, 1.8], hspace=0.28)

    ax_eff  = fig.add_subplot(inner[0])
    ax_code = fig.add_subplot(inner[1])

    # ── Effect plot ───────────────────────────────────────────────────────────
    levels   = disc['levels']
    gmeans   = np.array(disc['group_means'])
    ci95     = np.array(disc['group_ci95'])
    pt_data  = disc['participant_data']
    n_levels = len(levels)
    x_pos    = np.arange(n_levels, dtype=float)

    rng = np.random.default_rng(77)

    # Individual participant dots
    for xi, pts in zip(x_pos, pt_data):
        jitter = rng.normal(0, 0.07, len(pts))
        ax_eff.scatter(xi + jitter, pts,
                       s=5, color='#AAAAAA', alpha=0.40,
                       linewidths=0, zorder=1)

    # Group means + 95% CI
    ax_eff.errorbar(x_pos, gmeans, yerr=ci95,
                    fmt='o', color=style.NOVEL_COLOR,
                    markersize=6.5, markeredgewidth=0,
                    lw=1.6, capsize=3.5, capthick=1.2, zorder=3)

    # Y limits: span participant data range + generous top padding for title
    all_pts = np.concatenate(pt_data)
    lo = np.percentile(all_pts, 2)
    hi = np.percentile(all_pts, 98)
    span = hi - lo
    ax_eff.set_ylim(lo - span * 0.10, hi + span * 0.45)

    ax_eff.set_xticks(x_pos)
    ax_eff.set_xticklabels(levels, fontsize=style.FS_TICK)
    ax_eff.set_ylabel(disc['outcome_label'], fontsize=style.FS_LABEL)
    ax_eff.set_xlim(-0.65, n_levels - 0.35)

    # Factor name as title (inside the padded top region)
    ax_eff.text(0.5, 0.97, disc['factor_name'],
                transform=ax_eff.transAxes,
                ha='center', va='top',
                fontsize=style.FS_LABEL, fontweight='bold')

    # Dataset + N as italic subtitle
    ax_eff.text(0.5, 0.88, disc['dataset'],
                transform=ax_eff.transAxes,
                ha='center', va='top',
                fontsize=style.FS_ANNOT, color='#666666', style='italic')

    style.despine(ax_eff)

    # ── Code inset (full cell width) ──────────────────────────────────────────
    ax_code.set_xlim(0, 1)
    ax_code.set_ylim(0, 1)
    ax_code.axis('off')

    # Gray background spanning full inset height
    ax_code.add_patch(Rectangle((0, 0), 1, 1,
                                facecolor=CODE_BG, edgecolor=CODE_EDG,
                                lw=0.5, transform=ax_code.transAxes,
                                clip_on=False))

    # Header label
    ax_code.text(0.04, 0.94, 'compute_factor()',
                 transform=ax_code.transAxes, ha='left', va='top',
                 fontsize=style.FS_CODE, color='#999999',
                 fontfamily='monospace', clip_on=True)

    # Code body — clip_on=True prevents text from running past the box edge
    ax_code.text(0.04, 0.78, disc['code'],
                 transform=ax_code.transAxes, ha='left', va='top',
                 fontsize=style.FS_CODE, fontfamily='monospace',
                 color='#1A1A1A', linespacing=1.65, clip_on=True)

    # ΔLL + reference below the code inset
    dl = disc['delta_ll']
    ax_code.text(0.025, -0.07,
                 f'ΔLL = {dl:.1f}  ·  held-out set',
                 transform=ax_code.transAxes, ha='left', va='top',
                 fontsize=style.FS_ANNOT, color='#444444', clip_on=False)
    ref = disc.get('literature_ref')
    if ref:
        ax_code.text(0.025, -0.22, ref,
                     transform=ax_code.transAxes, ha='left', va='top',
                     fontsize=style.FS_ANNOT, color='#888888',
                     style='italic', clip_on=False)


# ── Render each discovery ─────────────────────────────────────────────────────
for idx, disc in enumerate(NOVEL_DISCOVERIES):
    row_i = idx // N_COLS
    col_i = idx % N_COLS
    plot_discovery(outer_gs[row_i, col_i], disc)

# Hide unused cells
for idx in range(N, N_ROWS * N_COLS):
    ax_empty = fig.add_subplot(outer_gs[idx // N_COLS, idx % N_COLS])
    ax_empty.axis('off')

fig.text(0.5, 0.5, 'MOCKUP DATA', ha='center', va='center',
         fontsize=28, color='lightgray', alpha=0.35,
         rotation=30, transform=fig.transFigure, zorder=0)

out_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'figure4.png'))
fig.savefig(os.path.join(out_dir, 'figure4.pdf'))
print('Saved figure4.png / figure4.pdf')
plt.show()
