"""Shared visual style constants for all AutoFactor paper figures."""

import colorsys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# ── Factor-type color palette ─────────────────────────────────────────────────
# Blue family = discrete factors; red family = continuous factors.
# Darker shade = within-trial; lighter shade = across-trial (continuous also hatched).
FACTOR_COLORS = {
    'within_trial_discrete':   '#2166AC',  # dark blue
    'window_discrete':         '#74ADD1',  # light blue
    'within_trial_continuous': '#D73027',  # dark red
    'window_continuous':       '#F4A582',  # light salmon
}

# ── Metric colors (Panel A bars) — green shades, distinct from factor colors ──
METRIC_COLORS = {
    'precision': '#1B7837',
    'recall':    '#5AAE61',
    'f1':        '#A6D96A',
}

# ── Empirical / novel colors ──────────────────────────────────────────────────
EMPIRICAL_COLOR = '#1B7837'
NOVEL_COLOR     = '#1B7837'
KNOWN_COLOR     = '#4393C3'

# ── Reference line ─────────────────────────────────────────────────────────────
REF_LINE_COLOR  = '#888888'

# ── Typography ────────────────────────────────────────────────────────────────
FS_LABEL   = 8    # axis labels
FS_TICK    = 7    # tick labels
FS_TITLE   = 8.5  # panel titles / sub-headers
FS_PANEL   = 10   # bold panel letters (A, B, C)
FS_ANNOT   = 6.5  # small in-plot annotations
FS_CODE    = 5.5  # monospace code insets

# ── Figure widths (inches, Nature single/double column) ───────────────────────
W1 = 3.5
W2 = 7.2

# ── Hatch pattern for continuous-factor bars ──────────────────────────────────
CONTINUOUS_HATCH = '///'


def scale_saturation(hex_color, factor):
    """Return an RGB tuple with the HLS saturation of hex_color scaled by factor (0–1)."""
    r = int(hex_color[1:3], 16) / 255.0
    g = int(hex_color[3:5], 16) / 255.0
    b = int(hex_color[5:7], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l, s * factor)


def apply_style():
    mpl.rcParams.update({
        'font.family':          'Arial',
        'font.size':             FS_TICK,
        'axes.labelsize':        FS_LABEL,
        'axes.titlesize':        FS_TITLE,
        'xtick.labelsize':       FS_TICK,
        'ytick.labelsize':       FS_TICK,
        'legend.fontsize':       FS_TICK,
        'axes.linewidth':        0.7,
        'xtick.major.width':     0.7,
        'ytick.major.width':     0.7,
        'xtick.minor.width':     0.5,
        'ytick.minor.width':     0.5,
        'xtick.major.size':      3.0,
        'ytick.major.size':      3.0,
        'axes.spines.top':       False,
        'axes.spines.right':     False,
        'figure.dpi':            150,
        'savefig.dpi':           300,
        'savefig.bbox':          'tight',
        'savefig.pad_inches':    0.08,
        'pdf.fonttype':          42,
        'ps.fonttype':           42,
        'legend.frameon':        False,
        'legend.handlelength':   1.2,
    })


def despine(ax, left=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if left:
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)


def panel_label(ax, letter, x=-0.14, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight='bold', va='top', ha='right',
            clip_on=False)


def blended(ax):
    """Transform: x in axis coords [0,1], y in data coords."""
    return mtransforms.blended_transform_factory(ax.transAxes, ax.transData)


def factor_color(ftype, fclass):
    key = f'{ftype}_{fclass}'
    return FACTOR_COLORS.get(key, '#999999')


def factor_type_legend(ax, loc='lower right', continuous_label=True):
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=FACTOR_COLORS['within_trial_discrete'],
              label='Within-trial, discrete'),
        Patch(facecolor=FACTOR_COLORS['window_discrete'],
              label='Across-trial, discrete'),
    ]
    if continuous_label:
        handles += [
            Patch(facecolor=FACTOR_COLORS['within_trial_continuous'],
                  hatch=CONTINUOUS_HATCH, edgecolor=FACTOR_COLORS['within_trial_continuous'],
                  label='Within-trial, continuous (|ρ|)'),
            Patch(facecolor=FACTOR_COLORS['window_continuous'],
                  hatch=CONTINUOUS_HATCH, edgecolor=FACTOR_COLORS['window_continuous'],
                  label='Across-trial, continuous (|ρ|)'),
        ]
    ax.legend(handles=handles, loc=loc, fontsize=FS_ANNOT,
              frameon=False, handlelength=1.0, handleheight=0.8)
