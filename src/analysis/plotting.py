"""
Plotting utilities for discovered main effects and two-way interactions.

SEM is always computed over participants: for each condition cell, per-participant
means are first calculated, then mean ± SEM is computed across those participant
means.  This gives the correct within-subjects SEM for visualising effect estimates.

Dispatch rules
--------------
Main effects:
  discrete  factor → vertical bar chart with SEM error bars
  continuous factor → line plot (values binned into quantile groups)

Two-way interactions:
  discrete  × discrete   → line plot  (x = factor A levels, lines = factor B levels)
  discrete  × continuous → line plot  (x = factor A levels, lines = binned factor B)
  continuous × discrete  → line plot  (x = binned factor A, lines = factor B levels)
  continuous × continuous → heatmap   (both factors binned, colour = mean outcome)
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # headless / file-output only — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


# ---------------------------------------------------------------------------
# SEM helper
# ---------------------------------------------------------------------------

def _cell_means(
    df: pd.DataFrame,
    group_cols: List[str],
    outcome_col: str,
    participant_col: str,
) -> pd.DataFrame:
    """
    Return a DataFrame with columns *group_cols* + ["mean", "sem"].

    Algorithm:
      1. Compute per-participant mean within each cell (participant × group_cols).
      2. Aggregate across participants: cell mean and SEM of those means.
    """
    ppt = (
        df.groupby([participant_col] + group_cols, observed=True)[outcome_col]
        .mean()
        .reset_index()
        .rename(columns={outcome_col: "_y"})
    )
    g = ppt.groupby(group_cols, observed=True)["_y"]
    out = pd.DataFrame({"mean": g.mean(), "sem": g.sem(ddof=1)}).reset_index()
    out["sem"] = out["sem"].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# Continuous-factor binning
# ---------------------------------------------------------------------------

def _bin(series: pd.Series, n: int = 5) -> pd.Series:
    """Quantile-based binning; falls back to equal-width on duplicate-value failure."""
    try:
        return pd.qcut(series, q=n, duplicates="drop")
    except ValueError:
        return pd.cut(series, bins=n)


def _iv_label(iv) -> str:
    try:
        return f"{iv.mid:.3g}"
    except Exception:
        return str(iv)


# ---------------------------------------------------------------------------
# Title helpers
# ---------------------------------------------------------------------------

def _p_str(p: Optional[float]) -> str:
    if p is None:
        return ""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _title(base: str, p: Optional[float]) -> str:
    s = _p_str(p)
    return f"{base}  ({s})" if s else base


# ---------------------------------------------------------------------------
# Main-effect plots
# ---------------------------------------------------------------------------

def plot_main_effect(
    df: pd.DataFrame,
    factor_name: str,
    factor_class: str,          # "discrete" or "continuous"
    outcome_col: str,
    participant_col: str,
    output_path: Path,
    lrt_pvalue: Optional[float] = None,
    display_name: Optional[str] = None,
) -> None:
    """Save a bar (discrete) or line (continuous, binned) plot for a main effect."""
    keep = [c for c in [participant_col, factor_name, outcome_col] if c in df.columns]
    plot_df = df[keep].dropna()

    if factor_class == "continuous":
        _main_continuous(plot_df, factor_name, outcome_col, participant_col,
                         output_path, lrt_pvalue, display_name)
    else:
        _main_discrete(plot_df, factor_name, outcome_col, participant_col,
                       output_path, lrt_pvalue, display_name)


def _make_title(raw_name: str, display_name: Optional[str], p: Optional[float],
                prefix: str = "Main effect") -> str:
    """
    Build a plot title.  If display_name differs from raw_name, show the LLM
    name on the first line and the raw variable name (greyed out) on the second.
    """
    p_str = _p_str(p)
    if display_name and display_name != raw_name:
        first = f"{display_name}  ({p_str})" if p_str else display_name
        return f"{first}\n{raw_name}"
    base = f"{prefix}: {raw_name}"
    return f"{base}  ({p_str})" if p_str else base


def _main_discrete(df, factor_name, outcome_col, participant_col, out, p,
                   display_name=None):
    agg = _cell_means(df, [factor_name], outcome_col, participant_col)
    agg[factor_name] = agg[factor_name].astype(str)
    levels = agg[factor_name].tolist()
    xs = range(len(levels))

    fig, ax = plt.subplots(figsize=(max(3.2, len(levels) * 0.9 + 1.4), 4))
    ax.bar(xs, agg["mean"], yerr=agg["sem"], capsize=5,
           color=_PALETTE[0], alpha=0.85, width=0.55,
           error_kw={"linewidth": 1.5, "ecolor": "black"})
    ax.set_xticks(list(xs))
    rot = 20 if len(levels) > 3 else 0
    ax.set_xticklabels(levels, rotation=rot, ha="right" if rot else "center")
    ax.set_xlabel(factor_name)
    ax.set_ylabel(f"Mean {outcome_col}")
    ax.set_title(_make_title(factor_name, display_name, p), fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _main_continuous(df, factor_name, outcome_col, participant_col, out, p,
                     display_name=None):
    df = df.copy()
    df["__bin"] = _bin(df[factor_name])
    agg = _cell_means(df, ["__bin"], outcome_col, participant_col)
    agg = agg.sort_values("__bin")
    labels = [_iv_label(iv) for iv in agg["__bin"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    xs = range(len(labels))
    ax.errorbar(xs, agg["mean"], yerr=agg["sem"],
                marker="o", capsize=5, color=_PALETTE[0], linewidth=1.8,
                markersize=5, error_kw={"linewidth": 1.5, "ecolor": "black"})
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel(factor_name)
    ax.set_ylabel(f"Mean {outcome_col}")
    ax.set_title(_make_title(factor_name, display_name, p), fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interaction plots
# ---------------------------------------------------------------------------

def plot_interaction(
    df: pd.DataFrame,
    factor_a_name: str,
    factor_a_class: str,        # "discrete" or "continuous"
    factor_b_name: str,
    factor_b_class: str,
    outcome_col: str,
    participant_col: str,
    output_path: Path,
    lrt_pvalue: Optional[float] = None,
    display_name: Optional[str] = None,
) -> None:
    """
    Save a two-way interaction plot.

    discrete  × discrete   → line plot
    discrete  × continuous → line plot (continuous binned)
    continuous × discrete  → line plot (continuous binned; axes swapped so
                              discrete factor is on x)
    continuous × continuous → heatmap (both binned)
    """
    keep = [c for c in [participant_col, factor_a_name, factor_b_name, outcome_col]
            if c in df.columns]
    plot_df = df[keep].dropna().copy()

    if factor_a_class == "continuous" and factor_b_class == "continuous":
        _interaction_heatmap(plot_df, factor_a_name, factor_b_name,
                             outcome_col, participant_col, output_path, lrt_pvalue,
                             display_name)
        return

    # For line plots, put the discrete factor (or factor_a if both discrete) on x.
    if factor_a_class == "continuous":
        # swap so discrete (b) is on x
        factor_a_name, factor_a_class, factor_b_name, factor_b_class = (
            factor_b_name, factor_b_class, factor_a_name, factor_a_class
        )

    # factor_a is now always discrete (x-axis); factor_b may be discrete or continuous
    if factor_b_class == "continuous":
        plot_df["__b_bin"] = _bin(plot_df[factor_b_name])
        b_col  = "__b_bin"
        b_label = factor_b_name
    else:
        b_col  = factor_b_name
        b_label = factor_b_name

    _interaction_lines(plot_df, factor_a_name, b_col, b_label,
                       outcome_col, participant_col, output_path, lrt_pvalue,
                       factor_a_name, factor_b_name, display_name)


def _interaction_lines(
    df, x_col, line_col, line_label,
    outcome_col, participant_col, out, p,
    fa_name, fb_name, display_name=None,
):
    agg = _cell_means(df, [x_col, line_col], outcome_col, participant_col)
    agg[x_col]    = agg[x_col].astype(str)
    agg[line_col] = agg[line_col].astype(str)

    x_levels    = sorted(agg[x_col].unique())
    line_levels = sorted(agg[line_col].unique())
    xs = range(len(x_levels))

    fig, ax = plt.subplots(figsize=(max(4, len(x_levels) * 1.1 + 1.6), 4))

    for i, lv in enumerate(line_levels):
        sub = agg[agg[line_col] == lv].set_index(x_col)
        ys   = [sub.loc[xl, "mean"] if xl in sub.index else np.nan for xl in x_levels]
        errs = [sub.loc[xl, "sem"]  if xl in sub.index else np.nan for xl in x_levels]
        ax.errorbar(xs, ys, yerr=errs, marker="o",
                    label=f"{line_label} = {lv}",
                    color=_PALETTE[i % len(_PALETTE)], linewidth=1.8,
                    markersize=5, capsize=4,
                    error_kw={"linewidth": 1.5, "ecolor": _PALETTE[i % len(_PALETTE)]})

    rot = 20 if len(x_levels) > 3 else 0
    ax.set_xticks(list(xs))
    ax.set_xticklabels(x_levels, rotation=rot, ha="right" if rot else "center")
    ax.set_xlabel(fa_name)
    ax.set_ylabel(f"Mean {outcome_col}")
    raw_label = f"{fa_name} × {fb_name}"
    ax.set_title(_make_title(raw_label, display_name, p, prefix="Interaction"),
                 fontsize=10)
    ax.legend(title=line_label, bbox_to_anchor=(1.02, 1), loc="upper left",
              borderaxespad=0, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _interaction_heatmap(df, fa, fb, outcome_col, participant_col, out, p,
                         display_name=None):
    df = df.copy()
    df["__a_bin"] = _bin(df[fa])
    df["__b_bin"] = _bin(df[fb])

    agg = _cell_means(df, ["__a_bin", "__b_bin"], outcome_col, participant_col)
    agg["__a_str"] = agg["__a_bin"].apply(_iv_label)
    agg["__b_str"] = agg["__b_bin"].apply(_iv_label)

    pivot = agg.pivot(index="__a_str", columns="__b_str", values="mean")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", origin="upper")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel(f"{fb} (binned)")
    ax.set_ylabel(f"{fa} (binned)")
    plt.colorbar(im, ax=ax, label=f"Mean {outcome_col}")
    raw_label = f"{fa} × {fb}"
    ax.set_title(_make_title(raw_label, display_name, p, prefix="Interaction"),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def plot_all_effects(
    df: pd.DataFrame,
    discovered_factors,                  # List[DiscoveredFactor]
    discovered_effects,                  # List[DiscoveredEffect]
    factor_class_lookup: Dict[str, str], # {name: "discrete"|"continuous"} for all factors
    outcome_col: str,
    participant_col: str,
    output_dir: Path,
    factor_stats_by_name: Dict[str, dict],
    effect_stats_by_term: Dict[str, dict],
    llm_name_lookup: Optional[Dict[str, str]] = None,
) -> None:
    """
    Generate and save one PNG per discovered main effect and per 2-way interaction.

    Files are written to output_dir:
      {factor_name}_effect.png
      {factor_a}__{factor_b}_interaction.png

    llm_name_lookup: optional dict mapping factor column_name → LLM-generated name
    (for main effects) and effect term → LLM-generated name (for interactions).
    When provided, the LLM name appears as the plot title and the raw name is
    shown as a subtitle.

    Errors in individual plots are caught and printed as warnings so that a
    single bad plot does not abort the rest.
    """
    names = llm_name_lookup or {}

    for f in discovered_factors:
        out = output_dir / f"{f.column_name}_effect.png"
        p   = factor_stats_by_name.get(f.column_name, {}).get("lrt_pvalue")
        try:
            plot_main_effect(
                df=df,
                factor_name=f.column_name,
                factor_class=f.candidate.factor_class,
                outcome_col=outcome_col,
                participant_col=participant_col,
                output_path=out,
                lrt_pvalue=p,
                display_name=names.get(f.column_name),
            )
            print(f"  Plot saved → {out}")
        except Exception as exc:
            print(f"  Warning: could not plot main effect '{f.column_name}': {exc}")

    for e in discovered_effects:
        if len(e.factor_names) != 2:
            continue
        fa, fb = e.factor_names
        if fa not in factor_class_lookup or fb not in factor_class_lookup:
            print(f"  Warning: skipping interaction plot for {e.term} "
                  f"(factor class unknown)")
            continue
        out = output_dir / f"{'__'.join(e.factor_names)}_interaction.png"
        p   = effect_stats_by_term.get(e.term, {}).get("lrt_pvalue")
        try:
            plot_interaction(
                df=df,
                factor_a_name=fa,
                factor_a_class=factor_class_lookup[fa],
                factor_b_name=fb,
                factor_b_class=factor_class_lookup[fb],
                outcome_col=outcome_col,
                participant_col=participant_col,
                output_path=out,
                lrt_pvalue=p,
                display_name=names.get(e.term),
            )
            print(f"  Plot saved → {out}")
        except Exception as exc:
            print(f"  Warning: could not plot interaction '{e.term}': {exc}")
