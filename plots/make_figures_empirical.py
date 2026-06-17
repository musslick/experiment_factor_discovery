"""make_figures_empirical.py — Generate Figure 3 (panels A, B) and Figure 4.

Figure 3:
  A  P/R/F1 per empirical dataset, with pooled synthetic F1 reference line
  B  Per-factor recovery on known ground-truth factors (across datasets)

Figure 4:
  Discovery-effect panels: top-K novel factors per dataset
  Each cell: effect plot (participant dots + group mean±SEM) + code inset
"""

import argparse
import json
import math
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

style.apply_style()

# ── Human-readable label dicts ────────────────────────────────────────────────
FACTOR_DISPLAY_NAMES = {
    "congruency":           "congruency",
    "distractor_type_prev": "prev. distractor type",
}

OUTCOME_DISPLAY = {
    "latency":    "RT (ms)",
    "accuracy":   "Accuracy (%)",
    "correct":    "Accuracy (%)",
    "chose_left": "P(chose left)",
}

# ── Configurable constants ────────────────────────────────────────────────────
TOP_K_PER_DATASET = 1
CODE_BG  = "#F5F5F5"
CODE_EDG = "#DDDDDD"


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _outcome_label(raw_label: str, null_formula: str = "") -> str:
    """Return display label for an outcome, optionally prefixing 'log '."""
    base = OUTCOME_DISPLAY.get(raw_label, raw_label)
    if "np.log(" in (null_formula or ""):
        base = "log " + base
    return base


def _factor_display(name: str) -> str:
    return FACTOR_DISPLAY_NAMES.get(name, name)


def _factor_color(ftype: str, fclass: str) -> str:
    key = f"{ftype}_{fclass}"
    return style.FACTOR_COLORS.get(key, "#999999")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_empirical(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def load_synthetic(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def pooled_synthetic_f1(syn_data: dict) -> float:
    """Mean F1 across all runs of all synthetic benchmarks."""
    all_f1 = []
    benchmarks = syn_data.get("benchmarks", {})
    for bench_name, bench in benchmarks.items():
        for run in bench.get("runs", []):
            if "f1" in run:
                all_f1.append(run["f1"])
    if not all_f1:
        return float("nan")
    return float(np.mean(all_f1))


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure3(emp_data: dict, syn_data: dict, out_dir: str, show: bool):
    datasets = emp_data.get("datasets", {})
    ds_names = list(datasets.keys())

    # ── Collect metrics (mean ± SE per dataset) ───────────────────────────────
    metrics_order = ["precision", "recall", "f1"]
    prf = {}  # ds_name -> {metric -> (mean, se)}
    n_gt = {}  # ds_name -> n_ground_truth
    for ds_name, ds in datasets.items():
        runs = ds.get("benchmark_runs", [])
        prf[ds_name] = {}
        for m in metrics_order:
            vals = [r[m] for r in runs if m in r]
            if vals:
                prf[ds_name][m] = (float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            else:
                prf[ds_name][m] = (0.0, 0.0)
        n_list = [r.get("n_ground_truth", 0) for r in runs]
        n_gt[ds_name] = int(round(np.mean(n_list))) if n_list else 0

    # ── Pooled synthetic F1 reference ─────────────────────────────────────────
    ref_f1 = pooled_synthetic_f1(syn_data)

    # ── Factor rows for Panel B ───────────────────────────────────────────────
    # Each row: (display_label, ds_display_name, ftype, fclass, n_lvl, mean, se)
    factor_rows = []
    for ds_name, ds in datasets.items():
        display_name = ds.get("display_name", ds_name)
        gt_factors = ds.get("ground_truth_factors", [])
        runs = ds.get("benchmark_runs", [])
        for gf in gt_factors:
            fname = gf["name"]
            ftype = gf.get("type", "within_trial")
            fclass = gf.get("factor_class", "discrete")
            n_lvl = gf.get("n_levels", 0)
            # Collect per-run recovery values
            if fclass == "continuous":
                vals = []
                for r in runs:
                    ccp = r.get("continuous_correlation_per_factor", {})
                    if fname in ccp:
                        vals.append(abs(ccp[fname]))
            else:
                vals = []
                for r in runs:
                    lrp = r.get("level_recovery_per_factor", {})
                    if fname in lrp:
                        v = lrp[fname]
                        if isinstance(v, dict):
                            vals.append(v.get("level_recall", 0.0))
                        else:
                            vals.append(float(v))
            if vals:
                mean_v = float(np.mean(vals))
                se_v = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            else:
                mean_v, se_v = 0.0, 0.0
            factor_rows.append((_factor_display(fname), display_name,
                                 ftype, fclass, n_lvl, mean_v, se_v))

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(style.W2, 3.8),
        gridspec_kw={"width_ratios": [2.8, 3.6], "wspace": 0.42}
    )

    # ── Panel A ───────────────────────────────────────────────────────────────
    metric_labels = ["Precision", "Recall", "F1"]
    n_ds  = len(ds_names)
    bar_w = 0.22
    x_centers = np.arange(n_ds) * 0.90

    for i, (metric, label) in enumerate(zip(metrics_order, metric_labels)):
        offset = (i - 1) * bar_w
        means_ = [prf[d][metric][0] for d in ds_names]
        ses_   = [prf[d][metric][1] for d in ds_names]
        ax_a.bar(x_centers + offset, means_, bar_w,
                 color=style.METRIC_COLORS[metric], label=label,
                 yerr=ses_,
                 error_kw=dict(lw=0.8, capsize=2.5, capthick=0.8),
                 zorder=3)

    # Pooled synthetic F1 reference — single horizontal dashed line
    if not math.isnan(ref_f1):
        x_lo = x_centers[0] - 0.50 if n_ds > 0 else -0.5
        x_hi = x_centers[-1] + 0.50 if n_ds > 0 else 0.5
        ax_a.plot([x_lo, x_hi], [ref_f1, ref_f1],
                  color=style.REF_LINE_COLOR, lw=1.2, ls="--", zorder=4)
        ref_line = plt.Line2D([0], [0], color=style.REF_LINE_COLOR,
                              lw=1.2, ls="--", label="Synthetic ref. (mean F1)")
    else:
        ref_line = None

    # Annotate N ground-truth factors above each group
    for xi, ds_name in zip(x_centers, ds_names):
        n = n_gt.get(ds_name, 0)
        label_txt = f'{n} factor{"s" if n != 1 else ""}'
        ax_a.text(xi, 1.04, label_txt,
                  ha="center", va="bottom",
                  fontsize=style.FS_ANNOT, color="#555555", clip_on=False)

    # X-axis labels from display_name
    ds_labels = [datasets[d].get("display_name", d) for d in ds_names]
    ax_a.set_xticks(x_centers)
    ax_a.set_xticklabels(ds_labels, fontsize=style.FS_TICK)
    ax_a.set_ylabel("Score", fontsize=style.FS_LABEL)
    ax_a.set_ylim(0, 1.12)
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    handles, labels_ = ax_a.get_legend_handles_labels()
    if ref_line is not None:
        handles = handles + [ref_line]
        labels_ = labels_ + ["Synthetic ref. (mean F1)"]
    ax_a.legend(handles, labels_,
                loc="lower left", fontsize=style.FS_ANNOT,
                frameon=False, handlelength=1.2)

    style.despine(ax_a)
    style.panel_label(ax_a, "A")

    # ── Panel B ───────────────────────────────────────────────────────────────
    n_factors = len(factor_rows)
    y_positions = np.arange(n_factors)

    for i, row in enumerate(factor_rows):
        disp_label, ds_display, ftype, fclass, n_lvl, mean_v, se_v = row
        color = _factor_color(ftype, fclass)
        is_continuous = (fclass == "continuous")

        ax_b.barh(i, mean_v, height=0.55,
                  color=color, alpha=0.90, zorder=3)
        if is_continuous:
            ax_b.barh(i, mean_v, height=0.55,
                      color="none", hatch=style.CONTINUOUS_HATCH,
                      edgecolor=color, linewidth=0.0, zorder=4)
        ax_b.errorbar(mean_v, i, xerr=se_v, fmt="none",
                      color="#444444", lw=0.8, capsize=2.5, capthick=0.8,
                      zorder=5)

    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([r[0] for r in factor_rows], fontsize=6.5)
    ax_b.invert_yaxis()

    # Dataset section separators and side labels
    dataset_groups = []
    if factor_rows:
        prev_ds, start_i = factor_rows[0][1], 0
        for i, row in enumerate(factor_rows[1:], start=1):
            if row[1] != prev_ds:
                dataset_groups.append((prev_ds, start_i, i - 1))
                prev_ds, start_i = row[1], i
        dataset_groups.append((prev_ds, start_i, len(factor_rows) - 1))

    for ds_display, start, end in dataset_groups:
        if start > 0:
            ax_b.axhline(start - 0.5, color="#BBBBBB", lw=0.8, zorder=0)
        ax_b.text(1.04, (start + end) / 2.0,
                  ds_display.replace("\n", " "),
                  transform=style.blended(ax_b),
                  ha="left", va="center",
                  fontsize=style.FS_ANNOT, fontweight="bold", color="#444444",
                  clip_on=False)

    # Full-recovery threshold line
    ax_b.axvline(0.95, color="#999999", lw=0.8, ls=":", zorder=1)
    ax_b.text(0.95, -0.6, "0.95", ha="center", va="top",
              fontsize=style.FS_ANNOT, color="#999999")

    ax_b.set_xlabel("Recovery score  (level recall  |  |Spearman ρ|)",
                    fontsize=style.FS_LABEL)
    ax_b.set_xlim(0, 1.0)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    style.despine(ax_b, left=True)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(left=False)
    style.panel_label(ax_b, "B", x=-0.04)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "figure3.png"))
    fig.savefig(os.path.join(out_dir, "figure3.pdf"))
    print("Saved figure3.png / figure3.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_discovery_cells(emp_data: dict) -> list:
    """Return list of cell dicts, one per (dataset × top-K factor)."""
    cells = []
    datasets = emp_data.get("datasets", {})
    for ds_name, ds in datasets.items():
        display_name = ds.get("display_name", ds_name)
        null_formula = ds.get("null_formula", "")
        discovery_runs = ds.get("discovery_runs", [])
        if not discovery_runs:
            continue
        best_idx = ds.get("best_discovery_run_idx", 0)
        best_idx = min(best_idx, len(discovery_runs) - 1)
        best_run = discovery_runs[best_idx]
        discovered = best_run.get("discovered_factors", [])
        # Sort by validation_improvement descending
        discovered_sorted = sorted(
            discovered,
            key=lambda f: f.get("validation_improvement", 0.0),
            reverse=True,
        )
        for factor in discovered_sorted[:TOP_K_PER_DATASET]:
            cells.append({
                "ds_name":      ds_name,
                "display_name": display_name,
                "null_formula": null_formula,
                "factor":       factor,
            })
    return cells


def _plot_discovery_cell(fig, outer_cell, cell: dict):
    """Render one discovery panel: effect plot (top) + code inset (bottom)."""
    disc = cell["factor"]
    display_name = cell["display_name"]
    null_formula = cell.get("null_formula", "")

    pl_data = disc.get("participant_level_data") or {}
    raw_outcome = pl_data.get("outcome_label", "")
    outcome_label = _outcome_label(raw_outcome, null_formula)
    levels = pl_data.get("levels", [])
    group_means = np.array(pl_data.get("group_means", []))
    group_sems  = np.array(pl_data.get("group_sems", []))
    participant_means = pl_data.get("participant_means", {})

    n_levels = len(levels)
    x_pos = np.arange(n_levels, dtype=float)

    # Determine title: llm_name if available, else display name from dict
    factor_name_raw = disc.get("name", "")
    llm_name = disc.get("llm_name", "")
    title_str = llm_name if llm_name else _factor_display(factor_name_raw)

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_cell,
        height_ratios=[3.5, 1.8], hspace=0.28
    )
    ax_eff  = fig.add_subplot(inner[0])
    ax_code = fig.add_subplot(inner[1])

    # ── Effect plot ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(77)

    # Flatten all participant data for y-limit computation
    all_pts = []
    for lvl in levels:
        pts = participant_means.get(lvl, [])
        all_pts.extend(pts)

    # Scatter participant dots per level
    for xi, lvl in zip(x_pos, levels):
        pts = np.array(participant_means.get(lvl, []), dtype=float)
        if len(pts) > 0:
            jitter = rng.normal(0, 0.07, len(pts))
            ax_eff.scatter(xi + jitter, pts,
                           s=5, color="#AAAAAA", alpha=0.40,
                           linewidths=0, zorder=1)

    # Group means ± SEM
    if len(group_means) == n_levels and n_levels > 0:
        ax_eff.errorbar(x_pos, group_means, yerr=group_sems,
                        fmt="o", color=style.NOVEL_COLOR,
                        markersize=6.5, markeredgewidth=0,
                        lw=1.6, capsize=3.5, capthick=1.2, zorder=3)

    # Y limits: percentile 2/98 of participant data ± margin
    if all_pts:
        all_arr = np.array(all_pts, dtype=float)
        lo = np.percentile(all_arr, 2)
        hi = np.percentile(all_arr, 98)
        span = hi - lo if hi > lo else 1.0
        ax_eff.set_ylim(lo - span * 0.10, hi + span * 0.45)

    ax_eff.set_xticks(x_pos)
    ax_eff.set_xticklabels(levels, fontsize=style.FS_TICK)
    ax_eff.set_ylabel(outcome_label, fontsize=style.FS_LABEL)
    if n_levels > 0:
        ax_eff.set_xlim(-0.65, n_levels - 0.35)

    # Title (factor name) inside the padded top region
    ax_eff.text(0.5, 0.97, title_str,
                transform=ax_eff.transAxes,
                ha="center", va="top",
                fontsize=style.FS_LABEL, fontweight="bold")

    # Italic subtitle: dataset display name
    ax_eff.text(0.5, 0.88, display_name,
                transform=ax_eff.transAxes,
                ha="center", va="top",
                fontsize=style.FS_ANNOT, color="#666666", style="italic")

    style.despine(ax_eff)

    # ── Code inset ────────────────────────────────────────────────────────────
    ax_code.set_xlim(0, 1)
    ax_code.set_ylim(0, 1)
    ax_code.axis("off")

    # Gray background rectangle
    ax_code.add_patch(Rectangle(
        (0, 0), 1, 1,
        facecolor=CODE_BG, edgecolor=CODE_EDG,
        lw=0.5, transform=ax_code.transAxes, clip_on=False
    ))

    # Header
    ax_code.text(0.04, 0.94, "compute_factor()",
                 transform=ax_code.transAxes, ha="left", va="top",
                 fontsize=style.FS_CODE, color="#999999",
                 fontfamily="monospace", clip_on=True)

    # Code body
    compute_code = disc.get("compute_code", "")
    ax_code.text(0.04, 0.78, compute_code,
                 transform=ax_code.transAxes, ha="left", va="top",
                 fontsize=style.FS_CODE, fontfamily="monospace",
                 color="#1A1A1A", linespacing=1.65, clip_on=True)

    # ΔLL annotation below the code inset
    delta_ll = disc.get("validation_improvement", float("nan"))
    ax_code.text(0.025, -0.07,
                 f"ΔLL = {delta_ll:.1f}  ·  held-out set",
                 transform=ax_code.transAxes, ha="left", va="top",
                 fontsize=style.FS_ANNOT, color="#444444", clip_on=False)


def make_figure4(emp_data: dict, out_dir: str, show: bool):
    cells = _collect_discovery_cells(emp_data)
    if not cells:
        print("No discovery cells to plot for Figure 4 — skipping.")
        return

    total_cells = len(cells)
    N_COLS = total_cells  # one column per (dataset × top-K)
    N_ROWS = math.ceil(total_cells / N_COLS)

    fig = plt.figure(figsize=(style.W2, 5.5 * N_ROWS))
    outer_gs = gridspec.GridSpec(N_ROWS, N_COLS, figure=fig,
                                 hspace=0.55, wspace=0.38)

    for idx, cell in enumerate(cells):
        row_i = idx // N_COLS
        col_i = idx % N_COLS
        _plot_discovery_cell(fig, outer_gs[row_i, col_i], cell)

    # Hide unused cells
    for idx in range(total_cells, N_ROWS * N_COLS):
        ax_empty = fig.add_subplot(outer_gs[idx // N_COLS, idx % N_COLS])
        ax_empty.axis("off")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "figure4.png"))
    fig.savefig(os.path.join(out_dir, "figure4.pdf"))
    print("Saved figure4.png / figure4.pdf")
    if show:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 (A, B) and Figure 4 from empirical results."
    )
    parser.add_argument(
        "--results",
        default="results/aggregated/empirical_results.json",
        help="Path to empirical_results.json",
    )
    parser.add_argument(
        "--synthetic-results",
        default="results/aggregated/synthetic_results.json",
        help="Path to synthetic_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/output",
        help="Directory where figure files are written",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful in headless environments)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    show = not args.no_show

    # Resolve paths relative to CWD (the repo root when invoked normally)
    results_path   = args.results
    synthetic_path = args.synthetic_results
    out_dir        = args.output_dir

    emp_data = load_empirical(results_path)
    syn_data = load_synthetic(synthetic_path)

    make_figure3(emp_data, syn_data, out_dir, show)
    make_figure4(emp_data, out_dir, show)


if __name__ == "__main__":
    main()
