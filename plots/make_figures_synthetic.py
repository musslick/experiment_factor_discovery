"""make_figures_synthetic.py — Figure 2 (Panels A, B, C) from real benchmark data.

Loads results/aggregated/synthetic_results.json and generates:
  A  Overall P/R/F1 per benchmark (grouped bar chart)
  B  Per-factor level/correlation recovery (horizontal bars)
  C  2x2 factor-type recall matrix (heatmap)

Usage:
  python plots/make_figures_synthetic.py \
      --results results/aggregated/synthetic_results.json \
      --output-dir plots/output \
      [--no-show]
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Import shared style from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style

# ── Human-readable factor display names ──────────────────────────────────────

FACTOR_DISPLAY_NAMES = {
    # Stroop-Simon
    "word_color_congruency":              "word-color congruency",
    "location_response_congruency":       "location-response congruency",
    "congruency_previous_trial":          "previous-trial congruency",
    "response_transition":                "response transition",
    # RDK Task-Switching
    "task_transition":                    "task transition",
    "current_stimulus_difficulty":        "current stimulus difficulty",
    "past_stimulus_difficulty":           "past stimulus difficulty",
    "n2_task_inhibition":                 "n-2 task inhibition",
    # Prospect Theory
    "expected_value_difference":          "expected value difference",
    "gain_difference":                    "gain difference",
    "loss_difference":                    "loss difference",
    "probability_difference":             "probability difference",
    "dominance_relation":                 "dominance relation",
    "previous_expected_value_difference": "prev. EV difference",
    "value_difference_transition":        "value diff. transition",
}

# ── Compact derivation formulas shown below each factor bar ──────────────────

FACTOR_FORMULAS = {
    # Response Interference (Stroop-Simon)
    "word_color_congruency":              "word = color",
    "location_response_congruency":       "location = response",
    "congruency_previous_trial":          "word[n−1] = color[n−1]",
    "response_transition":                "response[n] = response[n−1]",
    # Task Switching (RDK)
    "task_transition":                    "task[n] = task[n−1]",
    "current_stimulus_difficulty":        "1 − coh_{task[n]}[n]",
    "past_stimulus_difficulty":           "1 − coherence[n−1]",
    "n2_task_inhibition":                 "task[n] = task[n−2]",
    # Decision Making (Prospect Theory)
    "expected_value_difference":          "EV_L − EV_R",
    "gain_difference":                    "gain_L − gain_R",
    "loss_difference":                    "loss_L − loss_R",
    "dominance_relation":                 "gain_L ≥ gain_R, loss_L ≤ loss_R, p_L ≥ p_R",
    "previous_expected_value_difference": "EV_diff[n−1]",
    "value_difference_transition":        "sign(EV_diff[n]) = sign(EV_diff[n−1])",
}

# Factors to omit from Panel B entirely
EXCLUDE_FACTORS = {"previous_expected_value_difference", "value_difference_transition", "gain_difference"}

# Display-name ordering for benchmarks (detected by substring match on display_name)
BENCHMARK_ORDER_KEYS = ["Stroop", "RDK", "Prospect"]
BENCHMARK_XTICKLABELS = {
    "Stroop":   "Response\nInterference",
    "RDK":      "Task\nSwitching",
    "Prospect": "Decision\nMaking",
}


# ── Data loading helpers ──────────────────────────────────────────────────────

def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _sort_benchmarks(benchmarks_dict):
    """Return benchmark keys sorted by BENCHMARK_ORDER_KEYS."""
    ordered = []
    for key_fragment in BENCHMARK_ORDER_KEYS:
        for bk, bv in benchmarks_dict.items():
            dn = bv.get("display_name", bk)
            if key_fragment.lower() in dn.lower() or key_fragment.lower() in bk.lower():
                if bk not in ordered:
                    ordered.append(bk)
    # Append any remaining benchmarks not matched
    for bk in benchmarks_dict:
        if bk not in ordered:
            ordered.append(bk)
    return ordered


def _benchmark_xlabel(bk, bdata):
    dn = bdata.get("display_name", bk)
    for frag, label in BENCHMARK_XTICKLABELS.items():
        if frag.lower() in dn.lower() or frag.lower() in bk.lower():
            return label
    return dn


def _factor_key(factor_info):
    """Return the FACTOR_COLORS key: '<scope>_<class>'."""
    ftype = factor_info.get("type", "within_trial")       # e.g. 'within_trial' or 'window'
    fclass = factor_info.get("factor_class", "discrete")  # e.g. 'discrete' or 'continuous'
    return f"{ftype}_{fclass}"


def _is_continuous(factor_info):
    return factor_info.get("factor_class", "discrete") == "continuous"


def _factor_display(name):
    return FACTOR_DISPLAY_NAMES.get(name, name.replace("_", " "))


# ── Statistics helpers ────────────────────────────────────────────────────────

def _mean_se(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def _prf_stats(runs):
    """Return dict of metric -> (mean, se) over runs."""
    stats = {}
    for metric in ("precision", "recall", "f1"):
        vals = [r[metric] for r in runs if metric in r]
        stats[metric] = _mean_se(vals)
    return stats


def _factor_recovery(factor_name, factor_info, runs):
    """
    Compute mean and SE of recovery for one factor across runs.

    Discrete: level_recall from level_recovery_per_factor
    Continuous: absolute correlation from continuous_correlation_per_factor
    Missing data treated as 0.0.
    """
    if _is_continuous(factor_info):
        values = []
        for run in runs:
            corr_map = run.get("continuous_correlation_per_factor", {})
            val = corr_map.get(factor_name, 0.0)
            if val is None:
                val = 0.0
            values.append(abs(float(val)))
    else:
        values = []
        for run in runs:
            lvl_map = run.get("level_recovery_per_factor", {})
            factor_entry = lvl_map.get(factor_name, {})
            if isinstance(factor_entry, dict):
                val = factor_entry.get("level_recall", 0.0)
            else:
                val = float(factor_entry) if factor_entry is not None else 0.0
            if val is None:
                val = 0.0
            values.append(float(val))
    return _mean_se(values)


def _heatmap_matrix(benchmark_order, benchmarks_dict):
    """
    Build 2x2 matrix: rows = [within_trial, window], cols = [discrete, continuous].
    Cell value = fraction of (factor, run) pairs where recovery meets threshold:
      discrete: level_recall >= 0.95
      continuous: |correlation| >= 0.7
    """
    counts = np.zeros((2, 2), dtype=float)
    totals = np.zeros((2, 2), dtype=float)

    row_map = {"within_trial": 0, "window": 1}
    col_map = {"discrete": 0, "continuous": 1}

    for bk in benchmark_order:
        bdata = benchmarks_dict[bk]
        runs = bdata.get("runs", [])
        for factor_info in bdata.get("ground_truth_factors", []):
            fname = factor_info["name"]
            ftype = factor_info.get("type", "within_trial")
            fclass = factor_info.get("factor_class", "discrete")
            r = row_map.get(ftype)
            c = col_map.get(fclass)
            if r is None or c is None:
                continue
            for run in runs:
                totals[r, c] += 1
                if fclass == "continuous":
                    corr_map = run.get("continuous_correlation_per_factor", {})
                    val = corr_map.get(fname, 0.0)
                    if val is None:
                        val = 0.0
                    recovered = abs(float(val)) >= 0.7
                else:
                    lvl_map = run.get("level_recovery_per_factor", {})
                    entry = lvl_map.get(fname, {})
                    if isinstance(entry, dict):
                        val = entry.get("level_recall", 0.0)
                    else:
                        val = float(entry) if entry is not None else 0.0
                    if val is None:
                        val = 0.0
                    recovered = float(val) >= 0.95
                if recovered:
                    counts[r, c] += 1

    with np.errstate(invalid="ignore"):
        matrix = np.where(totals > 0, counts / totals, 0.0)
    return matrix


# ── Build Panel B factor rows ─────────────────────────────────────────────────

def _build_factor_rows(benchmark_order, benchmarks_dict):
    """
    Return list of dicts, one per factor, in benchmark order.
    Within each benchmark, factors follow ground_truth_factors order.
    """
    rows = []
    for bk in benchmark_order:
        bdata = benchmarks_dict[bk]
        runs = bdata.get("runs", [])
        xlabel = _benchmark_xlabel(bk, bdata)
        for factor_info in bdata.get("ground_truth_factors", []):
            fname = factor_info["name"]
            if fname in EXCLUDE_FACTORS:
                continue
            mean, se = _factor_recovery(fname, factor_info, runs)
            rows.append({
                "name":    fname,
                "label":   _factor_display(fname),
                "bench":   xlabel,
                "type":    factor_info.get("type", "within_trial"),
                "class":   factor_info.get("factor_class", "discrete"),
                "mean":    mean,
                "se":      se,
            })
    return rows


# ── Main plotting function ────────────────────────────────────────────────────

def make_figure(results_path, output_dir, show=True):
    style.apply_style()

    data = _load_json(results_path)
    benchmarks_dict = data["benchmarks"]
    benchmark_order = _sort_benchmarks(benchmarks_dict)

    # ── Pre-compute statistics ────────────────────────────────────────────────

    bench_xlabels = [_benchmark_xlabel(bk, benchmarks_dict[bk]) for bk in benchmark_order]
    bench_prf = {bk: _prf_stats(benchmarks_dict[bk].get("runs", [])) for bk in benchmark_order}
    bench_n_gt = []
    for bk in benchmark_order:
        runs = benchmarks_dict[bk].get("runs", [])
        if runs:
            bench_n_gt.append(runs[0].get("n_ground_truth",
                              len(benchmarks_dict[bk].get("ground_truth_factors", []))))
        else:
            bench_n_gt.append(len(benchmarks_dict[bk].get("ground_truth_factors", [])))

    factor_rows = _build_factor_rows(benchmark_order, benchmarks_dict)
    heatmap_matrix = _heatmap_matrix(benchmark_order, benchmarks_dict)

    # ── Figure layout ─────────────────────────────────────────────────────────

    fig = plt.figure(figsize=(style.W2, 6.5))

    outer = gridspec.GridSpec(1, 2, figure=fig,
                              width_ratios=[2.6, 4.2], wspace=0.90)

    left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0],
        height_ratios=[1.6, 1.6], hspace=0.65)

    ax_a = fig.add_subplot(left[0])
    ax_c = fig.add_subplot(left[1])
    ax_b = fig.add_subplot(outer[1])

    # ═════════════════════════════════════════════════════════════════════════
    # Panel A — Benchmark P / R / F1
    # ═════════════════════════════════════════════════════════════════════════

    metrics = ["precision", "recall", "f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    n_bench = len(benchmark_order)
    bar_w = 0.22
    group_gap = 1.05
    x_centers = np.arange(n_bench) * group_gap

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - 1) * bar_w
        means = [bench_prf[bk][metric][0] for bk in benchmark_order]
        ses   = [bench_prf[bk][metric][1] for bk in benchmark_order]
        ax_a.bar(x_centers + offset, means, bar_w,
                 color=style.METRIC_COLORS[metric], label=label,
                 yerr=ses, error_kw=dict(lw=0.8, capsize=2.5, capthick=0.8),
                 zorder=3)

    ax_a.set_xticks(x_centers)
    ax_a.set_xticklabels(bench_xlabels, fontsize=style.FS_TICK)
    ax_a.set_ylabel("Score", fontsize=style.FS_LABEL)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.legend(loc="upper right", fontsize=style.FS_ANNOT,
                frameon=False, ncol=1, handlelength=1.0)
    style.despine(ax_a)
    style.panel_label(ax_a, "A")

    # ═════════════════════════════════════════════════════════════════════════
    # Panel C — 2x2 Factor-type recall matrix
    # ═════════════════════════════════════════════════════════════════════════

    # Build per-cell RGBA using flat factor-type colors
    _c_color_keys = [
        ['within_trial_discrete', 'within_trial_continuous'],
        ['window_discrete',       'window_continuous'],
    ]
    cell_rgba = np.zeros((2, 2, 4))
    for r in range(2):
        for c in range(2):
            hex_color = style.FACTOR_COLORS[_c_color_keys[r][c]]
            cell_rgba[r, c, 0] = int(hex_color[1:3], 16) / 255.0
            cell_rgba[r, c, 1] = int(hex_color[3:5], 16) / 255.0
            cell_rgba[r, c, 2] = int(hex_color[5:7], 16) / 255.0
            cell_rgba[r, c, 3] = 1.0

    ax_c.imshow(cell_rgba, aspect="auto", interpolation="nearest")

    for row in range(2):
        for col in range(2):
            val = heatmap_matrix[row, col]
            rgb = cell_rgba[row, col, :3]
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            text_color = "white" if lum < 0.55 else "#333333"
            ax_c.text(col, row, f"{val:.2f}",
                      ha="center", va="center",
                      fontsize=style.FS_LABEL, color=text_color,
                      fontweight="bold")

    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["Discrete", "Continuous"], fontsize=style.FS_TICK)
    ax_c.set_yticks([0, 1])
    ax_c.set_yticklabels(["Within-trial", "Across-trial"], fontsize=style.FS_TICK)
    ax_c.set_title("Recovery score  (recall  |  |Spearman ρ|)", fontsize=style.FS_LABEL, pad=6)
    ax_c.set_xlabel("Factor class", fontsize=style.FS_LABEL, labelpad=4)
    ax_c.set_ylabel("Factor scope", fontsize=style.FS_LABEL, labelpad=4)
    ax_c.tick_params(length=0)
    for sp in ax_c.spines.values():
        sp.set_visible(False)

    style.panel_label(ax_c, "C", x=-0.28, y=1.18)

    # ═════════════════════════════════════════════════════════════════════════
    # Panel B — Per-factor level / correlation recovery
    # ═════════════════════════════════════════════════════════════════════════

    n_factors = len(factor_rows)
    BAR_SPACING = 0.75
    y_positions = np.arange(n_factors) * BAR_SPACING
    bar_height  = 0.44

    for i, row in enumerate(factor_rows):
        color_key = f"{row['type']}_{row['class']}"
        color = style.FACTOR_COLORS.get(color_key, "#999999")
        is_cont = row["class"] == "continuous"

        ax_b.barh(y_positions[i], row["mean"], height=bar_height,
                  color=color, alpha=0.90, zorder=3)

        ax_b.errorbar(row["mean"], y_positions[i], xerr=row["se"], fmt="none",
                      color="#444444", lw=0.8, capsize=2.5, capthick=0.8,
                      zorder=5)

    # Y-axis: factor labels colored to match their bars
    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([r["label"] for r in factor_rows], fontsize=style.FS_ANNOT)
    for tick, row in zip(ax_b.get_yticklabels(), factor_rows):
        color_key = f"{row['type']}_{row['class']}"
        tick.set_color(style.FACTOR_COLORS.get(color_key, "#999999"))
    ax_b.invert_yaxis()

    # Formula annotations below each factor name
    for i, row in enumerate(factor_rows):
        formula = FACTOR_FORMULAS.get(row["name"], "")
        if formula:
            ax_b.text(
                0.0, y_positions[i] + 0.28,
                formula,
                transform=style.blended(ax_b),
                ha="right", va="top",
                fontsize=style.FS_CODE,
                fontstyle="italic",
                color="#888888",
                clip_on=False,
            )

    # Benchmark section separators + right-side labels
    sep_color = "#BBBBBB"

    # Identify group boundaries
    current_bench = None
    group_start = 0
    groups = []
    for i, row in enumerate(factor_rows):
        if row["bench"] != current_bench:
            if current_bench is not None:
                groups.append((current_bench, group_start, i - 1))
            current_bench = row["bench"]
            group_start = i
    if current_bench is not None:
        groups.append((current_bench, group_start, len(factor_rows) - 1))

    for bench_name, start, end in groups:
        sep_y = y_positions[start] - 0.38

        if start > 0:
            ax_b.axhline(sep_y, color=sep_color, lw=0.8, zorder=0,
                         xmin=0, xmax=1)

        ax_b.text(0.98, sep_y,
                  bench_name,
                  transform=style.blended(ax_b),
                  ha="right", va="bottom",
                  fontsize=style.FS_ANNOT, fontweight="bold", color="#444444",
                  clip_on=False)

    ax_b.set_xlabel(
        "Recovery score  (recall  |  |Spearman ρ|)",
        fontsize=style.FS_LABEL)
    ax_b.set_xlim(0, 1.0)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.axvline(0, color="#999999", lw=0.6, zorder=0)

    style.despine(ax_b, left=True)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(left=False)

    # Legend for factor types
    legend_handles = [
        mpatches.Patch(facecolor=style.FACTOR_COLORS["within_trial_discrete"],
                       label="Within-trial, discrete"),
        mpatches.Patch(facecolor=style.FACTOR_COLORS["window_discrete"],
                       label="Across-trial, discrete"),
        mpatches.Patch(facecolor=style.FACTOR_COLORS["within_trial_continuous"],
                       hatch=style.CONTINUOUS_HATCH,
                       edgecolor=style.FACTOR_COLORS["within_trial_continuous"],
                       label="Within-trial, continuous (|ρ|)"),
        mpatches.Patch(facecolor=style.FACTOR_COLORS["window_continuous"],
                       hatch=style.CONTINUOUS_HATCH,
                       edgecolor=style.FACTOR_COLORS["window_continuous"],
                       label="Across-trial, continuous (|ρ|)"),
    ]
    ax_b.legend(handles=legend_handles, loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                bbox_transform=ax_b.transAxes,
                ncol=2,
                fontsize=style.FS_ANNOT, frameon=False,
                handlelength=1.2, handleheight=0.9,
                columnspacing=1.0)

    style.panel_label(ax_b, "B", x=-0.04)

    # ── Save ─────────────────────────────────────────────────────────────────

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "figure2.png")
    pdf_path = os.path.join(output_dir, "figure2.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")

    if show:
        plt.show()

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Figure 2 (panels A, B, C) from synthetic benchmark results.")
    parser.add_argument(
        "--results",
        default="results/aggregated/synthetic_results.json",
        help="Path to synthetic_results.json (default: %(default)s)")
    parser.add_argument(
        "--output-dir",
        default="plots/output",
        help="Directory for output PNG/PDF (default: %(default)s)")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Suppress plt.show()")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    make_figure(
        results_path=args.results,
        output_dir=args.output_dir,
        show=not args.no_show,
    )
