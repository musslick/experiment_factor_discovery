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

# ── Paths ─────────────────────────────────────────────────────────────────────
# Add the repo root (parent of plots/) so src.* and produce_results_empirical
# are importable when computing participant-level data on-the-fly.
_PLOTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_PLOTS_DIR)
sys.path.insert(0, _PLOTS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
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

def _load_cfg_meta(config_path: str) -> dict:
    """
    Read key dataset metadata directly from the YAML config without importing
    any src.* modules (avoids sweetpea / statsmodels version conflicts in the
    plotting environment).
    Returns a plain dict with 'path', 'participant_id_column', 'outcome_variable',
    'null_formula', and 'hidden_factor_columns', or {} on failure.
    """
    if not config_path:
        return {}
    try:
        import yaml
        with open(config_path) as fh:
            raw = yaml.safe_load(fh) or {}
        ds = raw.get("dataset", {})
        hidden_cols = []
        for hf in ds.get("hidden_factors", []):
            if isinstance(hf, dict):
                hidden_cols.append(hf.get("column") or hf.get("name", ""))
        return {
            "path":                  ds.get("path", ""),
            "participant_id_column": ds.get("participant_id_column", "participant_id"),
            "outcome_variable":      ds.get("outcome_variable", ""),
            "null_formula":          ds.get("null_formula", ""),
            "hidden_factor_columns": hidden_cols,
        }
    except Exception as exc:
        print(f"  [WARN] Could not read config {config_path}: {exc}")
        return {}


_SANDBOX_HARNESS = """\
import json, sys, collections

{predicate_code}

data         = json.loads(sys.stdin.buffer.read().decode('utf-8'))
rows         = data['rows']
factor_type  = data['factor_type']
window_width = data.get('window_width', 2)
n            = data['n']

results = [None] * n
by_group = collections.defaultdict(list)
for row in rows:
    by_group[(row['participant_id'], row.get('block_index', -1))].append(row)

try:
    for group_key in sorted(by_group.keys()):
        p_rows = sorted(by_group[group_key], key=lambda r: r['trial_index'])
        for i, row in enumerate(p_rows):
            orig = row['__idx__']
            if factor_type == 'within_trial':
                results[orig] = compute_factor(row)
            else:
                if i < window_width - 1:
                    results[orig] = None
                else:
                    results[orig] = compute_factor(p_rows[i - window_width + 1 : i + 1])
except Exception:
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print(json.dumps(results))
"""


def _compute_pld_for_plot(factor_dict: dict, meta: dict, outcome_label: str):
    """
    Compute participant-level data on-the-fly using only stdlib + pandas.
    Runs the predicate compute_code in a subprocess harness (same mechanism as
    the discovery pipeline sandbox) without importing any src.* modules.

    Returns the pld dict or None on failure.
    """
    import re, json, os, subprocess, tempfile
    import pandas as pd

    compute_code = factor_dict.get("compute_code", "")
    csv_path     = meta.get("path", "")
    if not compute_code or not csv_path:
        return None

    pid_col     = meta.get("participant_id_column", "participant_id")
    outcome_var = meta.get("outcome_variable", "")
    if not outcome_var:
        return None

    # ── Load CSV and pre-process ──────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  [WARN] Could not read {csv_path}: {exc}")
        return None

    if pid_col in df.columns and pid_col != "participant_id":
        df = df.rename(columns={pid_col: "participant_id"})
    if "participant_id" not in df.columns:
        return None
    if outcome_var not in df.columns:
        return None
    if "trial_index" not in df.columns:
        df["trial_index"] = df.groupby("participant_id", sort=False).cumcount()

    df = df.reset_index(drop=True)

    # ── Infer factor type / window width from predicate signature ─────────────
    name   = factor_dict.get("name", "factor")
    levels = factor_dict.get("levels", [])

    # ── Infer factor_type: check compute_factor signature, then fall back to
    #    _parent_compute_factor (used by contrast/level-vs-rest derivatives).
    sig_m = re.search(r"def\s+compute_factor\s*\(\s*(\w+)", compute_code)
    par_m = re.search(r"def\s+_parent_compute_factor\s*\(\s*(\w+)", compute_code)
    factor_type = "within_trial"
    for m in (sig_m, par_m):
        if m and m.group(1) in ("window", "w"):
            factor_type = "window"
            break

    # ── Infer window_width from literal indices used with the window variable.
    #    Positive index k → need at least k+1 elements.
    #    Negative index -k → need at least k elements (wraps from end).
    required = 2
    for idx_str in re.findall(r"\[(-?\d+)\]", compute_code):
        idx = int(idx_str)
        required = max(required, idx + 1 if idx >= 0 else abs(idx))
    window_width = required if factor_type == "window" else 1

    # ── Serialize DataFrame for subprocess harness ────────────────────────────
    records = json.loads(df.to_json(orient="records", default_handler=str))
    for i, row in enumerate(records):
        row["__idx__"] = i
    payload = json.dumps({
        "rows": records, "factor_type": factor_type,
        "window_width": window_width, "n": len(df),
    })

    harness_src = _SANDBOX_HARNESS.format(predicate_code=compute_code)

    fd, harness_path = tempfile.mkstemp(suffix=".py", prefix="fig4_harness_")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(harness_src)
        try:
            proc = subprocess.run(
                [sys.executable, harness_path],
                input=payload.encode("utf-8"),
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            print(f"  [WARN] Sandbox timed out for {name}")
            return None
    finally:
        os.unlink(harness_path)

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[:300]
        print(f"  [WARN] Sandbox error for {name}: {err}")
        return None

    try:
        raw_values = json.loads(proc.stdout.decode("utf-8"))
    except Exception:
        return None

    # ── Attach results and aggregate ──────────────────────────────────────────
    df["_fc"] = raw_values
    df = df[df["_fc"].notna() & df[outcome_var].notna()]
    if df.empty:
        return None

    unique_lvls = list(levels) if levels else sorted(df["_fc"].unique().tolist(), key=str)
    part_means  = {str(lv): [] for lv in unique_lvls}
    for (pid, lv), val in df.groupby(["participant_id", "_fc"])[outcome_var].mean().items():
        key = str(lv)
        if key in part_means:
            part_means[key].append(float(val))

    # Drop levels with no participant data
    unique_lvls = [lv for lv in unique_lvls if part_means.get(str(lv))]

    group_means, group_sems = [], []
    for lv in unique_lvls:
        vals = part_means[str(lv)]
        arr  = np.array(vals)
        group_means.append(float(np.mean(arr)))
        group_sems.append(
            float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        )

    return {
        "outcome_label":     outcome_label,
        "levels":            [str(lv) for lv in unique_lvls],
        "group_means":       group_means,
        "group_sems":        group_sems,
        "participant_means": part_means,
    }


def _collect_discovery_cells(emp_data: dict) -> list:
    """Return list of cell dicts, one per (dataset × top-K factor)."""
    cells = []
    datasets = emp_data.get("datasets", {})
    for ds_name, ds in datasets.items():
        display_name = ds.get("display_name", ds_name)
        config_path  = ds.get("config_path", "")
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
            pld = factor.get("participant_level_data")
            if not pld and config_path:
                # participant_level_data was not stored (trial_index bug); recompute now
                meta = _load_cfg_meta(config_path)
                if not null_formula:
                    null_formula = meta.get("null_formula", "")
                outcome_var = meta.get("outcome_variable", "")
                out_label   = _outcome_label(outcome_var, null_formula) if outcome_var else ""
                pld = _compute_pld_for_plot(factor, meta, out_label)

            cells.append({
                "ds_name":      ds_name,
                "display_name": display_name,
                "null_formula": null_formula,
                "factor":       {**factor, "participant_level_data": pld},
            })
    return cells


def _plot_discovery_cell(fig, ax_eff, ax_code, cell: dict):
    """Render one discovery row: effect plot (left) + code inset (right)."""
    disc = cell["factor"]
    display_name = cell["display_name"]
    null_formula = cell.get("null_formula", "")

    pl_data = disc.get("participant_level_data") or {}
    outcome_label = pl_data.get("outcome_label", "")
    levels = pl_data.get("levels", [])
    group_means = np.array(pl_data.get("group_means", []))
    group_sems  = np.array(pl_data.get("group_sems", []))
    participant_means = pl_data.get("participant_means", {})

    n_levels = len(levels)
    x_pos = np.arange(n_levels, dtype=float)

    factor_name_raw = disc.get("name", "")
    llm_name = disc.get("llm_name", "")
    title_str = llm_name if llm_name else _factor_display(factor_name_raw)

    # ── Effect plot (left panel) ───────────────────────────────────────────────
    rng = np.random.RandomState(77)

    all_pts = []
    for lvl in levels:
        pts = participant_means.get(lvl, [])
        all_pts.extend(pts)

    for xi, lvl in zip(x_pos, levels):
        pts = np.array(participant_means.get(lvl, []), dtype=float)
        if len(pts) > 0:
            jitter = rng.normal(0, 0.07, len(pts))
            ax_eff.scatter(xi + jitter, pts,
                           s=5, color="#AAAAAA", alpha=0.40,
                           linewidths=0, zorder=1)

    if len(group_means) == n_levels and n_levels > 0:
        ax_eff.errorbar(x_pos, group_means, yerr=group_sems,
                        fmt="o", color=style.NOVEL_COLOR,
                        markersize=6.5, markeredgewidth=0,
                        lw=1.6, capsize=3.5, capthick=1.2, zorder=3)

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

    ax_eff.text(0.5, 0.97, title_str,
                transform=ax_eff.transAxes,
                ha="center", va="top",
                fontsize=style.FS_LABEL, fontweight="bold")

    ax_eff.text(0.5, 0.88, display_name,
                transform=ax_eff.transAxes,
                ha="center", va="top",
                fontsize=style.FS_ANNOT, color="#666666", style="italic")

    style.despine(ax_eff)

    # ── Code inset (right panel) ───────────────────────────────────────────────
    ax_code.set_xlim(0, 1)
    ax_code.set_ylim(0, 1)
    ax_code.axis("off")

    ax_code.add_patch(Rectangle(
        (0, 0), 1, 1,
        facecolor=CODE_BG, edgecolor=CODE_EDG,
        lw=0.5, transform=ax_code.transAxes, clip_on=False
    ))

    ax_code.text(0.04, 0.96, "compute_factor()",
                 transform=ax_code.transAxes, ha="left", va="top",
                 fontsize=style.FS_CODE, color="#999999",
                 fontfamily="monospace", clip_on=True)

    compute_code = disc.get("compute_code", "")
    ax_code.text(0.04, 0.86, compute_code,
                 transform=ax_code.transAxes, ha="left", va="top",
                 fontsize=4.0, fontfamily="monospace",
                 color="#1A1A1A", linespacing=1.4, clip_on=True)

    delta_ll = disc.get("validation_improvement", float("nan"))
    ax_code.text(0.04, 0.04,
                 f"ΔLL = {delta_ll:.2f}  ·  held-out set",
                 transform=ax_code.transAxes, ha="left", va="bottom",
                 fontsize=style.FS_ANNOT, color="#444444", clip_on=False)


def make_figure4(emp_data: dict, out_dir: str, show: bool):
    cells = _collect_discovery_cells(emp_data)
    if not cells:
        print("No discovery cells to plot for Figure 4 — skipping.")
        return

    total_cells = len(cells)
    row_height  = 2.5   # inches per dataset row

    fig = plt.figure(figsize=(style.W2, row_height * total_cells))
    gs  = gridspec.GridSpec(
        total_cells, 2,
        figure=fig,
        width_ratios=[0.75, 1.25],
        hspace=0.45,
        wspace=0.30,
    )

    for idx, cell in enumerate(cells):
        ax_eff  = fig.add_subplot(gs[idx, 0])
        ax_code = fig.add_subplot(gs[idx, 1])
        _plot_discovery_cell(fig, ax_eff, ax_code, cell)

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
