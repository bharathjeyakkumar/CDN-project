"""
plotter.py -- paper figures from a results CSV (no QAOA, no channels, no PennyLane needed).

Input : the long-format CSV written by qaoa_beam_assignment_datasets.py
        (one row per snapshot and method; columns used: set, scenario, snapshot, method,
         gain_db / gain_linear, interference_db, jain, and optionally users, beams, rf_chains, backend).
Output: one figure per data set ("set" column): three equal panels side by side
            (a) total beam gain (dB)   (b) interference   (c) Jain's fairness index
        x axis = the snapshots of that set, labelled with the scenario names.

All style and layout settings are in the STYLE section below, so a change of look is
"edit, run this file again" without re-running any simulation:

    python plotter.py results/run_20261024_153012_pennylane_6u2b2rf/qaoa_dataset_results.csv
    python plotter.py my_results.csv --metrics gain fairness --formats pdf png
    python plotter.py my_results.csv --methods Greedy OMP QAOA-XY Optimal --linear-gain --show

By default the figures go into a NEW folder  plots_<date>_<time>/  next to the CSV, and an existing file is
never overwritten (a numeric suffix is added), so earlier figures are always kept.

From Python:
    import plotter
    paths, figs = plotter.plot_from_csv("results.csv", out_dir="some/folder", close=False)
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ======================================================
# STYLE (edit here)
# ======================================================

PAPER_RC = {
    # Base font (falls back to the next installed Times-like font if Times New Roman is missing)
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,

    # Axes
    "axes.linewidth": 0.5,

    # Ticks
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,

    # Legend
    "legend.framealpha": 0.9,

    # Lines and markers
    "lines.linewidth": 0.75,
    "lines.markersize": 0.75,

    # Saving
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

FIG_WIDTH_IN = 7.0        # width of the THREE-panel figure = full text width of a two-column paper (figure*)
FIG_HEIGHT_IN = 2.35      # height of every figure; each panel is one third of FIG_WIDTH_IN wide
PAD_INCHES = 0.02         # margin kept by the tight bounding box (keeps the PDF just under FIG_WIDTH_IN)
MARKER_SIZE = 2.5         # explicit marker size (the rcParams value of 0.75 pt would be invisible)
PANEL_WSPACE = 0.42       # horizontal gap between panels (fraction of one panel width)
LEFT_MARGIN_IN, RIGHT_MARGIN_IN = 0.455, 0.035   # = 0.065 and 0.995 of the 7 in three-panel figure
TOP_FRAC, BOTTOM_FRAC = 0.86, 0.30
LEGEND_ANCHOR_Y = 0.93

FORMATS = ("pdf",)        # add "png" for a PNG copy
DEFAULT_METHODS = ["Greedy", "OMP", "QAOA-XY"]   # "Optimal" is in the CSV too; add it with --methods
COLORS = {"Greedy": "tab:blue", "OMP": "tab:orange", "QAOA-XY": "tab:green", "Optimal": "tab:red"}
MARKERS = {"Greedy": "o", "OMP": "s", "QAOA-XY": "^", "Optimal": "D"}
LINESTYLES = {"Greedy": "--", "OMP": ":", "QAOA-XY": "-", "Optimal": "-."}

# panel definitions: name -> (title, y label, CSV column, upper y limit or None)
METRICS = {
    "gain":         dict(title="Total beam gain", ylabel="Gain (dB)", col="gain_db", ytop=None),
    "interference": dict(title="Interference", ylabel="Interference-to-noise (dB)", col="interference_db", ytop=None),
    "fairness":     dict(title="Fairness", ylabel="Jain's index", col="jain", ytop=1.02),
}


# ======================================================
# DATA
# ======================================================

def load_results(csv_path):
    df = pd.read_csv(csv_path)
    missing = [c for c in ("snapshot", "set", "scenario", "method") if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: required column(s) {missing} not found; columns are {list(df.columns)}")
    return df


def _column_for(df, metric, linear_gain):
    """CSV column that feeds a panel (gain_db is derived from gain_linear if only that is stored)."""
    if metric == "gain":
        if linear_gain:
            col = "gain_linear"
        else:
            if "gain_db" not in df.columns and "gain_linear" in df.columns:
                df["gain_db"] = 10 * np.log10(df["gain_linear"].clip(lower=1e-30))
            col = "gain_db"
    else:
        col = METRICS[metric]["col"]
    if col not in df.columns:
        raise ValueError(f"the '{metric}' panel needs column '{col}', which is not in the CSV "
                         f"(columns: {list(df.columns)}). It was probably written by an older version of the "
                         f"dataset script; re-run it, or leave this metric out with --metrics.")
    return col


def unique_path(path):
    """path, or path with _1, _2, ... added if it already exists (an old file is never overwritten)."""
    path = Path(path)
    if not path.exists():
        return path
    k = 1
    while True:
        cand = path.with_name(f"{path.stem}_{k}{path.suffix}")
        if not cand.exists():
            return cand
        k += 1


# ======================================================
# FIGURES
# ======================================================

def _figure(sub, metrics, cols, methods, linear_gain):
    """One figure from the rows of ONE data set. Panel width is always FIG_WIDTH_IN / 3."""
    n_panels = len(metrics)
    width = FIG_WIDTH_IN * n_panels / 3.0
    fig, axes = plt.subplots(1, n_panels, figsize=(width, FIG_HEIGHT_IN),
                             gridspec_kw=dict(wspace=PANEL_WSPACE), squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=LEFT_MARGIN_IN / width, right=1 - RIGHT_MARGIN_IN / width,
                        top=TOP_FRAC, bottom=BOTTOM_FRAC)

    snaps = np.sort(sub["snapshot"].unique())
    x = np.arange(1, len(snaps) + 1)
    labels = (sub.drop_duplicates("snapshot").set_index("snapshot").loc[snaps, "scenario"]
              .astype(str).tolist())

    for k, (ax, metric) in enumerate(zip(axes, metrics)):
        spec, col = METRICS[metric], cols[metric]
        for mth in methods:
            s = sub[sub["method"] == mth].groupby("snapshot")[col].mean().reindex(snaps)
            ax.plot(x, s.to_numpy(), color=COLORS.get(mth), marker=MARKERS.get(mth, "o"),
                    markersize=MARKER_SIZE, linestyle=LINESTYLES.get(mth, "-"), label=mth)
        title = spec["title"]
        if n_panels > 1:
            title = f"({chr(97 + k)}) {title}"
        ylabel = spec["ylabel"]
        if metric == "gain" and linear_gain:
            ylabel = "Gain"
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_xlim(0.5, len(snaps) + 0.5)
        ax.grid(True, linewidth=0.3, alpha=0.5)
        if spec["ytop"] is not None:
            ax.set_ylim(top=spec["ytop"])

    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y), ncol=len(names),
               frameon=True, borderpad=0.3, columnspacing=1.5, handlelength=2.0)
    return fig


def plot_from_csv(csv_path, out_dir=None, metrics=tuple(METRICS), methods=None, formats=None,
                  linear_gain=False, close=True):
    """
    Draw one figure per data set found in the CSV and save it.

    csv_path    results CSV (see the module docstring)
    out_dir     folder for the files; default: a NEW  plots_<date>_<time>/  next to the CSV
    metrics     any of "gain", "interference", "fairness" (default: all three = the paper figure)
    methods     rows of the "method" column to draw (default: Greedy, OMP, QAOA-XY)
    formats     file types, default FORMATS
    linear_gain gain panel in linear units instead of dB
    close       False keeps the figures open (for plt.show())

    Returns (list of written paths, list of figures).
    """
    df = load_results(csv_path)
    metrics = list(metrics)
    for m in metrics:
        if m not in METRICS:
            raise ValueError(f"unknown metric '{m}'; choose from {list(METRICS)}")
    cols = {m: _column_for(df, m, linear_gain) for m in metrics}
    methods = [m for m in (methods or DEFAULT_METHODS) if m in set(df["method"])]
    if not methods:
        raise ValueError(f"none of the requested methods is in the CSV (methods there: {sorted(set(df['method']))})")
    formats = tuple(formats or FORMATS)

    if out_dir is None:
        out_dir = Path(csv_path).resolve().parent / f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "" if metrics == list(METRICS) else "_" + "_".join(metrics)
    paths, figs = [], []
    with mpl.rc_context(PAPER_RC):                      # style applies only inside this block
        for g in dict.fromkeys(df["set"]):              # data sets in order of appearance
            sub = df[df["set"] == g]
            fig = _figure(sub, metrics, cols, methods, linear_gain)
            first = sub.iloc[0]
            tag = ""
            if all(c in sub.columns for c in ("users", "beams", "rf_chains")):
                tag = f"_{int(first['users'])}u{int(first['beams'])}b{int(first['rf_chains'])}rf"
            if "backend" in sub.columns:
                tag += f"_{first['backend']}"
            stem = f"{str(g).replace(' ', '')}{tag}{suffix}"
            for ext in formats:
                path = unique_path(out_dir / f"{stem}.{ext}")
                fig.savefig(path, pad_inches=PAD_INCHES)
                paths.append(path)
            figs.append(fig)
            if close:
                plt.close(fig)
    return paths, figs


# ======================================================
# COMMAND LINE
# ======================================================

def main():
    p = argparse.ArgumentParser(description="Paper figures (gain | interference | fairness) from a results CSV")
    p.add_argument("csv", help="results CSV written by qaoa_beam_assignment_datasets.py")
    p.add_argument("--out-dir", default=None, help="output folder (default: new plots_<time>/ next to the CSV)")
    p.add_argument("--metrics", nargs="+", choices=list(METRICS), default=list(METRICS),
                   help="panels to draw (default: all three in one figure)")
    p.add_argument("--methods", nargs="+", default=None, help="methods to draw (default: Greedy OMP QAOA-XY)")
    p.add_argument("--formats", nargs="+", default=None, help="file types, e.g. pdf png (default: pdf)")
    p.add_argument("--linear-gain", action="store_true", help="gain panel in linear units instead of dB")
    p.add_argument("--show", action="store_true", help="also open the figures on screen")
    a = p.parse_args()
    paths, _ = plot_from_csv(a.csv, a.out_dir, a.metrics, a.methods, a.formats, a.linear_gain, close=not a.show)
    for path in paths:
        print(f"Figure saved to {path}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()