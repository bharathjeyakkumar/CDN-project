"""
plotter.py -- publication-style figures from a results CSV.

Input:
    Long-format CSV written by qaoa_beam_assignment_datasets.py.
    Expected columns include set, scenario, snapshot, method, and the selected
    metric columns (gain_db or gain_linear, interference_db, jain).

Output:
    One separate figure per set and per metric. With three sets and three
    metrics, this produces nine figures:
        set1_gain, set1_fairness, set1_interference,
        set2_gain, set2_fairness, set2_interference,
        set3_gain, set3_fairness, set3_interference.

QAOA, channel, and simulation code are not imported or modified.

Usage:
    python plotter.py path/to/qaoa_dataset_results.csv
    python plotter.py results.csv --metrics gain fairness --formats pdf png
    python plotter.py results.csv --methods Greedy OMP QAOA-XY Optimal --linear-gain --show

Figures are saved in a new plots_<date>_<time> folder next to the CSV by default.
Existing output files are never overwritten.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ======================================================
# STYLE -- compact IEEE ICC-style single-panel figures
# ======================================================

PAPER_RC = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman", "Times", "Nimbus Roman",
        "Liberation Serif", "DejaVu Serif"
    ],
    "mathtext.fontset": "stix",

    # Slightly larger fonts for readability at 100% PDF zoom
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 12,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,

    # Axes
    "axes.linewidth": 0.6,

    # Ticks
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,

    # Legend
    "legend.framealpha": 0.95,

    # Lines and markers
    "lines.linewidth": 0.9,
    "lines.markersize": 2.0,

    # Saving
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Each output is a single graph, approximately one-third of a 7-inch,
# two-column-wide IEEE figure.
FIG_WIDTH_IN = 7.0 / 3.0
FIG_HEIGHT_IN = 2.65
PAD_INCHES = 0.01
MARKER_SIZE = 2.0

# Single-panel figures do not need inter-panel spacing.
LEFT_MARGIN_IN = 0.48
RIGHT_MARGIN_IN = 0.10
TOP_FRAC = 0.82
BOTTOM_FRAC = 0.34
LEGEND_ANCHOR_Y = 1.00

FORMATS = ("pdf",)
DEFAULT_METHODS = ["Greedy", "OMP", "QAOA-XY", "ILP"]
COLORS = {"Greedy": "tab:blue", "OMP": "tab:orange", "QAOA-XY": "tab:green", "ILP": "tab:red", "Optimal": "tab:red"}
MARKERS = {"Greedy": "o", "OMP": "s", "QAOA-XY": "^", "ILP": "D", "Optimal": "D"}
LINESTYLES = {"Greedy": "--", "OMP": ":", "QAOA-XY": "-", "ILP": "-.", "Optimal": "-."}


METRICS = {
    "gain": dict(
        title="Total beam gain",
        ylabel="Gain (dB)",
        col="gain_db",
        ytop=None
    ),
    "interference": dict(
        title="Interference",
        ylabel="Interference(dB)",
        col="interference_db",
        ytop=None
    ),
    "fairness": dict(
        title="Fairness",
        ylabel="Jain's index",
        col="jain",
        ytop=1.02
    ),
}


# ======================================================
# DATA
# ======================================================

def load_results(csv_path):
    df = pd.read_csv(csv_path)

    required = ("snapshot", "set", "scenario", "method")
    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{csv_path}: required column(s) {missing} not found; "
            f"columns are {list(df.columns)}"
        )

    return df


def _column_for(df, metric, linear_gain):
    """Select the CSV column used for a metric."""

    if metric == "gain":
        if linear_gain:
            col = "gain_linear"
        else:
            if "gain_db" not in df.columns and "gain_linear" in df.columns:
                df["gain_db"] = 10 * np.log10(
                    pd.to_numeric(
                        df["gain_linear"],
                        errors="coerce"
                    ).clip(lower=1e-30)
                )

            col = "gain_db"

    else:
        col = METRICS[metric]["col"]

    if col not in df.columns:
        raise ValueError(
            f"The '{metric}' figure needs column '{col}', "
            f"which is not in the CSV. "
            f"Available columns: {list(df.columns)}"
        )

    return col


def unique_path(path):
    """Return a non-existing path, adding _1, _2, ... if necessary."""

    path = Path(path)

    if not path.exists():
        return path

    k = 1

    while True:
        candidate = path.with_name(
            f"{path.stem}_{k}{path.suffix}"
        )

        if not candidate.exists():
            return candidate

        k += 1


def _safe_tag(value):
    """Make a set name safe for use in a filename."""

    text = str(value).strip().replace(" ", "")

    return "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in text
    )


# ======================================================
# FIGURES
# ======================================================

def _figure(sub, metric, col, methods, linear_gain):
    """Create one single-metric figure for one data set."""

    spec = METRICS[metric]

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN)
    )

    fig.subplots_adjust(
        left=LEFT_MARGIN_IN / FIG_WIDTH_IN,
        right=1.0 - RIGHT_MARGIN_IN / FIG_WIDTH_IN,
        top=TOP_FRAC,
        bottom=BOTTOM_FRAC,
    )

    snaps = np.sort(
        sub["snapshot"].dropna().unique()
    )

    x = np.arange(1, len(snaps) + 1)

    # Use one scenario label per snapshot, in snapshot order.
    scenario_by_snapshot = (
        sub.drop_duplicates("snapshot")
        .set_index("snapshot")["scenario"]
    )

    labels = (
        r'$x_1$', r'$x_2$', r'$x_3$',r'$x_4$', r'$x_5$', r'$x_6$',r'$x_7$', r'$x_8$', r'$x_9$',r'$x_{10}$'
    )

    # Plot each selected method.
    for method in methods:

        series = (
            sub[sub["method"] == method]
            .groupby("snapshot")[col]
            .mean()
            .reindex(snaps)
        )

        ax.plot(
            x,
            series.to_numpy(),
            color=COLORS.get(method, None),
            marker=MARKERS.get(method, "o"),
            markersize=MARKER_SIZE,
            linestyle=LINESTYLES.get(method, "-"),
            linewidth=0.9,
            label=method,
        )

    # Axis labels.
    # The graph title is deliberately removed.
    ylabel = spec["ylabel"]

    if metric == "gain" and linear_gain:
        ylabel = "Gain"

    ax.set_xlabel(
        "Scenario",
        labelpad=3
    )

    ax.set_ylabel(
        ylabel,
        labelpad=3
    )

    # Snapshot/scenario labels.
    ax.set_xticks(x)

    ax.set_xticklabels(
        labels
    )

    ax.set_xlim(
        0.5,
        len(snaps) + 0.5
    )

    # Grid.
    ax.grid(
        True,
        linewidth=0.3,
        alpha=0.5
    )

    # Apply an upper limit where specified.
    if spec["ytop"] is not None:
        ax.set_ylim(
            top=spec["ytop"]
        )

    # Legend.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
        ncol=max(1, len(methods)),
        frameon=True,
        borderpad=0.25,
        columnspacing=0.8,
        handlelength=1.5,
        handletextpad=0.4,
    )

    return fig


# ======================================================
# PLOT FROM CSV
# ======================================================

def plot_from_csv(
    csv_path,
    out_dir=None,
    metrics=tuple(METRICS),
    methods=None,
    formats=None,
    linear_gain=False,
    close=True,
):
    """
    Draw one figure per set and per metric.

    csv_path:
        Results CSV.

    out_dir:
        Output folder; defaults to a new plots_<timestamp> folder.

    metrics:
        Any of "gain", "interference", "fairness".

    methods:
        Methods to plot; defaults to Greedy, OMP, QAOA-XY.

    formats:
        Output formats; defaults to PDF.

    linear_gain:
        Plot gain in linear units instead of dB.

    close:
        Close figures after saving.

    Returns:
        (list of written paths, list of figures)
    """

    df = load_results(csv_path)

    metrics = list(metrics)

    for metric in metrics:
        if metric not in METRICS:
            raise ValueError(
                f"Unknown metric '{metric}'; "
                f"choose from {list(METRICS)}"
            )

    cols = {
        metric: _column_for(
            df,
            metric,
            linear_gain
        )
        for metric in metrics
    }

    available_methods = set(
        df["method"].dropna().astype(str)
    )

    methods = [
        method
        for method in (methods or DEFAULT_METHODS)
        if method in available_methods
    ]

    if not methods:
        raise ValueError(
            "None of the requested methods exists in the CSV. "
            f"Methods found: {sorted(available_methods)}"
        )

    formats = tuple(
        formats or FORMATS
    )

    if out_dir is None:
        out_dir = (
            Path(csv_path).resolve().parent
            / f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    out_dir = Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    paths = []
    figures = []

    with mpl.rc_context(PAPER_RC):

        # Each set gets an individual figure for each requested metric.
        for set_value in dict.fromkeys(
            df["set"].dropna()
        ):

            subset = df[
                df["set"] == set_value
            ]

            set_tag = _safe_tag(set_value)

            for metric in metrics:

                fig = _figure(
                    subset,
                    metric,
                    cols[metric],
                    methods,
                    linear_gain
                )

                # Include configuration details in the filename
                # when those columns are available.
                tag = ""

                if all(
                    column in subset.columns
                    for column in ("users", "beams", "rf_chains")
                ):

                    first = subset.iloc[0]

                    try:
                        tag = (
                            f"_{int(first['users'])}u"
                            f"{int(first['beams'])}b"
                            f"{int(first['rf_chains'])}rf"
                        )

                    except (TypeError, ValueError):
                        tag = ""

                if "backend" in subset.columns and not subset.empty:

                    backend = _safe_tag(
                        subset.iloc[0]["backend"]
                    )

                    if backend and backend.lower() != "nan":
                        tag += f"_{backend}"

                stem = f"{set_tag}_{metric}{tag}"

                # Save each requested output format.
                for extension in formats:

                    output_path = unique_path(
                        out_dir
                        / f"{stem}.{extension.lstrip('.')}"
                    )

                    fig.savefig(
                        output_path,
                        pad_inches=PAD_INCHES
                    )

                    paths.append(output_path)

                figures.append(fig)

                if close:
                    plt.close(fig)

    return paths, figures


# ======================================================
# COMMAND LINE
# ======================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Create separate gain, interference, and fairness "
            "figures from a results CSV."
        )
    )

    parser.add_argument(
        "csv",
        help="CSV written by qaoa_beam_assignment_datasets.py"
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output folder "
            "(default: new plots_<time>/ folder beside the CSV)"
        )
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=list(METRICS),
        default=list(METRICS),
        help=(
            "Metrics to plot "
            "(default: all three, each in its own figure)"
        )
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to plot (default: Greedy OMP QAOA-XY)"
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Output formats, e.g. pdf png (default: pdf)"
    )

    parser.add_argument(
        "--linear-gain",
        action="store_true",
        help="Plot gain in linear units instead of dB"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the figures on screen"
    )

    args = parser.parse_args()

    paths, _ = plot_from_csv(
        args.csv,
        out_dir=args.out_dir,
        metrics=args.metrics,
        methods=args.methods,
        formats=args.formats,
        linear_gain=args.linear_gain,
        close=not args.show,
    )

    for path in paths:
        print(
            f"Figure saved to {path}"
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()