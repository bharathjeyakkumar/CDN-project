
"""
Render EXACTLY THREE charts from a precomputed scenario-means CSV.

The plotter never recalculates gain, interference, or fairness.
All metrics are read directly from qaoa_scenario_means.csv.

Dataset aggregation is performed by qaoa_beam_assignment_datasets.py.

Usage:
    python plotter.py results/run_.../qaoa_scenario_means.csv
    python plotter.py results/run_.../qaoa_scenario_means.csv --formats pdf,png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PAPER CONFIGURATION
# ============================================================

PAPER_RC = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",

    # Font sizes
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


# ============================================================
# FIGURE DIMENSIONS AND LAYOUT
# ============================================================

# Approximately one-third of a 7-inch-wide IEEE figure.
FIG_WIDTH_IN = 7.0 / 3.0
FIG_HEIGHT_IN = 2.65

PAD_INCHES = 0.01
MARKER_SIZE = 2.0

LEFT_MARGIN_IN = 0.48
RIGHT_MARGIN_IN = 0.10

TOP_FRAC = 0.82
BOTTOM_FRAC = 0.34

LEGEND_ANCHOR_Y = 1.00


# ============================================================
# PLOTTING OPTIONS
# ============================================================

FORMATS = ("pdf",)

DEFAULT_METHODS = [
    "Greedy",
    "OMP",
    "QAOA-XY",
    "ILP",
]

COLORS = {
    "Greedy": "tab:blue",
    "OMP": "tab:orange",
    "QAOA-XY": "tab:green",
    "ILP": "tab:red",
    "Optimal": "tab:red",
}

MARKERS = {
    "Greedy": "o",
    "OMP": "s",
    "QAOA-XY": "^",
    "ILP": "D",
    "Optimal": "D",
}

LINESTYLES = {
    "Greedy": "--",
    "OMP": ":",
    "QAOA-XY": "-",
    "ILP": "-.",
    "Optimal": "-.",
}


# ============================================================
# METRIC DEFINITIONS
# ============================================================

# These values define chart labels and output filenames only.
# Metric values are read directly from the CSV without conversion.

METRICS = {
    "gain_linear": {
        "title": "",
        "ylabel": "Gain",
        "stem": "gain",
    },

    "interference_linear": {
        "title": "",
        "ylabel": "Interference",
        "stem": "interference",
    },

    "fairness": {
        "title": "",
        "ylabel": "Fairness penalty",
        "stem": "fairness",
    },
}


# ============================================================
# LOAD PRECOMPUTED RESULTS
# ============================================================

def load_results(csv_path):
    """
    Load already-aggregated scenario means.

    No aggregation, metric calculations, or unit conversions
    are performed by this function.
    """

    df = pd.read_csv(csv_path)

    required = {
        "scenario",
        "scenario_order",
        "method",
        *METRICS.keys(),
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            "Supply qaoa_scenario_means.csv, not the "
            "individual-snapshot CSV."
        )

    if df[list(METRICS)].isna().any().any():
        raise ValueError(
            "The scenario-means CSV contains missing metric values."
        )

    if df.duplicated(["scenario", "method"]).any():
        raise ValueError(
            "Expected exactly one precomputed value per "
            "scenario and method."
        )

    if df[["scenario", "scenario_order"]].isna().any().any():
        raise ValueError(
            "Scenario names and scenario_order cannot be missing."
        )

    return df.sort_values(
        "scenario_order",
        kind="stable",
    )


# ============================================================
# PAPER-STYLE FIGURE LAYOUT
# ============================================================

def configure_axes(ax, scenarios, ylabel, title):
    """
    Apply consistent paper formatting to a single chart.
    """

    xs = np.arange(len(scenarios))

    ax.set_xticks(range(10))
    ax.set_xticklabels([
        r'$S_1$', r'$S_2$', r'$S_3$', r'$S_4$', r'$S_5$',
        r'$S_6$', r'$S_7$', r'$S_8$', r'$S_9$', r'$S_{10}$'
    ])

    ax.set_xlabel("Channel scenario")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Light grid for readability in printed papers.
    ax.grid(
        axis="y",
        which="major",
        alpha=0.28,
        linewidth=0.55,
    )

    ax.grid(
        axis="y",
        which="minor",
        alpha=0.10,
        linewidth=0.35,
    )

    ax.margins(y=0.16)

    return xs


def configure_legend(ax, methods):
    """
    Position the legend above the plotting area.
    """

    handles, labels = ax.get_legend_handles_labels()

    if not handles:
        return

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
        ncol=min(4, len(methods)),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.0,
        handletextpad=0.4,
    )


# ============================================================
# PLOT THREE SCENARIO-MEAN CHARTS
# ============================================================

def plot_from_csv(
    csv_path,
    out_dir=None,
    methods=None,
    formats=FORMATS,
    close=True,
):
    """
    Generate exactly three figures:

        1. Scenario-mean beam gain
        2. Scenario-mean interference
        3. Scenario-mean fairness penalty

    Returns:
        list_of_output_paths, list_of_figures
    """

    df = load_results(csv_path)

    out = (
        Path(out_dir)
        if out_dir is not None
        else Path(csv_path).resolve().parent
    )

    out.mkdir(parents=True, exist_ok=True)

    methods = (
        DEFAULT_METHODS
        if methods is None
        else list(methods)
    )

    formats = [
        fmt.strip().lower().lstrip(".")
        for fmt in formats
        if fmt.strip()
    ]

    if not formats:
        raise ValueError("Specify at least one output format.")

    supported_formats = {"pdf", "png", "svg"}

    invalid_formats = set(formats) - supported_formats

    if invalid_formats:
        raise ValueError(
            f"Unsupported formats: {sorted(invalid_formats)}. "
            f"Supported formats: {sorted(supported_formats)}."
        )

    if not methods:
        raise ValueError("At least one method must be selected.")

    # Preserve the scenario ordering provided by the dataset runner.
    order = (
        df[["scenario", "scenario_order"]]
        .drop_duplicates()
        .sort_values("scenario_order", kind="stable")
    )

    scenarios = order["scenario"].tolist()

    xs = np.arange(len(scenarios))

    paths = []
    figs = []

    # Apply the PAPER_RC settings consistently throughout plotting.
    with plt.rc_context(PAPER_RC):

        # Iterate over exactly the three configured metrics.
        for metric, settings in METRICS.items():

            fig, ax = plt.subplots(
                figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
            )

            # Reserve space using the configured physical margins.
            fig.subplots_adjust(
                left=LEFT_MARGIN_IN / FIG_WIDTH_IN,
                right=1.0 - RIGHT_MARGIN_IN / FIG_WIDTH_IN,
                top=TOP_FRAC,
                bottom=BOTTOM_FRAC,
            )

            plotted_methods = []

            for method in methods:

                sub = df[df["method"] == method].set_index("scenario")

                # Skip methods that are not present in the CSV.
                if sub.empty:
                    continue

                missing_scenarios = (
                    set(scenarios) - set(sub.index)
                )

                if missing_scenarios:
                    raise ValueError(
                        f"{method}: missing stored {metric} values "
                        f"for scenarios "
                        f"{sorted(missing_scenarios)}."
                    )

                # Lookup only: no metric calculations or conversions.
                ys = sub.loc[scenarios, metric].to_numpy(dtype=float)

                ax.plot(
                    xs,
                    ys,
                    label=method,
                    color=COLORS.get(method, "black"),
                    marker=MARKERS.get(method, "o"),
                    linestyle=LINESTYLES.get(method, "-"),
                    markersize=MARKER_SIZE,
                )

                plotted_methods.append(method)

            if not plotted_methods:
                plt.close(fig)

                raise ValueError(
                    "None of the requested methods were found in "
                    "the scenario-means CSV."
                )

            # Use a logarithmic view when every stored value is positive.
            # Use symlog if zero or negative values are present.
            # This changes only the visual scale, not the data.
            all_y = df[metric].to_numpy(dtype=float)

            if np.all(all_y > 0):

                ax.set_yscale(
                    "log",
                    base=10,
                )

            else:

                positive = all_y[all_y > 0]

                linthresh = (
                    max(float(np.min(positive)) * 0.1, 1e-12)
                    if len(positive)
                    else 1.0
                )

                ax.set_yscale(
                    "symlog",
                    linthresh=linthresh,
                    base=10,
                )

            configure_axes(
                ax=ax,
                scenarios=scenarios,
                ylabel=settings["ylabel"],
                title=settings["title"],
            )

            configure_legend(
                ax=ax,
                methods=plotted_methods,
            )

            # Tight layout is not used because the paper margins
            # and legend placement are configured explicitly above.
            for fmt in formats:

                dest = out / (
                    f"scenario_mean_{settings['stem']}.{fmt}"
                )

                fig.savefig(
                    dest,
                    format=fmt,
                    dpi=PAPER_RC["savefig.dpi"],
                    bbox_inches=PAPER_RC["savefig.bbox"],
                    pad_inches=PAD_INCHES,
                )

                paths.append(dest)

            figs.append(fig)

            if close:
                plt.close(fig)

    return paths, figs


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate three paper-formatted comparison charts "
            "from precomputed scenario means."
        )
    )

    parser.add_argument(
        "csv",
        help="Path to qaoa_scenario_means.csv",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: directory containing the CSV).",
    )

    parser.add_argument(
        "--formats",
        default=",".join(FORMATS),
        help="Comma-separated output formats, e.g. pdf or pdf,png.",
    )

    args = parser.parse_args()

    formats = args.formats.split(",")

    files, _ = plot_from_csv(
        args.csv,
        out_dir=args.out_dir,
        formats=formats,
    )

    for file_path in files:
        print(file_path)


if __name__ == "__main__":
    main()