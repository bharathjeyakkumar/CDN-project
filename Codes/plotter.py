"""Render EXACTLY THREE charts from a precomputed scenario-means CSV.

The plotter never recalculates gain/interference/fairness or changes units.
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

DEFAULT_METHODS = ["Greedy", "OMP", "QAOA-XY", "ILP"]
COLORS = {"Greedy": "tab:blue", "OMP": "tab:orange", "QAOA-XY": "tab:green", "ILP": "tab:red"}
MARKERS = {"Greedy": "o", "OMP": "s", "QAOA-XY": "^", "ILP": "D"}
LINESTYLES = {"Greedy": "--", "OMP": ":", "QAOA-XY": "-", "ILP": "-."}
METRICS = {
    "gain_linear": ("Gain", "Mean gain (linear power)", "gain"),
    "interference_linear": ("Interference", "Mean pairwise interference (linear power)", "interference"),
    "fairness": ("Fairness penalty (lower is better)", "Mean unweighted fairness penalty", "fairness"),
}


def load_results(csv_path):
    """Require already-aggregated metrics; plotter performs no groupby or math."""
    df = pd.read_csv(csv_path)
    required = {"scenario", "scenario_order", "method", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}. Supply qaoa_scenario_means.csv, "
                         "not the individual-snapshot CSV.")
    if df[list(METRICS)].isna().any().any():
        raise ValueError("Means CSV has missing metric values")
    if df.duplicated(["scenario", "method"]).any():
        raise ValueError("Expected exactly one precomputed value per scenario and method")
    return df.sort_values("scenario_order", kind="stable")


def plot_from_csv(csv_path, out_dir=None, methods=None, formats=("pdf", "png"), close=True):
    """Return (list_of_output_paths, list_of_figures)."""
    df = load_results(csv_path)
    out = Path(out_dir) if out_dir is not None else Path(csv_path).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    methods = DEFAULT_METHODS if methods is None else list(methods)
    order = df[["scenario", "scenario_order"]].drop_duplicates().sort_values("scenario_order")
    scenarios = order["scenario"].tolist()
    xs = np.arange(len(scenarios))
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif", "Liberation Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8, "pdf.fonttype": 42,
    })
    paths, figs = [], []
    for metric, (title, ylabel, stem) in METRICS.items():
        fig, ax = plt.subplots(figsize=(9.5, 3.65))
        for method in methods:
            sub = df[df["method"] == method].set_index("scenario")
            if sub.empty:
                continue
            missing = set(scenarios) - set(sub.index)
            if missing:
                raise ValueError(f"{method}: no stored {metric} for scenarios {sorted(missing)}")
            # Lookup only: no metric conversions or statistics performed here.
            ys = sub.loc[scenarios, metric].to_numpy(float)
            ax.plot(xs, ys, label=method, color=COLORS.get(method, "black"),
                    marker=MARKERS.get(method, "o"),
                    linestyle=LINESTYLES.get(method, "-"), linewidth=1.35,
                    markersize=4.5)
        ax.set_xticks(xs, scenarios, rotation=35, ha="right")
        ax.set_xlabel("Channel scenario")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.legend(loc="best", frameon=False, ncol=min(4, len(methods)))
        fig.tight_layout()
        for fmt in formats:
            dest = out / f"scenario_mean_{stem}.{fmt}"
            fig.savefig(dest, dpi=300, bbox_inches="tight")
            paths.append(dest)
        figs.append(fig)
        if close:
            plt.close(fig)
    return paths, figs


def main():
    parser = argparse.ArgumentParser(description="Three QAOA comparison charts from stored scenario means")
    parser.add_argument("csv", help="qaoa_scenario_means.csv from dataset runner")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--formats", default="pdf,png", help="comma-separated formats")
    args = parser.parse_args()
    files, _ = plot_from_csv(args.csv, out_dir=args.out_dir, formats=args.formats.split(","))
    for f in files:
        print(f)


if __name__ == "__main__":
    main()
