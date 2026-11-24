"""Plot the three-scenario mean best-so-far curves produced by analyze_results.py."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


COLORS = {
    "Random": "#666666",
    "PSO": "#2474b5",
    "NSGA-II": "#29915c",
    "GCE3S": "#8e64a7",
    "DGGA": "#d68a20",
    "RMD-ES": "#bd2026",
}
LINESTYLES = {
    "Random": ":",
    "PSO": "--",
    "NSGA-II": "-.",
    "GCE3S": (0, (5, 1, 1, 1)),
    "DGGA": "--",
    "RMD-ES": "-",
}
SCENARIO_TITLES = {
    "HighWayCutInScenarioParse": "Highway cut-in",
    "CCRH": "CCRH",
    "CBNAO": "CBNAO",
}


def make_plot(csv_path: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    curves = defaultdict(lambda: defaultdict(lambda: ([], [])))
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            x, y = curves[row["scenario"]][row["method"]]
            x.append(int(row["valid_evaluations"]))
            y.append(float(row["mean_best_target_fit"]))

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    scenarios = list(SCENARIO_TITLES)
    methods = list(COLORS)
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.45), sharey=False)
    for scenario, ax in zip(scenarios, axes):
        for method in methods:
            xs, ys = curves[scenario][method]
            if xs:
                ax.step(xs, ys, where="post", color=COLORS[method],
                        linestyle=LINESTYLES[method], linewidth=1.65 if method == "RMD-ES" else 1.1)
        ax.set_title(SCENARIO_TITLES[scenario])
        ax.set_xlabel("Recorded valid evaluations")
        ax.set_xlim(50, 2300)
        ax.set_xticks([50, 500, 1000, 1500, 2000, 2300])
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.ticklabel_format(axis="y", useOffset=False)
    axes[0].set_ylabel("Mean best-so-far target_fit (lower is better)")
    handles = [Line2D([0], [0], color=COLORS[m], linestyle=LINESTYLES[m],
                      linewidth=1.6, label=m) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.025))
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.08, right=0.985, wspace=0.32)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "convergence_mean.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "convergence_mean.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
