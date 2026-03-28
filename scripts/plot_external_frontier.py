#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


METHOD_LABELS = {
    "tree_milp_l1": "Exact tree MILP",
    "utility_router_guarded": "Guarded router",
    "fixed_blend": "Fixed blend",
    "knn_mean_k5": "5-neighbor mean",
    "dice_random": "DiCE random",
    "dice_genetic": "DiCE genetic",
    "nn_positive_train": "Nearest positive",
}


METHOD_COLORS = {
    "tree_milp_l1": "#0b3c5d",
    "utility_router_guarded": "#b33f62",
    "fixed_blend": "#f0a202",
    "knn_mean_k5": "#3c91e6",
    "dice_random": "#5c8001",
    "dice_genetic": "#8f2d56",
    "nn_positive_train": "#6c757d",
}


def main() -> None:
    src = ROOT / "exp" / "tree_milp_multi3_q20_v2" / "matched_family_milp_summary.csv"
    out_dir = ROOT / "figures" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "external_frontier.png"

    df = pd.read_csv(src)
    df["runtime_ms"] = df["runtime_sec"] * 1000.0

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    constraint_order = ["r2_realistic", "r1_all_mutable"]
    title_map = {
        "r2_realistic": "Realistic Constraints",
        "r1_all_mutable": "All-Mutable Constraints",
    }

    handles: list[Line2D] = []
    seen_methods: set[str] = set()

    for ax, constraint in zip(axes, constraint_order):
        part = df[df["constraint"] == constraint].copy()
        part = part.sort_values("utility")
        for _, row in part.iterrows():
            method = str(row["method"])
            ax.scatter(
                row["runtime_ms"],
                row["utility"],
                s=130,
                color=METHOD_COLORS.get(method, "#333333"),
                edgecolors="white",
                linewidths=0.8,
            )
            if method not in seen_methods:
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor=METHOD_COLORS.get(method, "#333333"),
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        markersize=9,
                        label=METHOD_LABELS.get(method, method),
                    )
                )
                seen_methods.add(method)
        ax.set_xscale("log")
        ax.set_xlabel("Mean runtime per query (ms, log scale)")
        ax.set_title(title_map[constraint])
        ax.grid(True, alpha=0.25, which="both")

    axes[0].set_ylabel("Mean recourse utility")
    fig.suptitle("External baseline quality-speed frontier", fontsize=20)
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=False,
    )
    plt.tight_layout(rect=(0, 0.08, 1, 0.95))
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
