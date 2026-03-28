from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONSTRAINT_DIR = ROOT / "exp" / "adaptive_routing_constraint_sweep_v1_fix3"
MODEL_DIR = ROOT / "exp" / "adaptive_routing_model_sweep_v1_fix3"
CASCADE_DIR = ROOT / "exp" / "adaptive_routing_cascade_v1_fix6"
PAPER_FIG_DIR = ROOT / "figures" / "paper"

CONSTRAINT_ORDER = [
    "r1_all_mutable",
    "r1_immutable_only",
    "r1_structural",
    "r2_realistic",
]
CONSTRAINT_LABELS = {
    "r1_all_mutable": "All mutable",
    "r1_immutable_only": "Immutable only",
    "r1_structural": "Structural",
    "r2_realistic": "Realistic",
}
MODEL_LABELS = {
    "tree": "Tree",
    "logistic": "Logistic",
    "rf": "Random forest",
}
DATASET_LABELS = {
    "adult": "Adult",
    "bank": "Bank",
    "compas": "COMPAS",
    "german": "German",
}


def make_dirs() -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    (CONSTRAINT_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "analysis").mkdir(parents=True, exist_ok=True)
    (CASCADE_DIR / "analysis").mkdir(parents=True, exist_ok=True)


def build_constraint_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tuning = pd.read_csv(CONSTRAINT_DIR / "tuning.csv")
    query = pd.read_csv(CONSTRAINT_DIR / "query_results.csv")

    alpha_by_constraint = (
        tuning.groupby("constraint")["fixed_alpha"]
        .agg(mean="mean", median="median", min="min", max="max")
        .reset_index()
    )
    alpha_by_constraint["constraint"] = pd.Categorical(
        alpha_by_constraint["constraint"], categories=CONSTRAINT_ORDER, ordered=True
    )
    alpha_by_constraint = alpha_by_constraint.sort_values("constraint")

    alpha_by_dataset = (
        tuning.groupby(["dataset", "constraint"])["fixed_alpha"]
        .agg(mean="mean", median="median", min="min", max="max")
        .reset_index()
    )
    alpha_by_dataset["constraint"] = pd.Categorical(
        alpha_by_dataset["constraint"], categories=CONSTRAINT_ORDER, ordered=True
    )
    alpha_by_dataset = alpha_by_dataset.sort_values(["dataset", "constraint"])

    keep = query[query["method"].isin(["fixed_blend", "utility_router_guarded", "projection_only"])]
    utility_compare = (
        keep.groupby(["dataset", "constraint", "method"], as_index=False)[["utility", "valid", "runtime_sec"]]
        .mean()
    )
    utility_compare["constraint"] = pd.Categorical(
        utility_compare["constraint"], categories=CONSTRAINT_ORDER, ordered=True
    )
    utility_compare = utility_compare.sort_values(["dataset", "constraint", "method"])

    alpha_by_constraint.to_csv(CONSTRAINT_DIR / "analysis" / "fixed_alpha_by_constraint.csv", index=False)
    alpha_by_dataset.to_csv(CONSTRAINT_DIR / "analysis" / "fixed_alpha_by_dataset.csv", index=False)
    utility_compare.to_csv(CONSTRAINT_DIR / "analysis" / "constraint_method_summary.csv", index=False)
    return alpha_by_constraint, alpha_by_dataset, utility_compare


def build_model_summary() -> pd.DataFrame:
    query = pd.read_csv(MODEL_DIR / "query_results.csv")
    keep = query[query["method"].isin(["fixed_blend", "utility_router_guarded"])]
    pivot = (
        keep.pivot_table(
            index=["model_family", "constraint"],
            columns="method",
            values="utility",
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot["guard_gain"] = pivot["fixed_blend"] - pivot["utility_router_guarded"]
    pivot["constraint"] = pd.Categorical(
        pivot["constraint"], categories=["r1_all_mutable", "r2_realistic"], ordered=True
    )
    pivot = pivot.sort_values(["model_family", "constraint"])
    pivot.to_csv(MODEL_DIR / "analysis" / "model_guard_gain_summary.csv", index=False)
    return pivot


def build_cascade_summary() -> pd.DataFrame:
    query = pd.read_csv(CASCADE_DIR / "query_results.csv")
    keep = query[
        query["method"].isin(["fixed_blend", "utility_router_guarded", "exact_tree_milp", "cascade_exact_auto"])
    ]
    summary = keep.groupby(["dataset", "method"], as_index=False)[
        ["utility", "valid", "runtime_sec", "accepted_exact", "escalated_exact"]
    ].mean()
    summary.to_csv(CASCADE_DIR / "analysis" / "cascade_method_summary.csv", index=False)

    overall = keep.groupby("method", as_index=False)[
        ["utility", "valid", "runtime_sec", "accepted_exact", "escalated_exact"]
    ].mean()
    overall.to_csv(CASCADE_DIR / "analysis" / "cascade_overall_summary.csv", index=False)

    guarded = keep[keep["method"] == "utility_router_guarded"][
        ["query_id", "dataset", "utility"]
    ].rename(columns={"utility": "guarded_utility"})
    cascade = keep[keep["method"] == "cascade_exact_auto"][
        ["query_id", "dataset", "utility"]
    ].rename(columns={"utility": "cascade_utility"})
    compare = guarded.merge(cascade, on=["query_id", "dataset"], how="inner")
    compare["delta_utility"] = compare["cascade_utility"] - compare["guarded_utility"]
    compare["outcome"] = np.where(
        compare["delta_utility"] < -1e-9,
        "better",
        np.where(compare["delta_utility"] > 1e-9, "worse", "tie"),
    )
    compare.to_csv(CASCADE_DIR / "analysis" / "cascade_query_compare_vs_guarded.csv", index=False)

    compare_overall = (
        compare.groupby("outcome")
        .size()
        .rename("count")
        .reindex(["worse", "better", "tie"], fill_value=0)
        .reset_index()
    )
    compare_by_dataset = (
        compare.groupby(["dataset", "outcome"])
        .size()
        .rename("count")
        .reset_index()
        .pivot(index="dataset", columns="outcome", values="count")
        .fillna(0)
        .reset_index()
    )
    compare_overall.to_csv(CASCADE_DIR / "analysis" / "cascade_compare_overall.csv", index=False)
    compare_by_dataset.to_csv(CASCADE_DIR / "analysis" / "cascade_compare_by_dataset.csv", index=False)
    return summary


def plot_supporting_sweeps(alpha_by_dataset: pd.DataFrame, model_summary: pd.DataFrame) -> None:
    base_font = 17
    plt.rcParams.update(
        {
            "font.size": base_font,
            "axes.titlesize": base_font,
            "axes.labelsize": base_font,
            "xtick.labelsize": base_font - 1,
            "ytick.labelsize": base_font - 1,
            "legend.fontsize": base_font - 2,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    x = np.arange(len(CONSTRAINT_ORDER))
    palette = {
        "adult": "#1f77b4",
        "bank": "#d95f02",
        "compas": "#7570b3",
        "german": "#1b9e77",
    }
    for dataset in ["adult", "bank", "compas", "german"]:
        subset = alpha_by_dataset[alpha_by_dataset["dataset"] == dataset]
        ax.plot(
            x,
            subset["mean"].to_numpy(),
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=DATASET_LABELS[dataset],
            color=palette[dataset],
        )
    ax.set_title("Constraint-induced shift in tuned fixed policy")
    ax.set_ylabel("Mean tuned alpha")
    ax.set_xticks(x, [CONSTRAINT_LABELS[c] for c in CONSTRAINT_ORDER], rotation=15, ha="right")
    ax.set_ylim(0.2, 1.5)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[1]
    heat = model_summary.pivot(index="model_family", columns="constraint", values="guard_gain").loc[
        ["tree", "logistic", "rf"], ["r1_all_mutable", "r2_realistic"]
    ]
    im = ax.imshow(heat.to_numpy(), cmap="YlGn", aspect="auto", vmin=0.0, vmax=float(max(heat.to_numpy().max(), 0.1)))
    ax.set_title("Guard gain over fixed blend")
    ax.set_xticks(np.arange(heat.shape[1]), [CONSTRAINT_LABELS[c] for c in heat.columns], rotation=15, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]), [MODEL_LABELS[m] for m in heat.index])
    threshold = float(heat.to_numpy().max()) * 0.7
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            text_color = "white" if val >= threshold else "#222222"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=15)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Utility improvement")

    plt.tight_layout()
    fig.savefig(PAPER_FIG_DIR / "fix3_supporting_sweeps.png", dpi=100)
    plt.close(fig)


def main() -> None:
    make_dirs()
    _, alpha_by_dataset, _ = build_constraint_summaries()
    model_summary = build_model_summary()
    build_cascade_summary()
    plot_supporting_sweeps(alpha_by_dataset, model_summary)


if __name__ == "__main__":
    main()
