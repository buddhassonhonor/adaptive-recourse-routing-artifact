#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


METHOD_LABELS = {
    "fixed_blend": "Fixed blend",
    "learned_router": "Learned router",
    "utility_router_guarded": "Guarded router",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_experiment(exp_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exp_dir = ROOT / "exp" / exp_name
    query_df = pd.read_csv(exp_dir / "query_results.csv")
    summary_df = pd.read_csv(exp_dir / "summary.csv")
    disparity_df = pd.read_csv(exp_dir / "subgroup_disparity.csv")
    return query_df, summary_df, disparity_df


def make_pairwise_table(
    query_df: pd.DataFrame,
    method_a: str,
    method_b: str,
) -> pd.DataFrame:
    piv = (
        query_df[query_df["method"].isin([method_a, method_b])]
        .pivot_table(
            index=["query_id", "dataset", "seed", "depth", "setting", "constraint", "subgroup"],
            columns="method",
            values=["regret", "utility", "cost_l1", "valid", "chosen_alpha"],
            aggfunc="first",
        )
        .reset_index()
    )
    piv.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in piv.columns.to_flat_index()]
    piv["regret_delta"] = piv[f"regret_{method_b}"] - piv[f"regret_{method_a}"]
    piv["cost_delta"] = piv[f"cost_l1_{method_b}"] - piv[f"cost_l1_{method_a}"]
    piv["valid_delta"] = piv[f"valid_{method_b}"] - piv[f"valid_{method_a}"]
    return piv


def make_case_studies(pairwise_df: pd.DataFrame, method_a: str, method_b: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    r2 = pairwise_df[pairwise_df["constraint"] == "r2_realistic"].copy()
    r1 = pairwise_df[pairwise_df["constraint"] == "r1_all_mutable"].copy()
    best_r2 = (
        r2.sort_values(["regret_delta", "cost_delta"], ascending=[True, True])
        .head(15)
        .reset_index(drop=True)
    )
    worst_r1 = (
        r1.sort_values(["regret_delta", "cost_delta"], ascending=[False, False])
        .head(15)
        .reset_index(drop=True)
    )
    keep = [
        "query_id",
        "dataset",
        "seed",
        "setting",
        "constraint",
        "subgroup",
        f"regret_{method_a}",
        f"regret_{method_b}",
        "regret_delta",
        f"cost_l1_{method_a}",
        f"cost_l1_{method_b}",
        "cost_delta",
        f"valid_{method_a}",
        f"valid_{method_b}",
        f"chosen_alpha_{method_a}",
        f"chosen_alpha_{method_b}",
    ]
    return best_r2[keep], worst_r1[keep]


def make_stability_summary(summary_df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    cell_df = summary_df[summary_df["method"].isin(methods)].copy()
    grouped = (
        cell_df.groupby(["constraint", "method"])["regret"]
        .agg(["mean", "std", "median", lambda s: float(np.quantile(s, 0.10)), lambda s: float(np.quantile(s, 0.90))])
        .reset_index()
    )
    grouped.columns = ["constraint", "method", "mean_regret", "std_regret", "median_regret", "q10_regret", "q90_regret"]
    return grouped


def plot_regret_box(summary_df: pd.DataFrame, fig_dir: Path, methods: list[str]) -> None:
    data = summary_df[summary_df["method"].isin(methods)].copy()
    if data.empty:
        return
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    constraints = ["r1_all_mutable", "r2_realistic"]
    for ax, constraint in zip(axes, constraints):
        part = data[data["constraint"] == constraint]
        order = [m for m in methods if m in part["method"].unique()]
        values = [part.loc[part["method"] == m, "regret"].to_numpy() for m in order]
        ax.boxplot(values, tick_labels=order, showfliers=False)
        ax.set_title(constraint)
        ax.set_ylabel("Cell-level regret")
        ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(fig_dir / "regret_box_by_constraint.png", dpi=100)
    plt.close()


def plot_subgroup_gap(disparity_df: pd.DataFrame, fig_dir: Path, methods: list[str]) -> None:
    part = disparity_df[disparity_df["method"].isin(methods)].copy()
    if part.empty:
        return
    agg = part.groupby(["constraint", "method"], as_index=False)["regret_gap"].mean()
    agg["method_label"] = agg["method"].map(lambda x: METHOD_LABELS.get(x, x))
    pivot = agg.pivot(index="method_label", columns="constraint", values="regret_gap")
    ax = pivot.plot(kind="bar", figsize=(11, 6), rot=0)
    ax.set_xlabel("Method")
    plt.ylabel("Mean subgroup regret gap")
    plt.title("Subgroup Disparity")
    plt.legend(title="Constraint", ncol=2, loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "subgroup_regret_gap.png", dpi=100)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze adaptive routing experiment outputs.")
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--method-a", type=str, default="fixed_blend")
    parser.add_argument("--method-b", type=str, default="utility_router")
    args = parser.parse_args()

    query_df, summary_df, disparity_df = load_experiment(args.exp_name)
    out_dir = ROOT / "exp" / args.exp_name
    fig_dir = ROOT / "figures" / args.exp_name / "analysis"
    ensure_dir(fig_dir)

    pairwise_df = make_pairwise_table(query_df, args.method_a, args.method_b)
    pairwise_df.to_csv(out_dir / f"pairwise_{args.method_a}_vs_{args.method_b}_query.csv", index=False)

    best_r2, worst_r1 = make_case_studies(pairwise_df, args.method_a, args.method_b)
    best_r2.to_csv(out_dir / f"case_studies_best_{args.method_b}_in_r2.csv", index=False)
    worst_r1.to_csv(out_dir / f"case_studies_worst_{args.method_b}_in_r1.csv", index=False)

    methods = [args.method_a, args.method_b, "nn_positive_train", "knn_mean_k5", "learned_router"]
    stability_df = make_stability_summary(summary_df, methods)
    stability_df.to_csv(out_dir / "stability_summary.csv", index=False)

    plot_regret_box(summary_df, fig_dir, methods)
    plot_subgroup_gap(disparity_df, fig_dir, [args.method_a, args.method_b, "learned_router"])


if __name__ == "__main__":
    main()
