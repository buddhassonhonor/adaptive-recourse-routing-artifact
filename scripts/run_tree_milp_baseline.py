#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pulp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import run_adaptive_routing_experiments as expmod


ROOT = Path(__file__).resolve().parents[1]


def ensure_dirs(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)


def feature_bounds(bundle: expmod.TransformBundle) -> Dict[int, tuple[Optional[float], Optional[float]]]:
    bounds: Dict[int, tuple[Optional[float], Optional[float]]] = {}
    train_x = bundle.train_x
    for idx in range(train_x.shape[1]):
        lo = float(np.min(train_x[:, idx])) - 1.0
        hi = float(np.max(train_x[:, idx])) + 1.0
        bounds[idx] = (lo, hi)
    for idxs in bundle.onehot_groups.values():
        for idx in idxs:
            bounds[idx] = (0.0, 1.0)
    return bounds


def build_leaf_milp_candidate(
    base_query: np.ndarray,
    leaf_box: Dict[str, Dict[int, float]],
    bundle: expmod.TransformBundle,
    config: expmod.DatasetConfig,
    constraints: expmod.ConstraintSetting,
    bounds: Dict[int, tuple[Optional[float], Optional[float]]],
    time_limit_sec: float,
) -> Optional[np.ndarray]:
    prob = pulp.LpProblem("tree_milp_recourse", pulp.LpMinimize)
    x_vars: Dict[int, pulp.LpVariable] = {}
    d_vars: Dict[int, pulp.LpVariable] = {}
    binary_idxs: set[int] = set()

    for col, idxs in bundle.onehot_groups.items():
        if constraints.enforce_immutables and col in config.immutable_raw:
            continue
        binary_idxs.update(int(idx) for idx in idxs)

    for idx in range(len(base_query)):
        lo, hi = bounds[idx]
        if idx in binary_idxs:
            x_vars[idx] = pulp.LpVariable(f"x_{idx}", lowBound=0.0, upBound=1.0, cat=pulp.LpBinary)
        else:
            x_vars[idx] = pulp.LpVariable(f"x_{idx}", lowBound=lo, upBound=hi, cat="Continuous")
        d_vars[idx] = pulp.LpVariable(f"d_{idx}", lowBound=0.0, cat="Continuous")
        base_val = float(base_query[idx])
        prob += x_vars[idx] - base_val <= d_vars[idx]
        prob += base_val - x_vars[idx] <= d_vars[idx]

    for col in bundle.num_cols:
        idx = bundle.raw_to_indices[col][0]
        lo = expmod.lower_bound(leaf_box, idx)
        hi = expmod.upper_bound(leaf_box, idx)
        base_val = float(base_query[idx])
        if constraints.enforce_immutables and col in config.immutable_raw:
            prob += x_vars[idx] == base_val
        else:
            if math.isfinite(lo):
                prob += x_vars[idx] >= lo
            if math.isfinite(hi):
                prob += x_vars[idx] <= hi
            if constraints.enforce_monotonic:
                if col in config.monotone_increase_raw:
                    prob += x_vars[idx] >= base_val
                if col in config.monotone_decrease_raw:
                    prob += x_vars[idx] <= base_val

    for col, idxs in bundle.onehot_groups.items():
        if constraints.enforce_immutables and col in config.immutable_raw:
            frozen = int(np.argmax(base_query[idxs]))
            for pos, idx in enumerate(idxs):
                prob += x_vars[idx] == float(pos == frozen)
        else:
            prob += pulp.lpSum(x_vars[idx] for idx in idxs) == 1.0
        for idx in idxs:
            lo = expmod.lower_bound(leaf_box, idx)
            hi = expmod.upper_bound(leaf_box, idx)
            if math.isfinite(lo):
                prob += x_vars[idx] >= lo
            if math.isfinite(hi):
                prob += x_vars[idx] <= hi

    prob += pulp.lpSum(d_vars.values())
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=float(time_limit_sec))
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        return None

    candidate = np.array([float(pulp.value(x_vars[idx])) for idx in range(len(base_query))], dtype=float)
    for col, idxs in bundle.onehot_groups.items():
        if constraints.enforce_immutables and col in config.immutable_raw:
            continue
        chosen = expmod.choose_onehot_vector(candidate, idxs, box=None)
        if chosen is not None:
            candidate[idxs] = chosen
    return candidate


def best_milp_candidate(
    base_query: np.ndarray,
    clf: DecisionTreeClassifier,
    leaf_boxes: Dict[int, Dict[str, Dict[int, float]]],
    positive_leaf_nodes: List[int],
    bundle: expmod.TransformBundle,
    config: expmod.DatasetConfig,
    constraints: expmod.ConstraintSetting,
    time_limit_sec: float,
) -> Optional[np.ndarray]:
    _ = clf
    bounds = feature_bounds(bundle)
    best_candidate: Optional[np.ndarray] = None
    best_cost = float("inf")
    for leaf_id in positive_leaf_nodes:
        candidate = build_leaf_milp_candidate(
            base_query=base_query,
            leaf_box=leaf_boxes[int(leaf_id)],
            bundle=bundle,
            config=config,
            constraints=constraints,
            bounds=bounds,
            time_limit_sec=time_limit_sec,
        )
        if candidate is None:
            continue
        if int(clf.predict(candidate.reshape(1, -1))[0]) != 1:
            continue
        cost = float(np.sum(np.abs(candidate - base_query)))
        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate
    return best_candidate


def run_tree_milp(
    dataset_names: List[str],
    seeds: List[int],
    depths: List[int],
    max_rows: int,
    max_queries_per_cell: int,
    exp_name: str,
    time_limit_sec: float,
) -> None:
    exp_dir = ROOT / "exp" / exp_name
    ensure_dirs(exp_dir)

    dataset_map = expmod.build_datasets()
    query_rows: List[Dict[str, object]] = []

    for dataset_name in dataset_names:
        config = dataset_map[dataset_name]
        full_df, full_y = config.loader()
        full_df, full_y = expmod.downsample_frame(full_df, full_y, max_rows=max_rows, seed=11)

        for seed in seeds:
            idx_all = np.arange(len(full_y))
            train_idx, temp_idx = train_test_split(
                idx_all,
                test_size=0.40,
                stratify=full_y,
                random_state=int(seed),
            )
            _, test_idx = train_test_split(
                temp_idx,
                test_size=0.50,
                stratify=full_y[temp_idx],
                random_state=int(seed) + 1,
            )
            train_df = full_df.iloc[train_idx].reset_index(drop=True)
            test_df = full_df.iloc[test_idx].reset_index(drop=True)
            y_train = full_y[train_idx]

            bundle = expmod.build_preprocessor(train_df)
            train_x = bundle.train_x

            for depth in depths:
                clf = DecisionTreeClassifier(
                    max_depth=int(depth),
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=int(seed),
                )
                clf.fit(train_x, y_train)
                leaf_boxes = expmod.get_leaf_boxes(clf)
                positive_leaf_nodes = expmod.get_positive_leaf_nodes(clf, leaf_boxes)
                if not positive_leaf_nodes:
                    continue

                for constraints in [expmod.R1_ALL_MUTABLE, expmod.R2_REALISTIC]:
                    for setting in expmod.SETTINGS:
                        eval_raw, _ = expmod.apply_setting(setting, train_df, test_df, config)
                        eval_x = expmod.transform_df(bundle, eval_raw)
                        pred = clf.predict(eval_x)
                        neg_idx = np.where(pred == 0)[0][: max_queries_per_cell]

                        for idx in neg_idx:
                            x0 = eval_x[int(idx)]
                            start = time.perf_counter()
                            candidate = best_milp_candidate(
                                base_query=x0,
                                clf=clf,
                                leaf_boxes=leaf_boxes,
                                positive_leaf_nodes=positive_leaf_nodes,
                                bundle=bundle,
                                config=config,
                                constraints=constraints,
                                time_limit_sec=time_limit_sec,
                            )
                            result = expmod.evaluate_candidate(
                                method="tree_milp_l1",
                                alpha=0.0,
                                base_query=x0,
                                candidate=candidate,
                                bundle=bundle,
                                clf=clf,
                                start_time=start,
                            )
                            query_rows.append(
                                {
                                    "dataset": dataset_name,
                                    "seed": seed,
                                    "depth": depth,
                                    "setting": setting,
                                    "constraint": constraints.name,
                                    "method": "tree_milp_l1",
                                    "query_id": f"{dataset_name}|seed={seed}|depth={depth}|constraint={constraints.name}|setting={setting}|row={int(idx)}",
                                    "valid": float(result.valid),
                                    "cost_l1": result.cost_l1,
                                    "cost_l2": result.cost_l2,
                                    "sparsity": result.sparsity,
                                    "plausibility": result.plausibility,
                                    "utility": result.utility,
                                    "runtime_sec": result.runtime_sec,
                                }
                            )

    query_df = pd.DataFrame(query_rows)
    summary_df = (
        query_df.groupby(["dataset", "seed", "depth", "setting", "constraint", "method"], as_index=False)[
            ["valid", "cost_l1", "cost_l2", "sparsity", "plausibility", "utility", "runtime_sec"]
        ]
        .mean()
        if not query_df.empty
        else pd.DataFrame()
    )

    query_df.to_csv(exp_dir / "query_results.csv", index=False)
    summary_df.to_csv(exp_dir / "summary.csv", index=False)
    (exp_dir / "overview.json").write_text(
        json.dumps(
            {
                "datasets": dataset_names,
                "seeds": seeds,
                "depths": depths,
                "max_rows": max_rows,
                "max_queries_per_cell": max_queries_per_cell,
                "time_limit_sec": time_limit_sec,
                "n_query_rows": int(len(query_df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"exp_dir": str(exp_dir), "n_query_rows": int(len(query_df))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tree MILP L1 baselines for adaptive recourse experiments.")
    parser.add_argument("--datasets", nargs="+", default=["adult", "german", "bank"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--depths", nargs="+", type=int, default=[3])
    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--max-queries-per-cell", type=int, default=25)
    parser.add_argument("--exp-name", type=str, default="tree_milp_baseline")
    parser.add_argument("--time-limit-sec", type=float, default=5.0)
    args = parser.parse_args()

    run_tree_milp(
        dataset_names=list(args.datasets),
        seeds=[int(x) for x in args.seeds],
        depths=[int(x) for x in args.depths],
        max_rows=int(args.max_rows),
        max_queries_per_cell=int(args.max_queries_per_cell),
        exp_name=str(args.exp_name),
        time_limit_sec=float(args.time_limit_sec),
    )


if __name__ == "__main__":
    main()
