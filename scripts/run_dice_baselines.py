#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import dice_ml
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import run_adaptive_routing_experiments as expmod


ROOT = Path(__file__).resolve().parents[1]


def ensure_dirs(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)


def make_raw_pipeline(train_df: pd.DataFrame, y_train: np.ndarray, depth: int, seed: int) -> Pipeline:
    bundle = expmod.build_preprocessor(train_df)
    raw_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(bundle.preprocessor)),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=int(depth),
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=int(seed),
                ),
            ),
        ]
    )
    raw_pipeline.fit(train_df, y_train)
    return raw_pipeline


def mutable_features(config: expmod.DatasetConfig, columns: List[str], constraints: expmod.ConstraintSetting) -> List[str]:
    if not constraints.enforce_immutables:
        return list(columns)
    return [col for col in columns if col not in set(config.immutable_raw)]


def run_dice(
    dataset_names: List[str],
    seeds: List[int],
    depths: List[int],
    max_rows: int,
    max_queries_per_cell: int,
    exp_name: str,
    dice_method: str,
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
            val_idx, test_idx = train_test_split(
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

                raw_pipeline = make_raw_pipeline(train_df, y_train, depth=depth, seed=seed)

                target_col = "__target__"
                dice_df = train_df.copy()
                dice_df[target_col] = y_train
                data = dice_ml.Data(
                    dataframe=dice_df,
                    continuous_features=bundle.num_cols,
                    outcome_name=target_col,
                )
                model = dice_ml.Model(model=raw_pipeline, backend="sklearn", model_type="classifier")
                explainer = dice_ml.Dice(data, model, method=dice_method)

                for constraints in [expmod.R1_ALL_MUTABLE, expmod.R2_REALISTIC]:
                    for setting in expmod.SETTINGS:
                        eval_raw, _ = expmod.apply_setting(setting, train_df, test_df, config)
                        eval_x = expmod.transform_df(bundle, eval_raw)
                        pred = clf.predict(eval_x)
                        neg_idx = np.where(pred == 0)[0][: max_queries_per_cell]

                        features_to_vary = mutable_features(config, list(train_df.columns), constraints)
                        if not features_to_vary:
                            continue

                        for idx in neg_idx:
                            row_raw = eval_raw.iloc[[int(idx)]].copy()
                            x0 = eval_x[int(idx)]
                            start = time.perf_counter()
                            candidate = None
                            try:
                                cf = explainer.generate_counterfactuals(
                                    query_instances=row_raw,
                                    total_CFs=1,
                                    desired_class="opposite",
                                    features_to_vary=features_to_vary,
                                    verbose=False,
                                )
                                cf_list = cf.cf_examples_list
                                if cf_list and cf_list[0].final_cfs_df is not None and not cf_list[0].final_cfs_df.empty:
                                    cf_raw = cf_list[0].final_cfs_df.iloc[[0]][train_df.columns].copy()
                                    cf_x = expmod.transform_df(bundle, cf_raw)[0]
                                    candidate = expmod.enforce_actionability(
                                        base_query=x0,
                                        proposal=cf_x,
                                        bundle=bundle,
                                        config=config,
                                        constraints=constraints,
                                    )
                            except Exception:
                                candidate = None

                            result = expmod.evaluate_candidate(
                                method=f"dice_{dice_method}",
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
                                    "method": f"dice_{dice_method}",
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
                "dice_method": dice_method,
                "n_query_rows": int(len(query_df)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"exp_dir": str(exp_dir), "n_query_rows": int(len(query_df))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiCE baselines for adaptive recourse experiments.")
    parser.add_argument("--datasets", nargs="+", default=["adult", "german", "bank"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--depths", nargs="+", type=int, default=[3])
    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--max-queries-per-cell", type=int, default=25)
    parser.add_argument("--exp-name", type=str, default="dice_baseline")
    parser.add_argument("--dice-method", type=str, default="random")
    args = parser.parse_args()

    run_dice(
        dataset_names=list(args.datasets),
        seeds=[int(x) for x in args.seeds],
        depths=[int(x) for x in args.depths],
        max_rows=int(args.max_rows),
        max_queries_per_cell=int(args.max_queries_per_cell),
        exp_name=str(args.exp_name),
        dice_method=str(args.dice_method),
    )


if __name__ == "__main__":
    main()
