#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import pulp
except ImportError:  # pragma: no cover - optional dependency
    pulp = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXP_NAME = "adaptive_routing"
EXP_DIR = ROOT / "exp" / DEFAULT_EXP_NAME
FIG_DIR = ROOT / "figures" / DEFAULT_EXP_NAME
DATA_ROOT = Path(r"D:\data")

INVALID_PENALTY = 1000.0
SPARSITY_WEIGHT = 0.05
PLAUSIBILITY_WEIGHT = 0.15
ABSTAIN_BUFFER = 0.50
DIFF_EPS = 1e-6
BOUND_EPS = 1e-8

ALPHA_GRID = [0.25, 0.50, 0.75, 0.90, 1.00]
DEFAULT_SEEDS = [7, 13, 19]
DEFAULT_DEPTHS = [3, 5]
SETTINGS = ["in_domain", "covariate_shift", "exemplar_mismatch", "subgroup_shift"]
DEFAULT_MODEL_FAMILIES = ["tree"]
FULL_FEATURE_ORDER = [
    "d_proj",
    "d_nn",
    "d_mix",
    "ood_score",
    "pos_density",
    "immutable_gap",
    "category_gap",
    "repair_needed",
    "leaf_slack",
]
NO_SHIFT_FEATURE_ORDER = [
    "d_proj",
    "d_nn",
    "d_mix",
    "immutable_gap",
    "category_gap",
    "repair_needed",
    "leaf_slack",
]
NO_GEOMETRY_FEATURE_ORDER = [
    "ood_score",
    "pos_density",
    "immutable_gap",
    "category_gap",
    "repair_needed",
]


@dataclass(frozen=True)
class ConstraintSetting:
    name: str
    enforce_immutables: bool
    enforce_onehot: bool
    enforce_monotonic: bool


R1_ALL_MUTABLE = ConstraintSetting(
    name="r1_all_mutable",
    enforce_immutables=False,
    enforce_onehot=False,
    enforce_monotonic=False,
)
R1_IMMUTABLE_ONLY = ConstraintSetting(
    name="r1_immutable_only",
    enforce_immutables=True,
    enforce_onehot=False,
    enforce_monotonic=False,
)
R1_STRUCTURAL = ConstraintSetting(
    name="r1_structural",
    enforce_immutables=True,
    enforce_onehot=True,
    enforce_monotonic=False,
)
R2_REALISTIC = ConstraintSetting(
    name="r2_realistic",
    enforce_immutables=True,
    enforce_onehot=True,
    enforce_monotonic=True,
)
CONSTRAINT_REGIMES = [
    R1_ALL_MUTABLE,
    R1_IMMUTABLE_ONLY,
    R1_STRUCTURAL,
    R2_REALISTIC,
]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    loader: Callable[[], Tuple[pd.DataFrame, np.ndarray]]
    drop_cols: Tuple[str, ...]
    immutable_raw: Tuple[str, ...]
    monotone_increase_raw: Tuple[str, ...]
    monotone_decrease_raw: Tuple[str, ...]
    subgroup_raw: Optional[str]


@dataclass
class TransformBundle:
    preprocessor: ColumnTransformer
    num_cols: List[str]
    cat_cols: List[str]
    feature_names: List[str]
    raw_to_indices: Dict[str, List[int]]
    onehot_groups: Dict[str, List[int]]
    train_x: np.ndarray
    train_mean: np.ndarray
    knn_all: NearestNeighbors
    knn_pos: Optional[NearestNeighbors]


@dataclass(frozen=True)
class ProjectionCandidate:
    leaf_id: int
    candidate: np.ndarray
    cost_l2: float


@dataclass
class ActionResult:
    method: str
    alpha: float
    candidate: Optional[np.ndarray]
    valid: bool
    abstained: bool
    cost_l1: float
    cost_l2: float
    sparsity: float
    plausibility: float
    utility: float
    runtime_sec: float


def get_constraint_settings(names: Sequence[str]) -> List[ConstraintSetting]:
    name_map = {c.name: c for c in CONSTRAINT_REGIMES}
    out: List[ConstraintSetting] = []
    for name in names:
        if name not in name_map:
            raise ValueError(f"unknown constraint regime: {name}")
        out.append(name_map[name])
    return out


def clone_action_result(result: ActionResult, method: str, alpha: float, runtime_sec: Optional[float] = None) -> ActionResult:
    return ActionResult(
        method=method,
        alpha=float(alpha),
        candidate=result.candidate,
        valid=result.valid,
        abstained=result.abstained,
        cost_l1=result.cost_l1,
        cost_l2=result.cost_l2,
        sparsity=result.sparsity,
        plausibility=result.plausibility,
        utility=result.utility,
        runtime_sec=result.runtime_sec if runtime_sec is None else float(runtime_sec),
    )


def ensure_dirs(exp_dir: Path, fig_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)


def downsample_frame(df: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True), y.copy()
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep_pos = min(len(pos_idx), max_rows // 2)
    keep_neg = min(len(neg_idx), max_rows - keep_pos)
    if keep_pos + keep_neg < max_rows:
        remaining = max_rows - keep_pos - keep_neg
        extra_pos = min(len(pos_idx) - keep_pos, remaining)
        keep_pos += extra_pos
        remaining -= extra_pos
        keep_neg += min(len(neg_idx) - keep_neg, remaining)
    pick_pos = rng.choice(pos_idx, size=keep_pos, replace=False)
    pick_neg = rng.choice(neg_idx, size=keep_neg, replace=False)
    chosen = np.concatenate([pick_pos, pick_neg])
    rng.shuffle(chosen)
    return df.iloc[chosen].reset_index(drop=True), y[chosen]


def load_adult() -> Tuple[pd.DataFrame, np.ndarray]:
    cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    df = pd.read_csv(
        DATA_ROOT / "adult" / "adult.data",
        header=None,
        names=cols,
        na_values="?",
        skipinitialspace=True,
    )
    df = df.dropna().reset_index(drop=True)
    x = df.drop(columns=["income", "fnlwgt", "education"]).copy()
    for col in ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    valid_mask = x.notna().all(axis=1)
    x = x.loc[valid_mask].reset_index(drop=True)
    y = (df.loc[valid_mask, "income"].astype(str).str.strip() == ">50K").astype(int).to_numpy()
    return x, y


def load_german() -> Tuple[pd.DataFrame, np.ndarray]:
    cols = [
        "checking_status",
        "duration",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_status",
        "employment",
        "installment_commitment",
        "personal_status_sex",
        "other_parties",
        "residence_since",
        "property_magnitude",
        "age",
        "other_payment_plans",
        "housing",
        "existing_credits",
        "job",
        "num_dependents",
        "own_telephone",
        "foreign_worker",
        "credit_risk",
    ]
    df = pd.read_csv(DATA_ROOT / "german" / "german.data", sep=r"\s+", header=None, names=cols)
    x = df.drop(columns=["credit_risk"]).copy()
    for col in [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    valid_mask = x.notna().all(axis=1)
    x = x.loc[valid_mask].reset_index(drop=True)
    y = (df.loc[valid_mask, "credit_risk"].astype(int) == 1).astype(int).to_numpy()
    return x, y


def load_bank() -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_ROOT / "bank" / "bank-full.csv", sep=";")
    y = (df["y"].astype(str).str.strip() == "yes").astype(int).to_numpy()
    x = df.drop(columns=["y", "duration"]).copy()
    return x.reset_index(drop=True), y


def load_compas() -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA_ROOT / "compas" / "compas-scores-two-years.csv")
    keep = df.copy()
    keep = keep[
        keep["days_b_screening_arrest"].between(-30, 30, inclusive="both")
        & (keep["is_recid"] != -1)
        & (keep["c_charge_degree"] != "O")
        & keep["score_text"].notna()
    ].reset_index(drop=True)
    cols = [
        "sex",
        "age",
        "age_cat",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
    ]
    x = keep[cols].copy()
    for col in ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    valid_mask = x.notna().all(axis=1)
    x = x.loc[valid_mask].reset_index(drop=True)
    # Positive label denotes the desirable non-recidivism outcome.
    y = (1 - keep.loc[valid_mask, "two_year_recid"].astype(int)).astype(int).to_numpy()
    return x, y


def build_datasets() -> Dict[str, DatasetConfig]:
    return {
        "adult": DatasetConfig(
            name="adult",
            loader=load_adult,
            drop_cols=(),
            immutable_raw=("age", "sex", "race", "native_country"),
            monotone_increase_raw=("education_num", "capital_gain", "hours_per_week"),
            monotone_decrease_raw=(),
            subgroup_raw="sex",
        ),
        "german": DatasetConfig(
            name="german",
            loader=load_german,
            drop_cols=(),
            immutable_raw=("age", "personal_status_sex", "foreign_worker"),
            monotone_increase_raw=(),
            monotone_decrease_raw=(),
            subgroup_raw="personal_status_sex",
        ),
        "bank": DatasetConfig(
            name="bank",
            loader=load_bank,
            drop_cols=(),
            immutable_raw=("age", "marital"),
            monotone_increase_raw=("balance",),
            monotone_decrease_raw=(),
            subgroup_raw="marital",
        ),
        "compas": DatasetConfig(
            name="compas",
            loader=load_compas,
            drop_cols=(),
            immutable_raw=("age", "sex", "race"),
            monotone_increase_raw=(),
            monotone_decrease_raw=(),
            subgroup_raw="race",
        ),
    }


def build_preprocessor(train_df: pd.DataFrame) -> TransformBundle:
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in train_df.columns if c not in num_cols]
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    train_x = np.asarray(pre.fit_transform(train_df), dtype=float)
    raw_to_indices: Dict[str, List[int]] = {}
    onehot_groups: Dict[str, List[int]] = {}
    feature_names: List[str] = []

    for idx, col in enumerate(num_cols):
        raw_to_indices[col] = [idx]
        feature_names.append(col)

    offset = len(num_cols)
    if cat_cols:
        onehot = pre.named_transformers_["cat"].named_steps["onehot"]
        for col, cats in zip(cat_cols, onehot.categories_):
            idxs = list(range(offset, offset + len(cats)))
            raw_to_indices[col] = idxs
            onehot_groups[col] = idxs
            feature_names.extend([f"{col}={cat}" for cat in cats])
            offset += len(cats)

    knn_all = NearestNeighbors(n_neighbors=min(5, len(train_x)))
    knn_all.fit(train_x)

    return TransformBundle(
        preprocessor=pre,
        num_cols=num_cols,
        cat_cols=cat_cols,
        feature_names=feature_names,
        raw_to_indices=raw_to_indices,
        onehot_groups=onehot_groups,
        train_x=train_x,
        train_mean=np.mean(train_x, axis=0),
        knn_all=knn_all,
        knn_pos=None,
    )


def transform_df(bundle: TransformBundle, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(bundle.preprocessor.transform(df), dtype=float)


def fit_target_model(model_family: str, train_x: np.ndarray, y_train: np.ndarray, depth: int, seed: int):
    if model_family == "tree":
        model = DecisionTreeClassifier(
            max_depth=int(depth),
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=int(seed),
        )
    elif model_family == "logistic":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=int(seed),
        )
    elif model_family == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=max(int(depth) + 2, 6),
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=int(seed),
            n_jobs=-1,
        )
    elif model_family == "hgbt":
        model = HistGradientBoostingClassifier(
            max_depth=max(int(depth), 3),
            max_iter=250,
            random_state=int(seed),
        )
    else:
        raise ValueError(f"unknown model family: {model_family}")
    model.fit(train_x, y_train)
    return model


def fit_route_tree(train_x: np.ndarray, route_labels: np.ndarray, depth: int, seed: int) -> DecisionTreeClassifier:
    route_tree = DecisionTreeClassifier(
        max_depth=int(depth),
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=int(seed) + 101,
    )
    route_tree.fit(train_x, route_labels)
    return route_tree


def get_leaf_boxes(clf: DecisionTreeClassifier) -> Dict[int, Dict[str, Dict[int, float]]]:
    tree = clf.tree_
    boxes: Dict[int, Dict[str, Dict[int, float]]] = {}
    stack: List[Tuple[int, Dict[int, float], Dict[int, float]]] = [(0, {}, {})]
    while stack:
        node, lb, ub = stack.pop()
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == right:
            boxes[node] = {"lb": dict(lb), "ub": dict(ub)}
            continue
        feat = int(tree.feature[node])
        thr = float(tree.threshold[node])
        left_lb = dict(lb)
        left_ub = dict(ub)
        left_ub[feat] = min(left_ub.get(feat, float("inf")), thr)
        right_lb = dict(lb)
        right_ub = dict(ub)
        right_lb[feat] = max(right_lb.get(feat, -float("inf")), float(np.nextafter(thr, float("inf"))))
        stack.append((left, left_lb, left_ub))
        stack.append((right, right_lb, right_ub))
    return boxes


def lower_bound(box: Dict[str, Dict[int, float]], idx: int) -> float:
    return float(box["lb"].get(idx, -float("inf")))


def upper_bound(box: Dict[str, Dict[int, float]], idx: int) -> float:
    return float(box["ub"].get(idx, float("inf")))


def in_bounds(val: float, lo: float, hi: float) -> bool:
    return (val + BOUND_EPS) >= lo and (val - BOUND_EPS) <= hi


def clip_projection(x: np.ndarray, box: Dict[str, Dict[int, float]]) -> np.ndarray:
    z = x.copy()
    for idx in range(len(z)):
        z[idx] = float(np.clip(z[idx], lower_bound(box, idx), upper_bound(box, idx)))
    return z


def choose_onehot_vector(
    proposal: np.ndarray,
    idxs: List[int],
    box: Optional[Dict[str, Dict[int, float]]] = None,
) -> Optional[np.ndarray]:
    clipped = proposal[idxs].copy()
    if box is not None:
        for pos, idx in enumerate(idxs):
            clipped[pos] = float(np.clip(clipped[pos], lower_bound(box, idx), upper_bound(box, idx)))
    best_vec: Optional[np.ndarray] = None
    best_score = float("inf")
    for pos, _ in enumerate(idxs):
        candidate = np.zeros(len(idxs), dtype=float)
        candidate[pos] = 1.0
        if box is not None:
            feasible = True
            for j, idx in enumerate(idxs):
                if not in_bounds(candidate[j], lower_bound(box, idx), upper_bound(box, idx)):
                    feasible = False
                    break
            if not feasible:
                continue
        score = float(np.linalg.norm(candidate - clipped))
        if score < best_score:
            best_score = score
            best_vec = candidate
    return best_vec


def project_with_constraints(
    x: np.ndarray,
    box: Dict[str, Dict[int, float]],
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
) -> Optional[np.ndarray]:
    if not (constraints.enforce_immutables or constraints.enforce_onehot or constraints.enforce_monotonic):
        return clip_projection(x, box)

    z = x.copy()
    for col in bundle.num_cols:
        idx = bundle.raw_to_indices[col][0]
        lo = lower_bound(box, idx)
        hi = upper_bound(box, idx)
        base_val = float(x[idx])
        if constraints.enforce_immutables and col in config.immutable_raw:
            if not in_bounds(base_val, lo, hi):
                return None
            z[idx] = base_val
            continue
        if constraints.enforce_monotonic:
            if col in config.monotone_increase_raw:
                lo = max(lo, base_val)
            if col in config.monotone_decrease_raw:
                hi = min(hi, base_val)
        if lo > hi + BOUND_EPS:
            return None
        z[idx] = float(np.clip(base_val, lo, hi))

    for col, idxs in bundle.onehot_groups.items():
        if constraints.enforce_onehot:
            if constraints.enforce_immutables and col in config.immutable_raw:
                original = np.zeros(len(idxs), dtype=float)
                original[int(np.argmax(x[idxs]))] = 1.0
                for pos, idx in enumerate(idxs):
                    if not in_bounds(original[pos], lower_bound(box, idx), upper_bound(box, idx)):
                        return None
                z[idxs] = original
            else:
                chosen = choose_onehot_vector(x, idxs, box=box)
                if chosen is None:
                    return None
                z[idxs] = chosen
        else:
            for idx in idxs:
                z[idx] = float(np.clip(z[idx], lower_bound(box, idx), upper_bound(box, idx)))
    return z


def enforce_actionability(
    base_query: np.ndarray,
    proposal: np.ndarray,
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
) -> np.ndarray:
    if not (constraints.enforce_immutables or constraints.enforce_onehot or constraints.enforce_monotonic):
        return proposal.copy()

    out = proposal.copy()
    for col in bundle.num_cols:
        idx = bundle.raw_to_indices[col][0]
        base_val = float(base_query[idx])
        if constraints.enforce_immutables and col in config.immutable_raw:
            out[idx] = base_val
            continue
        if constraints.enforce_monotonic:
            if col in config.monotone_increase_raw:
                out[idx] = max(out[idx], base_val)
            if col in config.monotone_decrease_raw:
                out[idx] = min(out[idx], base_val)

    if constraints.enforce_onehot:
        for col, idxs in bundle.onehot_groups.items():
            if constraints.enforce_immutables and col in config.immutable_raw:
                frozen = np.zeros(len(idxs), dtype=float)
                frozen[int(np.argmax(base_query[idxs]))] = 1.0
                out[idxs] = frozen
            else:
                chosen = choose_onehot_vector(out, idxs, box=None)
                if chosen is not None:
                    out[idxs] = chosen
    return out


def get_positive_leaf_nodes(clf: DecisionTreeClassifier, leaf_boxes: Dict[int, Dict[str, Dict[int, float]]]) -> List[int]:
    nodes: List[int] = []
    tree = clf.tree_
    for leaf_id in leaf_boxes.keys():
        pred = int(np.argmax(tree.value[leaf_id][0]))
        if pred == 1:
            nodes.append(int(leaf_id))
    return nodes


def nearest_positive_exemplar(
    query_x: np.ndarray,
    query_row: pd.Series,
    pos_x: np.ndarray,
    pos_rows: pd.DataFrame,
    config: DatasetConfig,
    retrieval_mode: str,
) -> np.ndarray:
    return positive_neighbor_pool(
        query_x=query_x,
        query_row=query_row,
        pos_x=pos_x,
        pos_rows=pos_rows,
        config=config,
        retrieval_mode=retrieval_mode,
        k=1,
    )[0]


def positive_neighbor_pool(
    query_x: np.ndarray,
    query_row: pd.Series,
    pos_x: np.ndarray,
    pos_rows: pd.DataFrame,
    config: DatasetConfig,
    retrieval_mode: str,
    k: int,
) -> np.ndarray:
    if len(pos_x) == 0:
        raise ValueError("positive pool is empty")
    if retrieval_mode != "mismatch":
        dists = np.linalg.norm(pos_x - query_x, axis=1)
        idxs = np.argsort(dists)[: max(1, min(int(k), len(dists)))]
        return pos_x[idxs]

    if config.subgroup_raw is not None and config.subgroup_raw in pos_rows.columns:
        subgroup_value = query_row[config.subgroup_raw]
        mismatch_mask = pos_rows[config.subgroup_raw].astype(str) != str(subgroup_value)
        if mismatch_mask.any():
            subset_x = pos_x[mismatch_mask.to_numpy()]
            dists = np.linalg.norm(subset_x - query_x, axis=1)
            idxs = np.argsort(dists)[: max(1, min(int(k), len(dists)))]
            return subset_x[idxs]

    dists = np.linalg.norm(pos_x - query_x, axis=1)
    if len(dists) == 1:
        return pos_x[:1]
    topk = min(max(1, int(k)), len(dists))
    idxs = np.argsort(dists)[:topk]
    return pos_x[idxs]


def best_projection_candidate(
    query_x: np.ndarray,
    clf: DecisionTreeClassifier,
    leaf_boxes: Dict[int, Dict[str, Dict[int, float]]],
    positive_leaf_nodes: List[int],
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
) -> Optional[ProjectionCandidate]:
    best: Optional[ProjectionCandidate] = None
    for leaf_id in positive_leaf_nodes:
        _ = clf
        box = leaf_boxes[int(leaf_id)]
        candidate = project_with_constraints(query_x, box, bundle, config, constraints)
        if candidate is None:
            continue
        dist = float(np.linalg.norm(candidate - query_x))
        if best is None or dist < best.cost_l2:
            best = ProjectionCandidate(leaf_id=int(leaf_id), candidate=candidate, cost_l2=dist)
    return best


def repair_toward_reference(
    clf: DecisionTreeClassifier,
    base_query: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
    steps: int = 10,
) -> np.ndarray:
    if int(clf.predict(candidate.reshape(1, -1))[0]) == 1:
        return candidate
    if int(clf.predict(reference.reshape(1, -1))[0]) != 1:
        return candidate
    lo, hi = 0.0, 1.0
    best = reference.copy()
    for _ in range(max(1, int(steps))):
        mid = 0.5 * (lo + hi)
        probe = (1.0 - mid) * candidate + mid * reference
        probe = enforce_actionability(base_query, probe, bundle, config, constraints)
        if int(clf.predict(probe.reshape(1, -1))[0]) == 1:
            best = probe
            hi = mid
        else:
            lo = mid
    return best


def repair_toward_projection(
    clf: DecisionTreeClassifier,
    base_query: np.ndarray,
    candidate: np.ndarray,
    projection: np.ndarray,
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
    steps: int = 10,
) -> np.ndarray:
    return repair_toward_reference(
        clf=clf,
        base_query=base_query,
        candidate=candidate,
        reference=projection,
        bundle=bundle,
        config=config,
        constraints=constraints,
        steps=steps,
    )


def nearest_train_distance(bundle: TransformBundle, x: np.ndarray, positive_only: bool = False) -> float:
    knn = bundle.knn_pos if positive_only else bundle.knn_all
    if knn is None:
        return float("nan")
    dists, _ = knn.kneighbors(x.reshape(1, -1), n_neighbors=min(5, knn.n_samples_fit_))
    return float(np.mean(dists[0]))


def feature_bounds(bundle: TransformBundle) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    bounds: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
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
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
    bounds: Dict[int, Tuple[Optional[float], Optional[float]]],
    time_limit_sec: float,
) -> Optional[np.ndarray]:
    if pulp is None:
        return None
    prob = pulp.LpProblem("tree_milp_recourse", pulp.LpMinimize)
    x_vars: Dict[int, object] = {}
    d_vars: Dict[int, object] = {}
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
        lo = lower_bound(leaf_box, idx)
        hi = upper_bound(leaf_box, idx)
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
        elif constraints.enforce_onehot:
            prob += pulp.lpSum(x_vars[idx] for idx in idxs) == 1.0
        for idx in idxs:
            lo = lower_bound(leaf_box, idx)
            hi = upper_bound(leaf_box, idx)
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
    if constraints.enforce_onehot:
        for col, idxs in bundle.onehot_groups.items():
            if constraints.enforce_immutables and col in config.immutable_raw:
                continue
            chosen = choose_onehot_vector(candidate, idxs, box=None)
            if chosen is not None:
                candidate[idxs] = chosen
    return candidate


def best_exact_candidate(
    base_query: np.ndarray,
    eval_model,
    leaf_boxes: Dict[int, Dict[str, Dict[int, float]]],
    positive_leaf_nodes: List[int],
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
    time_limit_sec: float,
) -> Optional[np.ndarray]:
    if pulp is None:
        return None
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
        if int(eval_model.predict(candidate.reshape(1, -1))[0]) != 1:
            continue
        cost = float(np.sum(np.abs(candidate - base_query)))
        if cost < best_cost:
            best_cost = cost
            best_candidate = candidate
    return best_candidate


def leaf_slack(candidate: np.ndarray, box: Dict[str, Dict[int, float]]) -> float:
    slacks: List[float] = []
    for idx in range(len(candidate)):
        lo = lower_bound(box, idx)
        hi = upper_bound(box, idx)
        if math.isfinite(lo):
            slacks.append(float(candidate[idx] - lo))
        if math.isfinite(hi):
            slacks.append(float(hi - candidate[idx]))
    if not slacks:
        return 0.0
    return float(min(slacks))


def immutable_gap(
    base_query: np.ndarray,
    exemplar: np.ndarray,
    bundle: TransformBundle,
    config: DatasetConfig,
) -> float:
    total = 0.0
    for col in config.immutable_raw:
        idxs = bundle.raw_to_indices.get(col, [])
        if not idxs:
            continue
        if len(idxs) == 1:
            total += float(abs(base_query[idxs[0]] - exemplar[idxs[0]]))
        else:
            total += float(np.argmax(base_query[idxs]) != np.argmax(exemplar[idxs]))
    return total


def category_disagreement(
    projection: np.ndarray,
    exemplar: np.ndarray,
    bundle: TransformBundle,
) -> float:
    total = 0.0
    for idxs in bundle.onehot_groups.values():
        total += float(np.argmax(projection[idxs]) != np.argmax(exemplar[idxs]))
    return total


def evaluate_candidate(
    method: str,
    alpha: float,
    base_query: np.ndarray,
    candidate: Optional[np.ndarray],
    bundle: TransformBundle,
    clf: DecisionTreeClassifier,
    start_time: float,
    abstained: bool = False,
    abstain_penalty: Optional[float] = None,
) -> ActionResult:
    runtime = float(time.perf_counter() - start_time)
    if abstained or candidate is None:
        penalty = float(abstain_penalty if abstain_penalty is not None else INVALID_PENALTY)
        return ActionResult(
            method=method,
            alpha=float(alpha),
            candidate=None,
            valid=False,
            abstained=bool(abstained),
            cost_l1=float("nan"),
            cost_l2=float("nan"),
            sparsity=float("nan"),
            plausibility=float("nan"),
            utility=penalty,
            runtime_sec=runtime,
        )

    valid = int(clf.predict(candidate.reshape(1, -1))[0]) == 1
    cost_l1 = float(np.sum(np.abs(candidate - base_query)))
    cost_l2 = float(np.linalg.norm(candidate - base_query))
    sparsity = float(np.sum(np.abs(candidate - base_query) > DIFF_EPS))
    plausibility = nearest_train_distance(bundle, candidate, positive_only=False)
    utility = (
        (0.0 if valid else INVALID_PENALTY)
        + cost_l1
        + SPARSITY_WEIGHT * sparsity
        + PLAUSIBILITY_WEIGHT * (0.0 if math.isnan(plausibility) else plausibility)
    )
    return ActionResult(
        method=method,
        alpha=float(alpha),
        candidate=candidate,
        valid=valid,
        abstained=False,
        cost_l1=cost_l1,
        cost_l2=cost_l2,
        sparsity=sparsity,
        plausibility=plausibility,
        utility=float(utility),
        runtime_sec=runtime,
    )


def build_alpha_candidate(
    base_query: np.ndarray,
    projection: Optional[ProjectionCandidate],
    exemplar: np.ndarray,
    alpha: float,
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
    clf: DecisionTreeClassifier,
    do_repair: bool = True,
) -> Optional[np.ndarray]:
    if projection is None:
        return None
    if alpha >= 1.0 - 1e-9:
        return projection.candidate.copy()
    candidate = alpha * projection.candidate + (1.0 - alpha) * exemplar
    candidate = enforce_actionability(base_query, candidate, bundle, config, constraints)
    if do_repair:
        candidate = repair_toward_projection(
            clf=clf,
            base_query=base_query,
            candidate=candidate,
            projection=projection.candidate,
            bundle=bundle,
            config=config,
            constraints=constraints,
        )
    return candidate


def build_nn_candidate(
    base_query: np.ndarray,
    exemplar: np.ndarray,
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
) -> np.ndarray:
    return enforce_actionability(base_query, exemplar.copy(), bundle, config, constraints)


def build_knn_mean_candidate(
    base_query: np.ndarray,
    neighbors: np.ndarray,
    exemplar: np.ndarray,
    bundle: TransformBundle,
    config: DatasetConfig,
    constraints: ConstraintSetting,
    clf: DecisionTreeClassifier,
) -> np.ndarray:
    candidate = np.mean(neighbors, axis=0)
    candidate = enforce_actionability(base_query, candidate, bundle, config, constraints)
    return repair_toward_reference(
        clf=clf,
        base_query=base_query,
        candidate=candidate,
        reference=exemplar,
        bundle=bundle,
        config=config,
        constraints=constraints,
    )


def compute_query_features(
    base_query: np.ndarray,
    projection: ProjectionCandidate,
    exemplar: np.ndarray,
    bundle: TransformBundle,
    config: DatasetConfig,
    clf: DecisionTreeClassifier,
    fixed_alpha: float,
    constraints: ConstraintSetting,
    leaf_boxes: Dict[int, Dict[str, Dict[int, float]]],
) -> Dict[str, float]:
    d_proj = float(np.linalg.norm(projection.candidate - base_query))
    d_nn = float(np.linalg.norm(exemplar - base_query))
    d_mix = float(np.linalg.norm(projection.candidate - exemplar))
    ood_score = nearest_train_distance(bundle, base_query, positive_only=False)
    pos_density = nearest_train_distance(bundle, base_query, positive_only=True)
    imm_gap = immutable_gap(base_query, exemplar, bundle, config)
    cat_gap = category_disagreement(projection.candidate, exemplar, bundle)

    raw_blend = fixed_alpha * projection.candidate + (1.0 - fixed_alpha) * exemplar
    raw_blend = enforce_actionability(base_query, raw_blend, bundle, config, constraints)
    repair_needed = float(int(clf.predict(raw_blend.reshape(1, -1))[0]) != 1)

    return {
        "d_proj": d_proj,
        "d_nn": d_nn,
        "d_mix": d_mix,
        "ood_score": 0.0 if math.isnan(ood_score) else ood_score,
        "pos_density": 0.0 if math.isnan(pos_density) else pos_density,
        "immutable_gap": imm_gap,
        "category_gap": cat_gap,
        "repair_needed": repair_needed,
        "leaf_slack": leaf_slack(projection.candidate, leaf_boxes[projection.leaf_id]),
    }


def apply_setting(
    setting: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    config: DatasetConfig,
) -> Tuple[pd.DataFrame, str]:
    if setting == "in_domain":
        return eval_df.copy(), "standard"
    if setting == "exemplar_mismatch":
        return eval_df.copy(), "mismatch"
    if setting == "subgroup_shift":
        if config.subgroup_raw is None or config.subgroup_raw not in train_df.columns or config.subgroup_raw not in eval_df.columns:
            return eval_df.copy(), "standard"
        counts = train_df[config.subgroup_raw].astype(str).value_counts()
        if len(counts) <= 1:
            return eval_df.copy(), "standard"
        subgroup_value = str(counts.index[-1])
        subset = eval_df[eval_df[config.subgroup_raw].astype(str) == subgroup_value].copy()
        if subset.empty:
            return eval_df.copy(), "standard"
        return subset.reset_index(drop=True), "standard"

    shifted = eval_df.copy()
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    shift_cols = [c for c in num_cols if c not in config.immutable_raw]
    stats = train_df[shift_cols].agg(["mean", "std"]).T if shift_cols else pd.DataFrame()
    scales = [0.75, -0.45, 0.30, -0.20]
    for col, scale in zip(shift_cols[:4], scales):
        std = float(stats.loc[col, "std"]) if col in stats.index else 0.0
        if std > 0:
            shifted[col] = shifted[col].astype(float) + scale * std
    return shifted, "standard"


def sanitize_feature_matrix(rows: List[Dict[str, float]], feature_order: Sequence[str]) -> np.ndarray:
    x = np.array([[row[name] for name in feature_order] for row in rows], dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


def tune_fixed_alpha(calib_rows: List[Dict[str, object]]) -> float:
    alpha_scores: Dict[float, List[float]] = {alpha: [] for alpha in ALPHA_GRID}
    for row in calib_rows:
        per_alpha = row["alpha_results"]
        for alpha, result in per_alpha.items():
            alpha_scores[float(alpha)].append(float(result.utility))
    best_alpha = 1.0
    best_score = float("inf")
    for alpha, vals in alpha_scores.items():
        if not vals:
            continue
        score = float(np.mean(vals))
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def fit_router_model(
    calib_rows: List[Dict[str, object]],
    labels: np.ndarray,
    feature_order: Sequence[str],
) -> DummyClassifier | Pipeline:
    x = sanitize_feature_matrix([row["features"] for row in calib_rows], feature_order)
    if len(np.unique(labels)) <= 1:
        model: DummyClassifier | Pipeline = DummyClassifier(strategy="constant", constant=int(labels[0]))
    else:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
    model.fit(x, labels)
    return model


def fit_utility_model(
    calib_rows: List[Dict[str, object]],
    targets: np.ndarray,
    feature_order: Sequence[str],
) -> DummyRegressor | RandomForestRegressor:
    x = sanitize_feature_matrix([row["features"] for row in calib_rows], feature_order)
    if len(np.unique(np.round(targets, 8))) <= 1:
        model: DummyRegressor | RandomForestRegressor = DummyRegressor(strategy="constant", constant=float(targets[0]))
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=1,
            n_jobs=-1,
        )
    model.fit(x, targets)
    return model


def fit_learned_models(
    calib_rows: List[Dict[str, object]],
    fixed_alpha: float,
) -> Tuple[
    DummyRegressor | RandomForestRegressor,
    Dict[str, DummyClassifier | Pipeline],
    Dict[str, Dict[str, DummyRegressor | RandomForestRegressor]],
    Dict[str, float],
]:
    x = sanitize_feature_matrix([row["features"] for row in calib_rows], FULL_FEATURE_ORDER)
    y_alpha = np.array([row["best_alpha"] for row in calib_rows], dtype=float)
    y_router = np.array(
        [
            int(float(row["alpha_results"][1.0].utility) <= float(row["alpha_results"][fixed_alpha].utility))
            for row in calib_rows
        ],
        dtype=int,
    )

    if len(np.unique(y_alpha)) <= 1:
        alpha_model: DummyRegressor | RandomForestRegressor = DummyRegressor(strategy="constant", constant=float(y_alpha[0]))
    else:
        alpha_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            random_state=0,
            n_jobs=-1,
        )
    alpha_model.fit(x, y_alpha)

    proj_targets = np.array([float(row["alpha_results"][1.0].utility) for row in calib_rows], dtype=float)
    blend_targets = np.array([float(row["alpha_results"][fixed_alpha].utility) for row in calib_rows], dtype=float)
    oracle_margins = np.abs(proj_targets - blend_targets)

    routers = {
        "main": fit_router_model(calib_rows, y_router, FULL_FEATURE_ORDER),
        "no_shift": fit_router_model(calib_rows, y_router, NO_SHIFT_FEATURE_ORDER),
        "no_geometry": fit_router_model(calib_rows, y_router, NO_GEOMETRY_FEATURE_ORDER),
    }
    utility_models = {
        "main": {
            "projection": fit_utility_model(calib_rows, proj_targets, FULL_FEATURE_ORDER),
            "blend": fit_utility_model(calib_rows, blend_targets, FULL_FEATURE_ORDER),
        },
        "no_shift": {
            "projection": fit_utility_model(calib_rows, proj_targets, NO_SHIFT_FEATURE_ORDER),
            "blend": fit_utility_model(calib_rows, blend_targets, NO_SHIFT_FEATURE_ORDER),
        },
        "no_geometry": {
            "projection": fit_utility_model(calib_rows, proj_targets, NO_GEOMETRY_FEATURE_ORDER),
            "blend": fit_utility_model(calib_rows, blend_targets, NO_GEOMETRY_FEATURE_ORDER),
        },
    }

    ood_vals = [float(row["features"]["ood_score"]) for row in calib_rows]
    best_utils = [float(row["oracle_utility"]) for row in calib_rows if math.isfinite(float(row["oracle_utility"]))]
    thresholds = {
        "rule_ood_q75": float(np.quantile(ood_vals, 0.75)) if ood_vals else 0.0,
        "safe_utility_threshold": (
            float(np.quantile(best_utils, 0.85) + ABSTAIN_BUFFER) if best_utils else INVALID_PENALTY + ABSTAIN_BUFFER
        ),
        "exact_margin_q25": float(np.quantile(oracle_margins, 0.25)) if len(oracle_margins) else 0.0,
    }
    return alpha_model, routers, utility_models, thresholds


def alpha_prediction(alpha_model: DummyRegressor | RandomForestRegressor, feature_row: Dict[str, float]) -> float:
    x = sanitize_feature_matrix([feature_row], FULL_FEATURE_ORDER)
    pred = float(alpha_model.predict(x)[0])
    return float(np.clip(pred, min(ALPHA_GRID), max(ALPHA_GRID)))


def router_prediction(
    router_model: DummyClassifier | Pipeline,
    feature_row: Dict[str, float],
    feature_order: Sequence[str],
) -> int:
    x = sanitize_feature_matrix([feature_row], feature_order)
    return int(router_model.predict(x)[0])


def utility_route_prediction(
    utility_model_pair: Dict[str, DummyRegressor | RandomForestRegressor],
    feature_row: Dict[str, float],
    feature_order: Sequence[str],
) -> int:
    x = sanitize_feature_matrix([feature_row], feature_order)
    pred_proj = float(utility_model_pair["projection"].predict(x)[0])
    pred_blend = float(utility_model_pair["blend"].predict(x)[0])
    return int(pred_proj <= pred_blend)


def predict_route_utilities(
    utility_model_pair: Dict[str, DummyRegressor | RandomForestRegressor],
    feature_row: Dict[str, float],
    feature_order: Sequence[str],
) -> Tuple[float, float]:
    x = sanitize_feature_matrix([feature_row], feature_order)
    pred_proj = float(utility_model_pair["projection"].predict(x)[0])
    pred_blend = float(utility_model_pair["blend"].predict(x)[0])
    return pred_proj, pred_blend


def nearest_grid_alpha(alpha: float) -> float:
    return float(min(ALPHA_GRID, key=lambda a: abs(a - alpha)))


def prepare_calibration_rows(
    config: DatasetConfig,
    constraints: ConstraintSetting,
    bundle: TransformBundle,
    clf: DecisionTreeClassifier,
    leaf_boxes: Dict[int, Dict[str, Dict[int, float]]],
    positive_leaf_nodes: List[int],
    pos_train_x: np.ndarray,
    pos_train_rows: pd.DataFrame,
    train_df_raw: pd.DataFrame,
    calib_df_raw: pd.DataFrame,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for setting in SETTINGS:
        eval_raw, retrieval_mode = apply_setting(setting, train_df_raw, calib_df_raw, config)
        eval_x = transform_df(bundle, eval_raw)
        pred = clf.predict(eval_x)
        neg_idx = np.where(pred == 0)[0]
        for idx in neg_idx:
            x0 = eval_x[int(idx)]
            row_raw = eval_raw.iloc[int(idx)]
            projection = best_projection_candidate(
                query_x=x0,
                clf=clf,
                leaf_boxes=leaf_boxes,
                positive_leaf_nodes=positive_leaf_nodes,
                bundle=bundle,
                config=config,
                constraints=constraints,
            )
            if projection is None:
                continue
            exemplar = nearest_positive_exemplar(
                query_x=x0,
                query_row=row_raw,
                pos_x=pos_train_x,
                pos_rows=pos_train_rows,
                config=config,
                retrieval_mode=retrieval_mode,
            )
            alpha_results: Dict[float, ActionResult] = {}
            for alpha in ALPHA_GRID:
                start = time.perf_counter()
                candidate = build_alpha_candidate(
                    base_query=x0,
                    projection=projection,
                    exemplar=exemplar,
                    alpha=float(alpha),
                    bundle=bundle,
                    config=config,
                    constraints=constraints,
                    clf=clf,
                )
                result = evaluate_candidate(
                    method=f"alpha_{alpha:.2f}",
                    alpha=float(alpha),
                    base_query=x0,
                    candidate=candidate,
                    bundle=bundle,
                    clf=clf,
                    start_time=start,
                )
                alpha_results[float(alpha)] = result
            best_alpha = float(min(alpha_results.items(), key=lambda kv: kv[1].utility)[0])
            features = compute_query_features(
                base_query=x0,
                projection=projection,
                exemplar=exemplar,
                bundle=bundle,
                config=config,
                clf=clf,
                fixed_alpha=0.90,
                constraints=constraints,
                leaf_boxes=leaf_boxes,
            )
            rows.append(
                {
                    "setting": setting,
                    "features": features,
                    "best_alpha": best_alpha,
                    "alpha_results": alpha_results,
                    "oracle_utility": float(alpha_results[best_alpha].utility),
                }
            )
    return rows


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "model_family", "seed", "depth", "setting", "constraint", "method"], as_index=False)[
            [
                "valid",
                "abstained",
                "cost_l1",
                "cost_l2",
                "sparsity",
                "plausibility",
                "utility",
                "regret",
                "route_accuracy",
                "runtime_sec",
                "chosen_alpha",
                "projection_route",
            ]
        ]
        .mean()
    )


def summarize_subgroups(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subgroup_summary = (
        df.groupby(["dataset", "model_family", "seed", "depth", "setting", "constraint", "method", "subgroup"], as_index=False)[
            ["valid", "abstained", "cost_l1", "regret", "route_accuracy"]
        ]
        .mean()
    )
    disparity_rows: List[Dict[str, object]] = []
    for keys, part in subgroup_summary.groupby(["dataset", "model_family", "seed", "depth", "setting", "constraint", "method"]):
        if len(part) <= 1:
            continue
        disparity_rows.append(
            {
                "dataset": keys[0],
                "model_family": keys[1],
                "seed": keys[2],
                "depth": keys[3],
                "setting": keys[4],
                "constraint": keys[5],
                "method": keys[6],
                "n_subgroups": int(len(part)),
                "valid_gap": float(part["valid"].max() - part["valid"].min()),
                "cost_l1_gap": float(part["cost_l1"].max() - part["cost_l1"].min()),
                "regret_gap": float(part["regret"].max() - part["regret"].min()),
                "route_accuracy_gap": float(part["route_accuracy"].max() - part["route_accuracy"].min()),
            }
        )
    disparity_df = pd.DataFrame(disparity_rows)
    return subgroup_summary, disparity_df


def make_figures(summary_df: pd.DataFrame, fig_dir: Path) -> None:
    if summary_df.empty:
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

    method_df = (
        summary_df.groupby(["constraint", "method"], as_index=False)[["valid", "cost_l1"]]
        .mean()
        .sort_values(["constraint", "valid", "cost_l1"], ascending=[True, False, True])
    )
    if not method_df.empty:
        plt.figure(figsize=(10, 6))
        for constraint, part in method_df.groupby("constraint"):
            plt.scatter(part["cost_l1"], part["valid"], s=90, label=constraint)
            for _, row in part.iterrows():
                plt.text(row["cost_l1"], row["valid"], row["method"], fontsize=11)
        plt.xlabel("Mean L1 cost")
        plt.ylabel("Mean validity")
        plt.title("Adaptive Routing Pilot Pareto View")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "pilot_pareto.png", dpi=100)
        plt.close()

    regret_df = summary_df.groupby(["setting", "method"], as_index=False)["regret"].mean()
    if not regret_df.empty:
        pivot = regret_df.pivot(index="setting", columns="method", values="regret").fillna(np.nan)
        pivot.plot(kind="bar", figsize=(10, 6))
        plt.ylabel("Mean regret to alpha-grid oracle")
        plt.title("Regret by Shift Setting")
        plt.axhline(0.0, color="black", linewidth=1)
        plt.tight_layout()
        plt.savefig(fig_dir / "pilot_regret_by_setting.png", dpi=100)
        plt.close()


def run_experiment(
    dataset_names: Sequence[str],
    model_families: Sequence[str],
    constraint_settings: Sequence[ConstraintSetting],
    seeds: Sequence[int],
    depths: Sequence[int],
    max_rows: int,
    exp_dir: Path,
    fig_dir: Path,
    enable_exact_cascade: bool = False,
    include_exact_baseline: bool = False,
    exact_time_limit_sec: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    dataset_map = build_datasets()
    query_rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []

    for dataset_name in dataset_names:
        config = dataset_map[dataset_name]
        full_df, full_y = config.loader()
        full_df, full_y = downsample_frame(full_df, full_y, max_rows=max_rows, seed=11)

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
            val_df = full_df.iloc[val_idx].reset_index(drop=True)
            test_df = full_df.iloc[test_idx].reset_index(drop=True)
            y_train = full_y[train_idx]

            bundle = build_preprocessor(train_df)
            for depth in depths:
                for model_family in model_families:
                    eval_model = fit_target_model(
                        model_family=model_family,
                        train_x=bundle.train_x,
                        y_train=y_train,
                        depth=int(depth),
                        seed=int(seed),
                    )
                    train_pred = np.asarray(eval_model.predict(bundle.train_x), dtype=int)
                    if len(np.unique(train_pred)) <= 1:
                        continue

                    pos_mask = train_pred == 1
                    if int(np.sum(pos_mask)) == 0:
                        continue

                    pos_train_x = bundle.train_x[pos_mask]
                    pos_train_rows = train_df.iloc[np.where(pos_mask)[0]].reset_index(drop=True)
                    bundle.knn_pos = NearestNeighbors(n_neighbors=min(5, len(pos_train_x)))
                    bundle.knn_pos.fit(pos_train_x)

                    if model_family == "tree" and isinstance(eval_model, DecisionTreeClassifier):
                        route_tree = eval_model
                    else:
                        route_tree = fit_route_tree(
                            train_x=bundle.train_x,
                            route_labels=train_pred,
                            depth=int(depth),
                            seed=int(seed),
                        )

                    route_train_pred = np.asarray(route_tree.predict(bundle.train_x), dtype=int)
                    if len(np.unique(route_train_pred)) <= 1 or int(np.sum(route_train_pred == 1)) == 0:
                        continue

                    leaf_boxes = get_leaf_boxes(route_tree)
                    positive_leaf_nodes = get_positive_leaf_nodes(route_tree, leaf_boxes)
                    if not positive_leaf_nodes:
                        continue

                    for constraints in constraint_settings:
                        calib_rows = prepare_calibration_rows(
                            config=config,
                            constraints=constraints,
                            bundle=bundle,
                            clf=eval_model,
                            leaf_boxes=leaf_boxes,
                            positive_leaf_nodes=positive_leaf_nodes,
                            pos_train_x=pos_train_x,
                            pos_train_rows=pos_train_rows,
                            train_df_raw=train_df,
                            calib_df_raw=val_df,
                        )
                        if not calib_rows:
                            continue

                        fixed_alpha = tune_fixed_alpha(calib_rows)
                        alpha_model, router_models, utility_models, thresholds = fit_learned_models(
                            calib_rows, fixed_alpha=fixed_alpha
                        )
                        tuning_rows.append(
                            {
                                "dataset": dataset_name,
                                "model_family": model_family,
                                "seed": seed,
                                "depth": depth,
                                "constraint": constraints.name,
                                "fixed_alpha": fixed_alpha,
                                "rule_ood_q75": thresholds["rule_ood_q75"],
                                "safe_utility_threshold": thresholds["safe_utility_threshold"],
                                "exact_margin_q25": thresholds["exact_margin_q25"],
                                "n_calibration_queries": len(calib_rows),
                            }
                        )

                        for setting in SETTINGS:
                            eval_raw, retrieval_mode = apply_setting(setting, train_df, test_df, config)
                            eval_x = transform_df(bundle, eval_raw)
                            pred = eval_model.predict(eval_x)
                            neg_idx = np.where(pred == 0)[0]

                            for idx in neg_idx:
                                x0 = eval_x[int(idx)]
                                row_raw = eval_raw.iloc[int(idx)]
                                projection = best_projection_candidate(
                                    query_x=x0,
                                    clf=route_tree,
                                    leaf_boxes=leaf_boxes,
                                    positive_leaf_nodes=positive_leaf_nodes,
                                    bundle=bundle,
                                    config=config,
                                    constraints=constraints,
                                )
                                if projection is None:
                                    continue

                                exemplar = nearest_positive_exemplar(
                                    query_x=x0,
                                    query_row=row_raw,
                                    pos_x=pos_train_x,
                                    pos_rows=pos_train_rows,
                                    config=config,
                                    retrieval_mode=retrieval_mode,
                                )
                                neighbor_pool = positive_neighbor_pool(
                                    query_x=x0,
                                    query_row=row_raw,
                                    pos_x=pos_train_x,
                                    pos_rows=pos_train_rows,
                                    config=config,
                                    retrieval_mode=retrieval_mode,
                                    k=5,
                                )
                                features = compute_query_features(
                                    base_query=x0,
                                    projection=projection,
                                    exemplar=exemplar,
                                    bundle=bundle,
                                    config=config,
                                    clf=eval_model,
                                    fixed_alpha=fixed_alpha,
                                    constraints=constraints,
                                    leaf_boxes=leaf_boxes,
                                )

                                oracle_results: Dict[float, ActionResult] = {}
                                for alpha in ALPHA_GRID:
                                    start = time.perf_counter()
                                    candidate = build_alpha_candidate(
                                        base_query=x0,
                                        projection=projection,
                                        exemplar=exemplar,
                                        alpha=float(alpha),
                                        bundle=bundle,
                                        config=config,
                                        constraints=constraints,
                                        clf=eval_model,
                                    )
                                    oracle_results[float(alpha)] = evaluate_candidate(
                                        method=f"alpha_{alpha:.2f}",
                                        alpha=float(alpha),
                                        base_query=x0,
                                        candidate=candidate,
                                        bundle=bundle,
                                        clf=eval_model,
                                        start_time=start,
                                    )
                                oracle_alpha = float(min(oracle_results.items(), key=lambda kv: kv[1].utility)[0])
                                oracle_utility = float(oracle_results[oracle_alpha].utility)
                                safe_threshold = float(thresholds["safe_utility_threshold"])

                                learned_alpha = alpha_prediction(alpha_model, features)
                                if (
                                    features["ood_score"] > thresholds["rule_ood_q75"]
                                    or features["d_proj"] <= 0.90 * max(features["d_nn"], 1e-9)
                                    or features["repair_needed"] > 0.5
                                ):
                                    rule_alpha = 1.0
                                else:
                                    rule_alpha = fixed_alpha
                                router_alpha = (
                                    1.0 if router_prediction(router_models["main"], features, FULL_FEATURE_ORDER) == 1 else fixed_alpha
                                )
                                router_no_shift_alpha = (
                                    1.0 if router_prediction(router_models["no_shift"], features, NO_SHIFT_FEATURE_ORDER) == 1 else fixed_alpha
                                )
                                router_no_geometry_alpha = (
                                    1.0
                                    if router_prediction(router_models["no_geometry"], features, NO_GEOMETRY_FEATURE_ORDER) == 1
                                    else fixed_alpha
                                )
                                utility_router_alpha = (
                                    1.0
                                    if utility_route_prediction(utility_models["main"], features, FULL_FEATURE_ORDER) == 1
                                    else fixed_alpha
                                )
                                utility_router_no_shift_alpha = (
                                    1.0
                                    if utility_route_prediction(utility_models["no_shift"], features, NO_SHIFT_FEATURE_ORDER) == 1
                                    else fixed_alpha
                                )
                                utility_router_no_geometry_alpha = (
                                    1.0
                                    if utility_route_prediction(utility_models["no_geometry"], features, NO_GEOMETRY_FEATURE_ORDER) == 1
                                    else fixed_alpha
                                )
                                pred_proj_u, pred_blend_u = predict_route_utilities(
                                    utility_models["main"], features, FULL_FEATURE_ORDER
                                )
                                proj_valid = bool(oracle_results[1.0].valid)
                                fixed_valid = bool(oracle_results[nearest_grid_alpha(fixed_alpha)].valid)
                                utility_router_guarded_alpha = utility_router_alpha
                                if abs(utility_router_guarded_alpha - 1.0) < 1e-9 and (not proj_valid) and fixed_valid:
                                    utility_router_guarded_alpha = fixed_alpha
                                elif abs(utility_router_guarded_alpha - fixed_alpha) < 1e-9 and (not fixed_valid) and proj_valid:
                                    utility_router_guarded_alpha = 1.0
                                if abs(utility_router_guarded_alpha - nearest_grid_alpha(utility_router_guarded_alpha)) < 1e-9:
                                    guarded_result_cached = clone_action_result(
                                        oracle_results[nearest_grid_alpha(utility_router_guarded_alpha)],
                                        "utility_router_guarded",
                                        utility_router_guarded_alpha,
                                    )
                                else:
                                    guarded_candidate = build_alpha_candidate(
                                        base_query=x0,
                                        projection=projection,
                                        exemplar=exemplar,
                                        alpha=utility_router_guarded_alpha,
                                        bundle=bundle,
                                        config=config,
                                        constraints=constraints,
                                        clf=eval_model,
                                        do_repair=True,
                                    )
                                    guarded_result_cached = evaluate_candidate(
                                        method="utility_router_guarded",
                                        alpha=utility_router_guarded_alpha,
                                        base_query=x0,
                                        candidate=guarded_candidate,
                                        bundle=bundle,
                                        clf=eval_model,
                                        start_time=time.perf_counter(),
                                    )
                                nn_candidate = build_nn_candidate(
                                    base_query=x0,
                                    exemplar=exemplar,
                                    bundle=bundle,
                                    config=config,
                                    constraints=constraints,
                                )
                                knn_mean_candidate = build_knn_mean_candidate(
                                    base_query=x0,
                                    neighbors=neighbor_pool,
                                    exemplar=exemplar,
                                    bundle=bundle,
                                    config=config,
                                    constraints=constraints,
                                    clf=eval_model,
                                )

                                should_escalate_exact = bool(
                                    enable_exact_cascade
                                    and pulp is not None
                                    and (
                                        abs(pred_proj_u - pred_blend_u) <= thresholds["exact_margin_q25"]
                                        or ((not proj_valid) and (not fixed_valid))
                                        or (features["repair_needed"] > 0.5 and features["ood_score"] > thresholds["rule_ood_q75"])
                                    )
                                )
                                exact_candidate: Optional[np.ndarray] = None
                                exact_result: Optional[ActionResult] = None
                                accepted_exact = False

                                method_candidates = [
                                    ("nn_positive_train", 0.0, None, None),
                                    ("knn_mean_k5", 0.0, None, None),
                                    ("projection_only", 1.0, False, True),
                                    ("fixed_blend", fixed_alpha, False, True),
                                    ("fixed_blend_no_repair", fixed_alpha, False, False),
                                    ("learned_alpha", learned_alpha, False, True),
                                    ("rule_router", rule_alpha, False, True),
                                    ("learned_router", router_alpha, False, True),
                                    ("learned_router_no_shift", router_no_shift_alpha, False, True),
                                    ("learned_router_no_geometry", router_no_geometry_alpha, False, True),
                                    ("utility_router", utility_router_alpha, False, True),
                                    ("utility_router_guarded", utility_router_guarded_alpha, False, True),
                                    ("utility_router_no_shift", utility_router_no_shift_alpha, False, True),
                                    ("utility_router_no_geometry", utility_router_no_geometry_alpha, False, True),
                                    ("safe_router", router_alpha, True, True),
                                ]
                                if include_exact_baseline:
                                    method_candidates.append(("exact_tree_milp", -1.0, False, True))
                                if enable_exact_cascade:
                                    method_candidates.append(("cascade_exact_auto", utility_router_guarded_alpha, False, True))

                                query_id = (
                                    f"{dataset_name}|model={model_family}|seed={seed}|depth={depth}|"
                                    f"constraint={constraints.name}|setting={setting}|row={int(idx)}"
                                )
                                subgroup_value = (
                                    str(row_raw[config.subgroup_raw]) if config.subgroup_raw is not None and config.subgroup_raw in row_raw.index else "NA"
                                )

                                for method_name, alpha, can_abstain, use_repair in method_candidates:
                                    alpha = float(alpha)
                                    if method_name == "nn_positive_train":
                                        start = time.perf_counter()
                                        result = evaluate_candidate(
                                            method=method_name,
                                            alpha=alpha,
                                            base_query=x0,
                                            candidate=nn_candidate,
                                            bundle=bundle,
                                            clf=eval_model,
                                            start_time=start,
                                        )
                                    elif method_name == "knn_mean_k5":
                                        start = time.perf_counter()
                                        result = evaluate_candidate(
                                            method=method_name,
                                            alpha=alpha,
                                            base_query=x0,
                                            candidate=knn_mean_candidate,
                                            bundle=bundle,
                                            clf=eval_model,
                                            start_time=start,
                                        )
                                    elif method_name == "exact_tree_milp":
                                        start = time.perf_counter()
                                        if exact_candidate is None:
                                            exact_candidate = best_exact_candidate(
                                                base_query=x0,
                                                eval_model=eval_model,
                                                leaf_boxes=leaf_boxes,
                                                positive_leaf_nodes=positive_leaf_nodes,
                                                bundle=bundle,
                                                config=config,
                                                constraints=constraints,
                                                time_limit_sec=float(exact_time_limit_sec),
                                            )
                                        if exact_result is None:
                                            exact_result = evaluate_candidate(
                                                method=method_name,
                                                alpha=alpha,
                                                base_query=x0,
                                                candidate=exact_candidate,
                                                bundle=bundle,
                                                clf=eval_model,
                                                start_time=start,
                                            )
                                        result = clone_action_result(exact_result, method_name, alpha)
                                    elif method_name == "cascade_exact_auto":
                                        if should_escalate_exact:
                                            start = time.perf_counter()
                                            if exact_candidate is None:
                                                exact_candidate = best_exact_candidate(
                                                    base_query=x0,
                                                    eval_model=eval_model,
                                                    leaf_boxes=leaf_boxes,
                                                    positive_leaf_nodes=positive_leaf_nodes,
                                                    bundle=bundle,
                                                    config=config,
                                                    constraints=constraints,
                                                    time_limit_sec=float(exact_time_limit_sec),
                                                )
                                            if exact_result is None:
                                                exact_result = evaluate_candidate(
                                                    method="exact_tree_milp",
                                                    alpha=alpha,
                                                    base_query=x0,
                                                    candidate=exact_candidate,
                                                    bundle=bundle,
                                                    clf=eval_model,
                                                    start_time=start,
                                                )
                                            if exact_result.valid and exact_result.utility + 1e-9 < guarded_result_cached.utility:
                                                accepted_exact = True
                                                result = clone_action_result(
                                                    exact_result,
                                                    method_name,
                                                    alpha,
                                                    runtime_sec=exact_result.runtime_sec,
                                                )
                                            else:
                                                result = clone_action_result(
                                                    guarded_result_cached,
                                                    method_name,
                                                    alpha,
                                                    runtime_sec=exact_result.runtime_sec,
                                                )
                                        else:
                                            result = clone_action_result(guarded_result_cached, method_name, alpha)
                                    elif method_name == "utility_router_guarded":
                                        result = clone_action_result(guarded_result_cached, method_name, alpha)
                                    elif use_repair and any(abs(alpha - grid_alpha) < 1e-9 for grid_alpha in ALPHA_GRID):
                                        result = clone_action_result(oracle_results[nearest_grid_alpha(alpha)], method_name, alpha)
                                    else:
                                        start = time.perf_counter()
                                        candidate = build_alpha_candidate(
                                            base_query=x0,
                                            projection=projection,
                                            exemplar=exemplar,
                                            alpha=alpha,
                                            bundle=bundle,
                                            config=config,
                                            constraints=constraints,
                                            clf=eval_model,
                                            do_repair=use_repair,
                                        )
                                        result = evaluate_candidate(
                                            method=method_name,
                                            alpha=alpha,
                                            base_query=x0,
                                            candidate=candidate,
                                            bundle=bundle,
                                            clf=eval_model,
                                            start_time=start,
                                        )
                                    if can_abstain and result.utility > safe_threshold:
                                        result = evaluate_candidate(
                                            method=method_name,
                                            alpha=alpha,
                                            base_query=x0,
                                            candidate=None,
                                            bundle=bundle,
                                            clf=eval_model,
                                            start_time=start,
                                            abstained=True,
                                            abstain_penalty=safe_threshold,
                                        )

                                    if method_name in {"nn_positive_train", "knn_mean_k5"}:
                                        route_acc = float("nan")
                                    elif result.abstained:
                                        route_acc = float(oracle_utility >= safe_threshold)
                                    else:
                                        route_acc = float(abs(nearest_grid_alpha(alpha) - oracle_alpha) < 1e-9)

                                    query_rows.append(
                                        {
                                            "query_id": query_id,
                                            "dataset": dataset_name,
                                            "model_family": model_family,
                                            "seed": seed,
                                            "depth": depth,
                                            "setting": setting,
                                            "constraint": constraints.name,
                                            "subgroup": subgroup_value,
                                            "method": method_name,
                                            "valid": float(result.valid),
                                            "abstained": float(result.abstained),
                                            "cost_l1": result.cost_l1,
                                            "cost_l2": result.cost_l2,
                                            "sparsity": result.sparsity,
                                            "plausibility": result.plausibility,
                                            "utility": result.utility,
                                            "regret": float(result.utility - oracle_utility),
                                            "route_accuracy": route_acc,
                                            "runtime_sec": result.runtime_sec,
                                            "chosen_alpha": alpha,
                                            "oracle_alpha": oracle_alpha,
                                            "oracle_utility": oracle_utility,
                                            "projection_route": float(abs(alpha - 1.0) < 1e-9),
                                            "escalated_exact": float(method_name == "cascade_exact_auto" and should_escalate_exact),
                                            "accepted_exact": float(method_name == "cascade_exact_auto" and accepted_exact),
                                        }
                                    )

    query_df = pd.DataFrame(query_rows)
    tuning_df = pd.DataFrame(tuning_rows)
    summary_df = aggregate_results(query_df) if not query_df.empty else pd.DataFrame()
    subgroup_summary_df, subgroup_disparity_df = summarize_subgroups(query_df) if not query_df.empty else (pd.DataFrame(), pd.DataFrame())
    overview = {
        "datasets": list(dataset_names),
        "model_families": list(model_families),
        "seeds": [int(x) for x in seeds],
        "depths": [int(x) for x in depths],
        "settings": SETTINGS,
        "constraints": [c.name for c in constraint_settings],
        "alpha_grid": ALPHA_GRID,
        "n_query_rows": int(len(query_df)),
        "n_tuning_rows": int(len(tuning_df)),
        "n_subgroup_rows": int(len(subgroup_summary_df)),
        "n_disparity_rows": int(len(subgroup_disparity_df)),
        "exp_dir": str(exp_dir),
        "fig_dir": str(fig_dir),
    }
    return query_df, summary_df, {
        "overview": overview,
        "tuning": tuning_df,
        "subgroup_summary": subgroup_summary_df,
        "subgroup_disparity": subgroup_disparity_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive recourse routing pilot runner.")
    parser.add_argument("--datasets", nargs="+", default=["adult", "german", "bank"])
    parser.add_argument("--model-families", nargs="+", default=DEFAULT_MODEL_FAMILIES)
    parser.add_argument("--constraints", nargs="+", default=[R1_ALL_MUTABLE.name, R2_REALISTIC.name])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--exp-name", type=str, default=DEFAULT_EXP_NAME)
    parser.add_argument("--enable-exact-cascade", action="store_true")
    parser.add_argument("--include-exact-baseline", action="store_true")
    parser.add_argument("--exact-time-limit-sec", type=float, default=3.0)
    args = parser.parse_args()

    exp_dir = ROOT / "exp" / args.exp_name
    fig_dir = ROOT / "figures" / args.exp_name
    ensure_dirs(exp_dir, fig_dir)
    query_df, summary_df, extras = run_experiment(
        dataset_names=args.datasets,
        model_families=args.model_families,
        constraint_settings=get_constraint_settings(args.constraints),
        seeds=args.seeds,
        depths=args.depths,
        max_rows=int(args.max_rows),
        exp_dir=exp_dir,
        fig_dir=fig_dir,
        enable_exact_cascade=bool(args.enable_exact_cascade),
        include_exact_baseline=bool(args.include_exact_baseline),
        exact_time_limit_sec=float(args.exact_time_limit_sec),
    )

    query_df.to_csv(exp_dir / "query_results.csv", index=False)
    summary_df.to_csv(exp_dir / "summary.csv", index=False)
    extras["tuning"].to_csv(exp_dir / "tuning.csv", index=False)
    extras["subgroup_summary"].to_csv(exp_dir / "subgroup_summary.csv", index=False)
    extras["subgroup_disparity"].to_csv(exp_dir / "subgroup_disparity.csv", index=False)
    (exp_dir / "overview.json").write_text(json.dumps(extras["overview"], indent=2), encoding="utf-8")
    make_figures(summary_df, fig_dir)

    print(json.dumps(extras["overview"], indent=2))


if __name__ == "__main__":
    main()
