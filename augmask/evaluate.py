"""Evaluation in data space, following TabSyn's metric code.

- Shape / Trend: sdmetrics QualityReport (Column Shapes / Column Pair Trends).
- alpha-Precision / beta-Recall: the "naive" AlphaPrecision of synthcity on
  [raw numerical | one-hot categorical], min-max scaled on the real data.
- MLE: XGBoost grid search trained on the synthetic table, selected on a 1/9
  hold-out of it, scored on the real test split (AUROC or RMSE). Report it
  normalized by the same score of a model trained on the real train split.

Details that follow TabSyn's implementation exactly, rather than an idealised version, are marked "TabSyn:".
One departure: the validation targets used to pick the regression model are log-clipped like the train and
test targets (TabSyn compares them unclipped). XGBoost runs on CPU with fixed threads, so scores are identical
across machines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .data.schema import Schema

XGB_GRID = {"n_estimators": [10, 50, 100], "min_child_weight": [1, 10], "max_depth": [5, 10, 20], "gamma": [0.0, 1.0]}


def shape_trend(real: pd.DataFrame, syn: pd.DataFrame, schema: Schema) -> dict[str, float]:
    from sdmetrics.reports.single_table import QualityReport

    cols = list(schema.num_block) + list(schema.cat_block)
    meta = {"columns": {c: {"sdtype": "numerical", "computer_representation": "Float"} for c in schema.num_block}
            | {c: {"sdtype": "categorical"} for c in schema.cat_block}}
    report = QualityReport()
    report.generate(real[cols], syn[cols], meta, verbose=False)
    scores = report.get_properties().set_index("Property")["Score"]
    return {"shape": float(scores["Column Shapes"]), "trend": float(scores["Column Pair Trends"])}


def _tabsyn_order(schema: Schema) -> tuple[list[str], list[str]]:
    """TabSyn appends the target to the end of its type block."""
    num, cat = list(schema.num_cols), list(schema.cat_cols)
    (num if schema.is_regression else cat).append(schema.target)
    return num, cat


def alpha_beta(real: pd.DataFrame, syn: pd.DataFrame, schema: Schema) -> dict[str, float]:
    num, cat = _tabsyn_order(schema)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(real[cat].astype(str))
    enc = lambda df: np.hstack([df[num].to_numpy(np.float64), ohe.transform(df[cat].astype(str))])
    X, Y = enc(real), np.nan_to_num(enc(syn))
    if len(X) != len(Y):
        n = min(len(X), len(Y))
        X = pd.DataFrame(X).sample(n=n, random_state=42).to_numpy()
        Y = pd.DataFrame(Y).sample(n=n, random_state=42).to_numpy()
    # TabSyn: synthcity's data loader treats the last column as the target and drops it.
    X, Y = X[:, :-1], Y[:, :-1]
    scaler = MinMaxScaler().fit(X)
    X, Y = scaler.transform(X), scaler.transform(Y)

    alphas = np.linspace(0, 1, 30)
    center = X.mean(0)
    radii = np.quantile(np.linalg.norm(X - center, axis=1), alphas)
    syn_to_center = np.linalg.norm(Y - center, axis=1)
    real_to_real = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X)[0][:, 1]
    real_to_syn, idx = NearestNeighbors(n_neighbors=1).fit(Y).kneighbors(X)
    real_to_syn, idx = real_to_syn[:, 0], idx[:, 0]
    closest_d = np.linalg.norm(Y[idx] - Y.mean(0), axis=1)
    closest_radii = np.quantile(closest_d, alphas)
    precision = np.array([(syn_to_center <= r).mean() for r in radii])
    coverage = np.array([((real_to_syn <= real_to_real) & (closest_d <= r)).mean() for r in closest_radii])
    delta = lambda curve: 1 - np.abs(alphas - curve).sum() / alphas.sum()
    return {"alpha_precision": float(delta(precision)), "beta_recall": float(delta(coverage))}


def _features(df: pd.DataFrame, schema: Schema, fitted=None):
    """TabSyn's feat_transform: scaled numericals, one-hot categoricals, encoded target."""
    enc, labels, cmin, cmax = fitted or ({}, None, None, None)
    feats = []
    for c in schema.columns:
        col = df[c].to_numpy()
        if c == schema.target:
            if schema.is_regression:
                y = col.astype(np.float32)
            else:
                labels = labels if labels is not None else np.unique(col.astype(str))
                y = np.searchsorted(labels, col.astype(str))
        elif c in schema.num_cols:
            col = col.astype(np.float32)
            # TabSyn: one (cmin, cmax) pair is carried over across columns and from train to test.
            cmin = cmin if cmin else col.min()
            cmax = cmax if cmax else col.max()
            feats.append(np.log(np.maximum(col, 1e-2)) if cmin >= 0 and cmax >= 1e3
                         else (col - cmin) / (cmax - cmin) * 5)
        else:
            if c not in enc:
                enc[c] = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(col.astype(str).reshape(-1, 1))
            feats.append(enc[c].transform(col.astype(str).reshape(-1, 1)))
    X = np.column_stack([f if f.ndim == 2 else f[:, None] for f in feats])
    return X, y, (enc, labels, cmin, cmax)


def mle(train: pd.DataFrame, test: pd.DataFrame, schema: Schema, seed: int = 0, device: str = "cpu",
        n_jobs: int = 4) -> float:
    """AUROC (classification) or RMSE on log(clip(y, 1, 2e4)) (regression) on the test split."""
    from xgboost import XGBClassifier, XGBRegressor

    X, y, fitted = _features(train, schema)
    X_test, y_test, _ = _features(test, schema, fitted)
    perm = np.arange(len(X))
    np.random.RandomState(seed).shuffle(perm)
    n_val = len(X) // 9
    X_val, y_val, X, y = X[perm[:n_val]], y[perm[:n_val]], X[perm[n_val:]], y[perm[n_val:]]
    common = {"n_jobs": n_jobs, "tree_method": "hist", "device": device}

    if schema.is_regression:
        log_clip = lambda v: np.log(np.clip(v, 1, 20000))  # TabSyn's target transform
        y, y_val, y_test = log_clip(y), log_clip(y_val), log_clip(y_test)
        rmse = lambda m, A, b: mean_squared_error(b, m.predict(A)) ** 0.5
        fits = [XGBRegressor(objective="reg:squarederror", **p, **common).fit(X, y) for p in ParameterGrid(XGB_GRID)]
        best = min(fits, key=lambda m: rmse(m, X_val, y_val))
        return float(rmse(best, X_test, y_test))

    if len(np.unique(y)) < 2:
        return 0.5
    auc = lambda m, A, b: roc_auc_score(b, m.predict_proba(A)[:, 1]) if len(np.unique(b)) > 1 else 0.5
    fits = [XGBClassifier(objective="binary:logistic", **p, **common).fit(X, y) for p in ParameterGrid(XGB_GRID)]
    best = max(fits, key=lambda m: auc(m, X_val, y_val))
    return float(auc(best, X_test, y_test))


def sample_checks(real: pd.DataFrame, syn: pd.DataFrame, schema: Schema) -> dict:
    """Sanity checks of a synthetic table against the real train split."""
    cat = list(schema.cat_block)
    num = list(schema.num_block)
    unseen = {c: int((~syn[c].astype(str).isin(set(real[c].astype(str)))).sum()) for c in cat}
    lo, hi = real[num].min(), real[num].max()
    outside = {c: int(((syn[c] < lo[c]) | (syn[c] > hi[c])).sum()) for c in num}
    tvd = {c: float(0.5 * (real[c].astype(str).value_counts(normalize=True)
                           .sub(syn[c].astype(str).value_counts(normalize=True), fill_value=0).abs().sum()))
           for c in cat}
    return {
        "rows": len(syn), "rows_expected": len(real),
        "columns_match": list(syn.columns) == list(real.columns),
        "nan": int(syn.isna().sum().sum()),
        "unseen_categories": sum(unseen.values()),
        "outside_train_range": sum(outside.values()),
        "max_category_tvd": max(tvd.values(), default=0.0),
    }


def normalized_mle(score: float, reference: float, schema: Schema) -> float:
    """MLE relative to train-on-real (higher is better for both tasks)."""
    return reference / score if schema.is_regression else score / reference


def evaluate(real_train: pd.DataFrame, syn: pd.DataFrame, test: pd.DataFrame, schema: Schema,
             mle_reference: float | None = None, seed: int = 0, device: str = "cpu") -> dict[str, float]:
    out = {**shape_trend(real_train, syn, schema), **alpha_beta(real_train, syn, schema)}
    out["mle_raw"] = mle(syn, test, schema, seed, device)
    if mle_reference is not None:
        out["mle"] = normalized_mle(out["mle_raw"], mle_reference, schema)
    return out
