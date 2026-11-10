"""Step 5: fill the missing entries of the (encoded) inputs.

``LGBMSAugmenter`` is the paper's factorized conditional stochastic augmentation
(Algorithm 1, "LGBM-S"); ``MeanModeFill`` is the mean/mode fill of NoAug and MeanMask.
Any object with the ``Augmenter`` interface can be used instead.
Both are fit on the train split and fill train and any further splits (val).
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Protocol, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

# -E[log chi^2_1]: for Gaussian residuals r, E[log r^2] = log sigma^2 - LOG_CHI2_BIAS
LOG_CHI2_BIAS = float(np.euler_gamma + np.log(2))

# Table 4 of the paper; the last four entries only set threading, determinism and logging.
LGBM_PARAMS = dict(
    boosting_type="gbdt",
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.1,
    n_estimators=50,
    subsample_for_bin=200000,
    min_split_gain=0.0,
    min_child_weight=1e-3,
    min_child_samples=20,
    subsample=1.0,
    subsample_freq=0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42,
    importance_type="split",
    n_jobs=1,
    deterministic=True,
    force_row_wise=True,
    verbosity=-1,
)


class Encoded(NamedTuple):
    """One split in encoded space. Masks are True where observed."""

    X_num: np.ndarray  # float, quantile-normal values (NaN where missing)
    X_cat: np.ndarray  # int codes 0..K_j-1 (-1 where missing)
    M_num: np.ndarray  # bool
    M_cat: np.ndarray  # bool


class Augmenter(Protocol):
    """Fills the missing entries: fit on ``train``, then return filled (X_num, X_cat) for ``train``
    followed by each of ``others``. Observed entries stay unchanged; a missing numerical entry gets a
    float and a missing categorical entry a code in 0..K_j-1. With several fills per scenario
    (``n_fills`` > 1), ``augment`` is called once per fill and should return a new random draw each time."""

    def augment(self, train: Encoded, others: Sequence[Encoded], n_categories: Sequence[int]
                ) -> list[tuple[np.ndarray, np.ndarray]]: ...


class MeanModeFill:
    """Train mean (numerical) and train mode (categorical) of the observed entries."""

    def augment(self, train: Encoded, others: Sequence[Encoded], n_categories: Sequence[int]
                ) -> list[tuple[np.ndarray, np.ndarray]]:
        means = np.nan_to_num(np.nanmean(np.where(train.M_num, train.X_num, np.nan), axis=0))
        modes = np.array([np.bincount(train.X_cat[train.M_cat[:, k], k], minlength=max(K, 1)).argmax()
                          for k, K in enumerate(n_categories)], dtype=np.int64)
        return [(np.where(b.M_num, b.X_num, means), np.where(b.M_cat, b.X_cat, modes)) for b in (train, *others)]


def _design(block: Encoded, n_categories: Sequence[int]) -> pd.DataFrame:
    """LightGBM inputs u = (x~, m): values with NaN where missing, plus mask indicators."""
    cols = {}
    for k in range(block.X_num.shape[1]):
        cols[f"n{k}"] = np.where(block.M_num[:, k], block.X_num[:, k], np.nan)
    for k, K in enumerate(n_categories):
        codes = np.where(block.M_cat[:, k], block.X_cat[:, k], -1)
        cols[f"c{k}"] = pd.Categorical.from_codes(codes, categories=np.arange(K))
    for k in range(block.M_num.shape[1]):
        cols[f"mn{k}"] = block.M_num[:, k].astype(np.uint8)
    for k in range(block.M_cat.shape[1]):
        cols[f"mc{k}"] = block.M_cat[:, k].astype(np.uint8)
    return pd.DataFrame(cols)


class LGBMSAugmenter:
    """Factorized conditional stochastic augmentation with per-feature LightGBM models.

    For each feature j, models are fit on the train rows where j is observed, with
    inputs u = (x~_{-j}, m_{-j}). Numerical features are sampled from
    N(mu_j(u), exp(f_j(u) + c)), where f_j regresses log(r^2 + eps) on u and r are the
    residuals of mu_j. The default c = 0 is Algorithm 1; ``log_var_offset=LOG_CHI2_BIAS``
    removes the bias of the log-scale fit (Harvey, 1976), which widens the fills by about
    1.9x without consistently improving the generators. Categorical features are sampled from the predicted class
    probabilities after label smoothing with alpha over the K_j categories.
    Every model conditions on the incomplete data as observed (no chaining).
    Features are fit in parallel on ``n_jobs`` threads (-1: all cores); the fills do not depend on it.
    ``draw`` indexes independent fills from the same models (draw 0 is the default stream).
    """

    def __init__(self, alpha: float = 0.05, eps: float = 1e-6, log_var_offset: float = 0.0,
                 seed: int = 42, n_jobs: int = -1, lgbm_params: dict | None = None, draw: int = 0):
        self.alpha = alpha
        self.eps = eps
        self.log_var_offset = log_var_offset
        self.seed = seed
        self.draw = draw
        self.n_jobs = (os.cpu_count() or 1) if n_jobs == -1 else n_jobs
        self.lgbm_params = {**LGBM_PARAMS, **(lgbm_params or {})}

    def augment(self, train: Encoded, others: Sequence[Encoded], n_categories: Sequence[int]
                ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return filled (X_num, X_cat) for ``train`` followed by each of ``others``."""
        blocks = [train, *others]
        designs = [_design(b, n_categories) for b in blocks]
        d_num = train.X_num.shape[1]
        features = [("num", k) for k in range(d_num)] + [("cat", k) for k in range(len(n_categories))]
        root = np.random.SeedSequence(self.seed if self.draw == 0 else [self.seed, self.draw])
        rngs = [np.random.default_rng(s) for s in root.spawn(len(features))]

        def run(i):
            kind, k = features[i]
            return self._fill_feature(kind, k, blocks, designs, n_categories, rngs[i])

        if self.n_jobs > 1:
            with ThreadPoolExecutor(self.n_jobs) as pool:
                results = list(pool.map(run, range(len(features))))
        else:
            results = [run(i) for i in range(len(features))]

        out = []
        for b_idx, block in enumerate(blocks):
            X_num = np.where(block.M_num, block.X_num, np.nan)
            X_cat = np.where(block.M_cat, block.X_cat, -1)
            for (kind, k), fills in zip(features, results):
                if fills is None:
                    continue
                if kind == "num":
                    X_num[~block.M_num[:, k], k] = fills[b_idx]
                else:
                    X_cat[~block.M_cat[:, k], k] = fills[b_idx]
            out.append((X_num, X_cat))
        return out

    def _fill_feature(self, kind, k, blocks, designs, n_categories, rng):
        num = kind == "num"
        name = f"n{k}" if num else f"c{k}"
        drop = [name, f"m{name}"]
        masks = [b.M_num[:, k] if num else b.M_cat[:, k] for b in blocks]
        if not any((~m).any() for m in masks):
            return None
        obs = masks[0]
        X_fit = designs[0].loc[obs].drop(columns=drop)
        X_miss = [d.loc[~m].drop(columns=drop) for d, m in zip(designs, masks)]

        if num:
            y = blocks[0].X_num[obs, k]
            mu_model = LGBMRegressor(**self.lgbm_params).fit(X_fit, y)
            log_var = np.log((y - mu_model.predict(X_fit)) ** 2 + self.eps)
            var_model = LGBMRegressor(**self.lgbm_params).fit(X_fit, log_var)
            fills = []
            for X in X_miss:
                if len(X) == 0:
                    fills.append(np.empty(0))
                    continue
                mu, sigma = mu_model.predict(X), np.exp(0.5 * (var_model.predict(X) + self.log_var_offset))
                fills.append(mu + sigma * rng.standard_normal(len(X)))
            return fills

        K = n_categories[k]
        y = blocks[0].X_cat[obs, k]
        classes = np.unique(y)
        clf = LGBMClassifier(**self.lgbm_params).fit(X_fit, y) if len(classes) > 1 else None
        fills = []
        for X in X_miss:
            if len(X) == 0:
                fills.append(np.empty(0, dtype=np.int64))
                continue
            probs = np.zeros((len(X), K))
            if clf is None:
                probs[:, classes[0]] = 1.0
            else:
                probs[:, clf.classes_] = clf.predict_proba(X)
            probs = (1 - self.alpha) * probs + self.alpha / K
            u = rng.random(len(X))[:, None]
            fills.append(np.minimum((probs.cumsum(axis=1) < u).sum(axis=1), K - 1))
        return fills
