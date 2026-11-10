"""Step 3: synthetic missingness. Masks use 1/True = observed.

MCAR: every entry is missing independently with probability p.

MAR (logistic): for every column j, P(x_ij missing) = sigmoid(w_j' z_{i,-j} + b_j), where
z_{-j} encodes the complete values of all other columns (rank-normal numerical,
one-hot categorical; encoders fit on the train split). The weights w_j ~ N(0, I) are
rescaled so the logit has unit standard deviation on train, and b_j is solved so the
mean missing probability on train equals p. The same (w_j, b_j) is used for val.
Since z_{-j} may itself be masked, this is "MAR-like" rather than strictly MAR.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

MECHANISMS = ("mcar", "mar")
MASKED_SPLITS = ("train", "val")  # the test split stays complete


def mcar_mask(shape: tuple[int, int], p: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(shape) >= p


def mar_covariates(tables: dict, num_cols, cat_cols, columns) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Encode complete train/val tables for the MAR propensities.

    Returns the encoded splits and, for each entry of ``columns``, the positions of
    its encoded features (so they can be excluded from its own propensity).
    """
    num_cols, cat_cols = list(num_cols), list(cat_cols)
    parts, owner = {s: [] for s in tables}, []
    if num_cols:
        n = len(tables["train"])
        qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, n),
                                 subsample=int(1e9), random_state=0)
        qt.fit(tables["train"][num_cols].to_numpy(np.float64))
        for s, t in tables.items():
            parts[s].append(qt.transform(t[num_cols].to_numpy(np.float64)))
        owner += num_cols
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(tables["train"][cat_cols].to_numpy(object))
        for s, t in tables.items():
            parts[s].append(ohe.transform(t[cat_cols].to_numpy(object)))
        owner += [c for c, cats in zip(cat_cols, ohe.categories_) for _ in cats]
    Z = {s: np.concatenate(v, axis=1) for s, v in parts.items()}
    owner = np.asarray(owner, dtype=object)
    groups = [np.flatnonzero(owner == c) for c in columns]
    return Z, groups


def _logistic_masks(Z: dict[str, np.ndarray], groups, p: float, rng_w, rngs) -> dict[str, np.ndarray]:
    masks = {s: np.ones((len(z), len(groups)), dtype=bool) for s, z in Z.items()}
    if p == 0:
        return masks
    d = Z["train"].shape[1]
    for j, own in enumerate(groups):
        keep = np.setdiff1d(np.arange(d), own)
        w = rng_w.standard_normal(len(keep))
        logit = {s: z[:, keep] @ w for s, z in Z.items()}
        sd = logit["train"].std()
        if sd > 0:
            logit = {s: v / sd for s, v in logit.items()}
        b = brentq(lambda b: expit(logit["train"] + b).mean() - p, -50.0, 50.0, xtol=1e-12)
        for s in Z:
            masks[s][:, j] = rngs[s].random(len(Z[s])) >= expit(logit[s] + b)
    return masks


def make_masks(
    sizes: dict[str, int],
    n_cols: int,
    mechanism: str,
    p: float,
    seed: int,
    covariates: tuple[dict[str, np.ndarray], list[np.ndarray]] | None = None,
) -> dict[str, np.ndarray]:
    """Observed-masks for the train and val splits over all columns, target included.

    ``covariates`` (from ``mar_covariates``) is required for MAR.
    """
    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}, got {mechanism!r}")
    if not 0.0 <= p < 1.0:
        raise ValueError(f"p must be in [0, 1), got {p}")
    seeds = np.random.SeedSequence(seed).spawn(len(MASKED_SPLITS) + 1)
    rngs = {s: np.random.default_rng(ss) for s, ss in zip(MASKED_SPLITS, seeds)}
    if mechanism == "mcar":
        return {s: mcar_mask((sizes[s], n_cols), p, rngs[s]) for s in MASKED_SPLITS}
    if covariates is None:
        raise ValueError("MAR masks need covariates (see mar_covariates)")
    Z, groups = covariates
    return _logistic_masks(Z, groups, p, np.random.default_rng(seeds[-1]), rngs)
