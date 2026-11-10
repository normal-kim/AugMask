"""ForestDiff baseline: the official ForestDiffusion package with the paper's settings.

Appendix C.6: n_t = 20 and duplicate_K = 10 (the defaults 50/100 did not finish
within 3600 s); on News the training data is streamed in chunks of 8,000 rows.
Everything else uses the package defaults (flow matching, XGBoost hyperparameters).
The model is trained in data space on the incomplete train table (NaN = missing).

One fix: the package one-hot encodes categoricals with >= 3 levels via
pd.get_dummies, which turns a missing entry into the reference level. We keep
those entries missing (NaN in every dummy), so they are skipped as targets and
treated as missing inputs, as for numerical and binary columns.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data.schema import Schema

CHUNK_SIZE = {"news": 8000}


def _model_class():
    from ForestDiffusion import ForestDiffusionModel

    class _Model(ForestDiffusionModel):
        def dummify(self, X):
            out, before, after = super().dummify(X)
            for i in self.cat_indexes:
                dummies = [k for k, name in enumerate(after) if name.startswith(f"{i}_")]
                out[np.ix_(np.isnan(X[:, i]), dummies)] = np.nan
            return out, before, after

    return _Model


class ForestDiff:
    def __init__(self, n_t: int = 20, duplicate_K: int = 10, chunk_size: int | None = None,
                 n_jobs: int = -1, seed: int = 0, **forest_kwargs):
        self.n_t, self.duplicate_K, self.chunk_size = n_t, duplicate_K, chunk_size
        # one XGBoost thread per model: the package already runs n_jobs models in parallel
        self.n_jobs, self.seed, self.forest_kwargs = n_jobs, seed, {"nthread": 1, **forest_kwargs}

    def fit(self, table: pd.DataFrame, schema: Schema) -> "ForestDiff":
        self.schema = schema
        num, cat = list(schema.num_block), list(schema.cat_block)
        self.vocab_ = [sorted(table[c].dropna().unique().tolist()) for c in cat]
        codes = [table[c].map({v: i for i, v in enumerate(voc)}).to_numpy(np.float64)
                 for c, voc in zip(cat, self.vocab_)]
        X = np.column_stack([table[num].to_numpy(np.float64), *codes]) if codes else table[num].to_numpy(np.float64)
        d_num = len(num)
        K = [len(v) for v in self.vocab_]
        int_idx = [j for j, c in enumerate(num) if (table[c].dropna() % 1 == 0).all()]  # rounded when sampling
        bin_idx = [d_num + k for k, n in enumerate(K) if n <= 2]
        cat_idx = [d_num + k for k, n in enumerate(K) if n > 2]
        n_batch = math.ceil(len(X) / self.chunk_size) if self.chunk_size else 1  # the package default
        self.model_ = _model_class()(
            X, n_t=self.n_t, duplicate_K=self.duplicate_K, bin_indexes=bin_idx, cat_indexes=cat_idx, int_indexes=int_idx,
            n_jobs=self.n_jobs, n_batch=n_batch, seed=self.seed, **self.forest_kwargs,
        )
        return self

    def sample(self, n: int) -> pd.DataFrame:
        np.random.seed(self.seed)
        Xs = self.model_.generate(batch_size=n)
        d_num = len(self.schema.num_block)
        out = pd.DataFrame(Xs[:, :d_num], columns=list(self.schema.num_block))
        for k, c in enumerate(self.schema.cat_block):
            codes = np.clip(np.rint(Xs[:, d_num + k]).astype(np.int64), 0, len(self.vocab_[k]) - 1)
            out[c] = np.asarray(self.vocab_[k], dtype=object)[codes]
        return out[list(self.schema.columns)]
