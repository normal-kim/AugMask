"""Steps 4 and 6: encoders fit on the observed entries of the train split."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import QuantileTransformer


def n_quantiles_for(n_rows: int) -> int:
    return max(min(n_rows // 30, 1000), 10)


class NumQuantile:
    """Quantile transform to a standard normal (CDTD/TabDDPM setting).

    NaN entries (missing) are ignored when fitting and kept as NaN when transforming.
    """

    def fit(self, X: np.ndarray) -> "NumQuantile":
        self.qt_ = QuantileTransformer(
            output_distribution="normal",
            n_quantiles=n_quantiles_for(X.shape[0]),
            subsample=int(1e9),
            random_state=42,
        ).fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.qt_.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return self.qt_.inverse_transform(Z)


class CatVocab:
    """Per-column sorted vocabulary of the observed train categories.

    Codes are 0..K_j-1; missing entries get code -1. A category not in the
    vocabulary (e.g. seen only in val) is treated as missing.
    """

    def fit(self, X: np.ndarray, M: np.ndarray) -> "CatVocab":
        self.vocab_ = [sorted(set(X[M[:, j], j].tolist())) for j in range(X.shape[1])]
        return self

    @property
    def n_categories(self) -> list[int]:
        return [len(v) for v in self.vocab_]

    def transform(self, X: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (codes, mask) where the mask also drops out-of-vocabulary entries."""
        codes = np.full(X.shape, -1, dtype=np.int64)
        for j, vocab in enumerate(self.vocab_):
            lookup = {v: i for i, v in enumerate(vocab)}
            codes[:, j] = [lookup.get(v, -1) if m else -1 for v, m in zip(X[:, j], M[:, j])]
        return codes, codes >= 0

    def inverse_transform(self, codes: np.ndarray) -> np.ndarray:
        out = np.empty(codes.shape, dtype=object)
        for j, vocab in enumerate(self.vocab_):
            out[:, j] = np.asarray(vocab, dtype=object)[codes[:, j]]
        return out


class NumStandardizer:
    """Standard scaling with mean/std (ddof=0) of the observed entries only."""

    def fit(self, X: np.ndarray, M: np.ndarray) -> "NumStandardizer":
        Xo = np.where(M, X, np.nan)
        self.mean_ = np.nanmean(Xo, axis=0)
        std = np.nanstd(Xo, axis=0)
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std_ + self.mean_
