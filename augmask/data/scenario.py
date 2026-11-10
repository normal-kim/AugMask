"""Steps 3-6 for one (dataset, mechanism, p, seed) scenario, and the inverse to data space.

A scenario holds the train/val splits in model space:
numerical block -> quantile-normal (step 4) -> filled (step 5) -> standardized (step 6);
categorical block -> integer codes (step 4) -> filled (step 5).
The test split is never masked and stays in data space for evaluation.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..augment import Augmenter, Encoded, LGBMSAugmenter
from .missingness import MASKED_SPLITS, make_masks, mar_covariates
from .schema import Schema
from .transforms import CatVocab, NumQuantile, NumStandardizer

ARTIFACT_VERSION = 1  # bump when the LGBM-S fills change


@dataclass
class Scenario:
    schema: Schema
    mechanism: str
    p: float
    seed: int
    fill: str  # name of the augmenter
    default_fills: bool  # filled by the default LGBM-S (these fills are saved in artifacts)
    split: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]  # synthetic masks over schema.columns (train/val)
    num_tf: NumQuantile
    cat_vocab: CatVocab
    scaler: NumStandardizer
    encoded: dict[str, Encoded]  # step-4 space with NaN / -1 where missing
    filled: dict[str, tuple[np.ndarray, np.ndarray]]  # step-4 space after step 5
    X_num: dict[str, np.ndarray]  # model space, float32
    X_cat: dict[str, np.ndarray]  # model space, int64
    cat_modes: np.ndarray  # mode of the filled train codes (for out-of-range samples)
    num_integer: np.ndarray  # numerical columns whose observed train values are all integers

    @property
    def M_num(self) -> dict[str, np.ndarray]:
        return {s: e.M_num for s, e in self.encoded.items()}

    @property
    def M_cat(self) -> dict[str, np.ndarray]:
        return {s: e.M_cat for s, e in self.encoded.items()}

    @property
    def n_categories(self) -> list[int]:
        return self.cat_vocab.n_categories

    def incomplete_table(self, df: pd.DataFrame, split: str = "train") -> pd.DataFrame:
        """Data-space rows of ``split`` with NaN where the synthetic mask hides an entry."""
        rows = df.iloc[self.split[split]].reset_index(drop=True)[list(self.schema.columns)]
        return rows.astype(object).where(self.masks[split], np.nan).astype(
            {c: np.float64 for c in self.schema.num_block})

    def to_data_space(self, X_num: np.ndarray, X_cat: np.ndarray) -> pd.DataFrame:
        """Map model-space samples back to a data-space table in ``schema.columns`` order."""
        num = self.num_tf.inverse_transform(self.scaler.inverse_transform(np.asarray(X_num, np.float64)))
        num[:, self.num_integer] = np.round(num[:, self.num_integer])
        codes = np.rint(X_cat).astype(np.int64)
        codes[codes < 0] = 0
        K = np.asarray(self.n_categories)
        too_high = codes >= K
        codes[too_high] = np.broadcast_to(self.cat_modes, codes.shape)[too_high]
        cat = self.cat_vocab.inverse_transform(codes)
        cols = {**dict(zip(self.schema.num_block, num.T)), **dict(zip(self.schema.cat_block, cat.T))}
        return pd.DataFrame(cols)[list(self.schema.columns)]


def _encode(df, schema, split, masks):
    num_pos, cat_pos = schema.block_positions()
    raw = {}
    for s in MASKED_SPLITS:
        rows = df.iloc[split[s]]
        Mn, Mc = masks[s][:, num_pos], masks[s][:, cat_pos]
        Xn = np.where(Mn, rows[list(schema.num_block)].to_numpy(np.float64), np.nan)
        Xc = rows[list(schema.cat_block)].to_numpy(object)
        raw[s] = (Xn, Xc, Mn, Mc)
    num_tf = NumQuantile().fit(raw["train"][0])
    cat_vocab = CatVocab().fit(raw["train"][1], raw["train"][3])
    encoded = {}
    for s, (Xn, Xc, Mn, Mc) in raw.items():
        codes, Mc_eff = cat_vocab.transform(Xc, Mc)
        encoded[s] = Encoded(num_tf.transform(Xn), codes, Mn, Mc_eff)
    Xn, Mn = raw["train"][0], raw["train"][2]
    num_integer = np.array([np.all(Xn[Mn[:, j], j] % 1 == 0) for j in range(Xn.shape[1])], dtype=bool)
    return num_tf, cat_vocab, encoded, num_integer


def build_scenario(
    df: pd.DataFrame,
    schema: Schema,
    split: dict[str, np.ndarray],
    mechanism: str,
    p: float,
    seed: int,
    augmenter: Augmenter | None = None,
    artifact: dict | None = None,
) -> Scenario:
    """Build a scenario. ``augmenter`` fills the missing inputs; the default is LGBM-S
    (``LGBMSAugmenter(seed=42 + seed)``). With ``artifact`` (see ``load_artifact``) the masks, and
    the fills of the default augmenter, are taken from a saved artifact instead of being regenerated."""
    default = augmenter is None
    augmenter = augmenter or LGBMSAugmenter(seed=42 + seed)
    if artifact is not None:
        masks = {s: artifact[f"mask_{s}"] for s in MASKED_SPLITS}
    else:
        covariates = None
        if mechanism == "mar":
            tables = {s: df.iloc[split[s]] for s in MASKED_SPLITS}
            covariates = mar_covariates(tables, schema.num_block, schema.cat_block, schema.columns)
        masks = make_masks({s: len(split[s]) for s in MASKED_SPLITS}, len(schema.columns),
                           mechanism, p, seed, covariates)

    num_tf, cat_vocab, encoded, num_integer = _encode(df, schema, split, masks)
    K = cat_vocab.n_categories
    if artifact is not None:
        _check_stats(artifact, num_tf, cat_vocab)

    if default and artifact is not None:
        if artifact["meta"].get("version") != ARTIFACT_VERSION:
            raise ValueError("the artifact's fills come from an older LGBM-S version; delete it to regenerate")
        filled = {s: _apply_fills(encoded[s], artifact, s) for s in MASKED_SPLITS}
    else:
        filled = dict(zip(MASKED_SPLITS, augmenter.augment(encoded["train"], [encoded["val"]], K)))

    scaler = NumStandardizer().fit(encoded["train"].X_num, encoded["train"].M_num)
    X_num = {s: scaler.transform(filled[s][0]).astype(np.float32) for s in MASKED_SPLITS}
    X_cat = {s: filled[s][1].astype(np.int64) for s in MASKED_SPLITS}
    cat_modes = np.array(
        [np.bincount(X_cat["train"][:, k], minlength=max(n, 1)).argmax() for k, n in enumerate(K)],
        dtype=np.int64,
    )
    return Scenario(schema, mechanism, p, seed, type(augmenter).__name__, default, split, masks, num_tf, cat_vocab, scaler,
                    encoded, filled, X_num, X_cat, cat_modes, num_integer)


# ----------------------------------------------------------------------------- artifacts
def save_artifact(scn: Scenario, path) -> None:
    """Save masks, encoder stats and, for an LGBM-S scenario, the fills
    (step-4 space, missing positions only, row-major order)."""
    payload = {
        "meta": np.array(json.dumps({
            "version": ARTIFACT_VERSION, "dataset": scn.schema.name, "mechanism": scn.mechanism, "p": scn.p,
            "seed": scn.seed,
            "n_rows": {s: int(len(scn.split[s])) for s in scn.split},
        })),
        "vocab": np.array(json.dumps(scn.cat_vocab.vocab_)),
        "quantiles": scn.num_tf.qt_.quantiles_,
    }
    for s in MASKED_SPLITS:
        e = scn.encoded[s]
        payload[f"mask_{s}"] = scn.masks[s]
        if scn.default_fills:
            payload[f"fill_num_{s}"] = scn.filled[s][0][~e.M_num]
            payload[f"fill_cat_{s}"] = scn.filled[s][1][~e.M_cat].astype(np.int32)
    tmp = f"{path}.{os.getpid()}.tmp"  # write then rename, so parallel runs never read a partial file
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


def load_artifact(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        art = {k: z[k] for k in z.files}
    art["meta"] = json.loads(str(art["meta"]))
    art["vocab"] = json.loads(str(art["vocab"]))
    return art


def _check_stats(art, num_tf, cat_vocab):
    if art["vocab"] != cat_vocab.vocab_ or not np.allclose(art["quantiles"], num_tf.qt_.quantiles_, equal_nan=True):
        raise ValueError("artifact does not match the rebuilt encoders; check the dataset version")


def _apply_fills(e: Encoded, art: dict, split: str) -> tuple[np.ndarray, np.ndarray]:
    X_num, X_cat = e.X_num.copy(), e.X_cat.copy()
    X_num[~e.M_num] = art[f"fill_num_{split}"]
    X_cat[~e.M_cat] = art[f"fill_cat_{split}"]
    return X_num, X_cat
