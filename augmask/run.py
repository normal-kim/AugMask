"""Train one method on one scenario, sample, and evaluate in data space.

    python -m augmask.run --dataset adult --mechanism mcar --p 0.5 --method cdtd-augmask

Methods: {tabddpm,cdtd,tabdiff}-{noaug,meanmask,augfull,augmask}, forestdiff, miwae.
In augfull/augmask, the missing train inputs get ``--fills`` (default 5) independent fills, used in turn
during training (one per step); ``--fills 1`` is the single fill of the paper.
From Python, ``run(..., augmenter=...)`` swaps LGBM-S for any ``augmask.Augmenter`` in augfull/augmask.
Masks and LGBM-S fills are read from (or written to) <artifacts>/<dataset>/<mechanism>/p<p>/seed<seed>.npz.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import pandas as pd

from .augment import MeanModeFill
from .data import build_scenario, load_artifact, save_artifact, split_indices
from .data.scenario import artifact_fills
from .data.datasets import DATASETS, load_dataset
from .evaluate import evaluate, mle, sample_checks

# strategy -> (fill of the missing inputs, mask the loss to observed entries)
STRATEGIES = {"noaug": ("mean", False), "meanmask": ("mean", True),
              "augfull": ("augment", False), "augmask": ("augment", True)}
BACKBONES = {"tabddpm": "TabDDPM", "cdtd": "CDTD", "tabdiff": "TabDiff"}
BASELINES = ("forestdiff", "miwae")
FILLS = 5  # augmented fills per scenario in augfull/augmask
METHODS = tuple(f"{b}-{s}" for b in BACKBONES for s in STRATEGIES) + BASELINES


def artifact_path(root, dataset, mechanism, p, seed) -> Path:
    return Path(root) / dataset / mechanism / f"p{p}" / f"seed{seed}.npz"


def scenario_for(df, schema, split, dataset, mechanism, p, seed, augmenter, artifacts, n_fills=1):
    """Build a scenario, reusing saved masks/fills when available (and saving them otherwise)."""
    path = artifact_path(artifacts, dataset, mechanism, p, seed)
    art = load_artifact(path) if path.exists() else None
    scn = build_scenario(df, schema, split, mechanism, p, seed, augmenter=augmenter, artifact=art, n_fills=n_fills)
    if scn.default_fills and (art is None or artifact_fills(art) < len(scn.train_draws)):
        path.parent.mkdir(parents=True, exist_ok=True)
        save_artifact(scn, path)
    return scn


def run(dataset, mechanism, p, method, seed=0, artifacts="artifacts", out="runs", device="cuda", augmenter=None,
        fills=FILLS, **config):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    df, schema = load_dataset(dataset)
    y = None if schema.is_regression else df[schema.target].to_numpy()
    split = split_indices(len(df), y=y, seed=0)
    real = df.iloc[split["train"]].reset_index(drop=True)
    test = df.iloc[split["test"]].reset_index(drop=True)
    n = len(real)

    backbone, _, strategy = method.partition("-")
    fill, loss_mask = STRATEGIES.get(strategy, ("mean", False))
    augmenter = MeanModeFill() if fill == "mean" else augmenter  # None: the default LGBM-S
    scn = scenario_for(df, schema, split, dataset, mechanism, p, seed, augmenter, artifacts,
                       n_fills=fills if fill == "augment" else 1)
    train = [(X_num, X_cat, scn.M_num["train"], scn.M_cat["train"]) for X_num, X_cat in scn.train_fills]
    train = train if len(train) > 1 else train[0]
    val = (scn.X_num["val"], scn.X_cat["val"], scn.M_num["val"], scn.M_cat["val"])

    start = time.time()
    if backbone == "forestdiff":
        from .baselines.forestdiff import CHUNK_SIZE, ForestDiff
        gpu = {"gpu_hist": True, "n_jobs": 8} if str(device).startswith("cuda") else {}  # faster than all CPU cores
        model = ForestDiff(chunk_size=CHUNK_SIZE.get(dataset), seed=seed, **{**gpu, **config})
        model.fit(scn.incomplete_table(df), schema)
        syn = model.sample(n)
    elif backbone == "miwae":
        from .baselines.miwae import MIWAE
        model = MIWAE(**config, seed=seed, device=device).fit(*train, scn.n_categories)
        syn = scn.to_data_space(*model.sample(n))
    else:
        cls = getattr(importlib.import_module(f".models.{backbone}", __package__), BACKBONES[backbone])
        model = cls(scn.X_num["train"].shape[1], scn.n_categories, loss_mask=loss_mask, device=device, **config)
        syn = scn.to_data_space(*model.fit(train, val, seed=seed).sample(n, seed=42 + seed))

    seconds = time.time() - start

    run_dir = Path(out) / dataset / mechanism / f"p{p}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    syn.to_csv(run_dir / f"{method}.csv", index=False)
    if getattr(model, "history_", None):
        pd.DataFrame(model.history_).to_csv(run_dir / f"{method}_loss.csv", index=False)
    ref_path = Path(out) / dataset / "trtr.json"
    if not ref_path.exists():  # write then rename, so parallel runs never read a partial file
        tmp = ref_path.with_name(f"trtr.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"mle": mle(real, test, schema)}))
        os.replace(tmp, ref_path)
    metrics = evaluate(real, syn, test, schema, json.loads(ref_path.read_text())["mle"])
    metrics |= {"seconds": round(seconds, 1), "fills": len(scn.train_draws), "checks": sample_checks(real, syn, schema)}
    (run_dir / f"{method}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--mechanism", required=True, choices=("mcar", "mar"))
    ap.add_argument("--p", required=True, type=float)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fills", type=int, default=FILLS, help="augmented fills of the train inputs (augfull/augmask)")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
