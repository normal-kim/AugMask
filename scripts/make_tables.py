"""Rank tables (Tables 3 and 7 of the paper) from runs/<dataset>/<mechanism>/p<p>/seed<s>/<method>.json.

Methods are ranked within each scenario (dataset x missing rate x seed) for every
metric (1 = best). Per mechanism we report the mean rank per metric and the
Summary Rank (mean over metrics). Markers compare a method with the better
missing-aware baseline (ForestDiff or MIWAE) on that metric over all scenarios,
using a paired Wilcoxon signed-rank test at 0.05:
  ** significantly better,  * not significantly different.

    python scripts/make_tables.py --runs runs
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from augmask.run import METHODS

METRICS = {"alpha_precision": "α-P", "beta_recall": "β-R", "trend": "Trend", "shape": "Shape", "mle": "Utility"}
BASELINES = ("forestdiff", "miwae")


def load(runs: Path) -> pd.DataFrame:
    rows = []
    for f in runs.glob("*/*/p*/seed*/*.json"):
        if f.stem not in METHODS:
            continue
        dataset, mechanism, p, seed = f.parts[-5:-1]
        scores = json.loads(f.read_text())
        rows += [dict(dataset=dataset, mechanism=mechanism, p=float(p[1:]), seed=int(seed[4:]),
                      method=f.stem, metric=m, score=scores[m]) for m in METRICS if m in scores]
    return pd.DataFrame(rows)


def marker(df: pd.DataFrame, method: str, metric: str, ranks: pd.DataFrame) -> str:
    wide = df[df.metric == metric].pivot_table(index=["dataset", "p", "seed"], columns="method", values="score")
    best = min(BASELINES, key=lambda b: ranks.loc[b, metric])
    if method in BASELINES or method not in wide or best not in wide:
        return ""
    pair = wide[[method, best]].dropna()  # scenarios both methods were run on
    if pair.empty:
        return ""
    a, b = pair[method], pair[best]
    if (a - b).abs().sum() == 0:
        return "*"
    if wilcoxon(a, b, alternative="greater").pvalue < 0.05:
        return "**"
    return "*" if wilcoxon(a, b).pvalue >= 0.05 else ""


def table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = df.groupby(["dataset", "p", "seed", "metric"])["score"].rank(ascending=False)
    ranks = df.pivot_table(index="method", columns="metric", values="rank", aggfunc="mean")[list(METRICS)]
    ranks["Summary"] = ranks.mean(axis=1)
    out = ranks.round(1).astype(str)
    for m in METRICS:
        for method in out.index:
            out.loc[method, m] += marker(df, method, m, ranks)
    return out.loc[ranks["Summary"].sort_values().index].rename(columns=METRICS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", type=Path)
    args = ap.parse_args()
    df = load(args.runs)
    for mechanism, part in df.groupby("mechanism"):
        print(f"\n## {mechanism.upper()} ({part[['dataset', 'p', 'seed']].drop_duplicates().shape[0]} scenarios)\n")
        print(table(part).to_string())


if __name__ == "__main__":
    main()
