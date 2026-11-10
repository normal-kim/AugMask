"""Step 1: download and clean the six UCI benchmark tables.

Each loader returns the complete table (no NaN) and its Schema. Numerical
columns are float64; categorical columns and a classification target are str.
Native missing categories (Adult's "?") are kept as their own category; rows
with a missing numerical value or target are dropped (Beijing's pm2.5).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import Schema

DATASETS = ("adult", "default", "shoppers", "magic", "beijing", "news")

# archive URL and sha256 of the data file(s) used inside it
SOURCES = {
    "adult": ("https://archive.ics.uci.edu/static/public/2/adult.zip", {
        "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
        "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"}),
    "default": ("https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip", {
        "default of credit card clients.xls": "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"}),
    "shoppers": ("https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip", {
        "online_shoppers_intention.csv": "b3055ee355f59134d851d32641183cb4a8b45def7124d2f50442a042f358e0d9"}),
    "magic": ("https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip", {
        "magic04.data": "e9314b7ebd4b4b59a3b3d65f7316663963777b16a46786877651dbbaa640b36a"}),
    "beijing": ("https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip", {
        "PRSA_data_2010.1.1-2014.12.31.csv": "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c"}),
    "news": ("https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip", {
        "OnlineNewsPopularity/OnlineNewsPopularity.csv":
            "b66d9088632308cc27fa35af847650d174a5a50503987c4e511de94a99d1c218"}),
}


def data_root() -> Path:
    return Path(os.environ.get("AUGMASK_DATA", "datasets"))


def load_dataset(name: str, root: str | Path | None = None, verify: bool = True) -> tuple[pd.DataFrame, Schema]:
    """Complete table and schema of ``name``; downloads and caches on first use."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    root = Path(root) if root is not None else data_root()
    out = root / name
    if (out / "complete.csv").exists() and (out / "schema.json").exists():
        schema = Schema.from_json(out / "schema.json")
        df = pd.read_csv(out / "complete.csv", dtype={c: str for c in schema.cat_block}, keep_default_na=False)
        return _finalize(df, schema), schema
    raw = _download(name, root / "raw", verify)
    df, schema = LOADERS[name](raw)
    df = _finalize(df, schema)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "complete.csv", index=False)
    schema.to_json(out / "schema.json")
    return df, schema


def _finalize(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    df = df[list(schema.columns)].reset_index(drop=True)
    for c in schema.num_block:
        df[c] = df[c].astype(np.float64)
    for c in schema.cat_block:
        df[c] = df[c].astype(str)
    if df.isna().any().any():
        raise ValueError(f"{schema.name}: complete table contains NaN")
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(name: str, raw_root: Path, verify: bool) -> Path:
    url, files = SOURCES[name]
    dest = raw_root / name
    if not all((dest / f).exists() for f in files):
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / "archive.zip"
        with urllib.request.urlopen(url) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f)
        _safe_extract(archive, dest)
        for nested in list(dest.rglob("*.zip")):
            if nested != archive:
                _safe_extract(nested, nested.parent)
    if verify:
        for f, digest in files.items():
            if _sha256(dest / f) != digest:
                raise ValueError(f"{name}: {f} does not match the expected sha256; the source may have changed")
    return dest


def _safe_extract(archive: Path, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            if not (root / member).resolve().is_relative_to(root):
                raise ValueError(f"unsafe path in archive: {member}")
        z.extractall(root)


def _cat_str(s: pd.Series) -> pd.Series:
    """Categories as strings; integer-valued codes without a trailing '.0'."""
    if pd.api.types.is_bool_dtype(s):
        return s.map({True: "True", False: "False"})
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.int64).astype(str)
    return s.astype(str).str.strip()


# ----------------------------------------------------------------------------- loaders
def _adult(raw: Path):
    cols = ["age", "workclass", "fnlwgt", "education", "education.num", "marital.status", "occupation",
            "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week",
            "native.country", "income"]
    read = dict(header=None, names=cols, skipinitialspace=True, keep_default_na=False)
    df = pd.concat([pd.read_csv(raw / "adult.data", **read),
                    pd.read_csv(raw / "adult.test", skiprows=1, **read)], ignore_index=True)
    df["income"] = df["income"].str.rstrip(".")
    num = ("age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week")
    cat = tuple(c for c in cols[:-1] if c not in num)
    for c in cat + ("income",):
        df[c] = _cat_str(df[c])
    return df, Schema("adult", "binclass", tuple(cols), num, cat, "income")


def _default(raw: Path):
    df = pd.read_excel(raw / "default of credit card clients.xls", sheet_name="Data", header=1).drop(columns="ID")
    target = "default payment next month"
    cat = ("SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
    num = tuple(c for c in df.columns if c not in cat + (target,))
    for c in cat + (target,):
        df[c] = _cat_str(df[c])
    return df, Schema("default", "binclass", tuple(df.columns), num, cat, target)


def _shoppers(raw: Path):
    df = pd.read_csv(raw / "online_shoppers_intention.csv")
    cat = ("Month", "OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType", "Weekend")
    num = tuple(df.columns[:10])
    for c in cat + ("Revenue",):
        df[c] = _cat_str(df[c])
    return df, Schema("shoppers", "binclass", tuple(df.columns), num, cat, "Revenue")


def _magic(raw: Path):
    cols = ["Length", "Width", "Size", "Conc", "Conc1", "Asym", "M3Long", "M3Trans", "Alpha", "Dist", "class"]
    df = pd.read_csv(raw / "magic04.data", header=None, names=cols)
    df["class"] = _cat_str(df["class"])
    return df, Schema("magic", "binclass", tuple(cols), tuple(cols[:-1]), (), "class")


def _beijing(raw: Path):
    df = pd.read_csv(raw / "PRSA_data_2010.1.1-2014.12.31.csv").drop(columns="No")
    df = df.dropna(subset=["pm2.5"]).reset_index(drop=True)
    cat = ("year", "month", "day", "hour", "cbwd")
    num = ("DEWP", "TEMP", "PRES", "Iws", "Is", "Ir")
    for c in cat:
        df[c] = _cat_str(df[c])
    return df, Schema("beijing", "regression", tuple(df.columns), num, cat, "pm2.5")


def _news(raw: Path):
    df = pd.read_csv(raw / "OnlineNewsPopularity" / "OnlineNewsPopularity.csv", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    channels = [c for c in df.columns if c.startswith("data_channel_is_")]
    days = [c for c in df.columns if c.startswith("weekday_is_")]
    onehot = df[channels].to_numpy()
    df["data_channel"] = np.where(onehot.sum(1) == 0, "none",
                                  np.array([c.removeprefix("data_channel_is_") for c in channels])[onehot.argmax(1)])
    df["weekday"] = np.array([d.removeprefix("weekday_is_") for d in days])[df[days].to_numpy().argmax(1)]
    df = df.drop(columns=["url", "is_weekend", *channels, *days])
    num = tuple(c for c in df.columns if c not in ("shares", "data_channel", "weekday"))
    cols = num + ("shares", "data_channel", "weekday")
    return df[list(cols)], Schema("news", "regression", cols, num, ("data_channel", "weekday"), "shares")


LOADERS = {"adult": _adult, "default": _default, "shoppers": _shoppers,
           "magic": _magic, "beijing": _beijing, "news": _news}
