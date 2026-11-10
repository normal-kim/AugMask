"""Small synthetic tables shared by the tests."""
import numpy as np
import pandas as pd

from augmask.data import Schema


def make_table(task, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    df = pd.DataFrame({
        "a": z + 0.3 * rng.normal(size=n),
        "b": np.exp(z + 0.5 * rng.normal(size=n)),
        "c": np.where(rng.random(n) < 0.6, 0.0, rng.integers(1, 50, n)).astype(float),
        "u": np.where(z > 0, "hi", "lo"),
        "v": rng.choice(["x", "y", "z", "?"], size=n),
    })
    if task == "regression":
        df["t"] = 2 * z + rng.normal(size=n)
        schema = Schema("toy", "regression", tuple(df.columns), ("a", "b", "c"), ("u", "v"), "t")
    else:
        df["t"] = np.where(z + 0.5 * rng.normal(size=n) > 0, "yes", "no")
        schema = Schema("toy", "binclass", tuple(df.columns), ("a", "b", "c"), ("u", "v"), "t")
    return df, schema
