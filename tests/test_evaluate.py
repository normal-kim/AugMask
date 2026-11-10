import numpy as np
import pytest

from augmask import evaluate as E

from toy import make_table


@pytest.mark.parametrize("task", ["binclass", "regression"])
def test_alpha_beta_of_identical_tables(task):
    df, schema = make_table(task, n=600)
    ab = E.alpha_beta(df, df, schema)
    assert ab["alpha_precision"] > 0.95 and ab["beta_recall"] > 0.95


@pytest.mark.parametrize("task", ["binclass", "regression"])
def test_shape_trend_of_identical_tables(task):
    pytest.importorskip("sdmetrics")
    df, schema = make_table(task, n=600)
    st = E.shape_trend(df, df, schema)
    assert st["shape"] == pytest.approx(1.0) and st["trend"] == pytest.approx(1.0)


def test_shuffled_columns_score_lower():
    pytest.importorskip("sdmetrics")
    df, schema = make_table("binclass", n=600)
    syn = df.copy()
    syn["a"] = np.random.default_rng(0).permutation(syn["a"].to_numpy())
    assert E.shape_trend(df, syn, schema)["trend"] < E.shape_trend(df, df, schema)["trend"]


@pytest.mark.parametrize("task", ["binclass", "regression"])
def test_mle_and_normalization(task):
    pytest.importorskip("xgboost")
    df, schema = make_table(task, n=900)
    train, test = df.iloc[:600], df.iloc[600:]
    score = E.mle(train, test, schema, device="cpu")
    if task == "binclass":
        assert 0.5 < score <= 1.0
    else:
        assert score > 0
    assert E.normalized_mle(score, score, schema) == pytest.approx(1.0)
