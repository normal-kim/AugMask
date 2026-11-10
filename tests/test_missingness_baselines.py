import numpy as np
import pytest

from augmask import MeanModeFill
from augmask.data import build_scenario, split_indices
from augmask.data.missingness import make_masks, mar_covariates

from toy import make_table


def mar_scenario(p=0.5, seed=0):
    df, schema = make_table("binclass", n=3000)
    split = split_indices(len(df), y=df[schema.target].to_numpy(), seed=0)
    return df, schema, split, build_scenario(df, schema, split, "mar", p, seed=seed, augmenter=MeanModeFill())


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_mar_rate_is_calibrated_per_column(p):
    _, _, _, scn = mar_scenario(p)
    np.testing.assert_allclose(1 - scn.masks["train"].mean(0), p, atol=0.04)
    np.testing.assert_allclose(1 - scn.masks["val"].mean(), p, atol=0.04)


def test_mar_depends_on_other_columns_only():
    df, schema, split, scn = mar_scenario(0.5)
    train = df.iloc[split["train"]]
    tables = {s: df.iloc[split[s]] for s in ("train", "val")}
    Z, groups = mar_covariates(tables, schema.num_block, schema.cat_block, schema.columns)
    # every column owns disjoint covariates, and the masks are not MCAR:
    assert sum(len(g) for g in groups) == Z["train"].shape[1]
    a = schema.columns.index("a")
    other = np.delete(np.arange(len(schema.columns)), a)
    # missingness of some column must correlate with a covariate it may depend on
    corrs = [abs(np.corrcoef(~scn.masks["train"][:, j], train["a"].to_numpy())[0, 1]) for j in other]
    assert max(corrs) > 0.1


def test_mar_is_reproducible_and_seed_dependent():
    _, _, _, a = mar_scenario(seed=0)
    _, _, _, b = mar_scenario(seed=0)
    _, _, _, c = mar_scenario(seed=1)
    assert np.array_equal(a.masks["train"], b.masks["train"])
    assert not np.array_equal(a.masks["train"], c.masks["train"])


def test_mcar_requires_no_covariates():
    m = make_masks({"train": 100, "val": 50}, 4, "mcar", 0.3, seed=0)
    assert m["train"].shape == (100, 4) and m["val"].shape == (50, 4)


def test_incomplete_table_matches_masks():
    df, schema, split, scn = mar_scenario(0.5)
    t = scn.incomplete_table(df)
    assert np.array_equal(t.notna().to_numpy(), scn.masks["train"])
    assert t["a"].dtype == np.float64


def test_forestdiff_runs_on_incomplete_data():
    pytest.importorskip("ForestDiffusion")
    from augmask.baselines.forestdiff import ForestDiff
    df, schema, split, scn = mar_scenario(0.3)
    fd = ForestDiff(n_t=3, duplicate_K=2, n_jobs=2).fit(scn.incomplete_table(df), schema)
    out = fd.sample(50)
    assert list(out.columns) == list(schema.columns) and len(out) == 50
    assert out.notna().all().all()
    assert set(out["v"]) <= set(df["v"])


def test_miwae_trains_and_samples():
    pytest.importorskip("torch")
    from augmask.baselines.miwae import MIWAE
    _, _, _, scn = mar_scenario(0.3)
    m = MIWAE(n_epochs=2, latent_size=4, n_hidden=16, K=5, device="cpu")
    tr = scn.encoded["train"]
    m.fit(scn.X_num["train"], scn.X_cat["train"], tr.M_num, tr.M_cat, scn.n_categories)
    X_num, X_cat = m.sample(40)
    out = scn.to_data_space(X_num, X_cat)
    assert len(out) == 40 and out.notna().all().all()


def test_forestdiff_keeps_missing_categoricals_missing():
    pytest.importorskip("ForestDiffusion")
    from augmask.baselines.forestdiff import _model_class
    model = object.__new__(_model_class())
    model.cat_indexes = [1]
    X = np.array([[0.1, 0.0], [0.2, 1.0], [0.3, 2.0], [0.4, np.nan]])
    out, _, after = model.dummify(X)
    assert list(after) == ["0", "1_1.0", "1_2.0"]
    assert np.isnan(out[3, 1:]).all() and not np.isnan(out[:3]).any()
