import numpy as np
import pytest

from augmask import LGBMSAugmenter, MeanModeFill
from augmask.data import build_scenario, load_artifact, save_artifact, split_indices

from toy import make_table


def build(task, p=0.5, augmenter=None, **kw):
    df, schema = make_table(task)
    y = None if schema.is_regression else df[schema.target].to_numpy()
    split = split_indices(len(df), y=y, seed=0)
    return df, schema, split, build_scenario(df, schema, split, "mcar", p, seed=0, augmenter=augmenter, **kw)


def test_split_is_disjoint_and_sized():
    df, schema = make_table("binclass")
    split = split_indices(len(df), y=df["t"].to_numpy(), seed=0)
    allidx = np.concatenate(list(split.values()))
    assert len(np.unique(allidx)) == len(df)
    assert [len(split[s]) for s in ("train", "val", "test")] == [900, 300, 300]


@pytest.mark.parametrize("task", ["binclass", "regression"])
def test_blocks_and_masks(task):
    df, schema, split, scn = build(task)
    assert scn.schema.num_block[0] == ("t" if task == "regression" else "a")
    assert scn.schema.cat_block[0] == ("u" if task == "regression" else "t")
    for s in ("train", "val"):
        assert scn.masks[s].shape == (len(split[s]), len(schema.columns))
        assert abs(scn.masks[s].mean() - 0.5) < 0.05
        assert np.isfinite(scn.X_num[s]).all()
        assert (scn.X_cat[s] >= 0).all()
        assert (scn.X_cat[s] < np.array(scn.n_categories)).all()
    assert "test" not in scn.masks


def test_observed_entries_are_untouched_and_standardized():
    df, schema, split, scn = build("binclass")
    e = scn.encoded["train"]
    filled_num, filled_cat = scn.filled["train"]
    assert np.array_equal(filled_num[e.M_num], e.X_num[e.M_num])
    assert np.array_equal(filled_cat[e.M_cat], e.X_cat[e.M_cat])
    Xo = np.where(e.M_num, scn.X_num["train"], np.nan)
    np.testing.assert_allclose(np.nanmean(Xo, 0), 0, atol=1e-5)
    np.testing.assert_allclose(np.nanstd(Xo, 0), 1, atol=1e-5)


def test_mean_fill_is_zero_in_model_space():
    df, schema, split, scn = build("binclass", augmenter=MeanModeFill())
    e = scn.encoded["train"]
    np.testing.assert_allclose(scn.X_num["train"][~e.M_num], 0, atol=1e-5)


def test_unknown_val_category_becomes_missing():
    df, schema = make_table("binclass")
    y = df["t"].to_numpy()
    split = split_indices(len(df), y=y, seed=0)
    df.loc[df.index[split["val"][:5]], "v"] = "only_in_val"
    scn = build_scenario(df, schema, split, "mcar", 0.0, seed=0)
    v = schema.cat_block.index("v")
    assert "only_in_val" not in scn.cat_vocab.vocab_[v]
    assert (~scn.M_cat["val"][:5, v]).all() and scn.M_cat["train"].all()


def test_lgbm_s_is_deterministic_and_thread_independent():
    _, _, _, a = build("regression", augmenter=LGBMSAugmenter(n_jobs=1))
    _, _, _, b = build("regression", augmenter=LGBMSAugmenter(n_jobs=4))
    for s in ("train", "val"):
        assert np.array_equal(a.X_num[s], b.X_num[s]) and np.array_equal(a.X_cat[s], b.X_cat[s])


def test_lgbm_s_is_conditional_and_stochastic():
    # 'u' is a deterministic function of the latent z, so its fills must track 'a'.
    _, schema, _, scn = build("regression", p=0.3)
    e = scn.encoded["train"]
    u = schema.cat_block.index("u")
    a = schema.num_block.index("a")
    miss = ~e.M_cat[:, u] & e.M_num[:, a]
    hi = scn.cat_vocab.vocab_[u].index("hi")
    agree = (scn.X_cat["train"][miss, u] == hi) == (e.X_num[miss, a] > 0)
    assert agree.mean() > 0.8
    fills = scn.filled["train"][0][~e.M_num[:, a], a]
    assert np.std(fills) > 0.1


def test_artifact_roundtrip(tmp_path):
    df, schema, split, scn = build("binclass")
    save_artifact(scn, tmp_path / "s.npz")
    art = load_artifact(tmp_path / "s.npz")
    again = build_scenario(df, schema, split, "mcar", 0.5, seed=0, artifact=art)
    for s in ("train", "val"):
        assert np.array_equal(scn.masks[s], again.masks[s])
        assert np.array_equal(scn.X_num[s], again.X_num[s])
        assert np.array_equal(scn.X_cat[s], again.X_cat[s])


def test_to_data_space_recovers_observed_values():
    df, schema, split, scn = build("binclass", p=0.3)
    out = scn.to_data_space(scn.X_num["train"], scn.X_cat["train"])
    truth = df.iloc[split["train"]].reset_index(drop=True)
    assert list(out.columns) == list(schema.columns)
    M = scn.masks["train"]
    for j, c in enumerate(schema.columns):
        obs = M[:, j]
        if c in schema.cat_block:
            assert (out.loc[obs, c].to_numpy() == truth.loc[obs, c].to_numpy()).all()
        else:
            np.testing.assert_allclose(out.loc[obs, c], truth.loc[obs, c], rtol=1e-3, atol=1e-3)


def test_augmenter_sampling_matches_smoothing():
    # A constant input cannot be split on, so the classifier predicts the observed
    # marginal and the fills must follow (1 - alpha) * marginal + alpha / K.
    rng = np.random.default_rng(1)
    n = 20000
    from augmask import Encoded
    X_cat = rng.choice(3, size=(n, 1), p=[0.7, 0.2, 0.1])
    M_cat = rng.random((n, 1)) > 0.5
    block = Encoded(np.zeros((n, 1)), np.where(M_cat, X_cat, -1), np.ones((n, 1), bool), M_cat)
    (_, filled), = LGBMSAugmenter().augment(block, [], [3])
    marginal = np.bincount(X_cat[M_cat[:, 0], 0], minlength=3) / M_cat.sum()
    freq = np.bincount(filled[~M_cat[:, 0], 0], minlength=3) / (~M_cat).sum()
    np.testing.assert_allclose(freq, 0.95 * marginal + 0.05 / 3, atol=0.015)


def test_integer_columns_are_rounded_in_data_space():
    df, schema, split, scn = build("binclass", p=0.3)
    c = schema.num_block.index("c")  # integer-valued toy column
    assert scn.num_integer[c] and not scn.num_integer[schema.num_block.index("a")]
    out = scn.to_data_space(scn.X_num["train"] + 0.1, scn.X_cat["train"])
    assert (out["c"] % 1 == 0).all() and not (out["a"] % 1 == 0).all()


def test_lgbm_s_variance_matches_gaussian_residuals():
    # x1 = x0 + 0.5 * noise: fills of x1 minus x0 have std close to 0.5 with the
    # log-variance correction, and about 0.53x smaller with Algorithm 1 as written (default).
    from augmask import Encoded
    from augmask.augment import LOG_CHI2_BIAS
    rng = np.random.default_rng(0)
    n = 20000
    x0 = rng.normal(size=n)
    X_num = np.column_stack([x0, x0 + 0.5 * rng.normal(size=n)])
    M_num = np.column_stack([np.ones(n, bool), rng.random(n) < 0.5])
    block = Encoded(np.where(M_num, X_num, np.nan), np.zeros((n, 0), np.int64), M_num, np.zeros((n, 0), bool))
    spread = {}
    for name, kw in {"corrected": {"log_var_offset": LOG_CHI2_BIAS}, "default": {}}.items():
        (filled, _), = LGBMSAugmenter(**kw).augment(block, [], [])
        spread[name] = np.std((filled[:, 1] - x0)[~M_num[:, 1]])
    assert 0.42 < spread["corrected"] < 0.58
    assert spread["default"] < 0.8 * spread["corrected"]


def test_custom_augmenter(tmp_path):
    class ZeroFill:  # any object with augment(train, others, n_categories)
        def augment(self, train, others, n_categories):
            return [(np.where(b.M_num, b.X_num, 0.0), np.where(b.M_cat, b.X_cat, 0)) for b in (train, *others)]

    df, schema, split, scn = build("binclass", augmenter=ZeroFill())
    e = scn.encoded["train"]
    assert scn.fill == "ZeroFill" and not scn.default_fills
    assert (scn.filled["train"][0][~e.M_num] == 0).all() and (scn.filled["train"][1][~e.M_cat] == 0).all()
    save_artifact(scn, tmp_path / "s.npz")
    assert "fill_num_train" not in load_artifact(tmp_path / "s.npz")  # only the default LGBM-S fills are cached
