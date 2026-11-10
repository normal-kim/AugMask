import numpy as np
import pytest

pytest.importorskip("torch")
from augmask.models.tabdiff import TabDiff  # noqa: E402

K = [3, 2, 4]
TINY = dict(d_token=4, dim_t=16, n_layers=2, n_units=32, batch_size=32, train_steps=5, num_timesteps=4,
            sample_batch_size=25)


def toy(n=64, p_obs=0.7, seed=0):
    rng = np.random.default_rng(seed)
    X_num = rng.standard_normal((n, 3)).astype(np.float32)
    X_cat = np.stack([rng.integers(0, k, n) for k in K], 1)
    return X_num, X_cat, rng.random((n, 3)) < p_obs, rng.random((n, len(K))) < p_obs


def fit_sample(data, loss_mask, n=60):
    return TabDiff(3, K, loss_mask, device="cpu", **TINY).fit(data, seed=0).sample(n, seed=42)


@pytest.mark.parametrize("loss_mask", [False, True])
def test_fit_and_sample_shapes(loss_mask):
    X_num, X_cat = fit_sample(toy(), loss_mask)
    assert X_num.shape == (60, 3) and X_num.dtype == np.float64 and np.isfinite(X_num).all()
    assert X_cat.shape == (60, 3) and X_cat.dtype == np.int64
    assert ((X_cat >= 0) & (X_cat < np.array(K))).all()


def test_loss_mask_with_everything_observed_equals_unmasked_loss():
    X_num, X_cat, M_num, M_cat = toy()
    full = fit_sample((X_num, X_cat, np.ones_like(M_num), np.ones_like(M_cat)), True)
    plain = fit_sample(toy(), False)  # the unmasked loss ignores the masks
    assert all(np.array_equal(a, b) for a, b in zip(full, plain))
    masked = fit_sample(toy(), True)
    assert not np.array_equal(masked[0], plain[0])


def test_last_feature_with_fewer_categories_than_features():
    # e.g. Beijing: 5 categorical features, the last one with 4 categories
    K2, rng = [5, 12, 31, 24, 4], np.random.default_rng(1)
    X_cat = np.stack([rng.integers(0, k, 64) for k in K2], 1)
    X_num = rng.standard_normal((64, 2)).astype(np.float32)
    data = (X_num, X_cat, rng.random((64, 2)) < 0.7, rng.random((64, 5)) < 0.7)
    X_num, X_cat = TabDiff(2, K2, True, device="cpu", **TINY).fit(data).sample(30)
    assert X_num.shape == (30, 2) and X_cat.shape == (30, 5) and ((X_cat >= 0) & (X_cat < np.array(K2))).all()
