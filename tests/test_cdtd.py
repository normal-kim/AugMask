"""CDTD backbone: tiny CPU runs and the AugMask loss restriction."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from augmask.models.cdtd import CDTD  # noqa: E402

K = [3, 2]
TINY = dict(dim=4, mlp_emb_dim=8, mlp_n_layers=2, mlp_n_units=16, batch_size=32, num_steps_train=8,
            num_steps_lr_warmup=2, validation_interval=2, generation_steps=3, batch_size_eval=16)


@pytest.fixture(autouse=True)
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)  # tiny ops are much faster single-threaded
    yield
    torch.set_num_threads(n)


def toy(n, seed):
    rng = np.random.default_rng(seed)
    X_num = rng.normal(size=(n, 3)).astype(np.float32)
    X_cat = np.stack([np.arange(n) % k for k in K], 1).astype(np.int64)
    return X_num, rng.permutation(X_cat), rng.random((n, 3)) > 0.3, rng.random((n, 2)) > 0.3


@pytest.mark.parametrize("loss_mask", [False, True])
@pytest.mark.parametrize("with_val", [False, True])
def test_fit_and_sample(loss_mask, with_val):
    m = CDTD(3, K, loss_mask, device="cpu", **TINY).fit(toy(100, 0), toy(40, 1) if with_val else None)
    X_num, X_cat = m.sample(37)
    assert X_num.shape == (37, 3) and X_num.dtype == np.float64 and np.isfinite(X_num).all()
    assert X_cat.shape == (37, 2) and X_cat.dtype == np.int64
    assert ((X_cat >= 0) & (X_cat <= np.array(K))).all()  # code K_j is the extra "unknown" class
    assert np.array_equal(m.sample(37, seed=3)[0], m.sample(37, seed=3)[0])
    if with_val:  # validation is logged, and training runs the full schedule
        assert max(h["step"] for h in m.history_ if "val" in h) == TINY["num_steps_train"]


def test_loss_mask_drops_unobserved_entries():
    """With loss_mask, a batch without observed entries gives zero loss and zero gradients."""
    net = CDTD(3, K, True, device="cpu", **TINY).fit(toy(100, 0)).net_
    X_num, X_cat, _, _ = (torch.as_tensor(a) for a in toy(32, 2))
    unobserved = torch.zeros(32, 3), torch.zeros(32, 2)
    for loss_mask in (True, False):
        net.zero_grad()
        loss = net.loss(X_num, X_cat, *unobserved, loss_mask)["loss"]
        loss.backward()
        grad_max = max(p.grad.abs().max().item() for p in net.parameters())
        if loss_mask:
            assert loss.item() == 0 and grad_max == 0
        else:
            assert loss.item() > 0 and grad_max > 0


def test_fills_are_rotated():
    fit = lambda train: CDTD(3, K, True, device="cpu", **TINY).fit(train, toy(40, 1)).sample(20, seed=3)
    one, listed = fit(toy(100, 0)), fit([toy(100, 0)])
    assert all(np.array_equal(a, b) for a, b in zip(one, listed))  # one fill in a list is the same run
    assert np.isfinite(fit([toy(100, 0), toy(100, 5)])[0]).all()
