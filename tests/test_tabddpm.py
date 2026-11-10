import numpy as np
import pytest

torch = pytest.importorskip("torch")
from augmask.models.common import to_tensors  # noqa: E402
from augmask.models.tabddpm import TabDDPM  # noqa: E402

TINY = dict(n_layers_hidden=1, n_units_hidden=16, dim_emb=8, num_timesteps=10, batch_size=32, train_steps=3,
            sample_batch_size=16)
N_NUM, K = 3, [2, 4, 3]


def toy(n=96, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, N_NUM)).astype(np.float32), np.stack([rng.integers(0, k, n) for k in K], 1),
            rng.random((n, N_NUM)) < 0.6, rng.random((n, len(K))) < 0.6)


@pytest.mark.parametrize("loss_mask", [False, True])
def test_fit_and_sample(loss_mask):
    model = TabDDPM(N_NUM, K, loss_mask, device="cpu", **TINY).fit(toy(), toy(seed=1), seed=0)
    x_num, x_cat = model.sample(40)
    assert x_num.shape == (40, N_NUM) and x_num.dtype == np.float64 and np.isfinite(x_num).all()
    assert x_cat.shape == (40, len(K)) and x_cat.dtype == np.int64
    assert ((x_cat >= 0) & (x_cat <= np.array(K))).all()  # code K_j is the spare class
    again = TabDDPM(N_NUM, K, loss_mask, device="cpu", **TINY).fit(toy(), seed=0).sample(40)
    assert np.array_equal(x_num, again[0]) and np.array_equal(x_cat, again[1])


def test_unknown_config_key_is_rejected():
    with pytest.raises(TypeError):
        TabDDPM(N_NUM, K, True, device="cpu", scheduler="linear")


def test_loss_mask_ignores_outputs_at_unobserved_entries():
    model = TabDDPM(N_NUM, K, True, device="cpu", **{**TINY, "train_steps": 0}).fit(toy())
    x_num, x_cat, m_num, m_cat = to_tensors(toy(), "cpu")
    # Perturb the denoiser output at unobserved numerical entries and, where the last categorical
    # feature is unobserved, its logits. (Only the last one: the per-feature normaliser is a
    # cumulative logsumexp, so an earlier feature would move later ones at rounding level.)
    delta = torch.zeros(len(x_num), N_NUM + sum(model.K))
    delta[:, :N_NUM] = 1 - m_num
    delta[:, -model.K[-1]:] = 1 - m_cat[:, -1:]
    delta *= 3 * torch.randn(delta.shape, generator=torch.Generator().manual_seed(1))
    net = model.net_

    def loss_and_grads(shift, mask):
        model.net_ = lambda x, t: net(x, t) + shift * delta
        net.zero_grad()
        torch.manual_seed(0)
        loss = model._loss(x_num, x_cat, m_num if mask else None, m_cat if mask else None)["loss"]
        loss.backward()
        return [loss] + [p.grad.clone() for p in net.parameters()]

    a, b = loss_and_grads(0.0, True), loss_and_grads(1.0, True)
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert loss_and_grads(0.0, False)[0] != loss_and_grads(1.0, False)[0]


def test_fills_are_rotated():
    one = TabDDPM(N_NUM, K, True, device="cpu", **TINY).fit(toy(), seed=0).sample(20, seed=1)
    listed = TabDDPM(N_NUM, K, True, device="cpu", **TINY).fit([toy()], seed=0).sample(20, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(one, listed))  # one fill in a list is the same run
    plain = TabDDPM(N_NUM, K, True, device="cpu", **TINY).fit(list(toy()), seed=0).sample(20, seed=1)
    assert all(np.array_equal(a, b) for a, b in zip(one, plain))  # so is one fill given as a list of arrays
    two = TabDDPM(N_NUM, K, True, device="cpu", **TINY).fit([toy(), toy(seed=5)], seed=0).sample(20, seed=1)
    assert np.isfinite(two[0]).all()
