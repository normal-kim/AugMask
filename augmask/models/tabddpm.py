"""TabDDPM backbone (Kotelnikov et al., 2023) with an optional AugMask loss mask.

Gaussian diffusion (eps-prediction, MSE) on the numerical block and multinomial
diffusion (x0-parametrisation, stochastic variational bound) on the categorical
block, one MLP denoiser and a cosine schedule (synthcity's TabDDPM). ``loss_mask=False``
trains on every entry of the filled table (NoAug / AugFull); ``loss_mask=True``
averages each sample's Gaussian and multinomial terms over its observed entries
only (AugMask).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import as_fills, ema_update, rotating_batches, set_seed, to_tensors

DEFAULTS = {  # paper configuration (Appendix C.6)
    "n_layers_hidden": 3, "n_units_hidden": 256, "dim_emb": 128, "num_timesteps": 1000,
    "batch_size": 4096, "train_steps": 30000, "lr": 2e-4, "weight_decay": 0.0, "ema_decay": 0.999,
    "sample_batch_size": 2000, "log_steps": 100,
}


def _log_add_exp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.max(a, b)
    return m + torch.log(torch.exp(a - m) + torch.exp(b - m))


def _sliced_logsumexp(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Per-feature logsumexp, broadcast back to every class (via a cumulative logsumexp)."""
    lse = torch.logcumsumexp(F.pad(x, [1, 0], value=-float("inf")), dim=-1)
    a, b = lse[:, offsets[1:]], lse[:, offsets[:-1]]
    m = torch.maximum(a, b)
    d = (torch.exp(a - m) - torch.exp(b - m)).clamp(min=1e-10)
    out = torch.where(a > b, torch.log(d) + m, torch.full_like(m, -float("inf")))
    return out.repeat_interleave(offsets[1:] - offsets[:-1], dim=-1)


def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (m * x).sum(1) / (m.sum(1) + 1e-8)


def _schedule(T: int, device) -> dict[str, torch.Tensor]:
    """Cosine schedule (Nichol & Dhariwal, 2021) and the coefficients used below."""
    f = lambda s: math.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
    alphas = torch.tensor(1.0 - np.array([min(1 - f((i + 1) / T) / f(i / T), 0.999) for i in range(T)]))
    betas = 1.0 - alphas
    log_a, ab = alphas.log(), alphas.cumprod(0)
    log_ab, ab_prev = log_a.cumsum(0), torch.cat([torch.ones(1, dtype=ab.dtype), ab[:-1]])
    log_1m = lambda a: torch.log(1 - a.exp() + 1e-40)
    # reverse-process std: posterior variance at t = 1, beta_t (in float32) afterwards; kept in float64
    var = torch.cat([(betas * (1 - ab_prev) / (1 - ab))[1:2], (1.0 - alphas.float())[1:].double()])
    s = {"sqrt_ab": ab.sqrt(), "sqrt_1m_ab": (1 - ab).sqrt(), "sqrt_recip_ab": (1 / ab).sqrt(),
         "sqrt_recipm1_ab": (1 / ab - 1).sqrt(), "coef1": betas * ab_prev.sqrt() / (1 - ab),
         "coef2": (1 - ab_prev) * alphas.sqrt() / (1 - ab), "log_a": log_a, "log_1m_a": log_1m(log_a),
         "log_ab": log_ab, "log_1m_ab": log_1m(log_ab)}
    s = {k: v.float().to(device) for k, v in s.items()}
    s["sigma"] = torch.exp(0.5 * var.to(device).log())
    return s


class _MLP(nn.Module):
    """Denoiser: ReLU MLP on proj(x) + embedding(t)."""

    def __init__(self, d: int, d_emb: int, n_layers: int, n_units: int):
        super().__init__()
        half = d_emb // 2
        freqs = torch.exp(-math.log(10000) / half * torch.arange(half, dtype=torch.float32))
        self.register_buffer("freqs", freqs, persistent=False)
        self.time_fc = nn.Sequential(nn.Linear(d_emb, d_emb), nn.SiLU(), nn.Linear(d_emb, d_emb))
        dims = [d_emb] + n_layers * [n_units]
        self.fc = nn.Sequential(*[m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.ReLU())],
                                nn.Linear(n_units, d))
        self.proj = nn.Linear(d, d_emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.freqs
        return self.fc(self.proj(x) + self.time_fc(torch.cat([args.cos(), args.sin()], 1)))


class TabDDPM:
    """Fit on a filled table in model space (masks: True = observed); sample from the EMA denoiser."""

    def __init__(self, n_num: int, n_categories: list[int], loss_mask: bool, device="cuda", **config):
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown config keys: {sorted(unknown)}")
        self.config = {**DEFAULTS, **config}
        self.n_num, self.loss_mask, self.device = n_num, loss_mask, torch.device(device)
        # One extra class per feature (index K_j), never present in the data; it is part of the
        # uniform noise and the network outputs, and sampled extra codes are mapped to the most
        # frequent category in data space.
        self.K = [k + 1 for k in n_categories]
        self.T = self.config["num_timesteps"]
        vars(self).update(_schedule(self.T, self.device))
        self.log_K = torch.tensor(np.repeat(self.K, self.K), dtype=torch.float32, device=self.device).log()
        self.offsets = torch.tensor(np.cumsum([0] + self.K), device=self.device)

    # ---- multinomial diffusion (log one-hot space) ----
    def _log_onehot(self, x_cat: torch.Tensor) -> torch.Tensor:
        oh = torch.cat([F.one_hot(x_cat[:, j], k) for j, k in enumerate(self.K)], 1)
        return oh.float().clamp(min=1e-30).log()

    def _sample_onehot(self, logits: torch.Tensor) -> torch.Tensor:
        """Gumbel-max draw for each feature, returned in log one-hot form."""
        idx = []
        for s in logits.split(self.K, 1):
            s = s.clamp(max=50)
            idx.append((-torch.log(-torch.log(torch.rand_like(s) + 1e-30) + 1e-30) + s).argmax(1))
        return self._log_onehot(torch.stack(idx, 1))

    def _q_pred(self, log_x0, t):  # log q(x_t | x_0)
        return _log_add_exp(log_x0 + self.log_ab[t, None], self.log_1m_ab[t, None] - self.log_K)

    def _q_posterior(self, log_x0, log_xt, t):  # log q(x_{t-1} | x_t, x_0)
        prev = torch.where((t == 0)[:, None], log_x0, self._q_pred(log_x0, (t - 1).clamp(min=0)))
        un = prev + _log_add_exp(log_xt + self.log_a[t, None], self.log_1m_a[t, None] - self.log_K)
        return un - _sliced_logsumexp(un, self.offsets)

    def _p_pred(self, out_cat, log_xt, t):  # log p(x_{t-1} | x_t)
        log_x0 = torch.cat([F.log_softmax(s, 1) for s in out_cat.split(self.K, 1)], 1)
        return self._q_posterior(log_x0, log_xt, t)

    def _cat_reduce(self, x, m):
        """Sum per-class terms over all features, or average them over the observed ones."""
        if m is None:
            return x.sum(1)
        return _masked_mean(torch.stack([s.sum(1) for s in x.split(self.K, 1)], 1), m)

    # ---- training ----
    def _loss(self, x_num, x_cat, m_num=None, m_cat=None) -> torch.Tensor:
        t = torch.randint(0, self.T, (len(x_num),), device=x_num.device)
        pt = torch.ones_like(t).float() / self.T
        noise = torch.randn_like(x_num)
        x_num_t = self.sqrt_ab[t, None] * x_num + self.sqrt_1m_ab[t, None] * noise
        log_x0 = self._log_onehot(x_cat)
        log_xt = self._sample_onehot(self._q_pred(log_x0, t))
        out = self.net_(torch.cat([x_num_t, log_xt], 1), t)

        mse = (noise - out[:, :self.n_num]) ** 2
        loss_gauss = mse.mean(1) if m_num is None else _masked_mean(mse, m_num)

        log_true = self._q_posterior(log_x0, log_xt, t)
        log_model = self._p_pred(out[:, self.n_num:], log_xt, t)
        kl = self._cat_reduce(log_true.exp() * (log_true - log_model), m_cat)
        nll = self._cat_reduce(-log_x0.exp() * log_model, m_cat)  # decoder term, used at t = 0
        t0 = (t == 0).float()
        log_qT = self._q_pred(log_x0, torch.full_like(t, self.T - 1))
        prior = self._cat_reduce(log_qT.exp() * (log_qT + self.log_K), m_cat)
        loss_multi = (t0 * nll + (1.0 - t0) * kl) / pt + prior
        if m_cat is None:  # per-class terms were summed over features: average them
            loss_multi = loss_multi / len(self.K)
        return {"loss": loss_multi.mean() + loss_gauss.mean(), "cat": loss_multi.mean().detach(),
                "num": loss_gauss.mean().detach()}

    def fit(self, train, val=None, seed: int = 0) -> "TabDDPM":
        """Train for ``train_steps`` steps (AdamW, linear lr decay, EMA); ``val`` is unused.
        ``train`` may be a list of fills of the same rows, used in turn (one per step)."""
        c = self.config
        set_seed(seed)
        fills = [to_tensors(t, self.device) for t in as_fills(train)]
        if not self.loss_mask:
            for f in fills:
                f[2] = f[3] = None
        self.net_ = _MLP(self.n_num + sum(self.K), c["dim_emb"], c["n_layers_hidden"], c["n_units_hidden"])
        self.net_.to(self.device)
        self.ema_ = copy.deepcopy(self.net_).requires_grad_(False)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        batches = rotating_batches(fills, c["batch_size"])
        self.history_ = []
        for step in range(c["train_steps"]):
            opt.zero_grad()
            out = self._loss(*next(batches))
            out["loss"].backward()
            opt.step()
            ema_update(self.ema_.parameters(), self.net_.parameters(), c["ema_decay"])
            if step % c["log_steps"] == 0:
                self.history_.append({"step": step, "lr": opt.param_groups[0]["lr"],
                                      **{k: v.item() for k, v in out.items()}})
            for g in opt.param_groups:
                g["lr"] = c["lr"] * (1 - step / c["train_steps"])
        return self

    # ---- sampling ----
    @torch.no_grad()
    def _sample_batch(self, b: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(b, self.n_num, device=self.device)
        log_z = self._sample_onehot(torch.zeros(b, sum(self.K), device=self.device))
        for i in reversed(range(self.T)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            out = self.ema_(torch.cat([z, log_z], 1).float(), t)
            x0 = self.sqrt_recip_ab[t, None] * z - self.sqrt_recipm1_ab[t, None] * out[:, :self.n_num]
            mean = self.coef1[t, None] * x0 + self.coef2[t, None] * z
            # sigma is float64, so z is carried in float64 after the first step
            z = mean + (t != 0).float()[:, None] * self.sigma[t, None] * torch.randn_like(z)
            log_z = self._sample_onehot(self._p_pred(out[:, self.n_num:], log_z, t))
        if torch.isnan(z).any():
            raise ValueError("found NaNs in sample")
        return z.cpu(), torch.stack([s.argmax(1) for s in log_z.split(self.K, 1)], 1).cpu()

    def sample(self, n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """Ancestral sampling over all T steps with the EMA denoiser, in batches of ``sample_batch_size``."""
        set_seed(seed)
        bs = min(self.config["sample_batch_size"], n)
        parts = [self._sample_batch(int(b)) for b in np.diff([*range(0, n, bs), n])]
        return tuple(torch.cat(p).numpy() for p in zip(*parts))
