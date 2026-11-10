"""TabDiff backbone (Shi et al., 2025): continuous-time mixed-type diffusion with an absorbing [MASK] class.

Paper configuration (Appendix C.6). ``loss_mask=False`` trains on every entry of the filled table
(NoAug / AugFull); ``loss_mask=True`` restricts the loss to the observed entries (AugMask).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import as_fills, ema_update, rotating_batches, set_seed, to_tensors

DEFAULTS = dict(
    # UniModMLP denoiser
    d_token=4, dim_t=256, n_layers=5, n_units=801,
    # power-mean (EDM) numerical and log-linear categorical schedules (both fixed), EDM preconditioning;
    # num_timesteps is also the number of sampling steps
    num_timesteps=50, sigma_min=0.002, sigma_max=80, rho=7, eps_max=1e-3, eps_min=1e-5, sigma_data=1.0,
    # training (early stopping is off, so no validation)
    batch_size=4096, train_steps=30000, lr=1e-3, weight_decay=0.0, ema_decay=0.997,
    log_steps=100, lr_factor=0.9, lr_patience=100, min_lr=1e-6,
    sample_batch_size=10000,
)
NEG = -1e6  # log-probability used for impossible classes


class _Net(nn.Module):
    """UniModMLP with EDM preconditioning: per-feature tokens -> MLP with time embedding -> per-feature outputs."""

    def __init__(self, n_num: int, K: list[int], d_token: int, dim_t: int, n_layers: int, n_units: int,
                 sigma_data: float):
        super().__init__()
        K1, n_tok = [k + 1 for k in K], n_num + len(K)
        self.n_num, self.d_token, self.sigma_data = n_num, d_token, sigma_data
        ends = np.cumsum(K1).tolist()
        self.slices = list(zip([0] + ends[:-1], ends))  # one token per feature from its K_j+1 one-hot columns
        # tokenizer (row 0 of `weight` is the [CLS] token, which the MLP never sees)
        self.cat_weight = nn.Parameter(torch.empty(sum(K1), d_token))
        self.weight = nn.Parameter(torch.empty(n_num + 1, d_token))
        self.bias = nn.Parameter(torch.empty(n_tok, d_token))
        for p in (self.cat_weight, self.weight, self.bias):
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))
        self.proj = nn.Linear(d_token * n_tok, dim_t)
        dims = [dim_t] + [n_units] * n_layers
        self.mlp = nn.Sequential(*[m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.ReLU())],
                                 nn.Linear(n_units, d_token * n_tok))
        self.time_fc = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))
        half = dim_t // 2
        self.register_buffer("freqs", torch.exp(-math.log(10000) / half * torch.arange(half, dtype=torch.float32)),
                             persistent=False)
        # detokenizer
        self.out_weight = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.xavier_uniform_(self.out_weight, gain=1 / math.sqrt(2))
        self.out_cat = nn.ModuleList()
        for k in K1:
            self.out_cat.append(nn.Linear(d_token, k))
            nn.init.xavier_uniform_(self.out_cat[-1].weight, gain=1 / math.sqrt(2))

    def forward(self, x_num, x_cat, t, sigma):
        """x_cat: one-hot incl. [MASK] (K_j+1 columns each); t: [b]; sigma: [b, 1]. Returns D(x_num), raw logits."""
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1 / (sd ** 2 + sigma ** 2).sqrt()
        c_noise = (0.002 ** (1 / 7) + t * (80 ** (1 / 7) - 0.002 ** (1 / 7))).pow(7).log() / 4  # fixed rho=7 sigma
        # the [CLS] token is computed and then dropped; it is kept so results are reproducible bit for bit
        h = [self.weight[None] * F.pad(c_in * x_num, (1, 0), value=1.0)[:, :, None]]
        h += [x_cat[:, s:e].unsqueeze(1) @ self.cat_weight[s:e][None] for s, e in self.slices]
        h = (torch.cat(h, 1) + F.pad(self.bias, (0, 0, 1, 0))[None])[:, 1:]
        a = c_noise.flatten()[:, None].float() * self.freqs[None]
        t_emb = self.time_fc(torch.cat([torch.cos(a), torch.sin(a)], -1))
        y = self.mlp(self.proj(h.flatten(1)) + t_emb).view(len(h), -1, self.d_token)
        f_num = (y[:, :self.n_num] * self.out_weight[None]).sum(-1)
        logits = torch.cat([lin(y[:, self.n_num + j]) for j, lin in enumerate(self.out_cat)], -1)
        return c_skip * x_num + c_out * f_num, logits


def _masked_mean(loss, mask):
    """Sum over observed entries / their count (the AugMask normalisation, per feature type)."""
    n = mask.sum()
    return (mask * loss).sum() / (n + 1e-8) if n > 0 else loss.new_zeros(())


class TabDiff:
    """TabDiff on (X_num, X_cat) in model space; ``config`` overrides ``DEFAULTS``."""

    def __init__(self, n_num: int, n_categories: list[int], loss_mask: bool, device="cuda", **config):
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown TabDiff options: {sorted(unknown)}")
        if n_num == 0 or len(n_categories) == 0:
            raise ValueError("TabDiff needs at least one numerical and one categorical feature")
        self.config = {**DEFAULTS, **config}
        self.n_num, self.K, self.loss_mask, self.device = n_num, [int(k) for k in n_categories], loss_mask, device
        self.mask_index = torch.tensor(self.K, device=device)  # class K_j is feature j's [MASK]
        self.net_ = None

    # schedules: power mean (numerical) and log-linear (categorical), t in [0, 1]
    def _sigma_num(self, t):
        rho = torch.tensor(self.config["rho"])
        lo, hi = self.config["sigma_min"] ** (1 / rho), self.config["sigma_max"] ** (1 / rho)
        return (lo + t * (hi - lo)).pow(rho)

    def _t_of_sigma(self, sigma):
        rho = torch.tensor(self.config["rho"])
        lo, hi = self.config["sigma_min"] ** (1 / rho), self.config["sigma_max"] ** (1 / rho)
        return (sigma.pow(1 / rho) - lo) / (hi - lo)

    def _sigma_cat(self, t):
        e_max, e_min = self.config["eps_max"], self.config["eps_min"]
        return -torch.log1p(-((1 - e_max - e_min) * t + e_min))

    def _mask(self, x, p):
        return torch.where(torch.rand(*x.shape, device=x.device) < p, self.mask_index, x)

    def _onehot(self, x):
        """One-hot encoding including each feature's [MASK] column, as in TabDiff."""
        return torch.cat([F.one_hot(x[:, j], k + 1) for j, k in enumerate(self.K)], 1).float()

    def _logits(self, raw, xt):
        """SUBS parameterisation: [b, n_cat, max K+1] log-probs; [MASK] impossible, unmasked entries copied."""
        k_max = max(self.K) + 1
        u = torch.stack([F.pad(r, (0, k_max - r.shape[-1]), value=NEG)
                         for r in raw.split([k + 1 for k in self.K], -1)], -2)
        u[:, range(len(self.K)), self.mask_index] += NEG
        logits = u - torch.logsumexp(u, dim=-1, keepdim=True)
        keep = xt != self.mask_index
        logits[keep] = NEG
        logits[keep, xt[keep]] = 0
        return logits

    def _loss(self, x_num, x_cat, m_num, m_cat):
        t = torch.rand(len(x_num), device=x_num.device, dtype=x_num.dtype)[:, None]
        s_num, s_cat = self._sigma_num(t), self._sigma_cat(t)
        x_num_t = x_num + torch.randn_like(x_num) * s_num
        x_cat_t = self._mask(x_cat, -torch.expm1(-s_cat))
        d, raw = self.net_(x_num_t, self._onehot(x_cat_t), t.squeeze(), s_num)
        sd = self.config["sigma_data"]
        l_num = (s_num ** 2 + sd ** 2) / (s_num * sd) ** 2 * ((d - x_num) ** 2)
        l_cat = -1 / (1 - torch.exp(-s_cat)) * self._logits(raw, x_cat_t).gather(-1, x_cat[:, :, None]).squeeze(-1)
        cat, num = (_masked_mean(l_cat, m_cat), _masked_mean(l_num, m_num)) if self.loss_mask \
            else (l_cat.mean(), l_num.mean())
        return {"loss": cat + num, "cat": cat.detach(), "num": num.detach()}

    def fit(self, train, val=None, seed: int = 0) -> "TabDiff":
        """Train for ``train_steps`` (``val`` is unused: early stopping is off). Afterwards ``self.net_`` holds the EMA
        weights, which are used for sampling. ``train`` may be a list of fills of the same rows, used in turn."""
        c = self.config
        set_seed(seed)
        self.net_ = _Net(self.n_num, self.K, c["d_token"], c["dim_t"], c["n_layers"], c["n_units"],
                        c["sigma_data"]).to(self.device)
        ema = copy.deepcopy(self.net_).requires_grad_(False)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=c["lr_factor"],
                                                           patience=c["lr_patience"], min_lr=c["min_lr"])
        batches = rotating_batches([to_tensors(t, self.device) for t in as_fills(train)], c["batch_size"])
        self.history_ = []
        for step in range(c["train_steps"]):
            opt.zero_grad()
            out = self._loss(*next(batches))
            out["loss"].backward()
            opt.step()
            ema_update(ema.parameters(), self.net_.parameters(), c["ema_decay"])
            if step % c["log_steps"] == 0:
                self.history_.append({"step": step, "lr": opt.param_groups[0]["lr"],
                                      **{k: v.item() for k, v in out.items()}})
                sched.step(out["loss"].item())
        # load the EMA weights into the trained module (matmul kernels depend on requires_grad)
        self.net_.load_state_dict(ema.state_dict())
        return self

    @torch.no_grad()
    def _sample_batch(self, b: int) -> torch.Tensor:
        """Stochastic (S_churn=1) second-order EDM sampler for x_num, MDLM unmasking for x_cat."""
        T = self.config["num_timesteps"]
        t = torch.linspace(0, 1, T, dtype=torch.float32, device=self.device)[:, None]
        s_num, s_cat = self._sigma_num(t), self._sigma_cat(t)
        s_num_next, s_cat_next = torch.zeros_like(s_num), torch.zeros_like(s_cat)
        s_num_next[1:], s_cat_next[1:] = s_num[:-1], s_cat[:-1]
        gamma = min(1 / T, np.sqrt(2) - 1) * (s_num >= 0)
        s_num_hat = s_num + gamma * s_num
        t_hat = self._t_of_sigma(s_num_hat)
        out = (t_hat > 1).squeeze()  # no churn where it would push t past 1
        s_num_hat[out], t_hat[out] = s_num[out], t[out]
        s_cat_hat = self._sigma_cat(t_hat)

        x_num = torch.randn((b, self.n_num), device=self.device) * s_num[-1]
        x_cat = self.mask_index.repeat(b, 1)
        for i in reversed(range(T)):
            x_hat = x_num + (s_num_hat[i] ** 2 - s_num[i] ** 2).sqrt() * torch.randn_like(x_num)
            c_hat = self._mask(x_cat, -torch.expm1(s_cat[i] - s_cat_hat[i]))
            oh = self._onehot(c_hat)
            d, raw = self.net_(x_hat, oh, t_hat[i].expand(b), s_num_hat[i].expand(b, 1))
            d_cur = (x_hat - d) / s_num_hat[i]
            x_num = x_hat + (s_num_next[i] - s_num_hat[i]) * d_cur
            move_t, move_s = 1 - torch.exp(-s_cat_hat[i]), 1 - torch.exp(-s_cat_next[i])
            q = self._logits(raw, c_hat).exp() * (move_t - move_s)
            q[:, range(len(self.K)), self.mask_index] = move_s
            draw = (q / (1e-10 - (torch.rand_like(q) + 1e-10).log())).argmax(-1)
            x_cat = torch.where(c_hat != self.mask_index, c_hat, draw)
            if i > 0:
                d, _ = self.net_(x_num, oh, t[i - 1].expand(b), s_num_next[i].expand(b, 1))
                d_prime = (x_num - d) / s_num_next[i]
                x_num = x_hat + (s_num_next[i] - s_num_hat[i]) * (0.5 * d_cur + 0.5 * d_prime)
        assert torch.all(x_cat < self.mask_index)
        return torch.cat([x_num, x_cat], 1).cpu()

    @torch.no_grad()
    def sample(self, n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """``n`` rows in model space. A row containing NaN is multiplied by 0 (codes become 0, NaN entries stay NaN)."""
        set_seed(seed)
        bs = min(self.config["sample_batch_size"], n)
        out = []
        for _ in range(-(-n // bs)):
            s = self._sample_batch(bs)
            out.append(s * ~s.isnan().any(1, keepdim=True))
        s = torch.cat(out)[:n].numpy()
        return s[:, :self.n_num].astype(np.float64), s[:, self.n_num:].astype(np.int64)
