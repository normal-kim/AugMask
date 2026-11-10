"""CDTD backbone (Mueller et al., 2023) with an optional AugMask loss mask.

Paper configuration (Appendix C.6): TabDDPM-style MLP score network, losses
calibrated by the category entropies and reweighted by a learned function of
the noise level, a learnable logistic noise schedule per feature, and a
deterministic Euler sampler with score interpolation for the categoricals.

``loss_mask=False`` trains on every entry of the filled data (AugFull);
``loss_mask=True`` restricts every loss term to the observed entries (AugMask).
Inputs and outputs live in the scenario's model space (see augmask.data.scenario).
"""
from __future__ import annotations

import math
from itertools import islice

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .common import eval_batches, set_seed, to_tensors, train_batches

DEFAULTS = {
    # score network
    "dim": 128,  # categorical embedding size
    "mlp_emb_dim": 256,
    "mlp_n_layers": 5,
    "mlp_n_units": 796,
    "cat_emb_init_sigma": 0.001,
    # noise levels and schedule
    "sigma_min_cat": 0.0,
    "sigma_max_cat": 100.0,
    "sigma_min_cont": 0.0,
    "sigma_max_cont": 80.0,
    "sigma_data_cat": 1.0,
    "sigma_data_cont": 1.0,
    "timewarp_weight_low_noise": 3.0,
    # optimisation
    "num_steps_train": 30_000,
    "num_steps_lr_warmup": 1000,
    "batch_size": 4096,
    "lr": 1e-3,
    "betas": (0.9, 0.99),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "ema_decay": 0.999,
    # validation loss, logged every 300 steps (monitoring only: the final EMA weights are kept)
    "validation_interval": 300,
    "val_batches": 3,
    "log_steps": 100,
    # sampling
    "generation_steps": 200,
    "batch_size_eval": 4096,
}


class _Timewarp(nn.Module):
    """Noise schedule: one learnable logistic CDF per feature, u = F(sigma) up to a scale."""

    def __init__(self, sigma_min: torch.Tensor, sigma_max: torch.Tensor, weight_low_noise: float):
        super().__init__()
        n = len(sigma_min)
        v = torch.tensor(1.01)
        p_large_noise = torch.tensor(1 / (weight_low_noise + 1))
        self.logits_v = nn.Parameter(torch.full((n,), torch.log(torch.exp(v - 1) - 1)))
        self.logits_mu = nn.Parameter(torch.full((n,), torch.log(1 / (1 - p_large_noise) - 1) / v))
        self.logits_gamma = nn.Parameter((torch.ones(n, 1).exp() - 1).log())
        self.register_buffer("sigma_min", sigma_min)
        self.register_buffer("sigma_max", sigma_max)

    def _params(self):
        return self.logits_mu[:, None], (1 + F.softplus(self.logits_v))[:, None]

    def sigma(self, u: torch.Tensor) -> torch.Tensor:
        """Inverse CDF: u [B] -> sigma [B, d]."""
        mu, v = self._params()
        out = torch.sigmoid(mu + 1 / v * torch.special.logit(u.expand(len(mu), -1), eps=1e-7)).T
        out = out.masked_fill((u == 0)[:, None], 0.0).masked_fill((u == 1)[:, None], 1.0)
        return out * (self.sigma_max - self.sigma_min) + self.sigma_min

    def loss(self, sigma: torch.Tensor, losses: torch.Tensor) -> torch.Tensor:
        """Fit the scaled CDF to the per-entry losses, weighted by the inverse density."""
        mu, v = self._params()
        x = ((sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)).clamp(1e-7, 1 - 1e-7).T
        z = ((x / (1 - x)) / mu.exp()) ** (-v)
        cdf = (1 / (1 + z) * F.softplus(self.logits_gamma)).T
        pdf = ((v / (x * (1 - x))) * (z / ((1 + z) ** 2))).T.detach()
        return ((cdf - losses) ** 2) / (pdf + 1e-7)


class _ScoreNet(nn.Module):
    """TabDDPM-style MLP: noisy inputs + time embedding -> category logits and numerical predictions."""

    def __init__(self, n_num: int, n_classes: list[int], dim: int, emb_dim: int, n_layers: int,
                 n_units: int, bias_init: torch.Tensor):
        super().__init__()
        half = emb_dim // 2
        freqs = torch.exp(-math.log(10000) / half * torch.arange(half, dtype=torch.float32))
        self.register_buffer("freqs", freqs, persistent=False)
        self.time_emb = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        dims = [emb_dim] + n_layers * [n_units]
        self.fc = nn.Sequential(*[m for i in range(n_layers) for m in (nn.Linear(dims[i], dims[i + 1]), nn.ReLU())])
        self.proj = nn.Linear(n_num + len(n_classes) * dim, emb_dim)
        self.out = nn.Linear(n_units, n_num + sum(n_classes))
        nn.init.zeros_(self.out.weight)
        self.out.bias.data.copy_(bias_init)
        self.split = [n_num, *n_classes]

    def forward(self, x_cat_emb, x_num, t):
        args = t[:, None].float() * self.freqs[None]
        cond = self.time_emb(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))
        h = self.fc(self.proj(torch.cat((x_cat_emb.flatten(1), x_num), dim=-1)) + cond)
        out = torch.split(self.out(h), self.split, dim=-1)
        return out[1:], out[0]


class _Diffusion(nn.Module):
    """Mixed-type score-based diffusion (EDM preconditioning, embedded categoricals)."""

    def __init__(self, n_num: int, n_categories: list[int], proportions: list[torch.Tensor], cfg: dict):
        super().__init__()
        n_classes = [k + 1 for k in n_categories]  # one extra class per feature, never present in the data
        self.n_cat, self.n_num, self.dim = len(n_categories), n_num, cfg["dim"]
        bias_init = torch.cat((torch.zeros(n_num),
                               torch.cat([torch.cat([p, torch.tensor([1e-8])]) for p in proportions]).log()))
        self.model = _ScoreNet(n_num, n_classes, self.dim, cfg["mlp_emb_dim"], cfg["mlp_n_layers"],
                               cfg["mlp_n_units"], bias_init)
        self.cat_emb = nn.Embedding(sum(n_classes), self.dim)
        nn.init.normal_(self.cat_emb.weight, std=cfg["cat_emb_init_sigma"])
        self.cat_bias = nn.Parameter(torch.zeros(self.n_cat, self.dim))
        self.register_buffer("offsets", torch.tensor([0, *np.cumsum(n_classes)[:-1].tolist()]), persistent=False)
        self.n_classes, self.emb_scale = n_classes, torch.tensor(self.dim).sqrt().item()
        entropy = torch.tensor([-torch.sum(p * p.log()) for p in proportions])
        self.register_buffer("normal_const", torch.cat((entropy, torch.ones(n_num))))
        # adaptive weighting: log of the expected loss, from Fourier features of log u
        self.register_buffer("fourier", torch.randn(512))
        self.weight_net = nn.Linear(1024, 1)
        nn.init.zeros_(self.weight_net.weight)
        nn.init.zeros_(self.weight_net.bias)
        self.register_buffer("sigma_data_cat", torch.tensor(cfg["sigma_data_cat"]))
        self.register_buffer("sigma_data_cont", torch.tensor(cfg["sigma_data_cont"]))
        bounds = [torch.tensor([cfg[f"sigma_{b}_cat"]] * self.n_cat + [cfg[f"sigma_{b}_cont"]] * n_num)
                  for b in ("min", "max")]
        self.timewarp = _Timewarp(*bounds, cfg["timewarp_weight_low_noise"])

    def embed(self, x_cat):
        return F.normalize(self.cat_emb(x_cat + self.offsets) + self.cat_bias, dim=2, eps=1e-20) * self.emb_scale

    def feature_emb(self, j: int):
        """Embeddings of all classes of categorical feature j."""
        off = int(self.offsets[j])
        emb = self.cat_emb.weight[off:off + self.n_classes[j]] + self.cat_bias[j]
        return F.normalize(emb, dim=1, eps=1e-20) * self.emb_scale

    def denoise(self, x_cat_t, x_num_t, u, sigma):
        """Preconditioned network: category logits and the denoised numericals."""
        s_cat, s_num, sd = sigma[:, :self.n_cat], sigma[:, self.n_cat:], self.sigma_data_cont
        c_in_cat = 1 / (self.sigma_data_cat**2 + s_cat.unsqueeze(2) ** 2 + 1e-8).sqrt()
        c_in_num = 1 / (sd**2 + s_num**2 + 1e-8).sqrt()
        logits, pred = self.model(c_in_cat * x_cat_t, c_in_num * x_num_t, torch.log(u + 1e-8) * 0.25 * 1000)
        c_skip = sd**2 / (s_num**2 + sd**2)
        c_out = s_num * sd / (s_num**2 + sd**2).sqrt()
        return logits, c_skip * x_num_t + c_out * pred

    def loss(self, x_num, x_cat, m_num, m_cat, loss_mask: bool, u=None):
        """Training loss and its parts; "obs" is the mean calibrated loss over the observed entries.

        With ``loss_mask`` every per-entry term is zeroed at unobserved entries; the
        means still run over all B x d entries."""
        B = x_num.shape[0]
        x_cat_emb = self.embed(x_cat)
        with torch.no_grad():
            if u is None:  # low-discrepancy sampler
                u = (torch.rand((1,), device=x_num.device, dtype=torch.float64)
                     + torch.arange(B, device=x_num.device, dtype=torch.float64) / B) % 1
            sigma = self.timewarp.sigma(u).detach().to(torch.float32)
            u = u.to(torch.float32)
        s_cat, s_num = sigma[:, :self.n_cat], sigma[:, self.n_cat:]
        x_cat_t = x_cat_emb + torch.randn_like(x_cat_emb) * s_cat.unsqueeze(2)
        x_num_t = x_num + torch.randn_like(x_num) * s_num
        mask = torch.cat([m_cat, m_num], 1)
        keep = mask if loss_mask else torch.ones_like(mask)

        logits, pred = self.denoise(x_cat_t, x_num_t, u, sigma)
        ce = torch.stack([F.cross_entropy(lg, x_cat[:, j], reduction="none") for j, lg in enumerate(logits)], dim=1)
        ce, mse = keep[:, :self.n_cat] * ce, keep[:, self.n_cat:] * (pred - x_num) ** 2
        sd = self.sigma_data_cont
        num_weight = (s_num**2 + sd**2) / ((s_num * sd) ** 2 + 1e-7)
        calibrated = torch.cat((ce, mse), dim=1) / self.normal_const
        weighted = torch.cat((ce, num_weight * mse), dim=1) / self.normal_const

        f = (torch.log(u + 1e-8) * 0.25)[:, None] * self.fourier * 2 * math.pi
        log_w = self.weight_net(torch.cat((f.sin(), f.cos()), dim=-1))
        timewarp_loss = self.timewarp.loss(sigma.detach(), calibrated.detach())
        weight_loss = (log_w.exp() - weighted.detach()) ** 2
        diffusion = weighted / log_w.exp().detach()
        timewarp_term, weight_term = (keep * timewarp_loss).mean(), (keep * weight_loss).mean()
        with torch.no_grad():
            parts = {"cat": diffusion[:, :self.n_cat].sum() / diffusion.numel(),
                     "num": diffusion[:, self.n_cat:].sum() / diffusion.numel(),
                     "timewarp": timewarp_term, "weightnet": weight_term,
                     "obs": (mask * weighted).sum() / mask.sum()}
        return {"loss": diffusion.mean() + timewarp_term + weight_term, **parts}

    @torch.inference_mode()
    def sample(self, n: int, steps: int):
        nc, dev = self.n_cat, self.cat_bias.device
        x_cat = torch.randn((n, nc, self.dim), device=dev).to(torch.float64)
        x_num = torch.randn((n, self.n_num), device=dev).to(torch.float64)
        u_steps = torch.linspace(1, 0, steps + 1, device=dev, dtype=torch.float64)
        t_steps = self.timewarp.sigma(u_steps)
        x_cat, x_num = x_cat * t_steps[0, :nc].unsqueeze(1), x_num * t_steps[0, nc:]

        def step(x_cat, x_num, u, t):
            logits, x_num_0 = self.denoise(x_cat.float(), x_num.float(), u.float().repeat((n,)), t.float())
            probs = [F.softmax(lg.to(torch.float64), dim=1) for lg in logits]
            x_cat_0 = torch.stack([p @ self.feature_emb(j).to(torch.float64) for j, p in enumerate(probs)], dim=1)
            return probs, x_cat_0, x_num_0.to(torch.float64)

        for t_cur, t_next, u in zip(t_steps[:-1], t_steps[1:], u_steps[:-1]):
            t_cur, t_next = t_cur.repeat((n, 1)), t_next.repeat((n, 1))
            _, x_cat_0, x_num_0 = step(x_cat, x_num, u, t_cur)
            h = t_next - t_cur
            x_cat = x_cat + h[:, :nc].unsqueeze(2) * ((x_cat - x_cat_0) / t_cur[:, :nc].unsqueeze(2))
            x_num = x_num + h[:, nc:] * ((x_num - x_num_0) / t_cur[:, nc:])
        probs, _, _ = step(x_cat, x_num, u_steps[-2], t_steps[-2].repeat(n, 1))
        return x_num.cpu(), torch.stack([p.argmax(1) for p in probs], dim=1).cpu()


def _lr(step: int, c: dict) -> float:
    """Linear warm-up from 1e-6, then linear decay to 1e-6 at the last step."""
    base, warm, total = c["lr"], c["num_steps_lr_warmup"], c["num_steps_train"]
    if step < warm:
        return 1e-6 + (base - 1e-6) * float(step) / float(warm)
    return (1e-6 - base) / (total - warm) * (step - warm) + base


@torch.no_grad()
def _ema_update(ema: list[torch.Tensor], params: list[torch.Tensor], decay: float, n_updates: int) -> None:
    """EMA update whose decay warms up as min(decay, (1 + n) / (10 + n)) (as in torch_ema)."""
    d = min(decay, (1 + n_updates) / (10 + n_updates))
    for e, p in zip(ema, params):
        e.sub_((e - p).mul_(1.0 - d))


class CDTD:
    """CDTD trained on filled data; ``loss_mask`` restricts the loss to the observed entries."""

    def __init__(self, n_num: int, n_categories: list[int], loss_mask: bool, device="cuda", **config):
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown config keys: {sorted(unknown)}")
        self.n_num, self.n_categories, self.loss_mask = n_num, list(n_categories), loss_mask
        self.device = torch.device(device)
        self.config = {**DEFAULTS, **config}

    def fit(self, train, val=None, seed: int = 0) -> "CDTD":
        """train/val: (X_num, X_cat, M_num, M_cat). Trains the full schedule and keeps the final
        EMA weights; with val, the validation loss is logged in ``history_``."""
        c, dev = self.config, self.device
        x_cat = torch.as_tensor(np.asarray(train[1]), dtype=torch.long)
        proportions = [torch.bincount(x_cat[:, j], minlength=k) / len(x_cat) for j, k in enumerate(self.n_categories)]
        set_seed(seed)
        self.net_ = net = _Diffusion(self.n_num, self.n_categories, proportions, c).to(dev)
        params = list(net.parameters())
        ema = [p.detach().clone() for p in params]
        opt = torch.optim.AdamW(params, lr=c["lr"], betas=c["betas"], eps=c["eps"], weight_decay=c["weight_decay"])
        tr = to_tensors(train, dev)
        va = None if val is None else to_tensors(val, dev)
        batches = train_batches(tr, c["batch_size"])
        self.history_ = []
        for step in range(1, c["num_steps_train"] + 1):
            opt.zero_grad()
            out = net.loss(*next(batches), self.loss_mask)
            out["loss"].backward()
            opt.step()
            _ema_update(ema, params, c["ema_decay"], step)
            if step % c["log_steps"] == 0:
                self.history_.append({"step": step, "lr": opt.param_groups[0]["lr"],
                                      **{k: v.item() for k, v in out.items()}})
            if va is not None and step % c["validation_interval"] == 0:
                with torch.no_grad():  # loss of the current (non-EMA) weights
                    vals = [net.loss(*b, self.loss_mask)["obs"].item()
                            for b in islice(eval_batches(va, min(c["batch_size"], len(tr[0]))), c["val_batches"])]
                self.history_.append({"step": step, "val": torch.tensor(vals).mean().item()})
            for g in opt.param_groups:  # the first step runs at the base lr
                g["lr"] = _lr(step, c)
        with torch.no_grad():
            for p, e in zip(params, ema):
                p.copy_(e)
        return self

    def sample(self, n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """n rows in model space: (X_num float64, X_cat int64 codes; code K_j is the unknown class)."""
        set_seed(seed)
        bs, steps = self.config["batch_size_eval"], self.config["generation_steps"]
        out = [self.net_.sample(min(bs, n - i), steps) for i in range(0, n, bs)]
        return torch.cat([o[0] for o in out]).numpy(), torch.cat([o[1] for o in out]).numpy()
