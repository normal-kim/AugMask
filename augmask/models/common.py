"""Small utilities shared by the diffusion backbones."""
from __future__ import annotations

import random
from typing import Iterator, Sequence

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_tensors(arrays: Sequence[np.ndarray | None], device) -> list[torch.Tensor | None]:
    """(X_num, X_cat, M_num, M_cat) -> float / long / float / float tensors."""
    dtypes = (torch.float32, torch.long, torch.float32, torch.float32)
    return [None if a is None else torch.as_tensor(np.asarray(a), dtype=dt, device=device)
            for a, dt in zip(arrays, dtypes)]


def train_batches(tensors: Sequence[torch.Tensor | None], batch_size: int) -> Iterator[list]:
    """Endless shuffled mini-batches over all rows; the last incomplete batch of each epoch is dropped."""
    n = next(t for t in tensors if t is not None).shape[0]
    bs = min(batch_size, n)
    while True:
        perm = torch.randperm(n)
        for start in range(0, n - bs + 1, bs):
            idx = perm[start:start + bs]
            yield [None if t is None else t[idx.to(t.device)] for t in tensors]


def eval_batches(tensors: Sequence[torch.Tensor | None], batch_size: int) -> Iterator[list]:
    """Ordered mini-batches over all rows (the last one may be smaller)."""
    n = next(t for t in tensors if t is not None).shape[0]
    for start in range(0, n, batch_size):
        yield [None if t is None else t[start:start + batch_size] for t in tensors]


@torch.no_grad()
def ema_update(target: Sequence[torch.Tensor], source: Sequence[torch.Tensor], decay: float) -> None:
    for t, s in zip(target, source):
        t.mul_(decay).add_(s.detach(), alpha=1 - decay)
