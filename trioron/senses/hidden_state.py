"""Hidden-state sense — a frozen language model's internal state as leaf input (s064).

The reflex brain (a small frozen LLM, world-side) exposes one hidden-state
vector per query.  This sense turns that vector into what a quad leaf can use:
z-score, project onto the top ``k`` principal directions fitted on a training
stream, whiten, and (by default) duplicate the block so every pair of
components shares a dendritic branch (branch-partition limit, spec §3.6).

The sense is fitted once, unsupervised, and carried with the organism.  It
knows nothing about the model that produced the vectors.
See docs/design/incubator_matron.md §2 (the "sense for the reflex's state").
"""
from __future__ import annotations

import torch


class HiddenStateSense:
    """``[N, d]`` hidden states -> ``[N, k]`` (or ``[N, 2k]`` with ``dup``)."""

    def __init__(self, H: torch.Tensor, k: int = 64, *, dup: bool = True) -> None:
        H = H.float()
        self.mu, self.sd = H.mean(0), H.std(0) + 1e-6
        Z = (H - self.mu) / self.sd
        k = min(k, Z.shape[0] - 1, Z.shape[1])
        _, S, Vt = torch.linalg.svd(Z, full_matrices=False)
        self.basis = Vt[:k].T.contiguous()                    # [d, k]
        self.scale = S[:k] / (Z.shape[0] - 1) ** 0.5 + 1e-6   # per-component std
        self.k, self.dup = k, dup

    @property
    def out_dim(self) -> int:
        return self.k * (2 if self.dup else 1)

    def __call__(self, H: torch.Tensor) -> torch.Tensor:
        P = ((H.float() - self.mu) / self.sd) @ self.basis / self.scale
        return torch.cat([P, P], 1) if self.dup else P
