"""Organism base — the stable surface shared by Logophosis and Kinopsis (s062).

An *organism* is a saveable composition of frozen/trainable substrates plus
its calibration statistics. The contract:

* ``fit``-style stage methods return ``self`` (chainable acquisition schedule);
* inference methods are ``@torch.no_grad`` and take the organism's native
  input (scene + token ids / feature-stream dicts), never file paths;
* ``save(path) -> bytes``, ``load(path)``, ``n_params()`` are uniform;
* everything that learns is substrate (``trainable_tensors``) except the
  explicitly named tiny torch parts (Kinopsis router, ~60 params).

Shared helpers: ``leaf`` (a Seeded quad leaf), ``train_leaf`` (Adam + clip,
freezes on exit), ``chunked`` (no-grad batched forward with optional H-codes),
``Std`` (z-score calibration carried with the organism), ``group_contrast`` /
``train_contrastive`` (pull same-group rows together; s065 fact-memory keys).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F

from trioron.bases.seeded import Seeded
from trioron.core import Envelope, construct
from trioron.phenotype import default_dispatch_table

API_VERSION = "1.1"   # s063: +LinkChain(norm,dup,penalty), Logophosis(seq_dup,rewake,lock_links); additive, 1.0 pickles load


class Std:
    """Per-feature z-score fitted on a training stream."""

    def __init__(self, Z: torch.Tensor) -> None:
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6

    def __call__(self, Z: torch.Tensor) -> torch.Tensor:
        return ((Z - self.mu) / self.sd).float()


def leaf(d: int, n_out: int, seed: int, *, interior: int = 48):
    """One ``Seeded(d -> n_out)`` quad leaf, compiled and ready to train."""
    torch.manual_seed(seed)
    sub = construct(base=Seeded(d, n_out, interior_cells=interior, nonlinear=True),
                    envelope=Envelope(max_parameter_bytes=400_000),
                    dispatch_table=default_dispatch_table(),
                    capacity=max(d + interior + n_out + 8,
                                 (d * interior + interior * n_out) * 5 // (4 * 64) + 8),
                    sparsity_k=0)
    sub.compile(); sub.prepare_training()
    return sub


def train_leaf(sub, Z: torch.Tensor, y: torch.Tensor, seed: int, epochs: int, *,
               lr: float = 1e-3, bs: int = 256):
    """Adam + grad-clip on ``Z -> y``; the leaf is FROZEN on return and its
    grad-bearing ``last_activations`` dropped (deepcopy/pickle-safe)."""
    params = sub.trainable_tensors()
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for bi in torch.randperm(len(y), generator=g).split(bs):
            opt.zero_grad()
            F.cross_entropy(sub(Z[bi]), y[bi]).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    for p in params: p.requires_grad_(False)
    with torch.no_grad(): sub(Z[:1])
    return sub


def group_contrast(E: torch.Tensor, group: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Supervised-contrastive loss: rows sharing a ``group`` id are pulled
    together, every other row in the batch is pushed away (cosine, ``temp``)."""
    S = F.normalize(E, dim=1) @ F.normalize(E, dim=1).T / temp
    eye = torch.eye(len(E), dtype=torch.bool)
    pos = (group[:, None] == group[None, :]) & ~eye
    S = S.masked_fill(eye, float("-inf"))
    has = pos.any(1)
    num = S.masked_fill(~pos, float("-inf")).logsumexp(1)
    return (S.logsumexp(1) - num)[has].mean()


def train_contrastive(fwd, params, Z: torch.Tensor, group: torch.Tensor, block: torch.Tensor,
                      seed: int, epochs: int, *, lr: float = 1e-3, blocks_per_batch: int = 96,
                      temp: float = 0.1):
    """Adam + grad-clip on ``group_contrast`` of ``fwd(Z)``.  A batch is every
    row of ``blocks_per_batch`` blocks, so rows that share a block but not a
    group (hard negatives) always meet.  Params are FROZEN on return."""
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    ids = torch.unique(block)
    for _ in range(epochs):
        for bi in ids[torch.randperm(len(ids), generator=g)].split(blocks_per_batch):
            m = torch.isin(block, bi)
            opt.zero_grad()
            group_contrast(fwd(Z[m]), group[m], temp).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
    for p in params: p.requires_grad_(False)


@torch.no_grad()
def chunked(sub, Z: torch.Tensor, n: int = 4000, h_ids: Optional[torch.Tensor] = None):
    """Logits (and interior H-codes when ``h_ids`` is given) in chunks."""
    Ls, Hs = [], []
    for i in range(0, len(Z), n):
        Ls.append(sub(Z[i:i + n]))
        if h_ids is not None: Hs.append(sub.last_activations[:, h_ids])
    return (torch.cat(Ls), torch.cat(Hs)) if h_ids is not None else torch.cat(Ls)


class Organism:
    """Base class: uniform save / load / n_params."""
    name: str = "organism"
    api_version: str = API_VERSION

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def substrates(self) -> list:
        """Every substrate the organism carries (override)."""
        return []

    def torch_modules(self) -> list:
        """Non-substrate torch parts (tiny routers); default none."""
        return []

    def n_params(self) -> int:
        n = sum(int(p.numel()) for s in self.substrates() for p in s.trainable_tensors())
        return n + sum(p.numel() for m in self.torch_modules() for p in m.parameters())

    def save(self, path: str) -> int:
        torch.save(self, path)
        return os.path.getsize(path)

    @classmethod
    def load(cls, path: str):
        obj = torch.load(path, weights_only=False)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} holds {type(obj).__name__}, not {cls.__name__}")
        return obj
