"""Tanh phenotype — bounded-saturation affine.  See spec §3.10.

y_v = tanh(b_v + Σ w_{vu}·a_u). The bounded primitive the palette lacked:
linear is unbounded-affine, the dendrite quad (z+z²) is unbounded-
expansive (the GCU detonation, s028, is what unbounded expansion does in
depth). Tanh composes safely in depth (|y| ≤ 1 regardless of fan-in) and
reduces to ~linear for |z| ≪ 1. Added s030 with its own council seat
group.
"""
from __future__ import annotations

import torch

from trioron.core.arena import Arena
from trioron.core.scheduler import Bucket

from . import linear


def forward_batch(
    act: torch.Tensor,
    bucket: Bucket,
    arena: Arena,
) -> torch.Tensor:
    """Batched tanh-affine for all cells in *bucket* — the linear combine
    followed by the bounded saturation."""
    return torch.tanh(linear.forward_batch(act, bucket, arena))


GENE_BIT = 12
NAME = "tanh"
