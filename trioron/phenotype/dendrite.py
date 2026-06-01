"""Dendrite phenotype — NMDA-style supralinear (quad) combine.  See spec §3.6.

The base trioron forward (``linear``) is purely linear: ``y = b + Σ w·a``. A
stack of linear cells is mathematically a single linear map — no depth or
composition gain, and relational operations (comparison, XOR, products) are
unrepresentable. The dendrite phenotype adds the quad nonlinearity

    σ(z) = z + z²,   z = b + Σ w·a

The ``z²`` term expands to ``Σ_i Σ_j w_i w_j · a_i a_j`` — all pairwise input
products in one cell. This is the supralinear (NMDA-style) summation of spec
§3.6, and it is the multiplicative primitive that makes coincidence detection /
comparison learnable where a linear cell cannot (see the DMS / quad-dendrite
result, 2026-06-01). The linear ``z`` term is retained so the cell still does
ordinary (additive, sensing) work — the quad is strictly more expressive than
linear, not a replacement of its linear behaviour.

Stability: ``z²`` is unbounded; in deep stacks or recurrent loops it compounds.
Train with gradient clipping; recurrent/memory pathways should normalize the
fed-back trace (RMS norm) rather than hard-clamp it.
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
    """Quad combine: σ(z) = z + z² over the linear pre-activation z."""
    z = linear.forward_batch(act, bucket, arena)
    return z + z * z


GENE_BIT = 4
NAME = "dendrite"
