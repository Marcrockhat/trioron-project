"""Attention phenotype — scaled dot-product over the cell's fan-in.  See spec §3.3.

The spec's full attention is a Q/K/V/OUT 4-cell *spawn lineage* (§3.3). That
lineage can only be spawned and exercised where there are multiple tokens to
attend over — i.e. the multi-data-type phase — so it is deferred. What ships here
is the genuine attention *operation* an OUT cell performs once it has a fan-in:
self-attention over its incoming edges, with each edge contributing a weighted
"token" that serves as query, key, and value:

    value_i = w_{vi}·a_i                     (the weighted source activation)
    score_i = value_i / √d_k                 (d_k = 1 for scalar tokens)
    α_i     = softmax_i over the cell's own fan-in (segment softmax per dst cell)
    y_v     = b_v + Σ_i α_i · value_i

**Reduction guarantee.** With a single incoming edge the per-cell softmax is over
one element ⇒ α = 1 ⇒ y_v = b_v + w·a — byte-identical to :mod:`linear`. On a 1-D
scalar feature an attention cell is therefore exactly linear; the data-dependent
gating only engages once a cell has ≥2 tokens to weigh. This is what lets the
council stand attention cells up on the 1-D probe without perturbing the linear
floor.

The softmax is computed per *(batch row, destination cell)* via segment reduction
over ``edge_dst_local`` — fully vectorized, no per-cell Python loop (phenotype
contract §3.1, requirement 2).
"""
from __future__ import annotations

import torch

from trioron.core.arena import Arena
from trioron.core.scheduler import Bucket


def forward_batch(
    act: torch.Tensor,
    bucket: Bucket,
    arena: Arena,
) -> torch.Tensor:
    n_cells = bucket.cell_ids.numel()
    batch = act.shape[0]

    out = arena.bias[bucket.bias_ids.long()].unsqueeze(0).expand(batch, -1).clone()
    if bucket.edge_indices.numel() == 0:
        return out

    src_act = act[:, bucket.edge_src.long()]              # [batch, E]
    weights = arena.edge_weight[bucket.edge_indices]      # [E]
    value = src_act * weights.unsqueeze(0)                # [batch, E] weighted token
    score = value                                         # d_k = 1 (scalar tokens) → no scaling

    idx = bucket.edge_dst_local.unsqueeze(0).expand(batch, -1)  # [batch, E]

    # Segment softmax over each cell's fan-in (per batch row), numerically stable.
    seg_max = torch.full((batch, n_cells), float("-inf"), device=act.device)
    seg_max = seg_max.scatter_reduce(1, idx, score, reduce="amax", include_self=False)
    expv = (score - seg_max.gather(1, idx)).exp()         # [batch, E]
    seg_sum = torch.zeros(batch, n_cells, device=act.device).scatter_add(1, idx, expv)
    attn = expv / seg_sum.gather(1, idx).clamp_min(1e-20)

    ctx = torch.zeros(batch, n_cells, device=act.device).scatter_add(1, idx, attn * value)
    return out + ctx


GENE_BIT = 1
NAME = "attention"
