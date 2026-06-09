"""Recurrent phenotype — self/lateral back-edges unrolled in place.  See spec §3.5.

A recurrent cell behaves as a linear cell on its feed-forward inputs and then
accumulates contributions from its *recurrent* edges (back-edges whose source is
one of the cells dispatched in this same bucket — i.e. self-loops and same-rank
laterals) across ``K_unroll`` steps:

    a⁽⁰⁾ = b_v + Σ_{u ∈ feed-forward}  w_{vu}·a_u
    a⁽ᵗ⁾ = a⁽ᵗ⁻¹⁾ + Σ_{u ∈ recurrent}  w_{vu}·a_u⁽ᵗ⁻¹⁾     for t = 1..K_unroll

The unroll is **fixed-depth, not fixed-point** (spec §3.5): exactly ``K_unroll``
steps, no convergence test, so CPU latency stays predictable. ``K_unroll`` is
per-cell (``arena.k_unroll``, capped at 8 — spec §1.4).

**Reduction guarantee.** A recurrent cell with *no* recurrent edge computes
``a = b + Σ w·a`` — byte-identical to :mod:`linear`. On a single static scalar
feature (no self-edge possible) the recurrent phenotype is therefore exactly
linear; its distinct behavior only engages once a back-edge exists. This is what
lets the council stand recurrent cells up on the 1-D probe without changing the
linear floor — they differentiate only when temporal/sequential structure (the
multi-data-type phase) gives them a loop to unroll.

Note (scheduler interaction): the substrate runs one rank-ordered forward pass,
so a recurrent cell's *cross-rank* back-edge (source at a higher rank, computed
later) reads a stale 0 at dispatch time. Only self/lateral loops within the
bucket evolve correctly during the in-op unroll; cross-rank recurrence is out of
scope here and documented as a known limit.
"""
from __future__ import annotations

import torch

from trioron.core.arena import Arena
from trioron.core.scheduler import Bucket

_K_CAP = 8  # serial-depth bound (spec §1.4)


def forward_batch(
    act: torch.Tensor,
    bucket: Bucket,
    arena: Arena,
) -> torch.Tensor:
    cells = bucket.cell_ids.long()
    n_cells = cells.numel()
    batch = act.shape[0]

    out = arena.bias[bucket.bias_ids.long()].unsqueeze(0).expand(batch, -1).clone()
    if bucket.edge_indices.numel() == 0:
        return out

    src_act = act[:, bucket.edge_src.long()]              # [batch, E]
    weights = arena.edge_weight[bucket.edge_indices]      # [E]
    weighted = src_act * weights.unsqueeze(0)             # [batch, E]
    dst_local = bucket.edge_dst_local                     # [E] → 0..n_cells-1

    # Local index lookup: global cell id → 0..n_cells-1 (−1 if not in bucket).
    local_of = torch.full((arena.capacity,), -1, dtype=torch.long, device=act.device)
    local_of[cells] = torch.arange(n_cells, device=act.device)
    src_local = local_of[bucket.edge_src.long()]          # [E]
    is_rec = src_local >= 0                                # recurrent = src is a bucket cell

    # Feed-forward base: a⁽⁰⁾ = bias + Σ_ff w·a
    ff = ~is_rec
    base = out
    if bool(ff.any()):
        base = base.scatter_add(
            1, dst_local[ff].unsqueeze(0).expand(batch, -1), weighted[:, ff]
        )

    if not bool(is_rec.any()):
        return base  # no back-edge → linear-identical (reduction guarantee)

    # Unroll the recurrent edges in place.
    rec_w = weights[is_rec]                               # [R]
    rec_dst = dst_local[is_rec]                           # [R]
    rec_src_local = src_local[is_rec]                     # [R]
    k_cells = arena.k_unroll[cells].clamp(1, _K_CAP).long()  # [n_cells]
    k_max = int(k_cells.max().item())

    a = base.clone()
    for t in range(k_max):
        step = torch.zeros(batch, n_cells, device=act.device, dtype=a.dtype)
        contrib = a[:, rec_src_local] * rec_w.unsqueeze(0)  # [batch, R]
        step = step.scatter_add(1, rec_dst.unsqueeze(0).expand(batch, -1), contrib)
        active = (k_cells > t).to(a.dtype)                  # cells still unrolling
        a = a + step * active.unsqueeze(0)
    return a


GENE_BIT = 3
NAME = "recurrent"
