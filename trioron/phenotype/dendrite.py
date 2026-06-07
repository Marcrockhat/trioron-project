"""Dendrite phenotype — branch-partitioned supralinear combine.  See spec §3.6.

The base trioron forward (``linear``) is purely linear: ``y = b + Σ w·a``. A
stack of linear cells is mathematically a single linear map — no depth or
composition gain, and relational operations (comparison, XOR, products) are
unrepresentable. The dendrite phenotype (Axis 5) partitions a cell's fan-in into
``K`` branches; each branch computes a *local* linear combine over its input
partition, passes it through a per-branch nonlinearity (default ``quad``,
σ(z) = z + z² — the NMDA-style supralinear activation from v1.1), and the branch
outputs are pooled at the soma with learned per-branch weights α:

    y_v = b_v + Σ_k α_{v,k} · σ( Σ_{u ∈ B_k(v)} w_{vu}·a_u )

The per-branch ``z²`` term yields all pairwise products *within a branch* — the
multiplicative primitive that makes coincidence detection / comparison learnable
where a linear cell cannot (DMS / quad-dendrite result, 2026-06-01). Because the
partition is per-branch, different branches can specialize to *disjoint* regions
of the fan-in — exactly what a disjoint-band decision boundary needs.

**K = 1 is byte-identical to linear** (spec §3.6 back-compat guarantee): with a
single branch the quad term is suppressed and α₀ = 1, so y = b + Σ w·a. The quad
nonlinearity engages only once a second branch is grown (``arena.grow_branch``).
This is the heterogeneous-substrate design: cells stay linear (sensing + CL
stability) and grow branches *only* where relational frustration demands it
(``selective_quad_growth`` verdict: a uniform always-on quad regresses CL −11pp).

Stability: ``z²`` is unbounded; in deep stacks or recurrent loops it compounds.
Train with gradient clipping; recurrent/memory pathways should normalize the
fed-back trace (RMS norm) rather than hard-clamp it.
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
    """Branch-partitioned quad combine for all dendrite cells in *bucket*.

    y_v = b_v + Σ_k α_{v,k} · σ(z_{v,k}),  z_{v,k} = Σ_{u∈B_k(v)} w_{vu}·a_u,
    σ(z) = z + z²  (suppressed to σ(z) = z for K=1 cells → linear-equivalent).
    """
    cells = bucket.cell_ids.long()
    n_cells = cells.numel()
    batch = act.shape[0]

    # Soma bias (added once, at the soma — outside the branch pooling).
    out = arena.bias[bucket.bias_ids.long()].unsqueeze(0).expand(batch, -1).clone()

    if bucket.edge_indices.numel() == 0:
        return out

    src_act = act[:, bucket.edge_src.long()]                  # [batch, E]
    weights = arena.edge_weight[bucket.edge_indices]          # [E]
    weighted = src_act * weights.unsqueeze(0)                 # [batch, E]
    e_branch = arena.edge_branch[bucket.edge_indices].long()  # [E]
    dst_local = bucket.edge_dst_local                         # [E] → 0..n_cells-1

    # Per-cell branch state. is_multi gates the quad term so K=1 stays linear.
    is_multi = (arena.n_branches[cells] >= 2).float().unsqueeze(0)  # [1, n_cells]
    alpha = arena.branch_alpha[cells]                              # [n_cells, B_max]

    for k in range(arena.branch_cap):
        kmask = e_branch == k
        if not bool(kmask.any()):
            continue
        zk = torch.zeros(batch, n_cells, device=act.device, dtype=out.dtype)
        zk.scatter_add_(
            1,
            dst_local[kmask].unsqueeze(0).expand(batch, -1),
            weighted[:, kmask],
        )
        sigma = zk + is_multi * zk * zk          # σ(z)=z+z² (K≥2) or z (K=1)
        out = out + alpha[:, k].unsqueeze(0) * sigma

    return out


GENE_BIT = 4
NAME = "dendrite"
