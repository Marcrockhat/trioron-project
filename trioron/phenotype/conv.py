"""Conv phenotype — kernel weights shared across receptors by lineage.  See spec §3.4.

The defining property of convolution in this substrate is **parameter sharing**:
a conv cell is a *sensor* (its identity is where it samples), and the kernel
weights live once at the cell's ``lineage_root`` (spec §3.4 — receptors are
zero-weight sensor cells that read their weights through the lineage tie). Many
receptors at different positions share one kernel; growing spatial coverage is
free (new sensor cells, no new weights).

This op implements that sharing directly: for each incoming edge, the weight
applied is read from the edge at the *same tap* (same ordinal position in the
fan-in) into the cell's ``lineage_root``. Gradients flow back into the shared
root edges, so all receptors of a kernel are tied.

**Reduction guarantee.** When a cell is its own root (``lineage_root`` unset or
== self — the default for a hand-wired or council trial cell, before any
astrocyte/receptor lineage is spawned), each edge reads *its own* weight, so the
op is ``y = b + Σ w·a`` — byte-identical to :mod:`linear`. On a 1-D scalar there
is no spatial axis and no second receptor to share with, so a conv cell is
exactly linear; the sharing only does work once a receptor lineage exists.

Deferred to the multi-data-type phase (where there is a spatial manifold to
sample): the astrocyte/receptor *spawn* convention (§3.8) and positional kernel
footprints. What ships here is the weight-tying primitive those will use.
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
    cells = bucket.cell_ids.long()
    n_cells = cells.numel()
    batch = act.shape[0]

    out = arena.bias[bucket.bias_ids.long()].unsqueeze(0).expand(batch, -1).clone()
    E = bucket.edge_indices.numel()
    if E == 0:
        return out

    dst_local = bucket.edge_dst_local                    # [E] → 0..n_cells-1
    own_w = arena.edge_weight[bucket.edge_indices]       # [E] each edge's own weight

    # Tap ordinal: position of each edge within its dst cell's fan-in (bucket order).
    order = torch.argsort(dst_local, stable=True)
    inv = torch.empty_like(order); inv[order] = torch.arange(E, device=act.device)
    sorted_dst = dst_local[order]
    is_start = torch.ones(E, dtype=torch.bool, device=act.device)
    if E > 1:
        is_start[1:] = sorted_dst[1:] != sorted_dst[:-1]
    arangeE = torch.arange(E, device=act.device)
    group_start = torch.cummax(torch.where(is_start, arangeE, torch.zeros_like(arangeE)), 0).values
    tap = (arangeE - group_start)[inv]                   # [E]
    n_taps = int(tap.max().item()) + 1

    # Kernel source per cell: lineage_root if it's a distinct in-bucket cell, else self.
    local_of = torch.full((arena.capacity,), -1, dtype=torch.long, device=act.device)
    local_of[cells] = torch.arange(n_cells, device=act.device)
    lr = arena.lineage_root[cells].long()                # [n_cells] global root id or -1
    root_global = torch.where(lr >= 0, lr, cells)
    root_local = local_of[root_global.clamp(min=0)]
    root_local = torch.where(root_local >= 0, root_local,
                             torch.arange(n_cells, device=act.device))  # off-bucket root → self

    # For each edge, find the edge index that holds the shared weight:
    # the edge into root_local[dst] at the same tap. (cell, tap) keys are unique.
    key = dst_local * n_taps + tap
    pos_table = torch.full((n_cells * n_taps,), -1, dtype=torch.long, device=act.device)
    pos_table[key] = arangeE
    want = root_local[dst_local] * n_taps + tap
    src_edge = pos_table[want]
    src_edge = torch.where(src_edge >= 0, src_edge, arangeE)  # missing tap at root → own weight
    kw = own_w[src_edge]                                  # differentiable gather → ties gradients

    weighted = act[:, bucket.edge_src.long()] * kw.unsqueeze(0)   # [batch, E]
    out = out.scatter_add(1, dst_local.unsqueeze(0).expand(batch, -1), weighted)
    return out


GENE_BIT = 2
NAME = "conv"
