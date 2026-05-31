"""Cellular division — the primary structural plasticity mechanism.  See spec §5.1."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from trioron.core.arena import Arena
from trioron.core.epigenome import (
    PERCEPTION, OUTPUT, CREDIT_ELIGIBLE, has_gene, set_gene,
)
from trioron.core.state import CellState


@dataclass
class GrowthConfig:
    frustration_threshold: float = 1.46
    frustration_window: int = 20
    frustration_sustained_frac: float = 0.635
    rank_saturation_eps: float = 0.1
    g_min: float = 1e-4
    g_max: float = 1.0
    inherit_frac: float = 0.246
    new_edges: int = 12
    position_jitter: float = 0.047
    same_rank_edges: bool = False  # allow interior↔interior edges → self-organized depth


@dataclass
class GrowthEvent:
    parent_id: int
    child_id: int
    rank: int
    n_inherited_edges: int
    n_new_edges: int


def divide(
    arena: Arena,
    parent_id: int,
    cfg: GrowthConfig | None = None,
) -> GrowthEvent | None:
    """Divide *parent_id* into one child cell with inherited + new edges.

    Returns a GrowthEvent on success, None if the envelope blocks growth.
    """
    cfg = cfg or GrowthConfig()
    a = arena

    if not a.allows_growth(add_cells=1, add_edges=cfg.new_edges + 1):
        return None

    child_ids = a.alloc(1)
    cid = int(child_ids[0].item())
    pid = parent_id

    a.parent[cid] = pid
    root = int(a.lineage_root[pid].item())
    a.lineage_root[cid] = root if root >= 0 else pid
    a.epigenome[cid] = int(a.epigenome[pid].item())
    a.output_dim[cid] = int(a.output_dim[pid].item())

    # Division orientation: symmetric (lateral) vs asymmetric (axial)
    sym_prob = a.division_mode[pid].item()
    is_symmetric = torch.rand(1, device=a.device).item() < sym_prob

    parent_pos = a.position[pid]
    if is_symmetric:
        # Lateral: same z, offset in y (builds width)
        offset = torch.tensor([0.0, torch.randn(1).item() * 0.05, 0.0], device=a.device)
    else:
        # Axial: shift toward higher z (builds depth)
        offset = torch.tensor([0.0, torch.randn(1).item() * 0.02, 0.03], device=a.device)

    a.position[cid] = (parent_pos + offset).clamp(0.0, 1.0)

    # Inherit division mode with slight mutation
    a.division_mode[cid] = (a.division_mode[pid] + torch.randn(1, device=a.device).item() * 0.05).clamp(0.0, 1.0)

    a.rank[cid] = int(a.rank[pid].item())
    a.refresh_phenotype(cid)

    parent_src, parent_w = a.inputs_of(pid)
    n_inherit = max(1, int(parent_src.numel() * cfg.inherit_frac))
    if parent_src.numel() > 0:
        perm = torch.randperm(parent_src.numel(), device=a.device)[:n_inherit]
        inh_src = parent_src[perm]
        inh_w = (parent_w[perm] * 0.5).detach()
        a.add_edges(inh_src, torch.full_like(inh_src, cid), inh_w)
    else:
        n_inherit = 0

    child_rank = int(a.rank[cid].item())
    # Edge-source policy.  Strict `<` keeps the substrate bipartite (a
    # 1-hidden-layer MLP).  Relaxed `<=` lets a new cell draw from same-rank
    # cells; recompute_ranks (Kahn's BFS) then promotes it to rank+1, so depth
    # self-organizes through growth.  The child has zero out-edges at division
    # time (it is a sink), so any incoming edge is cycle-safe by construction —
    # no _creates_forbidden_cycle check needed.  Exclude the child itself to
    # avoid a degenerate self-edge.
    rank_ok = (a.rank <= child_rank) if cfg.same_rank_edges else (a.rank < child_rank)
    lower_mask = (
        a.alive
        & (a.state == CellState.ACTIVE)
        & rank_ok
        & (torch.arange(a.capacity, device=a.device) != cid)
        & ~has_gene(a.epigenome, OUTPUT).bool()
    )
    lower_ids = lower_mask.nonzero(as_tuple=False).squeeze(-1)

    n_new = 0
    if lower_ids.numel() > 0:
        k = min(cfg.new_edges, lower_ids.numel())
        perm = torch.randperm(lower_ids.numel(), device=a.device)[:k]
        new_src = lower_ids[perm].to(torch.int32)
        new_dst = torch.full((k,), cid, dtype=torch.int32, device=a.device)
        a.add_edges(new_src, new_dst)
        n_new = k

    return GrowthEvent(
        parent_id=pid,
        child_id=cid,
        rank=child_rank,
        n_inherited_edges=n_inherit,
        n_new_edges=n_new,
    )


def check_growth_trigger(
    frustration_multiplier: float,
    frustration_steps: int,
    cfg: GrowthConfig | None = None,
) -> bool:
    """Simplified growth trigger check: frustration sustained above threshold."""
    cfg = cfg or GrowthConfig()
    min_sustained = int(cfg.frustration_window * cfg.frustration_sustained_frac)
    return (
        frustration_multiplier >= cfg.frustration_threshold
        and frustration_steps >= min_sustained
    )
