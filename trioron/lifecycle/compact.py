"""Compaction — reclaim low-saliency cells.  See spec §5.5."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from trioron.core.arena import Arena
from trioron.core.epigenome import RECYCLABLE, has_gene
from trioron.core.state import CellState
from .saliency import compute_saliency, SaliencyConfig


@dataclass
class CompactConfig:
    max_recyclings: int = 10
    utility_floor_cycles: int = 5
    saliency: SaliencyConfig | None = None


@dataclass
class CompactResult:
    recycled_ids: list[int]
    defragged: bool


def compact(
    arena: Arena,
    cfg: CompactConfig | None = None,
) -> CompactResult:
    """Recycle low-saliency dormant cells and optionally defrag.

    Returns CompactResult with recycled cell ids and whether defrag ran.
    """
    cfg = cfg or CompactConfig()
    a = arena

    sal = compute_saliency(a, cfg.saliency)

    candidates = (
        a.alive
        & (a.state == CellState.DORMANT)
        & has_gene(a.epigenome, RECYCLABLE).bool()
    )
    cand_ids = candidates.nonzero(as_tuple=False).squeeze(-1)

    if cand_ids.numel() == 0:
        return CompactResult(recycled_ids=[], defragged=False)

    cand_sal = sal[cand_ids]
    cand_age = a.age[cand_ids].float()
    sort_key = cand_sal - cand_age * 1e-8
    order = sort_key.argsort()
    sorted_ids = cand_ids[order]

    n_recycle = min(cfg.max_recyclings, sorted_ids.numel())
    recycled: list[int] = []
    for i in range(n_recycle):
        cid = int(sorted_ids[i].item())
        _mark_recycled(a, cid)
        recycled.append(cid)

    p = a.pressure()
    threshold = defrag_threshold(p)
    free_frac = _free_fraction(a)
    did_defrag = False
    if free_frac > threshold and len(recycled) > 0:
        _defrag(a)
        did_defrag = True

    return CompactResult(recycled_ids=recycled, defragged=did_defrag)


def defrag_threshold(pressure: float) -> float:
    """Pressure-adaptive defrag trigger: tight at high pressure, loose when empty."""
    return max(0.05, 0.5 * (1.0 - pressure))


def _free_fraction(arena: Arena) -> float:
    if arena.cursor == 0:
        return 0.0
    dead = (~arena.alive[:arena.cursor]).sum().item()
    return dead / arena.cursor


def _mark_recycled(arena: Arena, cell_id: int) -> None:
    """Reclaim a cell slot: mark dead, remove its edges."""
    a = arena
    a.alive[cell_id] = False
    a.state[cell_id] = 0
    a.engagement[cell_id] = 0.0
    a.utility[cell_id] = 0.0
    a.bias[cell_id] = 0.0

    if a.edge_cursor > 0:
        mask = (
            (a.edge_src[:a.edge_cursor] == cell_id)
            | (a.edge_dst[:a.edge_cursor] == cell_id)
        )
        if mask.any():
            keep = ~mask
            n_keep = keep.sum().item()
            if n_keep < a.edge_cursor:
                a.edge_src[:n_keep] = a.edge_src[:a.edge_cursor][keep]
                a.edge_dst[:n_keep] = a.edge_dst[:a.edge_cursor][keep]
                with torch.no_grad():
                    a.edge_weight[:n_keep] = a.edge_weight[:a.edge_cursor][keep]
                a.edge_cursor = n_keep


def _defrag(arena: Arena) -> None:
    """Re-pack alive cells to close gaps. Remaps cell ids in edges."""
    a = arena
    alive_ids = a.alive[:a.cursor].nonzero(as_tuple=False).squeeze(-1)
    if alive_ids.numel() == a.cursor:
        return

    remap = torch.full((a.cursor,), -1, dtype=torch.int32, device=a.device)
    new_cursor = alive_ids.numel()

    for new_idx, old_idx in enumerate(alive_ids.tolist()):
        remap[old_idx] = new_idx
        if new_idx != old_idx:
            a.bias[new_idx] = a.bias[old_idx].clone()
            a.engagement[new_idx] = a.engagement[old_idx].clone()
            a.utility[new_idx] = a.utility[old_idx].clone()
            a.position[new_idx] = a.position[old_idx].clone()
            a.epigenome[new_idx] = a.epigenome[old_idx].clone()
            a.phenotype_cache[new_idx] = a.phenotype_cache[old_idx].clone()
            a.lineage_root[new_idx] = a.lineage_root[old_idx].clone()
            a.parent[new_idx] = a.parent[old_idx].clone()
            a.rank[new_idx] = a.rank[old_idx].clone()
            a.state[new_idx] = a.state[old_idx].clone()
            a.age[new_idx] = a.age[old_idx].clone()
            a.output_dim[new_idx] = a.output_dim[old_idx].clone()
            a.forward_inclusion[new_idx] = a.forward_inclusion[old_idx].clone()
            a.alive[new_idx] = True

    for i in range(new_cursor, a.cursor):
        a.alive[i] = False

    a.cursor = new_cursor

    if a.edge_cursor > 0:
        old_src = a.edge_src[:a.edge_cursor].long().clamp(max=remap.shape[0] - 1)
        old_dst = a.edge_dst[:a.edge_cursor].long().clamp(max=remap.shape[0] - 1)
        a.edge_src[:a.edge_cursor] = remap[old_src]
        a.edge_dst[:a.edge_cursor] = remap[old_dst]

    if a.edge_cursor > 0:
        valid = (a.edge_src[:a.edge_cursor] >= 0) & (a.edge_dst[:a.edge_cursor] >= 0)
        if valid.sum().item() < a.edge_cursor:
            n_valid = valid.sum().item()
            a.edge_src[:n_valid] = a.edge_src[:a.edge_cursor][valid]
            a.edge_dst[:n_valid] = a.edge_dst[:a.edge_cursor][valid]
            with torch.no_grad():
                a.edge_weight[:n_valid] = a.edge_weight[:a.edge_cursor][valid]
            a.edge_cursor = n_valid

    for i in range(new_cursor):
        pid = int(a.parent[i].item())
        if pid >= 0 and pid < remap.shape[0]:
            a.parent[i] = remap[pid]
        lr = int(a.lineage_root[i].item())
        if lr >= 0 and lr < remap.shape[0]:
            a.lineage_root[i] = remap[lr]

    a.rank_dirty = True
