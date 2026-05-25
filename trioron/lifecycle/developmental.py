"""Developmental program — stem cells, morphogens, axon guidance.  See spec §5.7."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from trioron.core.arena import Arena
from trioron.core.epigenome import (
    LINEAR, ATTENTION, CONV, RECURRENT, DENDRITE,
    PERCEPTION, OUTPUT, CREDIT_ELIGIBLE, WEIGHT_TIED_LINEAGE,
    EXPRESSION_GENES, has_gene, set_gene, clear_gene, primary_phenotype,
)
from trioron.core.state import CellState


STEM_SENTINEL: int = -1
PHENOTYPE_CHANNELS: list[int] = [CONV, LINEAR, ATTENTION, LINEAR]
N_CHANNELS: int = len(PHENOTYPE_CHANNELS)


# ── Morphogen Field ─────────────────────────────────────────────

class MorphogenField:
    """Learnable position→phenotype projection.  See spec §5.7.2.

    Maps 3D position to R regional identity channels via
    phi(c) = sigmoid(W @ pos + b).  16 parameters total.
    """

    def __init__(self, n_channels: int = N_CHANNELS, device: str | torch.device = "cpu"):
        self.n_channels = n_channels
        self.device = torch.device(device)
        self.W = torch.zeros(n_channels, 3, device=self.device)
        self.b = torch.zeros(n_channels, device=self.device)
        self._init_z_dominant()

    def _init_z_dominant(self):
        """Initialize so z-component dominates: approximates static z-gradient."""
        with torch.no_grad():
            self.W.zero_()
            self.b.zero_()
            # Channel 0 (CONV): high when z is low
            self.W[0, 2] = -6.0
            self.b[0] = 4.0
            # Channel 1 (LINEAR mid): peaked around z=0.35
            self.W[1, 2] = -4.0
            self.b[1] = 1.5
            # Channel 2 (ATTENTION): high when z is high
            self.W[2, 2] = 6.0
            self.b[2] = -3.0
            # Channel 3 (LINEAR out): high near z=1
            self.W[3, 2] = 8.0
            self.b[3] = -6.0

    def evaluate(self, positions: torch.Tensor) -> torch.Tensor:
        """Evaluate morphogen at positions [N, 3]. Returns [N, n_channels]."""
        return torch.sigmoid(positions @ self.W.T + self.b)

    def assign_phenotype(self, position: torch.Tensor) -> int:
        """Return the phenotype gene for a single cell's position [3]."""
        phi = self.evaluate(position.unsqueeze(0))[0]
        channel = int(phi.argmax().item())
        return PHENOTYPE_CHANNELS[channel]

    def trainable_tensors(self) -> list[torch.Tensor]:
        return [self.W, self.b]

    def prepare_training(self):
        self.W.requires_grad_(True)
        self.b.requires_grad_(True)


# ── Stem Cell Spawning ──────────────────────────────────────────

def spawn_stem(
    arena: Arena,
    n: int,
    z_range: tuple[float, float] = (0.05, 0.95),
    y_jitter: float = 0.1,
) -> torch.Tensor:
    """Spawn n undifferentiated stem cells as a column from perception to output.

    Cells are distributed evenly along z (the depth axis), with slight
    y-jitter for spatial diversity. x is fixed at 0.5. This ensures
    stems near z=0 can reach perception cells and stems near z=1 can
    reach output cells via axon guidance.

    Returns int32 tensor of allocated cell ids.
    """
    ids = arena.alloc(n)
    z_lo, z_hi = z_range

    # Arrange in bands: n_bands layers, each with n/n_bands cells spread in y
    n_bands = max(1, min(4, n // 4))
    cells_per_band = n // n_bands
    remainder = n - cells_per_band * n_bands

    idx = 0
    for band in range(n_bands):
        band_z = z_lo + (z_hi - z_lo) * (band + 0.5) / n_bands
        band_n = cells_per_band + (1 if band < remainder else 0)

        for j in range(band_n):
            cid = int(ids[idx].item())
            arena.epigenome[cid] = 0
            arena.phenotype_cache[cid] = LINEAR

            y = (j + 0.5) / band_n + torch.randn(1, device=arena.device).item() * y_jitter
            arena.position[cid] = torch.tensor(
                [0.5, max(0.0, min(1.0, y)), band_z],
                device=arena.device,
            )

            arena.division_mode[cid] = 0.9 - 0.5 * band_z

            arena.rank[cid] = 1
            arena.lineage_root[cid] = cid
            arena.parent[cid] = -1
            arena.forward_inclusion[cid] = True
            idx += 1

    arena.rank_dirty = True
    return ids


def is_stem(arena: Arena, cell_id: int) -> bool:
    """Check if a cell is undifferentiated (no expression genes set)."""
    epi = int(arena.epigenome[cell_id].item())
    for gene in EXPRESSION_GENES:
        if (epi >> gene) & 1:
            return False
    if has_gene(epi, PERCEPTION) or has_gene(epi, OUTPUT):
        return False
    return True


# ── Axon Guidance ───────────────────────────────────────────────

@dataclass
class GuidanceConfig:
    max_edges: int = 8
    radius: float = 0.25
    feedforward_frac: float = 1.0
    lateral_frac: float = 0.0


def axon_guidance(
    arena: Arena,
    cell_ids: torch.Tensor | None = None,
    cfg: GuidanceConfig | None = None,
) -> int:
    """Wire cells based on position proximity.  See spec §5.7.4.

    For each cell, finds nearby candidates at lower z (feedforward)
    and same z (lateral), wires the closest ones.

    Args:
        arena: the arena.
        cell_ids: cells to wire. If None, wires all alive non-perception cells.
        cfg: guidance configuration.

    Returns:
        Number of edges added.
    """
    cfg = cfg or GuidanceConfig()
    a = arena

    if cell_ids is None:
        mask = a.alive & ~has_gene(a.epigenome, PERCEPTION).bool()
        cell_ids = mask.nonzero(as_tuple=False).squeeze(-1).to(torch.int32)

    if cell_ids.numel() == 0:
        return 0

    all_alive = a.alive_ids()
    all_pos = a.position[all_alive.long()]
    all_z = all_pos[:, 2]

    n_ff = max(1, int(cfg.max_edges * cfg.feedforward_frac))
    n_lat = max(0, cfg.max_edges - n_ff)

    total_added = 0
    src_list = []
    dst_list = []

    for cid in cell_ids.tolist():
        if has_gene(int(a.epigenome[cid].item()), PERCEPTION):
            continue

        pos_c = a.position[cid]
        z_c = pos_c[2].item()

        dists = (all_pos - pos_c).norm(dim=1)
        within = dists < cfg.radius
        within_ids = all_alive[within]
        within_dists = dists[within]
        within_z = all_z[within]

        # Exclude self
        not_self = within_ids != cid
        within_ids = within_ids[not_self]
        within_dists = within_dists[not_self]
        within_z = within_z[not_self]

        if within_ids.numel() == 0:
            continue

        # Feedforward: lower z sources
        ff_mask = within_z < z_c - 0.01
        if ff_mask.any():
            ff_ids = within_ids[ff_mask]
            ff_dists = within_dists[ff_mask]
            k = min(n_ff, ff_ids.numel())
            _, topk = ff_dists.topk(k, largest=False)
            for idx in topk.tolist():
                src_list.append(int(ff_ids[idx].item()))
                dst_list.append(cid)

        # Lateral: same z
        if n_lat > 0:
            lat_mask = (within_z >= z_c - 0.01) & (within_z <= z_c + 0.01)
            lat_mask = lat_mask & ~has_gene(a.epigenome[within_ids.long()], OUTPUT).bool()
            if lat_mask.any():
                lat_ids = within_ids[lat_mask]
                lat_dists = within_dists[lat_mask]
                k = min(n_lat, lat_ids.numel())
                _, topk = lat_dists.topk(k, largest=False)
                for idx in topk.tolist():
                    src_list.append(int(lat_ids[idx].item()))
                    dst_list.append(cid)

    if src_list:
        src_t = torch.tensor(src_list, dtype=torch.int32, device=a.device)
        dst_t = torch.tensor(dst_list, dtype=torch.int32, device=a.device)
        # He initialization: std = sqrt(2 / fan_in) per destination cell
        fan_ins = torch.zeros(a.capacity, device=a.device)
        fan_ins.scatter_add_(0, dst_t.long(), torch.ones(len(dst_list), device=a.device))
        per_edge_fan = fan_ins[dst_t.long()].clamp(min=1)
        weights = torch.randn(len(src_list), device=a.device) * (2.0 / per_edge_fan).sqrt()
        a.add_edges(src_t, dst_t, weights)
        total_added = len(src_list)

    a.rank_dirty = True
    return total_added


# ── Differentiation ─────────────────────────────────────────────

def differentiate(
    arena: Arena,
    cell_id: int,
    morphogen: MorphogenField,
    neighbor_phenotypes: list[int] | None = None,
) -> int:
    """Commit a stem cell to a phenotype based on morphogen + lateral signal.

    Returns the assigned phenotype gene, or -1 if already committed.
    """
    if not is_stem(arena, cell_id):
        return -1

    pos = arena.position[cell_id]
    phi = morphogen.evaluate(pos.unsqueeze(0))[0]

    # Lateral inhibition: if neighbors are already this phenotype, try next
    channel = int(phi.argmax().item())
    if neighbor_phenotypes:
        assigned = PHENOTYPE_CHANNELS[channel]
        same_count = sum(1 for p in neighbor_phenotypes if p == assigned)
        if same_count >= len(neighbor_phenotypes) * 0.5 and len(neighbor_phenotypes) >= 2:
            # Pick next-best channel
            sorted_channels = phi.argsort(descending=True)
            for alt in sorted_channels[1:].tolist():
                alt_pheno = PHENOTYPE_CHANNELS[alt]
                alt_count = sum(1 for p in neighbor_phenotypes if p == alt_pheno)
                if alt_count < len(neighbor_phenotypes) * 0.5:
                    channel = alt
                    break

    phenotype = PHENOTYPE_CHANNELS[channel]
    epi = int(arena.epigenome[cell_id].item())
    epi = set_gene(epi, phenotype)
    epi = set_gene(epi, CREDIT_ELIGIBLE)

    if phenotype == CONV:
        epi = set_gene(epi, WEIGHT_TIED_LINEAGE)

    arena.epigenome[cell_id] = epi
    arena.refresh_phenotype(cell_id)

    return phenotype


def differentiate_all_stems(
    arena: Arena,
    morphogen: MorphogenField,
    lateral_radius: float = 0.15,
) -> list[tuple[int, int]]:
    """Differentiate all stem cells. Returns list of (cell_id, phenotype)."""
    results = []
    alive = arena.alive_ids().tolist()
    stems = [cid for cid in alive if is_stem(arena, cid)]

    for cid in stems:
        # Gather neighbor phenotypes for lateral inhibition
        pos_c = arena.position[cid]
        neighbors = []
        for other in alive:
            if other == cid:
                continue
            d = (arena.position[other] - pos_c).norm().item()
            if d < lateral_radius:
                p = primary_phenotype(int(arena.epigenome[other].item()))
                if not is_stem(arena, other):
                    neighbors.append(p)

        pheno = differentiate(arena, cid, morphogen, neighbors)
        if pheno >= 0:
            results.append((cid, pheno))

    return results


# ── Lateral Signaling Helpers ───────────────────────────────────

def neighbor_count(arena: Arena, cell_id: int, radius: float = 0.15) -> int:
    """Count alive cells within radius of cell_id."""
    pos = arena.position[cell_id]
    alive = arena.alive_ids()
    if alive.numel() == 0:
        return 0
    dists = (arena.position[alive.long()] - pos).norm(dim=1)
    return int((dists < radius).sum().item()) - 1  # exclude self


def is_dense(arena: Arena, cell_id: int, radius: float = 0.15, threshold: int = 12) -> bool:
    """Check if the region around cell_id is too dense for further growth."""
    return neighbor_count(arena, cell_id, radius) >= threshold
