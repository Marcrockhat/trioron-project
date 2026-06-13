"""Retinal compression — the wide progenitor's first-sitting merge.

Design §3.2/§3.6 (progenitor_council.md); the genesis aim per Rocky
(s036): through one period the progenitor spawns only IMPORTANT
receptors — constant columns starve (census, already in genesis), and
adjacent REDUNDANT columns merge into one region sensor with an
imposed position (the center-surround equivalent). The dimensionality
ceiling is absorbed at the sensor; everything downstream (lock-in,
division, council, composer) reads the compressed positional set,
never the raw width. This module is dimension-generic: the same code
runs a 28x28 bench retina and the 1536x1024 ("1.5 Mi") aperture probe.

Geometry is BODY, not harness: input_shape=(H, W) declares the sensor
sheet's adjacency the way an eye has a retina. Without a declared
geometry there is no "adjacent", and the merge pass does not run —
flat feature vectors keep the status-quo 1:1 surface (this also keeps
every existing gate byte-identical).

Redundancy statistic: for grid-adjacent surviving continuous columns
(i, j), the resultant length of the per-sample phase-DIFFERENCE
stream, R_diff = |Σ Z_i·conj(Z_j)| / n — offset-invariant (a constant
phase shift carries the same information), computed in the same
per-sample frame the live injection uses, under the lock-in mask rule
(q ∈ {0, N} are reference, not evidence — spec §10.3).

Merge floor REDUNDANT_R = 1 − GAIN_D, derived, not bare: GAIN_D is
division's acceptance quantum — the smallest coherence structure the
organism can ever act on. A pair whose mutual phase wander stays
within that quantum carries no difference the discrimination law
could use, so pooling it is lossless to the organism. The evidence
floor reuses MIN_MEMBERS (division's own don't-judge-on-less floor):
a pair without that much co-active evidence is not merged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from trioron.core.epigenome import RECEPTOR, clear_gene, set_gene
from trioron.core.receptor import N_QUANTA, quantize_frame
from trioron.core.state import CellState

from .division import GAIN_D, MIN_MEMBERS

REDUNDANT_R = 1.0 - GAIN_D   # division's acceptance quantum, inverted
_ELEMS_PER_STEP = 2_000_000  # row-chunk bound on the [rows, pairs] work tensor


@dataclass
class RegionSensor:
    """One spawned region sensor: a pooled receptor owning member world
    columns, with an imposed retinotopic position (x, y, scale)."""
    cell_id: int
    members: List[int]
    position: Tuple[float, float, float]
    evidence: float = 0.0


def grid_positions(cols: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """(x, y, scale) ∈ [0,1]³ for world columns on the (H, W) sheet.
    Single-pixel scale = 1/√(H·W) — the linear extent of one pixel."""
    H, W = shape
    xs = (cols % W).float() / max(W - 1, 1)
    ys = (cols // W).float() / max(H - 1, 1)
    sc = torch.full_like(xs, 1.0 / math.sqrt(H * W))
    return torch.stack([xs, ys, sc], dim=1)


def adjacent_pairs(eligible: Sequence[int],
                   shape: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Right/down 4-neighbour pairs among eligible world columns."""
    H, W = shape
    in_set = set(eligible)
    pairs = []
    for c in eligible:
        if c % W != W - 1 and (c + 1) in in_set:
            pairs.append((c, c + 1))
        if c + W < H * W and (c + W) in in_set:
            pairs.append((c, c + W))
    return pairs


def pair_redundancy(chunks: Sequence[torch.Tensor],
                    eligible: Sequence[int],
                    pairs: List[Tuple[int, int]],
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(R_diff, co-active count) per adjacent pair, streamed over the
    period's buffered chunks. Phasors use the live injection's math:
    per-sample frame over the eligible continuous columns; q at the
    floor/ceiling is reference, not evidence."""
    cols = torch.as_tensor(list(eligible), dtype=torch.long)
    idx_of = {int(c): i for i, c in enumerate(cols.tolist())}
    pi = torch.tensor([idx_of[a] for a, _ in pairs], dtype=torch.long)
    pj = torch.tensor([idx_of[b] for _, b in pairs], dtype=torch.long)
    acc = torch.zeros(len(pairs), dtype=torch.complex64)
    cnt = torch.zeros(len(pairs))
    rows_per_step = max(1, _ELEMS_PER_STEP // max(len(pairs), 1))
    for x in chunks:
        for s in range(0, x.shape[0], rows_per_step):
            q, _, _ = quantize_frame(x[s:s + rows_per_step, cols])
            Z = torch.exp(1j * 2 * math.pi * q / N_QUANTA)
            valid = (q > 0) & (q < N_QUANTA)
            v = (valid[:, pi] & valid[:, pj])
            acc += (Z[:, pi] * Z[:, pj].conj() * v).sum(0).to(torch.complex64)
            cnt += v.sum(0)
    return acc.abs() / cnt.clamp_min(1.0), cnt


def find_regions(eligible: Sequence[int],
                 pairs: List[Tuple[int, int]],
                 r_diff: torch.Tensor,
                 cnt: torch.Tensor) -> List[List[int]]:
    """Union-find over pairs that clear both floors; multi-member
    components are the regions to merge."""
    parent: Dict[int, int] = {c: c for c in eligible}

    def find(c: int) -> int:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    keep = (cnt >= MIN_MEMBERS) & (r_diff >= REDUNDANT_R)
    for k, (a, b) in enumerate(pairs):
        if bool(keep[k]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    groups: Dict[int, List[int]] = {}
    for c in eligible:
        groups.setdefault(find(c), []).append(c)
    return [sorted(g) for g in groups.values() if len(g) >= 2]


def spawn_region_sensors(germline, regions: List[List[int]],
                         perception_ids: torch.Tensor,
                         shape: Tuple[int, int]) -> List[RegionSensor]:
    """Commit the merge: one pooled PERCEPTION+RECEPTOR sensor per
    region (parented to the progenitor, uniform-mean aperture, imposed
    position = member centroid, scale = √k pixel extents); member
    columns withdraw their receptor and go dormant — the same husk
    treatment as starvation, so the column mapping is preserved."""
    a = germline.substrate.arena
    H, W = shape
    out: List[RegionSensor] = []
    for members in regions:
        cid = int(a.alloc(1).item())
        epi = int(a.epigenome[cid].item())
        from trioron.core.epigenome import PERCEPTION
        a.epigenome[cid] = set_gene(set_gene(epi, PERCEPTION), RECEPTOR)
        a.parent[cid] = germline.progenitor_id
        mem = torch.tensor(members, dtype=torch.long)
        a.add_pool(cid, mem)
        pos = grid_positions(mem, shape)
        cx, cy = float(pos[:, 0].mean()), float(pos[:, 1].mean())
        scale = math.sqrt(len(members)) / math.sqrt(H * W)
        a.position[cid] = torch.tensor([cx, cy, scale])
        for c in members:
            mcid = int(perception_ids[c].item())
            a.epigenome[mcid] = clear_gene(int(a.epigenome[mcid].item()),
                                           RECEPTOR)
            a.state[mcid] = CellState.DORMANT
        out.append(RegionSensor(cid, members, (cx, cy, scale)))
    return out


def first_sitting_compress(germline,
                           chunks: Sequence[torch.Tensor],
                           eligible: Sequence[int],
                           perception_ids: torch.Tensor,
                           shape: Optional[Tuple[int, int]],
                           ) -> List[RegionSensor]:
    """The full merge pass: adjacency → redundancy → regions → spawn.
    No-op (empty list) without a declared body geometry."""
    if shape is None or len(eligible) < 2:
        return []
    pairs = adjacent_pairs(eligible, shape)
    if not pairs:
        return []
    r_diff, cnt = pair_redundancy(chunks, eligible, pairs)
    regions = find_regions(eligible, pairs, r_diff, cnt)
    return spawn_region_sensors(germline, regions, perception_ids, shape)
