"""Pool-matched L1 absorption.

Architectural premise (per session 2026-05-22): absorption works at the
POOL level, not the cell level. A donor L1 cell drawn from pool k can
land on any recipient L1 cell-slot in pool k — pools are interchangeable
within themselves. The pool is defined by positional partition of the
unit square (default 4x4 grid → 16 pools), matching the chained-15
initial L1 layout.

This unblocks positional constraints on L1 (LCN, alignment-driven
topography): they no longer conflict with absorption, because donor
and recipient share the pool convention and donor cells preserve
their pool when absorbed.

API:
    pool_id(position, grid_size=4)         → int in [0, grid_size²)
    pool_centroid(pool_id, grid_size=4)    → (cx, cy) float in [0,1]²
    absorb_l1_cell(recipient, donor, idx)  → new cell idx in recipient
    pool_matched_absorb(recipient, donor)  → list of (donor_idx, new_idx)

Both nets MUST share the L1 fan_in (== L0 width, guaranteed by R·S
handshake if same L0 seed) and head width. Caller is responsible for
rebuilding any optimizer that referenced recipient's L1 or head.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def pool_id(position: Sequence[float], grid_size: int = 4) -> int:
    """Map a 2D position in [0,1]² to a pool ID in [0, grid_size²)."""
    # Clamp to avoid index out-of-range at exactly 1.0.
    x = max(0.0, min(0.999999, float(position[0])))
    y = max(0.0, min(0.999999, float(position[1])))
    col = int(x * grid_size)
    row = int(y * grid_size)
    return row * grid_size + col


def pool_centroid(pool_id_int: int, grid_size: int = 4) -> Tuple[float, float]:
    """Inverse of pool_id: return the (cx, cy) centroid of the pool."""
    row = pool_id_int // grid_size
    col = pool_id_int % grid_size
    cx = (col + 0.5) / grid_size
    cy = (row + 0.5) / grid_size
    return cx, cy


def absorb_l1_cell(
    recipient_net,
    donor_net,
    donor_l1_idx: int,
    l1_layer_idx: int = 1,
    snap_to_pool_centroid: bool = False,
    grid_size: int = 4,
    donor_trained_classes: Optional[Sequence[int]] = None,
) -> int:
    """Absorb one donor L1 cell into recipient. Returns new recipient idx.

    Both nets must have the same L1 fan_in and head width. The recipient
    grows by one cell at L1 (with donor's W row + bias) and gains one
    new head-input column (donor's head column). After return, the
    caller MUST rebuild any optimizer holding recipient's L1/head params.

    snap_to_pool_centroid: if True, set the new cell's position to the
    centroid of its pool; if False, copy donor's exact position
    (preserves migration history).

    donor_trained_classes: if provided, ZERO the donor's head column for
    every class NOT in this set. Donor head weights for classes the
    donor never trained on are just random init noise — when transferred,
    they interfere with recipient's predictions on those classes. Zeroing
    eliminates the interference without losing real signal (donor's
    contribution on its trained classes is fully preserved).
    """
    donor_L1 = donor_net.layers[l1_layer_idx]
    donor_head = donor_net.layers[l1_layer_idx + 1]
    recipient_L1 = recipient_net.layers[l1_layer_idx]
    recipient_head = recipient_net.layers[l1_layer_idx + 1]

    if donor_L1.fan_in != recipient_L1.fan_in:
        raise ValueError(
            f"L1 fan_in mismatch: donor={donor_L1.fan_in} "
            f"recipient={recipient_L1.fan_in} — both must share L0 width"
        )
    if donor_head.n_nodes != recipient_head.n_nodes:
        raise ValueError(
            f"Head width mismatch: donor={donor_head.n_nodes} "
            f"recipient={recipient_head.n_nodes}"
        )
    if not (0 <= donor_l1_idx < donor_L1.n_nodes):
        raise IndexError(
            f"donor_l1_idx {donor_l1_idx} out of range [0, {donor_L1.n_nodes})"
        )

    donor_W_row = donor_L1.W[donor_l1_idx].detach().clone()
    donor_b = float(donor_L1.b[donor_l1_idx].item())
    donor_head_col = donor_head.W[:, donor_l1_idx].detach().clone()
    donor_pos = donor_L1.cell_position[donor_l1_idx, :].detach().clone()

    # Selective head merging: zero out columns for classes the donor never
    # trained on. Donor's head weights for untrained classes are random
    # init noise; transferring them produces additive interference on the
    # recipient's predictions for those classes.
    if donor_trained_classes is not None:
        trained_set = set(int(c) for c in donor_trained_classes)
        n_classes = donor_head_col.shape[0]
        for c in range(n_classes):
            if c not in trained_set:
                donor_head_col[c] = 0.0

    new_idx = recipient_net.grow_layer(
        l1_layer_idx,
        init_vec=donor_W_row,
        peer_init_for_next=donor_head_col,
    )

    with torch.no_grad():
        recipient_L1.b.data[new_idx] = donor_b
        if snap_to_pool_centroid:
            pid = pool_id(donor_pos[:2].tolist(), grid_size=grid_size)
            cx, cy = pool_centroid(pid, grid_size=grid_size)
            recipient_L1.cell_position[new_idx, 0] = cx
            recipient_L1.cell_position[new_idx, 1] = cy
        else:
            recipient_L1.cell_position[new_idx, :2] = donor_pos[:2]
        # Always set z=0 to match the 2D convention used in axis6_credit_chained15.
        if recipient_L1.cell_position.shape[1] >= 3:
            recipient_L1.cell_position[new_idx, 2] = 0.0
    recipient_L1._field_kernel = None
    return new_idx


def pool_matched_absorb(
    recipient_net,
    donor_net,
    grid_size: int = 4,
    pool_capacity: Optional[int] = None,
    l1_layer_idx: int = 1,
    snap_to_pool_centroid: bool = False,
    donor_trained_classes: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int]]:
    """Absorb donor's L1 cells into recipient via pool match.

    For each donor cell, compute its pool ID from position. If
    pool_capacity is set and the recipient is at capacity in that pool,
    skip the donor cell. Otherwise append it via absorb_l1_cell.

    donor_trained_classes (recommended): set of class IDs the donor was
    actually trained on. Forwarded to absorb_l1_cell to zero head columns
    for untrained classes (eliminates absorption-time interference on
    recipient's predictions). Without this, recipient classes degrade
    ~10pp post-absorption from B's random-init head weights.

    Returns list of (donor_idx, new_recipient_idx) for absorbed cells.
    Skipped cells are not in the list.
    """
    donor_L1 = donor_net.layers[l1_layer_idx]
    recipient_L1 = recipient_net.layers[l1_layer_idx]

    pool_counts = {}
    for i in range(recipient_L1.n_nodes):
        pid = pool_id(recipient_L1.cell_position[i, :2].tolist(), grid_size)
        pool_counts[pid] = pool_counts.get(pid, 0) + 1

    absorbed: List[Tuple[int, int]] = []
    for donor_idx in range(donor_L1.n_nodes):
        donor_pos = donor_L1.cell_position[donor_idx, :2].tolist()
        pid = pool_id(donor_pos, grid_size)
        if pool_capacity is not None and pool_counts.get(pid, 0) >= pool_capacity:
            continue
        new_idx = absorb_l1_cell(
            recipient_net,
            donor_net,
            donor_idx,
            l1_layer_idx=l1_layer_idx,
            snap_to_pool_centroid=snap_to_pool_centroid,
            grid_size=grid_size,
            donor_trained_classes=donor_trained_classes,
        )
        pool_counts[pid] = pool_counts.get(pid, 0) + 1
        absorbed.append((donor_idx, new_idx))
    return absorbed


def merge_manifold(recipient_manifold, donor_manifold) -> int:
    """Merge donor's per-class L0 (μ, σ) into recipient's ManifoldStore.

    Both stores must have compatible n_l0 (same L0 width — guaranteed
    by R·S handshake). For each class in donor's store:
      - If recipient doesn't have it: copy donor's (μ, σ)
      - If recipient has it: skip (recipient's own training is more
        trustworthy for shared classes)

    Returns number of classes added. The recipient's manifold can then
    drive post-absorption head re-calibration via manifold-replay
    forwards.
    """
    if donor_manifold.n_l0 != recipient_manifold.n_l0:
        raise ValueError(
            f"manifold n_l0 mismatch: donor={donor_manifold.n_l0} "
            f"recipient={recipient_manifold.n_l0}"
        )
    n_added = 0
    for c, mu in donor_manifold.mu_per_class.items():
        if c in recipient_manifold.mu_per_class:
            continue
        recipient_manifold.mu_per_class[c] = mu.detach().clone()
        recipient_manifold.sigma_per_class[c] = (
            donor_manifold.sigma_per_class[c].detach().clone()
        )
        n_added += 1
    return n_added


# ---------------------------------------------------------------------
# Sanity test for pool_id (run as __main__)
# ---------------------------------------------------------------------

def _self_test() -> int:
    """Sanity: cells on 4x4 init grid get distinct pool IDs 0..15."""
    # The 4x4 grid in axis6_credit_chained15 lays cells at
    # ((i+0.5)/4, (j+0.5)/4) for i,j in [0,4).
    seen = set()
    for j in range(4):
        for i in range(4):
            cx = (i + 0.5) / 4.0
            cy = (j + 0.5) / 4.0
            pid = pool_id((cx, cy), grid_size=4)
            seen.add(pid)
            print(f"  cell at ({cx:.3f}, {cy:.3f}) → pool {pid}")
    assert len(seen) == 16, f"expected 16 distinct pools, got {len(seen)}"
    print(f"  → {len(seen)} distinct pools ✓")

    # Inverse: pool_centroid recovers grid centers.
    for pid in range(16):
        cx, cy = pool_centroid(pid, grid_size=4)
        pid2 = pool_id((cx, cy), grid_size=4)
        assert pid == pid2, f"round trip {pid} → {(cx, cy)} → {pid2}"
    print("  centroid round-trip ✓")

    # Edge cases: corners and boundaries.
    assert pool_id((0.0, 0.0), 4) == 0
    assert pool_id((0.999, 0.999), 4) == 15
    assert pool_id((0.249, 0.249), 4) == 0
    assert pool_id((0.251, 0.249), 4) == 1
    assert pool_id((0.249, 0.251), 4) == 4
    print("  edge cases ✓")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
