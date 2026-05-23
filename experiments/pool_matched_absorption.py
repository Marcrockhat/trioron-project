"""Deprecated shim — pool-matched absorption now lives in trioron/.

The pool/LCN/absorption stack was promoted from this experiments-grade
module into the trioron package on 2026-05-22:

  - ``pool_id`` / ``pool_centroid``                → trioron.spatial
  - ``absorb_l1_cell`` / ``pool_matched_absorb``   → trioron.network
                                                     (TrioronNetwork
                                                      methods +
                                                      api.pool_matched_absorb)
  - ``merge_manifold``                             → trioron.network
                                                     (module level +
                                                      api.merge_manifold)
  - LCN mask + cell_position                       → already on
                                                     TrioronLayer
                                                     (enable_lcn,
                                                      extend_lcn_mask)

This file remains so the existing test scripts continue to work
unchanged. New code should import from ``trioron.spatial`` /
``trioron.network`` / ``trioron.api`` directly.

The function-level signatures here forward to the trioron package
implementations. ``absorb_l1_cell`` is preserved as a free-function
wrapper around ``TrioronNetwork.absorb_cell``; the legacy
``l1_layer_idx`` kwarg is mapped to the new ``layer_idx`` kwarg.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import merge_manifold
from trioron.spatial import pool_centroid, pool_id


def absorb_l1_cell(
    recipient_net,
    donor_net,
    donor_l1_idx: int,
    l1_layer_idx: int = 1,
    snap_to_pool_centroid: bool = False,
    grid_size: int = 4,
    donor_trained_classes: Optional[Sequence[int]] = None,
) -> int:
    """Legacy free-function alias for ``TrioronNetwork.absorb_cell``.

    The original module exposed this as the per-cell entry point;
    callers should migrate to ``recipient.absorb_cell(donor, ...)``
    or :func:`trioron.api.pool_matched_absorb` for the bulk path.
    """
    return recipient_net.absorb_cell(
        donor_net,
        donor_l1_idx,
        layer_idx=l1_layer_idx,
        snap_to_pool_centroid=snap_to_pool_centroid,
        grid_size=grid_size,
        donor_trained_classes=donor_trained_classes,
    )


def pool_matched_absorb(
    recipient_net,
    donor_net,
    grid_size: int = 4,
    pool_capacity: Optional[int] = None,
    l1_layer_idx: int = 1,
    snap_to_pool_centroid: bool = False,
    donor_trained_classes: Optional[Sequence[int]] = None,
    recipient_trained_classes: Optional[Sequence[int]] = None,
    position_jitter: float = 0.0,
) -> List[Tuple[int, int]]:
    """Legacy free-function alias for ``TrioronNetwork.pool_matched_absorb``.

    Positional args preserved for compatibility with the original
    experiments call sites. New code should use
    :func:`trioron.api.pool_matched_absorb` or the method directly.
    """
    return recipient_net.pool_matched_absorb(
        donor_net,
        layer_idx=l1_layer_idx,
        grid_size=grid_size,
        pool_capacity=pool_capacity,
        snap_to_pool_centroid=snap_to_pool_centroid,
        donor_trained_classes=donor_trained_classes,
        recipient_trained_classes=recipient_trained_classes,
        position_jitter=position_jitter,
    )


__all__ = [
    "pool_id",
    "pool_centroid",
    "absorb_l1_cell",
    "pool_matched_absorb",
    "merge_manifold",
]


# ---------------------------------------------------------------------
# Sanity test for the promoted pool_id (run as __main__)
# ---------------------------------------------------------------------


def _self_test() -> int:
    """Sanity: cells on a 4x4 init grid get distinct pool IDs 0..15."""
    seen = set()
    for j in range(4):
        for i in range(4):
            cx = (i + 0.5) / 4.0
            cy = (j + 0.5) / 4.0
            pid = pool_id((cx, cy), grid_size=4)
            seen.add(pid)
            print(f"  cell at ({cx:.3f}, {cy:.3f}) -> pool {pid}")
    assert len(seen) == 16, f"expected 16 distinct pools, got {len(seen)}"
    print(f"  -> {len(seen)} distinct pools ok")

    for pid in range(16):
        cx, cy = pool_centroid(pid, grid_size=4)
        pid2 = pool_id((cx, cy), grid_size=4)
        assert pid == pid2, f"round trip {pid} -> {(cx, cy)} -> {pid2}"
    print("  centroid round-trip ok")

    assert pool_id((0.0, 0.0), 4) == 0
    assert pool_id((0.999, 0.999), 4) == 15
    print("  edge cases ok")
    return 0


if __name__ == "__main__":
    sys.exit(_self_test())
