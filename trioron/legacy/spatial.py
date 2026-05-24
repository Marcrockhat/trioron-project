"""Spatial primitives — positions, pools, locally-connected masks.

The trioron substrate is natively flat-dense (no convolution, no
weight-sharing). Spatial structure is opt-in and lives in this module:

  - **Positions.** Every TrioronLayer carries a `cell_position` buffer
    of shape (n_nodes, 3). The position itself is just a coordinate;
    the meaning is whatever downstream policies make of it. Two
    helpers seed image-style coordinates: `grid_positions_2d` for
    the L0 cell layout (a regular grid over [0, 1]²) and
    `pixel_positions_2d` for the input-side (one position per pixel
    of an H×W image).

  - **Pools.** `pool_id` partitions [0, 1]² into a grid of pools.
    Pool identity is the architectural invariant under absorption:
    a donor cell at pool k lands on a recipient cell-slot in pool k
    (`trioron.network.pool_matched_absorb`). Within a pool, cells
    are interchangeable; across pools they are not.

  - **LCN masks.** Locally-Connected-Network masks tie each output
    cell to a window of input cells in position space. `build_lcn_mask`
    returns an (n_out, n_in) tensor in [0, 1]: "soft" Gaussian falloff
    or "hard" top-K nearest-neighbour. Apply the mask multiplicatively
    to a weight matrix to constrain its read pattern; `TrioronLayer
    .enable_lcn` wires this end-to-end (install, apply, extend on
    grow). `locality_metric` summarises how much of a masked layer's
    weight mass sits inside the window.

The architectural unlock from 2026-05-22 is that LCN and pool-matched
absorption compose without conflict — donor and recipient share the
pool convention, so the recipient's mask extension is idempotent on
donor cells. See [[pool_matched_absorption]] in memory for the
verification result.

These helpers are pure (no nn.Module classes, no state). They take
positions and return tensors. The integration points are
`TrioronLayer.enable_lcn` / `extend_lcn_mask` in `node.py` and
`TrioronNetwork.absorb_cell` / `pool_matched_absorb` in `network.py`.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch


# ---------------------------------------------------------------------
# Position seeds
# ---------------------------------------------------------------------


def grid_positions_2d(
    n_x: int,
    n_y: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return (n_x * n_y, 2) cell-center positions on a regular grid
    over [0, 1]².

    The cells are laid out in column-major order (the outer loop is
    over y), matching the chained-15 L0 layout: cell index
    `j * n_x + i` sits at (cx, cy) = ((i + 0.5) / n_x, (j + 0.5) / n_y).
    Pad with `cell_position[:, 2] = 0` on the caller side to fill the
    3D position buffer.
    """
    if n_x < 1 or n_y < 1:
        raise ValueError(f"grid dims must be >= 1, got ({n_x}, {n_y})")
    cx = (torch.arange(n_x, dtype=dtype, device=device) + 0.5) / n_x
    cy = (torch.arange(n_y, dtype=dtype, device=device) + 0.5) / n_y
    gx, gy = torch.meshgrid(cx, cy, indexing="ij")
    return torch.stack([gx, gy], dim=-1).reshape(-1, 2)


def pixel_positions_2d(
    height: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return (height * width, 2) pixel-center positions on [0, 1]²
    in row-major flatten order: index `r * width + c` sits at
    ((c + 0.5) / width, (r + 0.5) / height).

    Use this to seed input-side positions for an L0 layer that reads
    a flattened image: `lcn_in_positions = pixel_positions_2d(28, 28)`.
    """
    if height < 1 or width < 1:
        raise ValueError(f"image dims must be >= 1, got ({height}, {width})")
    px = (torch.arange(width, dtype=dtype, device=device) + 0.5) / width
    py = (torch.arange(height, dtype=dtype, device=device) + 0.5) / height
    rows, cols = torch.meshgrid(py, px, indexing="ij")
    return torch.stack([cols, rows], dim=-1).reshape(-1, 2)


# ---------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------


def pool_id(position: Sequence[float], grid_size: int = 4) -> int:
    """Map a 2D position in [0, 1]² to a pool ID in [0, grid_size²).

    The pool partition is the same for donor and recipient under the
    pool-matched absorption protocol; the position is clamped just
    below 1.0 to keep the upper boundary in the last pool (otherwise
    a cell exactly at 1.0 would index out of range).
    """
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1, got {grid_size}")
    x = max(0.0, min(0.999999, float(position[0])))
    y = max(0.0, min(0.999999, float(position[1])))
    col = int(x * grid_size)
    row = int(y * grid_size)
    return row * grid_size + col


def pool_centroid(
    pool_id_int: int, grid_size: int = 4,
) -> Tuple[float, float]:
    """Inverse of `pool_id`: return the (cx, cy) centroid of the pool."""
    if not (0 <= pool_id_int < grid_size * grid_size):
        raise ValueError(
            f"pool_id_int {pool_id_int} out of range [0, {grid_size**2})"
        )
    row = pool_id_int // grid_size
    col = pool_id_int % grid_size
    cx = (col + 0.5) / grid_size
    cy = (row + 0.5) / grid_size
    return cx, cy


# ---------------------------------------------------------------------
# LCN masks
# ---------------------------------------------------------------------


def build_lcn_mask(
    out_positions: torch.Tensor,
    in_positions: torch.Tensor,
    *,
    mode: str = "soft",
    sigma: float = 0.25,
    k: int = 8,
) -> torch.Tensor:
    """Build an (n_out, n_in) locality mask in [0, 1].

    Positions are 2D (x, y in [0, 1]²); higher dimensions are accepted
    and reduced via squared Euclidean distance in all coordinates.

    mode="soft": exp(-d² / 2σ²). Real-valued falloff — cells in the
        window contribute proportionally to their distance. σ ≈ 0.25
        gives a window roughly a quarter of the unit square wide.

    mode="hard": top-K nearest input cells per output cell get mask=1,
        all others get 0. K is clamped to [1, n_in].

    The returned tensor lives on `out_positions.device` with float32
    dtype unless caller casts. The same builder is used for L0 (in =
    pixel positions) and L1 (in = L0 cell positions).
    """
    if mode not in ("soft", "hard"):
        raise ValueError(
            f"mode must be 'soft' or 'hard', got {mode!r}"
        )
    if out_positions.ndim != 2 or in_positions.ndim != 2:
        raise ValueError(
            f"positions must be 2-D, got out={out_positions.shape}, "
            f"in={in_positions.shape}"
        )
    if out_positions.shape[1] != in_positions.shape[1]:
        raise ValueError(
            f"position dims mismatch: out={out_positions.shape[1]}, "
            f"in={in_positions.shape[1]}"
        )
    out_pos = out_positions.to(dtype=torch.float32)
    in_pos = in_positions.to(dtype=torch.float32, device=out_pos.device)
    d2 = ((out_pos.unsqueeze(1) - in_pos.unsqueeze(0)) ** 2).sum(-1)
    if mode == "soft":
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        return torch.exp(-d2 / (2.0 * sigma * sigma))
    # hard top-K
    k_eff = max(1, min(int(k), d2.shape[1]))
    _, idx = torch.topk(-d2, k_eff, dim=1)
    mask = torch.zeros_like(d2)
    mask.scatter_(1, idx, 1.0)
    return mask


def locality_metric(
    W: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float = 0.1,
) -> float:
    """Locality = in-window |W| mass / total |W| mass.

    "In-window" is defined as `mask > threshold`; for soft Gaussian
    masks this captures the visible window without including the long
    tail. For hard masks the threshold is irrelevant (mask ∈ {0, 1}).

    Returns a float in [0, 1] — higher = more locality. 0.5 is a
    diffuse layer (mass split evenly); 0.9+ is tightly local. Used as
    the regression sentinel for pool-matched absorption: locality
    should INCREASE post-absorb when both donor and recipient share
    the LCN convention.
    """
    if W.shape != mask.shape:
        raise ValueError(
            f"W {tuple(W.shape)} and mask {tuple(mask.shape)} must match"
        )
    total = W.detach().abs().sum().item()
    if total <= 0.0:
        return 0.0
    in_win = (mask.detach() > threshold).to(W.dtype)
    in_mass = (W.detach().abs() * in_win).sum().item()
    return float(in_mass / total)


__all__ = [
    "grid_positions_2d",
    "pixel_positions_2d",
    "pool_id",
    "pool_centroid",
    "build_lcn_mask",
    "locality_metric",
]
