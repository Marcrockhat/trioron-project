"""Chained-15 + LCN perception + Axis 6 emergence + credit-based freezing.

Three modes (controlled by AXIS6_MODE_RUN env var):
  - native  (default): train all 15 tasks from scratch with credit-based freezing
  - extend            : train tasks 0..9 → freeze L1 substrate → extend with 10..14
  - absorb            : train two donors on overlapping subsets, api.absorb fuse

Architecture:
  784  → L0 (128, LCN-masked random projection, FROZEN)
       → L1 (16, growable via Axis 6, credit-frozen per cell)
       → head (30, gradient-isolated per task via outs[:, task.global_classes])

Perception (L0): 16×8 retinotopic grid on image [0,1]² with Gaussian-overlap
receptive fields (σ=0.10, ~50% overlap between adjacent cells). L0 outputs
have meaningful positional identity — cell i = "weighted activity in region
X of the image".

Substrate (L1): credit on a cell = (engagement_frac > AXIS6_CREDIT_THR)
where engagement_frac is the mean per-input activation rate of the cell
over the task's training data. Credited cells get L1.W[i,:] and L1.b[i]
gradient masks for all subsequent tasks. Axis 6 field_conditional_growth
spawns new cells where epi_A is hot; fresh cells start uncredited =
fully plastic.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron import masked_cross_entropy
from experiments.datasets import (
    DatasetBundle, TaskDataView, chained_15_specs, build_task_views,
    DEFAULT_DATA_ROOT,
)
from experiments.bench_chained_15task import build_lcn_mask, _lcn_cell_centers


# ----------------------------------------------------------------------
# Architecture constants
# ----------------------------------------------------------------------
INPUT_DIM = 784
L0_WIDTH = 128
LCN_GRID_X = 16
LCN_GRID_Y = 8
LCN_SIGMA = 0.10
H_INIT = int(os.environ.get("AXIS6_H_INIT", "16"))
N_CLASSES = 30
# Cosine-head temperature: scales the [-1, 1] cosine similarities into
# CE-friendly logits. Higher = sharper softmax. 16 is a common choice
# (matches CosFace / ArcFace conventions for shallow heads).
COSINE_TEMPERATURE = float(os.environ.get("AXIS6_COS_TEMP", "16.0"))

# LCN-on-L1: each L1 cell at position p reads only from L0 cells near p.
# Co-designed with diffusion-of-W: with shared input topology across
# position-neighbors, diffusion of W smooths same-aligned local filters
# → emergent locally-connected proto-conv. Three modes:
#   "off"  : no mask (full read).
#   "soft" : Gaussian falloff in image-position space (mirrors L0 LCN).
#   "hard" : binary top-k window — each L1 cell reads from its K
#            nearest L0 cells, rest pinned at zero.
# Sigma default 0.25 = L1 cell spacing on the 4×4 grid (~50% overlap
# between adjacent windows, matches the L0 "1× spacing" convention).
L1_LCN_MODE = os.environ.get("AXIS6_L1_LCN_MODE", "off").lower()
L1_LCN_SIGMA = float(os.environ.get("AXIS6_L1_LCN_SIGMA", "0.25"))
L1_LCN_K = int(os.environ.get("AXIS6_L1_LCN_K", "8"))
if L1_LCN_MODE not in {"off", "soft", "hard"}:
    raise ValueError(
        f"AXIS6_L1_LCN_MODE must be one of off/soft/hard, got {L1_LCN_MODE!r}"
    )

# Hebbian (activity-driven) kernel for diffusion-of-W: replaces the
# Gaussian-over-hand-coded-position kernel with one derived from
# activation correlation between L1 cells. Cells that fire together on
# similar inputs become diffusion neighbors → their W rows smooth toward
# each other → emergent weight-sharing → conv-like organization,
# *earned by activity* rather than declared. Preserves the absorption
# invariant (no positional commitments — any cell can accept) and the
# substrate emergence principle (structure derives from what cells do,
# not from hand-coded coordinates).
#
# Modes:
#   "off"     : use Axis 6 Gaussian-over-cell_position kernel (default,
#               legacy behavior).
#   "softmax" : EMA of cosine similarity between L1 activation profiles,
#               row-softmaxed at use time with diagonal zeroed.
#   "topk"    : same EMA + cosine, but each row keeps only its K most-
#               correlated neighbors as a binary kernel (then normalized).
# Mutually exclusive with AXIS6_L1_LCN_MODE — the kernel and the read-
# mask are independent mechanisms but the comparison is muddled if both
# fire, so the smoke run gate insists on at most one being non-off.
HEBBIAN_MODE = os.environ.get("AXIS6_HEBBIAN", "off").lower()
HEBBIAN_DECAY = float(os.environ.get("AXIS6_HEBBIAN_DECAY", "0.9"))
HEBBIAN_K = int(os.environ.get("AXIS6_HEBBIAN_K", "4"))
HEBBIAN_TEMP = float(os.environ.get("AXIS6_HEBBIAN_TEMP", "1.0"))
if HEBBIAN_MODE not in {"off", "softmax", "topk"}:
    raise ValueError(
        f"AXIS6_HEBBIAN must be one of off/softmax/topk, got {HEBBIAN_MODE!r}"
    )
if HEBBIAN_MODE != "off" and L1_LCN_MODE != "off":
    raise RuntimeError(
        "AXIS6_HEBBIAN and AXIS6_L1_LCN_MODE are mutually exclusive: "
        "both are alternative answers to 'how do L1 cells decide who their "
        "neighbors are.' Pick one for a clean comparison."
    )

# Finite-space mechanics: trioron's flat substrate has positions but no
# mechanical conservation law. Spawn places a new cell at parent + jitter
# and no existing cell moves. Biology's finite-space constraint (skull,
# meristem disk) makes spawn DISPLACE neighbors, which packs cells into
# distinct positions over time. These helpers add that mechanic — after
# every axis6_spawn, run a few iterations of pairwise repulsion so any
# two cells within min_sep push apart, then clamp positions back into a
# bounding region. Cells now have to share finite space; spawn has a
# positional cost; high-density regions force later spawns elsewhere.
# Env-gated; default off preserves existing behavior.
FINITE_SPACE_ENABLED = os.environ.get("AXIS6_FINITE_SPACE", "0") == "1"
REPULSE_STEPS = int(os.environ.get("AXIS6_REPULSE_STEPS", "8"))
REPULSE_MIN_SEP = float(os.environ.get("AXIS6_REPULSE_MIN_SEP", "0.08"))
REPULSE_STRENGTH = float(os.environ.get("AXIS6_REPULSE_STRENGTH", "0.4"))
BOUND_REGION = os.environ.get("AXIS6_BOUND_REGION", "unit_box").lower()
REPULSE_DIMS = int(os.environ.get("AXIS6_REPULSE_DIMS", "2"))
if BOUND_REGION not in {"none", "unit_box", "unit_sphere"}:
    raise ValueError(
        f"AXIS6_BOUND_REGION must be none/unit_box/unit_sphere, "
        f"got {BOUND_REGION!r}"
    )
if REPULSE_DIMS not in {2, 3}:
    raise ValueError(f"AXIS6_REPULSE_DIMS must be 2 or 3, got {REPULSE_DIMS}")

# Boquila aux loss: rewards substrate ORGANIZATION instead of only
# output accuracy. Activation-correlated cells get pulled toward each
# other in position space — substrate layout emerges from substrate
# activity (Boquila vine reads its leaf shape from host morphology;
# trioron reads its substrate shape from its own activation profile).
#
#   L_align = mean over (i, j) of corr_ema[i, j] * dist²(p_i, p_j)
#
# Where corr_ema is the same activation-correlation EMA the Hebbian
# kernel uses (constant w.r.t. positions), and dist is differentiable
# through cell_position_learnable. Gradient pulls cells together;
# finite-space repulsion pushes them apart; equilibrium is a
# topographic map where layout reflects functional clustering.
#
# Requires: cells need learnable positions (cell_position in node.py
# is a buffer, so we install a parallel Parameter on L1 and copy it
# back to the buffer after every opt.step so the rest of the substrate
# sees the migrated positions). Env-gated, default off.
ALIGN_ENABLED = os.environ.get("AXIS6_ALIGN", "0") == "1"
ALIGN_WEIGHT = float(os.environ.get("AXIS6_ALIGN_WEIGHT", "0.01"))
ALIGN_DIMS = int(os.environ.get("AXIS6_ALIGN_DIMS", "2"))
if ALIGN_DIMS not in {2, 3}:
    raise ValueError(f"AXIS6_ALIGN_DIMS must be 2 or 3, got {ALIGN_DIMS}")


def diffused_l1_weight(
    L1,
    lambda_diff: float,
    credit_mask: Optional[torch.Tensor] = None,
    W_snapshot: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute L1's effective weight matrix under spatial diffusion + LCN.

      W_eff = ((1-λ) · W + λ · (kernel @ W))  ⊙  W_lcn_mask

    where `kernel` is the Gaussian-over-cell_position from Axis 6 field
    machinery, and `W_lcn_mask` (optional, on L1) enforces input-side
    positional locality. The mask is applied AFTER diffusion so that
    diffusion can mix neighbor rows but the effective read stays inside
    each cell's window — preserves the "L1 cells read positionally-
    local L0 cells" prior even when neighbors smear their weights.

    If a credit_mask + W_snapshot are provided, credited cells use the
    snapshot (their W_eff at credit-grant time) instead of the live
    diffusion+mask. This decouples credited cells from later neighbor
    changes: their feature identity stays locked even as uncredited
    neighbors learn.
    """
    n = L1.n_nodes
    lcn = getattr(L1, "W_lcn_mask", None)
    if lambda_diff <= 0.0 and lcn is None:
        # Degenerate: no diffusion, no mask → return W directly.
        return L1.W
    if lambda_diff > 0.0:
        # Kernel source: Hebbian (activity-driven) if armed and seeded,
        # otherwise fall back to Axis 6 Gaussian-over-cell_position.
        # Hebbian returns None when not yet seeded (first-batch warmup),
        # which auto-falls-through to position kernel — clean cold-start.
        kernel = hebbian_kernel(L1)
        if kernel is None:
            if (L1._field_kernel is None
                    or L1._field_kernel.shape[0] != n):
                L1._field_kernel = L1._compute_field_kernel()
            kernel = L1._field_kernel
        kernel = kernel.to(device=L1.W.device, dtype=L1.W.dtype)
        diffused = kernel @ L1.W
        W_eff_live = (1.0 - lambda_diff) * L1.W + lambda_diff * diffused
    else:
        # λ=0 but mask is on — diffusion is a no-op, just apply the mask.
        W_eff_live = L1.W
    if lcn is not None:
        # Mask may be stale-shaped if extend_l1_lcn_mask hasn't fired yet
        # for a fresh spawn. Pad with zeros (= no read) for missing rows
        # so the multiply is safe; extend will overwrite next opportunity.
        if lcn.shape[0] < n:
            pad = torch.zeros(
                n - lcn.shape[0], lcn.shape[1],
                dtype=lcn.dtype, device=lcn.device,
            )
            lcn = torch.cat([lcn, pad], dim=0)
        W_eff_live = W_eff_live * lcn.to(device=W_eff_live.device,
                                         dtype=W_eff_live.dtype)
    if (credit_mask is not None
            and W_snapshot is not None
            and credit_mask.numel() == n
            and credit_mask.any()):
        # Pad snapshot if L1 has grown since last sync (new rows = uncredited
        # so they'd never be selected by the where).
        if W_snapshot.shape[0] < n:
            pad = torch.zeros(
                n - W_snapshot.shape[0], W_snapshot.shape[1],
                dtype=W_snapshot.dtype, device=W_snapshot.device,
            )
            W_snapshot = torch.cat([W_snapshot, pad], dim=0)
        mask_2d = credit_mask.unsqueeze(1).to(device=L1.W.device)
        return torch.where(mask_2d, W_snapshot.to(device=L1.W.device), W_eff_live)
    return W_eff_live


def l1_forward_diffused(
    L1,
    h0: torch.Tensor,
    lambda_diff: float,
    credit_mask: Optional[torch.Tensor] = None,
    W_snapshot: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 forward with diffused effective weight. Mirrors TrioronLayer
    point-neuron forward (matmul + bias + ReLU), plus _last_y caching
    and backward hook for internal_stress (the field signal stays alive).
    """
    if lambda_diff <= 0.0:
        # No diffusion → standard forward.
        return L1(h0)
    W_eff = diffused_l1_weight(L1, lambda_diff, credit_mask, W_snapshot)
    y_pre = h0 @ W_eff.t() + L1.b
    if L1.activation == "relu":
        y = F.relu(y_pre)
    elif L1.activation == "tanh":
        y = torch.tanh(y_pre)
    else:
        y = y_pre
    # Same hook caching as TrioronLayer.forward, so internal_stress works.
    if y.requires_grad:
        L1._last_y = y.detach()
        y.register_hook(L1._capture_upstream)
    return y


def cosine_logits(
    l1_features: torch.Tensor,
    head_W: torch.Tensor,
    temperature: float = COSINE_TEMPERATURE,
) -> torch.Tensor:
    """Delegate to trioron.manifold.cosine_logits.

    Kept here as a thin wrapper so the env-tunable
    ``AXIS6_COS_TEMP`` default still applies for chained-15 callers
    who don't pass an explicit temperature.
    """
    from trioron.manifold import cosine_logits as _cosine
    return _cosine(l1_features, head_W, temperature=temperature)


# ----------------------------------------------------------------------
# LCN-on-L1 mask: positional locality on the input-read side of L1
# ----------------------------------------------------------------------


def _l1_lcn_rows(
    l1_positions: torch.Tensor,
    l0_positions: torch.Tensor,
    mode: str,
    sigma: float,
    k: int,
) -> torch.Tensor:
    """Return (n_l1, n_l0) mask using 2D distance in image position space.

    "soft" → exp(-d² / 2σ²). "hard" → 1 for each cell's k nearest L0
    cells, 0 elsewhere. Both rows live in [0, 1].
    """
    # (n_l1, n_l0) squared distance in xy.
    d2 = ((l1_positions.unsqueeze(1) - l0_positions.unsqueeze(0)) ** 2).sum(-1)
    if mode == "soft":
        return torch.exp(-d2 / (2.0 * sigma ** 2))
    if mode == "hard":
        k_eff = max(1, min(k, d2.shape[1]))
        # topk on -d2 → smallest distances → nearest neighbors.
        _, idx = torch.topk(-d2, k_eff, dim=1)
        mask = torch.zeros_like(d2)
        mask.scatter_(1, idx, 1.0)
        return mask
    raise ValueError(f"unsupported L1 LCN mode: {mode!r}")


def build_l1_lcn_mask(
    L1,
    L0,
    mode: str = L1_LCN_MODE,
    sigma: float = L1_LCN_SIGMA,
    k: int = L1_LCN_K,
) -> torch.Tensor:
    """(L1.n_nodes, L0.n_nodes) mask in L1.W dtype/device."""
    l1_pos = L1.cell_position[: L1.n_nodes, :2].detach().to(dtype=torch.float32)
    l0_pos = L0.cell_position[: L0.n_nodes, :2].detach().to(dtype=torch.float32)
    mask = _l1_lcn_rows(l1_pos, l0_pos, mode=mode, sigma=sigma, k=k)
    return mask.to(device=L1.W.device, dtype=L1.W.dtype)


def install_l1_lcn_mask(
    net: TrioronNetwork,
    mode: str = L1_LCN_MODE,
    sigma: float = L1_LCN_SIGMA,
    k: int = L1_LCN_K,
    l1_layer_idx: int = 1,
    l0_layer_idx: int = 0,
) -> None:
    """Install LCN-on-L1 via the substrate API.

    Thin wrapper around ``TrioronLayer.enable_lcn`` — pulls input
    positions from L0's ``cell_position`` (the upstream layer's image-
    plane coordinates) and delegates. The substrate caches
    ``lcn_in_positions`` so subsequent ``grow_node`` calls extend the
    mask automatically; legacy callers that hit ``extend_l1_lcn_mask``
    explicitly remain compatible (the substrate-level extend is
    idempotent on already-extended rows).

    No-op when mode='off'.
    """
    if mode == "off":
        return
    L1 = net.layers[l1_layer_idx]
    L0 = net.layers[l0_layer_idx]
    in_positions = L0.cell_position[: L0.n_nodes, :2].detach().clone()
    L1.enable_lcn(
        in_positions, mode=mode, sigma=float(sigma), k=int(k),
    )


def reapply_l1_lcn_mask(L1) -> None:
    """Multiply L1.W by W_lcn_mask in-place. Call after every optimizer
    step that touches L1 so off-window entries don't accumulate."""
    mask = getattr(L1, "W_lcn_mask", None)
    if mask is None:
        return
    with torch.no_grad():
        L1.W.data.mul_(mask)


def extend_l1_lcn_mask(
    net: TrioronNetwork,
    l1_layer_idx: int = 1,
    l0_layer_idx: int = 0,
) -> None:
    """Delegate to substrate-level extension.

    ``TrioronLayer.extend_lcn_mask`` already fires automatically inside
    ``grow_node`` (and therefore ``axis6_spawn`` and any other grow
    path). Calling this explicitly is idempotent — when shapes match
    it's a no-op. Kept for back-compat with legacy training loops that
    invoke it after every spawn event.
    """
    net.layers[l1_layer_idx].extend_lcn_mask()


# ----------------------------------------------------------------------
# Hebbian (activity-driven) kernel — alternative to Gaussian-over-position
# ----------------------------------------------------------------------


def _ensure_act_corr_buffer(L1) -> None:
    """Lazily allocate (n_nodes, n_nodes) EMA buffer for activation
    correlation, plus a 'has data yet' flag. Idempotent."""
    n = L1.n_nodes
    buf = getattr(L1, "_act_corr_ema", None)
    if buf is None or buf.shape[0] != n:
        L1.register_buffer(
            "_act_corr_ema",
            torch.zeros(n, n, device=L1.W.device, dtype=torch.float32),
        )
        L1._act_corr_initialized = False


def update_activation_correlation(
    L1,
    h1_batch: torch.Tensor,
    decay: float = HEBBIAN_DECAY,
) -> None:
    """EMA update: cosine similarity between L1 cells' per-batch
    activation profiles. h1_batch is (B, n_nodes) post-ReLU.

    cor[i, j] = mean over batch of normalized(h1[:, i]) · normalized(h1[:, j])
    Stored as L1._act_corr_ema with exponential decay:
        ema ← decay · ema + (1 - decay) · cor
    First call seeds ema directly (no smoothing toward zero)."""
    _ensure_act_corr_buffer(L1)
    n = L1.n_nodes
    if h1_batch.shape[1] != n:
        # Stale shape (mid-spawn). Skip — extend will reseed on next pass.
        return
    h = h1_batch.detach().to(dtype=torch.float32)
    # Center? No — we want raw co-activation, so cosine on raw activations
    # gives a [-1, 1] score that's positive when cells co-fire and zero
    # when they're orthogonal across the batch.
    norm = h.norm(dim=0, keepdim=True).clamp(min=1e-8)
    h_n = h / norm                                       # (B, n)
    cor = h_n.t() @ h_n                                  # (n, n)
    if not getattr(L1, "_act_corr_initialized", False):
        L1._act_corr_ema.copy_(cor)
        L1._act_corr_initialized = True
    else:
        L1._act_corr_ema.mul_(decay).add_(cor, alpha=(1.0 - decay))


def hebbian_kernel(
    L1,
    mode: str = HEBBIAN_MODE,
    k: int = HEBBIAN_K,
    temp: float = HEBBIAN_TEMP,
) -> Optional[torch.Tensor]:
    """Convert the EMA correlation buffer into a row-stochastic kernel
    suitable for `kernel @ W` diffusion. Returns None when mode='off'
    or when the buffer hasn't been seeded yet.

    Diagonal is always zeroed (a cell doesn't smooth toward itself —
    that's already covered by the (1-λ)·W term in the diffusion update)."""
    if mode == "off":
        return None
    ema = getattr(L1, "_act_corr_ema", None)
    if ema is None or not getattr(L1, "_act_corr_initialized", False):
        return None
    n = L1.n_nodes
    if ema.shape[0] != n:
        return None
    cor = ema.clone()
    cor.fill_diagonal_(0.0)
    if mode == "softmax":
        # Row-softmax with temperature. Negative correlations get
        # weak weight; positive correlations dominate.
        return torch.softmax(cor / max(temp, 1e-8), dim=1).to(
            device=L1.W.device, dtype=L1.W.dtype,
        )
    if mode == "topk":
        k_eff = max(1, min(k, n - 1))
        # Pick top-k correlations per row; everything else → 0.
        _, idx = torch.topk(cor, k_eff, dim=1)
        ker = torch.zeros_like(cor)
        ker.scatter_(1, idx, 1.0)
        # Row-normalize so each row sums to 1 (so diffusion is a
        # convex combination, no scale drift).
        row_sum = ker.sum(dim=1, keepdim=True).clamp(min=1e-8)
        ker = ker / row_sum
        return ker.to(device=L1.W.device, dtype=L1.W.dtype)
    raise ValueError(f"unsupported HEBBIAN mode {mode!r}")


def extend_act_corr_on_spawn(L1, parent_idx: int, new_idx: int) -> None:
    """After axis6_spawn grows L1, extend the activation-correlation
    EMA buffer. New cell inherits parent's row/column as a warm start
    (since axis6_spawn copies parent W + adds jitter — the new cell IS
    a noisy copy of parent at spawn time, so it should diffuse toward
    parent's neighbors)."""
    buf = getattr(L1, "_act_corr_ema", None)
    if buf is None:
        return
    n_new = L1.n_nodes
    n_old = buf.shape[0]
    if n_new <= n_old:
        return
    # Pad to new shape.
    new_buf = torch.zeros(n_new, n_new, device=buf.device, dtype=buf.dtype)
    new_buf[:n_old, :n_old] = buf
    if 0 <= parent_idx < n_old and n_old <= new_idx < n_new:
        # Inherit parent's correlations for the new cell. Then the
        # parent and new cell start strongly correlated with each other
        # (since they ARE near-copies) — set their pairwise entry to 1.
        new_buf[new_idx, :n_old] = buf[parent_idx, :n_old]
        new_buf[:n_old, new_idx] = buf[:n_old, parent_idx]
        new_buf[parent_idx, new_idx] = 1.0
        new_buf[new_idx, parent_idx] = 1.0
        new_buf[new_idx, new_idx] = 1.0
    del L1._buffers["_act_corr_ema"]
    L1.register_buffer("_act_corr_ema", new_buf)


# ----------------------------------------------------------------------
# Finite-space mechanics: spawn DISPLACES neighbors, positions bounded
# ----------------------------------------------------------------------


def relax_positions_step(
    positions: torch.Tensor,
    min_sep: float,
    strength: float,
    dims: int,
) -> torch.Tensor:
    """One iteration of pairwise repulsive relaxation. positions
    shape: (n, position_dim). Only the first `dims` columns participate
    in the repulsion — the rest (typically z) is held rigid. Returns
    updated positions, same shape."""
    n = positions.shape[0]
    if n < 2:
        return positions
    pos2d = positions[:, :dims]                        # (n, d)
    diff = pos2d.unsqueeze(1) - pos2d.unsqueeze(0)     # (n, n, d): i - j
    dist = diff.norm(dim=-1)                            # (n, n)
    # Avoid div-by-zero anywhere: clamp instead of adding eye, so two
    # cells at identical positions don't NaN the direction computation.
    # (The diagonal IS zero distance — but we zero out diagonal overlap
    # below so direction at i==j multiplies by zero anyway.) If two
    # off-diagonal cells coincide, push direction defaults to zero
    # (diff is 0), which is the right behavior — they'll separate as
    # soon as any other repulsion or alignment moves either of them.
    dist_safe = dist.clamp(min=1e-6)
    overlap = (min_sep - dist).clamp(min=0.0)           # (n, n) positive where too close
    overlap.fill_diagonal_(0.0)                          # no self-push
    direction = diff / dist_safe.unsqueeze(-1)          # (n, n, d) unit vector j → i
    push = (overlap.unsqueeze(-1) * direction).sum(dim=1)  # (n, d)
    pos2d_new = pos2d + strength * push
    new_positions = positions.clone()
    new_positions[:, :dims] = pos2d_new
    return new_positions


def clamp_to_region(
    positions: torch.Tensor,
    mode: str,
    dims: int,
) -> torch.Tensor:
    """Clamp first `dims` columns to a bounding region. 'unit_box':
    [0, 1]^d. 'unit_sphere': ball of radius 0.5 centered at 0.5. 'none':
    no clamp (substrate unbounded; equivalent to no skull)."""
    if mode == "none":
        return positions
    new_positions = positions.clone()
    if mode == "unit_box":
        new_positions[:, :dims] = new_positions[:, :dims].clamp(0.0, 1.0)
    elif mode == "unit_sphere":
        center = 0.5
        radius = 0.5
        offset = new_positions[:, :dims] - center
        norm = offset.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.where(norm > radius, radius / norm, torch.ones_like(norm))
        new_positions[:, :dims] = center + offset * scale
    return new_positions


def relax_after_spawn(
    L1,
    n_steps: int = REPULSE_STEPS,
    min_sep: float = REPULSE_MIN_SEP,
    strength: float = REPULSE_STRENGTH,
    mode: str = BOUND_REGION,
    dims: int = REPULSE_DIMS,
) -> float:
    """Run `n_steps` iterations of pairwise-repulsion + clamp on the
    LIVE positions (all cells, not just the newly-spawned one — biology's
    pressure propagates from the new cell outward through the substrate).
    Invalidates the field kernel cache so the next forward picks up the
    new positions. Returns max displacement (any cell, any axis) for
    diagnostics."""
    if n_steps <= 0:
        return 0.0
    with torch.no_grad():
        n = L1.n_nodes
        positions = L1.cell_position[:n].clone()
        original = positions.clone()
        for _ in range(n_steps):
            positions = relax_positions_step(positions, min_sep, strength, dims)
            positions = clamp_to_region(positions, mode, dims)
        L1.cell_position[:n] = positions
        max_displacement = float((positions - original).abs().max().item())
    L1._field_kernel = None
    return max_displacement


# ----------------------------------------------------------------------
# Boquila alignment: learnable positions + activation-position aux loss
# ----------------------------------------------------------------------


def install_learnable_positions(L1) -> None:
    """Register cell_position_learnable as a Parameter on L1, initialized
    from the current cell_position buffer. After every opt.step we copy
    the .data back to cell_position so the rest of the substrate (Axis 6,
    Hebbian kernel, LCN mask, repulsion) sees migrated positions. Idempotent.
    """
    if hasattr(L1, "cell_position_learnable"):
        return
    init = L1.cell_position[: L1.n_nodes].clone().detach()
    import torch.nn as nn
    L1.cell_position_learnable = nn.Parameter(init)


def extend_learnable_positions(L1) -> None:
    """After axis6_spawn extends L1, pad cell_position_learnable with
    rows for the new cells (taken from the substrate's cell_position
    buffer, which axis6_spawn already populated with parent+jitter)."""
    learn = getattr(L1, "cell_position_learnable", None)
    if learn is None:
        return
    n_new = L1.n_nodes
    n_old = learn.shape[0]
    if n_new <= n_old:
        return
    import torch.nn as nn
    new_rows = L1.cell_position[n_old:n_new].clone().detach()
    new_param = torch.cat([learn.data, new_rows], dim=0)
    # Re-register as a fresh Parameter; the caller is expected to rebuild
    # the optimizer (same pattern as W after grow_node).
    L1.cell_position_learnable = nn.Parameter(new_param)


def sync_positions_to_buffer(L1) -> None:
    """Copy the learnable parameter's data back into the cell_position
    buffer so Axis 6 / Hebbian / LCN / repulsion all see the updated
    positions. Also invalidates the field kernel cache since positions
    changed. Call after every opt.step that touched cell_position_learnable.
    """
    learn = getattr(L1, "cell_position_learnable", None)
    if learn is None:
        return
    with torch.no_grad():
        n = min(learn.shape[0], L1.cell_position.shape[0])
        L1.cell_position[:n] = learn.data[:n]
    L1._field_kernel = None


def alignment_aux_loss(L1, dims: int = ALIGN_DIMS) -> torch.Tensor:
    """Boquila aux loss: pull activation-correlated cells toward each
    other in position space.

      L_align = mean over (i, j) of corr_ema[i, j] · dist²(p_i, p_j)

    The correlation EMA is detached (constant w.r.t. positions); the
    gradient flows only through cell_position_learnable. Zeroes the
    diagonal so a cell isn't pulled toward itself.

    Returns 0.0 (no grad) when the Hebbian EMA hasn't been seeded yet
    (first batch), or when positions aren't learnable. Caller is
    expected to add this scaled by ALIGN_WEIGHT to the total loss."""
    learn = getattr(L1, "cell_position_learnable", None)
    if learn is None:
        return torch.tensor(0.0)
    corr = getattr(L1, "_act_corr_ema", None)
    if corr is None or not getattr(L1, "_act_corr_initialized", False):
        return torch.tensor(0.0)
    n = L1.n_nodes
    if learn.shape[0] != n or corr.shape[0] != n:
        return torch.tensor(0.0)
    pos = learn[:, :dims]                               # (n, d)
    diff = pos.unsqueeze(1) - pos.unsqueeze(0)          # (n, n, d)
    dist2 = (diff ** 2).sum(dim=-1)                     # (n, n)
    weight = corr.detach().to(device=dist2.device, dtype=dist2.dtype)
    weight = weight - torch.diag(torch.diag(weight))    # zero self
    # Use only positive correlations — negative would PUSH cells apart
    # via the position loss, but the finite-space repulsion already does
    # that. Keep the alignment loss strictly attractive.
    weight = weight.clamp(min=0.0)
    n_pairs = max(1, n * (n - 1))
    return (weight * dist2).sum() / n_pairs


# ----------------------------------------------------------------------
# Build a 3-layer trioron with frozen L0 (LCN or random) + growable L1 + head
# ----------------------------------------------------------------------
def build_net(
    seed: int,
    lcn_enabled: bool = True,
    l1_lcn_mode: str = L1_LCN_MODE,
    l1_lcn_sigma: float = L1_LCN_SIGMA,
    l1_lcn_k: int = L1_LCN_K,
) -> TrioronNetwork:
    torch.manual_seed(seed)
    net = TrioronNetwork([
        (INPUT_DIM, L0_WIDTH, "relu"),    # L0: perception
        (L0_WIDTH, H_INIT, "relu"),       # L1: growable substrate
        (H_INIT, N_CLASSES, "linear"),    # Head
    ])
    L0 = net.layers[0]
    L1 = net.layers[1]

    # Either LCN-masked (locally-connected with Gaussian receptive fields)
    # or pure random projection (no spatial inductive bias). Either way,
    # L0 is frozen — only L1 + head train.
    if lcn_enabled:
        mask = build_lcn_mask(INPUT_DIM, L0_WIDTH, LCN_SIGMA).to(
            device=L0.W.device, dtype=L0.W.dtype,
        )
        L0.register_buffer("W_lcn_mask", mask)
        with torch.no_grad():
            L0.W.data.mul_(mask)
            if hasattr(L0, "W_anchor"):
                L0.W_anchor.data.mul_(mask.to(L0.W_anchor.dtype))
    L0.W.requires_grad_(False)
    L0.b.requires_grad_(False)
    if hasattr(L0, "branch_weight"):
        L0.branch_weight.requires_grad_(False)

    # L0 cell positions live in image-space [0,1]² (third dim padded to 0).
    # Even without LCN we keep grid positions for diagnostic consistency.
    with torch.no_grad():
        l0_xy = _lcn_cell_centers()  # (128, 2)
        L0.cell_position[:, 0] = l0_xy[:, 0]
        L0.cell_position[:, 1] = l0_xy[:, 1]
        L0.cell_position[:, 2] = 0.0

    # Cosine head: head.b is unused (cosine logits have no bias term).
    # Freeze it at zero so it can't drift via any leftover gradient path.
    head = net.layers[2]
    with torch.no_grad():
        head.b.data.zero_()
    head.b.requires_grad_(False)

    # L1 cell positions: regular grid in image [0,1]² space. For H_init=16
    # → 4×4. This gives diffusion neighbors a meaningful structure:
    # close-in-position L1 cells are responsible for adjacent image
    # regions, so kernel-diffused weights effectively form a learned
    # local filter aligned with image space (conv by emergence).
    n = L1.n_nodes
    side = max(1, int(math.ceil(math.sqrt(n))))
    cx = (torch.arange(side).float() + 0.5) / side
    cy = (torch.arange(side).float() + 0.5) / side
    grid_x, grid_y = torch.meshgrid(cx, cy, indexing="ij")
    positions = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)[:n]
    with torch.no_grad():
        L1.cell_position[:n, 0] = positions[:, 0]
        L1.cell_position[:n, 1] = positions[:, 1]
        L1.cell_position[:n, 2] = 0.0

    # LCN-on-L1 (co-design with diffusion-of-W). Must come AFTER L1
    # cell_position is set so distances are computed in image space.
    # No-op when l1_lcn_mode='off'.
    install_l1_lcn_mask(
        net, mode=l1_lcn_mode, sigma=l1_lcn_sigma, k=l1_lcn_k,
    )

    # Boquila alignment: install learnable position parameter so cells
    # can migrate under aux-loss gradient + finite-space repulsion.
    # No-op when AXIS6_ALIGN=0.
    if ALIGN_ENABLED:
        install_learnable_positions(L1)

    return net


# ----------------------------------------------------------------------
# Manifold replay store — promoted to trioron.manifold.
# ----------------------------------------------------------------------
# ``ManifoldStore`` now lives in ``trioron/manifold.py``; this re-export
# preserves the experiments-level import path. ``store_task`` accepts
# any object with an ``all_examples() -> (X, y)`` method (TaskDataView
# qualifies), via duck typing in the promoted implementation.
from trioron.manifold import ManifoldStore  # noqa: E402, F401


# ----------------------------------------------------------------------
# Train one task
# ----------------------------------------------------------------------
@dataclass
class TaskTrainResult:
    events: List[Tuple[int, str]]
    engagement_frac: Optional[torch.Tensor]
    train_loss_final: float
    train_acc_final: float


def train_one_task(
    net: TrioronNetwork,
    view: TaskDataView,
    task_global_classes: List[int],
    n_epochs: int,
    batch_size: int,
    lr: float,
    axis6: bool,
    diff_floor: float,
    diff_b_thr: float,
    field_sigma: float,
    field_dt: float,
    stress_tolerance: float,
    spawn_cap: int,
    cooldown_steps: int,
    l1_credit_mask: Optional[torch.Tensor],
    track_engagement: bool,
    manifold: Optional[ManifoldStore] = None,
    lambda_manifold: float = 1.0,
    manifold_total_batch: int = 64,
    manifold_noise_scale: float = 1.0,
    seen_classes_global: Optional[Sequence[int]] = None,
    lambda_diff: float = 0.0,
    W_snapshot: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> TaskTrainResult:
    L1 = net.layers[1]
    if axis6:
        L1.enable_axis6_field(field_sigma=field_sigma)
        L1.reset_epi_field()

    # Optimizer: only L1 + head (L0 was set to requires_grad=False).
    opt = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr,
    )

    events: List[Tuple[int, str]] = []
    spawns_this_task = 0
    last_spawn_step = -10**9
    step = 0
    engagement_sum: Optional[torch.Tensor] = (
        torch.zeros(L1.n_nodes) if track_engagement else None
    )
    engagement_count: Optional[torch.Tensor] = (
        torch.zeros(L1.n_nodes) if track_engagement else None
    )
    last_loss = float("nan")
    last_acc = float("nan")

    head_class_idx = torch.tensor(task_global_classes, dtype=torch.long)
    gen = torch.Generator().manual_seed(int(time.time() * 1000) & 0xFFFFFFFF)

    for epoch in range(n_epochs):
        for xb, yb_global in view.iter_epoch(batch_size, generator=gen):
            opt.zero_grad()
            # Explicit L0 → L1 forward (so we can apply cosine head and
            # also have L1's _last_y / hook properly cached for stress).
            h0 = net.layers[0](xb)
            h1 = l1_forward_diffused(
                net.layers[1], h0, lambda_diff,
                credit_mask=l1_credit_mask, W_snapshot=W_snapshot,
            )
            # Update activation-correlation EMA. Needed by both Hebbian
            # kernel (when HEBBIAN_MODE != off) AND Boquila alignment
            # (when ALIGN_ENABLED). Skip when neither consumer is on to
            # save compute.
            if (HEBBIAN_MODE != "off" and lambda_diff > 0.0) or ALIGN_ENABLED:
                update_activation_correlation(L1, h1)
            outs = cosine_logits(h1, net.layers[2].W)
            # Task-local labels: map global → local-in-task ({0, 1})
            yb_local = torch.zeros_like(yb_global)
            for i, g in enumerate(task_global_classes):
                yb_local = torch.where(yb_global == g,
                                       torch.tensor(i, dtype=yb_local.dtype),
                                       yb_local)
            task_logits = outs[:, head_class_idx]
            real_loss = F.cross_entropy(task_logits, yb_local)
            # Manifold replay: head-only forward of synthetic L1 features
            # for every previously-stored class. Calibrates head logits
            # across all classes seen so the full-softmax metric stays
            # meaningful, independent of credit-mask + spawning on L1.
            synth_loss = torch.tensor(0.0)
            if (manifold is not None
                    and manifold.has_classes()
                    and seen_classes_global is not None):
                sample = manifold.sample_synthetic(
                    manifold_total_batch, noise_scale=manifold_noise_scale,
                )
                if sample is not None:
                    synth_feats, synth_labels = sample
                    # Forward synth L0 → L1 (using CURRENT L1 so the
                    # representation matches eval-time), THEN DETACH
                    # before head. The detach ensures the synth loss
                    # only updates the head — L1 stays purely a function
                    # of real data. Without this, synth pulls new
                    # (uncredited) cells toward predicting OLD classes,
                    # fighting the credit anchor and degrading task_aware.
                    with torch.no_grad():
                        synth_h1 = l1_forward_diffused(
                            net.layers[1], synth_feats, lambda_diff,
                            credit_mask=l1_credit_mask,
                            W_snapshot=W_snapshot,
                        )
                    synth_logits = cosine_logits(synth_h1, net.layers[2].W)
                    synth_loss = masked_cross_entropy(
                        synth_logits, synth_labels,
                        active_classes=list(seen_classes_global),
                    )
            loss = real_loss + lambda_manifold * synth_loss
            if ALIGN_ENABLED:
                # Boquila aux: pull activation-correlated cells together
                # in position space. Returns 0 (no grad) until Hebbian
                # EMA is seeded.
                align_loss = alignment_aux_loss(L1)
                loss = loss + ALIGN_WEIGHT * align_loss
            loss.backward()
            L1.update_internal_stress()

            # Track L1 engagement BEFORE the optimizer step (need the
            # cached _last_y from the just-completed forward pass).
            if track_engagement:
                with torch.no_grad():
                    last_y = L1._last_y
                    if last_y is not None:
                        if L1.activation == "relu":
                            active = (last_y > 0).float()
                        else:
                            active = (last_y.abs() > 0.05).float()
                        per_cell_rate = active.mean(dim=0).cpu()
                        n_cells = per_cell_rate.numel()
                        if engagement_sum is None or engagement_sum.numel() < n_cells:
                            pad_n = n_cells - (
                                0 if engagement_sum is None
                                else engagement_sum.numel()
                            )
                            zeros = torch.zeros(pad_n)
                            engagement_sum = (
                                zeros if engagement_sum is None
                                else torch.cat([engagement_sum, zeros])
                            )
                            engagement_count = (
                                torch.zeros_like(engagement_sum)
                                if engagement_count is None
                                else torch.cat([engagement_count, zeros])
                            )
                        engagement_sum[:n_cells] += per_cell_rate
                        engagement_count[:n_cells] += 1.0

            # Apply credit-based gradient mask on L1 (NOT L0 — L0 is frozen).
            if l1_credit_mask is not None and l1_credit_mask.any():
                with torch.no_grad():
                    cm = l1_credit_mask
                    n_cells = L1.W.shape[0]
                    if cm.numel() < n_cells:
                        pad = torch.zeros(
                            n_cells - cm.numel(),
                            dtype=torch.bool, device=cm.device,
                        )
                        cm = torch.cat([cm, pad])
                    if L1.W.grad is not None:
                        L1.W.grad[cm] = 0.0
                    if L1.b.grad is not None:
                        L1.b.grad[cm] = 0.0

            opt.step()
            # Boquila alignment: positions just moved under SGD. Clamp
            # back into the bounding region, then if finite-space is on,
            # run a relaxation step so the alignment pull (every step)
            # is BALANCED by repulsion (every step). Without per-step
            # repulsion, alignment dominates over many steps and cells
            # collapse to a centroid. Finally sync into the substrate
            # buffer so kernel/Hebbian/LCN see the migrated positions.
            if ALIGN_ENABLED:
                with torch.no_grad():
                    L1.cell_position_learnable.data = clamp_to_region(
                        L1.cell_position_learnable.data,
                        mode=BOUND_REGION,
                        dims=REPULSE_DIMS,
                    )
                    if FINITE_SPACE_ENABLED:
                        # Single relaxation step on the LEARNABLE param
                        # (not the buffer) so the next SGD step sees the
                        # post-repulsion positions.
                        L1.cell_position_learnable.data = relax_positions_step(
                            L1.cell_position_learnable.data,
                            min_sep=REPULSE_MIN_SEP,
                            strength=REPULSE_STRENGTH,
                            dims=REPULSE_DIMS,
                        )
                        L1.cell_position_learnable.data = clamp_to_region(
                            L1.cell_position_learnable.data,
                            mode=BOUND_REGION,
                            dims=REPULSE_DIMS,
                        )
                sync_positions_to_buffer(L1)
            # LCN-on-L1: we deliberately DO NOT call reapply_l1_lcn_mask
            # per step. Soft Gaussian masks with values in (0, 1) would
            # iteratively shrink in-window entries under repeated multiply
            # (degenerate behavior: a weight at mask=0.135 halves an order
            # of magnitude per ~7 steps). The forward-time multiply inside
            # diffused_l1_weight (W_eff_live * lcn) gives correct gradient
            # scaling on its own — off-window entries see grad ∝ mask, so
            # binary masks freeze them at init (mask=0 → grad=0) and soft
            # masks keep them small without runaway drift. Install-time
            # zeroing (in install_l1_lcn_mask) handles the starting point.
            step += 1

            if axis6:
                L1.update_epi_field(dt=field_dt, stress_tolerance=stress_tolerance)
                if (spawns_this_task < spawn_cap
                        and step - last_spawn_step >= cooldown_steps):
                    cand = L1.field_conditional_growth_candidate(
                        mode="absolute", k=1.0,
                        stress_floor=diff_floor, b_threshold=diff_b_thr,
                    )
                    if cand is not None:
                        new_idx = net.axis6_spawn(1, cand, position_jitter=0.05)
                        spawns_this_task += 1
                        last_spawn_step = step
                        events.append(
                            (step,
                             f"AXIS6 spawn: parent {cand} → new {new_idx} "
                             f"@ {L1.cell_position[new_idx][:2].tolist()}")
                        )
                        # LCN-on-L1: extend mask with new cell's window.
                        # Must come BEFORE opt rebuild so any later reapply
                        # finds the matching shape.
                        extend_l1_lcn_mask(net)
                        # Activation-correlation buffer extension (needed
                        # by Hebbian kernel and/or Boquila alignment).
                        # New cell inherits parent's row/col as warm
                        # start (it IS a noisy copy of parent post-spawn).
                        if HEBBIAN_MODE != "off" or ALIGN_ENABLED:
                            extend_act_corr_on_spawn(L1, cand, new_idx)
                        # Finite-space mechanics: the new cell pushes its
                        # neighbors, which push THEIR neighbors. Pressure
                        # propagates outward; positions repack so cells
                        # maintain at least min_sep separation, clamped to
                        # the bounding region. No-op when AXIS6_FINITE_
                        # SPACE=0. After this, _field_kernel is None so
                        # the next forward rebuilds from new positions.
                        if FINITE_SPACE_ENABLED:
                            max_disp = relax_after_spawn(L1)
                            if max_disp > 0.01:
                                events.append(
                                    (step,
                                     f"FINITE_SPACE relax: max_disp="
                                     f"{max_disp:.3f}")
                                )
                        # Boquila alignment: extend learnable positions
                        # with the new cell. The substrate's cell_position
                        # buffer already has parent+jitter (set by spawn);
                        # plus any finite-space displacement just applied.
                        # The new learnable param mirrors that.
                        if ALIGN_ENABLED:
                            extend_learnable_positions(L1)
                        # Rebuild opt to include new parameter slots.
                        opt = torch.optim.Adam(
                            [p for p in net.parameters() if p.requires_grad],
                            lr=lr,
                        )

            with torch.no_grad():
                pred = task_logits.argmax(dim=1)
                last_loss = float(loss.item())
                last_acc = float((pred == yb_local).float().mean().item())

        if verbose:
            extras = ""
            if axis6:
                extras = (
                    f" stress_max={float(L1.internal_stress.max().item()):.4f}"
                    f" epi_A_max={float(L1.epi_A.max().item()):.4f}"
                    f" epi_B_max={float(L1.epi_B.max().item()):.4f}"
                )
            print(f"    [epoch {epoch}] loss={last_loss:.4f} acc={last_acc:.3f} "
                  f"H={L1.n_nodes} spawns={spawns_this_task}{extras}")

    engagement_frac: Optional[torch.Tensor] = None
    if track_engagement and engagement_sum is not None and engagement_count is not None:
        n_cells = L1.n_nodes
        if engagement_sum.numel() < n_cells:
            pad = torch.zeros(n_cells - engagement_sum.numel())
            engagement_sum = torch.cat([engagement_sum, pad])
            engagement_count = torch.cat([engagement_count, torch.zeros_like(pad)])
        engagement_frac = engagement_sum / engagement_count.clamp(min=1.0)

    return TaskTrainResult(
        events=events,
        engagement_frac=engagement_frac,
        train_loss_final=last_loss,
        train_acc_final=last_acc,
    )


# ----------------------------------------------------------------------
# Evaluate a single task in the GLOBAL-class space (task-aware metric:
# argmax restricted to task's 2 class outputs).
# ----------------------------------------------------------------------
def _forward_logits(
    net: TrioronNetwork,
    x: torch.Tensor,
    lambda_diff: float = 0.0,
    credit_mask: Optional[torch.Tensor] = None,
    W_snapshot: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    h0 = net.layers[0](x)
    h1 = l1_forward_diffused(
        net.layers[1], h0, lambda_diff,
        credit_mask=credit_mask, W_snapshot=W_snapshot,
    )
    return cosine_logits(h1, net.layers[2].W)


def evaluate_task_aware(
    net: TrioronNetwork,
    view: TaskDataView,
    task_global_classes: List[int],
    lambda_diff: float = 0.0,
    credit_mask: Optional[torch.Tensor] = None,
    W_snapshot: Optional[torch.Tensor] = None,
) -> float:
    head_class_idx = torch.tensor(task_global_classes, dtype=torch.long)
    with torch.no_grad():
        x, y_global = view.all_examples()
        outs = _forward_logits(
            net, x, lambda_diff=lambda_diff,
            credit_mask=credit_mask, W_snapshot=W_snapshot,
        )
        task_logits = outs[:, head_class_idx]
        pred_local = task_logits.argmax(dim=1)
        pred_global = head_class_idx[pred_local]
    return float((pred_global == y_global).float().mean().item())


def evaluate_full(
    net: TrioronNetwork,
    view: TaskDataView,
    lambda_diff: float = 0.0,
    credit_mask: Optional[torch.Tensor] = None,
    W_snapshot: Optional[torch.Tensor] = None,
) -> float:
    with torch.no_grad():
        x, y_global = view.all_examples()
        outs = _forward_logits(
            net, x, lambda_diff=lambda_diff,
            credit_mask=credit_mask, W_snapshot=W_snapshot,
        )
        pred_global = outs.argmax(dim=1)
    return float((pred_global == y_global).float().mean().item())


# ----------------------------------------------------------------------
# Run a single arm
# ----------------------------------------------------------------------
def run_arm(
    seed: int,
    arm_label: str,
    axis6: bool,
    credit_enabled: bool,
    manifold_enabled: bool,
    train_views: Sequence[TaskDataView],
    eval_views: Sequence[TaskDataView],
    task_class_lists: Sequence[List[int]],
    n_epochs: int,
    batch_size: int,
    lr: float,
    hp: dict,
    starting_task_idx: int = 0,
    starting_net: Optional[TrioronNetwork] = None,
    starting_credit: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> dict:
    if starting_net is None:
        net = build_net(
            seed=seed,
            lcn_enabled=hp.get("lcn_enabled", True),
            l1_lcn_mode=hp.get("l1_lcn_mode", L1_LCN_MODE),
            l1_lcn_sigma=hp.get("l1_lcn_sigma", L1_LCN_SIGMA),
            l1_lcn_k=hp.get("l1_lcn_k", L1_LCN_K),
        )
        cell_credit = (
            torch.zeros(net.layers[1].n_nodes, dtype=torch.bool)
            if credit_enabled else None
        )
    else:
        net = starting_net
        cell_credit = starting_credit if starting_credit is not None else (
            torch.zeros(net.layers[1].n_nodes, dtype=torch.bool)
            if credit_enabled else None
        )

    manifold = (
        ManifoldStore(n_l0=net.layers[0].n_nodes)
        if manifold_enabled else None
    )

    lambda_diff = hp.get("lambda_diff", 0.0)
    # W_snapshot: per-cell snapshot of W_eff at credit-grant time. Used
    # to lock credited cells' effective behavior against later neighbor
    # changes under diffusion. None if diffusion is off or credit is off.
    W_snapshot: Optional[torch.Tensor] = None
    if lambda_diff > 0.0 and credit_enabled:
        L1 = net.layers[1]
        W_snapshot = torch.zeros(
            L1.n_nodes, L1.W.shape[1],
            dtype=L1.W.dtype, device=L1.W.device,
        )

    K = len(train_views)
    retention_task_aware: List[List[float]] = []
    retention_full: List[List[float]] = []
    all_events: List[Tuple[str, int, str]] = []
    seen_classes: List[int] = []
    t0 = time.time()
    for task_idx in range(starting_task_idx, K):
        # Track cumulative seen classes (current task included from the
        # start of its training — this gates the masked-CE active set).
        for c in task_class_lists[task_idx]:
            if c not in seen_classes:
                seen_classes.append(c)
        name = train_views[task_idx].name
        if verbose:
            print(f"  [phase {task_idx+1}/{K}: {name}] L1.H={net.layers[1].n_nodes}  "
                  f"frozen={int(cell_credit.sum().item()) if cell_credit is not None else 'n/a'}")
        res = train_one_task(
            net=net, view=train_views[task_idx],
            task_global_classes=task_class_lists[task_idx],
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            axis6=axis6,
            diff_floor=hp["diff_floor"], diff_b_thr=hp["diff_b_thr"],
            field_sigma=hp["field_sigma"], field_dt=hp["field_dt"],
            stress_tolerance=hp["stress_tolerance"],
            spawn_cap=hp["spawn_cap"], cooldown_steps=hp["cooldown_steps"],
            l1_credit_mask=cell_credit,
            track_engagement=credit_enabled,
            manifold=manifold,
            lambda_manifold=hp["lambda_manifold"],
            manifold_total_batch=hp["manifold_total_batch"],
            manifold_noise_scale=hp["manifold_noise_scale"],
            seen_classes_global=list(seen_classes),
            lambda_diff=lambda_diff,
            W_snapshot=W_snapshot,
            verbose=verbose,
        )
        for step, msg in res.events:
            all_events.append((name, step, msg))
        # End-of-task: snapshot retention on every task seen so far.
        row_ta = [
            evaluate_task_aware(
                net, eval_views[k], task_class_lists[k],
                lambda_diff=lambda_diff,
                credit_mask=cell_credit, W_snapshot=W_snapshot,
            )
            for k in range(K)
        ]
        row_full = [
            evaluate_full(
                net, eval_views[k],
                lambda_diff=lambda_diff,
                credit_mask=cell_credit, W_snapshot=W_snapshot,
            )
            for k in range(K)
        ]
        retention_task_aware.append(row_ta)
        retention_full.append(row_full)
        if verbose:
            print(f"    task_aware(this)={row_ta[task_idx]:.3f}  "
                  f"task_aware(mean_seen)={sum(row_ta[:task_idx+1])/(task_idx+1):.3f}  "
                  f"full(this)={row_full[task_idx]:.3f}  "
                  f"spawn_events={len(res.events)}")
        # Update L1 cell_credit by engagement.
        if credit_enabled and res.engagement_frac is not None:
            with torch.no_grad():
                L1 = net.layers[1]
                if cell_credit is None:
                    cell_credit = torch.zeros(L1.n_nodes, dtype=torch.bool)
                elif cell_credit.numel() < L1.n_nodes:
                    pad = torch.zeros(L1.n_nodes - cell_credit.numel(),
                                      dtype=torch.bool)
                    cell_credit = torch.cat([cell_credit, pad])
                ef = res.engagement_frac
                if ef.numel() < L1.n_nodes:
                    pad_e = torch.zeros(L1.n_nodes - ef.numel())
                    ef = torch.cat([ef, pad_e])
                earned = ef > hp["credit_thr"]
                newly = earned & (~cell_credit)
                cell_credit = cell_credit | earned
                if verbose:
                    print(f"    [credit] +{int(newly.sum().item())} new; "
                          f"total frozen = {int(cell_credit.sum().item())}/"
                          f"{cell_credit.numel()}")
                # Snapshot W_eff at credit-grant time for newly credited
                # cells (only when diffusion is on). The snapshot freezes
                # their effective behavior under later neighbor changes.
                if lambda_diff > 0.0 and W_snapshot is not None:
                    L1 = net.layers[1]
                    if W_snapshot.shape[0] < L1.n_nodes:
                        pad = torch.zeros(
                            L1.n_nodes - W_snapshot.shape[0],
                            W_snapshot.shape[1],
                            dtype=W_snapshot.dtype, device=W_snapshot.device,
                        )
                        W_snapshot = torch.cat([W_snapshot, pad], dim=0)
                    if newly.any():
                        # Compute current live W_eff for the L1 layer.
                        # We pass credit_mask=None so the snapshot doesn't
                        # use itself; we want the LIVE diffused value.
                        W_eff_live = diffused_l1_weight(
                            L1, lambda_diff,
                            credit_mask=None, W_snapshot=None,
                        )
                        if newly.numel() < L1.n_nodes:
                            newly_padded = torch.cat([
                                newly,
                                torch.zeros(
                                    L1.n_nodes - newly.numel(),
                                    dtype=torch.bool,
                                ),
                            ])
                        else:
                            newly_padded = newly
                        snap_idx = newly_padded.nonzero().flatten()
                        W_snapshot[snap_idx] = W_eff_live[snap_idx].detach()
        # Snapshot manifold per-class μ/σ AFTER credit is applied so the
        # stored features reflect the frozen-from-now-on representation.
        if manifold is not None:
            manifold.store_task(
                net=net, view=train_views[task_idx],
                task_global_classes=task_class_lists[task_idx],
            )
            if verbose:
                print(f"    [manifold] stored {len(task_class_lists[task_idx])} classes; "
                      f"total stored = {len(manifold.mu_per_class)} classes "
                      f"(n_l0={manifold.n_l0})")

    elapsed = time.time() - t0
    # Final pass: per-task task-aware + full accuracies
    final_task_aware = retention_task_aware[-1] if retention_task_aware else []
    final_full = retention_full[-1] if retention_full else []
    first_task_aware = [retention_task_aware[k][k] for k in range(len(retention_task_aware))]
    drift_task_aware = [
        first_task_aware[k] - final_task_aware[k]
        for k in range(len(first_task_aware))
    ]
    max_drift_ta = max((abs(d) for d in drift_task_aware), default=0.0)

    # n_spawns counts AXIS6_spawn events only (not relax-displacement
    # diagnostics which also go into all_events). Match on the message
    # prefix used by the spawn-event log in train_one_task.
    n_spawn_events = sum(
        1 for _, _, msg in all_events if msg.startswith("AXIS6 spawn:")
    )
    return {
        "seed": seed,
        "arm": arm_label,
        "elapsed_s": elapsed,
        "n_hidden_L1": net.layers[1].n_nodes,
        "n_spawns": n_spawn_events,
        "n_frozen": int(cell_credit.sum().item()) if cell_credit is not None else 0,
        "retention_task_aware": retention_task_aware,
        "retention_full": retention_full,
        "final_task_aware": final_task_aware,
        "final_full": final_full,
        "mean_final_task_aware": (
            sum(final_task_aware) / len(final_task_aware)
            if final_task_aware else 0.0
        ),
        "mean_final_full": (
            sum(final_full) / len(final_full)
            if final_full else 0.0
        ),
        "max_drift_task_aware": max_drift_ta,
        "events": all_events,
        "net": net,
        "cell_credit": cell_credit,
    }


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return m, v ** 0.5


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    n_epochs = int(os.environ.get("AXIS6_EPOCHS", "4"))
    batch_size = int(os.environ.get("AXIS6_BATCH", "256"))
    lr = float(os.environ.get("AXIS6_LR", "0.01"))
    n_seeds = int(os.environ.get("AXIS6_N_SEEDS", "1"))
    diff_floor = float(os.environ.get("AXIS6_FLOOR", "0.005"))
    diff_b_thr = float(os.environ.get("AXIS6_B_THR", "1.0"))
    field_sigma = float(os.environ.get("AXIS6_SIGMA", "0.2"))
    field_dt = float(os.environ.get("AXIS6_DT", "0.1"))
    stress_tolerance = float(os.environ.get("AXIS6_TOL", "0.0"))
    spawn_cap = int(os.environ.get("AXIS6_SPAWN_CAP", "5"))
    cooldown_steps = int(os.environ.get("AXIS6_COOLDOWN", "50"))
    credit_thr = float(os.environ.get("AXIS6_CREDIT_THR", "0.1"))
    lambda_manifold = float(os.environ.get("AXIS6_LAMBDA_MANIFOLD", "1.0"))
    manifold_total_batch = int(os.environ.get("AXIS6_MANIFOLD_BATCH", "64"))
    manifold_noise_scale = float(os.environ.get("AXIS6_MANIFOLD_NOISE", "1.0"))
    lcn_enabled = os.environ.get("AXIS6_LCN", "1") == "1"
    # Diffusion-of-W on L1: λ blends own weight with neighbor-averaged
    # weights via the same Gaussian kernel that diffuses internal_stress.
    # 0.0 = no diffusion (current behavior). 0.3 = moderate sharing.
    lambda_diff = float(os.environ.get("AXIS6_LAMBDA_DIFF", "0.0"))
    data_root = os.environ.get("AXIS6_DATA_ROOT", DEFAULT_DATA_ROOT)
    arms_env = os.environ.get(
        "AXIS6_ARMS",
        "BASELINE_frozen,AXIS6_emerge,AXIS6_credit,AXIS6_credit_manifold",
    )

    hp = dict(
        diff_floor=diff_floor, diff_b_thr=diff_b_thr,
        field_sigma=field_sigma, field_dt=field_dt,
        stress_tolerance=stress_tolerance,
        spawn_cap=spawn_cap, cooldown_steps=cooldown_steps,
        credit_thr=credit_thr,
        lambda_manifold=lambda_manifold,
        manifold_total_batch=manifold_total_batch,
        manifold_noise_scale=manifold_noise_scale,
        lcn_enabled=lcn_enabled,
        lambda_diff=lambda_diff,
    )

    print("=" * 88)
    print(f"axis6_credit_chained15 — LCN perception + L1 substrate")
    print(f"epochs/task={n_epochs}  batch={batch_size}  lr={lr}  "
          f"n_seeds={n_seeds}  spawn_cap={spawn_cap}  credit_thr={credit_thr}")
    print(f"H_init(L1)={H_INIT}  L0_width(frozen)={L0_WIDTH}  "
          f"LCN={'on' if lcn_enabled else 'OFF (random projection)'}  "
          f"grid={LCN_GRID_X}×{LCN_GRID_Y}  LCN_σ={LCN_SIGMA}")
    print(f"field_sigma={field_sigma}  diff_floor={diff_floor}  "
          f"cooldown={cooldown_steps}")
    print("=" * 88)

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=data_root,
        n_holdout_per_dataset=0,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]
    print(f"Curriculum: {[s.name for s in specs]}")
    print(f"Train sizes: {[v.n_examples() for v in train_views]}")
    print(f"Eval  sizes: {[v.n_examples() for v in eval_views]}")
    print()

    # Each arm = (label, axis6_on, credit_on, manifold_on)
    arms = [
        ("BASELINE_frozen", False, False, False),
        ("AXIS6_emerge", True, False, False),
        ("AXIS6_credit", True, True, False),
        ("AXIS6_credit_manifold", True, True, True),
    ]
    arms_set = {s.strip() for s in arms_env.split(",")}
    selected = [a for a in arms if a[0] in arms_set]

    results: dict = {a[0]: [] for a in selected}
    for seed in range(n_seeds):
        for arm_label, axis6, credit_enabled, manifold_enabled in selected:
            print(f"\n========== SEED {seed}  ARM: {arm_label} ==========")
            r = run_arm(
                seed=seed, arm_label=arm_label,
                axis6=axis6, credit_enabled=credit_enabled,
                manifold_enabled=manifold_enabled,
                train_views=train_views, eval_views=eval_views,
                task_class_lists=task_class_lists,
                n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                hp=hp, verbose=True,
            )
            results[arm_label].append(r)
            print(
                f"[seed {seed}  arm {arm_label}]  "
                f"task_aware_mean={r['mean_final_task_aware']:.3f}  "
                f"full_mean={r['mean_final_full']:.3f}  "
                f"max|drift|_ta={r['max_drift_task_aware']:.3f}  "
                f"H={r['n_hidden_L1']}  spawns={r['n_spawns']}  "
                f"frozen={r['n_frozen']}  elapsed={r['elapsed_s']:.1f}s"
            )

    print("\n" + "=" * 88)
    print(f"AGGREGATE (n={n_seeds} seeds, mean ± std)")
    print("=" * 88)
    print(
        f"  {'arm':<16}  {'task_aware':>14}  {'full':>14}  "
        f"{'max|drift|_ta':>14}  {'H_L1':>10}  {'spawns':>8}  {'frozen':>8}"
    )
    for arm_label, *_ in selected:
        rs = results[arm_label]
        ta_m, ta_s = mean_std([r["mean_final_task_aware"] for r in rs])
        f_m, f_s = mean_std([r["mean_final_full"] for r in rs])
        md_m, md_s = mean_std([r["max_drift_task_aware"] for r in rs])
        h_m, h_s = mean_std([float(r["n_hidden_L1"]) for r in rs])
        sp_m, sp_s = mean_std([float(r["n_spawns"]) for r in rs])
        fr_m, fr_s = mean_std([float(r["n_frozen"]) for r in rs])
        print(
            f"  {arm_label:<16}  {ta_m:>6.3f} ± {ta_s:>5.3f}  "
            f"{f_m:>6.3f} ± {f_s:>5.3f}  "
            f"{md_m:>6.3f} ± {md_s:>5.3f}  "
            f"{h_m:>4.1f} ± {h_s:>3.1f}  "
            f"{sp_m:>3.1f} ± {sp_s:>2.1f}  "
            f"{fr_m:>3.1f} ± {fr_s:>2.1f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
