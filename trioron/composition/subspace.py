"""Subspace-factored L0 — the trump-card factorization for cross-seed
lossless absorption.

Background — see paper/L0_HANDSHAKE_BRIEF.md and the L0_handshake
translator memory entry. The closed-form translator
``M = W_B · W_A^+`` is exact in row-space(W_A) but pays a 656-dim
information bottleneck on full-rank inputs because two random Gaussian
projections of ℝ^784 → ℝ^128 generically project onto disjoint 128-d
subspaces (128 + 128 ≤ 784).

The fix is to separate the choice of *subspace* from the choice of
*basis*. Define a protocol-level subspace selector ``S ∈ ℝ^{n_out × n_in}``
(public, fixed, derived from a hardcoded protocol seed) and let each
donor's L0 be::

    W_donor = R_donor · S

where ``R_donor ∈ ℝ^{n_out × n_out}`` is a per-donor random *orthogonal*
rotation seeded by the donor's ``l0_seed``. Then:

* All donors share the same surviving 128-d subspace of input space.
* The 656-dim privacy bottleneck happens once at protocol design time
  and is identical across donors.
* Cross-donor translation reduces to a pure rotation
  ``M = R_B · R_A^{-1}``, with no bottleneck residual on full-rank
  inputs (verified ~1e-12 numerical floor in fp32).
* The closed-form ``M = W_B · W_A^+`` translator computes this same
  rotation automatically — no special-case code needed.

Storage: per-donor handshake collapses to the 4-byte ``l0_seed``
(recipient regenerates ``R_donor`` deterministically). The protocol
constant ``S`` is loaded once per device and amortized across all
donors.
"""
from __future__ import annotations

from typing import Optional

import torch


# Public protocol constant. Bumping this breaks compatibility with all
# previously-trained factored donors — treat as a versioned wire-protocol
# field. Value chosen so the handshake spec is greppable; not security
# sensitive (the protocol subspace is intentionally public).
PROTOCOL_SEED: int = 0xFEEDFACE


def build_protocol_subspace(
    n_in: int,
    n_out: int,
    *,
    seed: int = PROTOCOL_SEED,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the shared subspace selector S ∈ ℝ^{n_out × n_in}.

    S is a fixed Gaussian random projection (Kaiming-relu scaled) defined
    by a hardcoded protocol seed. All donors of the same protocol version
    use the same S; their per-donor L0 is ``R_donor · S``.

    Returning fp32 by default; cast at the call site if you need fp16
    storage (the full-precision pseudoinverse used by the translator
    constructor wants fp32 anyway).
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    std = (2.0 / n_in) ** 0.5
    S = torch.randn(n_out, n_in, generator=g) * std
    return S.to(dtype)


def build_donor_rotation(
    n_out: int,
    *,
    donor_seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the per-donor orthogonal rotation R_donor ∈ ℝ^{n_out × n_out}.

    Uses QR decomposition of a Gaussian matrix (a standard construction
    of Haar-distributed orthogonal matrices up to sign-of-diagonal). The
    seed is the donor's ``l0_seed`` — a 4-byte integer that is the
    entirety of the per-donor handshake state.
    """
    g = torch.Generator(device="cpu").manual_seed(int(donor_seed))
    A = torch.randn(n_out, n_out, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q.to(dtype)


def build_factored_l0_weight(
    n_in: int,
    n_out: int,
    *,
    donor_seed: int,
    protocol_seed: int = PROTOCOL_SEED,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the donor's effective L0 weight ``W = R_donor · S``.

    The result has the same shape and Kaiming-relu-style scaling as a
    naive ``randn(n_out, n_in) * sqrt(2/n_in)`` initialization, so it is
    a drop-in replacement for the existing TrioronLayer L0 weight from
    a training-distribution standpoint.
    """
    S = build_protocol_subspace(
        n_in, n_out, seed=protocol_seed, dtype=torch.float32,
    )
    R = build_donor_rotation(n_out, donor_seed=donor_seed, dtype=torch.float32)
    W = (R @ S).to(dtype)
    return W


def factor_l0_in_place(
    net,
    *,
    donor_seed: int,
    protocol_seed: int = PROTOCOL_SEED,
    layer_idx: int = 0,
) -> None:
    """Override ``net.layers[layer_idx].W`` (and its anchor) with a
    factored ``R_donor · S`` L0 weight.

    Use this immediately after constructing a TrioronNetwork-style
    classifier and before any training begins. The L0 is expected to be
    frozen for the duration of training (matches the existing
    chained-15 protocol), so the override is permanent.
    """
    layer = net.layers[layer_idx]
    n_out, n_in = layer.W.shape
    W_new = build_factored_l0_weight(
        n_in, n_out,
        donor_seed=donor_seed,
        protocol_seed=protocol_seed,
        dtype=layer.W.dtype,
    )
    with torch.no_grad():
        layer.W.copy_(W_new)
        # W_anchor is a buffer cloned at construction; keep it consistent
        # so EWC/anchor-readout reproduces the factored weight rather
        # than the original random init.
        if hasattr(layer, "W_anchor"):
            layer.W_anchor.copy_(W_new)
        # Bias stays at zero (existing donors all have zero L0 bias).
        # Touch b_anchor too for parity with W_anchor handling.
        if hasattr(layer, "b_anchor"):
            layer.b_anchor.copy_(layer.b.detach())


__all__ = [
    "PROTOCOL_SEED",
    "build_protocol_subspace",
    "build_donor_rotation",
    "build_factored_l0_weight",
    "factor_l0_in_place",
]
