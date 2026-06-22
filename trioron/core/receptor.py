"""Receptor — adaptive per-sample contrast quantizer.  See spec §10.2.

Lives in core/ (not pcll/) because the scheduler performs the phase
injection and core must not import pcll; trioron/pcll/receptor.py
re-exports this module as the package surface (spec §9.15).

Each sample self-normalizes to its OWN range and drops into one of 1000
discrete pockets — gain-adaptive and history-independent, like a sensory
receptor adapting to the current stimulus (the eye encodes contrast, not
absolute luminance). The partition number IS the phase ("signal ×
radian"): a feature's magnitude becomes its angle in the period.

    lo = min(0, sample.min())   # floor at true-zero; a negative becomes the new min
    hi = sample.max()           # the peak feature saturates to 1000
    q  = round(1000 · (v − lo) / (hi − lo))     # pocket ∈ [0, 1000]
    θ  = 2π · q / 1000                          # phase ∈ [0, 2π]

Scale-invariant by construction: absolute magnitude is discarded, only
the within-sample pattern (each feature's ratio to the sample peak)
survives. q = 0 (silent) and q = 1000 (saturated) are reference, not
evidence — the mask rule lives in pcll/lockin.py (spec §10.3).
"""
from __future__ import annotations

import math

import torch

N_QUANTA = 1000


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Per-sample partition into [0, N_QUANTA]. Last dim = features of one
    sample; leading dims are batched independently (each sample uses its
    own lo/hi)."""
    lo = torch.minimum(x.amin(dim=-1, keepdim=True), x.new_zeros(1))
    hi = x.amax(dim=-1, keepdim=True)
    span = (hi - lo).clamp_min(1e-9)
    return torch.round(N_QUANTA * (x - lo) / span)


def quantize_frame(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """quantize() that also returns the per-sample gain frame (lo, hi) —
    the PCLL frame registry reads these (spec §10.5 translation layer)."""
    lo = torch.minimum(x.amin(dim=-1, keepdim=True), x.new_zeros(1))
    hi = x.amax(dim=-1, keepdim=True)
    span = (hi - lo).clamp_min(1e-9)
    q = torch.round(N_QUANTA * (x - lo) / span)
    return q, lo.squeeze(-1), hi.squeeze(-1)


def phase(x: torch.Tensor) -> torch.Tensor:
    """Receptor phase θ ∈ [0, 2π]: the partition mapped onto the full
    period (2π/1000 per quantum). q = 1000 → 2π = one full period."""
    return 2 * math.pi * quantize(x) / N_QUANTA


def quanta_to_phase(q: torch.Tensor) -> torch.Tensor:
    """Phase for already-quantized pockets (the injection's second half)."""
    return 2 * math.pi * q / N_QUANTA


def _unit(x: torch.Tensor, normalized: bool) -> torch.Tensor:
    """Normalized pocket value u ∈ [0, 1]. ``normalized=True``: x is already in
    [0, 1] under the CALLER's frame (use this for heterogeneous tables, which need
    a PER-FEATURE frame — core quantize() is per-sample and would mix incommensurate
    columns). ``normalized=False``: apply the per-sample receptor quantize()."""
    return x.clamp(0.0, 1.0) if normalized else (quantize(x) / N_QUANTA)


def arc_phase(x: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
    """Non-wrapping receptor phase θ = arcsin(u) ∈ [0, π/2]  (u = pocket / N_QUANTA).

    Unlike phase() (a 2π carrier), arc_phase is MONOTONE and INJECTIVE: u=0 and
    u=1 map to DISTINCT angles (0 and π/2), not the same point. This removes the
    wrap-collapse that makes the 2π carrier destroy bounded-discrete features — a
    binary 0/1 column collapses to a single phasor under phase() but maps to the
    orthogonal (1,0)/(0,1) under arc_phase.

    Use for discrete / categorical / bounded features. Keep the 2π phase() for
    continuous signals and for the periodic (Vernier) emitter, which NEED the wrap.
    """
    return torch.arcsin(_unit(x, normalized))


def arcsin_u_descriptor(x: torch.Tensor, *, normalized: bool = False) -> torch.Tensor:
    """"arcsin × u" encoding — magnitude-weighted arcsin phasor, concatenated
    [u·cos θ, u·sin θ] on the last dim (2× the feature count), θ = arcsin(u).

    The ANGLE encodes the value (arcsin → injective, no wrap-collapse); the
    MAGNITUDE is the value u itself (the "intensity" — features carry brightness,
    not just direction).

    Scope (measured, s045 taxonomy): a small real lift over the unit arcsin phasor,
    and it fixes the binary collapse. It does NOT beat raw features on a strong
    (quadratic/dendrite) readout — a learned readout builds its own nonlinear basis.
    So this is an OPTION for discrete/binary features and weak readouts, not a
    default front-end. For tables pass per-feature-normalized x with normalized=True.
    """
    u = _unit(x, normalized)
    th = torch.arcsin(u)
    return torch.cat([u * th.cos(), u * th.sin()], dim=-1)
