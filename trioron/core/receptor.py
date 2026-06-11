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
