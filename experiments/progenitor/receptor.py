"""Adaptive receptor — per-sample contrast quantizer (s028, Rocky's design).

Each sample self-normalizes to its OWN range and drops into one of 1000 discrete
pockets — gain-adaptive and history-independent, like a sensory receptor adapting to
the current stimulus (the eye encodes contrast, not absolute luminance; taste/skin
have few discrete channels). The partition number IS the phase ("signal × radian"):
a feature's magnitude becomes its angle in the period.

    lo = min(0, sample.min())      # floor at true-zero, OR a negative becomes the new min
    hi = sample.max()              # the peak feature saturates to 1000
    q  = round(1000 · (v - lo) / (hi - lo))     # partition in [0, 1000]
    θ  = 2π · q / 1000                          # phase in [0, 2π]  (2π/1000 per quantum)

q = 1000 wraps to θ = 2π = one full period — the "covered one period → called off" point.

Scale-invariant by construction: [5,10,15,20,25], [10,20,30,40,50] and [1,2,3,4,5]
all encode IDENTICALLY — absolute magnitude is discarded, only the within-sample
pattern (each feature's ratio to the sample peak) survives. The past max is irrelevant.
"""
from __future__ import annotations

import math

import torch

N_QUANTA = 1000


def quantize(x: torch.Tensor) -> torch.Tensor:
    """Per-sample partition into [0, N_QUANTA]. Last dim = features of one sample;
    leading dims are batched independently (each sample uses its own lo/hi)."""
    lo = torch.minimum(x.amin(dim=-1, keepdim=True), x.new_zeros(1))
    hi = x.amax(dim=-1, keepdim=True)
    span = (hi - lo).clamp_min(1e-9)
    return torch.round(N_QUANTA * (x - lo) / span)


def phase(x: torch.Tensor) -> torch.Tensor:
    """Receptor phase θ ∈ [0, 2π]: the partition mapped onto the full period (2π/1000 per
    quantum). q = 1000 → 2π = one full period (the 'called off' point)."""
    return 2 * math.pi * quantize(x) / N_QUANTA
