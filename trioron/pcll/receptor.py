"""Receptor surface for the pcll package.  See spec §9.15 / §10.2.

The quantizer math lives in core/receptor.py (the scheduler performs the
injection and core must not import pcll); this module re-exports it and
adds the discrete labeled-line placement.
"""
from __future__ import annotations

import math

import torch

from trioron.core.receptor import N_QUANTA, phase, quanta_to_phase, quantize

__all__ = ["N_QUANTA", "phase", "quanta_to_phase", "quantize", "theta_discrete"]


def theta_discrete(level: torch.Tensor, k: int) -> torch.Tensor:
    """Fixed phase for level j ∈ {0..k−1} of a k-level discrete feature:
    θ_j = (2j+1)·π/k, i.e. q_j = 1000·(2j+1)/(2k). Zero-mean over levels
    (k-th roots of unity rotated by π/k, so Σ e^{iθ_j} = 0 and the √N
    floor holds), always interior (discrete evidence always deposits),
    maximal pairwise separation (binary → ±i). The rejected
    (j+1)/(k+1) spacing sums to −1 → a 1/k DC per noisy feature
    (measured false-coherent at margin ~16, s029)."""
    return math.pi * (2 * level + 1) / k
