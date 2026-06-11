"""Lock-in deposits into arena state + trig units.  See spec §10.3.

The accumulator integrates each receptor against the single carrier (the
sweep phase, θ = 2π·q/1000) across the stream: a unit phasor per
observation, summed over one period. A coherent signal adds in phase →
amplitude ∝ N; an incoherent one random-walks → ∝ √N. Resolution is
margin over the √N floor (parameter-free).

Two load-bearing rules (s029 build-time corrections, design §11):

1. The quadrature pair is (cos θ, sin θ), NOT (gcos, sinc) — the √N floor
   requires zero-mean carriers (E[sinc] ≈ 0.226 would grow noise ∝ 0.23·N).
   gcos/sinc/tan remain post-perception units (TrigBank) for composing
   cells; the accumulator's carrier is the zero-mean pair.
2. Saturated (q = 1000, the gain reference) and silent (q = 0, the
   true-zero floor) receptors are reference, not evidence — the pinned max
   is a 1/F DC that makes uniform noise read coherent. evidence_mask()
   excludes both. Bonus: a flat input deposits nothing → reads EMPTY
   (the deprivation semantics of design §6).

State lives in the arena (lockin_re/im/n — spec §10.3), deposited by the
PCLL controller after each forward; LockInView snapshots the rows with
the read API the resolver/learner math expects.
"""
from __future__ import annotations

import math

import torch

from trioron.core.arena import Arena
from trioron.core.receptor import N_QUANTA


# ── the per-feature trig units (post-perception, never stacked in depth —
# ── the GCU-detonation fix; design §4)

def gcos(z: torch.Tensor) -> torch.Tensor:
    return z * torch.cos(z)


def sinc(z: torch.Tensor) -> torch.Tensor:
    safe = torch.where(z == 0, torch.ones_like(z), z)
    return torch.where(z == 0, torch.ones_like(z), torch.sin(z) / safe)


def tan_ramp(z: torch.Tensor) -> torch.Tensor:
    """Design §4: 'tan = z' — the ramp that flags the phase crossing."""
    return z


class TrigBank:
    """Per-feature gcos/sinc/tan on the receptor phase: (..., F) → (..., F, 3)."""

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.stack([gcos(theta), sinc(theta), tan_ramp(theta)], dim=-1)


def matched_k(k_levels: int | None = None) -> float:
    """Margin threshold matched to the channel's null statistics: binary
    deposits are antipodal → 1-D Gaussian null (heavier tail) → K=4;
    continuous and k ≥ 3 discrete span the plane → 2-D Rayleigh → K=3."""
    return 4.0 if k_levels == 2 else 3.0


# ── deposits (arena-backed)

def evidence_mask(q: torch.Tensor) -> torch.Tensor:
    """True where a receptor is a measurement: not saturated (q = 1000, the
    gain reference) and not silent (q = 0, the true-zero floor)."""
    return (q > 0) & (q < N_QUANTA)


def deposit(arena: Arena, receptor_ids: torch.Tensor, q: torch.Tensor) -> None:
    """Accumulate one batch of observations ([B, R] pockets, receptor_ids
    order — the scheduler's _last_receptor_q) into the arena lock-in rows,
    mask rule applied."""
    theta = 2 * math.pi * q / N_QUANTA
    m = evidence_mask(q).to(theta.dtype)
    ids = receptor_ids.long()
    arena.lockin_re[ids] += (m * torch.cos(theta)).sum(0)
    arena.lockin_im[ids] += (m * torch.sin(theta)).sum(0)
    arena.lockin_n[ids] += m.sum(0)


def reset(arena: Arena, receptor_ids: torch.Tensor) -> None:
    """Zero the period accumulators (boundary meeting step 5)."""
    ids = receptor_ids.long()
    arena.lockin_re[ids] = 0.0
    arena.lockin_im[ids] = 0.0
    arena.lockin_n[ids] = 0.0


class LockInView:
    """Snapshot of the arena lock-in rows with the read API the resolver
    and signature learner consume (.re/.im/.n + the √n statistics)."""

    def __init__(self, arena: Arena, receptor_ids: torch.Tensor) -> None:
        ids = receptor_ids.long()
        self.re = arena.lockin_re[ids]
        self.im = arena.lockin_im[ids]
        self.n = arena.lockin_n[ids]

    def amplitude(self) -> torch.Tensor:
        return torch.sqrt(self.re**2 + self.im**2)

    def noise_floor(self) -> torch.Tensor:
        """√n — random-walk statistics, parameter-free (design §5)."""
        return torch.sqrt(self.n.clamp_min(1.0))

    def margin(self) -> torch.Tensor:
        """Amplitude in units of the noise floor; the resolution signal."""
        return self.amplitude() / self.noise_floor()

    def coherent(self, k: float = 3.0) -> torch.Tensor:
        """Per-feature: does accumulated evidence clear k·√n?"""
        return self.margin() > k
