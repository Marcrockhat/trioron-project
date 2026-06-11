"""Trig units + phase-coherent lock-in over the period (s028 design §4–5).

Carrier decision (Rocky, s029): there is ONE carrier — the sweep phase itself
(v → 2πv/1000). A feature's evidence is tagged by the phase at which it fires;
class identity emerges from WHICH quanta its features slice, not from separate
per-class carriers.

The accumulator integrates each receptor against that carrier across the stream:
per feature, a unit phasor e^{iθ} per observation, summed over one period (a
recurrent accumulator — internal time, not an external clock). A coherent signal
(stable within-sample pattern) adds in phase → amplitude ∝ N; an incoherent one
random-walks → ∝ √N. Resolution = margin over the √N floor (parameter-free).

Two corrections found at build time (s029), both load-bearing:

1. The quadrature pair is (cos θ, sin θ), NOT (gcos, sinc). The √N floor math
   requires zero-mean carriers: E[cos θ] = E[sin θ] = 0 over a uniform phase, but
   E[sinc θ] = Si(2π)/2π ≈ 0.226 ≠ 0 — a DC bias that would grow NOISE ∝ 0.23·N
   and swamp the floor. gcos/sinc/tan remain the post-perception units (TrigBank)
   available to composing cells; the accumulator's carrier is the zero-mean pair.
   ("sin & cos give the quadrature/phase" — design §4.)

2. Saturated and silent receptors are reference, not evidence. The receptor pins
   each sample's max feature to q=1000 → θ=2π ≡ 0 → phasor exactly 1, every
   observation, for ANY input. For F-feature uniform noise that is a DC of
   P(max)=1/F per feature (amplitude 0.25·N at F=4 ≫ 3√N) — pure noise would read
   coherent. But the saturated channel carries zero within-sample information (the
   sample is normalized BY it), and a silent one (q=0) sits at true-zero floor —
   neither is a measurement. evidence_mask() excludes both. Bonus: a flat input
   (all features equal → all saturated) deposits nothing → reads EMPTY, which is
   exactly the sensory-deprivation semantics of design §6.
"""
from __future__ import annotations

import torch

N_QUANTA = 1000


# --- the per-feature trig units (design §4: one triplet per feature, placed right
# --- after perception, never stacked in depth — the GCU-detonation fix)

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


# --- lock-in accumulator

def evidence_mask(q: torch.Tensor) -> torch.Tensor:
    """True where a receptor is a measurement: not saturated (q=1000, the gain
    reference) and not silent (q=0, the true-zero floor). See module doc, point 2."""
    return (q > 0) & (q < N_QUANTA)


class LockIn:
    """Phase-coherent quadrature accumulator, one complex bin per feature.

    step() once per observation in the stream; after one period (the whole
    learning budget, design §5) read amplitude vs the √n noise floor.
    """

    def __init__(self, n_features: int):
        self.re = torch.zeros(n_features)
        self.im = torch.zeros(n_features)
        self.n = torch.zeros(n_features)  # per-feature deposit count (mask-aware)

    def step(self, theta: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        m = torch.ones_like(theta) if mask is None else mask.to(theta.dtype)
        self.re += m * torch.cos(theta)
        self.im += m * torch.sin(theta)
        self.n += m

    def amplitude(self) -> torch.Tensor:
        return torch.sqrt(self.re**2 + self.im**2)

    def noise_floor(self) -> torch.Tensor:
        """√n — random-walk statistics, parameter-free (design §5)."""
        return torch.sqrt(self.n.clamp_min(1.0))

    def margin(self) -> torch.Tensor:
        """Amplitude in units of the noise floor; the resolution signal."""
        return self.amplitude() / self.noise_floor()

    def coherent(self, k: float = 3.0) -> torch.Tensor:
        """Per-feature: does accumulated evidence clear k·√n? All-False = empty."""
        return self.margin() > k
