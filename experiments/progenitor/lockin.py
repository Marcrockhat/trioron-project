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

Discrete / binary features (Rocky, s029): under (2) a pure binary sample maps
every feature to a reference pocket and deposits nothing. Discrete features
therefore BYPASS the adaptive receptor — they have no gain or contrast to adapt
(a binary 1 is symbolic, not a magnitude, and must not win the sample max) — and
are fixed labeled lines at the midpoints of k equal bins:

    q_j = 1000·(2j+1)/(2k)        (binary: 0 → 250, 1 → 750)

Midpoints, not 1000·(j+1)/(k+1): the levels are then the k-th roots of unity
rotated by π/k, so Σ e^{iθ_j} = 0 — a feature flapping uniformly across its
levels random-walks at √N and the floor holds. The (j+1)/(k+1) spacing sums to
−1 → a 1/k DC per noisy feature (binary: 0.5·N — measured, false-coherent at
margin ~16). Always interior → discrete evidence always deposits.

Null-statistics footnote: zero mean with two unit phasors forces ANTIPODAL
placement, so a binary channel's noise is a 1-D random walk — floor still √N but
the null tail is |N(0,1)| (P>3 ≈ 0.27%), not 2-D Rayleigh (P>3 ≈ 0.01%). The
matched threshold is K≈4 for k=2; K=3 everywhere else (matched_k()). Unordered
categoricals k>3 get an arbitrary circular adjacency (midpoints is still the
max-separation choice); a high-k ORDINAL wraps (level 0 and k−1 become circle
neighbors) — once k is large, feed it through the adaptive receptor as continuous.
"""
from __future__ import annotations

import math

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


# --- discrete labeled lines (module doc, discrete section)

def theta_discrete(level: torch.Tensor, k: int) -> torch.Tensor:
    """Fixed phase for level j ∈ {0..k−1} of a k-level discrete feature:
    θ_j = (2j+1)·π/k, i.e. q_j = 1000·(2j+1)/(2k). Zero-mean over levels,
    always interior, maximal pairwise separation (binary → ±i)."""
    return math.pi * (2 * level + 1) / k


def matched_k(k_levels: int | None = None) -> float:
    """Margin threshold matched to the channel's null statistics: binary deposits
    are antipodal → 1-D Gaussian null (heavier tail) → K=4; continuous and k≥3
    discrete span the plane → 2-D Rayleigh → K=3. Same false-alarm rate."""
    return 4.0 if k_levels == 2 else 3.0


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

    def absorb(self, theta: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """Batch-equivalent of step() over a (T, F) stream chunk — same sums."""
        m = torch.ones_like(theta) if mask is None else mask.to(theta.dtype)
        self.re += (m * torch.cos(theta)).sum(0)
        self.im += (m * torch.sin(theta)).sum(0)
        self.n += m.sum(0)

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
