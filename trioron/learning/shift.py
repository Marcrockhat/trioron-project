"""Streaming context-shift detection — aperiodic surprise against an adaptive EWMA.

The deployment regime has no task boundaries: the router (learning/router.py)
needs a "the context just changed" signal to know when to re-route / rebuild.
This module is the aperiodic first cut of spec §10.2.1's design:

- Per-feature surprise ``r = |x − μ| / σ`` against a streaming baseline —
  the diagonal per-coordinate Mahalanobis term, unsupervised and O(d).
- The EWMA forgetting factor is DERIVED, not guessed: under a local-level
  model (random-walk mean, variance ``q``; observation noise, variance
  ``r_obs``) the steady-state Kalman gain gives the optimal EWMA rate in
  closed form from the SNR ``λ = q/r_obs``:

      s = (λ + sqrt(λ² + 4λ)) / 2,   α* = s / (s + 1)

  ``q`` and ``r_obs`` are identified online from the differenced stream
  ``d_t = x_t − x_{t−1}``:  Var(d) = q + 2·r_obs,  lag-1 Cov(d) = −r_obs.

- Shift declaration self-normalizes the scalar novelty trace (its own slow
  EWMA mean/std, FROZEN while a candidate shift is running so the shift
  cannot absorb itself) and fires on z > threshold for k consecutive
  samples, with a cooldown.

The phase-conditioned comb baseline (periodic feeds, §10.2.1 traps 1–4) is
deliberately NOT built here — it needs the period range ``P_min..P_max``
and a period detector; this aperiodic EWMA is its documented warm-up/
fallback path and the deployment-relevant first cut.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1e-8


class SurpriseBaseline:
    """Per-feature streaming (μ, σ) with a Kalman-derived adaptive EWMA rate.

    All state is O(d).  ``update(x)`` consumes one sample ``[d]`` and returns
    the per-feature surprise measured against the baseline BEFORE the update.
    """

    def __init__(self, dim: int, alpha_min: float = 1e-3, alpha_max: float = 0.5,
                 stat_beta: float = 0.02, warmup: int = 10, robust_clip: float = 3.0,
                 device: torch.device | str = "cpu") -> None:
        self.dim = dim
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.stat_beta = stat_beta          # slow EWMA for the q/r estimators
        self.warmup = warmup
        # Huber-style gate: samples beyond robust_clip·σ update the baseline
        # and the SNR statistics at down-weighted rate, so a suspected shift
        # cannot absorb itself into the baseline before the detector rules.
        self.robust_clip = robust_clip
        self.n = 0
        d = torch.device(device)
        self.mu = torch.zeros(dim, device=d)
        self.var = torch.ones(dim, device=d)
        self.alpha = torch.full((dim,), alpha_max, device=d)  # plastic until derived
        self._prev_x = torch.zeros(dim, device=d)
        self._prev_d = torch.zeros(dim, device=d)
        self._var_d = torch.zeros(dim, device=d)
        self._cov_d = torch.zeros(dim, device=d)

    def _derive_alpha(self) -> None:
        """Closed-form steady-state Kalman gain from the online SNR estimate."""
        r_obs = (-self._cov_d).clamp(min=_EPS)
        q = (self._var_d - 2.0 * r_obs).clamp(min=_EPS)
        lam = q / r_obs
        s = 0.5 * (lam + torch.sqrt(lam * lam + 4.0 * lam))
        self.alpha = (s / (s + 1.0)).clamp(self.alpha_min, self.alpha_max)

    @torch.no_grad()
    def score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-feature surprise of ``x`` against the current baseline. Read-only."""
        if self.n == 0:
            return torch.zeros_like(x)
        return (x - self.mu).abs() / (self.var + _EPS).sqrt()

    @torch.no_grad()
    def absorb(self, x: torch.Tensor) -> None:
        """Fold one sample ``[d]`` into the baseline and the SNR estimators."""
        self.n += 1
        if self.n == 1:
            self.mu = x.clone()
            self._prev_x = x.clone()
            return

        surprise = (x - self.mu).abs() / (self.var + _EPS).sqrt()
        w = torch.where(surprise > self.robust_clip,
                        self.robust_clip / (surprise + _EPS),
                        torch.ones_like(surprise))

        # q / r_obs identification on the differenced stream (outliers gated)
        d = x - self._prev_x
        b = self.stat_beta * w
        self._var_d = (1 - b) * self._var_d + b * d * d
        if self.n > 2:
            self._cov_d = (1 - b) * self._cov_d + b * d * self._prev_d
        self._prev_d = d
        self._prev_x = x.clone()
        if self.n > self.warmup:
            self._derive_alpha()

        # EWMA level + variance at the derived rate (outliers gated)
        a = self.alpha * w
        resid = x - self.mu
        self.mu = self.mu + a * resid
        self.var = (1 - a) * self.var + a * resid * resid
        self.var = self.var.clamp(min=_EPS)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> torch.Tensor:
        """score + absorb in one call (standalone use, no shift gating)."""
        surprise = self.score(x)
        self.absorb(x)
        if self.n <= self.warmup:
            return torch.zeros_like(x)
        return surprise

    @torch.no_grad()
    def reanchor(self, x: torch.Tensor) -> None:
        """Snap the level to the new context; keep scale + SNR estimators."""
        self.mu = x.clone()
        self._prev_x = x.clone()


@dataclass
class ShiftEvent:
    step: int          # sample index at which the shift was declared
    z: float           # normalized novelty at declaration
    novelty: float     # raw novelty at declaration


class ShiftDetector:
    """Declare context shifts from the self-normalized novelty trace.

    Feed one sample per call.  Returns a :class:`ShiftEvent` on the sample
    where a shift is declared, else ``None``.  The novelty trace's own
    (mean, std) baseline freezes while a candidate run is open so a real
    shift cannot inflate the baseline that judges it.
    """

    def __init__(self, dim: int, z_threshold: float = 6.0, k_consecutive: int = 3,
                 cooldown: int = 30, novelty_beta: float = 0.02,
                 reanchor: bool = True, baseline: SurpriseBaseline | None = None,
                 device: torch.device | str = "cpu") -> None:
        self.baseline = baseline or SurpriseBaseline(dim, device=device)
        self.z_threshold = z_threshold
        self.k_consecutive = k_consecutive
        self.cooldown = cooldown
        self.novelty_beta = novelty_beta
        self.auto_reanchor = reanchor
        self.step = 0
        self.novelty = 0.0
        self.z = 0.0
        self.events: list[ShiftEvent] = []
        self._nov_mean = 0.0
        self._nov_var = 1.0
        self._run = 0              # consecutive over-threshold samples
        self._cooldown_left = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> ShiftEvent | None:
        self.step += 1
        surprise = self.baseline.score(x)
        self.novelty = float(surprise.mean())

        if self.baseline.n <= self.baseline.warmup:
            self.baseline.absorb(x)
            return None

        sd = max(self._nov_var, _EPS) ** 0.5
        self.z = (self.novelty - self._nov_mean) / sd

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            self.baseline.absorb(x)
            self._absorb_novelty()
            return None

        if self.z > self.z_threshold:
            # Candidate open: BOTH baselines frozen — a suspected shift gets
            # zero absorption until the detector rules on it.
            self._run += 1
            if self._run >= self.k_consecutive:
                event = ShiftEvent(step=self.step, z=self.z, novelty=self.novelty)
                self.events.append(event)
                self._run = 0
                self._cooldown_left = self.cooldown
                if self.auto_reanchor:
                    self.baseline.reanchor(x)
                return event
            return None

        self._run = 0
        self.baseline.absorb(x)
        self._absorb_novelty()
        return None

    def _absorb_novelty(self) -> None:
        b = self.novelty_beta
        delta = self.novelty - self._nov_mean
        self._nov_mean += b * delta
        self._nov_var = (1 - b) * self._nov_var + b * delta * delta
