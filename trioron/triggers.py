"""Trioron — three-condition growth trigger.

Implements §4 of the blueprint. Growth requires ALL three of the
following to hold simultaneously over a sustained window:

(1) Contrastive loss plateau.
    The contrastive separation loss between conceptual opposites has
    not improved by more than ε_loss over W steps. Implemented as
    mean(prior W) − mean(recent W) < ε_loss.

(2) Effective rank saturation.
    The effective rank of the hidden activation matrix H — defined
    via the entropy of normalized singular values — is within
    ε_rank of the full latent dimension d.

(3) Gradient norm stability.
    The gradient norm sits inside [g_min, g_max] over the same window.
    This rules out optimization pathology being mistaken for capacity
    saturation.

Per §4 last paragraph: if (1)+(2) hold but (3) does not, the system
should attempt a learning-rate/optimizer reset before considering
growth. The trigger itself only REPORTS the per-condition state; the
reset policy is the orchestrator's responsibility.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch


def effective_rank(H: torch.Tensor, eps: float = 1e-12) -> float:
    """Entropy-based effective rank (Roy & Vetterli 2007).

    For an activation matrix H of shape (batch, d):
        σ = singular values of H
        p = σ / Σσ
        rank_eff = exp(-Σ p_i log p_i)

    Equals d when all singular directions are equally used; equals 1
    when only one direction is non-trivial; in between otherwise.
    """
    if H.ndim != 2:
        raise ValueError(f"H must be 2D, got shape {tuple(H.shape)}")
    with torch.no_grad():
        # CPU SVD is not implemented for BF16/FP16 — cast to FP32 for the
        # decomposition. This is purely diagnostic (used by the growth
        # trigger), not on the gradient path, so the cast is free.
        H_for_svd = H.detach()
        if H_for_svd.dtype not in (torch.float32, torch.float64):
            H_for_svd = H_for_svd.float()
        s = torch.linalg.svdvals(H_for_svd)
        s = s + eps
        p = s / s.sum()
        entropy = -(p * torch.log(p)).sum()
        return float(torch.exp(entropy))


def total_gradient_norm(parameters) -> float:
    """L2 norm across all parameter gradients. Call AFTER loss.backward()."""
    total_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            total_sq += float(p.grad.detach().pow(2).sum().item())
    return total_sq ** 0.5


@dataclass
class TriggerState:
    """Snapshot of trigger state at one step. Suitable for CSV logging."""
    step: int
    loss: float
    effective_rank: float
    grad_norm: float
    loss_plateau: bool
    rank_saturated: bool
    grad_stable: bool
    fire: bool
    loss_improvement: float       # mean(prior W) − mean(recent W)
    rank_recent_mean: float
    grad_recent_median: float
    warmup: bool                  # True while histories aren't full


class GrowthTrigger:
    """Three-condition growth trigger with independent per-condition logging.

    Usage per training step (AFTER loss.backward(), BEFORE optimizer.step()):

        state = trigger.observe(
            loss=loss.item(),
            hidden=h_a,                      # any latent activation matrix
            grad_norm=total_gradient_norm(net.parameters()),
        )
        if state.fire:
            # invoke cellular division (step 5)
            ...
        elif state.loss_plateau and state.rank_saturated and not state.grad_stable:
            # §4 escape valve: reset LR/optimizer instead of growing
            ...
    """

    def __init__(
        self,
        latent_dim: int,
        window: int = 1000,
        eps_loss: float = 0.0005,
        eps_rank: float = 0.1,
        g_min: float = 1e-4,
        g_max: float = 10.0,
    ):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.latent_dim = latent_dim
        self.window = window
        self.eps_loss = eps_loss
        self.eps_rank = eps_rank
        self.g_min = g_min
        self.g_max = g_max

        self._loss_hist: deque = deque(maxlen=2 * window)
        self._rank_hist: deque = deque(maxlen=window)
        self._grad_hist: deque = deque(maxlen=window)
        self._t: int = 0

    # ----- observation -----

    def observe(
        self,
        loss: float,
        hidden: torch.Tensor,
        grad_norm: float,
    ) -> TriggerState:
        rank = effective_rank(hidden)
        self._loss_hist.append(float(loss))
        self._rank_hist.append(float(rank))
        self._grad_hist.append(float(grad_norm))
        self._t += 1

        warmup = (
            len(self._loss_hist) < 2 * self.window
            or len(self._rank_hist) < self.window
            or len(self._grad_hist) < self.window
        )

        loss_imp = self._loss_improvement()
        rank_mean = self._rank_recent_mean()
        grad_med = self._grad_recent_median()

        loss_plateau = (not warmup) and (loss_imp < self.eps_loss)
        rank_saturated = (not warmup) and (
            (self.latent_dim - rank_mean) < self.eps_rank
        )
        grad_stable = (not warmup) and (self.g_min <= grad_med <= self.g_max)

        fire = loss_plateau and rank_saturated and grad_stable

        return TriggerState(
            step=self._t - 1,
            loss=float(loss),
            effective_rank=rank,
            grad_norm=float(grad_norm),
            loss_plateau=loss_plateau,
            rank_saturated=rank_saturated,
            grad_stable=grad_stable,
            fire=fire,
            loss_improvement=loss_imp,
            rank_recent_mean=rank_mean,
            grad_recent_median=grad_med,
            warmup=warmup,
        )

    # ----- introspection used by tests + orchestrator -----

    def reset(self) -> None:
        """Clear histories. Called after a successful division so the new
        topology is judged on its own evidence, not the pre-division window."""
        self._loss_hist.clear()
        self._rank_hist.clear()
        self._grad_hist.clear()

    def set_latent_dim(self, d: int) -> None:
        """Update the saturation reference dim. Called by the orchestrator
        after step 5 (cellular division) changes the network's latent size."""
        if d < 1:
            raise ValueError("latent_dim must be >= 1")
        self.latent_dim = d

    # ----- private helpers -----

    def _loss_improvement(self) -> float:
        """mean(prior W) − mean(recent W). Positive = loss is decreasing.
        Returns +inf during warmup so plateau is never True early."""
        if len(self._loss_hist) < 2 * self.window:
            return float("inf")
        h = list(self._loss_hist)
        prior = h[: self.window]
        recent = h[self.window : 2 * self.window]
        return (sum(prior) / self.window) - (sum(recent) / self.window)

    def _rank_recent_mean(self) -> float:
        if not self._rank_hist:
            return 0.0
        n = len(self._rank_hist)
        return sum(self._rank_hist) / n

    def _grad_recent_median(self) -> float:
        if not self._grad_hist:
            return 0.0
        sorted_g = sorted(self._grad_hist)
        return sorted_g[len(sorted_g) // 2]

    def __repr__(self) -> str:
        return (
            f"GrowthTrigger(d={self.latent_dim}, W={self.window}, "
            f"ε_loss={self.eps_loss}, ε_rank={self.eps_rank}, "
            f"g∈[{self.g_min}, {self.g_max}])"
        )
