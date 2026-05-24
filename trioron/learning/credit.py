"""Credit-based locking — the primary anchoring mechanism.  See spec §4.1."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from trioron.core.arena import Arena
from trioron.core.epigenome import CREDIT_ELIGIBLE, has_gene
from trioron.core.state import CellState


@dataclass
class CreditConfig:
    theta_e: float = 0.3
    consecutive_tasks: int = 2
    g_min: float = 1e-4
    engagement_decay: float = 0.01
    lock_base_rate: float = 0.5
    lock_floor: int = 1


class CreditTracker:
    """Tracks per-cell engagement and applies credit-based locking at task boundaries."""

    def __init__(self, arena: Arena, cfg: CreditConfig | None = None) -> None:
        self.arena = arena
        self.cfg = cfg or CreditConfig()
        self._consecutive_above = torch.zeros(arena.capacity, dtype=torch.int32, device=arena.device)
        self._cooldown = torch.zeros(arena.capacity, dtype=torch.int32, device=arena.device)

    def update_engagement(self, activations: torch.Tensor) -> None:
        """Update engagement EMA from a batch's activations.

        Args:
            activations: ``[batch, capacity]`` — per-cell activations for the batch.
        """
        a = self.arena
        rate = (activations.abs() > 0).float().mean(dim=0)
        alive = a.alive
        rho = self.cfg.engagement_decay
        a.engagement[alive] = (1 - rho) * a.engagement[alive] + rho * rate[alive]

    def update_utility(self) -> None:
        """Update utility from current gradient magnitudes on edge weights."""
        a = self.arena
        if a.edge_weight.grad is None or a.edge_cursor == 0:
            return
        grad_abs = a.edge_weight.grad[:a.edge_cursor].abs()
        dst = a.edge_dst[:a.edge_cursor].long()
        util = torch.zeros(a.capacity, device=a.device)
        util.scatter_add_(0, dst, grad_abs)
        counts = torch.zeros(a.capacity, device=a.device)
        counts.scatter_add_(0, dst, torch.ones_like(grad_abs))
        mask = counts > 0
        a.utility[mask] = util[mask] / counts[mask]

    def consolidate(self, stability_factor: float = 1.0) -> list[int]:
        """Evaluate lock condition at task boundary. Returns list of newly locked cell ids."""
        a = self.arena
        cfg = self.cfg

        eligible = (
            a.alive
            & (a.state == CellState.ACTIVE)
            & has_gene(a.epigenome, CREDIT_ELIGIBLE).bool()
            & (self._cooldown == 0)
        )
        elig_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
        if elig_ids.numel() == 0:
            self._decrement_cooldowns()
            return []

        above = a.engagement[elig_ids] >= cfg.theta_e
        self._consecutive_above[elig_ids[above]] += 1
        self._consecutive_above[elig_ids[~above]] = 0

        lock_ready = (
            (self._consecutive_above[elig_ids] >= cfg.consecutive_tasks)
            & (a.utility[elig_ids] < cfg.g_min)
        )

        cap = self._lock_cap(stability_factor)
        ready_ids = elig_ids[lock_ready]
        if ready_ids.numel() > cap:
            _, topk = a.engagement[ready_ids].topk(cap)
            ready_ids = ready_ids[topk]

        locked: list[int] = []
        for cid in ready_ids.tolist():
            a.state[cid] = CellState.DORMANT
            self._consecutive_above[cid] = 0
            locked.append(cid)

        self._decrement_cooldowns()
        return locked

    def set_cooldown(self, cell_id: int, tasks: int) -> None:
        """Exempt a cell from locking for *tasks* task boundaries (used after rejuvenation)."""
        self._cooldown[cell_id] = tasks

    def _lock_cap(self, phi: float) -> int:
        n_active = int((self.arena.alive & (self.arena.state == CellState.ACTIVE)).sum().item())
        cap = max(self.cfg.lock_floor, math.ceil(math.sqrt(n_active) * self.cfg.lock_base_rate * phi))
        return cap

    def _decrement_cooldowns(self) -> None:
        self._cooldown.clamp_min_(0)
        active = self._cooldown > 0
        self._cooldown[active] -= 1
