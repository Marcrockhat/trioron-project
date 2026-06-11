"""PCLL controller — the period loop + boundary meeting.  See spec §10.4.

The controller is the driver-facing object that wires PCLL onto a
substrate. Constructing it attaches it (substrate.attach_pcll), after
which the period boundary is automatic: Substrate.end_task() runs the
boundary meeting and returns its MeetingReport.

  observe(x)          — forward through the substrate (receptor cells are
                        injected as phase by the scheduler, spec §10.2),
                        then deposit the pockets into the arena lock-in
                        rows with the mask rule. Streaming, gradient-free.
  period_boundary()   — the meeting (spec §10.4):
                          1. resolve against the learned templates
                             (BEFORE learning — the verdict reflects
                             pre-meeting knowledge)
                          2. signature learn (match / birth / empty)
                          3. update the frustration adapter
                          4. reset accumulators (next period starts clean)
                        Stress routing into the council vote economy is
                        integration phase I3 and lands here.

PCLLResolution is the FrustrationDetector-consumer adapter: growth loops
read .is_frustrated / .multiplier and cannot tell which detector drives
them. Mapping (parameter-free up to the shared ceiling): a RESOLVED
period (or a birth — a new class that explains the stream is comprehension,
not stress) → multiplier 1.0; an EMPTY period → full stress (the ceiling;
deprivation is maximally aversive, design §6); FRUSTRATED interpolates by
the gap-margin deficit. The ceiling 5.04 matches FrustrationConfig so
consumers see the same dynamic range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from trioron.core.construct import Substrate
from trioron.core.state import CellState

from .lockin import LockInView, deposit, reset
from .resolve import EMPTY, RESOLVED, Resolution, Resolver
from .signature import ScheduleLearner

_CEILING = 5.04  # = FrustrationConfig ceiling (learning/frustration.py)

# Habituation patience (design §8 / integration §3): a READ receptor that is
# incoherent for this many consecutive signal-bearing periods retires to
# shadow (dormant, still depositing, no longer read). Mirrors the stress.py
# true-void → accepted-empty after exactly 3. Empty periods do NOT advance
# streaks — a period with nothing in the stream says nothing about channel
# quality (deprivation is the organism's stress, not the channels').
RETIRE_PATIENCE = 3


@dataclass
class MeetingReport:
    period: int                       # 1-based period index of this boundary
    resolution: Optional[Resolution]  # None until the first class is born
    event: str                        # 'empty' | 'match' | 'birth'
    class_name: Optional[str]         # learner's class for this period
    retired: List[int] = field(default_factory=list)  # receptor cols retired now
    n_read: int = 0                   # read-set size after this meeting
    status: str = ""                  # the period's exclusive status (spec §10.4)
    grow: Optional[str] = None        # stress driver decision (spec §10.6) or None


class PCLLResolution:
    """FrustrationDetector consumer interface, driven by PCLL margins."""

    def __init__(self, k_gap: float = 3.0) -> None:
        self.k_gap = k_gap
        self._multiplier = 1.0

    def update(self, resolution: Optional[Resolution], event: str) -> None:
        if event == "empty":
            self._multiplier = _CEILING            # empty-stress (design §6)
        elif event == "birth" or resolution is None or resolution.status == RESOLVED:
            self._multiplier = 1.0
        else:                                      # FRUSTRATED: margin deficit
            gap = resolution.gap_margin if resolution.gap_margin is not None else 0.0
            deficit = min(max((self.k_gap - gap) / self.k_gap, 0.0), 1.0)
            self._multiplier = 1.0 + (_CEILING - 1.0) * deficit

    @property
    def multiplier(self) -> float:
        return self._multiplier

    @property
    def is_frustrated(self) -> bool:
        return self._multiplier > 1.0


class PCLLController:
    def __init__(self, substrate: Substrate,
                 k_abs: float = 3.0, k_gap: float = 3.0, *,
                 codec: Optional[Callable] = None,
                 stress=None,
                 resolver_templates: Optional[dict] = None) -> None:
        self.substrate = substrate
        self.k_abs = k_abs
        self.k_gap = k_gap
        # Discrete-column translation (raw level values → level indices,
        # spec §10.2) — built by the period-1 census (progenitor.py).
        self.codec = codec
        # Stress router over the germline book (spec §10.6); None = no routing.
        self.stress = stress
        # Known-world mode: resolve against these templates instead of the
        # learner's (a provided world model / the stress-gate scaffolding).
        # Restriction to the read set happens via the masked view, so full
        # templates compose with recruit() naturally.
        self.resolver_templates = resolver_templates
        # Single source of truth for receptor identity/order: the compiled
        # dispatch plan (construct() compiles before we get here).
        self.receptor_ids = substrate.scheduler._plan.receptor_ids
        if self.receptor_ids.numel() == 0:
            raise ValueError(
                "substrate has no RECEPTOR cells — set the RECEPTOR gene "
                "(with PERCEPTION) on the input cells and compile first"
            )
        n = int(self.receptor_ids.numel())
        self.learner = ScheduleLearner(n)
        self.frustration = PCLLResolution(k_gap)
        self.periods = 0
        # Read set + habituation streaks (design §8). Retired receptors stay
        # in the plan and keep depositing (shadows); they leave the READ set.
        self.read_mask = torch.ones(n, dtype=torch.bool)
        self._streak = torch.zeros(n, dtype=torch.int32)
        substrate.attach_pcll(self)

    # ── streaming ─────────────────────────────────────────────────

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """One batch of observations through the substrate forward path;
        deposits the receptor pockets into the arena lock-in rows."""
        if self.codec is not None:
            x = self.codec(x)
        y = self.substrate.forward(x)
        q = self.substrate.scheduler._last_receptor_q
        deposit(self.substrate.arena, self.receptor_ids, q)
        return y

    # ── the boundary meeting (called by Substrate.end_task) ──────

    def period_boundary(self) -> MeetingReport:
        arena = self.substrate.arena
        full = LockInView(arena, self.receptor_ids)
        read = LockInView(arena, self.receptor_ids, mask=self.read_mask)

        templates = (self.resolver_templates if self.resolver_templates
                     is not None else self.learner.templates())
        resolution = (Resolver(templates).resolve(read, self.k_abs, self.k_gap)
                      if templates else None)
        event, name = self.learner.observe(read)
        status = self._status(resolution, event)

        grow = None
        if self.stress is not None:
            self.stress.settle(status)   # outcome of last boundary's growth
            grow = self.stress.decide(status)

        retired = self._habituate(arena, full, event)
        self.frustration.update(resolution, event)
        reset(arena, self.receptor_ids)
        self.periods += 1
        return MeetingReport(self.periods, resolution, event, name,
                             retired=retired,
                             n_read=int(self.read_mask.sum()),
                             status=status, grow=grow)

    def _status(self, resolution: Optional[Resolution], event: str) -> str:
        """The period's exclusive status (spec §10.4). In learned-world mode a
        BIRTH is comprehension, not stress — a new class that explains the
        stream resolves the period (the s029 disruptor semantics). In
        known-world mode (resolver_templates) the resolution speaks alone."""
        if event == "empty":
            return EMPTY
        if self.resolver_templates is not None and resolution is not None:
            return resolution.status
        if event == "birth" or resolution is None:
            return RESOLVED
        return resolution.status

    def _habituate(self, arena, full: LockInView, event: str) -> List[int]:
        """Streak-based retirement of read receptors that stay incoherent
        across signal-bearing periods (the per-channel threshold is matched
        to the channel's null: K=4 for binary labeled lines, else K=3)."""
        if event == "empty":
            return []
        levels = arena.receptor_levels[self.receptor_ids.long()]
        k_col = torch.where(levels == 2,
                            torch.full_like(full.n, 4.0),
                            torch.full_like(full.n, 3.0))
        coherent = full.margin() > k_col
        # A channel testifies only with n ≥ K² deposits: |A| ≤ n bounds the
        # margin at √n, so below K² deposits coherence is UNREACHABLE and the
        # period carries no testimony either way (e.g. a column that spent the
        # period saturated as the sample's gain reference). Streaks hold.
        testified = full.n >= k_col**2
        self._streak = torch.where(
            ~testified, self._streak,
            torch.where(coherent, torch.zeros_like(self._streak),
                        self._streak + 1))
        retire = self.read_mask & (self._streak >= RETIRE_PATIENCE)
        if not bool(retire.any()):
            return []
        self.read_mask &= ~retire
        for cid in self.receptor_ids[retire].tolist():
            arena.state[cid] = CellState.DORMANT
        return retire.nonzero(as_tuple=False).squeeze(-1).tolist()

    # ── growth actions (spec §10.6) ───────────────────────────────

    def recruit(self, col: int) -> None:
        """Discrimination growth: bring a sensed-but-unread receptor into the
        READ set (the s029 recruit-on-ambiguity mechanic). Its habituation
        streak restarts."""
        self.read_mask[col] = True
        self._streak[col] = 0
        cid = int(self.receptor_ids[col].item())
        self.substrate.arena.state[cid] = CellState.ACTIVE

    def refresh_receptors(self) -> None:
        """Re-pull the receptor set after the progenitor attaches/withdraws
        receptors (compile first). Learner classes and habituation state are
        remapped by CELL ID; new receptors enter unread-history (read=True,
        streak 0) and pad every learned class as inactive — structurally zero
        interference (spec §10.5)."""
        self.substrate.compile()
        plan = self.substrate.scheduler._plan
        new_ids = plan.receptor_ids
        old_pos = {int(c): i for i, c in enumerate(self.receptor_ids.tolist())}
        n = int(new_ids.numel())

        read = torch.ones(n, dtype=torch.bool)
        streak = torch.zeros(n, dtype=torch.int32)
        for j, c in enumerate(new_ids.tolist()):
            if c in old_pos:
                read[j] = self.read_mask[old_pos[c]]
                streak[j] = self._streak[old_pos[c]]
        for cls in self.learner.classes:
            T = torch.zeros(n, dtype=torch.complex64)
            act = torch.zeros(n, dtype=torch.bool)
            for j, c in enumerate(new_ids.tolist()):
                if c in old_pos:
                    T[j] = cls.T[old_pos[c]]
                    act[j] = cls.active[old_pos[c]]
            cls.T, cls.active = T, act
        self.learner.F = n
        self.receptor_ids = new_ids
        self.read_mask = read
        self._streak = streak
