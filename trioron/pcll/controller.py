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

from dataclasses import dataclass
from typing import Optional

import torch

from trioron.core.construct import Substrate

from .lockin import LockInView, deposit, reset
from .resolve import EMPTY, RESOLVED, Resolution, Resolver
from .signature import ScheduleLearner

_CEILING = 5.04  # = FrustrationConfig ceiling (learning/frustration.py)


@dataclass
class MeetingReport:
    period: int                       # 1-based period index of this boundary
    resolution: Optional[Resolution]  # None until the first class is born
    event: str                        # 'empty' | 'match' | 'birth'
    class_name: Optional[str]         # learner's class for this period


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
                 k_abs: float = 3.0, k_gap: float = 3.0) -> None:
        self.substrate = substrate
        self.k_abs = k_abs
        self.k_gap = k_gap
        # Single source of truth for receptor identity/order: the compiled
        # dispatch plan (construct() compiles before we get here).
        self.receptor_ids = substrate.scheduler._plan.receptor_ids
        if self.receptor_ids.numel() == 0:
            raise ValueError(
                "substrate has no RECEPTOR cells — set the RECEPTOR gene "
                "(with PERCEPTION) on the input cells and compile first"
            )
        self.learner = ScheduleLearner(int(self.receptor_ids.numel()))
        self.frustration = PCLLResolution(k_gap)
        self.periods = 0
        substrate.attach_pcll(self)

    # ── streaming ─────────────────────────────────────────────────

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """One batch of observations through the substrate forward path;
        deposits the receptor pockets into the arena lock-in rows."""
        y = self.substrate.forward(x)
        q = self.substrate.scheduler._last_receptor_q
        deposit(self.substrate.arena, self.receptor_ids, q)
        return y

    # ── the boundary meeting (called by Substrate.end_task) ──────

    def period_boundary(self) -> MeetingReport:
        arena = self.substrate.arena
        view = LockInView(arena, self.receptor_ids)

        templates = self.learner.templates()
        resolution = (Resolver(templates).resolve(view, self.k_abs, self.k_gap)
                      if templates else None)
        event, name = self.learner.observe(view)

        self.frustration.update(resolution, event)
        reset(arena, self.receptor_ids)
        self.periods += 1
        return MeetingReport(self.periods, resolution, event, name)
