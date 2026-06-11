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

from trioron.core.census import Census, census
from trioron.core.construct import Substrate
from trioron.core.state import CellState

from .lockin import LockInView, deposit, reset
from .resolve import EMPTY, FRUSTRATED, RESOLVED, Resolution, Resolver
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
    census: Optional[Census] = None   # structural anatomy after this meeting [D11]
    divisions: int = 0                # class divisions committed this meeting [D12]


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
                 codec_levels: Optional[dict] = None,
                 stress=None,
                 resolver_templates: Optional[dict] = None) -> None:
        self.substrate = substrate
        self.k_abs = k_abs
        self.k_gap = k_gap
        # Discrete-column level values (input column → sorted raw values,
        # from the period-1 census) → the raw→level-index codec the
        # scheduler's labeled-line contract needs (spec §10.2). Kept as data
        # so it serializes (D10e).
        self.codec_levels = codec_levels or {}
        self.codec = _build_codec(self.codec_levels)
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
        self.learner.set_levels(
            substrate.arena.receptor_levels[self.receptor_ids.long()])
        self.frustration = PCLLResolution(k_gap)
        self.periods = 0
        # Read set + habituation streaks (design §8). Retired receptors stay
        # in the plan and keep depositing (shadows); they leave the READ set.
        self.read_mask = torch.ones(n, dtype=torch.bool)
        self._streak = torch.zeros(n, dtype=torch.int32)
        # Frame registry (D10): running E[lo], E[hi] of the period's
        # per-sample gain frames — the translation layer's input.
        self._frame_lo = 0.0
        self._frame_hi = 0.0
        self._frame_n = 0
        substrate.attach_pcll(self)

    # ── streaming ─────────────────────────────────────────────────

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """One batch of observations through the substrate forward path;
        deposits the receptor pockets into the arena lock-in rows and
        accumulates the period's frame statistics."""
        if self.codec is not None:
            x = self.codec(x)
        y = self.substrate.forward(x)
        sched = self.substrate.scheduler
        deposit(self.substrate.arena, self.receptor_ids, sched._last_receptor_q)
        if sched._last_receptor_frame is not None:
            lo, hi = sched._last_receptor_frame
            self._frame_lo += float(lo.sum())
            self._frame_hi += float(hi.sum())
            self._frame_n += int(lo.numel())
        return y

    def _take_frame(self):
        """The period's frame statistics (E[lo], E[hi]); None when no
        continuous receptor deposited (all-discrete organisms)."""
        if self._frame_n == 0:
            return None
        frame = (self._frame_lo / self._frame_n, self._frame_hi / self._frame_n)
        self._frame_lo = self._frame_hi = 0.0
        self._frame_n = 0
        return frame

    # ── the boundary meeting (called by Substrate.end_task) ──────

    def period_boundary(self) -> MeetingReport:
        arena = self.substrate.arena
        full = LockInView(arena, self.receptor_ids)
        read = LockInView(arena, self.receptor_ids, mask=self.read_mask)
        frame = self._take_frame()

        templates = (self.resolver_templates if self.resolver_templates
                     is not None else self.learner.templates(frame))
        resolution = (Resolver(templates).resolve(read, self.k_abs, self.k_gap)
                      if templates else None)
        event, name = self.learner.observe(read, frame=frame, shadow=full)
        status = self._status(resolution, event)

        grow = None
        if self.stress is not None:
            # Per-class settlement [D13]: a pending growth's subject class
            # testifies only when THIS period is about it (the organism's
            # best read — the resolution winner). Resolved → its margin
            # cleared (success); still blurred while it leads (failure).
            # Any other period defers: the world changing is not evidence
            # about the growth.
            def testify(subject):
                if resolution is None or resolution.winner != subject:
                    return None
                return status == RESOLVED

            self.stress.settle(status, testify=testify)
            subject = (resolution.winner if status == FRUSTRATED
                       and resolution is not None else None)
            grow = self.stress.decide(status, subject=subject)

        retired = self._habituate(arena, full, event)
        self.frustration.update(resolution, event)
        self._sync_astrocytes(arena)
        reset(arena, self.receptor_ids)
        self.periods += 1
        return MeetingReport(self.periods, resolution, event, name,
                             retired=retired,
                             n_read=int(self.read_mask.sum()),
                             status=status, grow=grow,
                             census=census(arena))

    def _sync_astrocytes(self, arena) -> None:
        """Each learned class is backed by an astrocyte arena cell
        (forward_inclusion=False, spec §2.11 / D5): the class's existence and
        weight (Σm → engagement) are visible to the arena world; the tensors
        stay object-side and serialize via state_dict (D10e)."""
        for idx, c in enumerate(self.learner.classes):
            if c.cell_id == -1:
                try:
                    cid = int(arena.alloc(1).item())
                except RuntimeError:
                    c.cell_id = -2          # arena full — stop retrying
                    continue
                arena.forward_inclusion[cid] = False
                arena.epigenome[cid] = 0
                arena.position[cid] = torch.tensor([1.0, idx * 0.01, 0.9])
                c.cell_id = cid
            if c.cell_id >= 0:
                arena.engagement[c.cell_id] = float(c.m.sum())

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

    def recruit(self, col: int) -> int:
        """Discrimination growth: bring a sensed-but-unread receptor into the
        READ set (the s029 recruit-on-ambiguity mechanic). Every class holding
        shadow evidence on the dim extends INSTANTLY — zero relearning (D10c).
        Returns how many classes extended. The habituation streak restarts."""
        self.read_mask[col] = True
        self._streak[col] = 0
        cid = int(self.receptor_ids[col].item())
        self.substrate.arena.state[cid] = CellState.ACTIVE
        return self.learner.promote(col)

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
            m = torch.zeros(n, dtype=torch.float32)
            act = torch.zeros(n, dtype=torch.bool)
            for j, c in enumerate(new_ids.tolist()):
                if c in old_pos:
                    T[j] = cls.T[old_pos[c]]
                    m[j] = cls.m[old_pos[c]]
                    act[j] = cls.active[old_pos[c]]
            cls.T, cls.m, cls.active = T, m, act
        self.learner.F = n
        self.learner.set_levels(
            self.substrate.arena.receptor_levels[new_ids.long()])
        self.receptor_ids = new_ids
        self.read_mask = read
        self._streak = streak

    # ── persistence (D10e) ────────────────────────────────────────

    def state_dict(self) -> dict:
        """The learned world + read state, keyed by CELL ID so a woken
        substrate (which re-derives receptor identity from the epigenome)
        can restore it order-independently."""
        cells = self.receptor_ids.tolist()
        state = {
            "receptor_cells": cells,
            "read_mask": self.read_mask.clone(),
            "streak": self._streak.clone(),
            "periods": self.periods,
            "codec_levels": self.codec_levels,
            "classes": [{
                "name": c.name, "T": c.T.clone(), "m": c.m.clone(),
                "active": c.active.clone(), "frame": c.frame,
            } for c in self.learner.classes],
            "births": self.learner._births,
        }
        if self.stress is not None:
            state["stress"] = {
                "votes": dict(self.stress.germline.votes),
                "empty_drive": self.stress.empty_drive,
                "pending": [list(p) for p in self.stress.pending],
            }
        return state

    def load_state_dict(self, state: dict) -> None:
        from .signature import LearnedClass
        pos = {int(c): i for i, c in enumerate(self.receptor_ids.tolist())}
        remap = [pos.get(int(c), -1) for c in state["receptor_cells"]]
        n = int(self.receptor_ids.numel())

        def _vec(src, dtype, fill=0):
            out = torch.full((n,), fill, dtype=dtype)
            for old_i, new_i in enumerate(remap):
                if new_i >= 0:
                    out[new_i] = src[old_i]
            return out

        self.read_mask = _vec(state["read_mask"], torch.bool, True)
        self._streak = _vec(state["streak"], torch.int32)
        self.periods = state["periods"]
        self.codec_levels = state["codec_levels"]
        self.codec = _build_codec(self.codec_levels)
        self.learner.classes = [
            LearnedClass(s["name"], _vec(s["T"], torch.complex64),
                         _vec(s["m"], torch.float32),
                         _vec(s["active"], torch.bool), frame=s["frame"])
            for s in state["classes"]
        ]
        self.learner._births = state["births"]
        if self.stress is not None and "stress" in state:
            self.stress.germline.votes.update(state["stress"]["votes"])
            self.stress.empty_drive = state["stress"]["empty_drive"]
            pend = state["stress"]["pending"]
            if pend is None:                       # pre-D13 single slot
                pend = []
            elif isinstance(pend, str):
                pend = [(pend, None)]
            self.stress.pending = [tuple(p) for p in pend]


def _build_codec(levels_sorted: dict) -> Optional[Callable]:
    """Raw level values → level indices for discrete columns (the
    scheduler's labeled-line contract, spec §10.2). Bucketize on midpoints
    so the mapping is robust to float jitter."""
    if not levels_sorted:
        return None
    bounds = {
        c: torch.tensor([(lv[j] + lv[j + 1]) / 2 for j in range(len(lv) - 1)])
        for c, lv in levels_sorted.items()
    }

    def codec(x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for c, b in bounds.items():
            x[:, c] = torch.bucketize(x[:, c].contiguous(), b).to(x.dtype)
        return x

    return codec
