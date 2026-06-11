"""Mixed-stream controller — the division consumer.  [D12]

See docs/design/mixed_stream_growth.md §2. The mixed-stream regime
(Rocky, s031): the stream arrives shuffled, no labels, no class
boundaries; a period is a window of stream samples. Period 1
(PerceptionGenesis) spawns perception and births the ONE blurred
world-class; this controller takes over from period 2 and consumes the
council's discrimination decisions — `StressRouter.decide(FRUSTRATED)`
finally has a consumer (the s031 diag found 14 signals dropped).

Per boundary meeting:

  1. members  — each window sample joins its best-matched class
                (matched filter against buffer-mean templates) and
                enters that class's ROLLING buffer.
  2. judgment — the council trials a division on every judgable buffer
                (division.try_divide); any accepted candidate means the
                world is still blurred → the period is FRUSTRATED.
  3. economy  — stress.settle(last) then stress.decide(status); a
                DISCRIMINATION decision routes to the progenitor.
  4. execute  — the progenitor commits the divisions: parent astrocyte
                retires (DORMANT, lineage kept via arena.parent),
                two sibling classes are born with fresh astrocyte rows.
                Structural contract [D11]: division grows bookkeeping
                rows by design — classes are memory, not tissue.

Templates are buffer means recomputed each boundary (the probe's
semantics): accuracy lives in the structure — which buffers exist —
not in slowly-annealed weights. Settlement stays global-status in M2;
per-class attribution is M3 [D13].
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch

from trioron.core.census import census
from trioron.core.construct import Substrate
from trioron.core.receptor import N_QUANTA
from trioron.core.state import CellState

from .controller import MeetingReport, PCLLController
from .division import BUF, CLASS_CAP, try_divide
from .lockin import LockInView, deposit, matched_k, reset
from .resolve import EMPTY, FRUSTRATED, RESOLVED
from .signature import LearnedClass
from .stress import DISCRIMINATION


class MixedStreamController:
    """Attach after period 1; owns the boundary meeting for periods 2+."""

    def __init__(self, substrate: Substrate, *,
                 stress=None, adopt: Optional[PCLLController] = None) -> None:
        self.substrate = substrate
        self.stress = stress
        self.receptor_ids = substrate.scheduler._plan.receptor_ids
        if self.receptor_ids.numel() == 0:
            raise ValueError("substrate has no RECEPTOR cells")
        n = int(self.receptor_ids.numel())
        # Adopt the period-1 controller's world: codec, read set, the natal
        # world-class (already astrocyte-backed) — one organism, one history.
        self.codec = adopt.codec if adopt is not None else None
        self.read_mask = (adopt.read_mask.clone() if adopt is not None
                          else torch.ones(n, dtype=torch.bool))
        self._world: Optional[LearnedClass] = (
            adopt.learner.classes[0]
            if adopt is not None and adopt.learner.classes else None)
        self.periods = adopt.periods if adopt is not None else 0
        self.classes: List[LearnedClass] = []
        self.bufs: List[torch.Tensor] = []        # per-class rolling phasors
        self._window: List[torch.Tensor] = []     # this period's pockets
        self._births = 1                           # u1 = the world class
        self._lineage: dict = {}                   # child name → parent astrocyte
        # Canonical-frame registry (D10): per-sample contrast pockets are
        # translated into one canonical frame before membership and division —
        # frames move per sample by design (the receptor encodes contrast);
        # buffers must not mistake frame motion for mode structure (the s021
        # trigger lesson). The canonical frame is the running EXTREMES (not
        # the mean — a mean frame clamps true values to the pocket edges and
        # the clamp pileup divides forever), frozen at the genesis boundary:
        # the world window's extremes are the population's, causally.
        self._continuous = (
            substrate.arena.receptor_levels[self.receptor_ids.long()] < 2)
        self._flo = float("inf")       # running min of per-sample lo
        self._fhi = float("-inf")      # running max of per-sample hi
        self._frozen = False
        substrate.attach_pcll(self)

    # ── streaming ─────────────────────────────────────────────────

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """One batch through the substrate forward path; keeps the
        per-sample pockets + frames for the boundary's membership step."""
        if self.codec is not None:
            x = self.codec(x)
        y = self.substrate.forward(x)
        sched = self.substrate.scheduler
        q = sched._last_receptor_q
        deposit(self.substrate.arena, self.receptor_ids, q)
        frame = sched._last_receptor_frame
        self._window.append((q.clone(),
                             None if frame is None
                             else (frame[0].clone(), frame[1].clone())))
        if frame is not None and not self._frozen:
            self._flo = min(self._flo, float(frame[0].min()))
            self._fhi = max(self._fhi, float(frame[1].max()))
        return y

    def _canonical(self, q: torch.Tensor, frame) -> torch.Tensor:
        """Translate per-sample contrast pockets into the canonical frame
        (D10's affine, per sample): q' = a·q + b with
        a = (hi_s − lo_s)/(HI − LO), b = Q·(lo_s − LO)/(HI − LO).
        Exact inverse of the quantizer's per-sample map; values are interior
        by construction (the canonical frame holds the extremes). Discrete
        labeled lines are frame-free and pass through."""
        if frame is None or not math.isfinite(self._flo):
            return q.clone()
        span = self._fhi - self._flo
        if span <= 1e-9:
            return q.clone()
        lo, hi = frame
        a = ((hi - lo) / span).unsqueeze(1)
        b = (N_QUANTA * (lo - self._flo) / span).unsqueeze(1)
        qc = (a * q + b).clamp(1, N_QUANTA - 1)
        return torch.where(self._continuous, qc, q)

    # ── the boundary meeting (called by Substrate.end_task) ──────

    def period_boundary(self) -> MeetingReport:
        arena = self.substrate.arena
        read = LockInView(arena, self.receptor_ids, mask=self.read_mask)
        empty = not bool((read.margin() > matched_k()).any())
        q = (torch.cat([self._canonical(qi, fi) for qi, fi in self._window])
             if self._window else torch.empty(0, len(self.receptor_ids)))
        self._window = []
        Z = torch.exp(1j * 2 * math.pi * q / N_QUANTA)

        if not self.bufs:                          # genesis: the world-blur
            self._frozen = True                    # world extremes = the frame
            event, name = self._genesis(arena, Z)
            status = EMPTY if empty else RESOLVED  # birth = comprehension
            if self.stress is not None:
                self.stress.settle(status)
                self.stress.decide(status)
            return self._report(arena, event, name, status, None, 0)

        self._assign(Z)                            # members → rolling buffers

        candidates = []                            # the council's judgment
        for k, b in enumerate(self.bufs):
            if len(self.bufs) + len(candidates) >= CLASS_CAP:
                break                              # envelope guard, at commit
            verdict = try_divide(b)
            if verdict is not None:
                candidates.append((k, verdict))

        status = (EMPTY if empty
                  else FRUSTRATED if candidates else RESOLVED)
        grow = None
        if self.stress is not None:
            self.stress.settle(status)
            grow = self.stress.decide(status)

        n_div = 0
        if candidates and (grow == DISCRIMINATION or self.stress is None):
            n_div = self._execute(arena, candidates)

        return self._report(arena, "match", None, status, grow, n_div)

    # ── the steps ─────────────────────────────────────────────────

    def _genesis(self, arena, Z: torch.Tensor):
        """Birth (or adopt) the one blurred world-class."""
        if self._world is not None:
            world = self._world
        else:
            world = LearnedClass("u1", torch.zeros(Z.shape[1],
                                                   dtype=torch.complex64),
                                 torch.zeros(Z.shape[1]),
                                 self.read_mask.clone())
        self.classes = [world]
        self.bufs = [Z[-BUF:]]
        self._refresh(arena)
        return "birth", world.name

    def _assign(self, Z: torch.Tensor) -> None:
        """Matched-filter membership against buffer-mean templates."""
        if not len(Z):
            return
        T = torch.stack([b.mean(0) for b in self.bufs])
        member = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
        for k in range(len(self.bufs)):
            zk = Z[member == k]
            if len(zk):
                self.bufs[k] = torch.cat([self.bufs[k], zk])[-BUF:]

    def _execute(self, arena, candidates) -> int:
        """The progenitor commits the council's divisions: parent retires,
        two siblings are born (new buffers, new astrocyte rows, lineage)."""
        divided = {k for k, _ in candidates}
        verdict = dict(candidates)
        new_classes, new_bufs = [], []
        for k, (c, b) in enumerate(zip(self.classes, self.bufs)):
            if k not in divided:
                new_classes.append(c)
                new_bufs.append(b)
                continue
            side, _ = verdict[k]
            if c.cell_id >= 0:
                arena.state[c.cell_id] = CellState.DORMANT   # retire parent
            for child_side in (~side, side):
                self._births += 1
                child = LearnedClass(f"u{self._births}",
                                     torch.zeros_like(c.T),
                                     torch.zeros_like(c.m),
                                     c.active.clone())
                new_classes.append(child)
                new_bufs.append(b[child_side])
                self._lineage[child.name] = c.cell_id
        self.classes, self.bufs = new_classes, new_bufs
        return len(divided)

    def _refresh(self, arena) -> None:
        """Templates from buffers (the probe's semantics) + astrocyte sync:
        the class's existence, weight, and lineage are arena-visible [D11]."""
        for idx, (c, b) in enumerate(zip(self.classes, self.bufs)):
            c.T = b.mean(0)
            c.m = torch.full_like(c.m, float(len(b)))
            if c.cell_id == -1:
                try:
                    cid = int(arena.alloc(1).item())
                except RuntimeError:
                    c.cell_id = -2
                    continue
                arena.forward_inclusion[cid] = False
                arena.epigenome[cid] = 0
                arena.position[cid] = torch.tensor([1.0, idx * 0.01, 0.9])
                parent = self._lineage.get(c.name, -1)
                if parent >= 0:
                    arena.parent[cid] = parent
                c.cell_id = cid
            if c.cell_id >= 0:
                arena.engagement[c.cell_id] = float(len(b))

    def _report(self, arena, event, name, status, grow, n_div) -> MeetingReport:
        self._refresh(arena)
        reset(arena, self.receptor_ids)
        self.periods += 1
        return MeetingReport(self.periods, None, event, name,
                             n_read=int(self.read_mask.sum()),
                             status=status, grow=grow,
                             divisions=n_div, census=census(arena))

    # ── readouts ──────────────────────────────────────────────────

    def templates(self) -> torch.Tensor:
        """[C, F] buffer-mean templates (matched-filter readout)."""
        return torch.stack([b.mean(0) for b in self.bufs])

    def pockets_of(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample canonical-frame pockets through the substrate path —
        no deposits, no window, frame registry untouched (frozen-world
        eval)."""
        if self.codec is not None:
            x = self.codec(x)
        self.substrate.forward(x)
        sched = self.substrate.scheduler
        return self._canonical(sched._last_receptor_q,
                               sched._last_receptor_frame)
