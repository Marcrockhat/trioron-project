"""Progenitor — the period-1 developmental program.  See spec §10.6.

The organism starts as the GERMLINE only: one progenitor with her council
attached — born together, never frozen, never pruned (s022 design §3.1;
integration design §0). There is no genesis pre-pass: the council judges,
the progenitor spawns, and structure grows live from data.

  tick 1   — the first sample arrives. The progenitor reads it through her
             own perception node (a built-in aperture: sees the raw
             vector's dimensionality; transforms nothing) and spawns one
             PERCEPTION cell per input column — the status-quo input
             layer, but progenitor-spawned (cells carry her as parent).
  tick 2   — receptor adaptation is calculable (two samples = the first
             usable structure estimate); the progenitor equips RECEPTOR
             on the spawned cells. Deposits begin.
  ticks 3+ — receptors stream deposits; the census refines each column's
             STRUCTURAL verdict (distinct-value census, D7: ≤ K_DISCRETE
             distinct values → discrete labeled lines). The per-sample
             quantisation stays history-free — the history informs only
             the spawn-time structural decision (phenotype chosen from
             the problem at the growth event, manual §2.5).
  boundary — the council's FIRST SITTING (Substrate.end_task()), ending
             with the NATAL REPLAY (s030, Rocky): period-1 observations
             are buffered and, once the verdicts are committed, replayed
             through the FINAL receptor configuration (labeled lines,
             post-starve frame, codec) and learned as a normal period —
             the class that streams while the organ is being built is
             NOT lost. The distorted pre-census deposits are discarded.
             The sitting:
             - STARVE: census-constant columns (empty/dead sensors).
               The receptor gene is WITHDRAWN (the cell stays PERCEPTION
               and goes dormant, so input-column mapping is preserved)
               — with signed data a constant column is NOT at the q=0
               floor, so starvation must be structural, not mask-borne.
             - DISCRETE k: receptor_levels set; a codec translates raw
               level values → level indices (the scheduler's labeled-line
               contract, spec §10.2).
             - importance ranking from the period's deposit evidence —
               how the council judges features without gradients: by
               which columns produce coherent evidence. Seeds the vote
               ledger (the I3 economy).
             The sitting then hands over: a PCLLController (read set =
             all surviving receptors) attaches in the program's place,
             and habituation-driven retirement (controller meetings)
             handles the variance-bearing-but-incoherent columns genesis
             culled with its gradient probe. Zero gradient steps anywhere.

Genesis subsumption (integration design §1): variance apoptosis → the
census starve verdict; the |w·g| saliency probe → lock-in evidence +
habituation. The germline cells themselves are structural (the council's
vote ledger keys for I3); they are forward-invisible and never lock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from trioron.core.construct import Substrate
from trioron.core.epigenome import (
    EXPRESSION_GENES,
    PERCEPTION,
    RECEPTOR,
    clear_gene,
    set_gene,
)
from trioron.core.state import CellState

from .controller import PCLLController
from .lockin import LockInView, deposit, reset
from .retina import (RegionSensor, first_sitting_compress, grid_positions)

K_DISCRETE = 8           # D7: ≤ this many distinct values → discrete labeled lines
COUNCIL_PER_PHENOTYPE = 4  # one 4-cell group per expression gene (6×4=24 with tanh),
                           # even multiplicity → a split vote is a real "undecided"


class Germline:
    """The progenitor + her attached council — one unit, born together,
    never frozen, never pruned (s022 §3.1). The council holds the vote
    ledger and the judgment; the progenitor is the sole spawner."""

    def __init__(self, substrate: Substrate) -> None:
        a = substrate.arena
        self.substrate = substrate

        self.progenitor_id = int(a.alloc(1).item())
        a.forward_inclusion[self.progenitor_id] = False

        self.council_ids: Dict[int, List[int]] = {}
        for p in EXPRESSION_GENES:
            ids = a.alloc(COUNCIL_PER_PHENOTYPE)
            for cid in ids.tolist():
                a.epigenome[cid] = 1 << p
                a.forward_inclusion[cid] = False
                a.refresh_phenotype(cid)
            self.council_ids[p] = ids.tolist()

        # Conserved vote economy — ONE book, two readouts (spec §10.6):
        # 4 council cells per expression gene (the composer/discrimination
        # side, step4_grow's between-phenotype ranking) + PERCEPTION_SEATS
        # bookkeeping seats held by the progenitor (the sensation side —
        # decision D4: receptors are voted ON, they do not vote). Σ conserved.
        self.votes: Dict = {
            cid: 1.0 for ids in self.council_ids.values() for cid in ids
        }
        self.perception_seats: List[str] = []
        from .stress import PERCEPTION_SEATS
        for i in range(PERCEPTION_SEATS):
            seat = f"perception:{i}"
            self.perception_seats.append(seat)
            self.votes[seat] = 1.0
        self.perception_importance: List[tuple] = []

    def spawn_perception(self, n_cols: int) -> torch.Tensor:
        """Tick 1: one PERCEPTION cell per input column (status quo),
        parented to the progenitor."""
        a = self.substrate.arena
        ids = a.alloc(n_cols)
        for cid in ids.tolist():
            a.epigenome[cid] = set_gene(int(a.epigenome[cid].item()), PERCEPTION)
            a.parent[cid] = self.progenitor_id
        return ids

    def equip_receptors(self, cell_ids: torch.Tensor) -> None:
        """Tick 2: receptor adaptation is calculable — equip RECEPTOR."""
        a = self.substrate.arena
        for cid in cell_ids.tolist():
            a.epigenome[cid] = set_gene(int(a.epigenome[cid].item()), RECEPTOR)

    def attach_receptor(self, cell_id: int) -> None:
        """Sensation growth (spec §10.6): reach toward the world by equipping
        RECEPTOR on a perception cell that has none — a never-equipped
        candidate or a starved/withdrawn one (reversible by design). The
        caller recompiles via controller.refresh_receptors()."""
        a = self.substrate.arena
        a.epigenome[cell_id] = set_gene(int(a.epigenome[cell_id].item()), RECEPTOR)
        a.state[cell_id] = CellState.ACTIVE


@dataclass
class ColumnVerdict:
    kind: str            # 'continuous' | 'discrete' | 'starved'
    k_levels: int = 0
    evidence: float = 0.0  # first-sitting deposit margin (importance seed)


@dataclass
class FirstSittingReport:
    period: int
    verdicts: Dict[int, ColumnVerdict]      # input column → verdict
    starved: List[int] = field(default_factory=list)
    discrete: Dict[int, int] = field(default_factory=dict)   # column → k
    kept: List[int] = field(default_factory=list)            # surviving receptor columns
    importance: List[tuple] = field(default_factory=list)    # (column, margin) desc
    event: str = "empty"                    # natal-replay learning event
    class_name: Optional[str] = None        # the natal period's learned class
    # retinal compression (design §3.2/§3.6): spawned region sensors and
    # the member columns they absorbed (column → sensor cell id)
    regions: List[RegionSensor] = field(default_factory=list)
    merged: Dict[int, int] = field(default_factory=dict)


class PerceptionGenesis:
    """The period-1 program. Attach by construction; feed() the period's
    observations; Substrate.end_task() runs the first sitting, which
    commits the verdicts and hands over to a PCLLController."""

    def __init__(self, substrate: Substrate,
                 input_shape: Optional[tuple] = None) -> None:
        self.substrate = substrate
        self.germline = Germline(substrate)
        self.perception_ids: Optional[torch.Tensor] = None
        self._distinct: List[Optional[set]] = []   # None = overflowed → continuous
        self._buffer: List[torch.Tensor] = []      # period-1 obs, for natal replay
        self.controller: Optional[PCLLController] = None
        # Body geometry (design §3.2): (H, W) declares the sensor sheet's
        # adjacency — the retina the world projects onto. None = flat
        # feature vector, no adjacency, no merge pass (status quo).
        self.input_shape = input_shape
        substrate.attach_pcll(self)

    # ── period-1 streaming ────────────────────────────────────────

    def feed(self, x: torch.Tensor) -> None:
        assert self.controller is None, "period 1 is over — use .controller.observe"
        sub = self.substrate
        self._buffer.append(x)
        if self.perception_ids is None:
            # tick 1: spawn from the first sample, perceive it raw
            self.perception_ids = self.germline.spawn_perception(x.shape[1])
            if self.input_shape is not None:
                H, W = self.input_shape
                assert H * W == x.shape[1], \
                    f"body geometry {H}x{W} != world dim {x.shape[1]}"
                cols = torch.arange(x.shape[1], dtype=torch.long)
                sub.arena.position[self.perception_ids.long()] = \
                    grid_positions(cols, (H, W))
            self._distinct = [set() for _ in range(x.shape[1])]
            sub.compile()
            sub.forward(x[:1])
            self._census(x[:1])
            # tick 2: equip receptors; deposits begin with the second sample
            self.germline.equip_receptors(self.perception_ids)
            sub.compile()
            x = x[1:]
            if x.shape[0] == 0:
                return
        self._census(x)
        sub.forward(x)
        deposit(sub.arena, sub.scheduler._plan.receptor_ids,
                sub.scheduler._last_receptor_q)

    def _census(self, x: torch.Tensor) -> None:
        for c in range(x.shape[1]):
            seen = self._distinct[c]
            if seen is None:
                continue
            seen.update(x[:, c].tolist())
            if len(seen) > K_DISCRETE:
                self._distinct[c] = None   # overflowed → continuous

    # ── the first sitting (called by Substrate.end_task) ──────────

    def period_boundary(self) -> FirstSittingReport:
        sub, a = self.substrate, self.substrate.arena
        ids = self.perception_ids
        assert ids is not None, "first sitting before any data"

        verdicts: Dict[int, ColumnVerdict] = {}
        starved: List[int] = []
        discrete: Dict[int, int] = {}
        levels_sorted: Dict[int, List[float]] = {}

        for c, cid in enumerate(ids.tolist()):
            seen = self._distinct[c]
            if seen is not None and len(seen) <= 1:
                # constant column: empty/dead sensor. Withdraw the receptor
                # (a constant is NOT at the q=0 floor under signed data) —
                # the cell stays PERCEPTION + dormant so column mapping holds.
                a.epigenome[cid] = clear_gene(int(a.epigenome[cid].item()), RECEPTOR)
                a.state[cid] = CellState.DORMANT
                verdicts[c] = ColumnVerdict("starved")
                starved.append(c)
            elif seen is not None and 2 <= len(seen) <= K_DISCRETE:
                k = len(seen)
                a.receptor_levels[cid] = k
                verdicts[c] = ColumnVerdict("discrete", k)
                discrete[c] = k
                levels_sorted[c] = sorted(seen)
            else:
                verdicts[c] = ColumnVerdict("continuous")

        kept = [c for c in range(len(ids)) if verdicts[c].kind != "starved"]

        # Retinal compression (design §3.2/§3.6): adjacent redundant
        # continuous columns merge into pooled region sensors with imposed
        # positions; members withdraw to dormant husks (the starve
        # treatment, so column mapping is preserved). No-op without a
        # declared body geometry.
        regions: List[RegionSensor] = []
        merged: Dict[int, int] = {}
        if self.input_shape is not None:
            eligible = [c for c in kept if verdicts[c].kind == "continuous"]
            regions = first_sitting_compress(
                self.germline, self._buffer, eligible, ids, self.input_shape)
            merged = {c: r.cell_id for r in regions for c in r.members}
            kept = [c for c in kept if c not in merged]

        sub.compile()   # receptor set changed (starved withdrawn / merged)
        # handover: the controller attaches in this program's place, with the
        # stress router over the germline's book (spec §10.6)
        from .stress import StressRouter
        self.router = StressRouter(self.germline)
        self.controller = PCLLController(sub, codec_levels=levels_sorted,
                                         stress=self.router)

        # NATAL REPLAY (s030): discard the distorted pre-census deposits and
        # replay the buffered period through the FINAL receptor configuration
        # (labeled lines, post-starve frame, codec) — period 1 is then learned
        # as a normal period; no class is lost to organ-building.
        reset(a, self.controller.receptor_ids)
        x_all = torch.cat(self._buffer, dim=0)
        self._buffer = []
        self.controller.observe(x_all)

        # importance from the CLEAN replayed evidence (1:1 receptor cols
        # first; pooled region sensors carry their own margin tail)
        plan_cols = sub.scheduler._plan.receptor_cols.tolist()
        margins = LockInView(a, self.controller.receptor_ids).margin()
        for i, c in enumerate(plan_cols):
            verdicts[c].evidence = float(margins[i])
        for k, r in enumerate(regions):
            r.evidence = float(margins[len(plan_cols) + k])
        importance = sorted(
            ((c, verdicts[c].evidence) for c in kept),
            key=lambda t: t[1], reverse=True,
        )
        self.germline.perception_importance = importance

        # the natal period's own meeting: learn it, count it, reset for period 2
        natal = self.controller.period_boundary()
        return FirstSittingReport(
            period=1, verdicts=verdicts, starved=starved, discrete=discrete,
            kept=kept, importance=importance,
            event=natal.event, class_name=natal.class_name,
            regions=regions, merged=merged,
        )


def germline_base(substrate) -> None:
    """Base recipe: the organism is born as germline only (no input layer,
    no outputs — those are spawned from data in period 1)."""
    # Cells are alloc'd by PerceptionGenesis/Germline after construct();
    # nothing to do here — construct() requires a base, this is the empty one.
