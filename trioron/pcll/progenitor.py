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

        # Conserved vote economy (Σ = 4 × n_phenotypes): per-cell credibility,
        # the I3 routing currency. Seeded uniform; the first sitting's
        # importance ranking is recorded alongside for the perception group.
        self.votes: Dict[int, float] = {
            cid: 1.0 for ids in self.council_ids.values() for cid in ids
        }
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


class PerceptionGenesis:
    """The period-1 program. Attach by construction; feed() the period's
    observations; Substrate.end_task() runs the first sitting, which
    commits the verdicts and hands over to a PCLLController."""

    def __init__(self, substrate: Substrate) -> None:
        self.substrate = substrate
        self.germline = Germline(substrate)
        self.perception_ids: Optional[torch.Tensor] = None
        self._distinct: List[Optional[set]] = []   # None = overflowed → continuous
        self._buffer: List[torch.Tensor] = []      # period-1 obs, for natal replay
        self.controller: Optional[PCLLController] = None
        substrate.attach_pcll(self)

    # ── period-1 streaming ────────────────────────────────────────

    def feed(self, x: torch.Tensor) -> None:
        assert self.controller is None, "period 1 is over — use .controller.observe"
        sub = self.substrate
        self._buffer.append(x)
        if self.perception_ids is None:
            # tick 1: spawn from the first sample, perceive it raw
            self.perception_ids = self.germline.spawn_perception(x.shape[1])
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

        sub.compile()   # receptor set changed (starved withdrawn)
        codec = _build_codec(levels_sorted)
        # handover: the controller attaches in this program's place
        self.controller = PCLLController(sub, codec=codec)

        # NATAL REPLAY (s030): discard the distorted pre-census deposits and
        # replay the buffered period through the FINAL receptor configuration
        # (labeled lines, post-starve frame, codec) — period 1 is then learned
        # as a normal period; no class is lost to organ-building.
        reset(a, self.controller.receptor_ids)
        x_all = torch.cat(self._buffer, dim=0)
        self._buffer = []
        self.controller.observe(x_all)

        # importance from the CLEAN replayed evidence (receptor cols only)
        plan_cols = sub.scheduler._plan.receptor_cols.tolist()
        margins = LockInView(a, self.controller.receptor_ids).margin()
        for i, c in enumerate(plan_cols):
            verdicts[c].evidence = float(margins[i])
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
        )


def _build_codec(levels_sorted: Dict[int, List[float]]) -> Optional[Callable]:
    """Raw level values → level indices for discrete columns (the
    scheduler's labeled-line contract). Bucketize on midpoints so the
    mapping is robust to float jitter."""
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


def germline_base(substrate) -> None:
    """Base recipe: the organism is born as germline only (no input layer,
    no outputs — those are spawned from data in period 1)."""
    # Cells are alloc'd by PerceptionGenesis/Germline after construct();
    # nothing to do here — construct() requires a base, this is the empty one.
