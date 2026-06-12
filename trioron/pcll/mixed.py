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
not in slowly-annealed weights.

Settlement is per-class [D13]: each committed division is bound to the
boundary's DISCRIMINATION decision as a subject (child_a, child_b,
split dim, acceptance floor). At the next boundary — after fresh
stream members have been assigned — the division settles success iff
BOTH children still cohere on the split dim above the floor it was
accepted at: the membership that arrives after the growth exists is
the testimony (the D9 spirit). A noise slice scatters its members and
falls below the floor; the period's global status (which exponential
discovery keeps FRUSTRATED for unrelated classes) pays and drains
nothing.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch

from trioron.core.census import census
from trioron.core.construct import Substrate
from trioron.core.receptor import N_QUANTA
from trioron.core.state import CellState

from . import composer as comp
from .controller import MeetingReport, PCLLController
from .division import (BUF, CLASS_CAP, GAIN_D, MIN_CHILD, MIN_MEMBERS,
                       NULL_SPLIT, try_divide)
from .lockin import LockInView, deposit, matched_k, reset
from .resolve import EMPTY, FRUSTRATED, RESOLVED
from .signature import LearnedClass
from .stress import DISCRIMINATION

DIV_TRIES = 4   # dims judged per division when the composer arm is on
                # (§4b item 4; the division-only path keeps the M2
                # worst-dim semantics, tries=1)


class MixedStreamController:
    """Attach after period 1; owns the boundary meeting for periods 2+."""

    def __init__(self, substrate: Substrate, *,
                 stress=None, adopt: Optional[PCLLController] = None,
                 composer: bool = False,
                 divide_tries: int = 1) -> None:
        # divide_tries: dims judged per division (ascending coherence).
        # The default 1 is the M2 worst-dim semantics (byte-identical).
        # Worlds with pure-noise dims need > 1 — the worst dim is then
        # ALWAYS noise (rejected by the NULL_SPLIT floor) and division
        # never fires (M4 gate finding, s033).
        self.substrate = substrate
        self.stress = stress
        self.divide_tries = divide_tries
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
        # ── composer arm [D14+§4b] ────────────────────────────────
        if composer and stress is None:
            raise ValueError("the composer arm settles through the "
                             "stress economy — pass a StressRouter")
        self.composer = composer
        self.specs: List[comp.ComposerSpec] = []   # live composed dims
        self.composer_cells: List[int] = []        # arena cells, specs order
        self._watch: List[comp.ComposerPending] = []   # unsettled spawns
        self._to_prune: List[comp.ComposerPending] = []
        self._trial_seed = 0
        # lineage forest over class names (family pools, FAMILY_DEGREE)
        self._node: dict = {}          # class name → node id
        self._parent_node: dict = {}   # node id → parent node id | None
        self._depth: dict = {}
        self._next_node = 0
        substrate.attach_pcll(self)

    def _new_node(self, name: str, parent_name: Optional[str]) -> None:
        nid = self._next_node
        self._next_node += 1
        par = self._node.get(parent_name) if parent_name else None
        self._node[name] = nid
        self._parent_node[nid] = par
        self._depth[nid] = 0 if par is None else self._depth[par] + 1

    # ── streaming ─────────────────────────────────────────────────

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        """One batch through the substrate forward path; keeps the
        per-sample pockets + frames for the boundary's membership step.
        Spawned composer cells computed their composition IN the forward
        pass — their activations quantize through each spec's fixed
        static frame and deposit into their own lock-in rows [D14]."""
        if self.codec is not None:
            x = self.codec(x)
        y = self.substrate.forward(x)
        sched = self.substrate.scheduler
        q = sched._last_receptor_q
        deposit(self.substrate.arena, self.receptor_ids, q)
        qc = None
        if self.composer_cells:
            act = sched._last_activations[:, self.composer_cells]
            qc = torch.stack([self.specs[k].quantize(act[:, k])
                              for k in range(len(self.specs))], dim=1)
            deposit(self.substrate.arena,
                    torch.tensor(self.composer_cells), qc)
        frame = sched._last_receptor_frame
        self._window.append((q.clone(),
                             None if frame is None
                             else (frame[0].clone(), frame[1].clone()),
                             qc))
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
        if self._window:
            q = torch.cat([
                (self._canonical(qi, fi) if qc is None
                 else torch.cat([self._canonical(qi, fi), qc], dim=1))
                for qi, fi, qc in self._window])
        else:
            q = torch.empty(0, len(self.receptor_ids) + len(self.specs))
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

        member = self._assign(Z)                   # members → rolling buffers
        self._feed_fresh(q, member)                # future deposits [D9]

        candidates = []                            # the council's judgment
        tries = max(self.divide_tries, DIV_TRIES if self.composer else 1)
        for k, b in enumerate(self.bufs):
            if len(self.bufs) + len(candidates) >= CLASS_CAP:
                break                              # envelope guard, at commit
            verdict = try_divide(b, tries=tries)
            if verdict is not None:
                candidates.append((k, verdict))

        spawns = self._trials(dict(candidates)) if self.composer else []

        status = (EMPTY if empty
                  else FRUSTRATED if (candidates or spawns) else RESOLVED)
        grow = None
        for p in self._watch:
            p.age += 1
        if self.stress is not None:
            self.stress.settle(status, testify=self._testimony())
            grow = self.stress.decide(status)

        n_div, n_spawn = 0, 0
        if grow == DISCRIMINATION or self.stress is None:
            won = {k for k, _, _ in spawns}        # composer beat division
            candidates = [(k, v) for k, v in candidates if k not in won]
            records = []
            if candidates:
                n_div, records = self._execute(arena, candidates)
            records += [self._spawn(spec, names) for _, spec, names in spawns]
            n_spawn = len(spawns)
            if self.stress is not None:
                self.stress.attach_subjects(records)
        # prune AFTER execute: this boundary's division verdicts carry
        # side masks + dim indices over the pre-prune buffer width
        n_pruned = self._prune_composers()

        return self._report(arena, "match", None, status, grow, n_div,
                            spawns=n_spawn, pruned=n_pruned)

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
        self._new_node(world.name, None)
        self._refresh(arena)
        return "birth", world.name

    def _assign(self, Z: torch.Tensor) -> Optional[torch.Tensor]:
        """Matched-filter membership against buffer-mean templates.
        Returns the per-row class index (the fresh-store feed needs it)."""
        if not len(Z):
            return None
        T = torch.stack([b.mean(0) for b in self.bufs])
        member = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
        for k in range(len(self.bufs)):
            zk = Z[member == k]
            if len(zk):
                self.bufs[k] = torch.cat([self.bufs[k], zk])[-BUF:]
        return member

    # ── the composer arm [D14+§4b] ────────────────────────────────

    def _feed_fresh(self, q: torch.Tensor, member) -> None:
        """Future deposits [D9]: window members assigned to a pending
        spawn's lineage extend its virgin store (receptor pockets only —
        the trial statistic recomputes the composition itself)."""
        if not self._watch or member is None:
            return
        F = len(self.receptor_ids)
        for p in self._watch:
            idxs = [k for k, c in enumerate(self.classes)
                    if c.name in p.lineage]
            if not idxs:
                continue
            m = torch.isin(member, torch.tensor(idxs))
            if bool(m.any()):
                grew = torch.cat([p.fresh, q[m][:, :F]])
                p.fresh = grew[-2 * comp.FRESH_MIN:]

    def _recover_q(self, Zcols: torch.Tensor) -> torch.Tensor:
        """Receptor pockets back from buffered phasors (exact: canonical
        q ∈ [1, 999] → angle is injective on the circle)."""
        return (torch.angle(Zcols) / (2 * math.pi) * N_QUANTA) % N_QUANTA

    def _wired(self) -> List[int]:
        """Importance-gated wiring [D14/§4b]: receptor dims on which ANY
        judgable-size class buffer coheres — noise columns never enter a
        trial; small selected buffers confabulate (failure mode 4)."""
        wired = []
        for d in range(len(self.receptor_ids)):
            if any(len(b) >= MIN_MEMBERS
                   and float(b[:, d].mean().abs()) > comp.IMPORTANCE_R
                   for b in self.bufs):
                wired.append(d)
        return wired

    def _div_r(self, b: torch.Tensor, verdict) -> float:
        side, d = verdict
        return max(float(b[~side, d].mean().abs()),
                   float(b[side, d].mean().abs()))

    def _trials(self, div_verdicts: dict) -> List:
        """The council's overproduction (§4b): per-buffer trials COMPETE
        with division (the spawn must carve better than the best split);
        then ONE family trial over lineage-sibling pools within
        FAMILY_DEGREE generations (deepest pool first) — division
        fragments manifolds into raw-separable arcs before any
        relational buffer reaches trial size, so relational wholes
        reform for the trial only. Returns [(buffer idx | -1, spec,
        lineage names)]."""
        spawns: List = []
        room = comp.MAX_SPAWNED - len(self.specs)
        if room <= 0:
            return spawns
        wired = self._wired()
        if len(wired) < 2:
            return spawns
        taken = set(self.specs)
        F = len(self.receptor_ids)

        for k, b in enumerate(self.bufs):          # per-buffer, vs division
            if len(spawns) >= room:
                break
            if len(b) < 4 * MIN_CHILD or \
                    float(b.mean(0).abs().mean()) >= comp.TRIGGER_R:
                continue
            self._trial_seed += 1
            spec, r, _ = comp.best_candidate(
                self._recover_q(b[:, :F]), wired, taken, self._trial_seed)
            if spec is None:
                continue
            dv = div_verdicts.get(k)
            if dv is not None and self._div_r(b, dv) >= r:
                continue                           # division explains better
            taken.add(spec)
            spawns.append((k, spec, {self.classes[k].name}))

        if len(spawns) < room:                     # ≤1 family spawn/boundary
            pools: dict = {}
            for k, c in enumerate(self.classes):
                nid = self._node.get(c.name)
                if nid is None:
                    continue
                a, deg = self._parent_node.get(nid), 1
                while a is not None and deg <= comp.FAMILY_DEGREE:
                    pools.setdefault(a, set()).add(k)
                    a, deg = self._parent_node.get(a), deg + 1
            for a in sorted(pools, key=lambda n: -self._depth[n]):
                idxs = pools[a]
                if len(idxs) < 2:
                    continue
                pool = torch.cat([self.bufs[k] for k in idxs])[-BUF:]
                if len(pool) < 4 * MIN_CHILD or \
                        float(pool.mean(0).abs().mean()) >= comp.TRIGGER_R:
                    continue
                self._trial_seed += 1
                spec, _, _ = comp.best_candidate(
                    self._recover_q(pool[:, :F]), wired, taken,
                    self._trial_seed)
                if spec is not None:
                    spawns.append(
                        (-1, spec, {self.classes[k].name for k in idxs}))
                    break
        return spawns

    def _spawn(self, spec: comp.ComposerSpec, lineage) -> comp.ComposerPending:
        """Commit a winning candidate as real tissue [D11] and register
        its composed dim: buffers back-fill the new column from their own
        receptor pockets (the composition is a pure function of them);
        the stream's column comes from the forward path."""
        F = len(self.receptor_ids)
        src = (int(self.receptor_ids[spec.i]), int(self.receptor_ids[spec.j]))
        parent = (self.stress.germline.progenitor_id
                  if self.stress is not None else -1)
        cid = comp.spawn(self.substrate, spec, src, parent=parent)
        self.specs.append(spec)
        self.composer_cells.append(cid)
        for k, b in enumerate(self.bufs):          # back-fill history
            qc = spec.pockets(self._recover_q(b[:, :F]))
            zc = torch.exp(1j * 2 * math.pi * qc / N_QUANTA)
            self.bufs[k] = torch.cat([b, zc.unsqueeze(1)], dim=1)
        pend = comp.ComposerPending(
            spec, cid, dim=F + len(self.specs) - 1,
            fresh=torch.empty(0, F), lineage=set(lineage))
        self._watch.append(pend)
        return pend

    def _testimony(self):
        """Combined settlement testimony [D13]: division records settle
        against their children's split-dim coherence (M3); composer
        spawns settle on their virgin store — the §4b trial statistic
        recomputed on members that arrived AFTER the spawn existed.
        Failures queue the hard prune [D16]."""
        divide_testify = self._division_testimony()

        def testify(subject):
            if not isinstance(subject, comp.ComposerPending):
                return divide_testify(subject)
            p = subject
            ready = (len(p.fresh) >= comp.FRESH_MIN or
                     (p.age > comp.PATIENCE
                      and len(p.fresh) >= 2 * MIN_CHILD))
            if ready:
                g = torch.Generator().manual_seed(50_000 + p.cell_id)
                _, ratio = comp.trial_ratio(p.fresh, p.spec, g)
                ok = ratio > comp.NULL_RATIO
                self._watch.remove(p)
                if not ok:
                    self._to_prune.append(p)
                return ok
            if p.age > comp.PATIENCE:              # nobody deposited
                self._watch.remove(p)
                self._to_prune.append(p)
                return False
            return None

        return testify

    def _prune_composers(self) -> int:
        """PATIENCE failures hard-retire [D16]: cell out of the forward
        path, edges masked, the composed column leaves every buffer."""
        n = len(self._to_prune)
        for p in self._to_prune:
            k = self.specs.index(p.spec)
            comp.prune(self.substrate, p.cell_id)
            del self.specs[k]
            del self.composer_cells[k]
            col = len(self.receptor_ids) + k
            self.bufs = [torch.cat([b[:, :col], b[:, col + 1:]], dim=1)
                         for b in self.bufs]
        self._to_prune = []
        return n

    def _execute(self, arena, candidates):
        """The progenitor commits the council's divisions: parent retires,
        two siblings are born (new buffers, new astrocyte rows, lineage).
        Returns (count, settlement subjects [D13]) — one subject per
        division: (child_a, child_b, split dim, acceptance floor)."""
        divided = {k for k, _ in candidates}
        verdict = dict(candidates)
        new_classes, new_bufs, records = [], [], []
        for k, (c, b) in enumerate(zip(self.classes, self.bufs)):
            if k not in divided:
                new_classes.append(c)
                new_bufs.append(b)
                continue
            side, d = verdict[k]
            floor = max(float(b.mean(0).abs()[d]) + GAIN_D, NULL_SPLIT)
            if c.cell_id >= 0:
                arena.state[c.cell_id] = CellState.DORMANT   # retire parent
            names = []
            for child_side in (~side, side):
                self._births += 1
                child = LearnedClass(f"u{self._births}",
                                     torch.zeros_like(c.T),
                                     torch.zeros_like(c.m),
                                     c.active.clone())
                new_classes.append(child)
                new_bufs.append(b[child_side])
                self._lineage[child.name] = c.cell_id
                self._new_node(child.name, c.name)
                names.append(child.name)
            for p in self._watch:                # fresh follows the children
                if c.name in p.lineage:
                    p.lineage.discard(c.name)
                    p.lineage.update(names)
            records.append((names[0], names[1], d, floor))
        self.classes, self.bufs = new_classes, new_bufs
        return len(divided), records

    def _division_testimony(self):
        """[D13] Verdict for a pending division (child_a, child_b, dim,
        floor): its class's stress cleared iff BOTH children's buffers still
        cohere on the split dim above the acceptance floor, under the
        membership assigned since the division existed. Called BEFORE this
        boundary's divisions execute, so last boundary's children are
        always present."""
        by_name = {c.name: i for i, c in enumerate(self.classes)}

        def testify(subject):
            a, b, d, floor = subject
            ia, ib = by_name.get(a), by_name.get(b)
            if ia is None or ib is None:
                return None
            r_a = float(self.bufs[ia][:, d].mean().abs())
            r_b = float(self.bufs[ib][:, d].mean().abs())
            return min(r_a, r_b) > floor

        return testify

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

    def _report(self, arena, event, name, status, grow, n_div, *,
                spawns: int = 0, pruned: int = 0) -> MeetingReport:
        self._refresh(arena)
        reset(arena, self.receptor_ids)
        if self.composer_cells:
            reset(arena, torch.tensor(self.composer_cells))
        self.periods += 1
        return MeetingReport(self.periods, None, event, name,
                             n_read=int(self.read_mask.sum()),
                             status=status, grow=grow,
                             divisions=n_div, census=census(arena),
                             spawns=spawns, pruned=pruned)

    # ── readouts ──────────────────────────────────────────────────

    def templates(self) -> torch.Tensor:
        """[C, F] buffer-mean templates (matched-filter readout)."""
        return torch.stack([b.mean(0) for b in self.bufs])

    def pockets_of(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample canonical-frame pockets through the substrate path —
        no deposits, no window, frame registry untouched (frozen-world
        eval). Composed dims read from the spawned cells' forward-path
        activations through their static frames."""
        if self.codec is not None:
            x = self.codec(x)
        self.substrate.forward(x)
        sched = self.substrate.scheduler
        qr = self._canonical(sched._last_receptor_q,
                             sched._last_receptor_frame)
        if not self.composer_cells:
            return qr
        act = sched._last_activations[:, self.composer_cells]
        qc = torch.stack([self.specs[k].quantize(act[:, k])
                          for k in range(len(self.specs))], dim=1)
        return torch.cat([qr, qc], dim=1)
