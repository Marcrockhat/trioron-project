"""Growth must pay — the Numa rule for class growth.

Opt-in (``mixed.numa = NumaLedger()``); off, a division is final the moment
it is made, exactly as before. Written for the incubator organisms
(docs/design/vee_development.md §13, Rocky s071: "Yes to all, build it"):
a class's buffer is mostly its own replays, so without the rule a class can
divide on what the world never showed, and its children are fed replays in
turn.

The rule:

  1. BORN ON CREDIT. The two children of a division are a provisional pair
     (`born`): the pair, the dimension it split on, the coherence the split
     had to beat, the boundary of birth.
  2. ONLY THE WORLD PAYS. Each child's evidence is the members the world
     gave it since birth (`receive`): their count and their summed phasor on
     the split dimension. Replays never count.
  3. NO GROWTH ON CREDIT. A provisional class is not judged for division
     and receives no replays (`on_trial`, read by `mixed.py`).
  4. NUMA. Both children gather a batch of BATCH world members, each batch
     coherent on the split dimension above the mark; then a second batch of
     NEW members, coherent again — the re-test, evidence the division never
     saw. The pair are then ordinary classes.
  5. MIMA. A batch fails the mark, or a child receives nothing from the
     world for TRIAL boundaries on end. The pair is handed back
     (`verdicts`) and `mixed.py` joins the children again by the fold's
     routine: nothing lived is deleted, a boundary is taken back. A pair
     the world keeps feeding stays on trial until it has its batches,
     however long that takes (Rocky s072; in s071 the pair had TRIAL
     boundaries in all, and one division in 406 settled).

A pair one of whose children left by another road (consolidation emptied
it, the fold took it) is closed as "gone": the trial is void, nothing is
joined. BATCH is `division.MIN_CHILD` (the smallest child a division
allows); BATCHES and TRIAL are Chloe's values (s071).

State is keyed by class NAME (indices move on division / retirement).
``log`` keeps every event; ``taken_back`` keeps each Mima's record for
whatever is later built to hold what was taken back (plan §13.6).

``NumaLedger(witness=True)`` judges nothing: the rule stays off and the
ledger only writes down each division's split dimension (`dims`), so a
plain organism's divisions can be read beside a ruled one's.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..pcll.division import MIN_CHILD

BATCH = MIN_CHILD   # world members a child must gather for one reading
BATCHES = 2         # readings that must pass: the second is the re-test
TRIAL = 60          # boundaries a child may go unfed by the world (30 simulated days)


class NumaLedger:
    def __init__(self, batch: int = BATCH, batches: int = BATCHES,
                 trial: int = TRIAL, witness: bool = False) -> None:
        self.batch, self.batches, self.trial = batch, batches, trial
        self.witness = witness
        self.dims: Dict[str, List[int]] = {"born": [], "numa": []}
        self.pairs: Dict[str, dict] = {}       # first child's name → the trial
        self.child: Dict[str, str] = {}        # child name → its pair's key
        self.counts = {"born": 0, "numa": 0, "mima": 0, "gone": 0}
        self.log: List[tuple] = []             # (boundary, event, a, b)
        self.taken_back: List[dict] = []       # one record per Mima
        self.seen = self.refused = 0           # world rows met / claimed by no class

    def on_trial(self, name: str) -> bool:
        return name in self.child

    def born(self, period: int, a: str, b: str, dim: int, floor: float) -> None:
        if hasattr(self, "dims"):
            self.dims["born"].append(int(dim))
        if getattr(self, "witness", False):
            self.counts["born"] += 1
            return
        self.pairs[a] = {"a": a, "b": b, "dim": int(dim), "floor": float(floor),
                         "born": period, "passed": 0, "n": [0, 0],
                         "s": [0j, 0j], "total": [0, 0], "void": False,
                         "last": [period, period], "fed": [False, False]}
        self.child[a] = self.child[b] = a
        self.counts["born"] += 1
        self.log.append((period, "born", a, b))

    def receive(self, names: List[str], member: Optional[torch.Tensor],
                Z: torch.Tensor) -> None:
        """This boundary's membership of the WORLD's rows: row → class
        index (−1 = refused). Call before anything moves the indices."""
        if member is None or not len(member) or getattr(self, "witness", False):
            return
        self.seen = getattr(self, "seen", 0) + len(member)
        self.refused = getattr(self, "refused", 0) + int((member < 0).sum())
        if not self.pairs:
            return
        index = {nm: k for k, nm in enumerate(names)}
        for p in self.pairs.values():
            if p["dim"] >= Z.shape[1]:
                continue
            for i, nm in enumerate((p["a"], p["b"])):
                k = index.get(nm)
                if k is None:
                    continue
                rows = member == k
                n = int(rows.sum())
                if n:
                    p["n"][i] += n
                    p["total"][i] += n
                    p.setdefault("fed", [False, False])[i] = True
                    p["s"][i] += complex(Z[rows, p["dim"]].sum())

    def pruned(self, col: int) -> None:
        """A composed column left every buffer: a pair split on it has lost
        its evidence basis (taken back at the next verdict); dims to its
        right shift left."""
        for p in self.pairs.values():
            if p["dim"] == col:
                p["void"] = True
            elif p["dim"] > col:
                p["dim"] -= 1

    def _close(self, key: str, period: int, event: str) -> dict:
        p = self.pairs.pop(key)
        self.child.pop(p["a"], None)
        self.child.pop(p["b"], None)
        self.counts[event] += 1
        self.log.append((period, event, p["a"], p["b"]))
        return p

    def verdicts(self, period: int, live: List[str]) -> List[Tuple[str, str]]:
        """Settle what can be settled. Returns the Mima pairs to join, as
        (class to fold, class that keeps): the child the world fed more
        keeps."""
        alive, back = set(live), []
        for key in list(self.pairs):
            p = self.pairs[key]
            if p["a"] not in alive or p["b"] not in alive:
                self._close(key, period, "gone")
                continue
            last = p.setdefault("last", [p["born"], p["born"]])
            for i, fed in enumerate(p.setdefault("fed", [False, False])):
                if fed:
                    last[i], p["fed"][i] = period, False
            why = "void" if p["void"] else None
            mima = p["void"]
            if not mima and min(p["n"]) >= self.batch:
                r = min(abs(p["s"][i]) / p["n"][i] for i in (0, 1))
                if r > p["floor"]:
                    p["passed"] += 1
                    p["n"], p["s"] = [0, 0], [0j, 0j]
                    if p["passed"] >= self.batches:
                        self._close(key, period, "numa")
                        if hasattr(self, "dims"):
                            self.dims["numa"].append(p["dim"])
                        continue
                else:
                    mima, why = True, "failed"
            if not mima and period - min(last) >= self.trial:
                mima, why = True, "starved"
            if mima:
                self._close(key, period, "mima")
                keep, drop = ((p["a"], p["b"]) if p["total"][0] >= p["total"][1]
                              else (p["b"], p["a"]))
                self.taken_back.append({
                    "period": period, "born": p["born"], "kept": keep,
                    "folded": drop, "dim": p["dim"], "floor": p["floor"],
                    "passed": p["passed"], "world": list(p["total"]),
                    "why": why})
                back.append((drop, keep))
        return back

    @property
    def net(self) -> int:
        return self.counts["numa"] - self.counts["mima"]
