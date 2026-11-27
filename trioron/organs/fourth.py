"""The fourth organ — a division is tried here before a working organ carries it out.

Rocky, s071: "more classes means more creativity though but it will raise a
confused organism. Can you raise another organism to chew on this classes
instead? The fourth organs?"  s072: the living organ, as drafted in
docs/design/vee_development.md §14, "gate reads both".

A working organ (``mixed.court = FourthOrgan()``, one chamber per organ: each
organ reads its own numbers, so held classes of different organs are never
compared) that finds a division does not carry it out. It hands the candidate
over (`hear`) and stays whole: the parent is replayed and classifies as before.

Here the two children live as a held pair, fed by the WORLD only (`receive`:
every world moment the working organ gives the parent falls to one child or
the other, by the children's templates over all columns). No replay ever
enters. The pair is judged by the Numa rule (organs/numa.py): two coherent
batches of BATCH world members per child on the split column. Then

  * NUMA   the pair is handed back from `hear` and the working organ divides
           that parent, once and for good (`carried_out`).
  * a batch fails, or a child goes unfed for TRIAL boundaries: the pair falls
           ASLEEP. Its record is kept (no members); the same parent and column
           is not heard again for TRIAL boundaries, then it may be WOKEN.

Held children may divide again in here, on world members only (`inner`
pairs, to DEPTH). When the outer pair is carried out its children's inner
pairs become pairs of the new real classes. At most one living pair per class
and column.

`refine` gives the fine reading: the class a moment falls in, followed by the
held children it falls to. A gate may read it beside the working organ's own.

BATCH / BATCHES / TRIAL are the Numa rule's; KEEP, DEPTH and TRIES are Chloe's
values (s072).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ..pcll.division import GAIN_D, MIN_MEMBERS, NULL_SPLIT, try_divide
from .numa import BATCH, BATCHES, TRIAL

KEEP = 120    # world members a held child keeps (enough to be judged for a division)
DEPTH = 3     # how deep held children may divide in here
TRIES = 4     # columns tried for an inner division (the working organs' own number)


def _pair(Za: torch.Tensor, Zb: torch.Tensor, dim: int, floor: float,
          period: int, depth: int, woken: bool = False) -> dict:
    return {"dim": int(dim), "floor": float(floor),
            "T": torch.stack([Za.mean(0), Zb.mean(0)]), "born": period,
            "n": [0, 0], "s": [0j, 0j], "total": [0, 0], "passed": 0,
            "last": [period, period], "depth": depth, "woken": woken,
            "settled": False,
            "buf": [Za[:0].clone(), Zb[:0].clone()],
            "inner": [{}, {}], "rest": [{}, {}]}


def _floor(Z: torch.Tensor, dim: int) -> float:
    return max(float(Z.mean(0).abs()[dim]) + GAIN_D, NULL_SPLIT)


def _side(p: dict, Z: torch.Tensor) -> torch.Tensor:
    """Which child each row falls to (0 / 1): matched filter over the pair's columns."""
    T = p["T"]
    return (Z[:, :T.shape[1], None] * T.conj().T[None]).real.sum(1).argmax(1)


class FourthOrgan:
    def __init__(self, batch: int = BATCH, batches: int = BATCHES,
                 trial: int = TRIAL, keep: int = KEEP, depth: int = DEPTH) -> None:
        self.batch, self.batches, self.trial = batch, batches, trial
        self.keep, self.depth = keep, depth
        self.held: Dict[str, Dict[int, dict]] = {}     # parent class → split column → living pair
        self.rest: Dict[Tuple[str, int], int] = {}     # (parent, column) → boundary it may be heard again
        self.asleep: List[dict] = []                   # one record per pair that fell asleep
        self._carry: Dict[str, dict] = {}              # parent → the pair being carried out now
        self.counts = {k: 0 for k in (
            "heard", "held", "numa", "carried", "failed", "starved", "void",
            "gone", "woken", "woken_numa", "inner", "inner_numa", "inner_dropped")}
        self.log: List[tuple] = []                     # (boundary, event, parent, column)

    # ---- the world feeds the held ------------------------------------------------
    def receive(self, names: List[str], member: Optional[torch.Tensor],
                Z: torch.Tensor, period: int) -> None:
        if member is None or not len(member) or not self.held:
            return
        index = {nm: k for k, nm in enumerate(names)}
        for parent, pairs in self.held.items():
            k = index.get(parent)
            if k is None:
                continue
            rows = Z[member == k]
            if len(rows):
                self._feed(pairs, rows, period)

    def _feed(self, pairs: Dict[int, dict], rows: torch.Tensor, period: int) -> None:
        for p in list(pairs.values()):
            F = p["T"].shape[1]
            if rows.shape[1] < F:
                continue
            side = _side(p, rows)
            for i in (0, 1):
                r = rows[side == i][:, :F]
                if not len(r):
                    continue
                p["n"][i] += len(r)
                p["total"][i] += len(r)
                p["s"][i] += complex(r[:, p["dim"]].sum())
                p["last"][i] = period
                p["buf"][i] = torch.cat([p["buf"][i], r])[-self.keep:]
                self._feed(p["inner"][i], r, period)
                self._grow_inner(p, i, period)

    def _grow_inner(self, p: dict, i: int, period: int) -> None:
        """A held child divides again in here, on what the world gave it."""
        b = p["buf"][i]
        if p["depth"] >= self.depth or len(b) < MIN_MEMBERS:
            return
        v = try_divide(b, tries=TRIES)
        if v is None:
            return
        side, d = v
        if d in p["inner"][i] or p["rest"][i].get(d, -1) > period:
            return
        p["inner"][i][d] = _pair(b[~side], b[side], d, _floor(b, d), period, p["depth"] + 1)
        self.counts["inner"] += 1

    # ---- the judgment --------------------------------------------------------------
    def _judge(self, p: dict, period: int) -> Optional[str]:
        if p["settled"]:
            return "numa"
        if min(p["n"]) >= self.batch:
            r = min(abs(p["s"][i]) / p["n"][i] for i in (0, 1))
            if r <= p["floor"]:
                return "failed"
            p["passed"] += 1
            p["n"], p["s"] = [0, 0], [0j, 0j]
            if p["passed"] >= self.batches:
                p["settled"] = True
                return "numa"
        if period - min(p["last"]) >= self.trial:
            return "starved"
        return None

    def _judge_inner(self, p: dict, period: int) -> None:
        for i in (0, 1):
            for d, q in list(p["inner"][i].items()):
                self._judge_inner(q, period)
                if q["settled"]:
                    continue
                v = self._judge(q, period)
                if v == "numa":
                    self.counts["inner_numa"] += 1
                elif v is not None:
                    del p["inner"][i][d]
                    p["rest"][i][d] = period + self.trial
                    self.counts["inner_dropped"] += 1

    def _sleep(self, parent: str, p: dict, period: int, why: str) -> None:
        del self.held[parent][p["dim"]]
        if not self.held[parent]:
            del self.held[parent]
        self.rest[(parent, p["dim"])] = period + self.trial
        self.asleep.append({"period": period, "born": p["born"], "parent": parent,
                            "dim": p["dim"], "floor": p["floor"], "why": why,
                            "passed": p["passed"], "world": list(p["total"]),
                            "T": p["T"], "woken": p["woken"]})
        self.counts[why] += 1
        self.log.append((period, why, parent, p["dim"]))

    def hear(self, parents: List[str], bufs: List[torch.Tensor], candidates: list,
             period: int) -> list:
        """The working organ's candidates [(class index, (side, column))] are taken in;
        returned are the divisions that have settled here, to be carried out now, with the
        side of every member of the parent's present buffer."""
        index = {nm: k for k, nm in enumerate(parents)}
        for parent in [x for x in self.held if x not in index]:      # the parent left by another road
            self.counts["gone"] += len(self.held.pop(parent))
        for k, (side, d) in candidates:
            name, b = parents[k], bufs[k]
            self.counts["heard"] += 1
            if d in self.held.get(name, {}) or self.rest.get((name, d), -1) > period:
                continue
            woken = (name, d) in self.rest
            self.held.setdefault(name, {})[d] = _pair(
                b[~side], b[side], d, _floor(b, d), period, 1, woken)
            self.counts["held"] += 1
            self.counts["woken"] += woken
            self.log.append((period, "woken" if woken else "held", name, d))
        out = []
        for parent in list(self.held):
            for d, p in sorted(self.held.get(parent, {}).items()):
                self._judge_inner(p, period)
                v = self._judge(p, period)
                if v is None:
                    continue
                if v != "numa":
                    self._sleep(parent, p, period, v)
                    continue
                if parent in self._carry:
                    continue                                         # one division of a parent at a time
                b = bufs[index[parent]]
                side = _side(p, b) == 1 if b.shape[1] >= p["T"].shape[1] else None
                if side is None or not bool(side.any()) or bool(side.all()):
                    self._sleep(parent, p, period, "void")           # nothing of the parent on one side
                    continue
                self._carry[parent] = p
                out.append((index[parent], (side, d)))
        return out

    def carried_out(self, parent: str, a: str, b: str) -> None:
        """The working organ divided ``parent`` into ``a`` (child 0) and ``b`` (child 1)."""
        p = self._carry.pop(parent, None)
        if p is None:
            return
        self.counts["numa"] += 1
        self.counts["carried"] += 1
        self.counts["woken_numa"] += p["woken"]
        self.log.append((p["born"], "numa", parent, p["dim"]))
        others = self.held.pop(parent, {})
        self.counts["gone"] += len(others) - 1                       # the parent's other columns
        for i, child in enumerate((a, b)):                           # inner pairs follow the new real classes
            if p["inner"][i]:
                self.held[child] = p["inner"][i]
                self._lift(p["inner"][i])

    def _lift(self, pairs: Dict[int, dict]) -> None:
        for q in pairs.values():
            q["depth"] -= 1
            for i in (0, 1):
                self._lift(q["inner"][i])

    # ---- a composed column left the organ ---------------------------------------
    def pruned(self, col: int) -> None:
        self.rest = {(n, d - (d > col)): t for (n, d), t in self.rest.items() if d != col}
        for parent in list(self.held):
            self.held[parent] = self._drop_col(self.held[parent], col)
            if not self.held[parent]:
                del self.held[parent]

    def _drop_col(self, pairs: Dict[int, dict], col: int) -> Dict[int, dict]:
        out = {}
        for d, p in pairs.items():
            F = p["T"].shape[1]
            if col < F:
                if d == col:
                    self.counts["void"] += 1
                    continue
                keep = [j for j in range(F) if j != col]
                p["T"] = p["T"][:, keep]
                p["buf"] = [x[:, keep] for x in p["buf"]]
                p["dim"] = d - (d > col)
                for i in (0, 1):
                    p["inner"][i] = self._drop_col(p["inner"][i], col)
                    p["rest"][i] = {e - (e > col): t for e, t in p["rest"][i].items() if e != col}
            out[p["dim"]] = p
        return out

    # ---- the fine reading ---------------------------------------------------------
    def refine(self, names: List[str], member: torch.Tensor, Z: torch.Tensor) -> List[str]:
        """Per row: the working class's name, then the held children the row falls to."""
        out = [names[int(k)] if k >= 0 else "" for k in member]
        for parent, pairs in self.held.items():
            if parent not in names:
                continue
            rows = (member == names.index(parent)).nonzero().flatten()
            if len(rows):
                for j, tail in zip(rows.tolist(), self._path(pairs, Z[rows])):
                    out[j] += tail
        return out

    def _path(self, pairs: Dict[int, dict], Z: torch.Tensor) -> List[str]:
        tail = [""] * len(Z)
        for d, p in sorted(pairs.items()):
            if Z.shape[1] < p["T"].shape[1]:
                continue
            side = _side(p, Z)
            for i in (0, 1):
                rows = (side == i).nonzero().flatten()
                if not len(rows):
                    continue
                sub = self._path(p["inner"][i], Z[rows])
                for j, s in zip(rows.tolist(), sub):
                    tail[j] += f"/{d}:{i}{s}"
        return tail

    # ---- what it holds --------------------------------------------------------------
    def size(self) -> dict:
        def walk(pairs, acc):
            for p in pairs.values():
                acc["pairs"] += 1
                acc["settled_inner"] += p["settled"]
                acc["members"] += sum(len(x) for x in p["buf"])
                for i in (0, 1):
                    walk(p["inner"][i], acc)
        acc = {"pairs": 0, "settled_inner": 0, "members": 0}
        for pairs in self.held.values():
            walk(pairs, acc)
        return {"parents": len(self.held), "top": sum(len(v) for v in self.held.values()),
                **acc, "asleep": len(self.asleep)}
