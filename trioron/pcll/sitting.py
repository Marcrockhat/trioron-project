"""The sitting under a cap — which classes the council judges at a boundary.

Opt-in (``MixedStreamController(sitting_cap=True)``); off, every class is
judged at every boundary exactly as before. Written for the incubator
organisms (docs/design/vee_development.md §11.9, Rocky s069: "Square
root"): without it a sitting's work grows with the class count and never
levels off.

Two rules:

  1. A class that received no new member since it was last judged is not
     judged again — its buffer is what it was.
  2. Of the rest, at most ``cap = max(FLOOR, ceil(sqrt(classes) · RATE · φ))``
     are judged, the busiest first (most new members since last judged;
     ties to the class unjudged longest, never-judged first). The others
     wait for the next boundary and keep what they received.

The form is spec §4.3's scaled, dynamic cap; RATE is its lock rate and
φ = clip(1 − BETA · growth / sqrt(classes), 0.1, 1), growth = divisions +
spawns committed at the previous boundary (no frustration term: a
phasecyte period's status is exclusive, not a multiplier).

State is keyed by class NAME (indices move on division / retirement).
``work`` keeps one record per boundary: sleep is measured in work.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

RATE = 0.5     # spec §4.3 s_lock
FLOOR = 4      # classes judged a sitting at the least (Chloe's value, s069)
BETA = 0.3     # spec §4.3 growth-rate weight


class SittingCap:
    def __init__(self, rate: float = RATE, floor: int = FLOOR,
                 beta: float = BETA) -> None:
        self.rate, self.floor, self.beta = rate, floor, beta
        self.fresh: Dict[str, int] = {}    # members received since last judged
        self.last: Dict[str, int] = {}     # boundary at which last judged
        self.growth = 0                    # divisions + spawns, previous boundary
        self.work: List[dict] = []

    def receive(self, names: List[str], member: Optional[torch.Tensor]) -> None:
        """This boundary's membership: row → class index (−1 = refused)."""
        if member is None or not len(member):
            return
        counts = torch.bincount(member[member >= 0], minlength=len(names))
        for k in counts.nonzero().squeeze(1).tolist():
            self.fresh[names[k]] = self.fresh.get(names[k], 0) + int(counts[k])

    def cap(self, n: int) -> int:
        phi = min(1.0, max(0.1, 1.0 - self.beta * self.growth / math.sqrt(max(n, 1))))
        return max(self.floor, math.ceil(math.sqrt(n) * self.rate * phi))

    def choose(self, names: List[str], period: int) -> List[int]:
        """Class indices to judge now, busiest first; marks them judged."""
        live = set(names)
        self.fresh = {k: v for k, v in self.fresh.items() if k in live}
        self.last = {k: v for k, v in self.last.items() if k in live}
        due = [k for k, nm in enumerate(names)
               if self.fresh.get(nm, 0) > 0 or nm not in self.last]
        due.sort(key=lambda k: (-self.fresh.get(names[k], 0),
                                self.last.get(names[k], -1)))
        chosen = due[:self.cap(len(names))]
        for k in chosen:
            self.fresh[names[k]] = 0
            self.last[names[k]] = period
        self.work.append({"period": period, "classes": len(names),
                          "cap": self.cap(len(names)), "due": len(due),
                          "judged": len(chosen), "trials": 0,
                          "divisions": 0, "spawns": 0})
        return chosen

    def done(self, trials: int, divisions: int, spawns: int) -> None:
        """Close this boundary's record; its growth sets the next φ."""
        self.growth = divisions + spawns
        if self.work:
            self.work[-1].update(trials=trials, divisions=divisions, spawns=spawns)
