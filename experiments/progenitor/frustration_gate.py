"""Chance-anchored per-class frustration gate (s028, Rocky's design).

A class is FRUSTRATED only when both hold:
  • its loss has PLATEAUED — training no longer reduces it (the temporal signal), and
  • it is stuck at a BAD level — its cross-entropy is still a large fraction of chance
    (the absolute-magnitude signal).

Cross-entropy is anchored by **chance = log(C)** (the entropy of the uniform prior, and
exactly where an untrained zero-logit class sits — verified). Normalising by it gives

    f_c = CE_c / log(C)      # 1.0 = pure guessing, 0 = solved, >1 = worse than chance

a dataset-agnostic scale with physical meaning, replacing two flawed signals:
  • ``mean + k·std`` (no anchor — floats with the distribution, k is baseless), and
  • the bare loss-plateau detector (asymptote-blind — fires at ANY flat region, even at
    the converged floor; it can't tell "stuck at chance" from "done").

The gate answers WHO is frustrated. A caller decides what to do (widen / deepen / vote)
and when to stop: ``any_high`` false ⇒ every class is solved enough ⇒ converged; some
class high but not plateaued ⇒ keep TRAINING, don't grow yet; ``worst`` not None ⇒ grow
for that class. ``eps_improve`` is the per-class CE drop (over one train block) below
which a class counts as no-longer-improving.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GateResult:
    f: list[float]            # per-class CE / log(C)   (1.0 = chance)
    plateaued: list[bool]     # per-class: CE stopped dropping since last step
    frustrated: list[bool]    # plateaued AND f > tau
    worst: int | None         # most-frustrated class (highest f among frustrated), or None
    max_f: float              # max f over all classes
    any_high: bool            # max_f > tau  → some class still stuck at a bad level


class FrustrationGate:
    def __init__(self, n_classes: int, tau: float = 0.5, eps_improve: float = 0.01):
        self.logC = math.log(n_classes)
        self.tau = tau
        self.eps = eps_improve
        self.prev: list[float] | None = None

    def step(self, resid) -> GateResult:
        """Feed the current per-class mean cross-entropy; returns the gate verdict and
        advances the plateau reference. First call has no history → nothing plateaued."""
        r = [float(v) for v in resid]
        n = len(r)
        f = [r[i] / self.logC for i in range(n)]
        if self.prev is None:
            plateaued = [False] * n
        else:
            plateaued = [(self.prev[i] - r[i]) < self.eps for i in range(n)]
        frustrated = [plateaued[i] and f[i] > self.tau for i in range(n)]
        self.prev = r
        worst = max((c for c in range(n) if frustrated[c]),
                    key=lambda c: f[c], default=None)
        return GateResult(f=f, plateaued=plateaued, frustrated=frustrated,
                          worst=worst, max_f=max(f), any_high=max(f) > self.tau)
