"""Organ blur (plan §10.2 / §11.2; Rocky s067: "baby eyes are blur when they were born";
sharpen "at very slow rate probably relative to her size").

A blur sits in front of one organ.  The organ's senses still form at genesis on
full-resolution readings; from her first moment on, the organ sees each reading in a
small number of steps.  A reading that did not vary at genesis passes unchanged.  The
phasecyte is not changed.

Two kinds of reading, two ladders (Rocky s068):
  a LEVEL (busy, memory, a sentence number, an echo flag)        2 -> 3 -> 5 steps
      steps of equal width over the range the reading showed at genesis:
      low / high, then low / medium / high, then five grades.
  a CHANGE (how a reading moved since the moment before)         2 -> 3 -> 5 -> 7 steps
      2 = same / changed (how MUCH it moved, either way); 3 = down / same / up; then
      finer grades each side.  The range is symmetric about zero and "same" is the
      middle third of it at every rung, so "did not move" is a state of its own from her
      first moment ("Three states is mandatory": never a cut on zero).

"None" (Rocky s068: "yes add none"): a reading that can be SILENT (no text, the window
closed, a channel whose organ does not exist yet) reads exactly zero then, and the organ
sees exactly zero — a state of its own, never put into a step with real values.
A reading that showed no more than TOP distinct values at genesis (rain: no / closed /
yes; an echo flag) is already in steps: it passes unchanged.  Both hold only for a body
born with them; a blur saved before (the first sister's) is read as it was made.

Sharpening is one reading, one rung, at a time.  WHEN is decided by the body
(vee_body.Body: settled long enough for her size); WHICH is decided here: the reading,
not yet at the top of its ladder, that hides the most detail inside its present steps —
the spread of what the world showed around each step's own mean, as a share of the
reading's range.

    b = Blur(pool_part)                   pool_part = the organ's numbers at genesis, [N, d]
    b = Blur(pool_part, change=mask)      mask[j] = reading j is a change
    b = Blur(pool_part, none=mask)        mask[j] = reading j can be silent (exact zero = none)
    b.see(P)                              what the organ sees
    b.sharpen(P_recent, sitting)          one rung for one reading; returns its index, or None
"""
from __future__ import annotations
import torch

FLOOR = 1e-6                # hidden detail below this share of the range is nothing
LEVEL, CHANGE = (2, 3, 5), (2, 3, 5, 7)
TOP = 7                     # a reading with no more distinct values than this at genesis is already in steps


class Blur:
    change = none = None        # a blur saved before changes had their own ladder / before "none": all plain levels

    def __init__(self, pool: torch.Tensor, change: torch.Tensor | None = None, none: torch.Tensor | None = None):
        d = pool.shape[1]
        self.none = None if none is None else none.bool()
        self.change = torch.zeros(d, dtype=torch.bool) if change is None else change.bool()
        self.lo, self.hi = pool.min(0).values, pool.max(0).values
        top = torch.maximum(self.lo.abs(), self.hi.abs())           # a change: symmetric about zero
        self.lo, self.hi = torch.where(self.change, -top, self.lo), torch.where(self.change, top, self.hi)
        self.varies = self.hi > self.lo
        if self.none is not None:                                   # born with "none": readings already in steps pass unchanged
            self.varies &= torch.tensor([pool[:, j].unique().numel() > TOP for j in range(d)])
        self.width = (self.hi - self.lo).clamp(min=1e-12)
        self.steps = torch.full((d,), 2, dtype=torch.long)
        self.history: list[tuple[int, int, int]] = []               # (sitting, reading, steps after)

    def _is_change(self) -> torch.Tensor:
        return torch.zeros_like(self.varies) if self.change is None else self.change

    def _step(self, P: torch.Tensor) -> torch.Tensor:
        """Which step each number falls in.  A change at 2 steps: 0 = same, 1 = changed."""
        k = ((P - self.lo) / self.width * self.steps).floor().clamp(min=0).minimum(self.steps - 1)
        moved = (P.abs() >= self.hi / 3).to(k.dtype)
        return torch.where(self._is_change() & (self.steps == 2), moved, k)

    def see(self, P: torch.Tensor) -> torch.Tensor:
        k = self._step(P)
        seen = self.lo + (k + 0.5) * self.width / self.steps
        seen = torch.where(self._is_change() & (self.steps == 2), k * self.hi * 2 / 3, seen)   # same -> 0, changed -> one value
        keep = ~self.varies if self.none is None else ~self.varies | (self.none & (P == 0))
        return torch.where(keep, P, seen)

    def hidden(self, P: torch.Tensor) -> torch.Tensor:
        """Per reading: the detail its present steps hide, over the moments in ``P``."""
        K, out = self._step(P), torch.zeros(P.shape[1])
        for j in self.varies.nonzero().squeeze(1).tolist():
            live = torch.ones(len(P), dtype=torch.bool) if self.none is None or not self.none[j] else P[:, j] != 0
            for s in K[live, j].unique():
                m = (K[:, j] == s) & live
                out[j] += m.float().mean() * P[m, j].var(unbiased=False)
        return out / self.width ** 2

    def _next(self, j: int) -> int | None:
        ladder = CHANGE if bool(self._is_change()[j]) else LEVEL
        up = [n for n in ladder if n > int(self.steps[j])]
        return up[0] if up else None

    def sharpen(self, P: torch.Tensor, sitting: int) -> int | None:
        h = self.hidden(P)
        for j in h.argsort(descending=True).tolist():
            if h[j] <= FLOOR:
                return None
            n = self._next(j)
            if n is not None:
                self.steps[j] = n
                self.history.append((sitting, j, n))
                return j
        return None
