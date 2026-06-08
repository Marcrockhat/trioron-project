"""Numa / Mima — consolidated-learning currency for the embodied organism.

Adapted from the Aidos design (project-aidos/aidos/numa.py, blueprint §7.1):

  NUMA — a resolved predictive surprise that PERSISTS on re-test: a per-pair
         contrastive-prediction loss-drop that still holds when re-checked on
         held-out replay (the "dream"). Genuine consolidated learning. Food.
  MIMA — a loss-drop that REVERTS on re-test: it improved on the training
         batch but not on held-out replay → overfit/confabulation. Poison.

Net learning = Numa − Mima — anti-confabulation pressure, intrinsic. (My own
smoke-quad=1.000 that reverted at n=3 was, in this vocabulary, a Mima.)

Why this is the right metric (not survival): a reactive heuristic SURVIVES but
accrues zero Numa — it never learns, so no pair-loss ever drops-and-persists.
The organism is "alive" precisely by accruing Numa. Survival is the
precondition (you must live to keep learning); survival alone yields only the
witness baseline.

The five pairs (blueprint §5.3), as scalars in [-1,+1] from the tile world.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from experiments.world.tile_world import FOOD, WATER, BERRY, POISON, FIRE

PAIRS = ("hungry_stuffed", "cold_hot", "threat_safe",
         "reachable_unreachable", "owned_foreign")
N_PAIR = len(PAIRS)


def contrast_targets(w) -> torch.Tensor:
    """The 5 contrastive-pair labels in [-1,+1] from current world state
    (negative = first term). The organism learns to PREDICT these — getting
    better at prediction = resolving predictive surprise = Numa (if it sticks)."""
    s = w.size
    # hungry/stuffed, cold/hot — direct from drives
    hungry_stuffed = 2 * w.energy - 1
    cold_hot = 2 * w.temp - 1
    # threat/safe — predator chebyshev distance (far = safe = +1)
    dpx = min((w.pred[0] - w.px) % s, (w.px - w.pred[0]) % s)
    dpy = min((w.pred[1] - w.py) % s, (w.py - w.pred[1]) % s)
    dist = max(dpx, dpy) / (s / 2)
    threat_safe = max(-1.0, min(1.0, 2 * dist - 1))
    # reachable/unreachable — strength of food/water scent (near = reachable = -1)
    sc = max(float(w._scent(FOOD).abs().sum()), float(w._scent(WATER).abs().sum()))
    reachable_unreachable = max(-1.0, min(1.0, 1 - 4 * sc))   # strong scent → reachable
    # owned/foreign — nearest special tile: poison = foreign (+1), berry/food = owned (-1)
    pois = float(w._scent(POISON).abs().sum()); own = float(w._scent(BERRY).abs().sum())
    owned_foreign = max(-1.0, min(1.0, 4 * (pois - own)))
    return torch.tensor([hungry_stuffed, cold_hot, threat_safe,
                         reachable_unreachable, owned_foreign])


@dataclass
class NumaLedger:
    """Accrues Numa (consolidated, held-out-validated learning) and Mima
    (train-only loss-drops that don't survive the dream). Net = Numa − Mima."""
    drop_eps: float = 0.003
    numa: list = field(default_factory=list)   # (pair_idx, magnitude)
    mima: list = field(default_factory=list)
    _prev_dream: torch.Tensor = None
    _prev_train: torch.Tensor = None

    def checkpoint(self, train_loss: torch.Tensor, dream_loss: torch.Tensor):
        """train_loss/dream_loss: per-pair [5]. Numa = dream loss dropped (held
        out → consolidated). Mima = train dropped but dream did not (overfit)."""
        if self._prev_dream is not None:
            for p in range(N_PAIR):
                dream_drop = float(self._prev_dream[p] - dream_loss[p])
                train_drop = float(self._prev_train[p] - train_loss[p])
                if dream_drop > self.drop_eps:
                    self.numa.append((p, dream_drop))          # held-out gain = Numa
                elif train_drop > self.drop_eps and dream_drop <= 0:
                    self.mima.append((p, train_drop))          # overfit = Mima
        self._prev_dream = dream_loss.clone()
        self._prev_train = train_loss.clone()

    def numa_total(self):
        return sum(m for _, m in self.numa)

    def mima_total(self):
        return sum(m for _, m in self.mima)

    def net(self):
        return self.numa_total() - self.mima_total()

    def summary(self):
        return (f"Numa={self.numa_total():.3f}({len(self.numa)}) "
                f"Mima={self.mima_total():.3f}({len(self.mima)}) "
                f"net={self.net():+.3f}")
