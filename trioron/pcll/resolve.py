"""Resolution — matched filter + parameter-free floors.  See spec §10.4.

Per-class evidence is a matched filter in phasor space. A class's range
schedule gives a per-feature template phasor

    T_cf = mean of e^{iθ(q)} over the range's INTERIOR quanta

— a point range has |T| = 1, a wide (non-discriminative) range
self-down-weights, and reference pockets (q = 0/1000) contribute nothing,
mirroring evidence_mask. Evidence against the lock-in accumulator A
(complex, per feature):

    E_c = Σ_f Re(A_f · conj(T_cf))

Under noise A_f is a 2-D random walk (per-component variance n_f/2), so
E_c is exactly Gaussian with var σ_c² = Σ_f |T_cf|²·n_f/2 — both
resolution tests are parameter-free √-statistics:

  EMPTY      — no class clears K·σ_c: nothing in the stream (terminal AND
               stress, design §6)
  FRUSTRATED — evidence clears the floor but the top-two gap sits inside
               ITS noise band, σ_gap² = Σ_f |T_top,f − T_2nd,f|²·n_f/2
  RESOLVED   — the top class clearly leads (gap margin > K)

The output is graded: the full ranking of per-class margins, never a
one-hot. Promoted from experiments/progenitor/resolve.py (s029, validated
200/200 on the resolution trichotomy).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch

from trioron.core.receptor import N_QUANTA

Range = Tuple[int, int]            # inclusive (lo, hi) in quantum units
Schedule = Dict[int, List[Range]]  # feature index -> list of ranges

RESOLVED = "resolved"
FRUSTRATED = "frustrated"
EMPTY = "empty"


def template(schedule: Schedule, n_features: int,
             features: Optional[Set[int]] = None) -> torch.Tensor:
    """Per-feature template phasor for one class. `features` restricts which
    receptors the composer reads (discrimination growth = widening this set)."""
    T = torch.zeros(n_features, dtype=torch.complex64)
    for f in range(n_features):
        if features is not None and f not in features:
            continue
        qs = [q for lo, hi in schedule.get(f, ())
              for q in range(lo, hi + 1) if 0 < q < N_QUANTA]
        if qs:
            theta = 2 * math.pi * torch.tensor(qs, dtype=torch.float32) / N_QUANTA
            T[f] = torch.complex(theta.cos().mean(), theta.sin().mean())
    return T


@dataclass
class Resolution:
    status: str                          # RESOLVED | FRUSTRATED | EMPTY
    winner: Optional[str]                # top class (tentative when FRUSTRATED)
    ranking: List[Tuple[str, float]]     # graded output, margin-sorted desc
    gap_margin: Optional[float]          # top-vs-second, in gap-floor units


class Resolver:
    def __init__(self, templates: Dict[str, torch.Tensor]):
        self.templates = templates

    @classmethod
    def from_schedules(cls, classes: Dict[str, Schedule], n_features: int,
                       features: Optional[Set[int]] = None) -> "Resolver":
        return cls({name: template(s, n_features, features)
                    for name, s in classes.items()})

    def resolve(self, li, k_abs: float = 3.0, k_gap: float = 3.0) -> Resolution:
        """`li` is anything with .re/.im/.n per feature (LockInView)."""
        A = torch.complex(li.re, li.im)

        def evidence(T: torch.Tensor) -> Tuple[float, float]:
            e = (A * T.conj()).real.sum().item()
            var = ((T.abs() ** 2) * li.n / 2).sum().item()
            return e, math.sqrt(max(var, 1e-12))

        margins = {}
        for name, T in self.templates.items():
            e, sigma = evidence(T)
            margins[name] = e / sigma if sigma > 1e-6 else 0.0
        ranking = sorted(margins.items(), key=lambda kv: kv[1], reverse=True)

        if not ranking or ranking[0][1] < k_abs:
            return Resolution(EMPTY, None, ranking, None)
        if len(ranking) == 1:
            return Resolution(RESOLVED, ranking[0][0], ranking, None)

        top, second = ranking[0][0], ranking[1][0]
        gap_e, gap_sigma = evidence(self.templates[top] - self.templates[second])
        gap_margin = gap_e / gap_sigma if gap_sigma > 1e-6 else 0.0
        status = RESOLVED if gap_margin > k_gap else FRUSTRATED
        return Resolution(status, top, ranking, gap_margin)
