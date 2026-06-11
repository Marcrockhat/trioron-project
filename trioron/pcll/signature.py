"""Signatures — unsupervised schedule discovery.  See spec §10.5.

A period's lock-in state IS its signature: per feature, the mean phasor
S_f = A_f/n_f encodes both the range CENTER (arg S) and the range WIDTH
(|S| — the resultant length; wide ranges → low concentration). A learned
class is a running mean of matched signatures — so templates self-report
their spread and feed straight into resolve.Resolver.

Discovery loop, per period boundary:

  1. coherence gate — no feature clears the matched-K floor → EMPTY, no
     learning (noise can never birth a class).
  2. goodness-of-fit vs each known class — deposit-walk statistics around
     n·T with per-feature variance n_f·v_f, v_f = 1 − |T_cf|². The null is
     TANGENTIAL (narrow-arc deposits vary along the arc): χ²(df ≈ active
     features), std √2·mean — the isotropic null splits true classes
     (found empirically, s029). Criterion via Wilson–Hilferty at
     K_FIT = 4 (one notch above the universal K=3: the fit test repeats
     per known class per period and a false reject births a spurious
     class, while a loose accept is harmless). The (1+1/m) factor accounts
     for the template itself being an m-period estimate.
  3. best accepting class → MATCH: running-mean update
     T ← T + (S − T)/(m+1). No class accepts → BIRTH: the signature
     becomes a new class. New classes are new accumulators — structurally
     zero interference with what is already learned.

Storage note: the running mean/variance machinery here is slated to
migrate onto ManifoldArchive astrocytes (code_dim = 2F; integration phase
I4) — the math stays, the store changes. Promoted from
experiments/progenitor/schedule_learn.py (s029, validated 5-seed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from trioron.core.receptor import N_QUANTA
from .lockin import matched_k

V_FLOOR = (2 * math.pi / N_QUANTA) ** 2 / 6   # one-pocket quantization jitter
K_FIT = 4.0                                   # fit-test z (module doc, step 2)


def _chi2_quantile(df: int, zq: float) -> float:
    """Wilson–Hilferty approximation of the χ²(df) quantile at normal-z zq."""
    a = 2.0 / (9.0 * df)
    return df * (1.0 - a + zq * math.sqrt(a)) ** 3


@dataclass
class LearnedClass:
    name: str
    T: torch.Tensor                  # complex (F,) — running-mean signature
    active: torch.Tensor             # bool (F,) — features that ever deposited
    m: int = 1                       # periods absorbed

    def readable_schedule(self) -> Dict[int, Tuple[int, int]]:
        """Derived range view (center ± 2σ, circular stats) — reporting only."""
        out = {}
        for f in range(len(self.T)):
            if not self.active[f]:
                continue
            R = min(self.T[f].abs().item(), 1.0 - 1e-9)
            q = (math.atan2(self.T[f].imag.item(), self.T[f].real.item())
                 / (2 * math.pi) * N_QUANTA) % N_QUANTA
            sigma_q = math.sqrt(-2 * math.log(R)) / (2 * math.pi) * N_QUANTA
            out[f] = (round(q - 2 * sigma_q), round(q + 2 * sigma_q))
        return out


class ScheduleLearner:
    def __init__(self, n_features: int, k: float = 3.0):
        self.F = n_features
        self.k = k
        self.classes: List[LearnedClass] = []
        self._births = 0

    def observe(self, li) -> Tuple[str, Optional[str]]:
        """One period boundary (`li` = LockInView). Returns (event,
        class_name); event ∈ {'empty', 'match', 'birth'}."""
        if not (li.margin() > matched_k()).any():
            return "empty", None

        A = torch.complex(li.re, li.im)
        best: Optional[LearnedClass] = None
        best_ratio = float("inf")
        for c in self.classes:
            z, crit = self._fit(li, A, c)
            if z <= crit and z / max(crit, 1e-12) < best_ratio:
                best, best_ratio = c, z / max(crit, 1e-12)

        sig = torch.where(li.n > 0, A / li.n.clamp_min(1.0), torch.zeros_like(A))
        if best is None:
            self._births += 1
            c = LearnedClass(f"u{self._births}", sig.clone(), li.n > 0)
            self.classes.append(c)
            return "birth", c.name
        upd = best.active & (li.n > 0)
        best.T[upd] += (sig[upd] - best.T[upd]) / (best.m + 1)
        best.m += 1
        return "match", best.name

    def _fit(self, li, A: torch.Tensor, c: LearnedClass) -> Tuple[float, float]:
        f = c.active & (li.n > 0)
        df = int(f.sum())
        if df == 0:
            return float("inf"), 0.0
        v = (1.0 - c.T[f].abs() ** 2).clamp_min(V_FLOOR)
        z = ((A[f] - li.n[f] * c.T[f]).abs() ** 2).sum().item()
        crit = ((1 + 1 / c.m) * (li.n[f] * v).sum().item()
                * _chi2_quantile(df, K_FIT) / df)
        return z, crit

    def templates(self) -> Dict[str, torch.Tensor]:
        """Drop-in for resolve.Resolver(templates) — the learned world model."""
        return {c.name: c.T for c in self.classes}
