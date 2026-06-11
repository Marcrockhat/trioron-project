"""Signatures — unsupervised schedule discovery over a DYNAMIC perception
layer.  See spec §10.5 and integration decision D10.

A period's lock-in state IS its signature: per feature, the mean phasor
S_f = A_f/n_f encodes both the range CENTER (arg S) and the range WIDTH
(|S| — the resultant length). A learned class is a running mean of
matched signatures and feeds straight into resolve.Resolver.

Discovery loop, per period boundary:

  1. coherence gate — no feature clears the matched-K floor → EMPTY, no
     learning (noise can never birth a class).
  2. goodness-of-fit vs each known class — deposit-walk statistics around
     n·T with per-feature variance n_f·v_f, v_f = 1 − |T_cf|². The null
     is TANGENTIAL: χ²(df ≈ active features), Wilson–Hilferty at
     K_FIT = 4; the per-dim (1 + 1/m_f) factor accounts for each
     dimension's template being an m_f-period estimate.
  3. best accepting class → MATCH: per-dim running-mean update
     T_f ← T_f + (S_f − T_f)/(m_f + 1). No class accepts → BIRTH.
     New classes are new accumulators — zero structural interference.

Dynamic-perception machinery (D10, s030):

  PER-DIM COUNTS — m is a per-feature vector. A receptor added after a
  class existed accumulates from m_f = 0 at its own rate; a scalar m
  would divide the new dim's first evidence by the class's whole history
  and freeze it.

  SHADOWS — dims that deposit while NOT read (or not yet trusted)
  accumulate in the same (T, m) arrays with active = False: excluded
  from the fit, the matched filter, and templates(), but ready. This is
  the default extension path (the s029 recruit-without-relearn): a
  deliberate recruit() promotes a shadow dim instantly (m_f ≥ 1); the
  automatic sweep promotes at m_f ≥ PROMOTE_PATIENCE matched periods —
  the symmetric counterpart of habituation's RETIRE_PATIENCE.

  FRAME STAMPS + TRANSLATION — pocket coordinates are not stable: the
  per-sample gain frame moves when the continuous receptor set changes.
  Each class carries the frame statistics (E[lo], E[hi]) of its matched
  periods; at READ time the signature is translated closed-form from its
  stamp to the current period's frame (q' = a·q + b is exact on the
  (center, width) parametrization: θ' = a·θ + 2πb/Q, R' = R^(a²)).
  Same-class comparisons are near-identity; structural shifts (a new
  receptor in the frame) are absorbed by the map. Discrete dims never
  translate — labeled lines bypass the gain frame by construction. On
  match the stored signature is moved to the current frame and
  restamped; residual calibration error washes out in the running mean.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from trioron.core.receptor import N_QUANTA
from .lockin import matched_k

V_FLOOR = (2 * math.pi / N_QUANTA) ** 2 / 6   # one-pocket quantization jitter
K_FIT = 4.0                                   # fit-test z (module doc, step 2)
PROMOTE_PATIENCE = 3                          # matched periods before a shadow dim
                                              # auto-promotes (mirror of RETIRE_PATIENCE)

Frame = Tuple[float, float]                   # (E[lo], E[hi]) of a period's gain frames


def _chi2_quantile(df: int, zq: float) -> float:
    """Wilson–Hilferty approximation of the χ²(df) quantile at normal-z zq."""
    a = 2.0 / (9.0 * df)
    return df * (1.0 - a + zq * math.sqrt(a)) ** 3


def translate(T: torch.Tensor, stamp: Optional[Frame], frame: Optional[Frame],
              discrete: torch.Tensor) -> torch.Tensor:
    """Translate a signature from its stamp's pocket coordinates to the
    current frame's (D10). Identity when frames match, either is unknown,
    or per dim where the feature is discrete (labeled lines are
    frame-free)."""
    if stamp is None or frame is None:
        return T
    lo0, hi0 = stamp
    lo1, hi1 = frame
    span1 = hi1 - lo1
    if span1 <= 1e-9:
        return T
    a = (hi0 - lo0) / span1
    b = N_QUANTA * (lo0 - lo1) / span1
    if abs(a - 1.0) < 1e-3 and abs(b) < 0.5:
        return T
    R = T.abs().clamp(max=1.0 - 1e-9)
    # the affine map acts on the UNWRAPPED pocket coordinate q ∈ [0, 1000)
    # — torch.angle's signed principal value would scale the wrong lift
    # for centers past the half-period (q > 500)
    q_c = (torch.angle(T) / (2 * math.pi) * N_QUANTA) % N_QUANTA
    theta_n = 2 * math.pi * (a * q_c + b) / N_QUANTA
    Tn = (R ** (a * a)) * torch.exp(1j * theta_n)
    return torch.where(discrete, T, Tn.to(T.dtype))


@dataclass
class LearnedClass:
    name: str
    T: torch.Tensor                  # complex (F,) — running-mean signature
    m: torch.Tensor                  # float (F,) — PER-DIM periods absorbed
    active: torch.Tensor             # bool (F,) — read evidence (False = shadow)
    frame: Optional[Frame] = None    # frame stamp of the stored coordinates
    cell_id: int = -1                # astrocyte arena cell (-1 = not yet synced)

    def readable_schedule(self) -> Dict[int, Tuple[int, int]]:
        """Derived range view (center ± 2σ, circular stats) — reporting only."""
        out = {}
        for f in range(len(self.T)):
            if not self.active[f]:
                continue
            R = min(self.T[f].abs().item(), 1.0 - 1e-9)
            q = (math.atan2(self.T[f].imag.item(), self.T[f].real.item())
                 / (2 * math.pi) * N_QUANTA) % N_QUANTA
            sigma_q = math.sqrt(-2 * math.log(max(R, 1e-12))) / (2 * math.pi) * N_QUANTA
            out[f] = (round(q - 2 * sigma_q), round(q + 2 * sigma_q))
        return out


class ScheduleLearner:
    def __init__(self, n_features: int, k: float = 3.0):
        self.F = n_features
        self.k = k
        self.classes: List[LearnedClass] = []
        self._births = 0
        # which dims are discrete labeled lines (frame-free) — set by the
        # controller from arena.receptor_levels
        self.levels = torch.zeros(n_features, dtype=torch.int32)

    def set_levels(self, levels: torch.Tensor) -> None:
        self.levels = levels.clone()

    @property
    def _discrete(self) -> torch.Tensor:
        return self.levels >= 2

    # ── the boundary step ─────────────────────────────────────────

    def observe(self, li, frame: Optional[Frame] = None,
                shadow=None) -> Tuple[str, Optional[str]]:
        """One period boundary. `li` = READ view; `shadow` = optional FULL
        view (unread dims accumulate as shadows). Returns (event, name)."""
        if not (li.margin() > matched_k()).any():
            return "empty", None

        A = torch.complex(li.re, li.im)
        best: Optional[LearnedClass] = None
        best_ratio = float("inf")
        for c in self.classes:
            z, crit = self._fit(li, A, c, frame)
            if z <= crit and z / max(crit, 1e-12) < best_ratio:
                best, best_ratio = c, z / max(crit, 1e-12)

        sig = torch.where(li.n > 0, A / li.n.clamp_min(1.0), torch.zeros_like(A))
        sh_sig, sh_new = self._shadow_signature(li, shadow)

        if best is None:
            return self._birth(li, sig, sh_sig, sh_new, frame)
        return self._match(best, li, sig, sh_sig, sh_new, frame)

    def _shadow_signature(self, li, shadow):
        """Mean phasor + mask for dims depositing OUTSIDE the read view."""
        if shadow is None:
            z = torch.zeros(self.F, dtype=torch.complex64)
            return z, torch.zeros(self.F, dtype=torch.bool)
        A_s = torch.complex(shadow.re, shadow.im)
        sh_new = (shadow.n > 0) & (li.n == 0)
        sig = torch.where(sh_new, A_s / shadow.n.clamp_min(1.0),
                          torch.zeros_like(A_s))
        return sig, sh_new

    def _birth(self, li, sig, sh_sig, sh_new, frame):
        self._births += 1
        read = li.n > 0
        T = torch.where(read, sig, sh_sig)
        m = (read | sh_new).to(torch.float32)
        c = LearnedClass(f"u{self._births}", T.clone(), m, read.clone(),
                         frame=frame)
        self.classes.append(c)
        return "birth", c.name

    def _match(self, c: LearnedClass, li, sig, sh_sig, sh_new, frame):
        # move the stored signature into the current frame, restamp
        c.T = translate(c.T, c.frame, frame, self._discrete)
        c.frame = frame
        # per-dim running mean — read dims (active or warming) ...
        upd = li.n > 0
        c.T[upd] += (sig[upd] - c.T[upd]) / (c.m[upd] + 1)
        c.m[upd] += 1
        # ... and shadow dims (sensed but not read)
        c.T[sh_new] += (sh_sig[sh_new] - c.T[sh_new]) / (c.m[sh_new] + 1)
        c.m[sh_new] += 1
        # automatic promotion: a READ dim with enough matched periods of
        # evidence joins the class's read template
        c.active |= upd & (c.m >= PROMOTE_PATIENCE)
        return "match", c.name

    def _fit(self, li, A: torch.Tensor, c: LearnedClass,
             frame: Optional[Frame]) -> Tuple[float, float]:
        f = c.active & (li.n > 0)
        df = int(f.sum())
        if df == 0:
            return float("inf"), 0.0
        T = translate(c.T, c.frame, frame, self._discrete)
        v = (1.0 - T[f].abs() ** 2).clamp_min(V_FLOOR)
        z = ((A[f] - li.n[f] * T[f]).abs() ** 2).sum().item()
        crit = (((1 + 1 / c.m[f]) * li.n[f] * v).sum().item()
                * _chi2_quantile(df, K_FIT) / df)
        return z, crit

    # ── readouts ──────────────────────────────────────────────────

    def templates(self, frame: Optional[Frame] = None) -> Dict[str, torch.Tensor]:
        """Drop-in for resolve.Resolver — read (active) evidence only,
        translated into the current frame."""
        out = {}
        for c in self.classes:
            T = translate(c.T, c.frame, frame, self._discrete)
            out[c.name] = torch.where(c.active, T, torch.zeros_like(T))
        return out

    def promote(self, col: int) -> int:
        """Deliberate recruit (spec §10.6): instantly extend every class
        that holds ANY shadow evidence on this dim (m_f ≥ 1) — the s029
        recruit-without-relearn. Returns how many classes extended."""
        n = 0
        for c in self.classes:
            if not bool(c.active[col]) and float(c.m[col]) >= 1.0:
                c.active[col] = True
                n += 1
        return n
