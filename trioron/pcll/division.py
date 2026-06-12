"""Division — class discovery by frustration on a mixed stream.  [D12]

See docs/design/mixed_stream_growth.md §2 (spec §10 amendment pending,
M6). The regime (Rocky, s031): genesis births ONE blurred world-class;
division-by-frustration is the class-DISCOVERY mechanism, not a
refinement. Probe-validated in experiments/progenitor/run_mixed_division
(0.711 one-shot vs 0.386 class-sequential; self-arresting growth).

Mechanics, per class buffer at a sitting:

  1. worst dim — the buffer's least-coherent dimension (min resultant
     length over the buffered phasors).
  2. circular 2-means on that dim's phases proposes a partition.
  3. accept iff BOTH children have ≥ MIN_CHILD members (a mode must
     recur to be born) and BOTH beat the parent's coherence on the
     split dim by GAIN_D — the local split-vs-keep criterion. A tight
     unimodal dim shows no gain → rejected; slicing noise cannot make
     both halves cohere → rejected. Growth self-arrests from this
     criterion alone (probe: 0 divisions p12–16, no cap needed).

v0 falsified (probe doc): a global mean-gain criterion can never fire
on a deep mixture (~gain/F per split); window-only membership starves
sittings once classes multiply — hence per-dim gain + per-class
ROLLING buffers (the caller's job, see mixed.py).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

BUF = 400          # rolling deposit buffer per class
MIN_MEMBERS = 60   # don't judge a class on fewer buffered deposits
MIN_CHILD = 25     # a mode must recur to be born
GAIN_D = 0.08      # split-dim coherence gain BOTH children must show
NULL_SPLIT = 0.72  # noise-slicing null: halving a uniform dim yields 2/π=0.64
                   # coherence; both children must beat this floor or the
                   # organism tiles noise instead of frustrating (design §2)
CLASS_CAP = 128    # hard ceiling on live classes (envelope guard)


def circ_2means(phi: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Boolean partition of angles phi by circular 2-means."""
    z = torch.exp(1j * phi)
    mu = torch.angle(z.mean())
    sd = math.sqrt(max(-2 * math.log(float(z.mean().abs().clamp(min=1e-6))), 1e-4))
    c = torch.tensor([mu - sd, mu + sd])
    for _ in range(iters):
        d = 1 - torch.cos(phi.unsqueeze(1) - c.unsqueeze(0))   # [N, 2]
        side = d.argmin(1)
        for k in (0, 1):
            if (side == k).any():
                c[k] = torch.angle(z[side == k].mean())
    return side == 1


def try_divide(Z: torch.Tensor,
               tries: int = 1) -> Optional[Tuple[torch.Tensor, int]]:
    """The split-vs-keep judgment on one class's buffered phasors
    [n, F]. Returns (side mask, split dim) when the division is
    accepted, None when the buffer should stay whole.

    `tries` (composer arm, design §4b item 4): how many dims to judge,
    ascending coherence. The default 1 is the M2 worst-dim semantics
    (byte-identical); the composer-enabled controller passes
    DIV_TRIES=4 (the probe's loop) so spawned relational dims — rarely
    the argmin — can be divided on. The BEST accepted split among the
    tried dims wins (max child coherence)."""
    if len(Z) < MIN_MEMBERS:
        return None
    R = Z.mean(0).abs()
    best, best_r = None, 0.0
    for d in torch.argsort(R)[:tries].tolist():
        side = circ_2means(torch.angle(Z[:, d]))
        n_b = int(side.sum())
        if min(len(Z) - n_b, n_b) < MIN_CHILD:
            continue
        r_a = float(Z[~side, d].mean().abs())
        r_b = float(Z[side, d].mean().abs())
        if min(r_a, r_b) > max(float(R[d]) + GAIN_D, NULL_SPLIT) \
                and max(r_a, r_b) > best_r:
            best, best_r = (side, d), max(r_a, r_b)
    return best


MERGE_SIM = 0.80   # template cosine pre-filter: duplicates sit high; a
                   # freshly-divided pair coheres at DIFFERENT phases on
                   # its split dim, so it cannot immediately re-merge
                   # (the built-in hysteresis against merge/split thrash)
K_MERGE = 0.5      # per-dim standardized circular mean gap two fragments
                   # must stay UNDER on EVERY dim to be duplicates. Same
                   # mode split by membership accident: gap ≈ sampling
                   # noise √(1/n_a+1/n_b) ≈ 0.07-0.15. Distinct modes
                   # that division separated: gap ≳ 1-2σ on the split
                   # dim. 0.5 sits between with margin both ways.


def template_sim(Za: torch.Tensor, Zb: torch.Tensor) -> float:
    """Cosine between two buffers' phasor templates (complex space)."""
    Ta, Tb = Za.mean(0), Zb.mean(0)
    denom = float(Ta.abs().norm() * Tb.abs().norm())
    if denom < 1e-9:
        return 0.0
    return float((Ta * Tb.conj()).real.sum()) / denom


def _circ_gap(Za: torch.Tensor, Zb: torch.Tensor) -> float:
    """Max over dims of the standardized circular mean gap |Δμ|/σ_pool
    (σ from the resultant length, σ = √(−2 ln R), floored at one
    pocket's phase). The two-sample duplicate statistic."""
    Ta, Tb = Za.mean(0), Zb.mean(0)
    dphi = torch.angle(Ta * Tb.conj()).abs()
    floor = 2 * math.pi / 1000
    sig_a = (-2 * Ta.abs().clamp(1e-6, 1 - 1e-6).log()).sqrt()
    sig_b = (-2 * Tb.abs().clamp(1e-6, 1 - 1e-6).log()).sqrt()
    pooled = ((sig_a ** 2 + sig_b ** 2) / 2).sqrt().clamp_min(floor)
    return float((dphi / pooled).max())


def try_merge(Za: torch.Tensor, Zb: torch.Tensor,
              tries: int = 1) -> bool:
    """The merge consumer's judgment [D18] — a direct TWO-SAMPLE
    duplicate test, with division's law as a veto:

      1. template cosine ≥ MERGE_SIM (cheap pre-filter + hysteresis:
         fresh split children cohere at different phases);
      2. per-dim standardized circular mean gap < K_MERGE on EVERY dim
         — the distributions coincide up to sampling noise (duplicates
         the matched-filter membership split across two templates);
      3. the BALANCED union (up to BUF/2 newest from EACH side — a
         naive cat()[-BUF:] keeps only one side and the judgment
         degenerates) must fail try_divide: if a split clears the
         floors, they are two modes regardless of the gap test.

    Inverting try_divide ALONE is too permissive (first-gate
    falsification, s033: "not divisible by the strict acceptance
    floors" ≠ "same distribution" — two adjacent WIDE modes cannot
    seat coherent children yet are not duplicates; −0.147 accuracy)."""
    if min(len(Za), len(Zb)) < MIN_CHILD:
        return False
    if template_sim(Za, Zb) < MERGE_SIM:
        return False
    if _circ_gap(Za, Zb) >= K_MERGE:
        return False
    union = torch.cat([Za[-(BUF // 2):], Zb[-(BUF // 2):]])
    if len(union) < MIN_MEMBERS:
        return False
    return try_divide(union, tries=tries) is None
