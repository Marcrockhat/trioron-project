"""Per-species one-shot accuracy on the 10-animal taxonomy + the graded semantics
output Rocky asked about ("goat-ish chicken, not pig").

The one-shot number elsewhere is on the 32c x 3-mode HARD set (~12x chance). This
reads the interpretable easy set, with PROPER feature typing (the heterogeneous
tabular case): each feature is quantized against its OWN population range
(per-FEATURE, not the per-sample receptor in receptor.py — that one is for
commensurate signals like pixels; kg vs cm vs category share no scale), then:

  weight, height  -> continuous receptor phase (population min..max -> 0..1000)
  cover (k=4)     -> labeled lines theta_discrete(level, 4)
  eggs  (k=2)     -> labeled lines theta_discrete(level, 2)

A species template = mean phasor per feature (arg=typical value, |T|=tightness).
One-shot: single animal -> per-feature unit phasor -> matched filter
E_c = sum_f Re(P_f conj(T_cf)) -> argmax. The GRADED output is the full ranking
of E_c (relative membership), which is the semantics layer we have in resolve.py
(.ranking) but never surfaced readably. Run: python3 run_taxonomy10_oneshot.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

# data_taxonomy uses package-relative imports (`.data`); lockin is flat. Put both
# the repo root and this dir on the path so either style resolves. Run from here:
#   python3 run_taxonomy10_oneshot.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.progenitor.data_taxonomy import SPECIES, make_taxonomy
from lockin import N_QUANTA, theta_discrete

COVER_K, EGGS_K = 4, 2


def phases(d) -> torch.Tensor:
    """(N, 4) phase per feature, population-normalized per feature."""
    logw, logh = torch.log(d.kg), torch.log(d.cm)
    out = torch.zeros(len(d), 4)
    for i, v in enumerate((logw, logh)):
        q = 1000 * (v - v.min()) / (v.max() - v.min()).clamp_min(1e-9)
        out[:, i] = 2 * math.pi * q / N_QUANTA
    out[:, 2] = theta_discrete(d.cover.float(), COVER_K)
    out[:, 3] = theta_discrete(d.eggs.float(), EGGS_K)
    return out


def templates(P: torch.Tensor, y: torch.Tensor, n_class: int) -> torch.Tensor:
    """(C, 4) complex mean phasor per class."""
    Z = torch.exp(1j * P)
    return torch.stack([Z[y == c].mean(0) for c in range(n_class)])


def main() -> None:
    tr = make_taxonomy(n_per_class=256, seed=0)
    te = make_taxonomy(n_per_class=256, seed=1)
    names = tr.names
    C = len(names)

    T = templates(phases(tr), tr.y, C)
    Pte = phases(te)
    Z = torch.exp(1j * Pte)
    E = (Z @ T.conj().t()).real                      # (N, C) graded membership
    pred = E.argmax(1)

    print("10-animal taxonomy — per-species ONE-SHOT (single observation, "
          "circular matched filter)")
    print(f"{'species':10s} {'group':8s} {'acc':>6s}   most-confused-with")
    per = []
    for c in range(C):
        m = te.y == c
        acc = (pred[m] == c).float().mean().item()
        per.append(acc)
        wrong = pred[m][pred[m] != c]
        conf = names[torch.bincount(wrong, minlength=C).argmax()] if len(wrong) else "-"
        grp = SPECIES[names[c]].group
        print(f"{names[c]:10s} {grp:8s} {acc:6.3f}   {conf}")
    overall = (pred == te.y).float().mean().item()
    print(f"\noverall one-shot {overall:.3f}  (chance {1/C:.3f}, "
          f"= {overall*C:.1f}x chance)  worst {min(per):.3f}  best {max(per):.3f}")

    # the graded semantics output, on a few held-out animals
    print("\ngraded membership (the 'goat-ish chicken, not pig' output):")
    for c in (names.index("chicken"), names.index("goat"), names.index("dog")):
        idx = (te.y == c).nonzero()[0].item()
        e = E[idx]
        order = e.argsort(descending=True)
        top = order[:3].tolist()
        m = e - e[order[1]]                          # margin over runner-up, per class
        s = ", ".join(f"{names[j]}({e[j]:.2f})" for j in top)
        bot = names[order[-1].item()]
        print(f"  a {names[c]:8s} reads:  {s}  ...  least: {bot}  "
              f"[top margin {(e[order[0]]-e[order[1]]).item():+.2f}]")


if __name__ == "__main__":
    main()
