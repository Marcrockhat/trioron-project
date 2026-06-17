"""Run the phasor receptor on the 32-class hard taxonomy (s042 real-ish test).

Moves the phasor front-end off the toy 5x7 font onto data_hard's capacity-hard
taxonomy: K=32 classes, D=12 features, M=3 multimodal Gaussian modes/class, 15%
wide disruptors. Bayes ceiling 0.937; standing council ~0.775.

Caveats this test makes explicit (not an image):
  - TABULAR: the 12 features have no spatial locality, so the scattering lens's
    PATCH grouping is meaningless. What is tested is the phase-code RECEPTOR
    (theta=2*pi*q/1000) + complex descriptor, not the lens.
  - MULTIMODAL: each class is 3 modes. A single mean phasor per class averages
    phases ACROSS modes -> partial cancellation (low coherence) -> a single
    centroid should underfit. A K=3 per-class mode model is the honest back-end.

Receptors compared:
  raw            : the 12-d standardized features (no phase code) — reference
  phasor/sample  : quantize() verbatim (per-SAMPLE min/max over the 12 features)
                   -> theta -> [cos,sin] per feature (24-d). Literal "this receptor".
  phasor/feature : per-FEATURE global frame (sensible for heterogeneous columns)
                   -> theta -> [cos,sin] per feature (24-d).

Back-ends: nearest single centroid, vs nearest of K=3 per-class modes (k-means).

Run:  python3 experiments/progenitor/taxonomy_receptor.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from trioron.core.receptor import quantize
from experiments.progenitor.data_hard import make_split, bayes_accuracy

N_QUANTA = 1000


def phasor_persample(X):
    q = quantize(X)                                   # per-sample frame
    th = 2 * math.pi * q / N_QUANTA
    return torch.cat([th.cos(), th.sin()], dim=1).numpy()


def phasor_perfeature(X, lo=None, hi=None):
    if lo is None:
        lo = X.amin(0); hi = X.amax(0)
    span = (hi - lo).clamp_min(1e-9)
    q = torch.round(N_QUANTA * (X - lo) / span)
    th = 2 * math.pi * q / N_QUANTA
    return torch.cat([th.cos(), th.sin()], dim=1).numpy(), lo, hi


def kmeans(Xc, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = Xc[rng.choice(len(Xc), size=min(k, len(Xc)), replace=False)]
    for _ in range(iters):
        d = ((Xc[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        C = np.stack([Xc[a == j].mean(0) if (a == j).any() else C[j]
                      for j in range(len(C))])
    return C


def eval_centroid(Dtr, ytr, Dte, yte, K, modes=1):
    """Nearest-centroid (modes=1) or nearest-of-K-modes per class."""
    cents, owners = [], []
    for c in range(K):
        Xc = Dtr[ytr == c]
        cs = kmeans(Xc, modes) if modes > 1 else Xc.mean(0, keepdims=True)
        cents.append(cs); owners += [c] * len(cs)
    C = np.concatenate(cents); owners = np.array(owners)
    d = ((Dte[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    pred = owners[d.argmin(1)]
    return float((pred == yte).mean())


def main():
    tr, te = make_split()                             # K=32, D=12, M=3
    K = len(te.names)
    ytr, yte = tr.y.numpy(), te.y.numpy()
    bayes = bayes_accuracy(te)
    print(f"hard taxonomy: K={K}, D={tr.x.shape[1]}, M=3 modes, "
          f"{len(tr)} train / {len(te)} test")
    print(f"  Bayes ceiling = {bayes:.3f}   standing council ~0.775   chance = {1/K:.3f}\n")

    # descriptors
    raw_tr, raw_te = tr.x.numpy(), te.x.numpy()
    ps_tr, ps_te = phasor_persample(tr.x), phasor_persample(te.x)
    pf_tr, lo, hi = phasor_perfeature(tr.x)
    pf_te, _, _ = phasor_perfeature(te.x, lo, hi)

    sets = [("raw (12-d)", raw_tr, raw_te),
            ("phasor/sample (24-d)", ps_tr, ps_te),
            ("phasor/feature (24-d)", pf_tr, pf_te)]

    print(f"  {'receptor':<24} {'centroid':>10} {'K=3 modes':>11}")
    for name, Dtr, Dte in sets:
        a1 = eval_centroid(Dtr, ytr, Dte, yte, K, modes=1)
        a3 = eval_centroid(Dtr, ytr, Dte, yte, K, modes=3)
        print(f"  {name:<24} {a1:>10.3f} {a3:>11.3f}")

    print(f"\n  (single centroid underfits the 3 modes; K=3 modes is the honest "
          f"multimodal back-end)")


if __name__ == "__main__":
    main()
