"""Real ManifoldArchive back-end on the 32-class hard taxonomy (s042).

Replaces taxonomy_receptor.py's numpy centroid / k-means with the actual
substrate manifold (trioron/learning/manifold.py). The substrate's classifier
response to the receptor codes is the per-class Gaussian log-likelihood:

  diagonal   : ManifoldAstrocyte.log_likelihood        (diag-Gaussian sketch)
  Mahalanobis: ManifoldAstrocyte.log_likelihood_full   (full-cov QDA)
  mixture k=3: StreamingMixture (diag modes) — the multimodal manifold

argmax of the per-class log-likelihood = predicted class. Inputs: raw 12-d vs
the per-feature phasor receptor (the lift found in taxonomy_receptor). Reference:
Bayes 0.937, standing council ~0.775, numpy k-means K=3 phasor 0.848.

Run:  python3 experiments/progenitor/taxonomy_manifold.py
"""
from __future__ import annotations

import math
import torch

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.learning import ManifoldArchive
from trioron.learning.manifold import StreamingMixture

from experiments.progenitor.data_hard import make_split, bayes_accuracy
from experiments.progenitor.taxonomy_receptor import phasor_perfeature


def _archive(full_cov):
    arena = Arena(Envelope(), capacity=64)
    return ManifoldArchive(arena, full_cov=full_cov)


def classify_archive(Dtr, ytr, Dte, K, full_cov):
    arch = _archive(full_cov)
    for c in range(K):
        arch.update_class(c, Dtr[ytr == c])
    ll = torch.empty(len(Dte), K)
    for c in range(K):
        a = arch.get(c)
        ll[:, c] = a.log_likelihood_full(Dte) if full_cov else a.log_likelihood(Dte)
    return ll.argmax(1)


def classify_mixture(Dtr, ytr, Dte, K, k=3):
    """StreamingMixture per class; class score = logsumexp over diag modes."""
    dev = Dte.device
    mixes = []
    for c in range(K):
        mx = StreamingMixture(Dtr.shape[1], k, dev)
        mx.update(Dtr[ytr == c])
        mixes.append(mx)
    ll = torch.empty(len(Dte), K)
    log2pi = math.log(2 * math.pi)
    for c, mx in enumerate(mixes):
        mu, var = mx.mu, mx.var                       # (k, D)
        active = mx.n > 0
        z = (Dte[:, None, :] - mu[None, :, :]) / var.sqrt()[None, :, :]
        logN = (-0.5 * (z * z + var.log()[None] + log2pi)).sum(-1)   # (N, k)
        logN = logN.masked_fill(~active[None, :], float("-inf"))
        ll[:, c] = torch.logsumexp(logN, dim=1)
    return ll.argmax(1)


def main():
    tr, te = make_split()                             # K=32, D=12, M=3
    K = len(te.names)
    ytr, yte = tr.y, te.y
    print(f"hard taxonomy: K={K}, D={tr.x.shape[1]}, M=3 modes, "
          f"{len(tr)} train / {len(te)} test")
    print(f"  Bayes {bayes_accuracy(te):.3f} | council ~0.775 | "
          f"numpy kmeans-3 phasor 0.848 | chance {1/K:.3f}\n")

    raw_tr, raw_te = tr.x, te.x
    pf_tr, lo, hi = phasor_perfeature(tr.x)
    pf_te, _, _ = phasor_perfeature(te.x, lo, hi)
    pf_tr, pf_te = torch.from_numpy(pf_tr).float(), torch.from_numpy(pf_te).float()

    sets = [("raw (12-d)", raw_tr, raw_te), ("phasor/feature (24-d)", pf_tr, pf_te)]

    print(f"  {'input':<24} {'diag':>8} {'Mahalanobis':>12} {'mixture k=3':>12}")
    for name, Dtr, Dte in sets:
        ad = float((classify_archive(Dtr, ytr, Dte, K, full_cov=False) == yte).float().mean())
        am = float((classify_archive(Dtr, ytr, Dte, K, full_cov=True) == yte).float().mean())
        ax = float((classify_mixture(Dtr, ytr, Dte, K) == yte).float().mean())
        print(f"  {name:<24} {ad:>8.3f} {am:>12.3f} {ax:>12.3f}")


if __name__ == "__main__":
    main()
