"""Does NATIVE trioron retrieval (the pseudo-RAG paradigm) solve the disruptor-dog?

Rocky's question: dendritic cells should act like a pseudo-RAG — store-and-retrieve,
memorize when a problem can't be solved by a learned function. Before designing a
per-cell dendrite-RAG we look at the paradigm already in core: the ManifoldArchive
stores a per-class sketch (μ, Σ) as an astrocyte and scores a query by
log_likelihood_full (Mahalanobis); StreamingMixture stores K prototypes per class for
when one Gaussian underfits. Classification = argmax likelihood over classes. That IS
retrieval (memorize, then match) — no SGD, no growth.

We point it at the 6-class disruptor-dog data and compare to the known reference points:
  Bayes ceiling   0.852   (information limit, weight-only)
  linear readout  0.784   (cap — dog is non-linearly-separable)
  ReLU h=4 fit    0.814   (cheap nonlinear FUNCTION, ~33 params)

Run: python3 -m experiments.growth_exercise.retrieval_probe
"""
from __future__ import annotations

import math

import torch

from trioron.core import Envelope, construct
from trioron.bases import minimal
from trioron.phenotype import default_dispatch_table
from trioron.learning.manifold import ManifoldArchive

from experiments.growth_exercise.chicken_goat import make_animals, bayes_accuracy

SPECIES6 = ["chicken", "cat", "dog", "goat", "cow", "elephant"]
_LOG2PI = math.log(2.0 * math.pi)


def _single_gaussian_acc(archive: ManifoldArchive, x: torch.Tensor, y: torch.Tensor) -> float:
    """Argmax over per-class astrocyte Mahalanobis likelihood (1 prototype/class)."""
    cls = sorted(archive.class_ids)
    scores = torch.stack([archive.get(c).log_likelihood_full(x) for c in cls], dim=1)  # [N, C]
    pred = torch.tensor([cls[i] for i in scores.argmax(dim=1).tolist()])
    return float((pred == y).float().mean())


def _mixture_acc(archive: ManifoldArchive, x: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Argmax over per-class StreamingMixture: K diagonal-Gaussian prototypes/class.

    Per class, log p(x) = logsumexp_j [ log w_j + Σ_d log N(x_d; μ_jd, σ_jd) ]. This is
    the K-prototype memory retrieving the best-matching stored mode — the closest native
    analogue of a per-cell dendrite-RAG.
    """
    cls = sorted(archive._mixtures.keys())
    per_class = []
    for c in cls:
        mx = archive._mixtures[c]
        mu, var = mx.mu, mx.var                      # [k, 1]
        logw = (mx.n.clamp(min=1e-6) / mx.n.sum()).log()   # [k]
        diff = x.unsqueeze(1) - mu.unsqueeze(0)      # [N, k, 1]
        comp_ll = (-0.5 * ((diff ** 2) / var.unsqueeze(0) + var.unsqueeze(0).log() + _LOG2PI)).sum(-1)  # [N, k]
        per_class.append(torch.logsumexp(comp_ll + logw.unsqueeze(0), dim=1))  # [N]
    scores = torch.stack(per_class, dim=1)
    pred = torch.tensor([cls[i] for i in scores.argmax(dim=1).tolist()])
    return float((pred == y).float().mean())


def _dump_structure(archive: ManifoldArchive, k: int) -> None:
    """Report the composition of the K-prototype retrieval bank, per class.

    The 'structure' is NOT a feed-forward graph — it is a bank of per-class prototype
    astrocytes (forward_inclusion=False, never in any forward pass). So there are no
    inter-node forward edges to trace. The asymmetry lives in how many of the K
    prototypes each class actually recruits, and how widely their means spread.
    """
    cls = sorted(archive._mixtures.keys())
    total_modes = 0
    print(f"\n  ── structure of the K={k} retrieval bank ──")
    print(f"  {'class':9} {'modes':>5} {'occupancy (n/proto)':>22} {'μ span':>8}")
    for c in cls:
        mx = archive._mixtures[c]
        n = mx.n                                  # [k] samples assigned per prototype
        frac = (n / n.sum())
        eff = int((frac > 0.05).sum().item())     # prototypes carrying >5% of the class
        total_modes += eff
        mu = mx.mu.squeeze(-1)                     # [k]
        occ_mu = mu[frac > 0.05]
        span = float(occ_mu.max() - occ_mu.min()) if occ_mu.numel() else 0.0
        occ = " ".join(f"{int(v):3d}" for v in n.tolist())
        print(f"  {SPECIES6[c]:9} {eff:>5} {occ:>22} {span:>8.2f}")
    print(f"  total prototype-nodes recruited = {total_modes}   forward edges = 0 (retrieval bank)")


def main() -> None:
    torch.manual_seed(0)
    train = make_animals(SPECIES6, n_per_class=512, seed=0, log=True)
    test = make_animals(SPECIES6, n_per_class=512, seed=1, log=True)

    print(f"6-class disruptor-dog  |  Bayes ceiling = {bayes_accuracy(test):.3f}")
    print(f"  reference: linear 0.784   ReLU-h4 0.814\n")

    # Minimal arena to host the astrocytes (forward_inclusion=False; not in any forward pass).
    sub = construct(base=minimal(1, len(SPECIES6)), envelope=Envelope(),
                    dispatch_table=default_dispatch_table(), capacity=128)

    for k in (0, 4, 8):
        archive = ManifoldArchive(sub.arena, full_cov=True, mixture_k=k)
        for c in range(len(SPECIES6)):
            xc = train.x[train.y == c]
            archive.update_class(c, xc)
        single = _single_gaussian_acc(archive, test.x, test.y)
        if k == 0:
            print(f"  retrieval: 1 Gaussian/class (single prototype) = {single:.3f}")
        else:
            mix = _mixture_acc(archive, test.x, test.y, k)
            print(f"  retrieval: {k} prototypes/class (StreamingMixture) = {mix:.3f}")
            if k == 4:
                _dump_structure(archive, k)


if __name__ == "__main__":
    main()
