"""Genesis — grow the input layer from an oversized aperture (design §3.2, §3.6).

The perception layer is NOT hand-allocated; it is *discovered*. A wide progenitor
proposes far more candidate sensors than the problem needs (the toy stand-in for the
1.5 Mi retinal aperture); two cheap stages cull it down to the real features, which
then become the council's perception cells. Generalizes ``step2b_apoptosis`` from the
1-feature probe to an arbitrary F-feature table.

  Phase 1 — VARIANCE APOPTOSIS (a statistic, NO gradient steps): kill every
            zero-variance (empty) aperture cell in one pass. This removes the bulk
            "solvent" so the signal is never diluted — apoptosis is free.
  Phase 2 — SALIENCY u=|w·g| (a short probe train): of the variance-bearing
            survivors, keep those whose perception→output edges carry real gradient
            signal; recycle the variance-bearing-but-uninformative noise.

The survivors = the input layer. ``run_taxonomy_genesis`` then attaches the council
to exactly these cells, so the council sees a grown perception layer, not a given one.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from trioron.core import Envelope, construct
from trioron.core.epigenome import PERCEPTION, OUTPUT, set_gene
from trioron.phenotype import default_dispatch_table

from .positions import sensor_positions

VAR_EPS = 1e-4       # below this variance a cell is "empty" → apoptose
# Keep survivors with saliency ≥ this fraction of the max. step2b used 0.10 with a
# SINGLE planted feature (max ≈ that feature, noise ≪ 10%). With MULTIPLE real
# features of unequal strength (a strong continuous weight vs weaker categorical
# one-hots) 0.10 wrongly recycles the weak-but-real columns; the uninformative noise
# sits at ~0 saliency, far below any real feature, so a small fraction cleanly
# separates real from noise. (0.10 kept only 3 of 7 taxonomy features; 0.01 recovers all 7.)
SAL_FRAC = 0.01
# Saliency separates real features from noise by ~10 steps (noise weights collapse
# fast, and the real-vs-noise saliency gap is large); 21 clears that floor with
# margin. 200 was ~10× overkill — genesis only needs to DECIDE which sensors are real,
# not converge a classifier (verified: 5 steps keeps noise, ≥10 recovers all 7).
SAL_STEPS = 21       # probe-train length for the saliency stage
SEED = 0xC0FFEE


@dataclass
class Genesis:
    aperture_n: int          # candidate sensors proposed
    survived_variance: list[int]   # cols that passed apoptosis (variance-bearing)
    kept: list[int]          # cols that passed saliency → the input layer
    real_cols: list[int]     # ground-truth planted feature columns (for scoring)
    noise_cols: list[int]
    var: torch.Tensor
    sal: list[float]


def build_aperture(feats: torch.Tensor, aperture_n: int, n_noise: int, seed: int):
    """Embed the F real feature columns + ``n_noise`` random columns among
    ``aperture_n`` candidates; the rest are empty (zero). Returns (X_aperture,
    real_cols, noise_cols) so the SAME placement can be applied to train and test."""
    n, fdim = feats.shape
    if aperture_n < fdim + n_noise:
        raise ValueError("aperture too small for the planted features + noise")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(aperture_n, generator=g).tolist()
    real_cols = sorted(perm[:fdim])
    noise_cols = sorted(perm[fdim:fdim + n_noise])
    X = torch.zeros(n, aperture_n)
    X[:, real_cols] = feats                       # plant the real features
    for c in noise_cols:
        X[:, c] = torch.randn(n, generator=g)     # variance-bearing but meaningless
    return X, real_cols, noise_cols


def _saliency(cols, X, y, n_out, steps=SAL_STEPS, lr=0.05, seed=0):
    """perception(cols) → outputs probe; return per-col u=|w·g| (summed over edges)."""
    pos = sensor_positions(len(cols), seed=SEED, dim=1)

    def base(sub):
        a = sub.arena
        perc = a.alloc(len(cols)); outs = a.alloc(n_out)
        for k, p in enumerate(perc.tolist()):
            a.epigenome[p] = set_gene(int(a.epigenome[p].item()), PERCEPTION)
            a.rank[p] = 0; a.position[p] = pos[k]
        for o in outs.tolist():
            a.epigenome[o] = set_gene(int(a.epigenome[o].item()), OUTPUT); a.rank[o] = 1
        a.refresh_all_phenotypes()
        a.add_edges(perc.repeat(n_out), outs.repeat_interleave(len(cols)))
        base.perc = perc.tolist()

    sub = construct(base=base, envelope=Envelope(),
                    dispatch_table=default_dispatch_table(), capacity=len(cols) + n_out + 4)
    torch.manual_seed(seed); sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    Xc = X[:, cols]
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(sub(Xc), y).backward(); opt.step()
    a = sub.arena
    a.edge_weight.grad = None
    F.cross_entropy(sub(Xc), y).backward()
    g = a.edge_weight.grad
    sal = []
    w = a.edge_weight.detach()
    for s in base.perc:
        m = (a.edge_src[:a.edge_cursor] == s)
        sal.append(float((w[:a.edge_cursor][m] * g[:a.edge_cursor][m]).abs().sum()))
    return sal


def discover_input_layer(X_ap, y, n_out, real_cols, noise_cols, *,
                         var_eps=VAR_EPS, sal_frac=SAL_FRAC, steps=SAL_STEPS,
                         seed=0) -> Genesis:
    """Apoptosis-first genesis: variance cull → saliency keep. Returns the kept
    columns (the grown input layer) plus diagnostics."""
    var = X_ap.var(0)
    survived = (var >= var_eps).nonzero(as_tuple=False).squeeze(-1).tolist()
    sal = _saliency(survived, X_ap, y, n_out, steps=steps, seed=seed)
    smax = max(sal) or 1e-9
    kept = [survived[k] for k in range(len(survived)) if sal[k] >= sal_frac * smax]
    return Genesis(aperture_n=X_ap.shape[1], survived_variance=survived, kept=kept,
                   real_cols=real_cols, noise_cols=noise_cols, var=var, sal=sal)
