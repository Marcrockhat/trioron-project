"""Step 2b — variance apoptosis first, then saliency on the concentrated survivors.

The lesson from the failed hierarchical attempt: you cannot detect a signal
diluted 1/1.5 M against noise — diluted that far it IS noise. So the order is the
fix (Rocky):

  1. APOPTOSIS FIRST — kill every empty (zero-variance) cell IMMEDIATELY, en masse.
     Variance is a statistic, not a trained quantity — no gradient steps needed — so
     this culls the empty bulk of the 1.5 M aperture cheaply in a single pass. This
     is what removes the "solvent": the signal is never diluted because the empties
     die before any saliency is measured.
  2. SALIENCY SECOND — only the small variance-bearing survivor set goes to the
     council, where u=|w·g| separates the real feature from variance-bearing noise.
     Now the signal is concentrated (1-of-few, not 1-of-1.5 M), so saliency works.

Toy: a mostly-empty aperture (the realistic case) — one weight column, a few noise
columns, the rest empty. Apoptosis kills the empty bulk in one pass; saliency
finishes on the handful that remain.

Run: python3 -m experiments.progenitor.step2b_apoptosis
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from trioron.core import Envelope, construct
from trioron.core.epigenome import PERCEPTION, OUTPUT, set_gene
from trioron.phenotype import default_dispatch_table

from .data import make_data, bayes_accuracy
from .positions import sensor_positions

APERTURE_N = 256         # oversized intake (toy stand-in for 1.5 Mi)
WEIGHT_COL = 37          # the one informative column
N_NOISE = 5              # variance-bearing but uninformative columns
VAR_EPS = 1e-4
SAL_FRAC = 0.10
SEED = 0xC0FFEE


def build_aperture(data, N, weight_col, n_noise, seed):
    """Mostly-empty aperture: weight at one col, a few noise cols, the rest zero."""
    g = torch.Generator().manual_seed(seed)
    n = data.x.shape[0]
    X = torch.zeros(n, N)
    X[:, weight_col] = data.x[:, 0]
    noise_cols = torch.randperm(N, generator=g)
    noise_cols = [int(c) for c in noise_cols if int(c) != weight_col][:n_noise]
    for c in noise_cols:
        X[:, c] = torch.randn(n, generator=g)
    return X, sorted([weight_col] + noise_cols)


def saliency_on(cols, X, y, n_out, steps=200, lr=0.05, seed=0):
    """Build perception(cols) → outputs, train, return per-col saliency u=|w·g|."""
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
                    dispatch_table=default_dispatch_table(), capacity=len(cols) + 32)
    torch.manual_seed(seed); sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    Xc = X[:, cols]
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(sub(Xc), y).backward(); opt.step()
    a = sub.arena; a.edge_weight.grad = None
    F.cross_entropy(sub(Xc), y).backward(); g = a.edge_weight.grad
    sal = []
    for s in base.perc:
        m = (a.edge_src[:a.edge_cursor] == s)
        sal.append(float((a.edge_weight[:a.edge_cursor][m] * g[:a.edge_cursor][m]).abs().sum()))
    return sal


def main() -> None:
    data = make_data(seed=0, n_per_class=256)
    X, planted = build_aperture(data, APERTURE_N, WEIGHT_COL, N_NOISE, seed=1)
    n_out = len(data.names)

    print(f"Step 2b — apoptosis-first genesis  (aperture N={APERTURE_N}, "
          f"weight@{WEIGHT_COL}, {N_NOISE} noise, rest empty)\n")

    # ── Phase 1: variance apoptosis (immediate, mass, NO training) ──
    var = X.var(0)
    alive = (var >= VAR_EPS).nonzero(as_tuple=False).squeeze(-1).tolist()
    print(f"  Phase 1 — variance apoptosis (a statistic, no gradient steps):")
    print(f"    proposed {APERTURE_N} sensors → recycled {APERTURE_N - len(alive)} "
          f"empty (zero-variance) cells immediately → {len(alive)} survive")
    print(f"    survivors (col: variance): "
          f"{', '.join(f'{c}:{float(var[c]):.2f}' for c in alive)}")

    # ── Phase 2: saliency on the concentrated survivors ──
    sal = saliency_on(alive, X, data.y, n_out)
    smax = max(sal) or 1e-9
    print(f"\n  Phase 2 — saliency u=|w·g| on the {len(alive)} survivors "
          f"(keep ≥ {SAL_FRAC:.0%} of max):")
    kept = []
    for k, c in enumerate(alive):
        role = "weight" if c == WEIGHT_COL else "noise"
        verdict = "KEEP" if sal[k] >= SAL_FRAC * smax else "recycle (no learning)"
        if verdict == "KEEP":
            kept.append(c)
        print(f"    col {c:>3} ({role:>6})  u={sal[k]:.4f}  {verdict}")

    print(f"\n  → input layer = {kept}  "
          f"(planted feature at {WEIGHT_COL})")
    print(f"  cost: 1 variance pass over {APERTURE_N} cols (no training) + "
          f"training on {len(alive)} survivors only")
    print(f"  converged to the single planted feature: {kept == [WEIGHT_COL]}")


if __name__ == "__main__":
    main()
