"""s039 BAND-OFFSET encoding — Rocky's anti-superposition design.

The s037/s038 keystone: on raw pixels every feature's phasor lives on the
SAME unit circle centered at the origin, so distinct quanta COLLIDE
(q=0 background and q=1000 stroke land on the same point; templates 89%
identical) and all features superimpose in the matched-filter sum.

Rocky's fix (this diagnostic): stop flattening the image into one 784-bag
on one circle. Treat the data as a matrix [n_feat x n_dim] and give EACH
DIMENSION its own offset circle — band j is centered at d_j on the real
axis, so the bands sit PARALLEL ("stacked, not on top of each other"):

    Z[f, j] = W * exp(i * theta(q[f,j])) + d_j        d_j = D_SCALE * (j+1)

For a 28x28 image: 28 features x 28 dimensions -> 28 parallel band-circles.
A pixel in dimension 3 and a pixel in dimension 20 with the SAME intensity
(same quantum, same theta) no longer collide: their phasors orbit different
centers (3 vs 20). This re-injects the spatial (column) structure that the
flat bag destroys (substrate_no_spatial_memory) WITHOUT a conv.

This isolates the ENCODING from the council/division loop, exactly like
diag_s037: per-true-class mean-phasor ORACLE templates, nearest-template on
held-out test, 30-way, chance 0.033. Baselines: raw-pixel phasor (D=0)
~0.249 masked, template-cosine ~0.94 (the keystone collision).

Sweep S039_D (offset scale, 0 = reproduce raw) and S039_W (phasor gain).
The d^2 / d-linear terms grow with D_SCALE and can SWAMP the O(1) phase
term W^2*cos(dtheta) -- the sweep characterizes where band-separation helps
vs where the offset drowns the phase. n_feat x n_dim read from S039_GRID
(default 28x28; columns are the dimension/bands).

Run:  python3 -m experiments.progenitor.diag_s039_bands
Env:  S039_PER_CLASS (default 500), S039_D (csv sweep, default
      "0,0.1,0.5,1.0,2.0"), S039_W (default 1.0), S039_GRID (default 28x28).
"""
from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F

from trioron.core.receptor import quantize, N_QUANTA
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)

DATA_ROOT = os.environ.get(
    "AXIS6_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "outputs", "data"))
PER_CLASS = int(os.environ.get("S039_PER_CLASS", "500"))
D_SWEEP = [float(s) for s in os.environ.get("S039_D", "0,0.1,0.5,1.0,2.0").split(",")]
W_GAIN = float(os.environ.get("S039_W", "1.0"))
N_FEAT, N_DIM = (int(s) for s in os.environ.get("S039_GRID", "28x28").split("x"))
N_CLASSES = 30


def phasor_bands(x: torch.Tensor, d_scale: float, mask: bool = True
                 ) -> torch.Tensor:
    """[N, n_feat*n_dim] pixels -> [N, n_feat*n_dim] complex band phasors.
    Per-sample quantize (one 1000-quanta frame for the whole image, as the
    shipped receptor), then add a per-DIMENSION real offset d_j so each
    column/band orbits its own center. mask: q in {0, N_QUANTA} are
    reference (spec §10.3) -> zeroed, no phasor AND no offset."""
    q = quantize(x)                                   # [N, F]  per-sample
    theta = 2 * math.pi * q / N_QUANTA
    Z = (W_GAIN * torch.exp(1j * theta)).view(-1, N_FEAT, N_DIM)
    d = d_scale * (torch.arange(N_DIM, dtype=torch.float32) + 1.0)
    Z = Z + d.view(1, 1, N_DIM)                       # offset on the real axis
    if mask:
        qb = q.view(-1, N_FEAT, N_DIM)
        Z = Z.masked_fill((qb == 0) | (qb == N_QUANTA), 0)
    return Z.reshape(x.shape[0], -1)


def oracle_templates(Z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.stack([Z[y == k].mean(0) for k in range(N_CLASSES)])


def evidence(Z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    rows = max(1, int(25_000_000 / max(1, T.shape[0] * T.shape[1])))
    return torch.cat([
        (Z[i:i + rows].unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1)
        for i in range(0, len(Z), rows)])


def template_cosine(T: torch.Tensor) -> float:
    n = T.shape[0]
    norm = T.abs().pow(2).sum(-1).sqrt().clamp_min(1e-9)
    G = (T @ T.conj().t()).abs() / norm.unsqueeze(1) / norm.unsqueeze(0)
    off = G[~torch.eye(n, dtype=torch.bool)]
    return float(off.mean())


def balanced_slice(views, per_class, seed):
    X = torch.cat([v.images for v in views])
    y = torch.cat([v.labels_global for v in views])
    g = torch.Generator().manual_seed(seed)
    idx = []
    for k in range(N_CLASSES):
        ck = (y == k).nonzero().squeeze(1)
        idx.append(ck[torch.randperm(len(ck), generator=g)][:per_class])
    idx = torch.cat(idx)
    return X[idx], y[idx]


def main() -> None:
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()
    Xtr, ytr = balanced_slice(build_task_views(bundle, specs, split="train"),
                              PER_CLASS, seed=0)
    Xte, yte = balanced_slice(build_task_views(bundle, specs, split="test"),
                              PER_CLASS, seed=1)

    print(f"s039 BAND-OFFSET oracle — {N_CLASSES}-way, grid "
          f"{N_FEAT}x{N_DIM} (dims=bands), W={W_GAIN}")
    print(f"train {tuple(Xtr.shape)}  test {tuple(Xte.shape)}  "
          f"chance={1.0 / N_CLASSES:.3f}\n")
    print(f"{'D_SCALE':>8}{'masked_acc':>12}{'tmpl_cos':>10}{'t(s)':>7}")
    print("-" * 37)
    for d in D_SWEEP:
        t0 = time.time()
        T = oracle_templates(phasor_bands(Xtr, d), ytr)
        acc = float((evidence(phasor_bands(Xte, d), T).argmax(1) == yte)
                    .float().mean())
        cos = template_cosine(T)
        tag = "  (= raw baseline)" if d == 0 else ""
        print(f"{d:>8.2f}{acc:>12.3f}{cos:>10.3f}{time.time() - t0:>7.1f}{tag}")


if __name__ == "__main__":
    main()
