"""CONV proposer on the REAL chained-15 sensor field (s040) — pre-promotion
validation requested by Rocky.

Runs the conjoined-twin CONV cohort + trial-vote on actual MNIST/Fashion/
EMNIST images (30 global classes, 28x28), real vs column-SHUFFLED, to check
the proposer behaves correctly on the real sensor field before any package
promotion.

REGIME NOTE: this is the GRADIENT-BASED regime (kernels learned by Adam).
The chained-15 headline 0.736/0.446 is the GRADIENT-FREE PCLL organism — a
different pipeline. This validates "a learned conjoined-conv front-end on
the real sensor field + correct vote", NOT a drop-in to the matched filter.

Three arms (multi-channel cohort = C shared kernels, each tiled):
  raw-logreg     logistic regression on raw 784 pixels (tabular baseline)
  conv-maxpool   C kernels -> global-max pool -> linear (TRANSLATION-INV;
                 what the vote uses — this is what shuffling must break)
  conv-flatten   C kernels -> all positions kept -> linear (position-
                 preserving conv features; the centered-data fair test)

Run:  python3 experiments/progenitor/conv_proposer_chained15.py
"""
from __future__ import annotations

import os
import time
import torch

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.phenotype import conv
from trioron.legacy.donorkit.datasets import (DatasetBundle, chained_15_specs,
                                              build_task_views)
from experiments.progenitor.conv_proposer import tile_patches, spawn_conv_cohort

DATA_ROOT = os.environ.get("AXIS6_DATA_ROOT", "outputs/data")
H = W = 28
K = int(os.environ.get("CP_K", "5"))
STRIDE = int(os.environ.get("CP_STRIDE", "2"))
C = int(os.environ.get("CP_C", "12"))        # conv channels (shared kernels)
N_TRAIN, N_TEST = 6000, 2000
EPOCHS = int(os.environ.get("CP_EPOCHS", "8"))
BATCH, LR = 256, 0.02
MARGIN = 0.05               # vote margins (30-way, chance=0.033)


# ── data ────────────────────────────────────────────────────────────────

def load_3015(seed: int):
    b = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                      root=DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()[:15]
    tr = build_task_views(b, specs, split="train")
    te = build_task_views(b, specs, split="test")
    g = torch.Generator().manual_seed(seed)

    def pool(views, n):
        X = torch.cat([v.images for v in views])
        y = torch.cat([v.labels_global for v in views])
        idx = torch.randperm(len(X), generator=g)[:n]
        return X[idx], y[idx]

    return pool(tr, N_TRAIN), pool(te, N_TEST)


# ── multi-channel cohort ────────────────────────────────────────────────

def build_cohort(seed: int):
    patches, _ = tile_patches(H, W, K, STRIDE)
    n_pos = len(patches)
    a = Arena(Envelope(), capacity=H * W + C * n_pos + 64)
    a.alloc(H * W)
    in_cells = list(range(H * W))
    buckets = [spawn_conv_cohort(a, in_cells, patches, K, tied=True,
                                 seed=seed * 100 + ch)[1] for ch in range(C)]
    return a, in_cells, buckets, n_pos


def features(a, in_cells, buckets, x, maxpool: bool):
    X = torch.zeros(len(x), a.capacity); X[:, in_cells] = x
    chans = [conv.forward_batch(X, bk, a) for bk in buckets]   # each [N, n_pos]
    if maxpool:
        return torch.stack([c.max(1).values for c in chans], 1)  # [N, C]
    return torch.cat(chans, 1)                                   # [N, C*n_pos]


# ── trials ──────────────────────────────────────────────────────────────

def _train_head(feat_fn, in_dim, xtr, ytr, xte, yte, train_edge, a, seed):
    torch.manual_seed(seed)
    head = torch.nn.Linear(in_dim, 30)
    params = list(head.parameters()) + ([a.edge_weight] if train_edge else [])
    if train_edge:
        a.edge_weight.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=LR)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), BATCH):
            bi = perm[i:i + BATCH]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(feat_fn(xtr[bi])), ytr[bi])
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = float((head(feat_fn(xte)).argmax(1) == yte).float().mean())
    if train_edge:
        a.edge_weight.requires_grad_(False)
    return acc


def run(perm, seed):
    (xtr, ytr), (xte, yte) = load_3015(seed)
    if perm is not None:
        xtr, xte = xtr[:, perm], xte[:, perm]

    raw = _train_head(lambda x: x, H * W, xtr, ytr, xte, yte,
                      train_edge=False, a=Arena(Envelope(), capacity=1), seed=seed)

    a, inc, bks, n_pos = build_cohort(seed)
    cmax = _train_head(lambda x: features(a, inc, bks, x, maxpool=True),
                       C, xtr, ytr, xte, yte, train_edge=True, a=a, seed=seed)
    a, inc, bks, n_pos = build_cohort(seed)
    cflat = _train_head(lambda x: features(a, inc, bks, x, maxpool=False),
                        C * n_pos, xtr, ytr, xte, yte, train_edge=True, a=a, seed=seed)
    return raw, cmax, cflat


if __name__ == "__main__":
    seeds = [0, 1]
    print(f"REAL chained-15 sensor field, 30-way, {H}x{W}, K={K} stride={STRIDE} "
          f"C={C} channels, n={len(seeds)} seeds (chance=0.033).\n")

    t0 = time.time()
    real = torch.tensor([run(None, s) for s in seeds]).mean(0)
    g = torch.Generator().manual_seed(999)
    pm = torch.randperm(H * W, generator=g)
    shuf = torch.tensor([run(pm, s) for s in seeds]).mean(0)
    dt = time.time() - t0

    print(f"{'arm':<16}{'REAL':>10}{'SHUFFLED':>12}")
    for i, name in enumerate(["raw-logreg", "conv-maxpool", "conv-flatten"]):
        print(f"{name:<16}{real[i]:>10.3f}{shuf[i]:>12.3f}")

    # vote uses the translation-invariant arm (conv-maxpool) vs raw baseline
    adv_real = float(real[1] - real[0])
    adv_shuf = float(shuf[1] - shuf[0])
    vr = "SPAWN CONV" if (real[1] > 0.033 + MARGIN and adv_real > MARGIN) else "reject"
    vs = "SPAWN CONV" if (shuf[1] > 0.033 + MARGIN and adv_shuf > MARGIN) else "reject"
    print(f"\nVOTE (conv-maxpool vs raw): REAL -> {vr} (adv {adv_real:+.3f}); "
          f"SHUFFLED -> {vs} (adv {adv_shuf:+.3f})")
    print(f"[{dt:.0f}s]")
