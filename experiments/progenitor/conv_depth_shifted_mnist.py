"""Conv DEPTH (conv->pool->conv) on shifted-MNIST (s041) — the §8 "later stage".

Every conv result through s040 is SINGLE-LAYER. On centered chained-15 a lone
conv layer loses to raw logreg, and the honest gap was DEPTH: §8 deferred the
conv-on-conv stack ("needs the layer-1 feature map to carry positions too").
This builds it and demonstrates the POSITIVE the handoff NEXT#1 asks for, on a
TRANSLATION-STRUCTURED task where geometry actually matters.

TASK — shifted-MNIST. A 28x28 MNIST digit is dropped at a RANDOM (dr,dc) offset
onto a CANVAS x CANVAS field. The object MOVES, so a tabular learner (logreg)
that sums fixed columns cannot follow it; a translation-equivariant conv +
global-pool readout can. (Centered MNIST would let logreg win — the point is the
object is off-center and the position is random.)

THREE ARMS:
  raw-logreg   logistic regression on the raw CANVAS^2 pixels (tabular baseline)
  conv1        ONE conjoined-twin conv cohort -> global-max -> linear (s040 form)
  conv2        conv -> 2x2 maxpool -> conv -> global-max -> linear (NEW: DEPTH).
               L2 is a tied cohort tiling over the POOLED L1 feature map; each
               L2 channel reads ACROSS all L1 channels in its K2xK2 patch, tied
               across positions (real 2nd-layer convolution, not a wider L1).

Both conv arms use the REAL trioron conv.forward_batch (weight-tying by
lineage_root + tap). Pooling/chaining is the consumer's job (§8: "the consumer
must pool too"). GRADIENT-BASED regime (Adam on kernels), as in s040 — not the
gradient-free PCLL pipeline.

Run:  python3 experiments/progenitor/conv_depth_shifted_mnist.py
Env:  CANVAS, OFFSET, C1, C2, K1, S1, K2, S2, EPOCHS, N_TRAIN, N_TEST, SEEDS
"""
from __future__ import annotations

import os
import time
import torch
import torch.nn.functional as F

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.core.epigenome import CONV
from trioron.phenotype import conv
from trioron.legacy.donorkit.datasets import DatasetBundle
from experiments.progenitor.conv_proposer import tile_patches, _bucket_for

DATA_ROOT = os.environ.get("AXIS6_DATA_ROOT", "outputs/data")
CANVAS = int(os.environ.get("CANVAS", "36"))     # digit is 28x28, so offset 0..8
OFFSET = int(os.environ.get("OFFSET", str(CANVAS - 28)))
C1 = int(os.environ.get("C1", "8"))
C2 = int(os.environ.get("C2", "12"))
K1 = int(os.environ.get("K1", "5"))
S1 = int(os.environ.get("S1", "2"))
K2 = int(os.environ.get("K2", "3"))
S2 = int(os.environ.get("S2", "1"))
EPOCHS = int(os.environ.get("EPOCHS", "8"))
N_TRAIN = int(os.environ.get("N_TRAIN", "4000"))
N_TEST = int(os.environ.get("N_TEST", "1500"))
BATCH, LR = 256, 0.02


# ── data: shifted-MNIST ─────────────────────────────────────────────────

def _shift_onto_canvas(imgs28, g):
    """imgs28: [N,784] in [0,1]. Drop each onto a CANVAS^2 field of zeros at a
    random integer offset in [0,OFFSET]^2. Returns [N, CANVAS*CANVAS]."""
    n = imgs28.shape[0]
    out = torch.zeros(n, CANVAS, CANVAS)
    src = imgs28.view(n, 28, 28)
    dr = torch.randint(0, OFFSET + 1, (n,), generator=g)
    dc = torch.randint(0, OFFSET + 1, (n,), generator=g)
    for i in range(n):
        out[i, dr[i]:dr[i] + 28, dc[i]:dc[i] + 28] = src[i]
    return out.view(n, CANVAS * CANVAS)


def load_shifted(seed: int):
    b = DatasetBundle(["mnist"], root=DATA_ROOT)
    tr = b.task_view("mnist", list(range(10)), list(range(10)), split="train")
    te = b.task_view("mnist", list(range(10)), list(range(10)), split="test")
    g = torch.Generator().manual_seed(seed)

    def take(view, n):
        X, y = view.images, view.labels_global
        idx = torch.randperm(len(X), generator=g)[:n]
        return _shift_onto_canvas(X[idx], g), y[idx]

    return take(tr, N_TRAIN), take(te, N_TEST)


# ── multichannel patch tiler (for L2 over the pooled L1 map) ─────────────

def tile_patches_mc(C, H, W, k, stride):
    """k×k patches over a C-channel H×W map, in the channel-major flat layout
    idx = ch*(H*W) + r*W + c. Each patch lists ALL C channels at its K×K
    spatial footprint, in a fixed (ch, i, j) tap order so weight-tying lines up
    tap-t at one position with tap-t at the next (real cross-channel conv)."""
    plane = H * W
    patches, centers = [], []
    for r in range(0, H - k + 1, stride):
        for c in range(0, W - k + 1, stride):
            patch = [ch * plane + (r + i) * W + (c + j)
                     for ch in range(C) for i in range(k) for j in range(k)]
            patches.append(patch)
            centers.append((r, c))
    return patches, centers


def _grid(side, k, stride):
    return (side - k) // stride + 1


def spawn_cohort(arena, in_cells, patches, seed):
    """Conjoined-twin CONV cohort: one CONV cell per patch, all sharing
    cohort[0]'s lineage_root (ONE kernel, tiled). Kernel size = len(patch[0])
    so it handles BOTH a single-channel K×K fan-in (L1) and a cross-channel
    C·K2² fan-in (L2). Matched tap order across patches makes the weight-tie a
    real translation-shared filter. Returns the forward Bucket."""
    n = len(patches)
    ids = arena.alloc(n).tolist()
    for c in ids:
        arena.epigenome[c] = 1 << CONV
        arena.phenotype_cache[c] = CONV
        arena.lineage_root[c] = ids[0]
    g = torch.Generator().manual_seed(seed)
    fan = len(patches[0])
    for i, (cell, patch) in enumerate(zip(ids, patches)):
        src = torch.tensor([in_cells[p] for p in patch], dtype=torch.int32)
        dst = torch.full((len(patch),), cell, dtype=torch.int32)
        w = torch.randn(fan, generator=g) * 0.1 if i == 0 else None
        arena.add_edges(src, dst, w)
    return _bucket_for(arena, ids)


# ── L1 / L2 builders ────────────────────────────────────────────────────

def build_l1(seed):
    """Arena holding CANVAS^2 input cells + C1 tied conv cohorts over the
    input. Returns (arena, in_cells, buckets, (H1, W1))."""
    patches, _ = tile_patches(CANVAS, CANVAS, K1, S1)
    H1 = W1 = _grid(CANVAS, K1, S1)
    a = Arena(Envelope(), capacity=CANVAS * CANVAS + C1 * len(patches) + 64)
    a.alloc(CANVAS * CANVAS)
    in_cells = list(range(CANVAS * CANVAS))
    bks = [spawn_cohort(a, in_cells, patches, seed=seed * 100 + ch)
           for ch in range(C1)]
    a.edge_weight.requires_grad_(True)
    return a, in_cells, bks, (H1, W1)


def build_l2(seed, C, H, W):
    """Arena holding C*H*W pooled-input cells + C2 tied conv cohorts tiling
    K2×K2 over the C-channel pooled map. Returns (arena, in_cells, buckets,
    n_pos2)."""
    patches, _ = tile_patches_mc(C, H, W, K2, S2)
    n_in = C * H * W
    a = Arena(Envelope(), capacity=n_in + C2 * len(patches) + 64)
    a.alloc(n_in)
    in_cells = list(range(n_in))
    bks = [spawn_cohort(a, in_cells, patches, seed=seed * 200 + ch)
           for ch in range(C2)]
    a.edge_weight.requires_grad_(True)
    return a, in_cells, bks, len(patches)


def l1_map(a, in_cells, bks, x, H1, W1):
    """Run L1 cohorts -> spatial feature map [N, C1, H1, W1] (differentiable)."""
    X = torch.zeros(len(x), a.capacity)
    X[:, in_cells] = x
    chans = [conv.forward_batch(X, bk, a) for bk in bks]      # each [N, H1*W1]
    return torch.stack(chans, 1).view(len(x), C1, H1, W1)


def l2_feats(a, in_cells, bks, pooled_flat):
    """Run L2 cohorts over the flattened pooled map -> [N, C2] (global-max)."""
    X = torch.zeros(len(pooled_flat), a.capacity)
    X[:, in_cells] = pooled_flat
    chans = [conv.forward_batch(X, bk, a).max(1).values for bk in bks]
    return torch.stack(chans, 1)                              # [N, C2]


# ── training ────────────────────────────────────────────────────────────

def _fit_eval(feat_fn, in_dim, params, xtr, ytr, xte, yte, seed):
    torch.manual_seed(seed)
    head = torch.nn.Linear(in_dim, 10)
    opt = torch.optim.Adam(list(head.parameters()) + params, lr=LR)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), BATCH):
            bi = perm[i:i + BATCH]
            opt.zero_grad()
            loss = F.cross_entropy(head(feat_fn(xtr[bi])), ytr[bi])
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = float((head(feat_fn(xte)).argmax(1) == yte).float().mean())
    return acc


def run(seed, perm=None):
    (xtr, ytr), (xte, yte) = load_shifted(seed)
    D0 = CANVAS * CANVAS
    if perm is not None:                      # destroy spatial locality
        xtr, xte = xtr[:, perm], xte[:, perm]

    # arm 1: raw logreg
    raw = _fit_eval(lambda x: x, D0, [], xtr, ytr, xte, yte, seed)

    # arm 2: single-layer conv -> global-max
    a1, inc1, bk1, (H1, W1) = build_l1(seed)
    def f_conv1(x):
        return l1_map(a1, inc1, bk1, x, H1, W1).amax(dim=(2, 3))   # [N, C1]
    conv1 = _fit_eval(f_conv1, C1, [a1.edge_weight], xtr, ytr, xte, yte, seed)

    # arm 3: conv -> 2x2 maxpool -> conv -> global-max  (DEPTH)
    a1b, inc1b, bk1b, (H1b, W1b) = build_l1(seed)
    H1p, W1p = H1b // 2, W1b // 2
    a2, inc2, bk2, _ = build_l2(seed, C1, H1p, W1p)
    def f_conv2(x):
        m = l1_map(a1b, inc1b, bk1b, x, H1b, W1b)                  # [N,C1,H1,W1]
        m = F.max_pool2d(F.relu(m), 2)                            # [N,C1,H1p,W1p]
        pooled_flat = m.reshape(len(x), -1)                       # ch-major
        return l2_feats(a2, inc2, bk2, F.relu(pooled_flat))       # [N, C2]
    conv2 = _fit_eval(f_conv2, C2, [a1b.edge_weight, a2.edge_weight],
                      xtr, ytr, xte, yte, seed)
    return raw, conv1, conv2


if __name__ == "__main__":
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
    print(f"shifted-MNIST: 28x28 digit at random offset in [0,{OFFSET}]^2 on a "
          f"{CANVAS}x{CANVAS} canvas, 10-way (chance 0.10).")
    print(f"L1: C1={C1} K1={K1} s{S1}  pool 2x2  L2: C2={C2} K2={K2} s{S2}  "
          f"EPOCHS={EPOCHS}  n={len(seeds)} seeds.\n")

    do_shuf = os.environ.get("SHUFFLE", "1") == "1"

    t0 = time.time()
    m = torch.tensor([run(s) for s in seeds]).mean(0)
    if do_shuf:
        g = torch.Generator().manual_seed(999)
        pm = torch.randperm(CANVAS * CANVAS, generator=g)   # fixed, locality-breaking
        ms = torch.tensor([run(s, pm) for s in seeds]).mean(0)
    dt = time.time() - t0

    hdr = f"{'arm':<16}{'REAL':>10}" + (f"{'SHUFFLED':>12}" if do_shuf else "")
    print(hdr)
    for i, name in enumerate(["raw-logreg", "conv1 (1-layer)", "conv2 (DEPTH)"]):
        row = f"{name:<16}{m[i]:>10.3f}" + (f"{ms[i]:>12.3f}" if do_shuf else "")
        print(row)
    print(f"\nDEPTH lift  conv2 - conv1 = {float(m[2] - m[1]):+.3f}")
    print(f"conv2 - raw-logreg          = {float(m[2] - m[0]):+.3f}")
    if do_shuf:
        print(f"LOCALITY    conv2 REAL - SHUF = {float(m[2] - ms[2]):+.3f}  "
              f"(positive => conv2 uses geometry, not just capacity)")
    print(f"[{dt:.0f}s]")
