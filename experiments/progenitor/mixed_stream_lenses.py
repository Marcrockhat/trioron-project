"""Stage 1 of the MIXED cross-domain stream (s044): per-modality lenses ->
block-structured 52-class descriptors + static joint-accuracy sanity gate.

Rocky's design (recovered from the killed s043 session):
  * three domains, ONE 52-class union (full-softmax, no task id at test):
      taxonomy (32 tabular species, data_hard)  -> global classes  0..31
      MNIST    (10 digits, 28x28 grayscale)      -> global classes 32..41
      CIFAR    (10-class slice of CIFAR-100 luma) -> global classes 42..51
  * PURE-trioron lenses (no frozen cortex):
      taxonomy -> the 1D lens (per-feature phase-code + 1xk adaptive patches,
                  `taxonomy_lens1d`), the "certain setup" that works on 1D data
      MNIST    -> 2D fixed-Gabor scattering S1(+)S2 (`mnist_scatter_deep`)
      CIFAR    -> the SAME 2D scattering, retuned to the 32x32 grid (luma)
  * BLOCK-STRUCTURED input: each example fills ONLY its modality's block of a
    fixed-width vector (other blocks zero). The shared soma (Stage 2) reads all
    three blocks; the triparametric node does the continual-learning work.

This module is the perception front-end + a STATIC sanity gate (all data at
once, no CL): per-domain accuracy in isolation + joint 52-class centroid. It
confirms each lens separates its own classes before the CL stream is built.

Run:  python3 experiments/progenitor/mixed_stream_lenses.py
"""
from __future__ import annotations

import math
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from trioron.core.arena import Arena
from trioron.core.envelope import Envelope
from trioron.legacy.donorkit.datasets import DatasetBundle
from experiments.progenitor.conv_proposer import tile_patches
from experiments.progenitor.taxonomy_manifold import classify_archive
from experiments.progenitor.mnist_conv_fixed import balanced
from experiments.progenitor.mnist_scatter_deep import (
    gabor, pool_to, build_bank, modulus_maps,
    ORIENTS1, FREQS1, ORIENTS2, FREQS2, K1, K2,
)
from experiments.progenitor.data_hard import make_split, bayes_accuracy
from experiments.progenitor.taxonomy_lens1d import per_feature_q, lens_desc
from experiments.progenitor.fingerprint_lens import adaptive_patches

# ── global class layout (the 52-class union) ────────────────────────────
TAX_K, MNIST_K, CIFAR_K = 32, 10, 10
TAX_OFF, MNIST_OFF, CIFAR_OFF = 0, 32, 42
N_CLASS = TAX_K + MNIST_K + CIFAR_K            # 52
CIFAR_CLASSES = list(range(10))                # fixed 10-fine slice of CIFAR-100

PER_TRAIN = 128                                # per class (taxonomy ceiling = data_hard default)
PER_TEST = 256


# ── 2D scattering lens, parameterized by grid side (MNIST 28, CIFAR 32) ──
class ScatterLens:
    """Fixed-Gabor S1(+)S2 scattering for a single-channel grid x grid input.
    Reuses mnist_scatter_deep's leaf helpers; only `scatter` is grid-coupled,
    so it lives here with grid-derived H1/H2."""

    def __init__(self, grid: int, s1: int = 2, s2: int = 1,
                 s1_pool: int = 5, s2_pool: int = 2):
        self.grid, self.s1_pool, self.s2_pool = grid, s1_pool, s2_pool
        self.H1 = (grid - K1) // s1 + 1
        self.H2 = (self.H1 - K2) // s2 + 1
        self.n_l1 = len(ORIENTS1) * len(FREQS1)
        self.n_l2 = len(ORIENTS2) * len(FREQS2)

        p1, _ = tile_patches(grid, grid, K1, s1)
        self.a1 = Arena(Envelope(), capacity=grid * grid + 2 * self.n_l1 * len(p1) + 128)
        self.a1.alloc(grid * grid); self.inc1 = list(range(grid * grid))
        self.bank1 = build_bank(self.a1, self.inc1, p1, K1, ORIENTS1, FREQS1)

        p2, _ = tile_patches(self.H1, self.H1, K2, s2)
        self.a2 = Arena(Envelope(), capacity=self.H1 * self.H1 + 2 * self.n_l2 * len(p2) + 128)
        self.a2.alloc(self.H1 * self.H1); self.inc2 = list(range(self.H1 * self.H1))
        self.bank2 = build_bank(self.a2, self.inc2, p2, K2, ORIENTS2, FREQS2)

    @property
    def dim(self) -> int:
        return self.n_l1 * self.s1_pool ** 2 + self.n_l1 * self.n_l2 * self.s2_pool ** 2

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """[N, grid*grid] -> [N, dim] scattering descriptor (S1 (+) S2).
        If X is [N, C, grid*grid] (multi-channel), scatter each channel
        independently and concat -> [N, C*dim] (per-channel color scattering)."""
        if X.dim() == 3:
            return torch.cat([self(X[:, c]) for c in range(X.shape[1])], 1)
        U1 = modulus_maps(self.a1, self.inc1, self.bank1, X)
        S1 = torch.cat([pool_to(u, self.H1, self.s1_pool) for u in U1], 1)
        s2_chunks = []
        for u in U1:                                          # depthwise per L1 channel
            U2 = modulus_maps(self.a2, self.inc2, self.bank2, u)
            s2_chunks.extend(pool_to(v, self.H2, self.s2_pool) for v in U2)
        S2 = torch.cat(s2_chunks, 1)
        return torch.cat([S1, S2], 1)


# ── 1D lens for the tabular taxonomy (the "certain setup") ──────────────
TAX_K_PATCH, TAX_STRIDE = 2, 1                  # 1x2 overlapping patches (richest variant)


class TaxonomyLens:
    """Per-feature phase-code (quantize -> cos/sin carrier) + 1xk adaptive
    patches (fold adjacent features into a complex patch-mean). The lens setup
    that works on 1D/tabular data (taxonomy_lens1d)."""

    def __init__(self, n_feat: int):
        self.patches = adaptive_patches((n_feat,), k=TAX_K_PATCH, stride=TAX_STRIDE)
        self.lo = self.hi = None

    @property
    def dim(self) -> int:
        return 2 * len(self.patches)

    def fit(self, Xtr: torch.Tensor):
        _, self.lo, self.hi = per_feature_q(Xtr)
        return self

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        Q, _, _ = per_feature_q(X, self.lo, self.hi)
        return torch.from_numpy(lens_desc(Q, self.patches)).float()


# ── data loaders (one 10-class luma slice of CIFAR-100, on disk) ────────
def load_cifar_rgb(train: bool):
    """Per-channel CIFAR-100 luma-free: keep R,G,B. Returns [N, 3, 1024]
    (each channel a flattened 32x32 grid) + local labels 0..9."""
    from torchvision import datasets as tvd
    ds = tvd.CIFAR100(root="outputs/data", train=train, download=False)
    img = torch.from_numpy(ds.data).float().div_(255.0)        # (N,32,32,3)
    y = torch.tensor(ds.targets, dtype=torch.long)
    keep = torch.isin(y, torch.tensor(CIFAR_CLASSES))
    img, y = img[keep], y[keep]
    chans = img.permute(0, 3, 1, 2).reshape(len(img), 3, -1)   # [N,3,1024]
    remap = {c: i for i, c in enumerate(CIFAR_CLASSES)}
    y = torch.tensor([remap[int(v)] for v in y], dtype=torch.long)
    return chans, y


def load_domains(seed: int):
    """Return {domain: (Xtr, ytr_local, Xte, yte_local)} balanced per class."""
    # taxonomy
    tr, te = make_split(n_per_train=PER_TRAIN, n_per_test=PER_TEST)
    tax = (tr.x, tr.y, te.x, te.y)
    # MNIST
    b = DatasetBundle(["mnist"])
    Mtr, ymtr = b.task_view("mnist", list(range(10)), list(range(10)), split="train").all_examples()
    Mte, ymte = b.task_view("mnist", list(range(10)), list(range(10)), split="test").all_examples()
    Mtr, ymtr = balanced(Mtr, ymtr, PER_TRAIN, seed)
    Mte, ymte = balanced(Mte, ymte, PER_TEST, seed + 1000)
    mnist = (Mtr, ymtr, Mte, ymte)
    # CIFAR (per-channel color)
    Ctr, yctr = load_cifar_rgb(train=True)
    Cte, ycte = load_cifar_rgb(train=False)
    Ctr, yctr = balanced(Ctr, yctr, PER_TRAIN, seed)
    Cte, ycte = balanced(Cte, ycte, PER_TEST, seed + 1000)
    cifar = (Ctr, yctr, Cte, ycte)
    return {"taxonomy": tax, "mnist": mnist, "cifar": cifar}


# ── build descriptors + block assembly ──────────────────────────────────
def build_descriptors(seed: int = 0):
    """Per-domain lens descriptors + a fixed-width BLOCK layout. Returns
    (per_domain, block_dim, offsets) where per_domain[dom] = dict with
    Dtr/Dte (lens descriptors, torch float), ytr/yte (LOCAL), goff (global
    class offset), boff/bdim (block offset/width in the shared input)."""
    dom = load_domains(seed)

    tax_lens = TaxonomyLens(dom["taxonomy"][0].shape[1]).fit(dom["taxonomy"][0])
    mnist_lens = ScatterLens(grid=28)
    cifar_lens = ScatterLens(grid=32)
    lenses = {"taxonomy": tax_lens, "mnist": mnist_lens, "cifar": cifar_lens}
    goff = {"taxonomy": TAX_OFF, "mnist": MNIST_OFF, "cifar": CIFAR_OFF}

    per, boff, cur = {}, {}, 0
    for name in ("taxonomy", "mnist", "cifar"):
        Xtr, ytr, Xte, yte = dom[name]
        lens = lenses[name]
        Dtr, Dte = lens(Xtr), lens(Xte)
        per[name] = dict(Dtr=Dtr, Dte=Dte, ytr=ytr, yte=yte, goff=goff[name],
                         boff=cur, bdim=Dtr.shape[1])
        boff[name] = cur
        cur += Dtr.shape[1]
    return per, cur, boff


def to_block(per, name, split):
    """Place a domain's lens descriptor into the shared block vector (zeros
    elsewhere) and return (block_X [N, block_dim], global_y)."""
    block_dim = sum(per[n]["bdim"] for n in per)
    d = per[name]
    D = d["Dtr"] if split == "train" else d["Dte"]
    y = d["ytr"] if split == "train" else d["yte"]
    X = torch.zeros(len(D), block_dim)
    X[:, d["boff"]:d["boff"] + d["bdim"]] = D
    return X, y + d["goff"]


# ── static sanity gate ──────────────────────────────────────────────────
def centroid_predict(Dtr, ytr, Dte, classes):
    cents = torch.stack([Dtr[ytr == c].mean(0) for c in classes])
    pred = torch.cdist(Dte, cents).argmin(1)
    return torch.tensor([classes[i] for i in pred])


def main():
    t0 = time.time()
    seed = int(os.environ.get("SEED", "0"))
    per, block_dim, boff = build_descriptors(seed)

    print(f"MIXED stream Stage 1 — 3 lenses, {N_CLASS}-class union, block dim {block_dim}")
    print(f"  layout: taxonomy[{TAX_OFF}:{TAX_OFF+TAX_K}] block {per['taxonomy']['bdim']}d | "
          f"mnist[{MNIST_OFF}:{MNIST_OFF+MNIST_K}] block {per['mnist']['bdim']}d | "
          f"cifar[{CIFAR_OFF}:{CIFAR_OFF+CIFAR_K}] block {per['cifar']['bdim']}d")
    tr_t, te_t = make_split(n_per_train=PER_TRAIN, n_per_test=PER_TEST)
    print(f"  taxonomy Bayes ceiling = {bayes_accuracy(te_t):.3f}\n")

    # (1) per-domain accuracy in ISOLATION (own lens, own classes)
    print(f"  {'domain':<10} {'K':>3} {'dim':>5} {'centroid':>9} {'Mahalanobis':>12}")
    block_rows = []
    for name in ("taxonomy", "mnist", "cifar"):
        d = per[name]
        K = {"taxonomy": TAX_K, "mnist": MNIST_K, "cifar": CIFAR_K}[name]
        cls = list(range(K))
        cen = float((centroid_predict(d["Dtr"], d["ytr"], d["Dte"], cls) == d["yte"]).float().mean())
        mah = (float((classify_archive(d["Dtr"].float(), d["ytr"], d["Dte"].float(), K, full_cov=True)
                      == d["yte"]).float().mean()) if d["bdim"] < len(d["Dtr"]) else float("nan"))
        ms = "(dim>n)" if math.isnan(mah) else f"{mah:.3f}"
        print(f"  {name:<10} {K:>3} {d['bdim']:>5} {cen:>9.3f} {ms:>12}")

    # (2) joint 52-class on the BLOCK vectors (centroid)
    Xtr = torch.cat([to_block(per, n, "train")[0] for n in per])
    ytr = torch.cat([to_block(per, n, "train")[1] for n in per])
    Xte = torch.cat([to_block(per, n, "test")[0] for n in per])
    yte = torch.cat([to_block(per, n, "test")[1] for n in per])
    pred = centroid_predict(Xtr, ytr, Xte, list(range(N_CLASS)))
    joint = float((pred == yte).float().mean())
    print(f"\n  JOINT {N_CLASS}-class block centroid = {joint:.3f}  (chance {1/N_CLASS:.3f})")
    for name in ("taxonomy", "mnist", "cifar"):
        d = per[name]
        m = (yte >= d["goff"]) & (yte < d["goff"] + {"taxonomy": TAX_K, "mnist": MNIST_K, "cifar": CIFAR_K}[name])
        print(f"    {name:<10} joint-slice acc = {float((pred[m] == yte[m]).float().mean()):.3f}")
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
