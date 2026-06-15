"""s039 KNN encoding probe — is the class information PRESERVED in the
quanta, or lost in the encoding?

The keystone (s037/s038) is "organism full 0.30 << oracle 0.55". The
oracle is a GENERATIVE readout: per-class mean-phasor templates, then a
matched filter. We know those templates COLLIDE (cosine 0.94, 89%
identical) — but that is a property of the MEANS. KNN is a
DISCRIMINATIVE, non-parametric readout: it asks whether individual
samples have class-pure NEIGHBOURHOODS even when the class means
collapse.

So KNN separates two hypotheses cleanly:

  * info IS in the quanta  -> KNN clusters by digit despite the mean
    collision -> the loss is the readout (the mean-template oracle is
    itself leaving info on the table); perception is fine.
  * info is LOST in the encoding -> KNN also fails -> quantization is the
    real bottleneck.

Same data slice / encoding scaffolding as diag_s039_bands. 30-way,
chance 0.033. Representations (all per-sample receptor quantize, reference
quanta {0, N_QUANTA} masked):

  raw      raw pixels (cosine)              -- pre-quantization control
  quanta   q integer vector (cosine)        -- the pocket numbers as-is
  phasor   Z = exp(i.theta) (complex cos)   -- per-sample matched filter
  centered Z - global mean phasor (cx cos)  -- s039 common-mode removal

Cosine KNN: sim(z,t) = Re(<z, conj(t)>) / (|z||t|), k-NN majority vote,
balanced train (seed 0) vs held-out test (seed 1).

Run:  python3 -m experiments.progenitor.diag_s039_knn
Env:  S039_PER_CLASS (default 300), S039_K (csv, default "1,5"),
      S039_GABOR (1 = gabor energy instead of raw pixels).
"""
from __future__ import annotations

import math
import os
import time

import torch

from trioron.core.receptor import quantize, N_QUANTA
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)

DATA_ROOT = os.environ.get(
    "AXIS6_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "outputs", "data"))
PER_CLASS = int(os.environ.get("S039_PER_CLASS", "300"))
K_SWEEP = [int(s) for s in os.environ.get("S039_K", "1,5").split(",")]
USE_GABOR = os.environ.get("S039_GABOR", "0") == "1"
FE = os.environ.get("S039_FE", "gabor" if USE_GABOR else "raw")  # raw|gabor|scatter
INVERT_TEST = os.environ.get("S039_INVERT_TEST", "0") == "1"
N_CLASSES = 30


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


def _frontend(X: torch.Tensor) -> torch.Tensor:
    """Lazy import of the runner's front-ends so the two stay in sync."""
    if FE == "gabor":
        from experiments.progenitor.run_pcll_chained15 import _build_gabor_sense
        return _build_gabor_sense()(X)
    if FE == "scatter":
        from experiments.progenitor.run_pcll_chained15 import _build_scatter_sense
        return _build_scatter_sense(prune=True)(X)
    return X.flatten(1)


def reps(X: torch.Tensor, mu_phasor=None):
    """Return {name: (features, is_complex)} for one image batch.
    mu_phasor: global mean phasor from TRAIN, subtracted for 'centered'."""
    feat = _frontend(X)
    q = quantize(feat)                                  # [N, F] per-sample
    ref = (q == 0) | (q == N_QUANTA)                    # reference quanta
    Z = torch.exp(1j * 2 * math.pi * q / N_QUANTA)
    Z = Z.masked_fill(ref, 0)
    # half-circle: theta in [0, pi] -> NO wraparound, q=0 and q=N no longer
    # collide and cos(dtheta) stays monotonic in |dq| over the whole range.
    Zh = torch.exp(1j * math.pi * q / N_QUANTA).masked_fill(ref, 0)
    qm = q.clone().masked_fill(ref, 0)
    out = {
        "raw": (feat, False),
        "quanta": (qm, False),
        "phasor": (Z, True),
        "halfcirc": (Zh, True),
    }
    if mu_phasor is not None:
        out["centered"] = (Z - mu_phasor, True)
    return out, Z


def knn_acc(Ftr, ytr, Fte, yte, k, complex_feat):
    """Cosine-similarity k-NN majority vote, chunked over test rows."""
    if complex_feat:
        ntr = Ftr.abs().pow(2).sum(-1).sqrt().clamp_min(1e-9)
        nte = Fte.abs().pow(2).sum(-1).sqrt().clamp_min(1e-9)
    else:
        ntr = Ftr.pow(2).sum(-1).sqrt().clamp_min(1e-9)
        nte = Fte.pow(2).sum(-1).sqrt().clamp_min(1e-9)
    correct = 0
    for i in range(0, len(Fte), 1000):
        fe = Fte[i:i + 1000]
        if complex_feat:
            sim = (fe @ Ftr.conj().t()).real
        else:
            sim = fe @ Ftr.t()
        sim = sim / nte[i:i + 1000].unsqueeze(1) / ntr.unsqueeze(0)
        nn = sim.topk(k, dim=1).indices                 # [b, k]
        votes = ytr[nn]                                 # [b, k]
        pred = torch.mode(votes, dim=1).values
        correct += int((pred == yte[i:i + 1000]).sum())
    return correct / len(Fte)


def main() -> None:
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()
    Xtr, ytr = balanced_slice(build_task_views(bundle, specs, split="train"),
                              PER_CLASS, seed=0)
    Xte, yte = balanced_slice(build_task_views(bundle, specs, split="test"),
                              PER_CLASS, seed=1)
    if INVERT_TEST:
        Xte = 1.0 - Xte                                # background swap

    sense = {"gabor": "gabor energy", "scatter": "scattering (2nd-order)",
             "raw": "raw pixels"}.get(FE, FE)
    inv = "  [INVERTED test]" if INVERT_TEST else ""
    print(f"s039 KNN probe{inv} — {N_CLASSES}-way, {sense}, "
          f"train {tuple(Xtr.shape)} test {tuple(Xte.shape)}  "
          f"chance={1.0 / N_CLASSES:.3f}\n")

    _, Ztr0 = reps(Xtr)
    mu = Ztr0.mean(0, keepdim=True)                     # global mean phasor
    rtr, _ = reps(Xtr, mu)
    rte, _ = reps(Xte, mu)

    print(f"{'rep':>10}{'complex':>9}" +
          "".join(f"{'k=' + str(k):>9}" for k in K_SWEEP) + f"{'t(s)':>7}")
    print("-" * (19 + 9 * len(K_SWEEP) + 7))
    for name in rtr:
        Ftr, cx = rtr[name]
        Fte, _ = rte[name]
        t0 = time.time()
        accs = [knn_acc(Ftr, ytr, Fte, yte, k, cx) for k in K_SWEEP]
        print(f"{name:>10}{str(cx):>9}" +
              "".join(f"{a:>9.3f}" for a in accs) +
              f"{time.time() - t0:>7.1f}")


if __name__ == "__main__":
    main()
