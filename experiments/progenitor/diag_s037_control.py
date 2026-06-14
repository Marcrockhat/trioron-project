"""s037 CONTROL — is the matched filter sound, or is the encoding the gap?

The s036 keystone: on raw pixels the phasor matched filter clusters
QUANTA-SPACE (per-sample contrast), not CLASS-SPACE. Templates came out
94% identical (cosine 0.938) and 7/30 labels owned a class. The claim is
that the matched-filter MACHINERY is fine; it is fed an encoding that
does not carry class identity.

This diagnostic isolates that variable WITHOUT the slow council loop.
For each feature transform we build ORACLE templates — the per-true-class
mean phasor over the train split, i.e. the matched filter handed perfect,
un-fragmented class membership — and score nearest-template classification
on held-out test. This is the matched filter's BEST CASE under that
encoding:

  * If raw-pixel oracle accuracy ≈ chance (1/30 ≈ 0.033) with template
    cosine ≈ 0.94, the keystone is confirmed: the encoding, not the
    filter, is the gap (no clustering algorithm could do better than the
    oracle on this encoding).
  * If a DISCRIMINATIVE transform (frozen MobileNet cortex / hand-Gabor)
    lifts oracle accuracy well above chance, the matched filter is sound
    and the remaining gap is definitively the PERCEPTION encoding —
    route: grow class-discriminative receptors (Rocky's cell-spawning
    framing), or a perception transform upstream.

Transforms (Rocky's s037 ruling — BOTH):
  raw     identity 784-d pixels (reproduces the keystone)
  gabor   hand-designed oriented-edge / Gabor bank, conv+relu+pool — a
          self-contained discriminative feature map, no learned weights
  cortex  frozen ImageNet MobileNetV3-Small features (64-d projected) —
          guaranteed-discriminative ceiling; a CONTROL crutch, not a
          permanent transform

Run: python3 -m experiments.progenitor.diag_s037_control
Env: S037_PER_CLASS (train+test samples per class, default 500),
     S037_TRANSFORMS (csv subset, default "raw,gabor,cortex").
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
PER_CLASS = int(os.environ.get("S037_PER_CLASS", "500"))
TRANSFORMS = os.environ.get("S037_TRANSFORMS", "raw,gabor,cortex").split(",")
N_CLASSES = 30   # bench-15: 30 true global classes


# ---------------------------------------------------------------- transforms
def t_raw(x: torch.Tensor) -> torch.Tensor:
    return x


def _gabor_bank(ksize: int = 7) -> torch.Tensor:
    """Hand-designed bank: 4 orientations × 2 wavelengths Gabor + a
    center-surround (LoG-like) filter. No learned weights."""
    filters = []
    half = ksize // 2
    yy, xx = torch.meshgrid(torch.arange(-half, half + 1).float(),
                            torch.arange(-half, half + 1).float(),
                            indexing="ij")
    for wavelength in (3.0, 6.0):
        sigma = 0.5 * wavelength
        for theta in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
            x_t = xx * math.cos(theta) + yy * math.sin(theta)
            y_t = -xx * math.sin(theta) + yy * math.cos(theta)
            env = torch.exp(-(x_t ** 2 + y_t ** 2) / (2 * sigma ** 2))
            g = env * torch.cos(2 * math.pi * x_t / wavelength)
            g = g - g.mean()                       # zero DC
            filters.append(g / g.abs().sum().clamp_min(1e-6))
    # center-surround
    sig_c, sig_s = 1.0, 2.0
    log = (torch.exp(-(xx ** 2 + yy ** 2) / (2 * sig_c ** 2)) / sig_c ** 2
           - torch.exp(-(xx ** 2 + yy ** 2) / (2 * sig_s ** 2)) / sig_s ** 2)
    log = log - log.mean()
    filters.append(log / log.abs().sum().clamp_min(1e-6))
    return torch.stack(filters).unsqueeze(1)       # [C,1,k,k]


_GABOR_W = None


def t_gabor(x: torch.Tensor) -> torch.Tensor:
    global _GABOR_W
    if _GABOR_W is None:
        _GABOR_W = _gabor_bank()
    img = x.view(-1, 1, 28, 28)
    f = F.relu(F.conv2d(img, _GABOR_W, padding=_GABOR_W.shape[-1] // 2))
    f = F.avg_pool2d(f, 4)                          # [N,C,7,7]
    return f.flatten(1)


def t_cortex(x: torch.Tensor) -> torch.Tensor:
    from trioron.legacy.senses.cortex import sense_cortex
    img = x.view(-1, 1, 28, 28).repeat(1, 3, 1, 1)  # gray → 3-channel
    out = []
    for i in range(0, len(img), 1024):
        out.append(sense_cortex(img[i:i + 1024]))
    return torch.cat(out)


TRANSFORM_FN = {"raw": t_raw, "gabor": t_gabor, "cortex": t_cortex}


# ---------------------------------------------------------------- matched filter
def phasor(feats: torch.Tensor) -> torch.Tensor:
    """Per-sample receptor quanta → unit phasors Z[i,f]."""
    q = quantize(feats)
    return torch.exp(1j * 2 * math.pi * q / N_QUANTA)


def phasor_global(Ftr: torch.Tensor, Fte: torch.Tensor):
    """Phasor under a GLOBAL per-feature frame: each feature normalized
    by its dataset-wide [min,max] (a shared frame across samples), NOT
    per-sample. This is the receptor's per-sample frame swapped for a
    standard per-feature normalization — the decisive test of whether
    the per-sample frame is the lossy step."""
    lo = Ftr.amin(0, keepdim=True)
    hi = Ftr.amax(0, keepdim=True)
    span = (hi - lo).clamp_min(1e-9)

    def enc(Fx):
        q = torch.round(N_QUANTA * ((Fx - lo) / span).clamp(0, 1))
        return torch.exp(1j * 2 * math.pi * q / N_QUANTA)
    return enc(Ftr), enc(Fte)


def phasor_masked(feats: torch.Tensor) -> torch.Tensor:
    """Phasor with the lock-in mask rule (spec §10.3): q∈{0, N_QUANTA}
    are reference, not evidence — zeroed so they contribute nothing to
    the template mean or the matched-filter sum."""
    q = quantize(feats)
    Z = torch.exp(1j * 2 * math.pi * q / N_QUANTA)
    Z[(q == 0) | (q == N_QUANTA)] = 0
    return Z


def oracle_templates(Z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-true-class mean phasor [C,F] — the matched filter handed
    perfect, un-fragmented membership."""
    return torch.stack([Z[y == k].mean(0) for k in range(N_CLASSES)])


def evidence(Z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """E[i,k] = Σ_f Re(Z[i,f]·conj(T[k,f])), row-chunked."""
    rows = max(1, int(25_000_000 / max(1, T.shape[0] * T.shape[1])))
    return torch.cat([
        (Z[i:i + rows].unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1)
        for i in range(0, len(Z), rows)])


def centroid_acc(Ftr, ytr, Fte, yte) -> float:
    """Reference ceiling: nearest-centroid in RAW feature space (cosine),
    NO phasor quantization. Quantifies how much class signal the
    per-sample phasor encoding sheds vs the features themselves."""
    Fn = F.normalize(Ftr, dim=1)
    cent = F.normalize(
        torch.stack([Fn[ytr == k].mean(0) for k in range(N_CLASSES)]), dim=1)
    pred = (F.normalize(Fte, dim=1) @ cent.t()).argmax(1)
    return float((pred == yte).float().mean())


def template_cosine(T: torch.Tensor) -> float:
    """Mean off-diagonal pairwise template cosine (complex magnitude
    cosine) — the keystone reported 0.938 on raw."""
    n = T.shape[0]
    norm = T.abs().pow(2).sum(-1).sqrt().clamp_min(1e-9)
    G = (T @ T.conj().t()).abs() / norm.unsqueeze(1) / norm.unsqueeze(0)
    off = G[~torch.eye(n, dtype=torch.bool)]
    return float(off.mean())


# ---------------------------------------------------------------- harness
def balanced_slice(views, per_class, seed):
    """Concatenate task views, then take up to per_class samples of each
    of the 30 global classes."""
    X = torch.cat([v.images for v in views])
    y = torch.cat([v.labels_global for v in views])
    g = torch.Generator().manual_seed(seed)
    idx = []
    for k in range(N_CLASSES):
        ck = (y == k).nonzero().squeeze(1)
        perm = ck[torch.randperm(len(ck), generator=g)][:per_class]
        idx.append(perm)
    idx = torch.cat(idx)
    return X[idx], y[idx]


def main() -> None:
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    Xtr, ytr = balanced_slice(train_views, PER_CLASS, seed=0)
    Xte, yte = balanced_slice(eval_views, PER_CLASS, seed=1)

    chance = 1.0 / N_CLASSES
    print(f"s037 CONTROL — oracle matched-filter on {N_CLASSES} true classes")
    print(f"train {tuple(Xtr.shape)}  test {tuple(Xte.shape)}  "
          f"chance={chance:.3f}\n")
    print(f"{'transform':<10}{'dim':>6}{'persamp':>9}{'masked':>9}"
          f"{'global':>9}{'centroid':>10}{'tmpl_cos':>10}{'t(s)':>8}")
    print("-" * 73)

    for name in TRANSFORMS:
        fn = TRANSFORM_FN[name]
        t0 = time.time()
        Ftr, Fte = fn(Xtr), fn(Xte)
        # per-sample receptor frame (the shipped encoding)
        T = oracle_templates(phasor(Ftr), ytr)
        acc_ps = float((evidence(phasor(Fte), T).argmax(1) == yte)
                       .float().mean())
        # per-sample + lock-in mask (q∈{0,1000} = reference, not evidence)
        Tm = oracle_templates(phasor_masked(Ftr), ytr)
        acc_m = float((evidence(phasor_masked(Fte), Tm).argmax(1) == yte)
                      .float().mean())
        # global per-feature frame (the decisive swap)
        Zg_tr, Zg_te = phasor_global(Ftr, Fte)
        Tg = oracle_templates(Zg_tr, ytr)
        acc_g = float((evidence(Zg_te, Tg).argmax(1) == yte).float().mean())
        cent = centroid_acc(Ftr, ytr, Fte, yte)
        cos = template_cosine(T)
        print(f"{name:<10}{Ftr.shape[1]:>6}{acc_ps:>9.3f}{acc_m:>9.3f}"
              f"{acc_g:>9.3f}{cent:>10.3f}{cos:>10.3f}{time.time() - t0:>8.1f}")


if __name__ == "__main__":
    main()
