"""Dimensionality-adaptive scattering lens + per-class Gaussian fingerprints
(s041 pedagogy / design probe).

Rocky's spec: the data is a filter film; an emitter sweeps references through
it; where it "deflects" is the complex phasor response (cos = in-phase,
sin = quadrature). Group the responses with a lens whose patch is a HYPERCUBE
matched to the data's dimensionality:

    1D  ->  1xk   (k=2 -> 1x2 windows)
    2D  ->  kxk   (k=2 -> 2x2)
    3D  ->  kxkxk
    nD  ->  k^n

Each patch -> a mean phasor (re, im). Concatenate over patches -> a descriptor.
Per class, the descriptor cloud is a Gaussian fingerprint; the CENTROID
distance between classes is the recognition signal (= trioron's ManifoldArchive
mechanism). Cos/sin keep both quadratures, so the pi-fold doesn't collapse it.

Run:  python3 experiments/progenitor/fingerprint_lens.py
"""
from __future__ import annotations

import itertools
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trioron.core.receptor import quantize, quantize_frame
import experiments.progenitor.stream_sim_3class as S


# ── the dimensionality-adaptive filter lens ─────────────────────────────

def adaptive_patches(shape, k=2, stride=1):
    """Hypercube patches over an n-D grid (row-major flatten). shape is the
    grid's per-axis sizes; the patch is k along EVERY axis -> k^len(shape)
    taps. Returns a list of flat-index lists (matched tap order)."""
    dims = len(shape)
    strides = [1] * dims
    for d in range(dims - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    starts = [range(0, shape[d] - k + 1, stride) for d in range(dims)]
    patches = []
    for origin in itertools.product(*starts):
        idxs = []
        for off in itertools.product(*([range(k)] * dims)):
            idxs.append(sum((origin[d] + off[d]) * strides[d] for d in range(dims)))
        patches.append(idxs)
    return patches


def lens_descriptor(q, patches):
    """Per-patch mean phasor (re, im) -> concatenated descriptor. cos and sin
    are the two deflection axes; keeping both avoids the pi-fold."""
    theta = 2 * math.pi * q.float() / 1000.0
    feats = []
    for p in patches:
        z = torch.exp(1j * theta[p])          # unit phasors in the patch
        m = z.mean()
        feats += [m.real.item(), m.imag.item()]
    return np.array(feats)


# ── data + descriptors ──────────────────────────────────────────────────

def main():
    X, y = S.build_X(n_per_class=20, seed=7)
    GRID = (5, 3)                             # 5 rows x 3 cols  (2D)
    FLO, FHI = float("inf"), float("-inf")
    for i in range(len(X)):
        _, lo, hi = quantize_frame(X[i].unsqueeze(0))
        FLO, FHI = min(FLO, float(lo)), max(FHI, float(hi))

    def canon(i):
        qr = quantize(X[i]).to(torch.int64)
        _, lo, hi = quantize_frame(X[i].unsqueeze(0))
        return S.canonical(qr.float(), float(lo), float(hi), FLO, FHI).to(torch.int64)

    patches = adaptive_patches(GRID, k=2)     # 2D -> 2x2
    print(f"lens: {GRID} (2D) -> 2x2 patches -> {len(patches)} patches "
          f"-> {2*len(patches)}-D descriptor")
    for nm, shp in [("1D", (15,)), ("2D", (5, 3)), ("3D", (3, 3, 3)),
                    ("4D", (2, 2, 2, 2))]:
        print(f"   {nm} {shp} -> k=2 hypercube -> {len(adaptive_patches(shp,2))} patches")

    D = np.stack([lens_descriptor(canon(i), patches) for i in range(len(X))])
    labs = np.array([int(l) for l in y])

    # per-class Gaussian fingerprint = centroid (+ cloud)
    cents = {c: D[labs == c].mean(0) for c in (1, 2, 3)}
    dist = {(a, b): float(np.linalg.norm(cents[a] - cents[b]))
            for a, b in [(1, 2), (1, 3), (2, 3)]}
    # nearest-centroid in-sample separability
    pred = np.array([min((1, 2, 3), key=lambda c: np.linalg.norm(D[i] - cents[c]))
                     for i in range(len(D))])
    acc = float((pred == labs).mean())
    print(f"centroid distances: 1-2={dist[(1,2)]:.2f}  1-3={dist[(1,3)]:.2f}  "
          f"2-3={dist[(2,3)]:.2f}   nearest-centroid acc={acc:.3f}")

    # ── figure ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))

    # (a) GLOBAL 1D quanta density — value histogram alone (weak)
    for c, col in zip((1, 2, 3), ("crimson", "green", "navy")):
        qs = np.concatenate([canon(i).numpy() for i in range(len(X)) if labs[i] == c])
        ax[0].hist(qs, bins=40, range=(0, 1000), density=True, histtype="step",
                   color=col, lw=1.8, label=f"class {c}")
    ax[0].set_title("(a) global quanta density (1D) — overlaps", fontsize=9)
    ax[0].set_xlabel("quantum q"); ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)

    # (b) 2x2-scattering descriptors, PCA-2D, centroids + distances
    Dc = D - D.mean(0)
    U, Sv, Vt = np.linalg.svd(Dc, full_matrices=False)
    P = Dc @ Vt[:2].T
    Pc = {c: P[labs == c].mean(0) for c in (1, 2, 3)}
    for c, col in zip((1, 2, 3), ("crimson", "green", "navy")):
        pts = P[labs == c]
        ax[1].scatter(pts[:, 0], pts[:, 1], s=18, color=col, alpha=0.6, label=f"class {c}")
        ax[1].scatter(*Pc[c], s=180, color=col, marker="*", edgecolor="k", zorder=5)
    for (a, b) in [(1, 2), (1, 3), (2, 3)]:
        ax[1].plot([Pc[a][0], Pc[b][0]], [Pc[a][1], Pc[b][1]], "k--", lw=0.8)
    ax[1].set_title(f"(b) 2x2 scattering descriptors (PCA)  acc={acc:.2f}", fontsize=9)
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend(fontsize=8)

    # (c) the class fingerprints themselves (mean descriptor curves)
    for c, col in zip((1, 2, 3), ("crimson", "green", "navy")):
        ax[2].plot(cents[c], color=col, lw=1.6, marker="o", ms=3, label=f"class {c}")
    ax[2].set_title("(c) Gaussian fingerprints (mean descriptor)", fontsize=9)
    ax[2].set_xlabel("descriptor dim (patch re/im)"); ax[2].set_ylabel("value")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle("Dimensionality-adaptive scattering lens -> per-class fingerprints",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "outputs/fingerprint_lens.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
