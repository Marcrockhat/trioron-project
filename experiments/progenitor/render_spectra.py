"""Render canonical-q digits as VISIBLE-SPECTRUM color images (s041 pedagogy).

Maps each pocket q in [0,1000] onto the visible band 380-750 nm, converts
wavelength -> RGB (Bruton's approximation), and renders 5 samples x 3 classes
from the toy stream as a montage PNG. So a class's "spectrum" is something you
can literally look at.

Run:  python3 experiments/progenitor/render_spectra.py
"""
from __future__ import annotations

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trioron.core.receptor import quantize, quantize_frame
import experiments.progenitor.stream_sim_3class as S

LAM_LO, LAM_HI = 380.0, 750.0      # visible band (nm)


def wavelength_to_rgb(nm):
    """Bruton's wavelength->RGB (gamma 0.8). nm scalar -> (r,g,b) in [0,1]."""
    g = 0.8
    if nm < 440:
        r, gr, b = -(nm - 440) / 60, 0.0, 1.0
    elif nm < 490:
        r, gr, b = 0.0, (nm - 440) / 50, 1.0
    elif nm < 510:
        r, gr, b = 0.0, 1.0, -(nm - 510) / 20
    elif nm < 580:
        r, gr, b = (nm - 510) / 70, 1.0, 0.0
    elif nm < 645:
        r, gr, b = 1.0, -(nm - 645) / 65, 0.0
    else:
        r, gr, b = 1.0, 0.0, 0.0
    if nm < 420:
        f = 0.3 + 0.7 * (nm - 380) / 40
    elif nm > 700:
        f = 0.3 + 0.7 * (750 - nm) / 50
    else:
        f = 1.0
    return tuple((max(0.0, c) * f) ** g for c in (r, gr, b))


def q_to_rgb_grid(q):
    """q [15] -> RGB image [5,3,3] via visible-band wavelength."""
    img = np.zeros((5, 3, 3))
    for r in range(5):
        for c in range(3):
            nm = LAM_LO + (float(q[r * 3 + c]) / 1000.0) * (LAM_HI - LAM_LO)
            img[r, c] = wavelength_to_rgb(nm)
    return img


def main():
    X, y = S.build_X(n_per_class=20, seed=7)
    FLO, FHI = float("inf"), float("-inf")
    for i in range(len(X)):
        _, lo, hi = quantize_frame(X[i].unsqueeze(0))
        FLO, FHI = min(FLO, float(lo)), max(FHI, float(hi))

    def canon(i):
        qr = quantize(X[i]).to(torch.int64)
        _, lo, hi = quantize_frame(X[i].unsqueeze(0))
        return S.canonical(qr.float(), float(lo), float(hi), FLO, FHI).to(torch.int64)

    classes = ["1", "2", "3"]
    n = 5
    fig, axes = plt.subplots(len(classes), n, figsize=(n * 1.6, len(classes) * 2.4))
    fig.suptitle("Canonical q -> visible spectrum (380-750 nm)  |  5 samples x class",
                 fontsize=11)
    for rcls, lbl in enumerate(classes):
        idxs = [i for i in range(len(X)) if y[i] == lbl][-n:]
        for ccol, i in enumerate(idxs):
            ax = axes[rcls, ccol]
            ax.imshow(q_to_rgb_grid(canon(i)), interpolation="nearest", aspect="equal")
            ax.set_title(f"'{lbl}'  idx{i}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "outputs/spectra_visible.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  ({len(classes)}x{n} grid, visible-band RGB)")


if __name__ == "__main__":
    main()
