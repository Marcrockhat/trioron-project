"""WHAT + WHERE together: stereo position -> crop -> scattering-lens identity
(s041 pedagogy / design probe).

A grayscale digit (3x5, classes 1/2/3) is dropped at a RANDOM column on a
5 x W field and we must report BOTH where it is and what it is, with the two
phasor channels cooperating:

  WHERE : dual-frequency stereo emitter -> absolute column (Vernier unwrap)
  crop  : take the 3-col window at the estimated column (registration)
  WHAT  : contrast-normalized scattering-lens descriptor of the crop ->
          nearest class centroid (= ManifoldArchive fingerprint)

The point: identity is recovered from a MOVING, OFF-CENTER object because the
where-channel registers it first. A fixed lens with no registration would see
a different descriptor at every position.

Run:  python3 experiments/progenitor/combined_what_where.py
"""
from __future__ import annotations

import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trioron.core.receptor import quantize
import experiments.progenitor.stream_sim_3class as S
from experiments.progenitor.fingerprint_lens import adaptive_patches, lens_descriptor

ROWS, COLS = 5, 3
W = 40
PERIOD_A, PERIOD_B = 16.0, 13.0
WA, WB, OFFB = 2 * math.pi / PERIOD_A, 2 * math.pi / PERIOD_B, 0.3 * math.pi
PATCHES = adaptive_patches((ROWS, COLS), k=2)


def digit_field(label, col, g):
    """A 3x5 grayscale variant placed at column `col` on a 5 x W field."""
    d = S.variant(label, g).view(ROWS, COLS)            # per-sample variant
    field = 0.04 * torch.rand(ROWS, W, generator=g)
    field[:, col:col + COLS] = torch.maximum(field[:, col:col + COLS], d)
    return field


def where_estimate(field, grid):
    """Stereo dual-frequency column estimate of the object's centroid."""
    cols = np.arange(W)
    img = field.numpy()
    za = (img * np.exp(1j * (WA * cols))[None, :]).sum()
    zb = (img * np.exp(1j * (WB * cols + OFFB))[None, :]).sum()
    pa, pb = np.angle(za), np.angle(zb)
    pgrid_a, pgrid_b = WA * grid, WB * grid + OFFB
    cost = (np.abs(np.angle(np.exp(1j * (pgrid_a - pa))))
            + np.abs(np.angle(np.exp(1j * (pgrid_b - pb)))))
    return grid[np.argmin(cost)]                        # centroid column


def descriptor_of(window15):
    q = quantize(window15).to(torch.int64)              # per-sample contrast-norm
    return lens_descriptor(q, PATCHES)


def main():
    grid = np.linspace(0, W - 1, 2000)

    # train centroids from clean variants at column 0 (registered)
    gtr = torch.Generator().manual_seed(11)
    cent = {}
    for lbl in ("1", "2", "3"):
        ds = [descriptor_of(S.variant(lbl, gtr).view(ROWS, COLS).reshape(-1))
              for _ in range(40)]
        cent[int(lbl)] = np.mean(ds, 0)

    # test: random digit at random column on the field
    gte = torch.Generator().manual_seed(77)
    rng = np.random.default_rng(5)
    N = 60
    pos_err, id_ok, examples = [], [], []
    for n in range(N):
        lbl = ("1", "2", "3")[n % 3]
        col = int(rng.integers(0, W - COLS))
        field = digit_field(lbl, col, gte)
        c_hat = where_estimate(field, grid)             # centroid col
        col_hat = int(round(c_hat - 1))                 # left edge = centroid-1
        col_hat = max(0, min(W - COLS, col_hat))
        win = field[:, col_hat:col_hat + COLS].reshape(-1)
        d = descriptor_of(win)
        pred = min((1, 2, 3), key=lambda c: np.linalg.norm(d - cent[c]))
        pos_err.append(abs(col_hat - col))
        id_ok.append(int(pred == int(lbl)))
        if len(examples) < 4:
            examples.append((field, col, col_hat, lbl, pred))

    print(f"N={N}  position MAE = {np.mean(pos_err):.2f} cols   "
          f"identity acc = {np.mean(id_ok):.3f}   (field W={W}, chance id=0.33)")

    fig, ax = plt.subplots(4, 1, figsize=(9, 6.4))
    for k, (field, col, col_hat, lbl, pred) in enumerate(examples):
        ax[k].imshow(field.numpy(), cmap="magma", aspect="auto")
        ax[k].add_patch(plt.Rectangle((col_hat - 0.5, -0.5), COLS, ROWS,
                        fill=False, edgecolor="cyan", lw=2))
        ax[k].set_title(f"true: '{lbl}' @col{col}   |   read: '{pred}' @col{col_hat}"
                        f"   {'OK' if pred==int(lbl) else 'MISS'}", fontsize=9)
        ax[k].set_yticks([])
    fig.suptitle("WHAT + WHERE: stereo localizes -> crop -> lens identifies "
                 f"(pos MAE {np.mean(pos_err):.1f}, id {np.mean(id_ok):.2f})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "outputs/combined_what_where.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
