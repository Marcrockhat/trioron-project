"""Stereo (dual-frequency) emitter -> absolute position of a moving 2D object
(s041 pedagogy / design probe).

A grayscale blob translates across a HxW field over T frames. Two coherent
emitters impose phase ramps across the columns:
    emitter A : theta_A(c) = w_A * c + 0
    emitter B : theta_B(c) = w_B * c + 0.3*pi      (DIFFERENT rate w_B != w_A)
The object's lock-in response to each emitter is Z = sum_lit I * e^{i theta}.
phase(Z) ~ theta_emit(centroid column), so each emitter reads position modulo
its own period -> a sawtooth (wrap-ambiguous). With two incommensurate rates
the (phi_A, phi_B) pair is UNIQUE over the whole field -> absolute position
(the Vernier / dual-frequency unwrap). This is the "position is a relation
between references, not a value" fix (s039).

Run:  python3 experiments/progenitor/stereo_emitter_2d.py
"""
from __future__ import annotations

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H, W = 8, 48
T = 40
PERIOD_A, PERIOD_B = 16.0, 13.0          # incommensurate -> Vernier
W_A = 2 * math.pi / PERIOD_A
W_B = 2 * math.pi / PERIOD_B
OFF_B = 0.3 * math.pi


def frame(center_c, rng):
    """A soft vertical blob (3-col gaussian) at column center_c, light noise."""
    img = 0.05 * rng.random((H, W))
    cols = np.arange(W)
    prof = np.exp(-0.5 * ((cols - center_c) / 1.2) ** 2)
    img += 0.9 * prof[None, :] * (0.7 + 0.3 * rng.random((H, 1)))
    return np.clip(img, 0, 1)


def response(img, w, off):
    """Lock-in of the image against an emitter phase-ramp across columns."""
    cols = np.arange(W)
    theta = w * cols + off
    z = (img * np.exp(1j * theta)[None, :]).sum()
    return z


def estimate_single(phi, w, off):
    """Position mod period from one emitter (wrap-ambiguous)."""
    return ((phi - off) / w) % (2 * math.pi / w)


def estimate_stereo(phi_a, phi_b, grid):
    """Pick the column on a fine grid whose predicted (A,B) phases best match,
    in circular distance -> unambiguous over the full field."""
    def cdist(a, b):
        return np.abs(np.angle(np.exp(1j * (a - b))))
    pa = W_A * grid
    pb = W_B * grid + OFF_B
    cost = cdist(pa, phi_a) + cdist(pb, phi_b)
    return grid[np.argmin(cost)]


def main():
    rng = np.random.default_rng(0)
    true_c = np.linspace(3, W - 4, T)
    grid = np.linspace(0, W - 1, 2000)

    phiA, phiB, est_single, est_stereo = [], [], [], []
    for c in true_c:
        img = frame(c, rng)
        za = response(img, W_A, 0.0)
        zb = response(img, W_B, OFF_B)
        pa, pb = np.angle(za), np.angle(zb)
        phiA.append(pa); phiB.append(pb)
        est_single.append(estimate_single(pa, W_A, 0.0))
        est_stereo.append(estimate_stereo(pa, pb, grid))

    phiA, phiB = np.array(phiA), np.array(phiB)
    est_single, est_stereo = np.array(est_single), np.array(est_stereo)
    err = np.abs(est_stereo - true_c)
    print(f"stereo position MAE = {err.mean():.2f} columns (field W={W}, "
          f"periods A={PERIOD_A:.0f} B={PERIOD_B:.0f})")
    print(f"single-emitter range = one period ({2*math.pi/W_A:.0f} cols) -> "
          f"ambiguous beyond it")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

    # (a) a few frames of the moving object
    montage = np.concatenate([frame(c, np.random.default_rng(1))
                              for c in [6, 20, 34, W - 5]], axis=0)
    ax[0].imshow(montage, cmap="magma", aspect="auto")
    ax[0].set_title("(a) object translating across the field (4 frames)", fontsize=9)
    ax[0].set_xlabel("column"); ax[0].set_yticks([])

    # (b) the two emitter phases vs true position (two sawtooths)
    ax[1].plot(true_c, phiA, "crimson", marker="o", ms=3, label=f"emitter A (T={PERIOD_A:.0f})")
    ax[1].plot(true_c, phiB, "navy", marker="s", ms=3, label=f"emitter B (T={PERIOD_B:.0f})")
    ax[1].set_title("(b) each emitter phase WRAPS (ambiguous alone)", fontsize=9)
    ax[1].set_xlabel("true position (column)"); ax[1].set_ylabel("phase (rad)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    # (c) recovered position: single (sawtooth) vs stereo (matches truth)
    ax[2].plot(true_c, true_c, "k--", lw=1, label="truth")
    ax[2].plot(true_c, est_single, color="0.6", marker=".", ms=4,
               label="single emitter (wraps)")
    ax[2].plot(true_c, est_stereo, "green", marker="o", ms=3,
               label=f"stereo unwrap (MAE {err.mean():.1f})")
    ax[2].set_title("(c) stereo recovers ABSOLUTE position", fontsize=9)
    ax[2].set_xlabel("true position"); ax[2].set_ylabel("estimated position")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle("Stereo dual-frequency emitter: absolute position of a moving object",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "outputs/stereo_emitter_2d.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
