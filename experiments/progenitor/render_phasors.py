"""Visualize the lock-in phasor on the quanta (s041 pedagogy).

Each quantum q -> a unit phasor at angle theta = 2*pi*q/1000 on the complex
unit circle. Three panels:
  (1) one object's 15 feature-phasors (the prism reading one sample);
  (2) coherent integration: one feature across a class's samples -> the spokes
      cluster, their vector sum (resultant) is LONG (locks in, amplitude ~ N);
  (3) the cos-fold pathology: a feature that is discriminative in q across
      classes lands as MIRROR images across the real axis (q and 1000-q share
      a cos), so the receptor partly blurs it -> falsely coherent on the real
      axis. This is the "interfere after pi" effect, made visible.

Run:  python3 experiments/progenitor/render_phasors.py
"""
from __future__ import annotations

import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trioron.core.receptor import quantize, quantize_frame
import experiments.progenitor.stream_sim_3class as S
from experiments.progenitor.render_spectra import wavelength_to_rgb

LAM_LO, LAM_HI = 380.0, 750.0


def hue(q):
    return wavelength_to_rgb(LAM_LO + (float(q) / 1000.0) * (LAM_HI - LAM_LO))


def circle(ax, title):
    th = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="0.8", lw=1)
    ax.axhline(0, color="0.9", lw=0.6); ax.axvline(0, color="0.9", lw=0.6)
    ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)


def spoke(ax, q, color, lw=1.5, alpha=1.0, tip=True):
    t = 2 * math.pi * float(q) / 1000.0
    ax.plot([0, math.cos(t)], [0, math.sin(t)], color=color, lw=lw, alpha=alpha)
    if tip:
        ax.plot(math.cos(t), math.sin(t), "o", color=color, ms=4, alpha=alpha)


def resultant(ax, qs, color):
    z = np.mean([np.exp(1j * 2 * math.pi * float(q) / 1000.0) for q in qs])
    ax.plot([0, z.real], [0, z.imag], color=color, lw=3)
    ax.plot(z.real, z.imag, "o", color=color, ms=7)
    return abs(z)


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

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))

    # (1) one object: 15 feature phasors, colored by hue
    circle(ax[0], "(1) one object (idx0, '1'): 15 feature phasors")
    q0 = canon(0)
    for f in range(15):
        spoke(ax[0], int(q0[f]), hue(int(q0[f])), lw=1.6)

    # (2) coherent: feature pos1 (middle, always bright) across class '1'
    circle(ax[1], "(2) coherent: pos1 across 20 '1's")
    ones = [i for i in range(len(X)) if y[i] == "1"]
    qs2 = [int(canon(i)[1]) for i in ones]
    for q in qs2:
        spoke(ax[1], q, "0.6", lw=1.0, alpha=0.5)
    R2 = resultant(ax[1], qs2, "crimson")
    ax[1].text(-1.15, -1.12, f"R = {R2:.2f}  (long = locks in)", fontsize=8)

    # (3) cos-fold: feature pos0 across ALL classes (dark for '1', bright for '2'/'3')
    circle(ax[2], "(3) cos-fold: pos0 across ALL classes")
    qs3 = [int(canon(i)[0]) for i in range(len(X))]
    for q in qs3:
        spoke(ax[2], q, "0.6", lw=0.8, alpha=0.4)
    R3 = resultant(ax[2], qs3, "navy")
    ax[2].text(-1.15, -1.12, f"R = {R3:.2f}  (mirror across real axis)", fontsize=8)

    fig.suptitle("Lock-in phasors on the quanta  (theta = 2*pi*q/1000)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = "outputs/phasors.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}   R_coherent={R2:.3f}  R_fold={R3:.3f}")


if __name__ == "__main__":
    main()
