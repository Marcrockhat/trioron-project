"""Lock-in tightening over the stream (s041 pedagogy).

Watch a coherent feature's resultant grow as samples accumulate 1->N:
  top row : the unit circle at N = 1,3,6,10,15,20 — spokes pile up, the mean
            resultant (R) stabilizes in direction and lengthens.
  bottom  : amplitude |sum e^{i theta}| vs N for a COHERENT feature (rides ~N,
            above the sqrt(N) noise floor) vs an INCOHERENT one (hugs sqrt(N));
            and margin = amplitude / sqrt(N) — the resolver's signal — which
            clears the threshold for the coherent feature and not the other.

Run:  python3 experiments/progenitor/render_lockin_growth.py
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

    # coherent: pos1 (middle, always bright) over the 20 '1's in stream order
    ones = [i for i in range(len(X)) if y[i] == "1"]
    th_coh = [2 * math.pi * int(canon(i)[1]) / 1000.0 for i in ones]
    # incoherent: pos0 over the whole mixed stream
    th_inc = [2 * math.pi * int(canon(i)[0]) / 1000.0 for i in range(len(X))]

    def cumamp(ths):
        z = np.cumsum([np.exp(1j * t) for t in ths])
        return np.abs(z), z

    amp_c, z_c = cumamp(th_coh)
    amp_i, _ = cumamp(th_inc)

    fig = plt.figure(figsize=(13, 7.5))
    snaps = [1, 3, 6, 10, 15, 20]
    for k, N in enumerate(snaps):
        ax = fig.add_subplot(2, 6, k + 1)
        t = np.linspace(0, 2 * math.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color="0.85", lw=1)
        for tt in th_coh[:N]:
            ax.plot([0, math.cos(tt)], [0, math.sin(tt)], color="0.7", lw=0.8, alpha=0.5)
        zbar = z_c[N - 1] / N                       # mean resultant
        ax.plot([0, zbar.real], [0, zbar.imag], color="crimson", lw=2.5)
        ax.plot(zbar.real, zbar.imag, "o", color="crimson", ms=6)
        ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"N={N}  R={abs(zbar):.2f}", fontsize=9)

    Ns = np.arange(1, 21)
    axA = fig.add_subplot(2, 2, 3)
    axA.plot(Ns, amp_c, "crimson", marker="o", ms=3, label="coherent feature")
    axA.plot(np.arange(1, len(amp_i) + 1), amp_i, "navy", marker=".", ms=3,
             label="incoherent feature")
    axA.plot(Ns, np.sqrt(Ns), "k--", lw=1, label="$\\sqrt{N}$ noise floor")
    axA.plot(Ns, Ns, color="0.7", lw=1, ls=":", label="$N$ (perfect coherence)")
    axA.set_xlabel("samples accumulated N"); axA.set_ylabel("amplitude  |$\\Sigma e^{i\\theta}$|")
    axA.set_title("amplitude grows ~N (coherent) vs ~$\\sqrt{N}$ (noise)", fontsize=9)
    axA.legend(fontsize=7); axA.grid(alpha=0.3)

    axB = fig.add_subplot(2, 2, 4)
    axB.plot(Ns, amp_c / np.sqrt(Ns), "crimson", marker="o", ms=3, label="coherent")
    axB.plot(np.arange(1, len(amp_i) + 1), amp_i / np.sqrt(np.arange(1, len(amp_i) + 1)),
             "navy", marker=".", ms=3, label="incoherent")
    axB.axhline(3.0, color="k", ls="--", lw=1, label="k=3 threshold")
    axB.set_xlabel("samples accumulated N"); axB.set_ylabel("margin = amp / $\\sqrt{N}$")
    axB.set_title("margin: coherent clears k, incoherent stays ~1", fontsize=9)
    axB.legend(fontsize=7); axB.grid(alpha=0.3)

    fig.suptitle("Lock-in tightening as the stream accumulates (1 -> 20)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "outputs/lockin_growth.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}   coherent R@20={abs(z_c[-1]/20):.3f}  "
          f"margin@20 coh={amp_c[-1]/math.sqrt(20):.2f} inc={amp_i[-1]/math.sqrt(60):.2f}")


if __name__ == "__main__":
    main()
