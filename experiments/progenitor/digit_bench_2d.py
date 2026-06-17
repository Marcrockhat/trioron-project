"""Larger toy: 10 digits, 2D off-center, stereo (row+col) where + scattering
what, with an ambiguity-margin probe (s041 design probe).

Builds on the validated pieces:
  WHERE : TWO dual-frequency stereo pairs (one per axis) -> (row, col)
  crop  : register the 5x7 window at the estimated (row, col)
  WHAT  : contrast-normalized scattering-lens descriptor -> nearest centroid
  MARGIN: gap between nearest and 2nd-nearest centroid = confidence; a 2->Z
          morph drives it to collapse (the refusal / novelty signal)

Reports: identity accuracy (2-pair vs col-only ablation), position MAE (row,
col), and the margin curve over the 2->Z morph. 'Z' is NOT a training class.

Run:  python3 experiments/progenitor/digit_bench_2d.py
"""
from __future__ import annotations

import math
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trioron.core.receptor import quantize
from experiments.progenitor.fingerprint_lens import adaptive_patches, lens_descriptor

# ── 5x7 dot-matrix font (7 rows x 5 cols) ───────────────────────────────
FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
}
RH, RW = 7, 5
H = int(os.environ.get("FH", "16"))
W = int(os.environ.get("FW", "40"))
PATCHES = adaptive_patches((RH, RW), k=2)
# stereo: incommensurate periods per axis (lcm > field extent)
PC = (16.0, 13.0)          # column pair
PR = (11.0, 9.0)           # row pair
OFFB = 0.3 * math.pi


def template(ch):
    return torch.tensor([[float(c) for c in row] for row in FONT[ch]])


def variant(ch, g, morph_to=None, alpha=0.0):
    t = template(ch)
    if morph_to is not None:
        t = (1 - alpha) * t + alpha * template(morph_to)
    on = 0.6 + 0.4 * torch.rand(RH, RW, generator=g)
    off = 0.15 * torch.rand(RH, RW, generator=g)
    return (t * on + (1 - t) * off).clamp(0, 1)


def place(d, r, c, g):
    field = 0.04 * torch.rand(H, W, generator=g)
    field[r:r + RH, c:c + RW] = torch.maximum(field[r:r + RH, c:c + RW], d)
    return field


def _axis_estimate(weight_1d, periods, grid):
    """Dual-frequency unwrap along one axis. weight_1d: marginal intensity."""
    n = len(weight_1d)
    idx = np.arange(n)
    wa, wb = 2 * math.pi / periods[0], 2 * math.pi / periods[1]
    za = (weight_1d * np.exp(1j * wa * idx)).sum()
    zb = (weight_1d * np.exp(1j * (wb * idx + OFFB))).sum()
    pa, pb = np.angle(za), np.angle(zb)
    cost = (np.abs(np.angle(np.exp(1j * (wa * grid - pa))))
            + np.abs(np.angle(np.exp(1j * (wb * grid + OFFB - pb)))))
    return grid[np.argmin(cost)]


def where_2d(field):
    img = field.numpy()
    gcol = np.linspace(0, W - 1, 2000)
    grow = np.linspace(0, H - 1, 1000)
    c_cen = _axis_estimate(img.sum(0), PC, gcol)        # marginal over rows
    r_cen = _axis_estimate(img.sum(1), PR, grow)        # marginal over cols
    return r_cen, c_cen


def descriptor(win):
    q = quantize(win.reshape(-1)).to(torch.int64)
    return lens_descriptor(q, PATCHES)


def crop(field, r, c):
    r = max(0, min(H - RH, int(round(r)))); c = max(0, min(W - RW, int(round(c))))
    return field[r:r + RH, c:c + RW], r, c


def main():
    digits = [str(i) for i in range(10)]

    # train centroids (registered, clean-ish variants)
    gtr = torch.Generator().manual_seed(3)
    cent = {}
    for ch in digits:
        ds = [descriptor(variant(ch, gtr)) for _ in range(60)]
        cent[ch] = np.mean(ds, 0)

    def classify(d):
        dist = sorted(((np.linalg.norm(d - cent[ch]), ch) for ch in digits))
        gap = dist[1][0] - dist[0][0]
        return dist[0][1], gap

    def refine(field, r0, c0, span=3):
        """Move the aperture around the stereo guess to MAXIMIZE the identity
        margin (the screen seeks the lock-in). Returns (pred, gap, r, c)."""
        best = (-1.0, None, r0, c0)
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                win, rr, cc = crop(field, r0 + dr, c0 + dc)
                pred, gap = classify(descriptor(win))
                if gap > best[0]:
                    best = (gap, pred, rr, cc)
        return best[1], best[0], best[2], best[3]

    # ── bench: random digit at random (r,c) ─────────────────────────────
    gte = torch.Generator().manual_seed(99)
    rng = np.random.default_rng(7)
    N = 200
    id2, id1, rerr, cerr, examples = [], [], [], [], []
    for n in range(N):
        ch = digits[n % 10]
        r = int(rng.integers(0, H - RH)); c = int(rng.integers(0, W - RW))
        field = place(variant(ch, gte), r, c, gte)
        r_cen, c_cen = where_2d(field)
        # stereo coarse guess, then refine the aperture to maximize lock-in
        pred2, gap, rh, cw = refine(field, int(round(r_cen - RH / 2)),
                                    int(round(c_cen - RW / 2)))
        # ablation: column-only where, row GUESSED at field center
        win1, _, _ = crop(field, (H - RH) // 2, c_cen - RW // 2)
        pred1, _ = classify(descriptor(win1))
        id2.append(int(pred2 == ch)); id1.append(int(pred1 == ch))
        rerr.append(abs(rh - r)); cerr.append(abs(cw - c))
        if len(examples) < 4:
            examples.append((field, r, c, rh, cw, ch, pred2))

    print(f"N={N}  10-digit, field {H}x{W}, off-center 2D")
    print(f"  identity  2-pair(row+col) = {np.mean(id2):.3f}   "
          f"col-only(row guessed) = {np.mean(id1):.3f}   (chance 0.10)")
    print(f"  position MAE  row = {np.mean(rerr):.2f}  col = {np.mean(cerr):.2f}")

    # ── ambiguity: 2 -> Z morph, watch the margin collapse ──────────────
    gm = torch.Generator().manual_seed(21)
    alphas = np.linspace(0, 1, 11)
    gaps, preds = [], []
    for a in alphas:
        gs = [classify(descriptor(variant("2", gm, morph_to="Z", alpha=float(a))))
              for _ in range(30)]
        gaps.append(np.mean([g for _, g in gs]))
        preds.append(max(set(p for p, _ in gs), key=[p for p, _ in gs].count))
    # clean-digit margin baseline for the histogram split
    clean_gaps = [classify(descriptor(variant(digits[k % 10], gm)))[1]
                  for k in range(120)]
    morph_gaps = [classify(descriptor(variant("2", gm, morph_to="Z", alpha=1.0)))[1]
                  for _ in range(120)]
    print(f"  2->Z morph: margin alpha0={gaps[0]:.3f} -> alpha1={gaps[-1]:.3f}  "
          f"(label {preds[0]} -> {preds[-1]})")

    # ── figures ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for k, (field, r, c, rh, cw, ch, pred) in enumerate(examples):
        a = ax.flat[k]
        a.imshow(field.numpy(), cmap="magma", aspect="auto")
        a.add_patch(plt.Rectangle((cw - .5, rh - .5), RW, RH, fill=False,
                    edgecolor="cyan", lw=2))
        a.set_title(f"true '{ch}'@({r},{c})  read '{pred}'@({rh},{cw})  "
                    f"{'OK' if pred==ch else 'MISS'}", fontsize=9)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"WHAT+WHERE @scale: 10 digits, 2D off-center  "
                 f"(id {np.mean(id2):.2f}, pos MAE r{np.mean(rerr):.1f}/c{np.mean(cerr):.1f})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("outputs/digit_bench_2d.png", dpi=140)

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.4))
    ax2[0].plot(alphas, gaps, "purple", marker="o")
    ax2[0].set_title("ambiguity: 2->Z morph collapses the centroid margin", fontsize=10)
    ax2[0].set_xlabel("morph alpha (0='2'  1='Z')"); ax2[0].set_ylabel("margin (gap to 2nd centroid)")
    for x, p in zip(alphas, preds):
        ax2[0].annotate(p, (x, gaps[int(round(x * 10))]), fontsize=7,
                        textcoords="offset points", xytext=(0, 6))
    ax2[0].grid(alpha=0.3)
    ax2[1].hist(clean_gaps, bins=20, alpha=0.6, color="green", label="clean digits")
    ax2[1].hist(morph_gaps, bins=20, alpha=0.6, color="red", label="2->Z (alpha=1)")
    ax2[1].set_title("margin histogram: clean vs ambiguous (refusal threshold)", fontsize=10)
    ax2[1].set_xlabel("margin"); ax2[1].legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig("outputs/digit_bench_2d_ambiguity.png", dpi=140)
    print("wrote outputs/digit_bench_2d.png  outputs/digit_bench_2d_ambiguity.png")


if __name__ == "__main__":
    main()
