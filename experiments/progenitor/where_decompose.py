"""Decompose the 10-digit 2D bench's 0.645 identity into WHAT vs WHERE
(s042 diagnostic). The spectral-lens result shows the registered WHAT channel
is ~0.99; this asks how much of the bench loss is localization by measuring
identity at three crops:

  oracle : the TRUE (r,c) the digit was placed at        -> WHAT ceiling
  stereo : the dual-frequency where_2d() coarse guess    -> + where error
  refine : aperture moved to maximize identity margin    -> + coupling fix

Run:  python3 experiments/progenitor/where_decompose.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.progenitor import digit_bench_2d as DB


def main():
    digits = [str(i) for i in range(10)]
    gtr = torch.Generator().manual_seed(3)
    cent = {ch: np.mean([DB.descriptor(DB.variant(ch, gtr)) for _ in range(60)], 0)
            for ch in digits}

    def classify(d):
        dist = sorted(((np.linalg.norm(d - cent[ch]), ch) for ch in digits))
        return dist[0][1], dist[1][0] - dist[0][0]

    def refine(field, r0, c0, span=3):
        best = (-1.0, None, r0, c0)
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                win, rr, cc = DB.crop(field, r0 + dr, c0 + dc)
                pred, gap = classify(DB.descriptor(win))
                if gap > best[0]:
                    best = (gap, pred, rr, cc)
        return best[1], best[2], best[3]

    def slide(field, c0, rspan=3):
        """Stereo trusted on the clean COL axis only; the cramped ROW axis is
        registered by sliding the aperture to maximize the lens margin (the
        screen seeks the lock-in over the full row extent)."""
        best = (-1.0, None, 0, c0)
        for rr in range(0, DB.H - DB.RH + 1):
            for dc in range(-rspan, rspan + 1):
                win, r_, c_ = DB.crop(field, rr, c0 + dc)
                pred, gap = classify(DB.descriptor(win))
                if gap > best[0]:
                    best = (gap, pred, r_, c_)
        return best[1], best[2], best[3]

    gte = torch.Generator().manual_seed(99)
    rng = np.random.default_rng(7)
    N = 200
    acc_oracle, acc_stereo, acc_refine, acc_slide = [], [], [], []
    rerr_s, cerr_s, rerr_r, cerr_r, rerr_sl = [], [], [], [], []
    for n in range(N):
        ch = digits[n % 10]
        r = int(rng.integers(0, DB.H - DB.RH)); c = int(rng.integers(0, DB.W - DB.RW))
        field = DB.place(DB.variant(ch, gte), r, c, gte)

        # oracle crop
        win_o, _, _ = DB.crop(field, r, c)
        acc_oracle.append(int(classify(DB.descriptor(win_o))[0] == ch))

        # stereo coarse guess (no refine)
        r_cen, c_cen = DB.where_2d(field)
        rs = int(round(r_cen - DB.RH / 2)); cs = int(round(c_cen - DB.RW / 2))
        win_s, rss, css = DB.crop(field, rs, cs)
        acc_stereo.append(int(classify(DB.descriptor(win_s))[0] == ch))
        rerr_s.append(abs(rss - r)); cerr_s.append(abs(css - c))

        # refine (aperture seeks max margin)
        pred_r, rr, cc = refine(field, rs, cs)
        acc_refine.append(int(pred_r == ch))
        rerr_r.append(abs(rr - r)); cerr_r.append(abs(cc - c))

        # full row-slide: col from stereo, row from lens margin over full extent
        pred_sl, rsl, _ = slide(field, cs)
        acc_slide.append(int(pred_sl == ch))
        rerr_sl.append(abs(rsl - r))

    print(f"N={N}  field {DB.H}x{DB.W}, 10 digits off-center 2D")
    print(f"  identity @ ORACLE crop (true r,c)   = {np.mean(acc_oracle):.3f}   <- WHAT ceiling")
    print(f"  identity @ STEREO guess (no refine) = {np.mean(acc_stereo):.3f}   "
          f"(MAE r={np.mean(rerr_s):.2f} c={np.mean(cerr_s):.2f})")
    print(f"  identity @ REFINE (margin-seeking)  = {np.mean(acc_refine):.3f}   "
          f"(MAE r={np.mean(rerr_r):.2f} c={np.mean(cerr_r):.2f})")
    print(f"  identity @ ROW-SLIDE (lens-driven)  = {np.mean(acc_slide):.3f}   "
          f"(MAE r={np.mean(rerr_sl):.2f})")
    print(f"  chance = 0.10")
    gap_what = np.mean(acc_oracle) - np.mean(acc_refine)
    print(f"\n  WHERE loss (oracle - refine) = {gap_what:.3f}  "
          f"({100*gap_what/max(np.mean(acc_oracle),1e-9):.0f}% of the WHAT ceiling)")


if __name__ == "__main__":
    main()
