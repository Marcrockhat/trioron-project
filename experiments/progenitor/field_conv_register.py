"""Field-wide tied-lens conv response map: read identity AND position jointly
from the peak, retiring the brittle stereo row guess (s042 step 2).

The bench's standing cap (identity 0.675) is lateral ROW registration, not WHAT
(oracle-crop identity is 0.980) and not depth. The stereo row guess is biased by
the extended object's ink shape (where_decompose: row MAE 1.21). §8 fix: a
convolution is one SHARED filter swept over every position. So sweep the lens
over EVERY (r,c) crop -> a response map; the lens's own confidence (margin to
the nearest class centroid) peaks at the true registration. argmax over the map
gives position; the class there gives identity. No stereo, no separate where.

Compared:
  stereo+refine : current (stereo guess + -+3 margin refine)         baseline 0.645
  field-margin  : argmax over the whole field of the centroid MARGIN (gap)
  field-dist    : argmin over the whole field of nearest-centroid DISTANCE
  oracle        : identity at the true (r,c)                          ceiling 0.980

Run:  python3 experiments/progenitor/field_conv_register.py
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.progenitor import digit_bench_2d as DB


def main():
    digits = [str(i) for i in range(10)]
    gtr = torch.Generator().manual_seed(3)
    cent = {ch: np.mean([DB.descriptor(DB.variant(ch, gtr)) for _ in range(60)], 0)
            for ch in digits}
    C = np.stack([cent[ch] for ch in digits])           # [10, D]

    def scores(d):
        """nearest class, its margin (gap to 2nd), and nearest distance."""
        dist = np.linalg.norm(C - d[None, :], axis=1)
        order = np.argsort(dist)
        return order[0], dist[order[1]] - dist[order[0]], dist[order[0]]

    rows = list(range(0, DB.H - DB.RH + 1))
    cols = list(range(0, DB.W - DB.RW + 1))

    def response_map(field):
        """Sweep the shared lens over every (r,c); return margin & dist maps."""
        marg = np.full((len(rows), len(cols)), -1.0)
        dst = np.full((len(rows), len(cols)), 1e9)
        cls = np.zeros((len(rows), len(cols)), dtype=int)
        for ir, r in enumerate(rows):
            for ic, c in enumerate(cols):
                win = field[r:r + DB.RH, c:c + DB.RW]
                k, g, dd = scores(DB.descriptor(win))
                marg[ir, ic] = g; dst[ir, ic] = dd; cls[ir, ic] = k
        return marg, dst, cls

    def refine(field, r0, c0, span=3):
        best = (-1.0, None, r0, c0)
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                win, rr, cc = DB.crop(field, r0 + dr, c0 + dc)
                k, g, _ = scores(DB.descriptor(win))
                if g > best[0]:
                    best = (g, k, rr, cc)
        return best[1], best[2], best[3]

    gte = torch.Generator().manual_seed(99)
    rng = np.random.default_rng(7)
    N = 200
    id_oracle, id_refine, id_fmarg, id_fdist = [], [], [], []
    re_marg, ce_marg, re_dist, ce_dist = [], [], [], []
    for n in range(N):
        ch = digits[n % 10]
        r = int(rng.integers(0, DB.H - DB.RH)); c = int(rng.integers(0, DB.W - DB.RW))
        field = DB.place(DB.variant(ch, gte), r, c, gte)

        # oracle
        k_o, _, _ = scores(DB.descriptor(DB.crop(field, r, c)[0]))
        id_oracle.append(int(digits[k_o] == ch))

        # stereo + refine (current)
        r_cen, c_cen = DB.where_2d(field)
        k_r, rr, cc = refine(field, int(round(r_cen - DB.RH / 2)),
                             int(round(c_cen - DB.RW / 2)))
        id_refine.append(int(digits[k_r] == ch))

        # field-wide response map
        marg, dst, cls = response_map(field)
        im, jm = np.unravel_index(np.argmax(marg), marg.shape)
        idd, jd = np.unravel_index(np.argmin(dst), dst.shape)
        id_fmarg.append(int(digits[cls[im, jm]] == ch))
        id_fdist.append(int(digits[cls[idd, jd]] == ch))
        re_marg.append(abs(rows[im] - r)); ce_marg.append(abs(cols[jm] - c))
        re_dist.append(abs(rows[idd] - r)); ce_dist.append(abs(cols[jd] - c))

    print(f"N={N}  field {DB.H}x{DB.W}, 10 digits off-center 2D\n")
    print(f"  {'method':<22} {'identity':>9} {'row MAE':>8} {'col MAE':>8}")
    print(f"  {'oracle (true r,c)':<22} {np.mean(id_oracle):>9.3f} {'-':>8} {'-':>8}")
    print(f"  {'stereo+refine (cur)':<22} {np.mean(id_refine):>9.3f} {'-':>8} {'-':>8}")
    print(f"  {'field-margin (argmax)':<22} {np.mean(id_fmarg):>9.3f} "
          f"{np.mean(re_marg):>8.2f} {np.mean(ce_marg):>8.2f}")
    print(f"  {'field-dist (argmin)':<22} {np.mean(id_fdist):>9.3f} "
          f"{np.mean(re_dist):>8.2f} {np.mean(ce_dist):>8.2f}")


if __name__ == "__main__":
    main()
