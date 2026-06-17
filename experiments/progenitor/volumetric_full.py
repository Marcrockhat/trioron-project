"""Unified volumetric localizer: field-conv register (x,y) + Fresnel focal sweep
(z), the two solved s042 channels combined (s042 continuation).

volumetric_localize.py predates step 2 — it localizes lateral with the old
where_2d + row_slide (identity 0.675). This swaps in the field-wide tied-lens
conv response map (field_conv_register: register at min-centroid-DISTANCE over
every (r,c) -> identity 0.975) and keeps the depth-from-focus channel for z.
Expected: lateral identity ~oracle AND z near-lossless, jointly.

Run:  python3 experiments/progenitor/volumetric_full.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.progenitor import digit_bench_2d as DB

# depth model (shared with lens_depth_focus / volumetric_localize)
APER = np.arange(-5, 6)
MU = 0.30 * math.pi
ZRANGE = (1.0, 2.0)
FGRID = np.linspace(ZRANGE[0], ZRANGE[1], 81)


def aperture_coh(z_true, f, rng, mu):
    defocus = mu * (1.0 / z_true - 1.0 / f) * (APER ** 2)
    noise = 0.25 * rng.standard_normal(len(APER))
    return abs(np.exp(1j * (defocus + noise)).mean())


def recover_z(z_true, n_lit, rng, mu):
    coh = np.zeros(len(FGRID))
    for _ in range(n_lit):
        coh += [aperture_coh(z_true, f, rng, mu) for f in FGRID]
    return FGRID[int(np.argmax(coh))]


def main():
    digits = [str(i) for i in range(10)]
    gtr = torch.Generator().manual_seed(3)
    cent = {ch: np.mean([DB.descriptor(DB.variant(ch, gtr)) for _ in range(60)], 0)
            for ch in digits}
    C = np.stack([cent[ch] for ch in digits])

    rows = list(range(0, DB.H - DB.RH + 1))
    cols = list(range(0, DB.W - DB.RW + 1))

    def register(field):
        """Field-wide conv: min nearest-centroid DISTANCE over every (r,c)."""
        best = (1e9, 0, rows[0], cols[0])
        for r in rows:
            for c in cols:
                d = DB.descriptor(field[r:r + DB.RH, c:c + DB.RW])
                dist = np.linalg.norm(C - d[None, :], axis=1)
                k = int(np.argmin(dist))
                if dist[k] < best[0]:
                    best = (dist[k], k, r, c)
        return best[1], best[2], best[3]

    gte = torch.Generator().manual_seed(99)
    rng = np.random.default_rng(7)
    drng = np.random.default_rng(11)
    N = 150
    ez, er, ec, ident = [], [], [], []
    for n in range(N):
        ch = digits[n % 10]
        r = int(rng.integers(0, DB.H - DB.RH)); c = int(rng.integers(0, DB.W - DB.RW))
        z0 = float(drng.uniform(*ZRANGE))
        field = DB.place(DB.variant(ch, gte), r, c, gte)

        k, rh, cw = register(field)                       # lateral: conv response map
        n_lit = int((field[rh:rh + DB.RH, cw:cw + DB.RW] > 0.5).sum().item()) or 1
        z_hat = recover_z(z0, n_lit, drng, MU)            # depth: Fresnel focal sweep

        ident.append(int(digits[k] == ch))
        ez.append(abs(z_hat - z0)); er.append(abs(rh - r)); ec.append(abs(cw - c))

    span = ZRANGE[1] - ZRANGE[0]
    print(f"N={N}  volume depth{ZRANGE} x field {DB.H}x{DB.W}, flat digit at (z,r,c)\n")
    print(f"  unified (conv-register x,y  +  Fresnel focus z):")
    print(f"    identity = {np.mean(ident):.3f}   (chance 0.10; oracle 0.980)")
    print(f"    z MAE = {np.mean(ez):.3f}   (range {span:.2f}, chance~{span/3:.2f})")
    print(f"    r MAE = {np.mean(er):.2f}    c MAE = {np.mean(ec):.2f}")
    print(f"\n  (cf. volumetric_localize.py, old lateral: identity 0.675, "
          f"r/c MAE ~0.5)")


if __name__ == "__main__":
    main()
