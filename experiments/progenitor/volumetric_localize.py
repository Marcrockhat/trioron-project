"""Volumetric joint localization: recover (z, r, c) of a flat 2D digit placed at
a depth plane in a volume (s042, Rocky's "jointly localize z,r,c").

A flat object occupies ONE depth plane, so every lit tile shares the same z (the
object's plane); z is read from the AGGREGATE focus over the tiles (the "center
of the tiles"), not a per-tile map. Channels, each validated earlier this
session:

  c (col) : plane-wave dual-frequency Vernier on the column marginal   (clean axis)
  r (row) : lens-margin slide over the cramped row axis                (coupling fix)
  z (depth): aggregate aperture focal-curvature sweep over the lit tiles (NEW)

The plane wave is depth-blind (mu=0 -> flat focus), so its z is chance. The
curved focal sweep recovers the object's plane. Identity is read by the lens at
the recovered (r,c). Reported: MAE for z/r/c, identity, and the plane-wave z
baseline.

Run:  python3 experiments/progenitor/volumetric_localize.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.progenitor import digit_bench_2d as DB

# ── depth model (shared with lens_depth_focus) ──────────────────────────
APER = np.arange(-5, 6)
MU = 0.30 * math.pi
ZRANGE = (1.0, 2.0)
FGRID = np.linspace(ZRANGE[0], ZRANGE[1], 81)


def aperture_coh(z_true, f, rng, mu):
    defocus = mu * (1.0 / z_true - 1.0 / f) * (APER ** 2)
    noise = 0.25 * rng.standard_normal(len(APER))
    return abs(np.exp(1j * (defocus + noise)).mean())


def recover_z(z_true, n_lit, rng, mu):
    """Aggregate focus over n_lit tiles all at the object's plane z_true."""
    coh = np.zeros(len(FGRID))
    for t in range(n_lit):
        coh += [aperture_coh(z_true, f, rng, mu) for f in FGRID]
    return FGRID[int(np.argmax(coh))]


def main():
    digits = [str(i) for i in range(10)]
    gtr = torch.Generator().manual_seed(3)
    cent = {ch: np.mean([DB.descriptor(DB.variant(ch, gtr)) for _ in range(60)], 0)
            for ch in digits}

    def classify(d):
        dist = sorted(((np.linalg.norm(d - cent[ch]), ch) for ch in digits))
        return dist[0][1], dist[1][0] - dist[0][0]

    def row_slide(field, c0, rspan=3):
        best = (-1.0, None, 0)
        for rr in range(0, DB.H - DB.RH + 1):
            for dc in range(-rspan, rspan + 1):
                win, r_, _ = DB.crop(field, rr, c0 + dc)
                _, gap = classify(DB.descriptor(win))
                if gap > best[0]:
                    best = (gap, r_, c0 + dc)
        return best[1], best[2]

    gte = torch.Generator().manual_seed(99)
    rng = np.random.default_rng(7)
    drng = np.random.default_rng(11)
    N = 200
    ez, er, ec, ident = [], [], [], []
    ez_plane = []
    for n in range(N):
        ch = digits[n % 10]
        r = int(rng.integers(0, DB.H - DB.RH)); c = int(rng.integers(0, DB.W - DB.RW))
        z0 = float(drng.uniform(*ZRANGE))                    # object's depth plane
        field = DB.place(DB.variant(ch, gte), r, c, gte)

        # lateral: col from plane Vernier, row from lens-margin slide
        _, c_cen = DB.where_2d(field)
        cs = int(round(c_cen - DB.RW / 2))
        rh, cw = row_slide(field, cs)
        n_lit = int((DB.crop(field, rh, cw)[0] > 0.5).sum().item()) or 1

        # depth: aggregate focal sweep (curved) vs plane wave (depth-blind)
        z_hat = recover_z(z0, n_lit, drng, MU)
        z_plane = recover_z(z0, n_lit, drng, 0.0)

        pred, _ = classify(DB.descriptor(DB.crop(field, rh, cw)[0]))
        ez.append(abs(z_hat - z0)); ez_plane.append(abs(z_plane - z0))
        er.append(abs(rh - r)); ec.append(abs(cw - c)); ident.append(int(pred == ch))

    span = ZRANGE[1] - ZRANGE[0]
    print(f"N={N}  volume depth{ZRANGE} x field {DB.H}x{DB.W}, flat digit at (z,r,c)\n")
    print(f"  joint localization (curved z + plane col + slide row):")
    print(f"    z MAE = {np.mean(ez):.3f}   (range {span:.2f}, chance~{span/3:.2f})")
    print(f"    r MAE = {np.mean(er):.2f}    c MAE = {np.mean(ec):.2f}")
    print(f"    identity @ recovered (r,c) = {np.mean(ident):.3f}  (chance 0.10)")
    print(f"\n  plane-wave z baseline (depth-blind) = {np.mean(ez_plane):.3f}  "
          f"(chance~{span/3:.2f})")


if __name__ == "__main__":
    main()
