"""Perspective stereo: revise the 2nd emitter from a parallel plane wave to a
curved (Fresnel) wavefront so the two projections differ like two viewpoints
(s042, Rocky's revision).

Failure being fixed (where_decompose.py): the dual-frequency stereo localizes a
POINT well but an EXTENDED, shape-varying digit poorly on the cramped row axis
(MAE 1.21). Reason (math): both emitters are plane waves theta=w*idx, so the
lock-in phase is  w*p + angle(S(w))  where S(w) is the object's shape transform;
the angle(S(w)) term is a class-dependent bias that BOTH parallel emitters carry,
so a 2nd plane wave cannot triangulate it out.

Revision: make the 2nd emitter a NEAR-FIELD curved wavefront

    theta_B(idx) = w_B*idx + kappa*(idx - idx_c)^2 + off

whose local frequency dtheta/didx = 2*kappa*(idx-idx_c)+w_B fans out with
position (perspective rays from a point source). It samples the object's EXTENT,
not just its centroid -> a genuinely different projection. Compared here:

  plane-pair       : current (two plane waves, periods PR)
  plane+curved     : emitter A plane, emitter B curved (Rocky's literal revision)
  opp-curvature    : A=+kappa, B=-kappa toed-in pair (stereo triangulation)

Metric: ROW localization MAE of placed 0-9 digits (the axis that fails), plus the
identity you get by cropping at the estimated row (oracle row = the WHAT ceiling).

Run:  python3 experiments/progenitor/stereo_perspective.py
"""
from __future__ import annotations

import math
import numpy as np
import torch

from experiments.progenitor import digit_bench_2d as DB

IC = (DB.H - 1) / 2.0                         # field row center (focal point)
WR = 2 * math.pi / 11.0                       # base row carrier
WR2 = 2 * math.pi / 9.0                       # 2nd plane-pair carrier
OFFB = DB.OFFB


def _cdist(a, b):
    return np.abs(np.angle(np.exp(1j * (a - b))))


def est_plane(weight, grid):
    """Current dual-frequency plane-pair Vernier (two parallel projections)."""
    idx = np.arange(len(weight))
    za = (weight * np.exp(1j * WR * idx)).sum()
    zb = (weight * np.exp(1j * (WR2 * idx + OFFB))).sum()
    cost = _cdist(WR * grid, np.angle(za)) + _cdist(WR2 * grid + OFFB, np.angle(zb))
    return grid[np.argmin(cost)]


def est_plane_curved(weight, grid, kappa):
    """Emitter A plane (WR); emitter B curved (Fresnel) — Rocky's revision."""
    idx = np.arange(len(weight))
    za = (weight * np.exp(1j * WR * idx)).sum()
    zb = (weight * np.exp(1j * (WR2 * idx + kappa * (idx - IC) ** 2 + OFFB))).sum()
    cost = (_cdist(WR * grid, np.angle(za))
            + _cdist(WR2 * grid + kappa * (grid - IC) ** 2 + OFFB, np.angle(zb)))
    return grid[np.argmin(cost)]


def est_opp_curved(weight, grid, kappa):
    """Toed-in pair: same carrier, opposite curvature (+/-kappa) -> disparity."""
    idx = np.arange(len(weight))
    za = (weight * np.exp(1j * (WR * idx + kappa * (idx - IC) ** 2))).sum()
    zb = (weight * np.exp(1j * (WR * idx - kappa * (idx - IC) ** 2 + OFFB))).sum()
    cost = (_cdist(WR * grid + kappa * (grid - IC) ** 2, np.angle(za))
            + _cdist(WR * grid - kappa * (grid - IC) ** 2 + OFFB, np.angle(zb)))
    return grid[np.argmin(cost)]


def main():
    digits = [str(i) for i in range(10)]
    gtr = torch.Generator().manual_seed(3)
    cent = {ch: np.mean([DB.descriptor(DB.variant(ch, gtr)) for _ in range(60)], 0)
            for ch in digits}

    def classify(d):
        return min(digits, key=lambda ch: np.linalg.norm(d - cent[ch]))

    gte = torch.Generator().manual_seed(99)
    rng = np.random.default_rng(7)
    grow = np.linspace(0, DB.H - DB.RH, 1000)        # top-left row grid
    N = 300

    kappas = [0.04, 0.08, 0.15, 0.25]
    # accumulate row errors per method
    err = {"plane": [], "oracle_id": [], "plane_id": []}
    for k in kappas:
        err[f"curved@{k}"] = []
        err[f"opp@{k}"] = []
        err[f"curved_id@{k}"] = []
        err[f"opp_id@{k}"] = []

    for n in range(N):
        ch = digits[n % 10]
        r = int(rng.integers(0, DB.H - DB.RH)); c = int(rng.integers(0, DB.W - DB.RW))
        field = DB.place(DB.variant(ch, gte), r, c, gte)
        img = field.numpy()
        wrow = img.sum(1)                            # row marginal (the where signal)

        # true column from stereo col axis (clean) so identity isolates ROW error
        r_cen_plane = est_plane(wrow, grow)
        err["plane"].append(abs(r_cen_plane - r))

        # identity readouts: oracle row vs plane-row
        win_o, _, _ = DB.crop(field, r, c)
        err["oracle_id"].append(int(classify(DB.descriptor(win_o)) == ch))
        win_p, _, _ = DB.crop(field, r_cen_plane, c)
        err["plane_id"].append(int(classify(DB.descriptor(win_p)) == ch))

        for k in kappas:
            rc = est_plane_curved(wrow, grow, k)
            ro = est_opp_curved(wrow, grow, k)
            err[f"curved@{k}"].append(abs(rc - r))
            err[f"opp@{k}"].append(abs(ro - r))
            wc, _, _ = DB.crop(field, rc, c)
            wo, _, _ = DB.crop(field, ro, c)
            err[f"curved_id@{k}"].append(int(classify(DB.descriptor(wc)) == ch))
            err[f"opp_id@{k}"].append(int(classify(DB.descriptor(wo)) == ch))

    print(f"N={N}  field {DB.H}x{DB.W}, ROW localization of placed 0-9 digits")
    print(f"  (column taken from stereo so identity isolates ROW error)\n")
    print(f"  oracle-row identity (WHAT ceiling) = {np.mean(err['oracle_id']):.3f}")
    print(f"  {'method':<18} {'row MAE':>9} {'identity':>9}")
    print(f"  {'plane-pair (current)':<18} {np.mean(err['plane']):>9.2f} "
          f"{np.mean(err['plane_id']):>9.3f}")
    for k in kappas:
        print(f"  {'plane+curved k='+str(k):<18} {np.mean(err[f'curved@{k}']):>9.2f} "
              f"{np.mean(err[f'curved_id@{k}']):>9.3f}")
    for k in kappas:
        print(f"  {'opp-curv  k='+str(k):<18} {np.mean(err[f'opp@{k}']):>9.2f} "
              f"{np.mean(err[f'opp_id@{k}']):>9.3f}")


if __name__ == "__main__":
    main()
