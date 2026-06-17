"""Depth-from-focus: the curved (Fresnel) emitter recovers the depth a plane
wave is blind to (s042, Rocky's "perspective needs 3D / tiles need depth").

Why the 2D perspective test (stereo_perspective.py) was ~null: a flat digit is
one plane, so a curved wavefront only reparametrizes it -> no parallax. Depth is
a real axis only when the data HAS depth. And the plane wave cannot see it at
all: translating an object in z does not shift its 2D projection, so a linear
phase ramp theta=w*idx carries zero depth information.

A curved wavefront does. Model each lens tile as a point scatterer at depth z,
read by an aperture of A samples. An emitter focusing at depth f imposes a
DEFOCUS phase across the aperture:

    phi(a) = mu * (1/z - 1/f) * a^2          (thin-lens quadratic / Fresnel)

The aperture lock-in  Z(f) = mean_a exp(i*phi(a))  has |Z| = 1 only when f = z
(in focus) and decays as defocus grows. Sweep f, take argmax|Z(f)| -> z. A plane
wave is mu=0: |Z| == 1 for every f -> depth-blind.

"Each tile has its own depth/thickness": a relief template gives every tile a z;
sweeping focus recovers the whole depth MAP, and two reliefs are separable by it
(depth as a WHAT channel). Reported: depth MAE (curved focal sweep vs plane
wave), and identity from the recovered depth map.

Run:  python3 experiments/progenitor/lens_depth_focus.py
"""
from __future__ import annotations

import math
import numpy as np

# ── geometry ────────────────────────────────────────────────────────────
GH, GW = 5, 5                      # tile grid
APER = np.arange(-5, 6)            # aperture samples (a)
MU = 0.30 * math.pi               # defocus strength (mu*a^2*Delta(1/z))
ZRANGE = (1.0, 2.0)               # true depth range
FGRID = np.linspace(ZRANGE[0], ZRANGE[1], 81)   # focal-depth sweep

# two depth reliefs (classes): a slanted ramp vs a central bump
def relief(kind):
    r = np.linspace(0, 1, GH)[:, None] * np.ones((1, GW))
    c = np.linspace(0, 1, GW)[None, :] * np.ones((GH, 1))
    if kind == "ramp":
        z = r                                   # depth increases down the grid
    else:  # "bump"
        z = np.exp(-(( (r-0.5)**2 + (c-0.5)**2) / 0.08))   # central mound
    z = (z - z.min()) / (z.max() - z.min() + 1e-9)
    return ZRANGE[0] + (ZRANGE[1] - ZRANGE[0]) * z


def aperture_lockin(z_true, f, rng, mu):
    """Coherence |Z(f)| of one tile at depth z_true, focus f, with phase noise."""
    defocus = mu * (1.0 / z_true - 1.0 / f) * (APER ** 2)
    noise = 0.25 * rng.standard_normal(len(APER))      # per-sample phase jitter
    z = np.exp(1j * (defocus + noise)).mean()
    return abs(z)


def recover_depth(zmap, rng, mu):
    """Per-tile depth = focal sweep argmax coherence. mu=0 -> plane wave."""
    out = np.zeros_like(zmap)
    for i in range(GH):
        for j in range(GW):
            coh = [aperture_lockin(zmap[i, j], f, rng, mu) for f in FGRID]
            out[i, j] = FGRID[int(np.argmax(coh))]
    return out


def main():
    rng = np.random.default_rng(0)
    reliefs = {"ramp": relief("ramp"), "bump": relief("bump")}

    print(f"grid {GH}x{GW}, aperture {len(APER)}, depth range {ZRANGE}, "
          f"focal sweep {len(FGRID)} steps\n")

    # ── depth MAE: curved focal sweep vs plane wave ─────────────────────
    for name, zmap in reliefs.items():
        e_curved, e_plane = [], []
        for _ in range(20):                                  # repeats over noise
            rc = recover_depth(zmap, rng, MU)
            rp = recover_depth(zmap, rng, 0.0)               # plane wave (mu=0)
            e_curved.append(np.abs(rc - zmap).mean())
            e_plane.append(np.abs(rp - zmap).mean())
        rng_span = ZRANGE[1] - ZRANGE[0]
        print(f"  {name:<5} depth MAE  curved={np.mean(e_curved):.3f}  "
              f"plane={np.mean(e_plane):.3f}  (range {rng_span:.2f}, "
              f"chance~{rng_span/3:.2f})")

    # ── identity from the recovered depth map ───────────────────────────
    # build per-class depth-map centroids, then classify noisy recoveries
    cent = {}
    for name, zmap in reliefs.items():
        maps = [recover_depth(zmap, rng, MU).ravel() for _ in range(15)]
        cent[name] = np.mean(maps, 0)
    correct = 0; total = 0
    for name, zmap in reliefs.items():
        for _ in range(30):
            d = recover_depth(zmap, rng, MU).ravel()
            pred = min(cent, key=lambda k: np.linalg.norm(d - cent[k]))
            correct += int(pred == name); total += 1
    print(f"\n  identity from recovered depth map (curved) = {correct/total:.3f}  "
          f"(2 classes, chance 0.50)")

    # plane-wave identity (depth-blind -> should be chance)
    centp = {}
    for name, zmap in reliefs.items():
        centp[name] = np.mean([recover_depth(zmap, rng, 0.0).ravel()
                               for _ in range(15)], 0)
    cp = 0; tp = 0
    for name, zmap in reliefs.items():
        for _ in range(30):
            d = recover_depth(zmap, rng, 0.0).ravel()
            pred = min(centp, key=lambda k: np.linalg.norm(d - centp[k]))
            cp += int(pred == name); tp += 1
    print(f"  identity from recovered depth map (plane)  = {cp/tp:.3f}  "
          f"(depth-blind baseline)")


if __name__ == "__main__":
    main()
