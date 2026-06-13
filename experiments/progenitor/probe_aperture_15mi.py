"""1.5 Mi aperture probe (s036, Rocky-approved) — validate that the
SAME genesis retinal-compression code scales to a screen-resolution
aperture and DELIVERS compression where redundancy exists.

The bench finding (diag_s036): native 28x28 MNIST has no lossless
adjacent redundancy (max R_diff 0.857 < floor 0.920), so the merge
correctly declines and the starve is the only compression. This probe
supplies a regime where within-block redundancy is exact: a small
image presented on a large aperture by NEAREST-neighbour upsampling in
VALUE SPACE (before phasor encoding), so each source pixel becomes a
block of byte-identical columns -> identical q -> identical phasor ->
within-block R_diff = 1.0.

IMPORTANT (Rocky, s036): the naive "~2000:1 lossless, one region per
source pixel" expectation is WRONG, because the data is fed as phasors
through the per-sample contrast frame under the MASK RULE (q in
{0, 1000} are reference, not evidence). A block whose source pixel sits
at the per-sample floor (near-black) or is the per-sample brightest
(saturates to the ceiling) accumulates almost no co-active evidence,
never reaches MIN_MEMBERS, and does NOT merge despite identical
columns. So only the source pixels carrying INTERIOR contrast evidence
merge (measured: 86 of ~434 non-constant blocks at 1.5 Mi, ratio
~2.3:1). What this probe legitimately establishes: (a) the genesis
code is dimension-generic and runs at 1.5 M columns in ~76 s; (b) the
merge is GEOMETRICALLY EXACT where it fires (each region = one
source-pixel block, member count = exact block area). Compression is
real but bounded by interior-evidence coverage, NOT by pixel count —
the same lesson as native MNIST, one scale up.

Two paths, both the real retina.py code:
  * SWEEP (end-to-end): small apertures run the FULL genesis
    (census -> starve -> merge -> spawn) and confirm the receptor
    count drops to ~784.
  * HEADLINE (merge-only): the literal 1536x1024 ("1.5 Mi") aperture
    runs the redundancy + region-finding directly on the buffer
    (the spawn/census Python loops over 1.5M columns are an impl
    detail, validated at small scale in test_retina; the CLAIM is
    that the merge math finds the regions and the ratio holds). Reports
    compression ratio, regions, members/region, wall time.

Run: python3 -m experiments.progenitor.probe_aperture_15mi
Env: PROBE_SAMPLES (default 200), PROBE_FULL15MI (default 1).
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from trioron.core import construct
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)
from trioron.pcll import PerceptionGenesis, germline_base
from trioron.pcll.retina import (REDUNDANT_R, adjacent_pairs,
                                 find_regions, pair_redundancy)

DATA_ROOT = os.environ.get(
    "AXIS6_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "outputs", "data"))
SAMPLES = int(os.environ.get("PROBE_SAMPLES", "200"))
FULL15MI = os.environ.get("PROBE_FULL15MI", "1") == "1"


def _mnist(n: int) -> torch.Tensor:
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT, n_holdout_per_dataset=0)
    view = build_task_views(bundle, chained_15_specs()[:1], split="train")[0]
    return view.images[:n]          # [n, 784]


def _upsample(x: torch.Tensor, hw) -> torch.Tensor:
    """Nearest-neighbour blow-up 28x28 -> (H, W). Each source pixel
    becomes a block of identical columns (the lossless redundancy the
    merge is designed to recover)."""
    H, W = hw
    img = x.view(-1, 1, 28, 28)
    up = F.interpolate(img, size=(H, W), mode="nearest")
    return up.view(x.shape[0], H * W)


def _eligible(buf: torch.Tensor):
    """Non-constant columns (continuous) — the merge-eligible set, the
    same predicate genesis uses (constant -> starve)."""
    nonconst = buf.amax(0) != buf.amin(0)
    return nonconst.nonzero(as_tuple=False).squeeze(1).tolist()


def sweep_full(hw):
    """End-to-end genesis at a tractable aperture: census -> starve ->
    merge -> spawn. Confirms receptors collapse to ~784."""
    H, W = hw
    x = _upsample(_mnist(SAMPLES), hw)
    t0 = time.time()
    sub = construct(germline_base, capacity=H * W + 4096)
    pg = PerceptionGenesis(sub, input_shape=hw)
    pg.feed(x)
    rep = sub.end_task()
    n_rec = int(sub.scheduler._plan.receptor_ids.numel())
    n_pool = int(sub.scheduler._plan.pooled_ids.numel())
    print(f"  {H}x{W}={H*W:>9d}  starved={len(rep.starved):>6d}  "
          f"regions={len(rep.regions):>4d}  merged_cols={len(rep.merged):>7d}  "
          f"receptors->{n_rec:>5d} ({n_pool} pooled)  "
          f"t={time.time()-t0:.1f}s")


def headline_15mi(hw):
    """Merge-only at the literal 1.5 Mi aperture: redundancy +
    region-finding on the buffer. Reports the compression ratio."""
    H, W = hw
    print(f"\n=== HEADLINE: {H}x{W} = {H*W:,} columns "
          f"(nearest-upsampled MNIST, n={SAMPLES}) ===")
    t0 = time.time()
    x = _upsample(_mnist(SAMPLES), hw)
    elig = _eligible(x)
    t_up = time.time() - t0
    pairs = adjacent_pairs(elig, hw)
    t_pairs = time.time() - t0
    r_diff, cnt = pair_redundancy([x], elig, pairs)
    t_red = time.time() - t0
    regions = find_regions(elig, pairs, r_diff, cnt)
    t_reg = time.time() - t0
    merged_cols = sum(len(r) for r in regions)
    sizes = torch.tensor([len(r) for r in regions]) if regions \
        else torch.tensor([0])
    # surviving sensors = (eligible not merged) + one per region + starved husks
    survive = (len(elig) - merged_cols) + len(regions)
    print(f"  eligible (non-constant) cols : {len(elig):,} / {H*W:,}")
    print(f"  adjacent pairs               : {len(pairs):,}")
    print(f"  derived floor REDUNDANT_R    : {REDUNDANT_R:.3f}")
    print(f"  pairs clearing floor         : "
          f"{int((r_diff >= REDUNDANT_R).sum()):,} / {len(pairs):,}")
    print(f"  regions formed               : {len(regions):,}")
    print(f"  columns merged into regions  : {merged_cols:,}")
    print(f"  members/region  min/med/max  : "
          f"{int(sizes.min())}/{int(sizes.median())}/{int(sizes.max())}")
    print(f"  surviving sensors (council sees): {survive:,}")
    print(f"  COMPRESSION RATIO            : {H*W / max(survive,1):.1f}:1")
    print(f"  wall: upsample {t_up:.1f}s  pairs {t_pairs-t_up:.1f}s  "
          f"redundancy {t_red-t_pairs:.1f}s  regions {t_reg-t_red:.1f}s  "
          f"total {time.time()-t0:.1f}s")


def main():
    print(f"1.5 Mi aperture probe — nearest-upsampled MNIST, "
          f"n={SAMPLES} samples\n")
    print("=== SWEEP (end-to-end genesis, receptors should collapse "
          "to ~784) ===")
    for hw in [(56, 56), (112, 112), (224, 224)]:
        sweep_full(hw)
    if FULL15MI:
        headline_15mi((1536, 1024))


if __name__ == "__main__":
    main()
