"""Manifold Replay ablation — noise-scale sweep on grown_uncapped_dream.

Single-seed sweep that probes which moment of the per-class L0 Gaussian
is load-bearing:

  noise_scale = 0.0  →  μ-only (rehearse the per-class L0 mean as a
                         deterministic 'prototype'; no variance)
  noise_scale = 0.5  →  half variance
  noise_scale = 1.0  →  full diagonal Gaussian (default, already have
                         this from probe_manifold_replay)
  noise_scale = 1.5  →  over-variance

If μ-only matches noise_scale=1.0, the variance moment isn't load-
bearing. If μ-only collapses, σ is essential — and the per-class
covariance structure is what makes manifold replay work.

Run:
  python3 -m experiments.bench_manifold_ablation \
      > outputs/bench_manifold_ablation.log 2>&1
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import bench_chained_15task as bench

# Manifold replay only — all other rehearsal mechanisms OFF.
bench.MANIFOLD_REPLAY_ENABLED = True
bench.HIPPOCAMPAL_ENABLED = False
bench.HIPPOCAMPAL_SYNTHETIC = False
bench.REHEARSAL_ENABLED = False
bench.LWF_ENABLED = False
bench.BRAINSTEM_ENABLED = False
bench.ENGRAM_ENABLED = False
bench.DIFFERENTIAL_ENABLED = False


SWEEP = [
    ("0.0_mu_only",  0.0),
    ("0.5_half",     0.5),
    ("1.5_over",     1.5),
    # noise_scale=1.0 is the default and already covered by
    # probe_manifold_replay / bench_manifold_replay_n3 — skipping here.
]


def main() -> int:
    print("Manifold Replay — noise-scale ablation (single seed, "
          "grown_uncapped_dream)")
    print()
    rc = 0
    for tag, ns in SWEEP:
        print("=" * 78)
        print(f"NOISE_SCALE = {ns}  (tag: {tag})")
        print("=" * 78)
        bench.MANIFOLD_NOISE_SCALE = ns
        argv = [
            "--seed", "0",
            "--arms", "grown_uncapped_dream",
            "--csv", f"bench_manifold_ablation_ns{tag}.csv",
        ]
        rc = bench.main(argv) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
