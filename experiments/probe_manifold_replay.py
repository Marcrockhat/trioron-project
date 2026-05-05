"""Manifold Replay probe — trioron-native pseudo-rehearsal.

Stores per-class diagonal Gaussian (μ_c, σ_c) at L0 output (~30 KB total
for chained-15) and samples K=64 fresh codes per replay step. No per-
sample storage; codes are drawn on demand from the consolidated L0
distribution.

Headline against real K=50 hippo (n=3 means, from
outputs/bench_chained_15task_n3_HIPPO_K50_FIXED_multiseed.csv):

  fixed_ewc_small        full=0.624  task-aware=0.957
  grown_capped_no_dream  full=0.618  task-aware=0.958
  grown_capped_dream     full=0.612  task-aware=0.962
  grown_uncapped_dream   full=0.646  task-aware=0.966   <-- target

Storage budget comparison:
  hippo K=50            768 KB   (per-sample codes)
  manifold (this probe)  30 KB   (per-class μ + σ only)

Run:
  python3 -m experiments.probe_manifold_replay \
      > outputs/probe_manifold_replay.log 2>&1
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


def main() -> int:
    argv = [
        "--seed", "0",
        "--arms", "grown_uncapped_dream",
        "--csv", "probe_manifold_replay.csv",
    ]
    print("Manifold Replay — trioron-native pseudo-rehearsal")
    print(f"  MANIFOLD_REPLAY_ENABLED      = {bench.MANIFOLD_REPLAY_ENABLED}")
    print(f"  MANIFOLD_REPLAY_BATCH        = {bench.MANIFOLD_REPLAY_BATCH}")
    print(f"  MANIFOLD_REPLAY_LOSS_WEIGHT  = {bench.MANIFOLD_REPLAY_LOSS_WEIGHT}")
    print(f"  MANIFOLD_NOISE_SCALE         = {bench.MANIFOLD_NOISE_SCALE}")
    print(f"  MANIFOLD_MAX_SAMPLES_PER_CLASS = {bench.MANIFOLD_MAX_SAMPLES_PER_CLASS}")
    print()
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
