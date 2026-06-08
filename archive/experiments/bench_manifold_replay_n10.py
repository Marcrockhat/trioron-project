"""Manifold Replay multi-seed bench — n=10, all 4 chained-15 arms.

n=10 rerun for the ArXiv-ready paper draft. Same protocol as
bench_manifold_replay_n3, seeds 0..9.

Run:
  python3 -m experiments.bench_manifold_replay_n10 \
      > outputs/bench_chained_15task_n10_MANIFOLD_REPLAY.log 2>&1
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import bench_chained_15task as bench

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
        "--seeds", "0,1,2,3,4,5,6,7,8,9",
        "--arms", ",".join([
            "fixed_ewc_small",
            "grown_capped_no_dream",
            "grown_capped_dream",
            "grown_uncapped_dream",
        ]),
        "--csv", "bench_chained_15task_n10_MANIFOLD_REPLAY.csv",
    ]
    print("Manifold Replay — n=10 multi-seed, 4 arms")
    print(f"  MANIFOLD_REPLAY_ENABLED      = {bench.MANIFOLD_REPLAY_ENABLED}")
    print(f"  MANIFOLD_REPLAY_BATCH        = {bench.MANIFOLD_REPLAY_BATCH}")
    print(f"  MANIFOLD_REPLAY_LOSS_WEIGHT  = {bench.MANIFOLD_REPLAY_LOSS_WEIGHT}")
    print(f"  MANIFOLD_NOISE_SCALE         = {bench.MANIFOLD_NOISE_SCALE}")
    print()
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
