"""HAT n=10 multi-seed bench on chained-15.

n=10 rerun for the ArXiv-ready paper draft.

Run:
  python3 -m experiments.bench_hat_chained_15_n10 \
      > outputs/bench_chained_15task_n10_HAT.log 2>&1
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import bench_chained_15task as bench

bench.MANIFOLD_REPLAY_ENABLED = False
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
        "--arms", "hat_matched,hat_standard",
        "--csv", "bench_chained_15task_n10_HAT.csv",
    ]
    print("HAT n=10 multi-seed (chained-15)")
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
