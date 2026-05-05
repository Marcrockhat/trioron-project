"""HAT n=3 multi-seed bench on chained-15.

Confirms single-seed (hat_matched task=0.939, hat_standard task=0.886)
at σ-confidence and produces the head-to-head row vs manifold-equipped
grown_uncapped_dream (0.604±0.003 / 0.960±0.001).

Run:
  python3 -m experiments.bench_hat_chained_15_n3 \
      > outputs/bench_chained_15task_n3_HAT.log 2>&1
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
        "--seeds", "0,1,2",
        "--arms", "hat_matched,hat_standard",
        "--csv", "bench_chained_15task_n3_HAT.csv",
    ]
    print("HAT n=3 multi-seed (chained-15)")
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
