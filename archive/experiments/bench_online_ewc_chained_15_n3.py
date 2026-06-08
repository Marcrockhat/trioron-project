"""Online EWC n=3 multi-seed bench on chained-15.

Schwarz et al. 2018 — Fisher accumulates across tasks with decay γ
instead of per-task reset. Same shape as fixed_ewc_small (matched-
trainable, frozen L0); only the consolidation step differs.

Run:
  python3 -m experiments.bench_online_ewc_chained_15_n3 \
      > outputs/bench_chained_15task_n3_ONLINE_EWC.log 2>&1
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
        "--arms", "online_ewc",
        "--csv", "bench_chained_15task_n3_ONLINE_EWC.csv",
    ]
    print("Online EWC n=3 multi-seed (chained-15)")
    print(f"  online_ewc_gamma = {bench.ARM_DEFINITIONS['online_ewc']['online_ewc_gamma']}")
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
