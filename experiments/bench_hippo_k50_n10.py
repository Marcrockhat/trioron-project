"""Hippocampal K=50 n=10 multi-seed bench on chained-15.

n=10 rerun for the ArXiv-ready paper draft. Same configuration as the
historical bench_chained_15task_n3_HIPPO_K50_FIXED run with seeds 0..9.

Run:
  python3 -m experiments.bench_hippo_k50_n10 \
      > outputs/bench_chained_15task_n10_HIPPO_K50.log 2>&1
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import bench_chained_15task as bench

bench.MANIFOLD_REPLAY_ENABLED = False
bench.HIPPOCAMPAL_ENABLED = True
bench.HIPPOCAMPAL_K_PER_CLASS = 50
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
        "--csv", "bench_chained_15task_n10_HIPPO_K50.csv",
    ]
    print("Hippocampal K=50 n=10 multi-seed (chained-15)")
    print(f"  HIPPOCAMPAL_ENABLED  = {bench.HIPPOCAMPAL_ENABLED}")
    print(f"  K_PER_CLASS          = {bench.HIPPOCAMPAL_K_PER_CLASS}")
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
