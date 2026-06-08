"""PackNet n=3 multi-seed bench on chained-15.

Confirms the single-seed result (packnet_matched task=0.908 / full=0.047,
packnet_standard task=0.844 / full=0.035) at σ-confidence and produces
the head-to-head row against manifold-equipped grown_uncapped_dream
(0.604±0.003 / 0.960±0.001 from bench_chained_15task_n3_MANIFOLD_REPLAY).

Run:
  python3 -m experiments.bench_packnet_chained_15_n3 \
      > outputs/bench_chained_15task_n3_PACKNET.log 2>&1
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
        "--arms", "packnet_matched,packnet_standard",
        "--csv", "bench_chained_15task_n3_PACKNET.csv",
    ]
    print("PackNet n=3 multi-seed (chained-15)")
    print()
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
