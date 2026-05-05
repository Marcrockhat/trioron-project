"""Single-seed sanity probe for the PackNet port to chained-15.

Runs both packnet_matched and packnet_standard arms at seed 0 to verify
the wiring (begin_task, freeze_grads, end_task, per-task inference mask)
runs end-to-end without errors and produces sensible numbers.

Run:
  python3 -m experiments.probe_packnet \
      > outputs/probe_packnet.log 2>&1
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import bench_chained_15task as bench

# All rehearsal mechanisms OFF — PackNet is the comparison baseline.
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
        "--seed", "0",
        "--arms", "packnet_matched,packnet_standard",
        "--csv", "probe_packnet.csv",
    ]
    print("PackNet single-seed sanity probe (chained-15)")
    print()
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
