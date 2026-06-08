"""Manifold Replay + Dream-archive (Phase 1+2) — n=3, all 4 chained-15 arms.

Phase 1 marks stable rows as developmentally closed (snaps W/b to anchor,
drops Fisher contribution, masks grads). Phase 2 snaps archived rows from
FP32 to int8 at end-of-curriculum, re-evaluates, reports deployment KB.

Compare against bench_chained_15task_n3_MANIFOLD_REPLAY.csv (same panel,
no archive). Goal: confirm accuracy preserved within noise + quantify the
deployment-KB win.

Run:
  python3 -m experiments.bench_archive_n3 \
      > outputs/bench_chained_15task_n3_ARCHIVE_INT8.log 2>&1
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

bench.ARCHIVE_ENABLED = True
bench.QUANTIZE_ARCHIVED_AT_END = True
bench.QUANTIZE_MODE = "int8"


def main() -> int:
    argv = [
        "--seeds", "0,1,2",
        "--arms", ",".join([
            "fixed_ewc_small",
            "grown_capped_no_dream",
            "grown_capped_dream",
            "grown_uncapped_dream",
        ]),
        "--csv", "bench_chained_15task_n3_ARCHIVE_INT8.csv",
    ]
    print("Manifold + Dream-archive (Phase 1 + Phase 2 int8) — n=3, 4 arms")
    print(f"  ARCHIVE_ENABLED          = {bench.ARCHIVE_ENABLED}")
    print(f"  ARCHIVE_STREAK_THRESHOLD = {bench.ARCHIVE_STREAK_THRESHOLD}")
    print(f"  ARCHIVE_LAM_TOP_PERCENTILE = {bench.ARCHIVE_LAM_TOP_PERCENTILE}")
    print(f"  ARCHIVE_GRAD_MAG_FLOOR   = {bench.ARCHIVE_GRAD_MAG_FLOOR}")
    print(f"  ARCHIVE_PULSE_MAX        = {bench.ARCHIVE_PULSE_MAX}")
    print(f"  ARCHIVE_MAX_PER_LAYER    = {bench.ARCHIVE_MAX_PER_LAYER}")
    print(f"  QUANTIZE_ARCHIVED_AT_END = {bench.QUANTIZE_ARCHIVED_AT_END}")
    print(f"  QUANTIZE_MODE            = {bench.QUANTIZE_MODE}")
    print(f"  MANIFOLD_REPLAY_ENABLED  = {bench.MANIFOLD_REPLAY_ENABLED}")
    print()
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
