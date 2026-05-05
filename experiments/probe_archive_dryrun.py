"""Single-seed dry run: Manifold Replay + Dream-archive (Phase 1+2).

Confirms archive triggers fire on the chained-15 curriculum and reports
the post-quantization accuracy + deployment-KB breakdown for each arm.

Defaults: streak=3, lam_top=0.75, grad_mag_floor=0.1, pulse_max=0.1,
max=8/layer/call. Calibrate grad_mag_floor here if no archives fire.

Run:
  python3 -m experiments.probe_archive_dryrun \
      > outputs/probe_archive_dryrun.log 2>&1
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
bench.QUANTIZE_MODE = "int8"  # ternary kills full-acc by ~0.31 on
                              # frozen L0 random projection (smoke seed 0);
                              # int8 is essentially lossless. Use ternary
                              # only with QAT or scope-restricted layers.


def main() -> int:
    argv = [
        "--seed", "0",
        "--arms", ",".join([
            "fixed_ewc_small",
            "grown_capped_no_dream",
            "grown_capped_dream",
            "grown_uncapped_dream",
        ]),
        "--csv", "probe_archive_dryrun.csv",
    ]
    print("Dream-archive dry run — single seed, Manifold + Archive ON")
    print(f"  ARCHIVE_ENABLED          = {bench.ARCHIVE_ENABLED}")
    print(f"  ARCHIVE_STREAK_THRESHOLD = {bench.ARCHIVE_STREAK_THRESHOLD}")
    print(f"  ARCHIVE_LAM_TOP_PERCENTILE = {bench.ARCHIVE_LAM_TOP_PERCENTILE}")
    print(f"  ARCHIVE_GRAD_MAG_FLOOR   = {bench.ARCHIVE_GRAD_MAG_FLOOR}")
    print(f"  ARCHIVE_PULSE_MAX        = {bench.ARCHIVE_PULSE_MAX}")
    print(f"  ARCHIVE_MAX_PER_LAYER    = {bench.ARCHIVE_MAX_PER_LAYER}")
    print(f"  QUANTIZE_ARCHIVED_AT_END = {bench.QUANTIZE_ARCHIVED_AT_END}")
    print(f"  QUANTIZE_MODE            = {bench.QUANTIZE_MODE}")
    print()
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
