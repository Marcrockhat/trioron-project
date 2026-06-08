"""Phase A probe: DeepInversion-style synthetic L0 codes vs real-sample
hippocampal codes.

Question: can logit-driven inversion against the just-consolidated network
produce 128-dim L0-output codes that, when stored in HippocampalBuffer
and replayed via forward_from_layer(start=1), match real K=50 hippo's
0.65 full / 0.96 task-aware ceiling on chained-15?

Setup:
  - Flips HIPPOCAMPAL_ENABLED=True, HIPPOCAMPAL_SYNTHETIC=True at import.
  - Single seed (default 0). Single arm (grown_uncapped_dream — best
    real-hippo arm at K=50 per outputs/bench_chained_15task_n3_HIPPO_K50_FIXED).
  - K=50 to compare apples-to-apples against real K=50.

Headline against real K=50 (n=3 means, from existing CSV):
  fixed_ewc_small        full=0.624  task-aware=0.957
  grown_capped_no_dream  full=0.618  task-aware=0.958
  grown_capped_dream     full=0.612  task-aware=0.962
  grown_uncapped_dream   full=0.646  task-aware=0.966   <-- target

Run:
  python3 -m experiments.probe_synthetic_pseudo_rehearsal \
      > outputs/probe_synthetic_pseudo_rehearsal.log 2>&1
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import bench_chained_15task as bench

# Phase A flags — synthesize codes instead of encoding real samples.
bench.HIPPOCAMPAL_ENABLED = True
bench.HIPPOCAMPAL_SYNTHETIC = True
bench.HIPPOCAMPAL_K_PER_CLASS = 50
# All other rehearsal mechanisms OFF — isolate synthesis.
bench.REHEARSAL_ENABLED = False
bench.LWF_ENABLED = False
bench.BRAINSTEM_ENABLED = False
bench.ENGRAM_ENABLED = False
bench.DIFFERENTIAL_ENABLED = False


def main() -> int:
    argv = [
        "--seed", "0",
        "--arms", "grown_uncapped_dream",
        "--csv", "probe_synthetic_pseudo_rehearsal.csv",
    ]
    print("Phase A — synthetic pseudo-rehearsal (DeepInversion-style)")
    print(f"  HIPPOCAMPAL_ENABLED         = {bench.HIPPOCAMPAL_ENABLED}")
    print(f"  HIPPOCAMPAL_SYNTHETIC       = {bench.HIPPOCAMPAL_SYNTHETIC}")
    print(f"  HIPPOCAMPAL_K_PER_CLASS     = {bench.HIPPOCAMPAL_K_PER_CLASS}")
    print(f"  HIPPOCAMPAL_SYNTH_STEPS     = {bench.HIPPOCAMPAL_SYNTH_STEPS}")
    print(f"  HIPPOCAMPAL_SYNTH_LR        = {bench.HIPPOCAMPAL_SYNTH_LR}")
    print(f"  HIPPOCAMPAL_SYNTH_L2        = {bench.HIPPOCAMPAL_SYNTH_L2}")
    print(f"  HIPPOCAMPAL_SYNTH_INIT_SIGMA= {bench.HIPPOCAMPAL_SYNTH_INIT_SIGMA}")
    print()
    return bench.main(argv)


if __name__ == "__main__":
    sys.exit(main())
