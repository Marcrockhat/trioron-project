"""Manifold-replay n=3 head-to-head: baseline vs all-axes+plastic.

Re-runs the manifold-replay panel (grown_uncapped_dream arm, n_passes=1)
twice on the same 3 seeds:

  1. Baseline: only MANIFOLD_REPLAY=1 (the n10_headline_numbers config
     that reaches 0.601/0.677/0.961).
  2. All-axes-v2 + plastic: MANIFOLD_REPLAY=1 + AXIS_3+AXIS_4+AXIS_5
     + PLASTIC (Axis 5.5).

This is the corrected version of the 05-20/21 chained-15 regression
test, which ran on the `grown_capped_dream` arm without manifold
replay (i.e. on the impaired ~0.155 default config). Per
`all_axes_chained15_regression` memory's own caveat: "the architecture
comparison should be run on the paper config, not the default
smoke-arm config."

Two output CSVs (one per condition); compare paired-σ across seeds.

Run:
  python3 -m experiments.bench_manifold_replay_n3_plastic \
      > outputs/bench_chained_15task_n3_MANIFOLD_REPLAY_plastic.log 2>&1
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_condition(condition: str) -> int:
    """Run one of the two conditions in a clean env-var slate."""
    # Reset relevant env vars so a stale value from a prior shell
    # session can't leak into this run's gating.
    for k in [
        "TRIORON_MANIFOLD_REPLAY",
        "TRIORON_DENDRITIC_GROWTH",
        "TRIORON_AXIS_3", "TRIORON_AXIS_4", "TRIORON_AXIS_5",
        "TRIORON_PLASTIC",
        "TRIORON_PLASTIC_REWIRE_STEPS",
        "TRIORON_PLASTIC_TEMP_START",
        "TRIORON_PLASTIC_TEMP_END",
        "TRIORON_PLASTIC_INIT",
        "TRIORON_PLASTIC_LR",
    ]:
        os.environ.pop(k, None)

    os.environ["TRIORON_MANIFOLD_REPLAY"] = "1"

    if condition == "baseline":
        csv_name = "bench_chained_15task_n3_MANIFOLD_REPLAY_baseline.csv"
    elif condition == "all_axes_plastic":
        os.environ["TRIORON_DENDRITIC_GROWTH"] = "1"
        os.environ["TRIORON_AXIS_3"] = "1"
        os.environ["TRIORON_AXIS_4"] = "1"
        os.environ["TRIORON_AXIS_5"] = "1"
        os.environ["TRIORON_PLASTIC"] = "1"
        os.environ["TRIORON_PLASTIC_REWIRE_STEPS"] = "50"
        csv_name = "bench_chained_15task_n3_MANIFOLD_REPLAY_all_axes_plastic.csv"
    else:
        raise ValueError(f"unknown condition: {condition!r}")

    # Import the bench module AFTER env vars are set so its module-
    # level env reads pick up the right values.
    if "experiments.bench_chained_15task" in sys.modules:
        # Force re-import on the second condition so module-level env
        # reads re-evaluate. The bench reads env vars at module-load
        # time (e.g. AXIS_3_ENABLED, PLASTIC_ENABLED, ...).
        del sys.modules["experiments.bench_chained_15task"]
    from experiments import bench_chained_15task as bench

    # Defensive: make sure manifold replay is on at the module level
    # (the bench's flag was read from env, but be explicit anyway).
    bench.MANIFOLD_REPLAY_ENABLED = True
    bench.HIPPOCAMPAL_ENABLED = False
    bench.HIPPOCAMPAL_SYNTHETIC = False
    bench.REHEARSAL_ENABLED = False
    bench.LWF_ENABLED = False
    bench.BRAINSTEM_ENABLED = False
    bench.ENGRAM_ENABLED = False
    bench.DIFFERENTIAL_ENABLED = False

    argv = [
        "--seeds", "0,1,2",
        "--arms", "grown_uncapped_dream",
        "--csv", csv_name,
    ]
    print("=" * 78)
    print(f"CONDITION: {condition}")
    print(f"  MANIFOLD_REPLAY_ENABLED = {bench.MANIFOLD_REPLAY_ENABLED}")
    print(f"  AXIS_3_ENABLED          = {bench.AXIS_3_ENABLED}")
    print(f"  AXIS_4_ENABLED          = {bench.AXIS_4_ENABLED}")
    print(f"  AXIS_5_ENABLED          = {bench.AXIS_5_ENABLED}")
    print(f"  PLASTIC_ENABLED         = {bench.PLASTIC_ENABLED}")
    print("=" * 78)
    print()
    rc = bench.main(argv)
    print()
    print(f"[{condition}] returned rc={rc}")
    return rc


def main() -> int:
    rc1 = _run_condition("baseline")
    rc2 = _run_condition("all_axes_plastic")
    return rc1 or rc2


if __name__ == "__main__":
    sys.exit(main())
