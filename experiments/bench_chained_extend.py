"""Extension experiment: chained-15 → consolidation dream → chained-15+8.

Tests the device-conscience deployment loop:
  1. Train chained-15 with archive on (Phase 1) + manifold replay.
  2. Fire a "shipping consolidation" dream: full-coverage replay over
     all 15 tasks with archive-aware grad masking, then one more
     archive_block call to lock anything that just settled.
  3. Permanently snap archived rows to int8 (Phase 2).
  4. Lift the cap from 32 KB → 64 KB trainable.
  5. Continue with 8 new EMNIST-letters K..Z binary tasks
     (global classes 30..45). Watch the network grow new plastic
     capacity to handle them while archived rows stay locked at int8.

Headline questions:
  - Does the network grow back the plastic substrate it lost to archive?
  - Do the new tasks learn at the same accuracy as the original block?
  - Do the original 15 tasks survive the extension (no catastrophic
    forgetting from the new growth)?
  - What's the deployment-KB delta after 23 tasks?

Run:
  python3 -m experiments.bench_chained_extend \
      > outputs/bench_chained_extend_smoke.log 2>&1
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import bench_chained_15task as bench
from experiments.datasets import (
    DEFAULT_DATA_ROOT,
    DatasetBundle,
    build_task_views,
    chained_15_specs,
    chained_extension_specs,
)


# Extension cap: doubles the chained-15 cap so the network has room to
# grow plastic capacity for the 8 new tasks while keeping the ~4 KB
# locked from archive.
EXTENSION_CAP_BYTES = 16_000 * 4  # 16k trainable params → 64 KB
EXTENSION_N_TASKS = 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="4 epochs/task instead of 8.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds", default="",
        help="Comma-separated seed list (overrides --seed).",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--arms",
        default="grown_capped_dream",
        help="Arm subset (only frozen-L0 grown arms make sense for the "
             "extension experiment; archive + manifold both require "
             "frozen L0).",
    )
    parser.add_argument(
        "--csv", default="bench_chained_extend.csv",
    )
    parser.add_argument(
        "--no-permanent-int8", action="store_true",
        help="Skip permanent int8 snap at extension boundary "
             "(default: snap on, simulating shipped state).",
    )
    args = parser.parse_args()

    # Module-level config: archive on + manifold replay + Phase 2
    # quantization simulation at end-of-extension for the report.
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

    n_epochs = (bench.N_EPOCHS_PER_TASK_SMOKE if args.smoke
                else bench.N_EPOCHS_PER_TASK)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    # Build datasets: bundle has all 3 sources (mnist, fashion_mnist,
    # emnist_letters). Main views = chained_15_specs, extension =
    # chained_extension_specs.
    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=args.data_root,
        n_holdout_per_dataset=bench.N_INFANCY_PER_DATASET,
    )
    main_specs = chained_15_specs()
    ext_specs = chained_extension_specs(EXTENSION_N_TASKS)

    main_train_views = build_task_views(bundle, main_specs, split="train")
    main_eval_views = build_task_views(bundle, main_specs, split="test")
    main_class_lists = [s.global_classes for s in main_specs]

    ext_train_views = build_task_views(bundle, ext_specs, split="train")
    ext_eval_views = build_task_views(bundle, ext_specs, split="test")
    ext_class_lists = [s.global_classes for s in ext_specs]

    infancy_view = (bundle.infancy_view(main_specs)
                    if bench.WARMUP_ENABLED else None)

    print("=" * 78)
    print("Trioron — bench_chained_extend: chained-15 → +8 EMNIST K..Z")
    print("=" * 78)
    print(f"Epochs/task:        {n_epochs}{' [SMOKE]' if args.smoke else ''}")
    print(f"Main tasks:         15 (mnist+fashion+EMNIST A..J)")
    print(f"Extension tasks:    {EXTENSION_N_TASKS} (EMNIST K..Z, "
          f"global classes 30..{30 + 2*EXTENSION_N_TASKS - 1})")
    print(f"Main cap:           {bench.M_MAX_BYTES_CAPPED:_} B "
          f"({bench.M_MAX_BYTES_CAPPED//4:_} trainable params)")
    print(f"Extension cap:      {EXTENSION_CAP_BYTES:_} B "
          f"({EXTENSION_CAP_BYTES//4:_} trainable params)")
    print(f"Permanent int8:     {not args.no_permanent_int8}")
    print(f"Arms:               {arms}")
    print(f"Seeds:              {seeds}")
    print()

    all_results = []
    for seed in seeds:
        print(f"\n{'#'*78}\n#   SEED {seed}\n{'#'*78}")
        for arm in arms:
            r = bench.run_arm(
                arm,
                seed=seed + (hash(arm) % 7919),
                n_epochs_per_task=n_epochs,
                train_views=main_train_views,
                eval_views=main_eval_views,
                task_class_lists=main_class_lists,
                infancy_view=infancy_view,
                n_passes=1,
                extension_train_views=ext_train_views,
                extension_eval_views=ext_eval_views,
                extension_task_class_lists=ext_class_lists,
                extension_cap_bytes=EXTENSION_CAP_BYTES,
                extension_permanent_int8=not args.no_permanent_int8,
            )
            r["seed"] = seed
            all_results.append(r)
        bench.report([r for r in all_results if r["seed"] == seed])

    if len(seeds) > 1:
        bench.report_multiseed(all_results, arms)

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.csv)
    if len(seeds) > 1:
        bench.write_csv_multiseed(
            all_results, csv_path.replace(".csv", "_multiseed.csv"),
        )
    else:
        bench.write_csv(all_results, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
