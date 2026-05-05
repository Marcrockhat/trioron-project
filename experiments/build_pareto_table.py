"""Aggregate existing multi-seed CSVs into a single storage-vs-accuracy
Pareto table for the paper.

Reads each known *_multiseed.csv under outputs/, computes n-seed mean +
std on (full, domain, task-aware) per arm, and writes a tidy table:

    variant, arm, storage_KB, n_seeds,
    full_mean, full_std, domain_mean, domain_std, task_mean, task_std

Storage budgets are computed analytically from the variant configuration
(hippo K * 30 classes * L0_width * 4 bytes, etc.), not from the run logs.

Run:
  python3 -m experiments.build_pareto_table \
      > outputs/pareto_table.log 2>&1

Output:
  outputs/pareto_table.csv
"""
from __future__ import annotations
import csv
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

L0_WIDTH = 128
N_CLASSES = 30
L1_WIDTH_END = 48          # approx final L1 width on grown arms; for
                            # differential's per-class storage estimate
HEAD_WIDTH = 30


def hippo_kb(K: int) -> float:
    return N_CLASSES * K * L0_WIDTH * 4 / 1024.0


def differential_kb() -> float:
    # δL0 + δL1 + δlogit per class.
    return N_CLASSES * (L0_WIDTH + L1_WIDTH_END + HEAD_WIDTH) * 4 / 1024.0


def manifold_kb() -> float:
    # μ + σ per class at L0 width.
    return N_CLASSES * 2 * L0_WIDTH * 4 / 1024.0


def packnet_mask_kb(trainable_params: int, n_tasks: int = 15) -> float:
    # Per-task bool masks (~1 byte each) over trainable params.
    return trainable_params * n_tasks / 1024.0


def hat_mask_kb(masked_dims_total: int, n_tasks: int = 15) -> float:
    # Per-task float embeddings (one per masked-layer unit).
    return masked_dims_total * n_tasks * 4 / 1024.0


# (csv_filename, variant_label, storage_KB)
SOURCES: List[Tuple[str, str, float, Optional[str]]] = [
    ("bench_chained_15task_n3_HIPPO_K1_multiseed.csv",
     "hippo K=1", hippo_kb(1), None),
    ("bench_chained_15task_n3_HIPPO_K50_FIXED_multiseed.csv",
     "hippo K=50", hippo_kb(50), None),
    ("bench_chained_15task_n3_COMBINED_REANCHOR_multiseed.csv",
     "combined storage-free (differential)", differential_kb(), None),
    ("bench_chained_15task_n3_COMBINED_REANCHOR_HIPPO_K10_multiseed.csv",
     "combined + hippo K=10", differential_kb() + hippo_kb(10), None),
    ("bench_chained_15task_n3_COMBINED_REANCHOR_HIPPO_K20_multiseed.csv",
     "combined + hippo K=20", differential_kb() + hippo_kb(20), None),
    ("bench_chained_15task_n3_HIPPOK1_ENGRAM_DIFF_multiseed.csv",
     "hippo K=1 + engram + diff", differential_kb() + hippo_kb(1), None),
    ("bench_chained_15task_n3_ENGRAM_DIFF_COMBINED_multiseed.csv",
     "engram + diff combined", differential_kb(), None),
    ("bench_chained_15task_n3_MANIFOLD_REPLAY_multiseed.csv",
     "manifold (μ,σ)", manifold_kb(), None),
    # Architectural competitors — task masks instead of rehearsal.
    # Storage column reports their own bookkeeping overhead, NOT
    # directly Pareto-comparable with rehearsal-buffer storage.
    ("bench_chained_15task_n3_PACKNET_multiseed.csv",
     "PackNet matched (frozen L0)",
     packnet_mask_kb(14658), "packnet_matched"),
    ("bench_chained_15task_n3_PACKNET_multiseed.csv",
     "PackNet standard (full net)",
     packnet_mask_kb(48862), "packnet_standard"),
    ("bench_chained_15task_n3_HAT_multiseed.csv",
     "HAT matched (frozen L0)",
     hat_mask_kb(128 + 92), "hat_matched"),
    ("bench_chained_15task_n3_HAT_multiseed.csv",
     "HAT standard (full net)",
     hat_mask_kb(56 + 56), "hat_standard"),
]


def aggregate_csv(path: str) -> Dict[str, Dict[str, Tuple[float, float, int]]]:
    """Return {arm: {metric: (mean, std, n)}} for full/domain/task."""
    if not os.path.exists(path):
        return {}
    rows: List[Dict[str, str]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    by_arm: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_arm.setdefault(r["label"], []).append(r)
    out: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for arm, arm_rows in by_arm.items():
        agg: Dict[str, Tuple[float, float, int]] = {}
        for metric_csv, metric_label in [
            ("final_accuracy",        "full"),
            ("final_accuracy_domain", "domain"),
            ("final_accuracy_aware",  "task"),
        ]:
            vals = [float(r[metric_csv]) for r in arm_rows]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            agg[metric_label] = (mean, std, len(vals))
        out[arm] = agg
    return out


def main() -> int:
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs",
    )
    out_csv = os.path.join(out_dir, "pareto_table.csv")
    rows: List[Dict[str, object]] = []
    print(f"{'variant':<42} {'arm':<24} {'KB':>8}  "
          f"{'full':>14} {'domain':>14} {'task':>14}")
    print("-" * 122)
    for fname, variant, kb, arm_filter in SOURCES:
        path = os.path.join(out_dir, fname)
        agg = aggregate_csv(path)
        if not agg:
            print(f"{variant:<42} {'<missing>':<24} {kb:>8.1f}  "
                  f"file not found: {fname}")
            continue
        items = (
            [(arm_filter, agg[arm_filter])]
            if arm_filter is not None and arm_filter in agg
            else agg.items()
        )
        for arm, metrics in items:
            full = metrics["full"]
            domain = metrics["domain"]
            task = metrics["task"]
            print(f"{variant:<42} {arm:<24} {kb:>8.1f}  "
                  f"{full[0]:.3f}±{full[1]:.3f}({full[2]})  "
                  f"{domain[0]:.3f}±{domain[1]:.3f}({domain[2]})  "
                  f"{task[0]:.3f}±{task[1]:.3f}({task[2]})")
            rows.append({
                "variant": variant,
                "arm": arm,
                "storage_KB": f"{kb:.2f}",
                "n_seeds": full[2],
                "full_mean": f"{full[0]:.4f}",
                "full_std": f"{full[1]:.4f}",
                "domain_mean": f"{domain[0]:.4f}",
                "domain_std": f"{domain[1]:.4f}",
                "task_mean": f"{task[0]:.4f}",
                "task_std": f"{task[1]:.4f}",
            })

    fields = [
        "variant", "arm", "storage_KB", "n_seeds",
        "full_mean", "full_std",
        "domain_mean", "domain_std",
        "task_mean", "task_std",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
