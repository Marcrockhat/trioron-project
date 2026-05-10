"""Build a SensoryConductor over the trained sense donors and report
fusion lift on the CIFAR-100 first-slice held-out test set.

Compares each donor in isolation vs. the fused conductor under each of
the three fusion rules, and reports both task-aware accuracy (5-way
within group) and full-slice accuracy (25-way over the union).
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from trioron.senses.conductor import (
    SensoryConductor, load_sense_donor,
)
from experiments.cifar.datasets import (
    load_cifar100, SLICES, DEFAULT_DATA_ROOT,
)


def _slice_test_set(class_groups, root):
    images, labels = load_cifar100(root, train=False)
    all_classes = sorted({int(c) for g in class_groups for c in g})
    keep = torch.zeros(labels.shape[0], dtype=torch.bool)
    for c in all_classes:
        keep |= labels == c
    return images[keep], labels[keep], all_classes


def _task_for_class(c: int, class_groups: List[List[int]]) -> int:
    for i, g in enumerate(class_groups):
        if int(c) in g:
            return i
    raise KeyError(f"class {c} not in any task group")


def _eval_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    union_classes: List[int],
    class_groups: List[List[int]],
) -> Dict[str, float]:
    """Return {'full': full-slice accuracy, 'domain': 10-class-block
    accuracy, 'task': task-aware accuracy}.

    Domain matches bench_chained_15task: the 10-class block containing
    the current image's task. CIFAR-100 tasks are contiguous 5-class
    blocks aligned to multiples of 5, so two consecutive tasks share
    one domain — chance 1/10 when restricted to the 10-class superset.
    """
    union = list(union_classes)
    cls_to_col = {c: i for i, c in enumerate(union)}
    union_set = set(union)
    # Full-slice argmax over all union classes.
    pred_cols = logits.argmax(dim=1)
    pred_cls = torch.tensor([union[i] for i in pred_cols.tolist()],
                            dtype=labels.dtype)
    full = (pred_cls == labels).float().mean().item()

    correct_task = 0
    correct_domain = 0
    for i in range(labels.shape[0]):
        c = int(labels[i].item())
        t = _task_for_class(c, class_groups)
        # Task-aware: restrict to current task's 5 classes.
        active_cols = [cls_to_col[g] for g in class_groups[t]]
        local_pred_t = logits[i, active_cols].argmax().item()
        if class_groups[t][local_pred_t] == c:
            correct_task += 1
        # Domain: 10-class block containing the smallest class of this task.
        task0 = min(class_groups[t])
        domain_idx = task0 // 10
        domain_classes = [
            cls for cls in range(domain_idx * 10, (domain_idx + 1) * 10)
            if cls in union_set
        ]
        if domain_classes:
            domain_cols = [cls_to_col[cls] for cls in domain_classes]
            local_pred_d = logits[i, domain_cols].argmax().item()
            if domain_classes[local_pred_d] == c:
                correct_domain += 1
    n = labels.shape[0]
    return {
        "full": full,
        "domain": correct_domain / n,
        "task": correct_task / n,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slice", choices=sorted(SLICES), default="first",
        help="Class slice to evaluate against — must match the donors'.",
    )
    parser.add_argument(
        "--donor-dir", default=None,
        help="Default scales with --slice: outputs/cifar_donors[_full].",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--senses", nargs="+",
        default=["cortex", "color_smell", "frequency_print"],
    )
    parser.add_argument(
        "--greedy-select", action="store_true",
        help="Forward-greedy fusion subset selection: add donors one "
             "at a time by which one most improves task-aware accuracy. "
             "Stops when no further addition helps.",
    )
    args = parser.parse_args(argv)

    class_groups = SLICES[args.slice]
    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(DEFAULT_DATA_ROOT), sub)

    donor_paths = [
        os.path.join(args.donor_dir, f"sense_donor_{s}.pt")
        for s in args.senses
    ]
    for p in donor_paths:
        if not os.path.exists(p):
            print(f"missing donor checkpoint: {p}", file=sys.stderr)
            return 2

    donors = [load_sense_donor(p) for p in donor_paths]
    print(f"loaded {len(donors)} donors:")
    for d in donors:
        print(f"  sense={d.sense_name:14s}  classes={len(d.classes_covered)}  "
              f"head_arch={list(d.net.n_nodes_per_layer())}")

    test_imgs, test_labs, all_classes = _slice_test_set(
        class_groups, args.data_root,
    )
    print(f"slice={args.slice}  donor_dir={args.donor_dir}")
    print(f"test set: {test_imgs.shape[0]} images, "
          f"{len(all_classes)} classes")

    # Per-donor solo evaluation (using the conductor with a single
    # donor — ensures the same standardizer pipeline as fused).
    print("\n=== per-donor solo accuracy ===")
    print(f"{'sense':<14s} {'full':>8s} {'domain':>8s} {'task':>8s}")
    solo: Dict[str, Dict[str, float]] = {}
    for d in donors:
        cnd = SensoryConductor([d], fusion="mean_logit").eval()
        with torch.no_grad():
            logits = cnd(test_imgs)
        m = _eval_logits(logits, test_labs, cnd.union_classes,
                         class_groups)
        solo[d.sense_name] = m
        print(f"{d.sense_name:<14s} {m['full']:>8.4f} {m['domain']:>8.4f} "
              f"{m['task']:>8.4f}")

    # Fused conductor with all donors, each fusion rule.
    print("\n=== fused conductor (all donors) ===")
    print(f"{'fusion':<22s} {'full':>8s} {'domain':>8s} {'task':>8s}")
    fusion_results: Dict[str, Dict[str, float]] = {}
    for fusion in ("mean_logit", "sum_logit", "log_prob_mean",
                   "confidence_weighted"):
        cnd = SensoryConductor(donors, fusion=fusion).eval()
        with torch.no_grad():
            logits = cnd(test_imgs)
        m = _eval_logits(logits, test_labs, cnd.union_classes,
                         class_groups)
        fusion_results[fusion] = m
        print(f"{fusion:<22s} {m['full']:>8.4f} {m['domain']:>8.4f} "
              f"{m['task']:>8.4f}")

    # Leave-one-out ablation under the strongest fusion rule. Each row
    # shows what fusion accuracy looks like without one specific donor
    # — negative numbers indicate the donor is contributing; positive
    # mean removing it helps.
    print("\n=== leave-one-out ablation (confidence_weighted) ===")
    print(f"{'removed':<18s} {'full':>8s} {'domain':>8s} {'task':>8s}  "
          f"{'Δfull':>8s} {'Δdomain':>8s} {'Δtask':>8s}")
    base_full = fusion_results["confidence_weighted"]["full"]
    base_domain = fusion_results["confidence_weighted"]["domain"]
    base_task = fusion_results["confidence_weighted"]["task"]
    print(f"{'(none, baseline)':<18s} {base_full:>8.4f} {base_domain:>8.4f} "
          f"{base_task:>8.4f}")
    for i, d in enumerate(donors):
        kept = [donors[j] for j in range(len(donors)) if j != i]
        cnd = SensoryConductor(kept, fusion="confidence_weighted").eval()
        with torch.no_grad():
            logits = cnd(test_imgs)
        m = _eval_logits(logits, test_labs, cnd.union_classes,
                         class_groups)
        d_full = m["full"] - base_full
        d_domain = m["domain"] - base_domain
        d_task = m["task"] - base_task
        marker_full = " ↑" if d_full > 0 else ("  " if d_full == 0 else "")
        print(f"{d.sense_name:<18s} {m['full']:>8.4f} {m['domain']:>8.4f} "
              f"{m['task']:>8.4f}  "
              f"{d_full:>+8.4f} {d_domain:>+8.4f} {d_task:>+8.4f}{marker_full}")

    # Greedy forward selection: add donors one at a time, keep if
    # task-aware accuracy increases under mean_logit fusion.
    if args.greedy_select:
        print("\n=== greedy forward selection (mean_logit, task-aware metric) ===")
        remaining = list(range(len(donors)))
        chosen: List[int] = []
        best_task = 0.0
        best_full = 0.0
        step = 0
        while remaining:
            best_idx = None
            best_local_task = best_task
            best_local_full = best_full
            for i in remaining:
                trial = chosen + [i]
                cnd = SensoryConductor(
                    [donors[j] for j in trial], fusion="mean_logit",
                ).eval()
                with torch.no_grad():
                    logits = cnd(test_imgs)
                m = _eval_logits(logits, test_labs, cnd.union_classes,
                                 class_groups)
                if m["task"] > best_local_task or (
                    m["task"] == best_local_task and m["full"] > best_local_full
                ):
                    best_local_task = m["task"]
                    best_local_full = m["full"]
                    best_idx = i
            if best_idx is None:
                break
            step += 1
            chosen.append(best_idx)
            remaining.remove(best_idx)
            best_task = best_local_task
            best_full = best_local_full
            kept = [donors[j].sense_name for j in chosen]
            print(f"step {step:>2d}: +{donors[best_idx].sense_name:<18s} "
                  f"full={best_full:.4f}  task={best_task:.4f}  "
                  f"set={kept}")
        print(f"\nfinal set ({len(chosen)} senses):  full={best_full:.4f}  "
              f"task={best_task:.4f}")
        chosen_names = [donors[j].sense_name for j in chosen]
        print(f"  senses: {chosen_names}")

    # Lift summary.
    best_solo_full = max(s["full"] for s in solo.values())
    best_solo_task = max(s["task"] for s in solo.values())
    best_fused_full = max(f["full"] for f in fusion_results.values())
    best_fused_task = max(f["task"] for f in fusion_results.values())
    n_classes = len(all_classes)
    chance_full = 1.0 / n_classes
    chance_task = 1.0 / max(len(g) for g in class_groups)

    print("\n=== lift summary ===")
    print(f"chance:                full {chance_full:.4f}  task {chance_task:.4f}")
    print(f"best single donor:     full {best_solo_full:.4f}  task {best_solo_task:.4f}")
    print(f"best fused conductor:  full {best_fused_full:.4f}  task {best_fused_task:.4f}")
    print(f"fusion lift (Δ):       full {best_fused_full - best_solo_full:+.4f}  "
          f"task {best_fused_task - best_solo_task:+.4f}")
    print(f"fusion / best-solo:    full {best_fused_full / best_solo_full:.2f}×  "
          f"task {best_fused_task / best_solo_task:.2f}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
