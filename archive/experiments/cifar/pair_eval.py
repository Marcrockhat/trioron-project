"""Stage-C bench: single-look + pair-mode escalation on full-100 CIFAR.

Layers:
  1. Load organism + trained calibrator (Stage B router + ambig head)
     + class prototypes.
  2. Compute single-look fused logits and ambig probability over the
     test set (one forward pass).
  3. Compute per-branch per-class distances vs prototypes (one
     forward pass).
  4. Bench:
        baseline          — single-look uniform routing (0.143/0.631)
        pair_all_100      — argmin distance over ALL 100 classes
        pair_topK         — argmin distance over top-K from single-look
                            (K ∈ {3, 5, 10})
        escalated_topK    — keep single-look where ambig high, pair-mode
                            for the bottom-N% by ambig score
  5. Sweep escalation rate ∈ {10, 20, 30, 40, 50, 100}% to find best
     trade-off.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from trioron.senses.organism import SensoryOrganism
from trioron.senses.calibrator import (
    BRANCH_FEATURE_NAMES, stack_branch_features,
    CalibratedRouter, AmbiguityHead,
    fuse_with_router, fused_confidence_aux,
)
from trioron.senses.pair import (
    PrototypeBank, per_branch_distances, resolve_pair,
)
from experiments.cifar.datasets import (
    load_cifar100, SLICES, DEFAULT_DATA_ROOT,
)
from experiments.cifar.conductor_eval import _eval_logits


GREEDY_7 = ["cortex", "color_smell", "frequency_print", "taste",
            "random_walk"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", choices=sorted(SLICES), default="full")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--donor-dir", default=None)
    parser.add_argument("--senses", nargs="+", default=GREEDY_7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-ks", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument(
        "--escalation-fractions", type=float, nargs="+",
        default=[0.10, 0.20, 0.30, 0.40, 0.50, 1.0],
    )
    parser.add_argument(
        "--prototype-path", default=None,
        help="If set, load prototypes from this file instead of refitting.",
    )
    parser.add_argument(
        "--save-prototypes", action="store_true",
        help="Save the freshly fit bank to outputs/cifar_donors_full/"
             "class_prototypes.pt for reuse.",
    )
    args = parser.parse_args(argv)

    if args.donor_dir is None:
        sub = "cifar_donors" if args.slice == "first" else "cifar_donors_full"
        args.donor_dir = os.path.join(os.path.dirname(args.data_root), sub)

    donor_paths = [
        os.path.join(args.donor_dir, f"sense_donor_{s}.pt")
        for s in args.senses
    ]
    for p in donor_paths:
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr)
            return 2

    print(f"slice={args.slice}  donors={args.senses}")
    org = SensoryOrganism.from_sense_donors(donor_paths).eval()
    n_branches = len(org.branches)
    union_classes = org.union_classes
    class_groups = SLICES[args.slice]

    # ---- load calibrator ----
    calib_path = os.path.join(args.donor_dir, "sensory_calibrator.pt")
    payload = torch.load(calib_path, map_location="cpu", weights_only=False)
    router = CalibratedRouter(n_branches=n_branches)
    router.load_state_dict(payload["router_state_dict"])
    router.eval()
    head = AmbiguityHead(n_branches=n_branches)
    head.load_state_dict(payload["ambig_state_dict"])
    head.eval()
    print(f"loaded calibrator: router {sum(p.numel() for p in router.parameters())} params, "
          f"ambig {sum(p.numel() for p in head.parameters())} params")

    # ---- prototypes ----
    if args.prototype_path is None:
        args.prototype_path = os.path.join(args.donor_dir, "class_prototypes.pt")
    if os.path.exists(args.prototype_path):
        print(f"loading prototypes from {args.prototype_path}")
        bank = PrototypeBank.load(args.prototype_path)
    else:
        print("fitting prototype bank from CIFAR-100 train...")
        train_imgs, train_labs = load_cifar100(args.data_root, train=True)
        t0 = time.time()
        bank = PrototypeBank.fit_from_organism(
            org, train_imgs, train_labs, batch_size=args.batch_size,
        )
        print(f"  fit done in {time.time()-t0:.1f}s; "
              f"storage {bank.storage_bytes()/1024:.2f} KB")
        if args.save_prototypes or True:
            bank.save(args.prototype_path)
            print(f"  saved → {args.prototype_path}")

    # ---- test set ----
    test_imgs, test_labs = load_cifar100(args.data_root, train=False)
    N = test_imgs.shape[0]
    print(f"test: {N} images, {len(union_classes)} classes")

    # ---- forward pass: single-look features + padded logits + distances ----
    print("\ncomputing test-set features + distances...")
    t0 = time.time()
    feats_full = torch.empty(N, n_branches, len(BRANCH_FEATURE_NAMES))
    padded_full = torch.empty(N, n_branches, len(union_classes), dtype=torch.float16)
    dist_full = torch.empty(N, n_branches, len(bank.class_order))
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            j = min(i + args.batch_size, N)
            fd = org.branch_features(test_imgs[i:j])
            feats_full[i:j] = stack_branch_features(fd)
            padded_full[i:j] = fd["branch_logits_padded"].to(torch.float16)
            dist_full[i:j] = per_branch_distances(org, test_imgs[i:j], bank)
    print(f"  done in {time.time()-t0:.1f}s")

    # ---- single-look fused (uniform routing → matches conductor mean_logit) ----
    uniform_g = feats_full.new_full((N, n_branches), 1.0 / n_branches)
    with torch.no_grad():
        fused_uniform = fuse_with_router(padded_full.float(), uniform_g)
        fused_aux_uniform = fused_confidence_aux(fused_uniform)
        ambig_p = head(feats_full, fused_aux_uniform)
    pred_single = fused_uniform.argmax(-1)
    pred_single_classes = torch.tensor(
        [union_classes[i] for i in pred_single.tolist()], dtype=test_labs.dtype,
    )
    base_full = (pred_single_classes == test_labs).float().mean().item()
    print(f"\nbaseline single-look uniform: full={base_full:.4f}")

    # Also evaluate baseline task-aware via existing helper.
    base_metrics = _eval_logits(
        fused_uniform, test_labs, union_classes, class_groups,
    )
    print(f"baseline single-look: full={base_metrics['full']:.4f}  "
          f"task={base_metrics['task']:.4f}")

    # ---- single-look per-image task labels (which task group each image belongs to) ----
    def _task_class_groups_per_image(labels: torch.Tensor) -> List[List[int]]:
        """Return the in-task class group (length-5 list of class IDs)
        for each image's true label."""
        groups: List[List[int]] = []
        for c in labels.tolist():
            for g in class_groups:
                if int(c) in g:
                    groups.append(list(g))
                    break
        return groups
    per_img_task_classes = _task_class_groups_per_image(test_labs)
    # Tensorize to (N, 5) for vectorized pair-mode task-aware.
    task_cand = torch.tensor(per_img_task_classes, dtype=torch.long)

    # Single-look task-aware predicted class per image (for escalation merge).
    pred_single_task = torch.empty(N, dtype=test_labs.dtype)
    for i in range(N):
        active = [union_classes.index(c) for c in per_img_task_classes[i]]
        local = fused_uniform[i, active].argmax().item()
        pred_single_task[i] = per_img_task_classes[i][local]
    base_task = (pred_single_task == test_labs).float().mean().item()
    print(f"baseline single-look task-aware (recomputed): {base_task:.4f}")

    # ---- pair-mode-only configurations ----
    print("\n=== pair-mode standalone (no escalation gate) ===")
    print(f"{'mode':<24s} {'full':>8s} {'task':>8s}  Δfull   Δtask")
    pair_results: Dict[str, Dict[str, float]] = {}

    # (a) all 100 candidates for full; task group for task-aware.
    all_cand = torch.tensor(bank.class_order, dtype=torch.long).view(1, -1).expand(N, -1)
    picks_all_full = resolve_pair(dist_full, all_cand, bank)
    full_all = (picks_all_full == test_labs).float().mean().item()
    picks_all_task = resolve_pair(dist_full, task_cand, bank)
    task_all = (picks_all_task == test_labs).float().mean().item()
    print(f"{'pair full=100, task=5':<24s} {full_all:>8.4f} {task_all:>8.4f}  "
          f"{full_all - base_metrics['full']:+.4f}  {task_all - base_task:+.4f}")
    pair_results["pair_full"] = {"full": full_all, "task": task_all}

    # (b) pair-mode constrained to top-K from single-look (full metric).
    for K in args.top_ks:
        topk = fused_uniform.topk(K, dim=-1).indices                # (N, K)
        cand_classes = torch.tensor(
            [[union_classes[c] for c in row.tolist()] for row in topk],
            dtype=torch.long,
        )
        picks_k = resolve_pair(dist_full, cand_classes, bank)
        full_k = (picks_k == test_labs).float().mean().item()
        # Task-aware metric uses task_cand regardless of K.
        d_full = full_k - base_metrics['full']
        # Reusing pair_full's task value because pair task-aware doesn't
        # depend on K (it's restricted to the 5 task classes).
        d_task = task_all - base_task
        tag = f"pair top-{K}, task=5"
        print(f"{tag:<24s} {full_k:>8.4f} {task_all:>8.4f}  "
              f"{d_full:+.4f}  {d_task:+.4f}")
        pair_results[f"pair_top{K}"] = {"full": full_k, "task": task_all}

    # ---- escalated bench ----
    print("\n=== escalated (single-look + pair-mode for low-ambig) ===")
    print(f"top-K escalation candidates: {args.top_ks}  "
          f"(K applies only to full-metric branch; task-aware always uses 5)")
    print(f"{'esc-frac':>9s} {'thresh':>8s} {'K':>3s} {'full':>8s} {'task':>8s}  Δfull   Δtask")
    for esc_frac in args.escalation_fractions:
        if esc_frac >= 1.0:
            mask = torch.ones(N, dtype=torch.bool)
            thresh = float("inf")
        else:
            thresh = torch.quantile(ambig_p, esc_frac).item()
            mask = ambig_p <= thresh
        n_esc = int(mask.sum().item())
        for K in args.top_ks:
            # Top-K from single-look for full metric.
            topk = fused_uniform.topk(K, dim=-1).indices
            cand_classes = torch.tensor(
                [[union_classes[c] for c in row.tolist()] for row in topk],
                dtype=torch.long,
            )

            # FULL metric: where mask, escalate via top-K pair pick.
            final_full = pred_single_classes.clone()
            esc_picks_full = resolve_pair(
                dist_full[mask], cand_classes[mask], bank,
            )
            final_full[mask] = esc_picks_full
            full = (final_full == test_labs).float().mean().item()

            # TASK metric: where mask, escalate via task-group pair pick.
            final_task = pred_single_task.clone()
            esc_picks_task = resolve_pair(
                dist_full[mask], task_cand[mask], bank,
            )
            final_task[mask] = esc_picks_task
            task = (final_task == test_labs).float().mean().item()

            d_full = full - base_metrics['full']
            d_task = task - base_task
            print(f"{esc_frac:>9.2f} {thresh:>8.3f} {K:>3d} "
                  f"{full:>8.4f} {task:>8.4f}  "
                  f"{d_full:+.4f}  {d_task:+.4f}  (n_esc={n_esc})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
