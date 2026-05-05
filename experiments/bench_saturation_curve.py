"""Saturation curve for multi-branch absorption.

Sweeps N from 1 to len(--donors), assembling an organism from the first
N labels in order and evaluating on the union test set. Produces a
table of (N, donor-standalone upper bound, organism task-aware,
organism full-union) under hard / soft / soft+log-softmax routing,
plus a per-branch gate summary at each N. Used to find the bleed-flip
point where task-aware deviates from the upper bound.

Run:
  python3 -m experiments.bench_saturation_curve \\
      --donors digits,fashion,emnist,emnist_kt,emnist_uz \\
      > outputs/bench_saturation_curve_run1.log 2>&1
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments.datasets import (
    DatasetBundle, build_task_views, DEFAULT_DATA_ROOT,
)
from experiments.train_donor import SPLIT_BLOCKS
from experiments.test_multibranch_absorption import (
    _spec_block_for_donor, _datasets_for_donors, evaluate, gate_stats,
)
from trioron.multibranch import Branch, MultiBranchOrganism


def donor_standalone_mean(branch, eval_views_for_branch):
    with torch.no_grad():
        per_task = []
        for v in eval_views_for_branch:
            x, y = v.all_examples()
            logits = branch.net(x)
            cols = list(v.global_classes)
            sub = logits[:, cols]
            pred_local = sub.argmax(dim=-1)
            pred_global = torch.tensor(
                [cols[int(j)] for j in pred_local], dtype=torch.long,
            )
            per_task.append(float((pred_global == y).float().mean().item()))
    return sum(per_task) / len(per_task), per_task


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--donors",
        default="digits,fashion,emnist,emnist_kt,emnist_uz",
        help="Comma-separated donor labels in incremental-N order.",
    )
    parser.add_argument("--ckpt-prefix", default="outputs/poc_donor_")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    labels = [d.strip() for d in args.donors.split(",") if d.strip()]
    print(f"Saturation curve over donors (in order): {labels}")
    print(f"  T = {args.temperature}")

    # Pre-load every donor and its eval views once.
    branches = []
    eval_views_for = {}
    bundle_dataset_names = _datasets_for_donors(labels)
    bundle = DatasetBundle(
        bundle_dataset_names, root=args.data_root, n_holdout_per_dataset=0,
    )
    for lab in labels:
        ckpt = f"{args.ckpt_prefix}{lab}.pt"
        if not os.path.exists(ckpt):
            print(f"  [SKIP] {lab} — checkpoint not found at {ckpt}")
            continue
        b = Branch.from_checkpoint(ckpt, label=lab)
        branches.append(b)
        eval_views_for[lab] = build_task_views(
            bundle, _spec_block_for_donor(lab), split="test",
        )
        print(f"  loaded {lab:<10} arch={list(b.net.n_nodes_per_layer())}  "
              f"classes={b.classes_covered}  "
              f"l0_seed={b.l0_seed}  arm={b.arm}")

    if not branches:
        print("No donors loaded. Aborting.")
        return 1

    rows = []
    for k in range(1, len(branches) + 1):
        prefix = branches[:k]
        labels_k = [b.label for b in prefix]
        print()
        print("#" * 78)
        print(f"#   N = {k}   donors = {labels_k}")
        print("#" * 78)

        # Donor-standalone upper bound — mean over each donor's tasks.
        donor_means = []
        for b in prefix:
            mean_b, _ = donor_standalone_mean(b, eval_views_for[b.label])
            donor_means.append(mean_b)
        upper = sum(donor_means) / len(donor_means)

        # Assemble organism and evaluate on union.
        org = MultiBranchOrganism.from_branches(prefix)
        eval_views_union = []
        for b in prefix:
            eval_views_union.extend(eval_views_for[b.label])

        rows_hard = evaluate(org, eval_views_union, routing="hard")
        rows_soft = evaluate(org, eval_views_union, routing="soft",
                             temperature=args.temperature)
        rows_unif = evaluate(org, eval_views_union, routing="uniform")
        rows_softn = evaluate(
            org, eval_views_union, routing="soft",
            temperature=args.temperature, normalize_per_branch=True,
        )
        rows_hardn = evaluate(
            org, eval_views_union, routing="hard",
            normalize_per_branch=True,
        )

        def mta(rs): return sum(r["task_aware"] for r in rs) / len(rs)
        def mfu(rs): return sum(r["full_union"] for r in rs) / len(rs)

        result = {
            "N": k,
            "donors": labels_k,
            "donor_standalone_task_aware": upper,
            "hard_task_aware": mta(rows_hard),
            "hard_full_union": mfu(rows_hard),
            "soft_task_aware": mta(rows_soft),
            "soft_full_union": mfu(rows_soft),
            "uniform_task_aware": mta(rows_unif),
            "uniform_full_union": mfu(rows_unif),
            "soft_norm_task_aware": mta(rows_softn),
            "soft_norm_full_union": mfu(rows_softn),
            "hard_norm_task_aware": mta(rows_hardn),
            "hard_norm_full_union": mfu(rows_hardn),
        }
        rows.append(result)

        # Per-branch gate summary at this N.
        gates = gate_stats(org, eval_views_union, routing="soft",
                           temperature=args.temperature)
        bls = [b.label for b in prefix]
        print(f"\nGATES (soft, T={args.temperature}) — mean per-task gate weight per branch:")
        print(f"  {'task':<24}" + "".join(f"{lab:>12}" for lab in bls))
        print("  " + "-" * (24 + 12 * len(bls)))
        for r in gates:
            print(f"  {r['task']:<24}" +
                  "".join(f"{g:>12.4f}" for g in r["gates"]))

        print()
        print(f"  upper        task-aware = {upper:.4f}")
        print(f"  HARD         task-aware = {result['hard_task_aware']:.4f}  "
              f"full-union = {result['hard_full_union']:.4f}")
        print(f"  SOFT         task-aware = {result['soft_task_aware']:.4f}  "
              f"full-union = {result['soft_full_union']:.4f}")
        print(f"  UNIFORM      task-aware = {result['uniform_task_aware']:.4f}  "
              f"full-union = {result['uniform_full_union']:.4f}")
        print(f"  HARD + norm  task-aware = {result['hard_norm_task_aware']:.4f}  "
              f"full-union = {result['hard_norm_full_union']:.4f}")
        print(f"  SOFT + norm  task-aware = {result['soft_norm_task_aware']:.4f}  "
              f"full-union = {result['soft_norm_full_union']:.4f}")

    # Final summary table.
    print()
    print("=" * 78)
    print("SATURATION CURVE")
    print("=" * 78)
    header = (
        f"{'N':>3}  {'upper':>8}  "
        f"{'soft+norm TA':>14}  {'soft+norm FU':>14}  "
        f"{'hard+norm TA':>14}  {'soft TA':>10}  {'hard TA':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['N']:>3}  {r['donor_standalone_task_aware']:>8.4f}  "
              f"{r['soft_norm_task_aware']:>14.4f}  {r['soft_norm_full_union']:>14.4f}  "
              f"{r['hard_norm_task_aware']:>14.4f}  "
              f"{r['soft_task_aware']:>10.4f}  {r['hard_task_aware']:>10.4f}")

    # CSV alongside the log (gitignored per outputs/ convention).
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs",
    )
    csv_path = os.path.join(out_dir, "bench_saturation_curve.csv")
    keys = list(rows[0].keys()) if rows else []
    keys.remove("donors")
    with open(csv_path, "w") as f:
        f.write("N,donors," + ",".join(keys[1:]) + "\n")
        for r in rows:
            f.write(f"{r['N']},\"{'+'.join(r['donors'])}\","
                    + ",".join(f"{r[k]:.6f}" for k in keys[1:]) + "\n")
    print(f"\nCSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
