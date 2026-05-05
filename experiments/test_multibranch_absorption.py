"""Multi-branch absorption test — instantiates a recipient organism by
loading two pre-trained donors (digits + fashion) as parallel L1
branches off a shared frozen L0 and evaluates the assembled organism
on the union test set with no further training.

Headline question: does shared-L0 paste-and-go absorption preserve each
donor's task-aware accuracy under three routing modes?

  hard    — argmax over branches (single specialist fires per input)
  soft    — softmax(log_lik / T) over branches (bleed allowed)
  uniform — 1/N gates (pure ablation; routing disabled)

Upper bound for each per-task score is the donor's standalone
task-aware accuracy reported in `outputs/poc_dual_organism_*_run1.log`.

Run:
  python3 -m experiments.test_multibranch_absorption \\
      > outputs/test_multibranch_absorption_run1.log 2>&1
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from experiments.datasets import (
    DatasetBundle, build_task_views, chained_15_specs, DEFAULT_DATA_ROOT,
)
from trioron.multibranch import Branch, MultiBranchOrganism


def task_aware_accuracy(
    logits_union: torch.Tensor,
    union_classes: list,
    labels_global: torch.Tensor,
    active_classes: list,
) -> float:
    """Argmax restricted to `active_classes`. logits_union is shape
    (N, len(union_classes)) and indexed by union_classes[i] = global
    class id at column i."""
    union_idx = {c: i for i, c in enumerate(union_classes)}
    cols = [union_idx[c] for c in active_classes]
    sub = logits_union[:, cols]                    # (N, |active|)
    pred_local = sub.argmax(dim=-1)
    pred_global = torch.tensor(
        [active_classes[int(j)] for j in pred_local], dtype=torch.long,
    )
    return float((pred_global == labels_global).float().mean().item())


def full_softmax_accuracy(
    logits_union: torch.Tensor,
    union_classes: list,
    labels_global: torch.Tensor,
) -> float:
    """Argmax over the full union (all 20 classes from the assembled
    organism). Tests cross-branch competition."""
    pred_idx = logits_union.argmax(dim=-1)
    pred_global = torch.tensor(
        [union_classes[int(j)] for j in pred_idx], dtype=torch.long,
    )
    return float((pred_global == labels_global).float().mean().item())


def evaluate(
    org, eval_views, *, routing, temperature=1.0,
    normalize_per_branch=False, batch=512,
):
    """Forward each task's full eval set in batches, collect logits."""
    union = org.union_classes
    rows = []
    with torch.no_grad():
        for v in eval_views:
            x, y = v.all_examples()
            n = x.shape[0]
            logits_chunks = []
            for s in range(0, n, batch):
                logits_chunks.append(
                    org(
                        x[s:s + batch], routing=routing,
                        temperature=temperature,
                        normalize_per_branch=normalize_per_branch,
                    )
                )
            logits = torch.cat(logits_chunks, dim=0)
            ta = task_aware_accuracy(logits, union, y, list(v.global_classes))
            full = full_softmax_accuracy(logits, union, y)
            rows.append({
                "task": v.name,
                "n": n,
                "active": list(v.global_classes),
                "task_aware": ta,
                "full_union": full,
            })
    return rows


def report_block(title, rows):
    print(f"\n--- {title}")
    header = f"{'task':<22}{'n':>6}  {'active':<14}{'task-aware':>12}{'full-union':>12}"
    print(header)
    print("-" * len(header))
    ta_sum = 0.0
    fu_sum = 0.0
    for r in rows:
        print(f"{r['task']:<22}{r['n']:>6}  "
              f"{str(r['active']):<14}"
              f"{r['task_aware']:>12.4f}{r['full_union']:>12.4f}")
        ta_sum += r["task_aware"]
        fu_sum += r["full_union"]
    n = len(rows)
    print(f"{'mean':<22}{'':>6}  {'':<14}{ta_sum/n:>12.4f}{fu_sum/n:>12.4f}")


def gate_stats(org, eval_views, *, routing, temperature=1.0, batch=512):
    """Average per-branch gate weight per task — diagnostic for whether
    routing is firing the expected branch on the expected task."""
    n_branches = len(org.branches)
    rows = []
    with torch.no_grad():
        for v in eval_views:
            x, _ = v.all_examples()
            n = x.shape[0]
            gate_acc = torch.zeros(n_branches)
            for s in range(0, n, batch):
                _, extras = org(
                    x[s:s + batch], routing=routing,
                    temperature=temperature, return_extras=True,
                )
                gate_acc += extras["gates"].sum(dim=0)
            gate_acc = gate_acc / n
            rows.append({"task": v.name, "gates": gate_acc.tolist()})
    return rows


def _spec_block_for_donor(donor_label, all_specs):
    """Map a donor label to the chained_15 sub-block it was trained on
    (used to pick test specs for that donor's classes)."""
    if donor_label == "digits":
        return all_specs[0:5]   # MNIST     (global 0..9)
    if donor_label == "fashion":
        return all_specs[5:10]  # Fashion   (global 10..19)
    if donor_label == "emnist":
        return all_specs[10:15] # EMNIST    (global 20..29)
    raise ValueError(f"Unknown donor label: {donor_label}")


def _datasets_for_donors(donor_labels):
    """Map donor labels to the dataset names DatasetBundle should load."""
    name_for = {"digits": "mnist", "fashion": "fashion_mnist",
                "emnist": "emnist_letters"}
    return [name_for[d] for d in donor_labels]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--donors", default="digits,fashion",
        help=("Comma-separated donor labels in load order. Each label "
              "maps to outputs/poc_donor_<label>.pt and to the chained-15 "
              "sub-block that produced it. Supported: digits, fashion, "
              "emnist."),
    )
    parser.add_argument(
        "--ckpt-prefix", default="outputs/poc_donor_",
        help="Path prefix for donor checkpoints (default outputs/poc_donor_).",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Soft-routing temperature (default 1.0). Lower → sharper.",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    donor_labels = [d.strip() for d in args.donors.split(",") if d.strip()]
    print("Multi-branch absorption test")
    print(f"  donors              = {donor_labels}")
    print(f"  soft routing T      = {args.temperature}")

    branches = []
    for lab in donor_labels:
        ckpt_path = f"{args.ckpt_prefix}{lab}.pt"
        b = Branch.from_checkpoint(ckpt_path, label=lab)
        branches.append(b)
        print()
        print(f"  {lab:<8} ckpt={ckpt_path}")
        print(f"    arch={list(b.net.n_nodes_per_layer())}  "
              f"classes={b.classes_covered}  "
              f"l0_seed={b.l0_seed}  arm={b.arm}")

    seeds = {b.l0_seed for b in branches}
    if len(seeds) > 1:
        print(f"\n[WARN] donors have different l0_seed: {seeds} — "
              "absorption requires shared seed.")

    org = MultiBranchOrganism.from_branches(branches)
    print(f"\nOrganism assembled with {len(branches)} branch(es).")
    print(f"  union_classes = {org.union_classes}")
    sb = org.storage_bytes()
    print(f"  storage:")
    for k, v in sb.items():
        print(f"    {k:<28} {v:>10}  ({v/1024:>8.1f} KB)")

    # Eval views over the union of all donors' chained-15 sub-blocks.
    all_specs = chained_15_specs()
    union_specs = [
        s for lab in donor_labels for s in _spec_block_for_donor(lab, all_specs)
    ]
    bundle = DatasetBundle(
        _datasets_for_donors(donor_labels),
        root=args.data_root,
        n_holdout_per_dataset=0,
    )
    eval_views = build_task_views(bundle, union_specs, split="test")

    # Donor-standalone upper bound: each donor evaluated through its OWN
    # head on its OWN tasks (no organism wrapper). This reproduces the
    # numbers from the PoC log.
    print()
    print("=" * 78)
    print("DONOR STANDALONE (upper bound — each donor on its own tasks)")
    print("=" * 78)
    donor_rows = []
    with torch.no_grad():
        for branch in branches:
            specs_subset = _spec_block_for_donor(branch.label, all_specs)
            for v in build_task_views(bundle, specs_subset, split="test"):
                x, y = v.all_examples()
                logits = branch.net(x)
                cols = list(v.global_classes)
                sub = logits[:, cols]
                pred_local = sub.argmax(dim=-1)
                pred_global = torch.tensor(
                    [cols[int(j)] for j in pred_local], dtype=torch.long,
                )
                ta = float((pred_global == y).float().mean().item())
                donor_rows.append(
                    {"task": v.name, "donor": branch.label,
                     "n": int(x.shape[0]),
                     "active": cols, "task_aware": ta}
                )
    print(f"{'task':<22}{'donor':<10}{'n':>6}  {'active':<14}{'task-aware':>12}")
    print("-" * 64)
    for r in donor_rows:
        print(f"{r['task']:<22}{r['donor']:<10}{r['n']:>6}  "
              f"{str(r['active']):<14}{r['task_aware']:>12.4f}")
    print(f"  donor-standalone mean task-aware = "
          f"{sum(r['task_aware'] for r in donor_rows)/len(donor_rows):.4f}")

    # Organism evaluation under four configurations:
    #   HARD / SOFT / UNIFORM (raw-logit combine), and SOFT+norm
    #   (per-branch log-softmax — fixes full-union calibration mismatch).
    print()
    print("=" * 78)
    print("ORGANISM (assembled multi-branch, no re-training)")
    print("=" * 78)
    rows_hard = evaluate(org, eval_views, routing="hard")
    rows_soft = evaluate(org, eval_views, routing="soft",
                         temperature=args.temperature)
    rows_unif = evaluate(org, eval_views, routing="uniform")
    rows_soft_norm = evaluate(
        org, eval_views, routing="soft",
        temperature=args.temperature, normalize_per_branch=True,
    )
    rows_hard_norm = evaluate(
        org, eval_views, routing="hard", normalize_per_branch=True,
    )
    report_block("HARD routing (argmax over branches)", rows_hard)
    report_block(f"SOFT routing (T={args.temperature})", rows_soft)
    report_block("UNIFORM gates (ablation; routing disabled)", rows_unif)
    report_block(
        f"SOFT routing + per-branch log-softmax (T={args.temperature})",
        rows_soft_norm,
    )
    report_block(
        "HARD routing + per-branch log-softmax",
        rows_hard_norm,
    )

    # Gate-firing diagnostic — under soft routing, what fraction of
    # weight does each task pull from each branch?
    print()
    print("=" * 78)
    print("GATE FIRING (soft routing — mean per-branch gate weight per task)")
    print("=" * 78)
    gates = gate_stats(org, eval_views, routing="soft",
                       temperature=args.temperature)
    branch_labels = [b.label for b in org.branches]
    print(f"{'task':<22}" + "".join(f"{lab:>12}" for lab in branch_labels))
    print("-" * (22 + 12 * len(branch_labels)))
    for r in gates:
        print(f"{r['task']:<22}" +
              "".join(f"{g:>12.4f}" for g in r["gates"]))

    # Headline summary.
    def mean_ta(rows):
        return sum(r["task_aware"] for r in rows) / len(rows)
    def mean_full(rows):
        return sum(r["full_union"] for r in rows) / len(rows)
    print()
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    donor_mean = sum(r["task_aware"] for r in donor_rows) / len(donor_rows)
    print(f"  donor-standalone mean task-aware       = {donor_mean:.4f}  "
          "(upper bound)")
    print(f"  HARD              task-aware = {mean_ta(rows_hard):.4f}  "
          f"full-union = {mean_full(rows_hard):.4f}")
    print(f"  SOFT              task-aware = {mean_ta(rows_soft):.4f}  "
          f"full-union = {mean_full(rows_soft):.4f}")
    print(f"  UNIFORM           task-aware = {mean_ta(rows_unif):.4f}  "
          f"full-union = {mean_full(rows_unif):.4f}")
    print(f"  HARD + log-softmax task-aware = "
          f"{mean_ta(rows_hard_norm):.4f}  "
          f"full-union = {mean_full(rows_hard_norm):.4f}")
    print(f"  SOFT + log-softmax task-aware = "
          f"{mean_ta(rows_soft_norm):.4f}  "
          f"full-union = {mean_full(rows_soft_norm):.4f}")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
