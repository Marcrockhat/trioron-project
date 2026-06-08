"""Dream-cycle calibration for the multi-branch organism's full-union
argmax.

Each donor's head was independently trained with cross-entropy under
its own active-class mask, so even after the per-branch log-softmax
calibration fix, absolute logit MAGNITUDES across donors aren't tightly
calibrated — full-union argmax still leaks. This script trains a tiny
per-class global bias offset (one scalar per union class — typically
~30-46 parameters total) on synthetic samples drawn from each branch's
manifold archive. Branches stay frozen; the dream cycle only learns
the bias offset.

Compatible with the "no real data after donor training" goal: the
bias offset is calibrated using each branch's stored (μ_c, σ_c)
archive, which is part of the donor checkpoint already. No retraining
of L1, no head-weight updates, no recipient-side data — just a
calibration sleep cycle on the codes the donors brought with them.

Run:
  python3 -m experiments.dream_cycle_calibration \\
      --donors digits,fashion,emnist \\
      > outputs/dream_cycle_calibration_run1.log 2>&1
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.datasets import (
    DatasetBundle, build_task_views, DEFAULT_DATA_ROOT,
)
from experiments.train_donor import SPLIT_BLOCKS
from experiments.test_multibranch_absorption import (
    _spec_block_for_donor, _datasets_for_donors, evaluate,
)
from trioron.multibranch import Branch, MultiBranchOrganism


def union_archive_sampler(branches, union_classes, class_to_union):
    """Build a callable that draws batched (z, y_union) from the union
    of all branches' archives. Each step samples a class uniformly over
    union_classes, then draws z ~ N(μ_c, σ_c²) from the branch that
    covers c."""
    # Map union class index → (branch index, mu, sigma) for fast lookup.
    per_class = {}
    for bi, b in enumerate(branches):
        for c, (mu, sg) in b.manifold_stats.items():
            per_class[c] = (bi, mu.detach().cpu(), sg.detach().cpu())
    classes = list(union_classes)
    n_classes = len(classes)
    d = next(iter(per_class.values()))[1].shape[0]

    def sample(batch, *, generator=None, noise_scale=1.0):
        choice_idx = torch.randint(0, n_classes, (batch,), generator=generator)
        zs = torch.zeros(batch, d)
        ys_union = torch.zeros(batch, dtype=torch.long)
        for i in range(batch):
            c = classes[int(choice_idx[i])]
            _, mu, sg = per_class[c]
            noise = torch.randn(d, generator=generator) * noise_scale
            zs[i] = mu + sg * noise
            ys_union[i] = class_to_union[c]
        return zs, ys_union

    return sample


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donors", default="digits,fashion,emnist")
    parser.add_argument("--ckpt-prefix", default="outputs/poc_donor_")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="multiplier on σ when sampling from archive")
    parser.add_argument(
        "--bias-mode", choices=["per_class", "per_branch"],
        default="per_class",
        help=("per_class: one bias per union class (~|union| params, can "
              "perturb within-task argmax). per_branch: one bias per "
              "branch broadcast to its covered slots (=N params, "
              "mathematically cannot affect task-aware since active "
              "classes within a task share a single branch — pure "
              "cross-donor calibration knob)."),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    labels = [d.strip() for d in args.donors.split(",") if d.strip()]
    print(f"Dream-cycle calibration — donors={labels}")
    print(f"  steps={args.steps} batch={args.batch} lr={args.lr} "
          f"noise_scale={args.noise_scale}")

    branches = []
    for lab in labels:
        ckpt = f"{args.ckpt_prefix}{lab}.pt"
        b = Branch.from_checkpoint(ckpt, label=lab)
        branches.append(b)
        print(f"  loaded {lab:<10} arch={list(b.net.n_nodes_per_layer())}  "
              f"classes={b.classes_covered}")

    org = MultiBranchOrganism.from_branches(branches)
    n_union = len(org.union_classes)
    print(f"  union_classes={org.union_classes}")
    print(f"  n_union={n_union}")

    # Eval views over the union — for before/after measurement.
    bundle = DatasetBundle(
        _datasets_for_donors(labels), root=args.data_root,
        n_holdout_per_dataset=0,
    )
    eval_views = []
    for b in branches:
        eval_views.extend(
            build_task_views(bundle, _spec_block_for_donor(b.label), split="test")
        )

    def headline(rows, tag):
        ta = sum(r["task_aware"] for r in rows) / len(rows)
        fu = sum(r["full_union"] for r in rows) / len(rows)
        print(f"  {tag:<28} task-aware={ta:.4f}  full-union={fu:.4f}")
        return ta, fu

    print("\n" + "=" * 78)
    print("BEFORE dream-cycle (per-branch log-softmax only)")
    print("=" * 78)
    rows_before = evaluate(
        org, eval_views, routing="soft",
        temperature=args.temperature, normalize_per_branch=True,
    )
    ta_before, fu_before = headline(rows_before, "soft + per-branch log-softmax")

    # Dream cycle — train bias offset (per_class or per_branch).
    torch.manual_seed(args.seed)
    if args.bias_mode == "per_class":
        bias_param = nn.Parameter(torch.zeros(n_union))
        n_params = n_union
    else:
        bias_param = nn.Parameter(torch.zeros(len(branches)))
        n_params = len(branches)
    opt = torch.optim.Adam([bias_param], lr=args.lr)
    sample = union_archive_sampler(
        branches, org.union_classes, org._class_to_union,
    )
    gen = torch.Generator().manual_seed(args.seed + 1)

    # Build a (n_union,) broadcaster for per_branch mode that maps each
    # union slot to its branch's scalar bias. Rebuilt every step from
    # bias_param so gradients flow.
    if args.bias_mode == "per_branch":
        # union_slot → branch_index lookup
        slot_to_branch = torch.zeros(n_union, dtype=torch.long)
        for bi, b in enumerate(branches):
            for c in b.classes_covered:
                slot_to_branch[org._class_to_union[c]] = bi

        def materialize_bias():
            return bias_param[slot_to_branch]
    else:
        def materialize_bias():
            return bias_param

    print("\n" + "=" * 78)
    print(f"DREAM CYCLE  bias_mode={args.bias_mode}  "
          f"({args.steps} steps, params={n_params})")
    print("=" * 78)
    log_every = max(1, args.steps // 10)
    for step in range(args.steps):
        z, y = sample(args.batch, generator=gen, noise_scale=args.noise_scale)
        # Branches frozen — no_grad through them; gradient flows only
        # through the bias parameter via the additive offset.
        with torch.no_grad():
            base = org.forward_from_z(
                z, routing="soft",
                temperature=args.temperature,
                normalize_per_branch=True,
            )
        bias_vec = materialize_bias()
        logits = base + bias_vec
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == y).float().mean().item()
            print(f"  step {step+1:4d}/{args.steps}  "
                  f"loss {loss.item():.4f}  archive_acc {acc:.4f}  "
                  f"||bias||={bias_param.detach().norm().item():.3f}  "
                  f"max|bias|={bias_param.detach().abs().max().item():.3f}")

    # AFTER eval — same evaluate path but with bias_offset wired through
    # via a thin wrapper that mimics the organism's forward signature.
    class _OrgWithBias(nn.Module):
        def __init__(self, org, bias):
            super().__init__()
            self.org = org
            self.bias = bias

        @property
        def union_classes(self):
            return self.org.union_classes

        @property
        def branches(self):
            return self.org.branches

        def storage_bytes(self):
            sb = self.org.storage_bytes()
            extra = self.bias.numel() * self.bias.element_size()
            return {**sb, "calibration_bias_bytes": extra,
                    "total_bytes": sb["total_bytes"] + extra}

        def forward(self, x, *, routing="soft", temperature=1.0,
                    normalize_per_branch=False, return_extras=False):
            return self.org(
                x, routing=routing, temperature=temperature,
                normalize_per_branch=normalize_per_branch,
                bias_offset=self.bias,
                return_extras=return_extras,
            )

    final_bias = materialize_bias().detach()
    org_cal = _OrgWithBias(org, final_bias)
    org_cal.eval()
    print("\n" + "=" * 78)
    print("AFTER dream-cycle (per-branch log-softmax + global bias)")
    print("=" * 78)
    rows_after = evaluate(
        org_cal, eval_views, routing="soft",
        temperature=args.temperature, normalize_per_branch=True,
    )
    ta_after, fu_after = headline(rows_after, "soft + log-softmax + bias")

    # Per-task delta.
    print("\nPer-task full-union delta (before → after):")
    print(f"  {'task':<24}{'before':>10}{'after':>10}{'Δ':>10}")
    for rb, ra in zip(rows_before, rows_after):
        delta = ra["full_union"] - rb["full_union"]
        print(f"  {rb['task']:<24}"
              f"{rb['full_union']:>10.4f}{ra['full_union']:>10.4f}"
              f"{delta:>+10.4f}")

    # Bias inspection.
    print("\nLearned bias offsets:")
    if args.bias_mode == "per_class":
        for ui, c in enumerate(org.union_classes):
            print(f"  class {c:>3} → bias {bias_param[ui].item():+.4f}")
    else:
        for bi, b in enumerate(branches):
            print(f"  branch {b.label:<10} → bias {bias_param[bi].item():+.4f}")

    # Storage delta.
    print()
    sb = org.storage_bytes()
    extra = bias_param.numel() * bias_param.element_size()
    print(f"Calibration storage: bias_param {bias_param.numel()} floats × "
          f"{bias_param.element_size()} B = {extra} B "
          f"(<{(extra / (sb['total_bytes']) * 100):.3f}% of organism)")

    print()
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"  before  task-aware={ta_before:.4f}  full-union={fu_before:.4f}")
    print(f"  after   task-aware={ta_after:.4f}  full-union={fu_after:.4f}  "
          f"(Δfull = {fu_after - fu_before:+.4f}, "
          f"Δta = {ta_after - ta_before:+.4f})")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
