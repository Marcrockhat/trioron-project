"""Trioron command-line entry point.

Wraps the production path of trioron — train a donor, absorb donors
into a multi-branch organism, run inference / eval — into four
subcommands. No experimental knobs are exposed here; the underlying
research scripts (`experiments/*.py`) remain available for deeper
reproduction. Reviewers should be able to reproduce the lossless
absorption result on commodity CPU in <5 minutes via:

    pip install -e .                  # or pip install git+https://...
    trioron train  --donor digits   --out donor_digits.pt
    trioron train  --donor fashion  --out donor_fashion.pt
    trioron absorb --donors donor_digits.pt,donor_fashion.pt --out organism.pt
    trioron eval   --organism organism.pt
    trioron infer  --organism organism.pt --image path/to/image.png

All four subcommands are deterministic given the L0 seed (default 42)
and produce the same accuracy numbers reported in the paper at the
2-donor configuration.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List, Optional, Sequence

import torch


# ---------------------------------------------------------------------
# train
# ---------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    """Train one trioron donor on a chained-15 sub-block."""
    from experiments import train_donor as td
    if args.donor not in td.SPLIT_BLOCKS:
        print(f"error: unknown donor split '{args.donor}'. "
              f"choices: {sorted(td.SPLIT_BLOCKS)}",
              file=sys.stderr)
        return 2
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    sub_argv: List[str] = [
        "--label", args.donor,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--out-dir", out_dir,
    ]
    if args.data_root:
        sub_argv += ["--data-root", args.data_root]
    rc = td.main(sub_argv)
    if rc != 0:
        return rc
    src = os.path.join(out_dir, f"poc_donor_{args.donor}.pt")
    if os.path.abspath(src) != os.path.abspath(args.out):
        os.replace(src, args.out)
    print(f"\n[trioron train] saved donor → {args.out}")
    return 0


# ---------------------------------------------------------------------
# absorb
# ---------------------------------------------------------------------


def cmd_absorb(args: argparse.Namespace) -> int:
    """Assemble a multi-branch organism from saved donors."""
    from trioron.multibranch import Branch, MultiBranchOrganism
    paths = [p.strip() for p in args.donors.split(",") if p.strip()]
    if len(paths) < 1:
        print("error: --donors is empty", file=sys.stderr)
        return 2
    branches = []
    for p in paths:
        if not os.path.exists(p):
            print(f"error: donor checkpoint not found: {p}", file=sys.stderr)
            return 2
        b = Branch.from_checkpoint(p)
        branches.append(b)
        print(f"  loaded {b.label:<12} arch={list(b.net.n_nodes_per_layer())}  "
              f"classes={b.classes_covered}  "
              f"l0_seed={b.l0_seed}")
    seeds = {b.l0_seed for b in branches}
    if len(seeds) > 1:
        print(f"error: donors have mismatched L0 seeds {seeds} — "
              "shared-seed invariant is required for paste-and-go absorption.",
              file=sys.stderr)
        return 2
    org = MultiBranchOrganism.from_branches(branches)
    payload = {
        "version": 1,
        "kind": "multibranch_organism",
        "l0_seed": next(iter(seeds)),
        "l0_W": org.l0_W.detach().cpu(),
        "l0_b": org.l0_b.detach().cpu(),
        "l0_activation": org.l0_activation,
        "branches": [
            {
                "label": b.label,
                "classes_covered": list(b.classes_covered),
                "arm": b.arm,
                "l0_seed": b.l0_seed,
                "n_nodes_per_layer": list(b.net.n_nodes_per_layer()),
                "input_dim": b.net.layers[0].fan_in,
                "state_dict": {k: v.detach().cpu()
                               for k, v in b.net.state_dict().items()},
                "manifold_stats": {
                    int(c): (mu.detach().cpu(), sg.detach().cpu())
                    for c, (mu, sg) in b.manifold_stats.items()
                },
            }
            for b in branches
        ],
        "union_classes": list(org.union_classes),
    }
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, args.out)
    sb = org.storage_bytes()
    print(f"\n[trioron absorb] organism with {len(branches)} branch(es) → {args.out}")
    print(f"  union_classes = {org.union_classes}")
    print(f"  storage: {sb['total_bytes'] / 1024:.1f} KB total "
          f"(L0 {sb['l0_bytes']/1024:.0f} KB shared, "
          f"branch substrate {sb['branch_substrate_bytes']/1024:.0f} KB, "
          f"archive {sb['archive_bytes']/1024:.0f} KB)")
    return 0


# ---------------------------------------------------------------------
# Helper: rebuild an organism from a saved organism payload
# ---------------------------------------------------------------------


def _load_organism(path: str):
    """Reconstruct a MultiBranchOrganism from a `trioron absorb`
    checkpoint OR (for convenience) from a legacy single-donor
    poc_donor_*.pt — that becomes a 1-branch organism."""
    from trioron.multibranch import Branch, MultiBranchOrganism
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") == "multibranch_organism":
        # Rebuild branches inline (skips the per-branch checkpoint files).
        from trioron.network import TrioronNetwork
        branches = []
        for d in payload["branches"]:
            n_nodes = d["n_nodes_per_layer"]
            specs = []
            prev = d["input_dim"]
            for i, n in enumerate(n_nodes):
                act = "linear" if i == len(n_nodes) - 1 else "relu"
                specs.append((prev, n, act))
                prev = n
            net = TrioronNetwork(specs)
            net.load_state_dict(d["state_dict"])
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            branches.append(Branch(
                label=d["label"], classes_covered=d["classes_covered"],
                net=net, manifold_stats=d["manifold_stats"],
                l0_seed=d.get("l0_seed"), arm=d.get("arm"),
            ))
        return MultiBranchOrganism.from_branches(branches)
    if "manifold_stats" in payload and "state_dict" in payload:
        # Legacy single-donor: wrap as a 1-branch organism.
        b = Branch.from_checkpoint(path)
        return MultiBranchOrganism.from_branches([b])
    raise ValueError(
        f"file at {path} is not a recognized trioron checkpoint "
        "(missing 'kind' or 'manifold_stats')"
    )


# ---------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate an organism on the union test set of its donors' tasks."""
    from experiments.datasets import (
        DatasetBundle, build_task_views, DEFAULT_DATA_ROOT,
    )
    from experiments.train_donor import SPLIT_BLOCKS
    from experiments.test_multibranch_absorption import (
        evaluate as eval_views_,
    )
    org = _load_organism(args.organism)
    print(f"[trioron eval] loaded organism with {len(org.branches)} branch(es)")
    print(f"  branches      = {[b.label for b in org.branches]}")
    print(f"  union_classes = {org.union_classes}")

    bundle_dataset_names = []
    union_specs = []
    for b in org.branches:
        if b.label not in SPLIT_BLOCKS:
            print(f"error: branch label '{b.label}' is not in the trained "
                  f"split registry; eval needs the matching test split.",
                  file=sys.stderr)
            return 2
        specs_fn, ds_name = SPLIT_BLOCKS[b.label]
        if ds_name not in bundle_dataset_names:
            bundle_dataset_names.append(ds_name)
        union_specs.extend(specs_fn())
    bundle = DatasetBundle(
        bundle_dataset_names,
        root=args.data_root or DEFAULT_DATA_ROOT,
        n_holdout_per_dataset=0,
    )
    eval_views = build_task_views(bundle, union_specs, split="test")

    rows_norm = eval_views_(
        org, eval_views, routing="soft",
        temperature=args.temperature, normalize_per_branch=True,
    )
    rows_raw = eval_views_(
        org, eval_views, routing="soft",
        temperature=args.temperature, normalize_per_branch=False,
    )
    n = len(rows_norm)
    ta_norm = sum(r["task_aware"] for r in rows_norm) / n
    fu_norm = sum(r["full_union"] for r in rows_norm) / n
    ta_raw = sum(r["task_aware"] for r in rows_raw) / n
    fu_raw = sum(r["full_union"] for r in rows_raw) / n

    print()
    print("Per-task accuracy (soft routing, per-branch log-softmax):")
    print(f"  {'task':<24}{'n':>6}  {'active':<14}"
          f"{'task-aware':>12}{'full-union':>12}")
    print("  " + "-" * 64)
    for r in rows_norm:
        print(f"  {r['task']:<24}{r['n']:>6}  {str(r['active']):<14}"
              f"{r['task_aware']:>12.4f}{r['full_union']:>12.4f}")
    print()
    print("Headline (mean across union):")
    print(f"  task-aware (production) = {ta_norm:.4f}  "
          f"full-union = {fu_norm:.4f}  "
          "← soft + per-branch log-softmax")
    print(f"  task-aware (raw)        = {ta_raw:.4f}  "
          f"full-union = {fu_raw:.4f}  "
          "← soft, no normalization")
    return 0


# ---------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------


def _load_image_as_tensor(path: str) -> torch.Tensor:
    """Load an image and convert to the 28x28 grayscale flattened
    tensor the chained-15 organism expects. Greyscale-MNIST shape =
    (1, 784) float in [0, 1]."""
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert("L")
    pre = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # (1, 28, 28) in [0, 1]
    ])
    x = pre(img)
    return x.view(1, -1)        # (1, 784)


def cmd_infer(args: argparse.Namespace) -> int:
    """Single-image inference. Reports top-k predictions from the union
    softmax (per-branch log-softmax composition)."""
    org = _load_organism(args.organism)
    x = _load_image_as_tensor(args.image)
    with torch.no_grad():
        logits, extras = org(
            x, routing="soft",
            temperature=args.temperature,
            normalize_per_branch=True,
            return_extras=True,
        )
    probs = torch.softmax(logits[0], dim=-1)
    topk = torch.topk(probs, k=min(args.topk, probs.numel()))
    union = org.union_classes
    print(f"[trioron infer] image={args.image}")
    print(f"  branches      = {[b.label for b in org.branches]}")
    print(f"  union_classes = {union}")
    print(f"  routing gates = {extras['gates'][0].tolist()}")
    print()
    print(f"Top-{topk.values.numel()} predictions:")
    for p, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        print(f"  class {union[int(idx)]:>3}  prob {p:.4f}")
    return 0


# ---------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="trioron",
        description=(
            "Continual-learning architecture with archive-routed "
            "multi-branch absorption."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    p_train = sub.add_parser("train", help="train one donor on a "
                                            "chained-15 sub-block")
    p_train.add_argument(
        "--donor", required=True,
        choices=["digits", "fashion", "emnist", "emnist_kt", "emnist_uz"],
        help="which sub-block to train on (matches paper §4.6 splits)",
    )
    p_train.add_argument("--seed", type=int, default=42,
                         help="shared L0 seed (default 42, must match "
                              "across all donors)")
    p_train.add_argument("--epochs", type=int, default=8,
                         help="epochs per chained-15 task (default 8)")
    p_train.add_argument("--data-root", default=None,
                         help="dataset cache directory (default: "
                              "<repo>/outputs/data)")
    p_train.add_argument("--out", required=True,
                         help="output donor checkpoint path")
    p_train.set_defaults(func=cmd_train)

    p_absorb = sub.add_parser("absorb",
                              help="assemble a multi-branch organism "
                                   "from saved donors (zero-shot)")
    p_absorb.add_argument("--donors", required=True,
                          help="comma-separated donor checkpoint paths")
    p_absorb.add_argument("--out", required=True,
                          help="output organism checkpoint path")
    p_absorb.set_defaults(func=cmd_absorb)

    p_eval = sub.add_parser("eval",
                            help="evaluate an organism on the union "
                                 "test set of its donors' tasks")
    p_eval.add_argument("--organism", required=True,
                        help="organism checkpoint path (or a single donor "
                             ".pt — wraps as a 1-branch organism)")
    p_eval.add_argument("--temperature", type=float, default=1.0,
                        help="soft-routing temperature (default 1.0)")
    p_eval.add_argument("--data-root", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_infer = sub.add_parser("infer",
                             help="run single-image inference through "
                                  "an organism")
    p_infer.add_argument("--organism", required=True)
    p_infer.add_argument("--image", required=True,
                         help="path to a 28x28-resizable grayscale image")
    p_infer.add_argument("--topk", type=int, default=5)
    p_infer.add_argument("--temperature", type=float, default=1.0)
    p_infer.set_defaults(func=cmd_infer)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
