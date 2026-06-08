"""Committee evaluation over the K-seed donor population.

Loads K donors from outputs/seed_population_emnist_kt/seed_*/ and aggregates
their predictions on the EMNIST K-T eval set. Each donor is run independently
through its own L0 (publish-W_L0 / Path E from exp3_cross_seed_absorption — the
simplest lossless path; no closed-form translator needed). Logits are then
combined across donors via three aggregation rules:

    mean-logit:   logits_avg = mean_d(logits_d)               then argmax
    mean-softmax: probs_avg  = mean_d(softmax(logits_d))      then argmax
    hard-vote:    pred       = mode_d(argmax(logits_d))       (ties → mean-logit)

For each rule we report task-aware / full / domain — the same three concentric
metrics evaluate_all_tasks() returns. Baselines:

    K=1 (seed=42)               — single best donor (population max = 0.9495)
    K=3 (top-3 by task-aware)   — selection committee
    K=5 (top-5 by task-aware)   — broader committee
    K=10 (all)                  — full population

Compares against the σ=0.0033 weak-signal regime in seed_population_report:
does committee aggregation actually beat selection?

Run:
    python3 -m experiments.committee_eval \\
        > outputs/seed_population_emnist_kt/committee_run1.log 2>&1
"""
from __future__ import annotations
import argparse
import glob
import os
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from experiments.datasets import (
    DatasetBundle, build_task_views, DEFAULT_DATA_ROOT,
)
from experiments.train_donor import SPLIT_BLOCKS
from experiments.bench_chained_15task import (
    _quantize_archived_in_place, _round_active_to_bf16_in_place,
)
from trioron.classification import accuracy
from trioron.multibranch import Branch


def dream_compress(branch: Branch) -> None:
    """Apply Phase 1+2 dream-archive compression to this donor in place:
        * round non-archived W and ALL biases to BF16 (cast back to FP32
          storage so F.linear/eval continues to work);
        * snap any archived rows to int8 with per-row symmetric scale.

    Reuses the bench helpers so accuracy here matches the
    bench_chained_extend deployment path. For donors with no archived
    rows (typical when training stops before any merge/purge fires —
    `cum_purges=0` in this 10-seed population), the int8 step is a no-op.
    """
    _round_active_to_bf16_in_place(branch.net)
    _quantize_archived_in_place(branch.net, mode="int8")


def deployment_bytes_for_branch(
    branch: Branch, *, bf16: bool, share_l0: bool,
) -> Dict[str, float]:
    """Bytes required to deploy this single branch — counting ONLY W+b
    (training scaffolding stripped) and the archive.

    bf16=True   active weights + biases at 2 B/value; archived rows at
                1 B/value + 4 B per-row scale.
    bf16=False  full FP32 (4 B/value).
    share_l0=True drops the L0 contribution (it's protocol-shared across
                  donors under factored L0; this is the "smart" path).
    """
    base_w_bytes = 2 if bf16 else 4
    base_b_bytes = base_w_bytes
    layers = list(branch.net.layers)
    out = {"l0": 0.0, "l1_head": 0.0, "archive": 0.0}
    for L, layer in enumerate(layers):
        n_arch = int(layer.archived.sum().item()) if hasattr(layer, "archived") else 0
        n_active = layer.n_nodes - n_arch
        bytes_W = (
            n_active * layer.fan_in * base_w_bytes
            + n_arch * (layer.fan_in * 1 + 4)            # int8 + per-row scale
        )
        bytes_b = layer.n_nodes * base_b_bytes
        layer_bytes = bytes_W + bytes_b
        if L == 0 and share_l0:
            continue                                     # protocol-shared S
        if L == 0:
            out["l0"] += layer_bytes
        else:
            out["l1_head"] += layer_bytes
    for (mu, sg) in branch.manifold_stats.values():
        # Archive stats — keep at FP32 (they're (μ, σ) pairs, small).
        out["archive"] += (mu.numel() + sg.numel()) * mu.element_size()
    out["total"] = out["l0"] + out["l1_head"] + out["archive"]
    return out


def load_population(root: str, label: str = "emnist_kt") -> List[Tuple[int, Branch, float]]:
    """Yield (seed, branch, train_task_aware) for every donor in root."""
    pattern = os.path.join(root, "seed_*", f"poc_donor_{label}.pt")
    out: List[Tuple[int, Branch, float]] = []
    for path in sorted(glob.glob(pattern)):
        seed_dir = os.path.basename(os.path.dirname(path))
        seed = int(seed_dir.split("_")[-1])
        # Branch.from_checkpoint reconstructs the full TrioronNetwork (incl L0)
        # and freezes it.
        b = Branch.from_checkpoint(path, label=f"seed_{seed}")
        # Pull the train-time task-aware score from the payload too — we use
        # it as the selection signal for top-K committees.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        ta = float(payload["final_accuracy_aware"])
        out.append((seed, b, ta))
    return out


# ---------------------------------------------------------------------
# Per-donor forward — full path (own L0 → own L1 → own head).
# ---------------------------------------------------------------------


@torch.no_grad()
def donor_forward(branch: Branch, x: torch.Tensor) -> torch.Tensor:
    """Full forward through this donor: input → its own L0 → L1 → head.

    Equivalent to Path E (publish-W_L0) in exp3 — each branch consumes
    raw x, runs its own L0 matmul. Lossless (no information bottleneck);
    pays N L0 matmuls per batch when the committee has N donors.
    """
    x_dev = x.to(branch.l0_W().device, dtype=branch.l0_W().dtype)
    return branch.net(x_dev)


# ---------------------------------------------------------------------
# Aggregation rules.
# ---------------------------------------------------------------------


def aggregate(
    per_donor_logits: List[torch.Tensor],
    rule: str,
) -> torch.Tensor:
    """Combine (B, head_size) logit tensors across K donors.

    mean-logit:   stack and mean. Returns logits.
    mean-softmax: softmax each, mean, log. Returns log-probs (argmax-equivalent
                  to the prob average; same scale as logits for downstream
                  restrict_to logic).
    hard-vote:    argmax per donor; majority vote per row. Returns one-hot
                  tensor compatible with argmax (winner gets +inf, others 0).
                  Ties broken by mean-logit fallback.
    """
    K = len(per_donor_logits)
    stacked = torch.stack(per_donor_logits, dim=0)  # (K, B, H)
    if rule == "mean-logit":
        return stacked.mean(dim=0)
    if rule == "mean-softmax":
        probs = F.softmax(stacked, dim=-1).mean(dim=0)
        return probs.clamp_min(1e-30).log()
    if rule == "hard-vote":
        preds = stacked.argmax(dim=-1)                                    # (K, B)
        B = preds.shape[1]; H = stacked.shape[-1]
        votes = torch.zeros(B, H, device=stacked.device, dtype=stacked.dtype)
        for k in range(K):
            votes.scatter_add_(
                1, preds[k].unsqueeze(1),
                torch.ones(B, 1, device=stacked.device, dtype=stacked.dtype),
            )
        # Tie-break by mean-logit (add a tiny ε * mean-logit so ties resolve
        # toward the average prediction without changing strict majorities).
        tiebreak = stacked.mean(dim=0)
        return votes + 1e-6 * tiebreak
    raise ValueError(f"Unknown aggregation rule: {rule!r}")


# ---------------------------------------------------------------------
# Eval — task-aware / full / domain across all eval views.
# ---------------------------------------------------------------------


@torch.no_grad()
def committee_eval(
    branches: Sequence[Branch],
    eval_views,
    task_class_lists: Sequence[Sequence[int]],
    *,
    rule: str,
) -> Dict[str, float]:
    """Run a committee of `branches` on `eval_views`. Returns mean
    (over views) of full / task-aware / domain accuracy."""
    full_accs, aware_accs, domain_accs = [], [], []
    for i, v in enumerate(eval_views):
        x, y = v.all_examples()
        per_donor = [donor_forward(b, x) for b in branches]
        # Pad each donor's head to the max width — they should all be 40 in
        # this population, but be defensive in case a donor differed.
        max_h = max(l.shape[-1] for l in per_donor)
        per_donor_padded = []
        for l in per_donor:
            if l.shape[-1] < max_h:
                pad = torch.full(
                    (l.shape[0], max_h - l.shape[-1]),
                    float("-inf"), dtype=l.dtype, device=l.device,
                )
                l = torch.cat([l, pad], dim=-1)
            per_donor_padded.append(l)
        combined = aggregate(per_donor_padded, rule=rule)
        head_size = combined.shape[1]

        full_accs.append(accuracy(combined, y))
        active = task_class_lists[i]
        if max(active) < head_size:
            aware_accs.append(accuracy(combined, y, restrict_to=active))
        # Domain: emnist_kt → classes 30..39 (start_class_offset=30).
        domain_idx = active[0] // 10
        domain_full = list(range(domain_idx * 10, (domain_idx + 1) * 10))
        domain_avail = [c for c in domain_full if c < head_size]
        if domain_avail:
            domain_accs.append(accuracy(combined, y, restrict_to=domain_avail))

    def m(xs): return float(sum(xs) / len(xs)) if xs else float("nan")
    return {
        "full": m(full_accs),
        "task_aware": m(aware_accs),
        "domain": m(domain_accs),
        "n_views": len(eval_views),
        "n_branches": len(branches),
    }


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/seed_population_emnist_kt")
    p.add_argument("--label", default="emnist_kt")
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument(
        "--committee-sizes", type=int, nargs="+", default=[1, 3, 5, 10],
        help="K values to run; donors are picked top-K by train-time task-aware.",
    )
    p.add_argument(
        "--dream", action="store_true",
        help="Apply Phase 1+2 dream-archive compression to each donor "
             "(bf16 active + int8 archived) before evaluation.",
    )
    args = p.parse_args(argv)

    pop = load_population(args.root, args.label)
    if args.dream:
        print(f"[dream] applying Phase 1+2 (bf16 active + int8 archived) "
              f"to {len(pop)} donors")
        for (_, b, _) in pop:
            dream_compress(b)
        print()
    if not pop:
        print(f"No donors found under {args.root}")
        return 1
    # Sort by train-time task-aware (descending) so top-K is well defined.
    pop_sorted = sorted(pop, key=lambda t: -t[2])
    print(f"Loaded {len(pop_sorted)} donors. Train-time task-aware ranking:")
    print(f"  {'rank':>4}  {'seed':>5}  {'task-aware':>11}")
    for r, (seed, _, ta) in enumerate(pop_sorted):
        print(f"  {r+1:>4}  {seed:>5}  {ta:>11.4f}")
    print()

    # Build eval views once (shared across committees).
    specs_fn, ds_name = SPLIT_BLOCKS[args.label]
    specs = specs_fn()
    bundle = DatasetBundle([ds_name], root=args.data_root)
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]
    n_eval = sum(v.n_examples() for v in eval_views)
    print(f"Eval set: {len(eval_views)} views, {n_eval} total examples "
          f"(label={args.label})")
    print()

    rules = ["mean-logit", "mean-softmax", "hard-vote"]

    print(f"{'rule':<14}  {'K':>3}  {'seeds':>20}  "
          f"{'task-aware':>11}  {'full':>8}  {'domain':>8}")
    print("-" * 80)
    rows: Dict[Tuple[str, int], Dict[str, float]] = {}
    for rule in rules:
        for K in args.committee_sizes:
            picks = pop_sorted[:K]
            seeds = [s for (s, _, _) in picks]
            branches = [b for (_, b, _) in picks]
            r = committee_eval(
                branches, eval_views, task_class_lists, rule=rule,
            )
            rows[(rule, K)] = r
            print(f"{rule:<14}  {K:>3}  {str(seeds):>20}  "
                  f"{r['task_aware']:>11.4f}  {r['full']:>8.4f}  "
                  f"{r['domain']:>8.4f}")
        print()

    # Selection vs committee summary using mean-logit (the simplest rule).
    print("=" * 80)
    print("HEADLINE — selection vs committee  (rule = mean-logit)")
    print("=" * 80)
    pop_max_train = pop_sorted[0][2]
    pop_mean_train = sum(t[2] for t in pop_sorted) / len(pop_sorted)
    K1 = rows[("mean-logit", 1)]["task_aware"]
    K3 = rows[("mean-logit", 3)]["task_aware"] if 3 in args.committee_sizes else None
    K10 = rows[("mean-logit", 10)]["task_aware"] if 10 in args.committee_sizes else None
    print(f"  Population (train-time):  mean = {pop_mean_train:.4f}  "
          f"max = {pop_max_train:.4f}")
    print(f"  K=1   (top-1 selection)    eval task-aware = {K1:.4f}")
    if K3 is not None:
        print(f"  K=3   (top-3 committee)    eval task-aware = {K3:.4f}  "
              f"(Δ vs K=1 = {K3-K1:+.4f})")
    if K10 is not None:
        print(f"  K=10  (full committee)     eval task-aware = {K10:.4f}  "
              f"(Δ vs K=1 = {K10-K1:+.4f})")
    print()

    # Deployment byte report — count only W+b (scaffolding stripped) plus
    # archive. "Smart" assumes factored-L0 deployment shares S across donors.
    bf16 = args.dream
    print("=" * 80)
    print(f"DEPLOYMENT BYTES  (training scaffolding stripped, "
          f"{'bf16 active + int8 archived' if bf16 else 'fp32'})")
    print("=" * 80)
    print(f"  per-donor breakdown (avg over {len(pop_sorted)} donors):")
    sums = {"l0": 0.0, "l1_head": 0.0, "archive": 0.0, "total": 0.0}
    sums_smart = {"l0": 0.0, "l1_head": 0.0, "archive": 0.0, "total": 0.0}
    for (_, b, _) in pop_sorted:
        d = deployment_bytes_for_branch(b, bf16=bf16, share_l0=False)
        ds = deployment_bytes_for_branch(b, bf16=bf16, share_l0=True)
        for k in sums: sums[k] += d[k]
        for k in sums_smart: sums_smart[k] += ds[k]
    n = len(pop_sorted)
    print(f"    L0      = {sums['l0']/n/1024:>7.2f} KiB  ({'protocol-shared under factored L0' if True else ''})")
    print(f"    L1+head = {sums['l1_head']/n/1024:>7.2f} KiB")
    print(f"    archive = {sums['archive']/n/1024:>7.2f} KiB")
    print(f"    total   = {sums['total']/n/1024:>7.2f} KiB")
    print()
    # Per-donor "smart" cost (excluding L0): everything except the shared S.
    per_donor_smart = sums_smart['total'] / n     # already excludes L0
    one_l0 = (sums['l0'] - sums_smart['l0']) / n  # the L0 share that's collapsed
    print(f"  factored-L0 deployment (ship S once + per-donor R seed + "
          f"L1+head+archive):")
    print(f"    shared S         = {one_l0/1024:>8.2f} KiB    "
          f"(one copy across all donors)")
    print(f"    per-donor R seed = {4:>8d} B")
    print(f"    per-donor pack   = {per_donor_smart/1024:>8.2f} KiB    "
          f"(L1+head+archive)")
    print()
    print(f"  {'K':>3}  {'naive (KiB)':>14}  {'smart (KiB)':>14}  "
          f"{'smart/naive':>13}")
    for K in args.committee_sizes:
        per_donor_full = sums['total'] / n
        naive = K * per_donor_full / 1024.0
        smart = (one_l0 + K * (per_donor_smart + 4)) / 1024.0
        print(f"  {K:>3}  {naive:>14.2f}  {smart:>14.2f}  "
              f"{smart/naive:>13.2f}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
