"""Branch-Train-Merge baselines for the multi-branch absorption test.

Two BTM variants on the same 3-donor setup (digits + fashion + emnist),
held against the trioron archive-routed organism:

  BTM-avg     — average L1 substrate across donors (entry-wise),
                concatenate per-class head columns over the union.
                The L1 averaging is the destructive step: each averaged
                hidden unit is the mean of three different feature
                detectors, so none of the donors' specialty features
                survive intact. Zero-shot but lossy.
                Concretely the same flavor as Wortsman et al.'s
                Model Soup / Branch-Train-Merge param-averaging variant.

  BTM-MoE     — frozen donor branches + a learned router (small MLP
                from L0 → branch index). Router trained on (L0(x),
                branch_id) pairs sampled from each donor's training
                data. Substitutes our archive-likelihood gate with a
                learned classifier router. NOT zero-shot — requires
                a router-training pass — but should be the strongest
                BTM variant.

Reports task-aware and full-union per task across:
  - donor-standalone (upper bound)
  - ours: archive-routed, soft + log-softmax (zero-shot)
  - BTM-avg (zero-shot, destructive)
  - BTM-MoE (router trained, frozen branches)

Run:
  python3 -m experiments.bench_btm_baseline \\
      > outputs/bench_btm_baseline_run1.log 2>&1
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
    task_aware_accuracy, full_softmax_accuracy,
)
from trioron.multibranch import Branch, MultiBranchOrganism


# ---------------------------------------------------------------------
# BTM-avg — parameter averaging
# ---------------------------------------------------------------------


def build_btm_avg_organism(branches):
    """Average each donor's L1 weights/biases entry-wise and assemble a
    single 'soup' net + concatenated head over the union of classes.

    Returns an object that mimics MultiBranchOrganism's `forward(x)` API
    so the same evaluate() helper works on it.
    """
    # All donors share the same L0 (validated by MultiBranchOrganism on
    # build) and the same L1 width (52). We use any donor's L0 directly
    # and average L1 across donors.
    l0_W = branches[0].l0_W().clone()
    l0_b = branches[0].l0_b().clone()

    # L1 = average across donors.
    l1_Ws = torch.stack([b.net.layers[1].W.detach() for b in branches], dim=0)
    l1_bs = torch.stack([b.net.layers[1].b.detach() for b in branches], dim=0)
    l1_W = l1_Ws.mean(dim=0)   # (52, 128)
    l1_b = l1_bs.mean(dim=0)   # (52,)

    # Head columns concatenated by class id over the union (disjoint
    # coverage → each class supplied by exactly one donor).
    union = []
    for b in branches:
        for c in b.classes_covered:
            union.append((c, b))
    union.sort(key=lambda r: r[0])
    union_classes = [c for c, _ in union]
    head_W_rows = []
    head_b_rows = []
    for c, b in union:
        head_W_rows.append(b.net.layers[2].W.detach()[c])
        head_b_rows.append(b.net.layers[2].b.detach()[c])
    head_W = torch.stack(head_W_rows, dim=0)   # (|union|, 52)
    head_b = torch.stack(head_b_rows, dim=0)   # (|union|,)

    class _BTMAvg(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("l0_W", l0_W)
            self.register_buffer("l0_b", l0_b)
            self.register_buffer("l1_W", l1_W)
            self.register_buffer("l1_b", l1_b)
            self.register_buffer("head_W", head_W)
            self.register_buffer("head_b", head_b)
            self.union_classes = union_classes

        def forward(self, x, *, routing="soft", temperature=1.0,
                    normalize_per_branch=False, return_extras=False):
            del routing, temperature, normalize_per_branch
            z = F.relu(F.linear(x, self.l0_W, self.l0_b))
            h = F.relu(F.linear(z, self.l1_W, self.l1_b))
            o = F.linear(h, self.head_W, self.head_b)
            if return_extras:
                return o, {"z": z, "gates": None, "branch_logits_padded": None}
            return o

    return _BTMAvg()


# ---------------------------------------------------------------------
# BTM-MoE — learned router
# ---------------------------------------------------------------------


class LearnedRouter(nn.Module):
    """1-hidden-layer MLP from L0 code → branch logits.

    Trained on (L0(x), branch_id) pairs sampled from each donor's
    training data: a sample x is labeled with the branch that owns
    the class y. Soft routing at inference = softmax over branch
    logits / T.
    """
    def __init__(self, l0_dim, n_branches, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(l0_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_branches)

    def forward(self, z):
        return self.fc2(F.relu(self.fc1(z)))


class BTMMoEOrganism(nn.Module):
    """Same multi-branch structure as MultiBranchOrganism but the gate
    is a LEARNED router rather than archive likelihood. Branches stay
    frozen; only the router has trainable params."""
    def __init__(self, organism, router):
        super().__init__()
        self.org = organism
        self.router = router

    @property
    def branches(self):
        return self.org.branches

    @property
    def union_classes(self):
        return self.org.union_classes

    def storage_bytes(self):
        sb = self.org.storage_bytes()
        router_bytes = sum(p.numel() * p.element_size()
                           for p in self.router.parameters())
        return {**sb, "router_bytes": router_bytes,
                "total_bytes": sb["total_bytes"] + router_bytes}

    def gate_logits(self, z):
        return self.router(z)

    def gates(self, z, *, mode="soft", temperature=1.0):
        if mode == "uniform":
            n = len(self.org.branches)
            return z.new_full((z.shape[0], n), 1.0 / n)
        log_lik = self.gate_logits(z)
        if mode == "hard":
            idx = log_lik.argmax(dim=-1)
            g = torch.zeros_like(log_lik)
            g.scatter_(1, idx.unsqueeze(1), 1.0)
            return g
        return F.softmax(log_lik / max(temperature, 1e-6), dim=-1)

    def forward(self, x, *, routing="soft", temperature=1.0,
                normalize_per_branch=False, return_extras=False):
        # Reuse the organism's combine path but with the learned gate.
        z = self.org.project_l0(x)
        gates = self.gates(z, mode=routing, temperature=temperature)
        n_union = len(self.org._union_classes)
        B = z.shape[0]
        if normalize_per_branch:
            combined = z.new_full((B, n_union), float("-inf"))
            log_g = torch.log(gates.clamp_min(1e-30))
        else:
            combined = z.new_zeros(B, n_union)
            log_g = None
        branch_padded = z.new_zeros(B, len(self.org._branches), n_union) \
            if return_extras else None
        for bi, b in enumerate(self.org._branches):
            head_logits = b.forward_from_l0(z)
            cov = b.classes_covered
            cols = head_logits[:, cov]
            if normalize_per_branch:
                cols = F.log_softmax(cols, dim=-1)
                cols = cols + log_g[:, bi:bi + 1]
            else:
                cols = gates[:, bi:bi + 1] * cols
            for j, c in enumerate(cov):
                ui = self.org._class_to_union[c]
                combined[:, ui] = cols[:, j]
                if branch_padded is not None:
                    branch_padded[:, bi, ui] = cols[:, j]
        if return_extras:
            return combined, {
                "z": z, "gates": gates, "branch_logits_padded": branch_padded,
            }
        return combined


def train_router(
    organism, branches, bundle, *,
    n_steps=400, batch=128, lr=1e-3, log_every=50, seed=0,
):
    """Train a LearnedRouter on (L0(x), branch_id) pairs sampled from
    each donor's training set. Frozen L0 + frozen branches; only the
    router parameters update."""
    torch.manual_seed(seed)
    l0_dim = organism.l0_W.shape[0]
    router = LearnedRouter(l0_dim, n_branches=len(branches))
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    # Build per-branch training image tensors.
    train_views_per_branch = []
    for b in branches:
        v = build_task_views(
            bundle, _spec_block_for_donor(b.label), split="train",
        )
        # Concatenate this branch's tasks into one big tensor.
        xs = torch.cat([w.images for w in v], dim=0)
        train_views_per_branch.append(xs)
    n_branches = len(branches)
    print(f"[router] training {n_steps} steps, batch={batch}, lr={lr}")
    for step in range(n_steps):
        # Per step, sample equal counts from each branch.
        per = batch // n_branches
        x_chunks = []
        y_chunks = []
        for bi, xs in enumerate(train_views_per_branch):
            idx = torch.randint(0, xs.shape[0], (per,))
            x_chunks.append(xs[idx])
            y_chunks.append(torch.full((per,), bi, dtype=torch.long))
        x = torch.cat(x_chunks, dim=0)
        y = torch.cat(y_chunks, dim=0)
        with torch.no_grad():
            z = organism.project_l0(x)
        logits = router(z)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0 or (step + 1) % log_every == 0 or step == n_steps - 1:
            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == y).float().mean().item()
            print(f"  step {step+1:4d}/{n_steps}  loss {loss.item():.4f}  "
                  f"router_acc {acc:.4f}")
    return router


# ---------------------------------------------------------------------
# Headline reporting
# ---------------------------------------------------------------------


def report(name, rows):
    if not rows:
        print(f"\n--- {name}\n  (no rows)")
        return
    print(f"\n--- {name}")
    header = (f"{'task':<24}{'n':>6}  {'active':<14}"
              f"{'task-aware':>12}{'full-union':>12}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['task']:<24}{r['n']:>6}  {str(r['active']):<14}"
              f"{r['task_aware']:>12.4f}{r['full_union']:>12.4f}")
    n = len(rows)
    print(f"  mean    task-aware = "
          f"{sum(r['task_aware'] for r in rows)/n:.4f}    "
          f"full-union = {sum(r['full_union'] for r in rows)/n:.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donors", default="digits,fashion,emnist")
    parser.add_argument("--ckpt-prefix", default="outputs/poc_donor_")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--router-steps", type=int, default=400)
    parser.add_argument("--router-batch", type=int, default=128)
    parser.add_argument("--router-lr", type=float, default=1e-3)
    parser.add_argument("--router-seed", type=int, default=0)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    labels = [d.strip() for d in args.donors.split(",") if d.strip()]
    print(f"BTM baselines vs ours — donors={labels}")
    branches = []
    for lab in labels:
        ckpt = f"{args.ckpt_prefix}{lab}.pt"
        b = Branch.from_checkpoint(ckpt, label=lab)
        branches.append(b)
        print(f"  loaded {lab:<10} arch={list(b.net.n_nodes_per_layer())}  "
              f"classes={b.classes_covered}")

    org = MultiBranchOrganism.from_branches(branches)

    # Eval views over the union.
    bundle = DatasetBundle(
        _datasets_for_donors(labels), root=args.data_root,
        n_holdout_per_dataset=0,
    )
    eval_views_union = []
    for b in branches:
        eval_views_union.extend(
            build_task_views(bundle, _spec_block_for_donor(b.label), split="test")
        )

    # Donor-standalone upper bound.
    print("\n" + "=" * 78)
    print("DONOR STANDALONE (upper bound)")
    print("=" * 78)
    upper_rows = []
    with torch.no_grad():
        for b in branches:
            for v in build_task_views(
                bundle, _spec_block_for_donor(b.label), split="test"
            ):
                x, y = v.all_examples()
                logits = b.net(x)
                cols = list(v.global_classes)
                sub = logits[:, cols]
                pred_local = sub.argmax(dim=-1)
                pred_global = torch.tensor(
                    [cols[int(j)] for j in pred_local], dtype=torch.long,
                )
                ta = float((pred_global == y).float().mean().item())
                upper_rows.append({"task": v.name, "n": int(x.shape[0]),
                                   "active": cols, "task_aware": ta,
                                   "full_union": float("nan")})
    report("Donor standalone (each donor on its own tasks)", upper_rows)

    # Ours.
    print("\n" + "=" * 78)
    print("OURS — archive-routed, soft + log-softmax (ZERO-SHOT)")
    print("=" * 78)
    ours_rows = evaluate(
        org, eval_views_union, routing="soft",
        temperature=args.temperature, normalize_per_branch=True,
    )
    report(f"Ours (soft+norm, T={args.temperature})", ours_rows)

    # BTM-avg.
    print("\n" + "=" * 78)
    print("BTM-avg — average L1 substrate, concatenate head (ZERO-SHOT)")
    print("=" * 78)
    btm_avg = build_btm_avg_organism(branches)
    btm_avg_rows = evaluate(
        btm_avg, eval_views_union, routing="soft", temperature=1.0,
    )
    report("BTM-avg (param averaging)", btm_avg_rows)

    # BTM-MoE.
    print("\n" + "=" * 78)
    print(f"BTM-MoE — learned router ({args.router_steps} training steps "
          f"on real images)")
    print("=" * 78)
    router = train_router(
        org, branches, bundle,
        n_steps=args.router_steps, batch=args.router_batch,
        lr=args.router_lr, seed=args.router_seed,
    )
    moe = BTMMoEOrganism(org, router)
    moe.eval()
    moe_rows = evaluate(
        moe, eval_views_union, routing="soft",
        temperature=args.temperature, normalize_per_branch=True,
    )
    report(f"BTM-MoE (learned router, soft+norm, T={args.temperature})",
           moe_rows)

    moe_rows_no_norm = evaluate(
        moe, eval_views_union, routing="soft",
        temperature=args.temperature, normalize_per_branch=False,
    )
    report("BTM-MoE (learned router, soft, raw logits)", moe_rows_no_norm)

    # Headline.
    def mta(rs):
        return sum(r["task_aware"] for r in rs) / len(rs)
    def mfu(rs):
        vals = [r["full_union"] for r in rs
                if not (isinstance(r["full_union"], float)
                        and r["full_union"] != r["full_union"])]
        return sum(vals) / len(vals) if vals else float("nan")

    print()
    print("=" * 78)
    print("HEADLINE")
    print("=" * 78)
    print(f"  donor-standalone (upper bound) task-aware = {mta(upper_rows):.4f}")
    print(f"  Ours (zero-shot, archive route) task-aware = "
          f"{mta(ours_rows):.4f}  full-union = {mfu(ours_rows):.4f}")
    print(f"  BTM-avg (zero-shot, param avg)  task-aware = "
          f"{mta(btm_avg_rows):.4f}  full-union = {mfu(btm_avg_rows):.4f}")
    print(f"  BTM-MoE (learned router, +norm) task-aware = "
          f"{mta(moe_rows):.4f}  full-union = {mfu(moe_rows):.4f}")
    print(f"  BTM-MoE (learned router, raw)   task-aware = "
          f"{mta(moe_rows_no_norm):.4f}  full-union = "
          f"{mfu(moe_rows_no_norm):.4f}")
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
