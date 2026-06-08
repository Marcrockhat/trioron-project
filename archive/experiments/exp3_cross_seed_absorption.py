"""Exp 3 from paper/L0_HANDSHAKE_BRIEF.md — cross-seed multi-branch absorption.

Compares five paths to compose two donors into one organism:

  A. SAME-SEED CONTROL — both donors share l0_seed=42. Shared canonical
     L0; current paste-and-go path. Upper bound for cross-seed work.

  B. CROSS-SEED NAIVE — donors have different l0_seeds. The off-seed
     donor's L1 and archive are fed canonical z directly (no adapter).
     Lower bound; expected ~chance on the off-seed donor's classes.

  C. CROSS-SEED RANDOM-PROJECTION ADAPTER — current MANUAL §3 fallback
     (Solution 3, 2026-05-06). A deterministic Gaussian random projection
     rotates canonical z into off-seed donor space. Untested-accuracy path
     in the live codebase.

  D. CROSS-SEED CLOSED-FORM TRANSLATOR — paper L0_HANDSHAKE_BRIEF.md.
     M = W_B · W_A^+, c = b_B - M·b_A. Exact in row-space(W_A); the
     residual against full W_B·x is donor A's information bottleneck.

  E. CROSS-SEED PUBLISH-W_L0 — donors publish their own (W_L0, b_L0);
     recipient runs each branch's L0 directly on x. Bit-exact, no info
     loss; pays N parallel L0 matmuls instead of one canonical + N
     translators.

Run:
  python3 -m experiments.exp3_cross_seed_absorption \\
      > outputs/cross_seed_test/exp3_run1.log 2>&1
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from experiments.datasets import (
    DatasetBundle, build_task_views, chained_15_specs, DEFAULT_DATA_ROOT,
)
from experiments.train_donor import SPLIT_BLOCKS
from trioron.composition import L0Translator, transform_archive_to_canonical
from trioron.multibranch import Branch


def task_aware_accuracy(
    logits_union: torch.Tensor,
    union_classes: List[int],
    labels_global: torch.Tensor,
    active_classes: List[int],
) -> float:
    union_idx = {c: i for i, c in enumerate(union_classes)}
    cols = [union_idx[c] for c in active_classes]
    sub = logits_union[:, cols]
    pred_local = sub.argmax(dim=-1)
    pred_global = torch.tensor(
        [active_classes[int(j)] for j in pred_local], dtype=torch.long,
    )
    return float((pred_global == labels_global).float().mean().item())


def full_union_accuracy(
    logits_union: torch.Tensor,
    union_classes: List[int],
    labels_global: torch.Tensor,
) -> float:
    pred_idx = logits_union.argmax(dim=-1)
    pred_global = torch.tensor(
        [union_classes[int(j)] for j in pred_idx], dtype=torch.long,
    )
    return float((pred_global == labels_global).float().mean().item())


# ---------------------------------------------------------------------
# Per-mode forward — each one returns (logits_union, gate_means_per_task)
# given a list of branches and an evaluation batch x.
# ---------------------------------------------------------------------


def _per_branch_z_for_mode(
    mode: str,
    x: torch.Tensor,
    branches: List[Branch],
    *,
    canonical_branch_idx: int,
    translators: List[L0Translator | None] | None = None,
) -> List[torch.Tensor]:
    """For each branch, compute the z (post-ReLU L0 code) it should consume.

    mode = "same_seed" — every branch receives the canonical z (assumes
        all l0_seeds match; caller is responsible).
    mode = "naive" — every branch receives the canonical z (off-seed
        branches get alien input).
    mode = "rand_proj" — off-seed branches go through a Gaussian
        random-projection adapter (Solution 3). Caller pre-attaches
        `branch.projection`; we just call branch._project_to_donor_space.
    mode = "translator" — off-seed branches consume z = ReLU(M·pre_C + c).
        We compute pre_C (skipping the canonical ReLU) and translate.
        Translators must be passed in, indexed parallel to `branches`,
        with None for the canonical branch.
    mode = "publish_w_l0" — every branch runs its own L0 on x.
    """
    canon = branches[canonical_branch_idx]
    z_canon_pre = F.linear(x, canon.l0_W(), canon.l0_b())     # pre-ReLU canonical
    z_canon = z_canon_pre.clamp_min(0.0)                      # post-ReLU canonical

    out: List[torch.Tensor] = []
    for bi, b in enumerate(branches):
        if mode == "same_seed" or mode == "naive":
            out.append(z_canon)
        elif mode == "rand_proj":
            # branch.projection is set up by caller (None for canonical).
            out.append(b._project_to_donor_space(z_canon))
        elif mode == "translator":
            if bi == canonical_branch_idx:
                out.append(z_canon)
            else:
                t = translators[bi]
                pre_b = t.translate(z_canon_pre)              # exact in row-space(W_C)
                out.append(pre_b.clamp_min(0.0))
        elif mode == "publish_w_l0":
            pre = F.linear(x, b.l0_W(), b.l0_b())
            out.append(pre.clamp_min(0.0))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return out


def _combine_logits(
    z_per_branch: List[torch.Tensor],
    branches: List[Branch],
    union_classes: List[int],
    class_to_union: dict,
    *,
    routing: str = "soft",
    temperature: float = 1.0,
    return_gates: bool = False,
):
    """Soft / hard / uniform routing combine. Mirrors MultiBranchOrganism's
    forward but lets us inject per-branch z."""
    # Per-branch log-likelihoods under each branch's own archive (using
    # that branch's view of the input).
    log_liks = torch.stack(
        [b.archive_log_likelihood(z) for b, z in zip(branches, z_per_branch)],
        dim=-1,
    )  # (B, N_branches)

    if routing == "soft":
        gates = F.softmax(log_liks / max(temperature, 1e-6), dim=-1)
    elif routing == "hard":
        idx = log_liks.argmax(dim=-1)
        gates = torch.zeros_like(log_liks)
        gates.scatter_(1, idx.unsqueeze(1), 1.0)
    elif routing == "uniform":
        gates = log_liks.new_full(log_liks.shape, 1.0 / log_liks.shape[1])
    else:
        raise ValueError(f"unknown routing {routing}")

    B = z_per_branch[0].shape[0]
    n_union = len(union_classes)
    combined = z_per_branch[0].new_zeros(B, n_union)
    for bi, (b, z) in enumerate(zip(branches, z_per_branch)):
        head_logits = b.forward_from_l0(z)
        cov = b.classes_covered
        cols = head_logits[:, cov]                            # (B, |cov|)
        cols = gates[:, bi:bi + 1] * cols
        for j, c in enumerate(cov):
            combined[:, class_to_union[c]] = cols[:, j]
    if return_gates:
        return combined, gates
    return combined


def _diag_gauss_logpdf(z: torch.Tensor, mu: torch.Tensor,
                      sg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-row, per-class diagonal-Gaussian log-pdf. z (B, d), mu (C, d),
    sg (C, d). Returns (B, C). Mirrors Branch.per_class_log_likelihood."""
    sg = sg.clamp_min(eps)
    d = z.shape[-1]
    diff = z.unsqueeze(1) - mu.unsqueeze(0)                   # (B, C, d)
    norm = ((diff / sg.unsqueeze(0)) ** 2).sum(-1)            # (B, C)
    logdet = sg.log().sum(-1)                                 # (C,)
    return -0.5 * norm - logdet.unsqueeze(0) - 0.5 * d * math.log(2 * math.pi)


def _archive_loglik_from_stats(z: torch.Tensor, stats: dict) -> torch.Tensor:
    """logsumexp over per-class diagonal-Gaussians from a stats dict
    {class: (mu, sg)}. Uniform mixture, like Branch.archive_log_likelihood.
    Returns (B,)."""
    classes = sorted(stats.keys())
    mus = torch.stack([stats[c][0] for c in classes]).to(z.device, z.dtype)
    sgs = torch.stack([stats[c][1] for c in classes]).to(z.device, z.dtype)
    log_pdfs = _diag_gauss_logpdf(z, mus, sgs)
    return torch.logsumexp(log_pdfs, dim=-1) - math.log(len(classes))


def evaluate_path_F(
    branches: List[Branch],
    eval_views,
    *,
    canonical_branch_idx: int,
    transformed_archive_per_branch: List[dict | None],
    translators: List[L0Translator | None],
    routing: str = "soft",
    temperature: float = 1.0,
    batch: int = 512,
):
    """Path F: closed-form L1 translator + transformed-archive routing.

    For canonical-seed branch: native archive on canonical post-ReLU z_C.
    For off-seed branch: transformed archive on canonical post-ReLU z_C
        (no translator residual on routing) AND closed-form translator
        on canonical pre-ReLU pre_C → branch's pre-ReLU pre_B → ReLU →
        branch's L1.
    """
    union_classes: List[int] = []
    class_to_union: dict = {}
    for b in branches:
        for c in b.classes_covered:
            class_to_union[c] = len(union_classes)
            union_classes.append(c)

    canon = branches[canonical_branch_idx]
    n_branches = len(branches)
    rows = []
    gate_means_per_task = []

    with torch.no_grad():
        for v in eval_views:
            x_all, y_all = v.all_examples()
            n = x_all.shape[0]
            chunks = []
            gate_acc = torch.zeros(n_branches)
            for s in range(0, n, batch):
                x_b = x_all[s:s + batch]
                pre_C = F.linear(x_b, canon.l0_W(), canon.l0_b())
                z_C = pre_C.clamp_min(0.0)

                # ---- routing: score z_C against per-branch archive ----
                log_lik_cols = []
                for bi, b in enumerate(branches):
                    if bi == canonical_branch_idx:
                        # canonical branch's archive lives in canonical space already
                        log_lik_cols.append(b.archive_log_likelihood(z_C))
                    else:
                        stats = transformed_archive_per_branch[bi]
                        log_lik_cols.append(_archive_loglik_from_stats(z_C, stats))
                log_lik = torch.stack(log_lik_cols, dim=-1)         # (B, N)

                if routing == "soft":
                    gates = F.softmax(log_lik / max(temperature, 1e-6), dim=-1)
                elif routing == "hard":
                    idx = log_lik.argmax(dim=-1)
                    gates = torch.zeros_like(log_lik)
                    gates.scatter_(1, idx.unsqueeze(1), 1.0)
                else:
                    gates = log_lik.new_full(log_lik.shape, 1.0 / n_branches)
                gate_acc += gates.sum(dim=0)

                # ---- L1 forward ----
                B = x_b.shape[0]
                combined = z_C.new_zeros(B, len(union_classes))
                for bi, b in enumerate(branches):
                    if bi == canonical_branch_idx:
                        head_logits = b.forward_from_l0(z_C)
                    else:
                        t = translators[bi]
                        pre_B = t.translate(pre_C)
                        z_B = pre_B.clamp_min(0.0)
                        head_logits = b.forward_from_l0(z_B)
                    cov = b.classes_covered
                    cols = head_logits[:, cov] * gates[:, bi:bi + 1]
                    for j, c in enumerate(cov):
                        combined[:, class_to_union[c]] = cols[:, j]
                chunks.append(combined)

            logits_all = torch.cat(chunks, dim=0)
            ta = task_aware_accuracy(
                logits_all, union_classes, y_all, list(v.global_classes),
            )
            full = full_union_accuracy(logits_all, union_classes, y_all)
            rows.append({"task": v.name, "n": n,
                         "active": list(v.global_classes),
                         "task_aware": ta, "full_union": full})
            gate_means_per_task.append({
                "task": v.name, "gates": (gate_acc / n).tolist(),
            })
    return rows, gate_means_per_task, union_classes


def evaluate_mode(
    mode: str,
    branches: List[Branch],
    eval_views,
    *,
    canonical_branch_idx: int,
    translators: List[L0Translator | None] | None = None,
    routing: str = "soft",
    temperature: float = 1.0,
    batch: int = 512,
):
    union_classes: List[int] = []
    class_to_union: dict = {}
    for b in branches:
        for c in b.classes_covered:
            class_to_union[c] = len(union_classes)
            union_classes.append(c)

    rows = []
    n_branches = len(branches)
    gate_means_per_task = []
    with torch.no_grad():
        for v in eval_views:
            x_all, y_all = v.all_examples()
            n = x_all.shape[0]
            chunks = []
            gate_acc = torch.zeros(n_branches)
            for s in range(0, n, batch):
                x_b = x_all[s:s + batch]
                z_per_branch = _per_branch_z_for_mode(
                    mode, x_b, branches,
                    canonical_branch_idx=canonical_branch_idx,
                    translators=translators,
                )
                logits, gates = _combine_logits(
                    z_per_branch, branches, union_classes, class_to_union,
                    routing=routing, temperature=temperature,
                    return_gates=True,
                )
                chunks.append(logits)
                gate_acc += gates.sum(dim=0)
            logits_all = torch.cat(chunks, dim=0)
            ta = task_aware_accuracy(
                logits_all, union_classes, y_all, list(v.global_classes),
            )
            full = full_union_accuracy(logits_all, union_classes, y_all)
            rows.append({"task": v.name, "n": n,
                         "active": list(v.global_classes),
                         "task_aware": ta, "full_union": full})
            gate_means_per_task.append({
                "task": v.name,
                "gates": (gate_acc / n).tolist(),
            })
    return rows, gate_means_per_task, union_classes


def report(title, rows):
    print(f"\n--- {title}")
    header = f"{'task':<24}{'n':>6}  {'active':<14}{'task-aware':>12}{'full-union':>12}"
    print(header); print("-" * len(header))
    ta = 0.0; fu = 0.0
    for r in rows:
        print(f"{r['task']:<24}{r['n']:>6}  {str(r['active']):<14}"
              f"{r['task_aware']:>12.4f}{r['full_union']:>12.4f}")
        ta += r["task_aware"]; fu += r["full_union"]
    n = len(rows)
    print(f"{'mean':<24}{'':>6}  {'':<14}{ta/n:>12.4f}{fu/n:>12.4f}")
    return ta / n, fu / n


def report_gates(title, gate_rows, branch_labels):
    print(f"\n[gates] {title}")
    print(f"{'task':<24}" + "".join(f"{lab:>12}" for lab in branch_labels))
    for r in gate_rows:
        print(f"{r['task']:<24}" +
              "".join(f"{g:>12.4f}" for g in r["gates"]))


def _build_random_projection_for_branch(canon_branch, off_branch):
    """Mirror multibranch._build_random_projection. Sets off_branch.projection
    in place so branch._project_to_donor_space picks it up."""
    from trioron.multibranch import _build_random_projection
    canon_seed = int(canon_branch.l0_seed)
    donor_seed = int(off_branch.l0_seed)
    canon_dim = canon_branch.l0_W().shape[0]
    donor_dim = off_branch.l0_W().shape[0]
    off_branch.projection = _build_random_projection(
        canon_seed=canon_seed, donor_seed=donor_seed,
        canon_dim=canon_dim, donor_dim=donor_dim,
        dtype=canon_branch.l0_W().dtype,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canon-ckpt", default="outputs/poc_donor_digits.pt",
        help="Canonical donor (its L0 seed defines canonical-L0).",
    )
    parser.add_argument(
        "--same-seed-ckpt", default="outputs/poc_donor_fashion.pt",
        help="Off-donor at the SAME seed (Path A control).",
    )
    parser.add_argument(
        "--cross-seed-ckpt",
        default="outputs/cross_seed_test/poc_donor_fashion.pt",
        help="Off-donor at a DIFFERENT seed (Paths B/C/D/E).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--routing", default="soft",
                        choices=["soft", "hard", "uniform"])
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args(argv)

    print("Exp 3: cross-seed multi-branch absorption")
    print(f"  canon (digits)       : {args.canon_ckpt}")
    print(f"  same-seed (control)  : {args.same_seed_ckpt}")
    print(f"  cross-seed           : {args.cross_seed_ckpt}")
    print(f"  routing              : {args.routing}  T={args.temperature}")

    digits = Branch.from_checkpoint(args.canon_ckpt, label="digits")
    fashion_same = Branch.from_checkpoint(args.same_seed_ckpt,
                                          label="fashion@same")
    fashion_cross = Branch.from_checkpoint(args.cross_seed_ckpt,
                                           label="fashion@cross")
    print(f"  digits l0_seed       : {digits.l0_seed}")
    print(f"  fashion-same l0_seed : {fashion_same.l0_seed}")
    print(f"  fashion-cross l0_seed: {fashion_cross.l0_seed}")
    if digits.l0_seed == fashion_cross.l0_seed:
        raise SystemExit(
            "cross-seed checkpoint shares l0_seed with canon — train it "
            "with --seed 43 first."
        )

    # Eval views over the union (digits + fashion).
    union_specs = (
        SPLIT_BLOCKS["digits"][0]() + SPLIT_BLOCKS["fashion"][0]()
    )
    bundle = DatasetBundle(
        ["mnist", "fashion_mnist"], root=args.data_root,
        n_holdout_per_dataset=0,
    )
    eval_views = build_task_views(bundle, union_specs, split="test")

    # ---------- Path A: same-seed control ----------
    print("\n" + "=" * 78)
    print("Path A: SAME-SEED CONTROL (digits + fashion @ seed 42)")
    print("=" * 78)
    rows, gates, union = evaluate_mode(
        "same_seed", [digits, fashion_same], eval_views,
        canonical_branch_idx=0,
        routing=args.routing, temperature=args.temperature,
    )
    ta_A, fu_A = report("Path A — same-seed", rows)
    report_gates("Path A", gates, ["digits", "fashion@same"])

    # ---------- Path B: cross-seed naive ----------
    print("\n" + "=" * 78)
    print("Path B: CROSS-SEED NAIVE (no adapter, no translator)")
    print("=" * 78)
    fashion_cross.projection = None                       # ensure no adapter
    rows, gates, _ = evaluate_mode(
        "naive", [digits, fashion_cross], eval_views,
        canonical_branch_idx=0,
        routing=args.routing, temperature=args.temperature,
    )
    ta_B, fu_B = report("Path B — cross-seed naive", rows)
    report_gates("Path B", gates, ["digits", "fashion@cross"])

    # ---------- Path C: cross-seed random-projection adapter ----------
    print("\n" + "=" * 78)
    print("Path C: CROSS-SEED RANDOM-PROJECTION ADAPTER (Solution 3)")
    print("=" * 78)
    _build_random_projection_for_branch(digits, fashion_cross)
    rows, gates, _ = evaluate_mode(
        "rand_proj", [digits, fashion_cross], eval_views,
        canonical_branch_idx=0,
        routing=args.routing, temperature=args.temperature,
    )
    ta_C, fu_C = report("Path C — random-projection adapter", rows)
    report_gates("Path C", gates, ["digits", "fashion@cross"])
    fashion_cross.projection = None                       # detach

    # ---------- Path D: cross-seed closed-form translator ----------
    print("\n" + "=" * 78)
    print("Path D: CROSS-SEED CLOSED-FORM TRANSLATOR (L0 handshake)")
    print("=" * 78)
    translator = L0Translator.from_donors(digits, fashion_cross)
    print(f"  translator M shape   : {tuple(translator.M.shape)}")
    print(f"  translator c shape   : {tuple(translator.c.shape)}")
    rows, gates, _ = evaluate_mode(
        "translator", [digits, fashion_cross], eval_views,
        canonical_branch_idx=0,
        translators=[None, translator],
        routing=args.routing, temperature=args.temperature,
    )
    ta_D, fu_D = report("Path D — closed-form translator", rows)
    report_gates("Path D", gates, ["digits", "fashion@cross"])

    # ---------- Path E: cross-seed publish-W_L0 ----------
    print("\n" + "=" * 78)
    print("Path E: CROSS-SEED PUBLISH-W_L0 (recipient runs each L0)")
    print("=" * 78)
    rows, gates, _ = evaluate_mode(
        "publish_w_l0", [digits, fashion_cross], eval_views,
        canonical_branch_idx=0,
        routing=args.routing, temperature=args.temperature,
    )
    ta_E, fu_E = report("Path E — publish W_L0", rows)
    report_gates("Path E", gates, ["digits", "fashion@cross"])

    # ---------- Path F: translator (L1) + transformed archive (routing) ----------
    print("\n" + "=" * 78)
    print("Path F: TRANSLATOR + TRANSFORMED ARCHIVE (manifold pseudo-replay)")
    print("=" * 78)
    transformed_stats = transform_archive_to_canonical(
        fashion_cross,
        canon_W=digits.l0_W(), canon_b=digits.l0_b(),
        n_samples_per_class=256, seed=0,
        apply_relu_canonical=True,
    )
    # Print archive width comparison so we can see how the synthesis
    # compares to the donor's native archive on its own scale.
    native_sg_mean = sum(
        float(sg.mean()) for _, sg in fashion_cross.manifold_stats.values()
    ) / len(fashion_cross.manifold_stats)
    xform_sg_mean = sum(
        float(sg.mean()) for _, sg in transformed_stats.values()
    ) / len(transformed_stats)
    print(f"  refit {len(transformed_stats)} per-class archives via "
          f"256-sample pseudo-replay → canonical post-ReLU space")
    print(f"  native fashion σ_B mean    = {native_sg_mean:.4f}")
    print(f"  synthesized σ_C mean (xfer)= {xform_sg_mean:.4f}")
    rows, gates, _ = evaluate_path_F(
        [digits, fashion_cross], eval_views,
        canonical_branch_idx=0,
        transformed_archive_per_branch=[None, transformed_stats],
        translators=[None, translator],
        routing=args.routing, temperature=args.temperature,
    )
    ta_F, fu_F = report("Path F — translator + transformed archive", rows)
    report_gates("Path F", gates, ["digits", "fashion@cross"])

    # ---------- Path G: translator + GROUND-TRUTH canonical archive ----------
    # Diagnostic upper bound for any archive-transfer scheme. Refit
    # fashion's archive using REAL fashion training images forwarded
    # through the canonical L0. This is what perfect archive transfer
    # would produce, given probe data.
    print("\n" + "=" * 78)
    print("Path G: TRANSLATOR + GROUND-TRUTH CANONICAL ARCHIVE (diagnostic)")
    print("=" * 78)
    fashion_specs = SPLIT_BLOCKS["fashion"][0]()
    train_views = build_task_views(bundle, fashion_specs, split="train")
    gt_stats = {}
    with torch.no_grad():
        for v in train_views:
            x_all, y_all = v.all_examples()
            pre_C = F.linear(x_all, digits.l0_W(), digits.l0_b())
            z_C = pre_C.clamp_min(0.0)
            for c in v.global_classes:
                mask = (y_all == c)
                if mask.sum() == 0:
                    continue
                z_c = z_C[mask]
                gt_stats[int(c)] = (z_c.mean(0), z_c.std(0).clamp_min(1e-6))
    gt_sg_mean = sum(float(sg.mean()) for _, sg in gt_stats.values()) / len(gt_stats)
    print(f"  refit {len(gt_stats)} per-class archives from real fashion "
          f"training data → canonical post-ReLU space")
    print(f"  ground-truth σ_C mean      = {gt_sg_mean:.4f}")
    rows, gates, _ = evaluate_path_F(
        [digits, fashion_cross], eval_views,
        canonical_branch_idx=0,
        transformed_archive_per_branch=[None, gt_stats],
        translators=[None, translator],
        routing=args.routing, temperature=args.temperature,
    )
    ta_G, fu_G = report("Path G — translator + ground-truth archive", rows)
    report_gates("Path G", gates, ["digits", "fashion@cross"])

    # ---------- Headline ----------
    print("\n" + "=" * 78)
    print("HEADLINE — task-aware (mean over 10 binary tasks)")
    print("=" * 78)
    print(f"  A. SAME-SEED CONTROL          task-aware = {ta_A:.4f}  "
          f"full-union = {fu_A:.4f}    (upper bound)")
    print(f"  B. CROSS-SEED NAIVE           task-aware = {ta_B:.4f}  "
          f"full-union = {fu_B:.4f}    (lower bound)")
    print(f"  C. CROSS-SEED rand-proj adptr task-aware = {ta_C:.4f}  "
          f"full-union = {fu_C:.4f}    (Solution 3)")
    print(f"  D. CROSS-SEED translator      task-aware = {ta_D:.4f}  "
          f"full-union = {fu_D:.4f}    (closed-form, native archive)")
    print(f"  E. CROSS-SEED publish-W_L0    task-aware = {ta_E:.4f}  "
          f"full-union = {fu_E:.4f}    (lossless)")
    print(f"  F. CROSS-SEED translator+xfer task-aware = {ta_F:.4f}  "
          f"full-union = {fu_F:.4f}    (translator L1 + xform archive)")
    print(f"  G. CROSS-SEED translator+GT   task-aware = {ta_G:.4f}  "
          f"full-union = {fu_G:.4f}    (translator L1 + GT archive — diagnostic)")
    if ta_A > 0:
        print(f"\n  D / A   = {ta_D/ta_A*100:>5.1f}%  "
              f"(closed-form translator alone)")
        print(f"  E / A   = {ta_E/ta_A*100:>5.1f}%  "
              f"(publish-W_L0; should be ~100%)")
        print(f"  F / A   = {ta_F/ta_A*100:>5.1f}%  "
              f"(translator + manifold pseudo-replay archive)")
        print(f"  G / A   = {ta_G/ta_A*100:>5.1f}%  "
              f"(translator + GT canonical archive — upper bound for archive transfer)")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
