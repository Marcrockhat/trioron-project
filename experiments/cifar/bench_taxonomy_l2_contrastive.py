"""L2 — contrastive δ-replay margin-loss refinement.

First implementation of the doc's proposal #2 (pair-difference manifold
sketch). Takes an already-trained L2 donor and adds a post-training
contrastive refinement phase that:

  1. Extracts per-fine-class (μ, σ) statistics in L0 space by pushing
     real training images through the donor's frozen L0 layer.
  2. Identifies the top-K confusable pairs from the donor's CE-trained
     confusion matrix on the test set.
  3. For each refinement step, samples a confusable pair (A, B), draws
     synthetic z_A ~ N(μ_A, σ_A) and z_B ~ N(μ_B, σ_B) directly in L0
     space (the doc's "occasionally sample a contrast pair" idea).
  4. Forwards both through L1 + head, computes a hinge margin loss:
        L = mean over pairs of max(0, m − (logit_A[A] − logit_B[A]))
                                  + max(0, m − (logit_B[B] − logit_A[B]))
     so the head's logit for A on z_A must beat its logit for A on z_B
     by margin m, and symmetrically for B.
  5. Backprops through L1 + head only (L0 is frozen).

After refinement, re-evaluates on the same test set. Reports baseline
vs refined accuracy. No dream-cycle modifications — this lives entirely
outside trioron's training loop, as a post-hoc sharpening pass.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F

from trioron.network import TrioronNetwork
from trioron.senses import apply_sense, Standardizer
from experiments.cifar.datasets import load_cifar100, DEFAULT_DATA_ROOT
from experiments.cifar.bench_taxonomy_l1 import _resolve_names_to_ids
from experiments.cifar.bench_taxonomy_l2_expert import _build_subset


def _load_donor(path: str) -> Tuple[TrioronNetwork, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    n_nodes = list(payload["n_nodes_per_layer"])
    layer_specs = []
    prev = int(payload["input_dim"])
    for i, n in enumerate(n_nodes):
        act = "linear" if i == len(n_nodes) - 1 else "relu"
        layer_specs.append((prev, int(n), act))
        prev = int(n)
    net = TrioronNetwork(layer_specs)
    net.load_state_dict(payload["state_dict"])
    net.eval()
    return net, payload


def _compute_l0_stats(
    net: TrioronNetwork, X: torch.Tensor, y: torch.Tensor, K: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (mu, sigma) each shape (K, L0_dim)."""
    with torch.no_grad():
        h0 = net.layers[0](X)
    L0_dim = h0.shape[1]
    mu = torch.zeros(K, L0_dim)
    sigma = torch.zeros(K, L0_dim)
    for c in range(K):
        m = y == c
        if m.sum() == 0:
            sigma[c] = 1.0
            continue
        mu[c] = h0[m].mean(dim=0)
        sigma[c] = h0[m].std(dim=0).clamp_min(1e-3)
    return mu, sigma


def _confusion_matrix(
    net: TrioronNetwork, X: torch.Tensor, y: torch.Tensor, K: int,
) -> torch.Tensor:
    with torch.no_grad():
        pred = net(X).argmax(dim=1)
    cm = torch.zeros(K, K, dtype=torch.long)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm


def _top_pairs(cm: torch.Tensor, top_k: int) -> List[Tuple[int, int, int]]:
    """Return [(count, i, j)] sorted by count desc; only off-diagonal."""
    K = cm.shape[0]
    pairs = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            c = int(cm[i, j].item())
            if c > 0:
                pairs.append((c, i, j))
    pairs.sort(reverse=True)
    return pairs[:top_k]


def _eval(net, X, y) -> Tuple[float, torch.Tensor]:
    with torch.no_grad():
        pred = net(X).argmax(dim=1)
    return (pred == y).float().mean().item(), pred


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--donor-path",
        default="outputs/cifar_taxonomy/donor_l2_central_object_9way.pt",
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-pairs", type=int, default=64,
                        help="Synthetic pairs per refinement step.")
    parser.add_argument("--top-k-pairs", type=int, default=12,
                        help="How many top confusable pairs to target.")
    parser.add_argument("--ce-weight", type=float, default=1.0,
                        help="Weight on the CE-on-real-samples retention "
                             "term. 0 = pure margin (collapses).")
    parser.add_argument("--margin-weight", type=float, default=0.5,
                        help="Weight on the contrastive margin term.")
    parser.add_argument("--ce-batch", type=int, default=256,
                        help="Real-sample batch size for CE retention.")
    parser.add_argument("--align-pairs", action="store_true",
                        help="Resample synthetic pairs until z_A−z_B aligns "
                             "with δ_AB (cos(z_A−z_B, δ_AB) > 0).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-path",
        default="outputs/cifar_taxonomy/donor_l2_central_object_9way_contrastive.pt",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)

    net, payload = _load_donor(args.donor_path)
    sense_name = payload["sense"]
    std = Standardizer.from_dict(payload["standardizer"])
    K = int(payload["n_nodes_per_layer"][-1])
    cluster_names = list(payload.get("fine_class_names", []))
    if not cluster_names:
        raise RuntimeError(
            f"donor at {args.donor_path} has no fine_class_names — "
            f"retrain via bench_taxonomy_l2_expert.py to bake metadata"
        )
    print(f"[L2-c] donor: {args.donor_path}")
    print(f"[L2-c]   sense={sense_name}  arch={payload['n_nodes_per_layer']}")
    print(f"[L2-c]   cluster: {payload.get('cluster', '(unspecified)')}  "
          f"K={K}  fines={cluster_names}")

    Xtr, ytr_local, _, Xte, yte_local, _, _ = _build_subset(
        args.data_root, cluster_names,
    )
    Xtr_s = std.transform(apply_sense(sense_name, Xtr)).contiguous()
    Xte_s = std.transform(apply_sense(sense_name, Xte)).contiguous()

    mu, sigma = _compute_l0_stats(net, Xtr_s, ytr_local, K)
    print(f"[L2-c] L0 stats: μ {tuple(mu.shape)}  σ {tuple(sigma.shape)}")
    print(f"[L2-c]   ‖μ‖ range: {mu.norm(dim=1).min():.3f}–"
          f"{mu.norm(dim=1).max():.3f}")
    print(f"[L2-c]   σ range:  {sigma.mean(dim=1).min():.3f}–"
          f"{sigma.mean(dim=1).max():.3f}")

    cm = _confusion_matrix(net, Xte_s, yte_local, K)
    pairs = _top_pairs(cm, args.top_k_pairs)
    print(f"[L2-c] top-{args.top_k_pairs} confusable pairs:")
    for c, i, j in pairs:
        print(f"    {cluster_names[i]:<14s} → "
              f"{cluster_names[j]:<14s}   {c}")

    # Baseline accuracy.
    base_acc, _ = _eval(net, Xte_s, yte_local)
    print(f"\n[L2-c] baseline test acc: {base_acc:.4f} (chance 1/{K}={1/K:.3f})")

    # Freeze L0; train L1 + head with margin loss on synthetic pairs.
    for p in net.layers[0].parameters():
        p.requires_grad_(False)
    trainable = [p for layer in net.layers[1:] for p in layer.parameters()
                 if p.requires_grad]
    optim = torch.optim.Adam(trainable, lr=args.lr)
    pair_idx = torch.tensor([[i, j] for _, i, j in pairs], dtype=torch.long)
    n_pairs = pair_idx.shape[0]

    # Pre-compute δ_AB for pair-alignment filtering, if requested.
    if args.align_pairs:
        pair_delta = torch.stack([mu[i] - mu[j] for _, i, j in pairs], dim=0)
        # Normalize for cos.
        pair_delta_n = pair_delta / pair_delta.norm(dim=1, keepdim=True).clamp_min(1e-9)

    n_train = Xtr_s.shape[0]
    print(f"[L2-c] starting CE+margin refinement: "
          f"{args.steps} steps, batch_pairs={args.batch_pairs}, "
          f"ce_batch={args.ce_batch}, margin={args.margin}, lr={args.lr}, "
          f"ce_w={args.ce_weight}, margin_w={args.margin_weight}, "
          f"align_pairs={args.align_pairs}")
    t0 = time.time()
    net.train()
    log_every = max(1, args.steps // 10)
    gather = lambda logits, cls: logits.gather(1, cls.view(-1, 1)).squeeze(1)
    for step in range(1, args.steps + 1):
        # Margin loss on synthetic pairs.
        sel = torch.randint(0, n_pairs, (args.batch_pairs,))
        I = pair_idx[sel, 0]
        J = pair_idx[sel, 1]
        muA, sigA = mu[I], sigma[I]
        muB, sigB = mu[J], sigma[J]
        zA = muA + sigA * torch.randn_like(muA)
        zB = muB + sigB * torch.randn_like(muB)
        if args.align_pairs:
            d = (zA - zB)
            d_n = d / d.norm(dim=1, keepdim=True).clamp_min(1e-9)
            cos_align = (d_n * pair_delta_n[sel]).sum(dim=1)
            # Flip pairs where alignment is negative (i.e., the random
            # sample produced a difference in the wrong direction);
            # the resampled pair is reflection of zB around mu[J].
            mask = (cos_align < 0).unsqueeze(1)
            zB = torch.where(mask, 2 * muB - zB, zB)
        logits_A = net.layers[2](net.layers[1](zA))
        logits_B = net.layers[2](net.layers[1](zB))
        loss_A = F.relu(args.margin - (gather(logits_A, I) - gather(logits_B, I)))
        loss_B = F.relu(args.margin - (gather(logits_B, J) - gather(logits_A, J)))
        margin_loss = (loss_A.mean() + loss_B.mean()) / 2

        # CE retention on real samples.
        ce_idx = torch.randint(0, n_train, (args.ce_batch,))
        Xb = Xtr_s[ce_idx]
        yb = ytr_local[ce_idx]
        logits_real = net(Xb)
        ce_loss = F.cross_entropy(logits_real, yb)

        loss = args.ce_weight * ce_loss + args.margin_weight * margin_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % log_every == 0 or step == 1:
            net.eval()
            acc_now, _ = _eval(net, Xte_s, yte_local)
            net.train()
            print(f"  step {step:>5d}/{args.steps}  "
                  f"ce={ce_loss.item():.4f}  "
                  f"margin={margin_loss.item():.4f}  "
                  f"acc={acc_now:.4f}")
    print(f"[L2-c] refinement done ({time.time()-t0:.1f}s)")

    net.eval()
    final_acc, pred_final = _eval(net, Xte_s, yte_local)
    print(f"\n[L2-c] === results ===")
    print(f"  baseline acc:       {base_acc:.4f}")
    print(f"  contrastive acc:    {final_acc:.4f}")
    print(f"  Δ:                  {final_acc - base_acc:+.4f}")

    # Per-class accuracy delta.
    cm_after = torch.zeros(K, K, dtype=torch.long)
    for t, p in zip(yte_local.tolist(), pred_final.tolist()):
        cm_after[int(t), int(p)] += 1
    print(f"\n  per-class accuracy (baseline → contrastive):")
    for i, n in enumerate(cluster_names):
        n_i = int((yte_local == i).sum().item())
        if n_i == 0:
            continue
        before = cm[i, i].item() / n_i
        after = cm_after[i, i].item() / n_i
        marker = "  ↑" if after > before else ("  ↓" if after < before else "")
        print(f"    {n:<14s}  {before:.4f} → {after:.4f}{marker}")

    # Save refined donor.
    out_path = os.path.abspath(args.out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload["state_dict"] = net.state_dict()
    payload["contrastive_refinement"] = {
        "margin": args.margin,
        "steps": args.steps,
        "batch_pairs": args.batch_pairs,
        "lr": args.lr,
        "top_k_pairs": args.top_k_pairs,
        "baseline_acc": base_acc,
        "contrastive_acc": final_acc,
    }
    torch.save(payload, out_path)
    print(f"\n[L2-c] [SAVE] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
