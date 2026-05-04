"""Probe: training-time loss + gradient equivalence across the two
rehearsal injection paths.

Forward equivalence already verified by probe_injection_mechanism (both
math and curriculum tests pass). Yet bench shows raw rehearsal hits 0.62
on grown arms while hippocampal hits 0.27 at matched K=50.

This probe computes, on PAIRED (x, z=L0(x), y) data through a grown +
dream-mutated net:

    Path A (memory rehearsal):
        logits_A = net(x)
        l_A      = masked_cross_entropy(logits_A, y, all_seen)
        l_A.backward()  → record L1.W.grad, head.W.grad

    Path B (hippocampal):
        logits_B = net.forward_from_layer(z, start=1)
        l_B      = masked_cross_entropy(logits_B, y, all_seen)
        l_B.backward()  → record L1.W.grad, head.W.grad

If logits_A == logits_B then l_A == l_B.
If autograd is consistent, gradients must also be identical.

If everything matches: the gap MUST be in the bench's runtime
dynamics — sampling distribution, ordering, or the interaction with
other loss components (LwF, EWC penalty, etc.).

Run:
    python3 -m experiments.probe_injection_gradient \
        > outputs/probe_injection_gradient.log 2>&1
"""
from __future__ import annotations
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron.classification import masked_cross_entropy
from experiments.bench_chained_15task import (
    INPUT_DIM, L0_WIDTH, INIT_CLASSES,
)


SEED = 0


def build_post_curriculum_net(
    initial_h: int = 32,
    init_classes: int = 2,
    n_growth_events: int = 8,    # mimic ~2 tasks of growth
    n_archive: int = 1,
    n_starve: int = 2,
    seed: int = SEED,
) -> TrioronNetwork:
    """Build a net resembling end-of-curriculum grown_capped_dream:
    frozen L0, grown L1, head with multiple input columns added,
    routing-scale starvation, and a few archived rows."""
    torch.manual_seed(seed)
    net = TrioronNetwork(
        [
            (INPUT_DIM, L0_WIDTH, "relu"),
            (L0_WIDTH, initial_h, "relu"),
            (initial_h, init_classes, "linear"),
        ]
    )
    net.layers[0].W.requires_grad_(False)
    net.layers[0].b.requires_grad_(False)

    # Simulate growth on L1 (each event also extends head fan-in by 1).
    for k in range(n_growth_events):
        net.grow_layer(1, init_vec=None, peer_init_for_next=None, task_idx=k)

    # Extend head from init_classes (2) up to 30 (chained-15 finishes here).
    head = net.layers[-1]
    while head.n_nodes < 30:
        head.grow_node(init_vec=None, task_idx=0)

    # Dream-rescue mutations on L1: starve a few rows.
    with torch.no_grad():
        net.layers[1].routing_scale[0] = 0.45
        if n_starve >= 2:
            net.layers[1].routing_scale[2] = 0.05
        if n_starve >= 3:
            net.layers[1].routing_scale[5] = 0.0
    # Archive a row.
    for k in range(n_archive):
        net.layers[1].archive_row(3 + k)

    # Anchor (mimics consolidate firing once at end-of-task).
    net.anchor_all()
    return net


def main() -> int:
    print("=" * 78)
    print("Injection gradient/loss equivalence probe")
    print("=" * 78)
    torch.manual_seed(SEED)

    net = build_post_curriculum_net()
    print(f"\nnet arch: {net.n_nodes_per_layer()}")
    print(f"L1 archived: "
          f"{net.layers[1].archived.nonzero(as_tuple=True)[0].tolist()}")
    print(f"L1 routing_scale: {net.layers[1].routing_scale.tolist()[:8]} ...")

    # Build paired data: x is raw, z = L0(x) is the hippocampal code.
    B = 64
    n_seen_classes = 30
    torch.manual_seed(SEED + 1)
    x = torch.randn(B, INPUT_DIM)
    y = torch.randint(0, n_seen_classes, (B,))
    with torch.no_grad():
        z = net.layers[0](x)

    all_seen = list(range(n_seen_classes))

    # ----- Path A: memory rehearsal -----
    net.zero_grad(set_to_none=True)
    logits_A = net(x)
    l_A = masked_cross_entropy(logits_A, y, active_classes=all_seen)
    l_A.backward()
    grad_L1_A = net.layers[1].W.grad.detach().clone()
    grad_head_A = net.layers[-1].W.grad.detach().clone()

    # ----- Path B: hippocampal -----
    net.zero_grad(set_to_none=True)
    logits_B = net.forward_from_layer(z, start_layer=1)
    l_B = masked_cross_entropy(logits_B, y, active_classes=all_seen)
    l_B.backward()
    grad_L1_B = net.layers[1].W.grad.detach().clone()
    grad_head_B = net.layers[-1].W.grad.detach().clone()

    # ----- Compare -----
    print()
    print("Loss comparison:")
    print(f"  l_A (memory)       = {float(l_A):.6f}")
    print(f"  l_B (hippocampal)  = {float(l_B):.6f}")
    print(f"  |l_A − l_B|        = {abs(float(l_A) - float(l_B)):.2e}")

    print()
    print("Logits comparison:")
    print(f"  max|logits_A − logits_B| = {(logits_A - logits_B).abs().max().item():.2e}")

    print()
    print("Gradient comparison (after backward):")
    diff_L1 = (grad_L1_A - grad_L1_B).abs()
    diff_head = (grad_head_A - grad_head_B).abs()
    print(f"  L1.W.grad   shape {tuple(grad_L1_A.shape)}  "
          f"max|Δ| = {diff_L1.max().item():.2e}  "
          f"mean|Δ| = {diff_L1.mean().item():.2e}")
    print(f"            |grad_A| max = {grad_L1_A.abs().max().item():.4f}  "
          f"|grad_B| max = {grad_L1_B.abs().max().item():.4f}")
    print(f"  head.W.grad shape {tuple(grad_head_A.shape)}  "
          f"max|Δ| = {diff_head.max().item():.2e}  "
          f"mean|Δ| = {diff_head.mean().item():.2e}")
    print(f"            |grad_A| max = {grad_head_A.abs().max().item():.4f}  "
          f"|grad_B| max = {grad_head_B.abs().max().item():.4f}")

    ok = (diff_L1.max().item() < 1e-5
          and diff_head.max().item() < 1e-5
          and abs(float(l_A) - float(l_B)) < 1e-5)
    print()
    if ok:
        print("VERDICT: PASS — losses and gradients are identical between paths.")
        print()
        print("Implication: forward + backward through the two injection")
        print("paths is mathematically equivalent. The bench-time gap")
        print("(memory 0.62 vs hippo 0.27 on grown arms) must come from")
        print("something OUTSIDE the loss path:")
        print("  1. Per-batch sampling distribution differs (class balance,")
        print("     replacement policy, batch composition).")
        print("  2. Storage timing differs (memory stores raw x; hippo")
        print("     stores L0(x) snapshotted at consolidate). For frozen")
        print("     L0 these are identical, so this should NOT matter — but")
        print("     verify the storage path doesn't apply any subtle")
        print("     transformation (dtype cast, normalization, ReLU mask).")
        print("  3. The batch the bench actually feeds at each step.")
        print("     Memory.sample(64) draws 64 from a union of 100×n_tasks")
        print("     unique reals; HippocampalBuffer.sample(64) draws 64")
        print("     class-uniform with replacement from K codes per class.")
        print("     At K=50 the per-class diversity matches, but per-batch")
        print("     class composition variance differs (with-replacement")
        print("     over 30 classes vs without-replacement over the union).")
        return 0
    else:
        print("VERDICT: FAIL — divergence found.")
        print("Inspect which gradient component differs and why.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
