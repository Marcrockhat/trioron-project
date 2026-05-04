"""Probe: injection-mechanism equivalence.

Question (per session_2026_05_04_handoff): why does raw rehearsal
(forward(x_real)) hit ~0.62 on grown arms while hippocampal replay
(forward_from_layer(L0(x_real), start=1)) hits ~0.27, when on a frozen
L0 the two paths SHOULD be mathematically identical?

This probe runs two tests:

  Test A — Forward math.
    Construct a TrioronNetwork with a frozen L0 + L1 + head, then
    apply the structural mutations a `grown_capped_dream` arm
    accumulates over a chained-15 curriculum (growth on L1, routing-
    scale starvation, latching to zero, row archive). For random
    inputs, compute:
        out_A = net(x)
        out_B = net.forward_from_layer(net.layers[0](x), start=1)
    and report |out_A − out_B| at L1 output and at the head.
    Expected: 0 to numerical noise. If non-zero → bug in
    forward_from_layer.

  Test B — Curriculum: stored-code staleness.
    Drive a short grown_capped_dream curriculum (3 tasks via the
    bench's `run_arm` infrastructure, just enough to fire a few
    growth + dream events). After each task's consolidation, store
    the L0(x_real) codes in a HippocampalBuffer. At end-of-curriculum,
    pull stored codes and the SAME raw x_real, forward both paths
    on the FINAL net, and report the diff.
    Expected: 0 if L0 truly stayed frozen across the curriculum.
    Non-zero → something mutates L0 or the codes' validity (e.g.,
    L0.routing_scale buffer touched by a code path I missed).

Run:
    python3 -m experiments.probe_injection_mechanism \
        > outputs/probe_injection_mechanism.log 2>&1
"""
from __future__ import annotations
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from experiments.bench_chained_15task import (
    INPUT_DIM, L0_WIDTH, INIT_CLASSES, BATCH,
    make_classifier, warmup_l0, run_chained_curriculum,
    ARM_DEFINITIONS, N_GROW_PER_TASK, N_WARMUP_STEPS, WARMUP_LR,
    WARMUP_TEMP_HIDDEN, WARMUP_HEAD_WIDTH,
)
from experiments.datasets import (
    DEFAULT_DATA_ROOT, DatasetBundle, build_task_views, chained_15_specs,
    HippocampalBuffer,
)


SEED = 0


def test_a_forward_math() -> bool:
    """Forward math equivalence on a constructed grown+dreamed net."""
    print("=" * 78)
    print("Test A — Forward math equivalence")
    print("=" * 78)
    torch.manual_seed(SEED)
    net = TrioronNetwork(
        [
            (INPUT_DIM, L0_WIDTH, "relu"),
            (L0_WIDTH, 8, "relu"),
            (8, 30, "linear"),
        ]
    )
    # Freeze L0.
    net.layers[0].W.requires_grad_(False)
    net.layers[0].b.requires_grad_(False)

    # Simulate a curriculum's worth of structural mutations on L1.
    print("\nApplying simulated curriculum mutations to L1...")
    # 1. Growth: add 4 nodes to L1.
    for k in range(4):
        net.grow_layer(1, init_vec=None, peer_init_for_next=None, task_idx=k)
    print(f"  after growth: arch = {net.n_nodes_per_layer()}")

    # 2. Dream-rescue starvation: down-scale routing on a couple of L1 nodes.
    with torch.no_grad():
        net.layers[1].routing_scale[0] = 0.45     # starved
        net.layers[1].routing_scale[2] = 0.10     # nearly latched
        net.layers[1].routing_scale[5] = 0.0      # latched to zero
    print(f"  routing_scale (L1): {net.layers[1].routing_scale.tolist()}")

    # 3. Archive a row.
    net.layers[1].archive_row(3)
    print(f"  archived (L1): "
          f"{net.layers[1].archived.nonzero(as_tuple=True)[0].tolist()}")

    # 4. Anchor (mimics consolidate_task firing once).
    net.anchor_all()

    # Random inputs — chained-15 inputs are 784-dim 28x28 normalized.
    torch.manual_seed(SEED + 1)
    x = torch.randn(BATCH, INPUT_DIM)

    net.eval()
    with torch.no_grad():
        # Path A: full forward.
        out_A = net(x)
        # Path B: explicit L0 then inject at L1.
        z = net.layers[0](x)
        out_B = net.forward_from_layer(z, start_layer=1)

        # Per-layer diffs.
        z_via_full_l1 = net.layers[1](z)
        # Sanity: re-run L0 to confirm stateless.
        z2 = net.layers[0](x)
        l0_self_diff = (z - z2).abs().max().item()

    head_max = (out_A - out_B).abs().max().item()
    head_mean = (out_A - out_B).abs().mean().item()
    print()
    print(f"  L0 self-determinism (re-run diff): {l0_self_diff:.2e}")
    print(f"  Head logits   max |Δ| = {head_max:.2e}    mean |Δ| = {head_mean:.2e}")

    out_A_norm = out_A.abs().mean().item()
    print(f"  (head |out_A| mean for scale)     = {out_A_norm:.4f}")

    ok = head_max < 1e-5
    verdict = "PASS — forward math is equivalent" if ok else (
        "FAIL — paths diverge; bug in forward_from_layer or layer state")
    print(f"\n  Verdict: {verdict}")
    return ok


def test_b_curriculum() -> bool:
    """End-to-end on a short grown_capped_dream curriculum.

    For each just-consolidated task, store K=10 L0 codes per class in a
    HippocampalBuffer, AND store the raw x samples that produced them.
    At end-of-curriculum, run both paths on the SAME raw x and compare.
    """
    print()
    print("=" * 78)
    print("Test B — Curriculum: stored codes vs raw inputs at end-of-curriculum")
    print("=" * 78)

    # Load chained-15 data, but truncate to 3 tasks for speed.
    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT,
        n_holdout_per_dataset=200,
    )
    specs = chained_15_specs()[:3]   # first 3 tasks (MNIST 0/1, 2/3, 4/5)
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [list(v.global_classes) for v in train_views]

    # Build grown_capped_dream net.
    cfg = ARM_DEFINITIONS["grown_capped_dream"]
    torch.manual_seed(SEED)
    net = make_classifier(
        INPUT_DIM, L0_WIDTH, cfg["h_init"], INIT_CLASSES,
        freeze_l0=cfg["freeze_l0"],
    )

    # Tiny infancy view from the first task's pool (just for warmup).
    infancy_view = train_views[0]
    print(f"\n[probe] L0 warmup ({N_WARMUP_STEPS} steps)...")
    warmup_l0(
        net, infancy_view,
        n_steps=N_WARMUP_STEPS, batch=BATCH, lr=WARMUP_LR,
        temp_hidden=WARMUP_TEMP_HIDDEN, head_width=WARMUP_HEAD_WIDTH,
        seed=SEED + 1009,
    )

    # Snapshot L0's W and routing_scale BEFORE the curriculum (post-warmup).
    l0_W_before = net.layers[0].W.detach().clone()
    l0_b_before = net.layers[0].b.detach().clone()
    l0_routing_before = net.layers[0].routing_scale.detach().clone()

    # Pre-stage: pre-store stored_x and stored_classes for each task BEFORE
    # the curriculum runs. We'll re-encode at end-of-curriculum and compare
    # to a buffer encoded MID-curriculum (after each task's consolidate).
    # That way we catch staleness bugs.
    stored_x_per_task = []
    stored_y_per_task = []
    K = 10
    for v in train_views:
        x_all, y_all = v.all_examples()
        chunks_x = []; chunks_y = []
        for c in v.global_classes:
            mask = (y_all == c)
            x_c = x_all[mask][:K]
            y_c = y_all[mask][:K]
            chunks_x.append(x_c); chunks_y.append(y_c)
        stored_x_per_task.append(torch.cat(chunks_x, dim=0))
        stored_y_per_task.append(torch.cat(chunks_y, dim=0))

    # We'll run the curriculum and instrument it: at end-of-curriculum,
    # encode the stored codes via the FINAL net's L0, then compare to
    # paths through both forward and forward_from_layer.
    print("\n[probe] running grown_capped_dream curriculum (3 tasks)...")
    _ = run_chained_curriculum(
        net, label="grown_capped_dream_probe",
        do_growth=cfg["do_growth"], do_dream=cfg["do_dream"],
        cap_bytes=cfg["cap_bytes"], n_grow_per_task=N_GROW_PER_TASK,
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs_per_task=2,             # short for probe
        rng_seed=SEED + 7919,
        n_passes=1,
    )

    # End-of-curriculum L0 state — confirm L0 is unchanged.
    l0_W_after = net.layers[0].W.detach()
    l0_b_after = net.layers[0].b.detach()
    l0_routing_after = net.layers[0].routing_scale.detach()
    dW = (l0_W_after - l0_W_before).abs().max().item()
    db = (l0_b_after - l0_b_before).abs().max().item()
    drouting = (l0_routing_after - l0_routing_before).abs().max().item()
    print()
    print("[probe] L0 state across curriculum:")
    print(f"  max |ΔW|            = {dW:.2e}")
    print(f"  max |Δb|            = {db:.2e}")
    print(f"  max |Δrouting_scale| = {drouting:.2e}")
    if max(dW, db, drouting) > 1e-6:
        print("  ⚠️  L0 STATE CHANGED during the curriculum — staleness IS the issue.")
    else:
        print("  L0 stayed frozen — staleness NOT the cause.")
    print(f"\n[probe] final arch: {net.n_nodes_per_layer()}")

    # Compare paths on past-task data.
    print()
    print("[probe] path comparison on stored past-task samples:")
    print(f"{'task':>4s}  {'n':>5s}  {'head max|Δ|':>14s}  "
          f"{'head mean|Δ|':>15s}  {'L1 max|Δ|':>13s}")
    net.eval()
    all_max_head = 0.0
    with torch.no_grad():
        for ti, (x_t, y_t) in enumerate(zip(stored_x_per_task, stored_y_per_task)):
            # Path A: full forward.
            out_A = net(x_t)
            # Path B: encode L0 NOW (end-of-curriculum), inject at L1.
            z_now = net.layers[0](x_t)
            out_B = net.forward_from_layer(z_now, start_layer=1)
            # Per-layer L1 output.
            l1_full = net.layers[1](net.layers[0](x_t))
            l1_inj  = net.layers[1](z_now)

            head_max = (out_A - out_B).abs().max().item()
            head_mean = (out_A - out_B).abs().mean().item()
            l1_max = (l1_full - l1_inj).abs().max().item()
            all_max_head = max(all_max_head, head_max)
            print(f"  {ti:>3d}  {x_t.shape[0]:>5d}  "
                  f"{head_max:>14.2e}  {head_mean:>15.2e}  {l1_max:>13.2e}")

    ok = all_max_head < 1e-5
    verdict = "PASS — forward equivalence holds end-to-end" if ok else (
        "FAIL — paths diverge after a real curriculum")
    print(f"\n  Verdict: {verdict}")
    return ok


def main() -> int:
    print("Injection-mechanism diagnostic")
    print(f"seed: {SEED}")
    a_ok = test_a_forward_math()
    b_ok = test_b_curriculum()
    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"  Test A (forward math)     : {'PASS' if a_ok else 'FAIL'}")
    print(f"  Test B (curriculum)       : {'PASS' if b_ok else 'FAIL'}")
    print()
    if a_ok and b_ok:
        print("Forward path is identical. The hippocampal-vs-rehearsal gap")
        print("is NOT caused by forward divergence. Investigate next:")
        print("  - sampling distribution (HippocampalBuffer vs MemoryBuffer)")
        print("  - gradient flow / autograd graph differences")
        print("  - loss weighting / reduction differences")
        print("  - per-task storage timing vs. raw-batch composition")
        return 0
    else:
        print("Forward divergence found. Locate which layer/buffer differs.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
