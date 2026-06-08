"""Probe: real-seed gradient-ascent engrams vs uniform-noise legacy.

Quick check that ENGRAM_SEED_FROM_REAL=True repairs the off-manifold
collapse documented in probe_engram_diversity 2026-05-04 (engrams 7.5x
narrower than real samples, 9x off-manifold).

Setup:
  - Build a chained-15 frozen-L0 net, warm L0 briefly on infancy view
    (matches bench's grown_capped_dream).
  - Train one MNIST 0/1 task, consolidate (anchor).
  - Build engrams two ways:
      A) seed_from_real=False  (legacy uniform noise init)
      B) seed_from_real=True   (real-sample init, the new construction)
  - For each: report pairwise engram-engram L2 distance (within-class
    spread) and engram-real distance (off-manifold extent), per class
    and aggregate.

Decision rule:
  If real-seed engrams have engram-real distance comparable to real-real
  distance, they're on-manifold and the construction repair is good.

Run:
  python3 -m experiments.probe_engram_real_seed \
      > outputs/probe_engram_real_seed.log 2>&1
"""
from __future__ import annotations
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from experiments.datasets import (
    DEFAULT_DATA_ROOT, DatasetBundle, build_task_views, chained_15_specs,
    EngramBuffer,
)
from experiments.bench_chained_15task import (
    INPUT_DIM, L0_WIDTH, INIT_CLASSES, BATCH,
    make_classifier, warmup_l0,
    _consolidate_engrams, consolidate_task, train_one_task,
    N_WARMUP_STEPS, WARMUP_LR, WARMUP_TEMP_HIDDEN, WARMUP_HEAD_WIDTH,
    ARM_DEFINITIONS, EWC_INTERTASK,
)
from trioron.classification import extend_output_head


SEED = 0
N_PER_CLASS_REAL = 50    # real samples to compare against


def pairwise_l2(A, B):
    """Pairwise L2 distances between rows of A (n, d) and B (m, d).
    Returns (n, m) tensor."""
    return (A.unsqueeze(1) - B.unsqueeze(0)).norm(dim=-1)


def report_for_engrams(label, engrams: EngramBuffer, real_per_class):
    print(f"\n=== {label} ===")
    classes = engrams.stored_classes()
    print(f"stored classes: {classes}")

    # Stack engrams (one per class — current bench builds K=1 per call).
    engram_rows = []
    real_blocks = []
    for c in classes:
        e = engrams._engrams[c]   # (input_dim,) — 1-D per spec
        r = real_per_class[c]     # (N, input_dim)
        engram_rows.append(e)
        real_blocks.append(r)
    E = torch.stack(engram_rows, dim=0)  # (n_classes, input_dim)

    # Engram-engram pairwise (cross-class engram diversity).
    ee = pairwise_l2(E, E)
    eye = torch.eye(ee.shape[0], dtype=torch.bool)
    ee_off = ee.masked_fill(eye, float("nan"))
    print(f"engram-engram (cross-class): "
          f"min = {ee_off.flatten().nanquantile(0.01).item():.3f}  "
          f"mean = {ee_off.nanmean().item():.3f}  "
          f"max = {ee_off.flatten().nanquantile(0.99).item():.3f}")

    # Engram-real (within-class): distance from each engram to its own
    # class's real samples.
    er_per_class = []
    for c, e_row in zip(classes, engram_rows):
        r = real_per_class[c]
        d = (r - e_row).norm(dim=-1)
        er_per_class.append(d)
        print(f"  class {c:>3d}: engram-real (own class) "
              f"min={d.min().item():.3f}  median={d.median().item():.3f}  "
              f"max={d.max().item():.3f}")
    er_all = torch.cat(er_per_class)
    print(f"engram-real (own class) AGGREGATE: "
          f"min = {er_all.min().item():.3f}  "
          f"median = {er_all.median().item():.3f}  "
          f"mean = {er_all.mean().item():.3f}")

    # Real-real (within-class): for reference.
    rr_per_class = []
    for c in classes:
        r = real_per_class[c]
        d = pairwise_l2(r, r)
        # Off-diagonal only.
        n = d.shape[0]
        m = ~torch.eye(n, dtype=torch.bool)
        rr_per_class.append(d[m])
    rr_all = torch.cat(rr_per_class)
    print(f"real-real (within class) AGGREGATE: "
          f"min = {rr_all.min().item():.3f}  "
          f"median = {rr_all.median().item():.3f}  "
          f"mean = {rr_all.mean().item():.3f}")

    ratio = er_all.mean().item() / rr_all.mean().item()
    print(f"\n  Off-manifold ratio (engram-real / real-real) = {ratio:.3f}")
    if ratio > 3.0:
        print("  ⚠️  engrams are off-manifold (>3x further than real-real)")
    elif ratio > 1.5:
        print("  ⚠️  engrams partially off-manifold")
    else:
        print("  ✓ engrams are on-manifold (comparable to real-real spread)")


def main() -> int:
    print("=" * 78)
    print("Engram real-seed vs uniform-init diversity probe")
    print("=" * 78)
    print(f"seed: {SEED}")

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT,
        n_holdout_per_dataset=200,
    )
    specs = chained_15_specs()[:1]   # one task: MNIST 0/1, fast.
    train_views = build_task_views(bundle, specs, split="train")

    # Build grown_capped_dream-style net.
    cfg = ARM_DEFINITIONS["grown_capped_dream"]
    torch.manual_seed(SEED)
    net = make_classifier(
        INPUT_DIM, L0_WIDTH, cfg["h_init"], INIT_CLASSES,
        freeze_l0=cfg["freeze_l0"],
    )
    print(f"\n[probe] L0 warmup ({N_WARMUP_STEPS} steps)...")
    warmup_l0(
        net, train_views[0],
        n_steps=N_WARMUP_STEPS, batch=BATCH, lr=WARMUP_LR,
        temp_hidden=WARMUP_TEMP_HIDDEN, head_width=WARMUP_HEAD_WIDTH,
        seed=SEED + 1009,
    )

    # Train task 1 briefly so we have a meaningful anchored network.
    active = list(train_views[0].global_classes)
    head_size = net.layers[-1].n_nodes
    if max(active) >= head_size:
        extend_output_head(net, max(active) - head_size + 1)
    import torch.optim as optim
    opt = optim.Adam(
        (p for p in net.parameters() if p.requires_grad),
        lr=1e-3,
    )
    print(f"\n[probe] training task 1 (active {active}) for 2 epochs...")
    train_one_task(
        net, 0, train_views[0], active,
        n_epochs=2, opt=opt, ewc_baseline=0.0,
        label="probe", n_total_tasks=1,
    )
    consolidate_task(net, train_views[0], active)

    # Collect real samples per class.
    real_per_class = {}
    x_all, y_all = train_views[0].all_examples()
    for c in active:
        mask = (y_all == c)
        x_c = x_all[mask][:N_PER_CLASS_REAL]
        real_per_class[int(c)] = x_c.detach().clone().view(x_c.shape[0], -1)

    # Build engrams two ways.
    eng_legacy = EngramBuffer()
    eng_realseed = EngramBuffer()
    print("\n[probe] building LEGACY (uniform-noise init) engrams...")
    _consolidate_engrams(
        net, active, eng_legacy,
        seed_from_real=False,
    )
    print("[probe] building REAL-SEED engrams...")
    _consolidate_engrams(
        net, active, eng_realseed,
        train_view=train_views[0],
        seed_from_real=True,
    )

    report_for_engrams("LEGACY (uniform-noise init)", eng_legacy, real_per_class)
    report_for_engrams("REAL-SEED (Gaussian-perturbed real-x init)",
                       eng_realseed, real_per_class)

    print()
    print("=" * 78)
    print("Verdict: compare off-manifold ratios above. If REAL-SEED's ratio")
    print("is ≤ 1.5 and LEGACY's is > 3, the construction repair works.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
