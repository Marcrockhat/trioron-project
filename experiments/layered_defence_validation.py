"""Layered-defence validation: training-time L1 orthogonality regulariser
+ deployment-time simulation gate.

Per Rocky 2026-05-23, both layers preserve the paste-and-go invariant
(no cross-donor coordination needed):

  (1) Per-donor sparsity reg pushes each donor's L1 toward a sparse
      subspace independently. Statistically reduces cross-donor
      feature-direction collision probability.

  (2) Simulation gate runs the absorption + settle on a deep copy
      and scores the result; deployment refuses below threshold.

This script measures BOTH on the same n=12 seeds:
  - baseline (no reg, no gate) → already characterised in n=12_provenance
  - reg-on (LAMBDA_L1_ORTHO=0.1) → does the reg lift outcomes?
  - simulation_score per seed → does it predict outcomes?

If reg lifts the bad-seed outcomes AND simulation_score predicts the
residual collapses with usable precision, the layered defence works.

Override via env: ABSORB_SEEDS (csv), L1_ORTHO_LAMBDA (default 0.1).
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

from experiments.datasets import (
    DatasetBundle,
    DEFAULT_DATA_ROOT,
    chained_15_specs,
    build_task_views,
)
from experiments.pool_matched_absorb_lcn_multiseed import SEEDS, _locality_mean
from experiments.pool_matched_absorb_lcn_test import (
    train_donor_lcn,
    extend_l1_lcn_mask,
)
from experiments.pool_matched_absorb_test import eval_on_tasks
from experiments.pool_matched_absorption import (
    pool_matched_absorb,
    merge_manifold,
)
from trioron.api import settle_head_via_manifold
from trioron.manifold import simulate_absorption_score


SETTLE_KWARGS = dict(n_steps=200, batch_size=64, lr=0.01, noise_scale=1.0)


def run_seed(seed, lambda_ortho, train_views, eval_views, task_class_lists):
    donor_a, _, manifold_a, trained_a = train_donor_lcn(
        seed=seed,
        task_indices=[0, 1],
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs=4,
        label=f"A_seed{seed}",
        lambda_l1_ortho=lambda_ortho,
    )
    donor_b, _, manifold_b, trained_b = train_donor_lcn(
        seed=seed,
        task_indices=[2, 3],
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs=4,
        label=f"B_seed{seed}",
        lambda_l1_ortho=lambda_ortho,
    )

    # Pre-absorb simulation gate
    gen = torch.Generator().manual_seed(seed * 9001 + 7)
    sim_score = simulate_absorption_score(
        donor_a, donor_b, manifold_a, manifold_b,
        layer_idx=1,
        settle_steps=50,
        score_samples=256,
        generator=gen,
        pool_matched_absorb_kwargs=dict(
            grid_size=4,
            snap_to_pool_centroid=False,
            donor_trained_classes=trained_b,
            recipient_trained_classes=trained_a,
        ),
    )

    # Commit the real absorption
    pool_matched_absorb(
        donor_a, donor_b,
        grid_size=4, pool_capacity=None,
        snap_to_pool_centroid=False,
        donor_trained_classes=trained_b,
        recipient_trained_classes=trained_a,
    )
    extend_l1_lcn_mask(donor_a)
    locality = _locality_mean(donor_a.layers[1])
    merge_manifold(manifold_a, manifold_b)
    combined_seen = sorted(set(trained_a) | set(trained_b))
    torch.manual_seed(seed)
    settle_head_via_manifold(donor_a, manifold_a, combined_seen, **SETTLE_KWARGS)

    a_ta, a_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [0, 1], label_prefix="    ",
    )
    b_ta, b_full = eval_on_tasks(
        donor_a, eval_views, task_class_lists, [2, 3], label_prefix="    ",
    )

    return {
        "seed": seed,
        "lambda_ortho": lambda_ortho,
        "sim_score": sim_score,
        "locality": locality,
        "a_ta": a_ta,
        "a_full": a_full,
        "b_ta": b_ta,
        "b_full": b_full,
        "combined_full": (a_full + b_full) / 2,
    }


def _pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _spearman(xs, ys):
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return _pearson(ranks(xs), ranks(ys))


def main(seeds=None, lambda_ortho=None) -> int:
    seeds = seeds or _parse_seeds() or SEEDS
    if lambda_ortho is None:
        lambda_ortho = float(os.environ.get("L1_ORTHO_LAMBDA", "0.1"))

    print("=" * 78)
    print(
        f"Layered-defence validation  λ_l1_ortho={lambda_ortho}  "
        f"n_seeds={len(seeds)}: {seeds}"
    )
    print("=" * 78)

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=DEFAULT_DATA_ROOT,
        n_holdout_per_dataset=0,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    rows = []
    t_total = time.time()
    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        t0 = time.time()
        row = run_seed(seed, lambda_ortho, train_views, eval_views, task_class_lists)
        rows.append(row)
        print(
            f"  seed={seed} in {time.time()-t0:.0f}s  "
            f"sim_score={row['sim_score']:.3f}  "
            f"full=({row['a_full']:.3f},{row['b_full']:.3f})  "
            f"combined={row['combined_full']:.3f}"
        )

    print(f"\nTotal: {time.time()-t_total:.0f}s")
    print("\n" + "=" * 78)
    print(f"AGGREGATE (n={len(rows)})  λ_l1_ortho={lambda_ortho}")
    print("=" * 78)

    a_full = [r["a_full"] for r in rows]
    b_full = [r["b_full"] for r in rows]
    combined = [r["combined_full"] for r in rows]
    sim = [r["sim_score"] for r in rows]
    a_ta = [r["a_ta"] for r in rows]
    b_ta = [r["b_ta"] for r in rows]
    loc = [r["locality"] for r in rows]

    def _stat(label, x):
        m = statistics.mean(x); s = statistics.stdev(x) if len(x) > 1 else 0.0
        return f"  {label:>32}  mean={m:+.4f}  σ={s:.4f}"
    print(_stat("A-tasks full", a_full))
    print(_stat("B-tasks full", b_full))
    print(_stat("combined full", combined))
    print(_stat("A task-aware", a_ta))
    print(_stat("B task-aware", b_ta))
    print(_stat("locality", loc))
    print(_stat("simulation_score (pre-absorb)", sim))

    print("\nsimulation_score vs realised outcome:")
    for ylabel, ys in [("A_full", a_full), ("B_full", b_full), ("combined_full", combined)]:
        pr = _pearson(sim, ys); sr = _spearman(sim, ys)
        print(f"  sim_score vs {ylabel:<14}  Pearson r={pr:+.3f}  Spearman ρ={sr:+.3f}")

    # Gate calibration: at various thresholds, what fraction is refused
    # and what's the mean outcome of the kept seeds?
    print("\nSimulated gate (refuse if sim_score < threshold):")
    print(f"{'threshold':>10} {'refused':>8} {'kept':>5} {'kept_combined_mean':>20}")
    for thr in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        kept = [r["combined_full"] for r in rows if r["sim_score"] >= thr]
        refused = len(rows) - len(kept)
        kmean = statistics.mean(kept) if kept else float("nan")
        print(f"{thr:>10.2f} {refused:>8} {len(kept):>5} {kmean:>20.4f}")

    print("\nSeeds sorted by sim_score ascending:")
    print(f"{'seed':<6} {'sim_score':>10} {'combined':>10} {'a_full':>9} {'b_full':>9} {'a_ta':>7} {'b_ta':>7}")
    for r in sorted(rows, key=lambda r: r["sim_score"]):
        print(
            f"{r['seed']:<6} {r['sim_score']:>10.3f} {r['combined_full']:>10.3f} "
            f"{r['a_full']:>9.3f} {r['b_full']:>9.3f} "
            f"{r['a_ta']:>7.3f} {r['b_ta']:>7.3f}"
        )
    return 0


def _parse_seeds():
    raw = os.environ.get("ABSORB_SEEDS")
    if not raw:
        return None
    return [int(s) for s in raw.split(",") if s.strip()]


if __name__ == "__main__":
    sys.exit(main())
