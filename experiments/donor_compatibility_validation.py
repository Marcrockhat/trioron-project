"""Validation: does pre-absorb donor_compatibility_score predict the
post-absorb full-softmax outcome?

Plan
----
For each seed in the n=12 sentinel set:
  1. Train donor A (tasks 0, 1) and donor B (tasks 2, 3).
  2. Compute pre-absorb compatibility score: max + mean cross-donor
     L1-centroid cosine.
  3. Run pool_matched_absorb + manifold-merge + settle_head_via_manifold.
  4. Eval full-softmax on A-tasks and B-tasks post-absorb.

Then correlate (Pearson + Spearman):
  compatibility_max_cos  vs  post-absorb combined full-softmax

If correlation is strong (|r| > 0.6), the score is a usable gate
signal. Wire it into pool_matched_absorb as an opt-in refuse-when-
above-threshold gate in a follow-up commit.
"""
from __future__ import annotations

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
from trioron.manifold import donor_compatibility_score, post_absorb_separability


SETTLE_KWARGS = dict(n_steps=200, batch_size=64, lr=0.01, noise_scale=1.0)


def run_seed(seed, train_views, eval_views, task_class_lists):
    donor_a, _, manifold_a, trained_a = train_donor_lcn(
        seed=seed,
        task_indices=[0, 1],
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs=4,
        label=f"A_seed{seed}",
    )
    donor_b, _, manifold_b, trained_b = train_donor_lcn(
        seed=seed,
        task_indices=[2, 3],
        train_views=train_views,
        eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_epochs=4,
        label=f"B_seed{seed}",
    )

    # PRE-ABSORB compatibility scores
    gen = torch.Generator().manual_seed(seed * 9001)
    score = donor_compatibility_score(
        donor_a, donor_b, manifold_a, manifold_b,
        n_samples=256, noise_scale=1.0, generator=gen,
    )
    gen2 = torch.Generator().manual_seed(seed * 9001)
    sep = post_absorb_separability(
        donor_a, donor_b, manifold_a, manifold_b,
        n_samples=256, noise_scale=1.0, generator=gen2,
    )

    # Standard absorb + settle path (mirrors the multiseed sentinel)
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
        "score_max": score["max_cosine"],
        "score_mean": score["mean_cosine"],
        "max_pair": score["max_pair"],
        "sep_min": sep["min_separation"],
        "sep_mean": sep["mean_separation"],
        "sep_max_cos": sep["max_cosine"],
        "sep_max_pair": sep["max_pair"],
        "n_pairs": score["n_pairs"],
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
    """Spearman rank correlation."""
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


def main(seeds=None) -> int:
    seeds = seeds or SEEDS
    print("=" * 78)
    print(f"Donor compatibility validation (n={len(seeds)} seeds)")
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
        row = run_seed(seed, train_views, eval_views, task_class_lists)
        rows.append(row)
        print(
            f"  seed={seed} in {time.time()-t0:.0f}s  "
            f"score_max={row['score_max']:.3f}  "
            f"score_mean={row['score_mean']:.3f}  "
            f"max_pair={row['max_pair']}  "
            f"full=({row['a_full']:.3f},{row['b_full']:.3f})  "
            f"combined={row['combined_full']:.3f}"
        )

    print(f"\nTotal: {time.time()-t_total:.0f}s")
    print("\n" + "=" * 78)
    print(f"CORRELATION (n={len(rows)})  score_max vs post-absorb full")
    print("=" * 78)

    a_full = [r["a_full"] for r in rows]
    b_full = [r["b_full"] for r in rows]
    combined = [r["combined_full"] for r in rows]

    signals = [
        ("score_max",   [r["score_max"]   for r in rows]),
        ("score_mean",  [r["score_mean"]  for r in rows]),
        ("sep_min",     [r["sep_min"]     for r in rows]),
        ("sep_mean",    [r["sep_mean"]    for r in rows]),
        ("sep_max_cos", [r["sep_max_cos"] for r in rows]),
    ]
    for name, xs in signals:
        for ylabel, ys in [
            ("A_full", a_full),
            ("B_full", b_full),
            ("combined_full", combined),
        ]:
            pr = _pearson(xs, ys)
            sr = _spearman(xs, ys)
            print(f"  {name:<12} vs {ylabel:<14}  Pearson r={pr:+.3f}  Spearman ρ={sr:+.3f}")

    print("\nSeeds sorted by sep_min ascending (most-confused → least-confused):")
    print(f"{'seed':<6} {'sep_min':>8} {'sep_mean':>9} {'sep_max_cos':>12} {'sep_max_pair':>14} {'score_max':>10} {'combined':>10} {'a_full':>9} {'b_full':>9}")
    for r in sorted(rows, key=lambda r: r["sep_min"]):
        print(
            f"{r['seed']:<6} {r['sep_min']:>8.3f} {r['sep_mean']:>9.3f} "
            f"{r['sep_max_cos']:>12.3f} {str(r['sep_max_pair']):>14} "
            f"{r['score_max']:>10.3f} {r['combined_full']:>10.3f} "
            f"{r['a_full']:>9.3f} {r['b_full']:>9.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
