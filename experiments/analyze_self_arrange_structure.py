"""What STRUCTURE does self-arrange grow? — full substrate report card, n-seed.

Rocky's questions (2026-05-31): different structure from a self-generated conv
layer? how many clusters? cortex-like (functional, not visible) separation?
Plus internal-state census (cells, dendrites, clusters, axon size).

We can't SEE it (visualizer unfinished), so we MEASURE it: the standardized
Substrate Organization Metrics + morphometrics (experiments/structural_metrics.py),
bipartite (width-only) vs self-arrange (depth), n seeds, on the grounded world.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trioron.lifecycle import divide, GrowthConfig
from trioron.learning.manifold import get_interior_ids
from experiments.bench_arena_hierarchical import build_substrate, interior_parents
from experiments.bench_grounded_world import N_SENSE, N_CONSEQ
from experiments.bench_temporal_gate import balanced_cues
from experiments.structural_metrics import organization_metrics, morphometrics


def grow_and_train(arm, *, h_init=8, budget=40, every=20, epochs=50, lr=6e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x, y = balanced_cues(700, 0.1, g)
    sub = build_substrate(N_SENSE, N_CONSEQ, h_init, 300_000, seed)
    sub.compile(); sub.prepare_training()
    cfg = GrowthConfig(same_rank_edges=(arm == "self-arrange"))
    do_grow = (arm != "no-growth")
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    torch.manual_seed(seed + 1000)
    n = x.shape[0]; grown = 0; step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for b in range(max(1, n // 128)):
            idx = perm[b * 128:(b + 1) * 128]
            loss = torch.nn.functional.cross_entropy(sub(x[idx]), y[idx])
            loss.backward(); sub.zero_dormant_grads(); opt.step(); opt.zero_grad()
            if do_grow and grown < budget and step > 0 and step % every == 0:
                ip = interior_parents(sub.arena)
                if ip:
                    p = ip[torch.randint(0, len(ip), (1,)).item()]
                    if divide(sub.arena, p, cfg):
                        grown += 1; sub.compile()
                        opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
            step += 1
    return sub, x, y


def report_arm(arm, seeds, n_classes=N_CONSEQ, sampler=None):
    rows = []
    for seed in seeds:
        if sampler is None:
            sub, x, y = grow_and_train(arm, seed=seed)
        else:
            sub, x, y = sampler(arm, seed)
        a = sub.arena
        iids = get_interior_ids(a).long()
        a.rank_dirty = True
        ranks = {int(c): int(a.rank[c].item()) for c in iids.tolist()}
        iset = set(iids.tolist())
        ne = a.edge_cursor
        src = a.edge_src[:ne].tolist(); dst = a.edge_dst[:ne].tolist()
        edges_ii = [(s, d) for s, d in zip(src, dst)
                    if s != d and a.alive[s] and a.alive[d] and s in iset and d in iset]
        som = organization_metrics(sub, x, y, iids, ranks, n_classes, edges_ii)
        morph = morphometrics(sub)
        rows.append({**morph, **som})
    return rows


def agg(rows, key):
    xs = [r[key] for r in rows if r[key] == r[key]]   # drop nan
    if not xs:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    s = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def print_report(title, by_arm):
    print(f"\n{'='*66}\n{title}\n{'='*66}")
    morph_keys = ["n_cells", "n_interior", "n_dendrite_cells", "n_recurrent_cells",
                  "n_lineage_clusters", "avg_axon_fanout", "avg_dendritic_fanin", "n_edges"]
    som_keys = ["effective_depth", "max_rank", "ii_edges", "sparseness",
                "assortativity", "abstraction_grad", "module_silhouette", "n_modules"]
    arms = list(by_arm.keys())
    print(f"\n-- MORPHOMETRICS (mean over seeds) --")
    print(f"  {'metric':>20s} " + "".join(f"{a:>16s}" for a in arms))
    for k in morph_keys:
        line = f"  {k:>20s} "
        for a in arms:
            m, s = agg(by_arm[a], k)
            line += f"{m:>10.2f}±{s:<4.1f}"
        print(line)
    print(f"\n-- ORGANIZATION (mean over seeds; chance: assort=0, abstraction=0) --")
    print(f"  {'metric':>20s} " + "".join(f"{a:>16s}" for a in arms))
    for k in som_keys:
        line = f"  {k:>20s} "
        for a in arms:
            m, s = agg(by_arm[a], k)
            line += f"{m:>10.3f}±{s:<4.2f}"
        print(line)


if __name__ == "__main__":
    seeds = list(range(5))
    print(f"Self-arrange structure — coarse grounded world, n={len(seeds)} seeds")
    by_arm = {arm: report_arm(arm, seeds) for arm in ("bipartite", "self-arrange")}
    print_report("COARSE TASK (grounded consequence classification)", by_arm)
    print("\nConv contrast: conv = weight-tied lineage + spatial receptors")
    print("(shared weight tensor); self-depth = rank-layered general edges,")
    print("no weight-tying, no conv gene — topological, not lineage/weight.")
