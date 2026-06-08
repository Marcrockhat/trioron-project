"""Fine-comparison-over-time: delayed match-to-sample (DMS) on the substrate.

The coarse grounded task didn't induce cortex-like organization (no abstraction
gradient, no wiring modularity) — but coarse classification doesn't DEMAND
composition (Rocky: reasoning isn't for eyesight). DMS is the canonical FINE
temporal task: the same stimuli appear in both classes; only the RELATION
(does the test stimulus match the earlier sample?) separates them. Individually
every percept is non-discriminative — like telling good-human from bad-human,
the distinction is composed over time, not sensed.

Setup
-----
Trial: [sample(cat a)] → filler×(delay-1) → [test(cat b)]. Label = match (a==b)
vs non-match, balanced 50/50. The substrate carries its own leaky interior
trace across steps (BPTT) — internal recurrence, as validated by the Axis-7
gate. Topology is pre-grown (so the trace dim is fixed during training).

Arms
----
  temporal-off  : memoryless (trace zeroed) → can't compare → ~0.5 (chance).
  bipartite     : temporal-on, width-only topology (rank policy strict).
  self-arrange  : temporal-on, self-organized depth (rank policy relaxed).

Questions: (1) does memory solve DMS? (2) does DEPTH help on a FINE task where
it didn't on the coarse one? (3) does the cortex abstraction gradient emerge —
do deeper cells encode the match/non-match decision (the abstract relational
variable) more than shallow ones?  Structural report via structural_metrics.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trioron.lifecycle import divide, GrowthConfig
from trioron.learning.manifold import get_interior_ids
from experiments.bench_arena_hierarchical import build_substrate, interior_parents
from experiments.bench_grounded_world import N_SENSE
from experiments.structural_metrics import organization_metrics, morphometrics


def make_prototypes(k, gen):
    """K distinct category prototypes (binary percepts over N_SENSE)."""
    return (torch.rand(k, N_SENSE, generator=gen) < 0.4).float()


def dms_batch(b, protos, delay, noise, gen):
    """Balanced match/non-match delayed-match-to-sample trials."""
    k = protos.shape[0]
    a = torch.randint(0, k, (b,), generator=gen)
    match = torch.rand(b, generator=gen) < 0.5
    bcat = a.clone()
    # for non-match, pick a different category
    for i in range(b):
        if not match[i]:
            bcat[i] = (a[i] + torch.randint(1, k, (1,), generator=gen)) % k
    sample = protos[a] + noise * torch.randn(b, N_SENSE, generator=gen)
    test = protos[bcat] + noise * torch.randn(b, N_SENSE, generator=gen)
    fill = (torch.rand(b, max(0, delay - 1), N_SENSE, generator=gen) < 0.2).float()
    if fill.numel():
        fill = fill + noise * torch.randn_like(fill)
    return sample, fill, test, match.long()


def pregrow(arm, sub, budget):
    cfg = GrowthConfig(same_rank_edges=(arm == "self-arrange"))
    if arm == "off":
        cfg = GrowthConfig(same_rank_edges=False)
    grown = 0
    while grown < budget:
        ip = interior_parents(sub.arena)
        if not ip:
            break
        p = ip[torch.randint(0, len(ip), (1,)).item()]
        if divide(sub.arena, p, cfg):
            grown += 1
    sub.compile()
    return grown


def run_arm(arm, *, k, delay, h_init, budget, cap, seed, epochs, batch, lr,
            lam, noise):
    H = h_init + budget
    g = torch.Generator().manual_seed(seed)
    protos = make_prototypes(k, g)
    sub = build_substrate(N_SENSE + H, 2, h_init, cap, seed)
    sub.compile()
    torch.manual_seed(seed + 500)
    pregrow(arm, sub, budget)
    sub.prepare_training()
    iids = get_interior_ids(sub.arena).long()
    Htr = iids.numel()
    temporal_on = (arm != "off")
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)

    def run_seq(sample, fill, test, collect=False):
        b = sample.shape[0]
        m = torch.zeros(b, Htr)
        steps = [sample] + [fill[:, t] for t in range(fill.shape[1])] + [test]
        logits = None
        for percept in steps:
            tin = m if temporal_on else torch.zeros(b, Htr)
            logits = sub(torch.cat([percept, tin], dim=1))
            if temporal_on:
                m = lam * m + (1 - lam) * sub.live_activations[:, iids]
        dec_acts = sub.last_activations[:, iids] if collect else None
        return logits, dec_acts

    for ep in range(epochs):
        for _ in range(40):
            s, f, t, y = dms_batch(batch, protos, delay, noise, g)
            logits, _ = run_seq(s, f, t)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); sub.zero_dormant_grads(); opt.step()

    # eval + collect decision-step activations for structural analysis
    with torch.no_grad():
        s, f, t, y = dms_batch(2000, protos, delay, noise, g)
        correct = 0; acts_all = []; y_all = []
        for i in range(0, 2000, 250):
            sl = slice(i, i + 250)
            logits, dec = run_seq(s[sl], f[sl], t[sl], collect=True)
            correct += int((logits.argmax(-1) == y[sl]).sum())
            acts_all.append(dec); y_all.append(y[sl])
        acc = correct / 2000
        acts = torch.cat(acts_all); y_eval = torch.cat(y_all)

    # structural report (w.r.t. the match/non-match decision)
    a = sub.arena; a.rank_dirty = True
    ranks = {int(c): int(a.rank[c].item()) for c in iids.tolist()}
    iset = set(iids.tolist()); ne = a.edge_cursor
    src = a.edge_src[:ne].tolist(); dst = a.edge_dst[:ne].tolist()
    edges_ii = [(s_, d_) for s_, d_ in zip(src, dst)
                if s_ != d_ and a.alive[s_] and a.alive[d_] and s_ in iset and d_ in iset]
    som = organization_metrics(sub, None, y_eval, iids, ranks, 2, edges_ii, acts=acts)
    morph = morphometrics(sub)
    return acc, {**morph, **som}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--h-init", type=int, default=8)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--k", type=int, default=4, help="sample categories")
    ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--cap", type=int, default=400_000)
    ap.add_argument("--arms", default="off,bipartite,self-arrange")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 10

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"bench_fine_temporal (DMS): seeds={args.seeds} epochs={args.epochs} "
          f"K={args.k} delay={args.delay} h_init={args.h_init} budget={args.budget} "
          f"λ={args.lam} arms={arms}  (chance=0.500)")

    results = {a: {"acc": [], "rows": []} for a in arms}
    for seed in range(args.seeds):
        for arm in arms:
            t0 = time.time()
            acc, rep = run_arm(arm, k=args.k, delay=args.delay, h_init=args.h_init,
                               budget=args.budget, cap=args.cap, seed=seed,
                               epochs=args.epochs, batch=args.batch, lr=args.lr,
                               lam=args.lam, noise=args.noise)
            results[arm]["acc"].append(acc); results[arm]["rows"].append(rep)
            print(f"  [s{seed} {arm:>12s}] acc={acc:.3f} depth={rep['effective_depth']} "
                  f"ii_edges={rep['ii_edges']} abstraction_grad={rep['abstraction_grad']:.3f} "
                  f"assort={rep['assortativity']:.3f} sparseness={rep['sparseness']:.3f} "
                  f"({time.time()-t0:.0f}s)")

    def ms(xs):
        xs = [v for v in xs if v == v]
        return (statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (float("nan"), 0.0)
    print(f"\n=== aggregate (n={args.seeds}) ===")
    for arm in arms:
        am, as_ = ms(results[arm]["acc"])
        rows = results[arm]["rows"]
        ag = ms([r["abstraction_grad"] for r in rows])
        asrt = ms([r["assortativity"] for r in rows])
        sp = ms([r["sparseness"] for r in rows])
        sil = ms([r["module_silhouette"] for r in rows])
        dep = ms([r["effective_depth"] for r in rows])
        print(f"  {arm:>12s}  acc={am:.3f}±{as_:.3f}  depth={dep[0]:.1f}  "
              f"abstraction_grad={ag[0]:+.3f}±{ag[1]:.2f}  assort={asrt[0]:+.3f}  "
              f"sparseness={sp[0]:.3f}  silhouette={sil[0]:.3f}")
    print("\n(abstraction_grad > 0 ⇒ deeper cells encode match/non-match more — "
          "the cortex hierarchy hallmark, on a task that demands composition.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
