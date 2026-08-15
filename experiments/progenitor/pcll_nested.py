"""Nested PCLL organisms on chained-15 — s049.

Hypothesis (from the s048 gaming result, Rocky's "nested triorons"):
the PCLL chained-15 gap is DOWNSTREAM of perception — s039 diagnosed
cap-128 over-segmentation and a permutation-invariant matched-filter
readout competing across all 30 labels at once; richer front-ends made
the organism WORSE. The gaming arc showed the same failure shape for
flat policies and beat it with leaves + a router. This bench tests the
classification analog: split the monolithic organism into THREE domain
leaves (digits / fashion / letters), each a full gradient-free PCLL
organism, and route per sample with the s047 ManifoldRouter (full-cov
QDA over the sense descriptor — the "manifold-recognition shortlist"
router, the validated non-TD half of the s048 hybrid).

Honesty constraints:
  * total class budget matched to the monolithic cap (128 split 43/43/43
    by default; PCLLNEST_LEAFCAP overrides) — the nest gets no extra
    class capacity;
  * stream order unchanged (tasks 0..14 sequential; a leaf trains only
    while its domain's tasks stream — incremental enrollment);
  * router fitted from stream statistics only (per-domain mu/Sigma over
    descriptors, subsampled) — same storage class as manifold replay,
    no gradients anywhere;
  * baseline = monolithic gabor CENTERED seed 0: task_aware 0.736 /
    full 0.446 (outputs/pcll_chained15_s039_gabor_centered.log).

Arms reported (all on the identical trained leaves):
  routed   — ManifoldRouter picks the leaf, leaf's centered filter picks
             the class (the nest claim);
  oracle   — true domain picks the leaf (routing ceiling);
  concat   — global argmax over the union of all leaves' evidence
             (router-free flat-union control);
  task-aware — task known -> leaf forced + class subset (comparable to
             the monolithic task-aware number).

Run:  PCLL15_SENSE=gabor python3 -m experiments.progenitor.pcll_nested
Env:  PCLL15_* as in run_pcll_chained15 (SENSE/WINDOW/FRAC/SEEDS/
      MANIFOLD/COMPOSER honored); PCLLNEST_LEAFCAP (default 43);
      PCLLNEST_ROUTER_N (descriptor samples kept per domain, def 4000).
"""
from __future__ import annotations

import os
import time

import torch

from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)
from trioron.pcll import PhasecyteNest

from experiments.progenitor import run_pcll_chained15 as base

LEAFCAP = int(os.environ.get("PCLLNEST_LEAFCAP", "43"))
ROUTER_N = int(os.environ.get("PCLLNEST_ROUTER_N", "4000"))
DOMAINS = ("digits", "fashion", "letters")
N_DOM = 3


def run_seed(seed: int, return_state: bool = False):
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=base.DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()[:base.N_TASKS]
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    sense = base.make_sense(seed)
    n_dom = min(N_DOM, (len(specs) + 4) // 5)
    t0 = time.time()

    nest = PhasecyteNest(
        sense, seed=seed, router_n=ROUTER_N, window=base.WINDOW,
        class_cap=LEAFCAP if LEAFCAP > 0 else None,
        manifold=base.MANIFOLD, composer=base.COMPOSER,
        input_shape=base.SHAPE)

    # the continual stream — identical order/subsampling to the monolith;
    # a leaf is enrolled when its domain's first task arrives
    for ti, view in enumerate(train_views):
        dom = ti // 5
        if dom not in nest.leaves:
            pool = torch.cat([v.images for v in
                              train_views[5 * dom:5 * dom + 5]])
            nest.enroll(dom, pool)
        g = torch.Generator().manual_seed(100 * seed + ti)
        order = torch.randperm(len(view.labels_global), generator=g)
        if base.FRAC < 1.0:
            order = order[:max(1, int(base.FRAC * len(order)))]
        X = view.images[order]
        labs = [f"g{int(v):02d}" for v in view.labels_global[order]]
        for w0 in range(0, len(X), base.WINDOW):
            nest.observe(dom, X[w0:w0 + base.WINDOW],
                         labs[w0:w0 + base.WINDOW])
        leaf = nest.leaves[dom]
        print(f"  [task {ti + 1:>2d}/{len(specs)} {view.name} -> "
              f"{DOMAINS[dom]}] classes={len(leaf.mixed.classes)} "
              f"spawned={len(leaf.mixed.specs)} "
              f"t={time.time() - t0:.0f}s", flush=True)

    router = nest.fit_router()
    leaves = [nest.leaves[d] for d in range(n_dom)]

    # eval on the union test set
    Xt = torch.cat([v.images for v in eval_views])
    yt = torch.cat([v.labels_global for v in eval_views])
    dom_true = torch.cat([torch.full((len(v.labels_global),), ti // 5)
                          for ti, v in enumerate(eval_views)])

    dom_pred = nest.route(Xt)
    route_acc = float((dom_pred == dom_true).float().mean())

    names, evid = [], []
    for dom in range(n_dom):
        _, Ec = leaves[dom].evidence(sense, Xt)
        names.append(leaves[dom].names())
        evid.append(Ec)

    def leaf_pred(d: torch.Tensor) -> torch.Tensor:
        out = torch.empty(len(Xt), dtype=torch.long)
        for dom in range(n_dom):
            m = d == dom
            if m.any():
                out[m] = names[dom][evid[dom][m].argmax(1)]
        return out

    full_routed = float((leaf_pred(dom_pred) == yt).float().mean())
    full_oracle = float((leaf_pred(dom_true) == yt).float().mean())
    E_all = torch.cat(evid, dim=1)
    name_all = torch.cat(names)
    full_concat = float((name_all[E_all.argmax(1)] == yt).float().mean())

    per_task, off = [], 0
    for ti, (view, spec) in enumerate(zip(eval_views, specs)):
        dom = ti // 5
        n = len(view.labels_global)
        Et = evid[dom][off:off + n]
        off += n
        cand = torch.isin(names[dom],
                          torch.tensor(spec.global_classes)) \
            .nonzero().squeeze(1)
        if not len(cand):
            per_task.append(0.0)
            continue
        pred = names[dom][cand[Et[:, cand].argmax(1)]]
        per_task.append(float((pred == view.labels_global).float().mean()))
    ta = sum(per_task) / len(per_task)

    n_cls = sum(len(leaves[d].mixed.classes) for d in range(n_dom))
    print(f"  per-task: " + " ".join(f"{a:.2f}" for a in per_task))
    print(f"[seed {seed}] NEST routed full={full_routed:.3f}  "
          f"oracle full={full_oracle:.3f}  concat full={full_concat:.3f}")
    print(f"[seed {seed}] NEST task_aware={ta:.3f}  "
          f"routing_acc={route_acc:.3f}  classes={n_cls} "
          f"(caps {LEAFCAP}x{n_dom})  elapsed={time.time() - t0:.0f}s",
          flush=True)
    print(f"[seed {seed}] monolithic reference (s039 gabor centered): "
          f"task_aware 0.736 / full 0.446")
    if return_state:
        return ta, full_routed, leaves, router
    return ta, full_routed


def main() -> None:
    print(f"NESTED PCLL on chained-15 (s049) — {N_DOM} domain leaves + "
          f"manifold router; sense={base.SENSE}, leafcap={LEAFCAP}, "
          f"frac={base.FRAC}, tasks={base.N_TASKS}, window={base.WINDOW}, "
          f"manifold={'on' if base.MANIFOLD else 'off'}, "
          f"composer={'on' if base.COMPOSER else 'off'}, "
          f"seeds={base.SEEDS}")
    tas, fulls = [], []
    for seed in range(base.SEEDS):
        ta, full = run_seed(seed)
        tas.append(ta)
        fulls.append(full)
    if base.SEEDS > 1:
        import statistics
        print(f"\nmean task_aware {statistics.mean(tas):.3f} "
              f"± {statistics.stdev(tas):.3f}  "
              f"full {statistics.mean(fulls):.3f} "
              f"± {statistics.stdev(fulls):.3f}")


if __name__ == "__main__":
    main()
