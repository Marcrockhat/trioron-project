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

from trioron.core import construct
from trioron.learning.manifold import ManifoldArchive
from trioron.learning.router import ManifoldRouter
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)
from trioron.pcll import (MixedStreamController, PerceptionGenesis,
                          germline_base)

from experiments.progenitor import run_pcll_chained15 as base

LEAFCAP = int(os.environ.get("PCLLNEST_LEAFCAP", "43"))
ROUTER_N = int(os.environ.get("PCLLNEST_ROUTER_N", "4000"))
DOMAINS = ("digits", "fashion", "letters")
N_DOM = 3


class Leaf:
    def __init__(self, sense, pool: torch.Tensor, seed: int, dom: int):
        torch.manual_seed(1000 * seed + dom)
        self.sub = construct(germline_base, capacity=8192)
        pg = PerceptionGenesis(self.sub, input_shape=base.SHAPE)
        g = torch.Generator().manual_seed(1000 * seed + dom)
        gidx = torch.randperm(len(pool), generator=g)[:base.WINDOW]
        pg.feed(sense(pool[gidx]))
        rep = self.sub.end_task()
        n_rec = int(self.sub.scheduler._plan.receptor_ids.numel())
        print(f"  [genesis {DOMAINS[dom]}] starved={len(rep.starved)} "
              f"merged={len(rep.merged)}->{len(rep.regions)}  "
              f"receptors->{n_rec}", flush=True)
        self.mixed = MixedStreamController(
            self.sub, stress=pg.router, adopt=pg.controller,
            manifold=base.MANIFOLD, composer=base.COMPOSER,
            class_cap=LEAFCAP if LEAFCAP > 0 else None)


def run_seed(seed: int, return_state: bool = False):
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=base.DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()[:base.N_TASKS]
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    sense = base.make_sense(seed)
    n_dom = min(N_DOM, (len(specs) + 4) // 5)
    t0 = time.time()

    leaves: list[Leaf | None] = [None] * n_dom
    router_buf: list[list[torch.Tensor]] = [[] for _ in range(n_dom)]

    # the continual stream — identical order/subsampling to the monolith;
    # a leaf is enrolled when its domain's first task arrives
    for ti, view in enumerate(train_views):
        dom = ti // 5
        if leaves[dom] is None:
            pool = torch.cat([v.images for v in
                              train_views[5 * dom:5 * dom + 5]])
            leaves[dom] = Leaf(sense, pool, seed, dom)
        leaf = leaves[dom]
        g = torch.Generator().manual_seed(100 * seed + ti)
        order = torch.randperm(len(view.labels_global), generator=g)
        if base.FRAC < 1.0:
            order = order[:max(1, int(base.FRAC * len(order)))]
        X = view.images[order]
        labs = [f"g{int(v):02d}" for v in view.labels_global[order]]
        for w0 in range(0, len(X), base.WINDOW):
            xw = sense(X[w0:w0 + base.WINDOW])
            leaf.mixed.observe(xw, labels=labs[w0:w0 + base.WINDOW])
            leaf.sub.end_task()
            router_buf[dom].append(xw)
        print(f"  [task {ti + 1:>2d}/{len(specs)} {view.name} -> "
              f"{DOMAINS[dom]}] classes={len(leaf.mixed.classes)} "
              f"spawned={len(leaf.mixed.specs)} "
              f"t={time.time() - t0:.0f}s", flush=True)

    # router: per-domain full-cov Gaussian over the descriptor stream
    archive = ManifoldArchive(leaves[0].sub.arena, full_cov=True)
    for dom in range(n_dom):
        D = torch.cat(router_buf[dom])
        g = torch.Generator().manual_seed(7000 + seed + dom)
        D = D[torch.randperm(len(D), generator=g)[:ROUTER_N]]
        archive.update_class(dom, D)
    archive.finalize_all()
    router = ManifoldRouter(archive, full_cov=True)

    # eval on the union test set
    Xt = torch.cat([v.images for v in eval_views])
    yt = torch.cat([v.labels_global for v in eval_views])
    dom_true = torch.cat([torch.full((len(v.labels_global),), ti // 5)
                          for ti, v in enumerate(eval_views)])

    dom_pred = []
    for i in range(0, len(Xt), 2000):
        dom_pred.append(router.route_class(
            sense(Xt[i:i + 2000]), class_ids=list(range(n_dom)),
            n_classes=n_dom))
    dom_pred = torch.cat(dom_pred)
    route_acc = float((dom_pred == dom_true).float().mean())

    names, evid = [], []
    for dom in range(n_dom):
        _, Ec = base.evidence_both(leaves[dom].mixed, sense, Xt)
        names.append(base.names_of(leaves[dom].mixed))
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
