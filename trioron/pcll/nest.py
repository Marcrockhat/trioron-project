"""Nested phasecyte organisms + wake/dream distillation.  s049 promotion.

Validated in ``experiments/progenitor/{pcll_nested,hybrid_nest}.py``
(commits 77ef15a / 2f04978, logs ``outputs/*_s049_*.log``):

* nest of per-group phasecyte leaves + full-cov manifold recognition
  router — chained-15 paired vs the monolithic organism: full +0.071
  ±0.009, task-aware +0.064±0.028 (n=3), 8× wall-clock; the router is
  load-bearing (flat evidence union collapses to 0.261);
* wake/dream — a gradient trioron leaf trained on pseudo-pockets from
  the phasecyte's OWN manifold sketches beats its teacher in 9/9
  group×seed cells (routed full 0.540±0.037). The sketches cover every
  class of the group jointly, so the gradient leaf trains on a
  STATIONARY set: nesting converts continual learning into per-leaf
  joint learning; the dreamed leaf needs no CL machinery.

The router is recognition-only (`ManifoldRouter` QDA over the sense
descriptor) — no gradients anywhere in the wake path. RNG streams are
parameterized so the promotion drivers reproduce the archived numbers
bit-for-bit.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from trioron.bases.seeded import Seeded
from trioron.core import construct
from trioron.core.receptor import N_QUANTA
from trioron.learning.manifold import ManifoldArchive
from trioron.learning.router import ManifoldRouter

from .mixed import MixedStreamController
from .progenitor import PerceptionGenesis, germline_base


class PhasecyteLeaf:
    """One phasecyte organism serving one group (domain) of the stream."""

    def __init__(self, sense: Callable[[torch.Tensor], torch.Tensor],
                 genesis_pool: torch.Tensor, seed: int, group: int, *,
                 window: int = 1000, class_cap: Optional[int] = 43,
                 manifold: bool = True, composer: bool = True,
                 input_shape=None, capacity: int = 8192,
                 rng_base: int = 1000, verbose: bool = True) -> None:
        self.group = group
        torch.manual_seed(rng_base * seed + group)
        self.sub = construct(germline_base, capacity=capacity)
        pg = PerceptionGenesis(self.sub, input_shape=input_shape)
        g = torch.Generator().manual_seed(rng_base * seed + group)
        gidx = torch.randperm(len(genesis_pool), generator=g)[:window]
        pg.feed(sense(genesis_pool[gidx]))
        rep = self.sub.end_task()
        if verbose:
            n_rec = int(self.sub.scheduler._plan.receptor_ids.numel())
            print(f"  [genesis g{group}] starved={len(rep.starved)} "
                  f"merged={len(rep.merged)}->{len(rep.regions)}  "
                  f"receptors->{n_rec}", flush=True)
        self.mixed = MixedStreamController(
            self.sub, stress=pg.router, adopt=pg.controller,
            manifold=manifold, composer=composer, class_cap=class_cap)

    def observe(self, x_sensed: torch.Tensor, labels: Sequence[str]) -> None:
        self.mixed.observe(x_sensed, labels=list(labels))
        self.sub.end_task()

    def names(self) -> torch.Tensor:
        """Per-class global label from the internal count majority
        (-1 = unnamed). Organism-internal — no eval-side mapping."""
        out = []
        for c in self.mixed.classes:
            comp = (self.mixed.label_taps.composition_of(c.name)
                    if self.mixed.label_taps else {})
            out.append(int(max(comp, key=comp.get)[1:]) if comp else -1)
        return torch.tensor(out, dtype=torch.long)

    def evidence(self, sense, X: torch.Tensor, *, chunk: int = 2000
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """(plain, centered) matched-filter evidence on the SAME
        templates in one substrate pass per chunk (spec §10.4; the
        centered filter subtracts the common-mode phasor — s039)."""
        T = self.mixed.templates()
        mu = T.mean(0, keepdim=True)
        Tc = T - mu
        Ep, Ec = [], []
        for i in range(0, len(X), chunk):
            q = self.mixed.pockets_of(sense(X[i:i + chunk]))
            Z = torch.exp(1j * 2 * math.pi * q / N_QUANTA)
            Ep.append(self.mixed._evidence(Z, T))
            Ec.append(self.mixed._evidence(Z - mu, Tc))
        return torch.cat(Ep), torch.cat(Ec)


class PhasecyteNest:
    """Per-group phasecyte leaves + a manifold recognition router.

    Wake protocol: ``enroll`` a leaf when its group first appears in the
    stream, ``observe`` windows through it (the nest banks descriptor
    windows for the router), then ``fit_router`` once and ``route`` /
    predict. The router is fitted from stream statistics only (per-group
    mu/Sigma, subsampled) — same storage class as manifold replay."""

    def __init__(self, sense: Callable[[torch.Tensor], torch.Tensor], *,
                 seed: int = 0, router_n: int = 4000,
                 router_rng_base: int = 7000, **leaf_kwargs) -> None:
        self.sense = sense
        self.seed = seed
        self.router_n = router_n
        self.router_rng_base = router_rng_base
        self.leaf_kwargs = leaf_kwargs
        self.leaves: Dict[int, PhasecyteLeaf] = {}
        self._router_buf: Dict[int, List[torch.Tensor]] = {}
        self.router: Optional[ManifoldRouter] = None

    def enroll(self, group: int, genesis_pool: torch.Tensor) -> PhasecyteLeaf:
        leaf = PhasecyteLeaf(self.sense, genesis_pool, self.seed, group,
                             **self.leaf_kwargs)
        self.leaves[group] = leaf
        self._router_buf[group] = []
        return leaf

    def observe(self, group: int, X: torch.Tensor,
                labels: Sequence[str]) -> None:
        xw = self.sense(X)
        self.leaves[group].observe(xw, labels)
        self._router_buf[group].append(xw)

    def fit_router(self) -> ManifoldRouter:
        anchor = next(iter(self.leaves.values()))
        archive = ManifoldArchive(anchor.sub.arena, full_cov=True)
        for group, bufs in self._router_buf.items():
            D = torch.cat(bufs)
            g = torch.Generator().manual_seed(
                self.router_rng_base + self.seed + group)
            D = D[torch.randperm(len(D), generator=g)[:self.router_n]]
            archive.update_class(group, D)
        archive.finalize_all()
        self.router = ManifoldRouter(archive, full_cov=True)
        return self.router

    def route(self, X: torch.Tensor, *, chunk: int = 2000) -> torch.Tensor:
        assert self.router is not None, "fit_router() first"
        groups = sorted(self.leaves)
        out = []
        for i in range(0, len(X), chunk):
            out.append(self.router.route_class(
                self.sense(X[i:i + chunk]), class_ids=groups,
                n_classes=max(groups) + 1))
        return torch.cat(out)


def dream_distill(leaf: PhasecyteLeaf, names: torch.Tensor,
                  group_labels: Sequence[int], *, pseudo: int = 1500,
                  epochs: int = 8, hidden: int = 64, seed: int = 0,
                  min_sketch: int = 8, lr: float = 1e-3, batch: int = 256,
                  rng_base: int = 5000, verbose: bool = True):
    """Dream a gradient trioron leaf from a phasecyte leaf's manifold
    sketches — the wake/dream bridge (s049, 9/9 student>teacher).

    Pseudo-pockets are sampled for EVERY named class of the group at
    once, so the returned ``Seeded`` substrate (nonlinear quad, per the
    s048 leaf recipe) trains on a stationary joint set and needs no CL
    machinery. Synthetic deposits never touched the sketches (spec
    §10.9), so the dream cannot self-reinforce."""
    mixed = leaf.mixed
    local = {g: i for i, g in enumerate(group_labels)}
    Xs, ys, skipped = [], [], 0
    for ci, cls in enumerate(mixed.classes):
        g = int(names[ci])
        astro = mixed.manifold.sketches.get(cls.name) \
            if mixed.manifold else None
        if g < 0 or g not in local or astro is None \
                or astro._n < min_sketch:
            skipped += 1
            continue
        q = astro.sample(pseudo).clamp(0, N_QUANTA)
        Xs.append(q / N_QUANTA)
        ys.append(torch.full((len(q),), local[g], dtype=torch.long))
    X = torch.cat(Xs)
    y = torch.cat(ys)
    width = X.shape[1]
    torch.manual_seed(rng_base + seed + leaf.group)
    sub = construct(Seeded(width, len(group_labels),
                           interior_cells=hidden, nonlinear=True),
                    capacity=width + hidden + len(group_labels) + 8)
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    g0 = torch.Generator().manual_seed(rng_base + 1000 + seed + leaf.group)
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g0)
        tot = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = F.cross_entropy(sub(X[idx]), y[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            opt.step()
            tot += float(loss.detach()) * len(idx)
        if verbose and ep in (0, epochs - 1):
            print(f"    [dream g{leaf.group}] ep{ep} "
                  f"loss={tot / len(X):.3f} ({len(X)} pseudo, "
                  f"{skipped} sketches skipped)", flush=True)
    return sub, list(group_labels)


@torch.no_grad()
def dreamed_predict(sub, leaf: PhasecyteLeaf, sense,
                    X: torch.Tensor, group_labels: Sequence[int], *,
                    chunk: int = 2000) -> torch.Tensor:
    """Global-label predictions of a dreamed leaf: the shared receptor
    field is the common currency (x -> pockets/N_QUANTA -> logits)."""
    lab = torch.tensor(list(group_labels))
    out = []
    for i in range(0, len(X), chunk):
        q = leaf.mixed.pockets_of(sense(X[i:i + chunk])) / N_QUANTA
        out.append(lab[sub(q).argmax(1)])
    return torch.cat(out)
