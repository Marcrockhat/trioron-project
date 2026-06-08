"""Native selective-dendrite spawning inside the taxonomy CL loop (s024).

The gap s023 left open (HANDOFF next-up #2): every prior dendrite-CL number used
either a FIXED, always-on uniform quad (cl_dream_dendrite: 0.860, traded dog for
goat) or no growth at all (cl_incremental: bipartite linear). Neither spawns a
dendrite the native way. selective_quad_growth's own verdict: a uniform quad
regresses CL; the substrate must be HETEROGENEOUS — linear everywhere, a quad
dendrite GROWN only where the problem needs it.

This wires that mechanism into the real substrate + the WAKE/DREAM CL loop:

  WAKE  : learn the new species (replay prior from the manifold), all-linear.
  DREAM : co-replay every species seen so far; under sustained frustration that
          linear width can't relieve, divide() an interior cell and grow_branch
          it into a K=2 quad dendrite (spec §3.6). Dream is when overlapping
          species co-exist, so the boundary-carving branch grows exactly there.

Arms:
  linear      seeded interior, no growth (floor).
  uniform     seeded(nonlinear=True): every interior cell is a K=2 quad from the
              start — the cl_dream_dendrite-style always-on quad.
  selective   start all-linear; frustration-gated divide → grow_branch quad child
              during dream. The mechanism under test.

Win condition (joint diagnostic): lift dog toward its 0.682 Bayes recall WITHOUT
crushing goat (0.867) — i.e. carve the disjoint band instead of trading one class
for the other (which the uniform quad did).

Run: python3 -m experiments.growth_exercise.cl_selective_dendrite [--seeds N]
"""
from __future__ import annotations

import argparse
import statistics
from collections import deque

import torch
import torch.nn.functional as F

from trioron.core import Envelope, construct
from trioron.core.epigenome import OUTPUT, DENDRITE, has_gene
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle import divide, GrowthConfig
from trioron.learning import FrustrationDetector, FrustrationConfig
from trioron.learning.manifold import get_interior_ids

from experiments.growth_exercise.taxonomy import make_taxonomy, ALL
from experiments.growth_exercise.cl_dream_dendrite import Manifold

WAKE_STEPS = 150
DREAM_STEPS = 200
REPLAY_PER_CLASS = 64
LR = 0.03
H_INIT = 16
GROW_BUDGET = 12
CAP = 400_000
# Bayes ceilings (taxonomy.py, 4-feat): overall / dog / goat / duck
CEIL = (0.909, 0.682, 0.867, 0.830)


def _output_ids(arena) -> torch.Tensor:
    return torch.tensor(
        [c for c in arena.alive_ids().tolist()
         if has_gene(int(arena.epigenome[c].item()), OUTPUT)],
        dtype=torch.int32,
    )


def _count_quad(arena) -> int:
    return sum(1 for c in arena.alive_ids().tolist()
               if has_gene(int(arena.epigenome[c].item()), DENDRITE)
               and int(arena.n_branches[c].item()) >= 2)


def _make_child_quad(arena, cid):
    """Split the child's inherited fan-in into 2 branches → K=2 quad (spec §3.6)."""
    srcs, _ = arena.inputs_of(cid)
    if srcs.numel() >= 2:
        half = srcs.numel() // 2
        arena.grow_branch(cid, srcs[half:])


def run(train, test, arm, *, mix_k=6, seed=0):
    torch.manual_seed(seed)
    nonlinear = (arm == "uniform")
    sub = construct(base=seeded(7, 10, interior_cells=H_INIT, nonlinear=nonlinear),
                    envelope=Envelope(max_parameter_bytes=CAP),
                    dispatch_table=default_dispatch_table(), capacity=512)
    sub.prepare_training()
    sub.compile()
    a = sub.arena
    man = Manifold(k=mix_k)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)
    frust = FrustrationDetector(FrustrationConfig())
    out_ids = _output_ids(a)
    grown = 0
    frust_steps = 0

    def wire_to_output(cid):
        a.add_edges(torch.full_like(out_ids, cid), out_ids,
                    0.1 * torch.randn(out_ids.numel()))

    def maybe_grow(loss_val):
        nonlocal grown, frust_steps
        if arm != "selective" or grown >= GROW_BUDGET:
            return
        frust.step(loss_val)
        if frust.is_frustrated:
            frust_steps += 1
        if frust.is_frustrated and frust_steps >= 25:
            interior = get_interior_ids(a).long().tolist()
            if interior:
                parent = interior[torch.randint(0, len(interior), (1,)).item()]
                ev = divide(a, parent, GrowthConfig())
                if ev:
                    _make_child_quad(a, ev.child_id)
                    wire_to_output(ev.child_id)
                    grown += 1
                    a.rank_dirty = True
                    sub.compile()
                    frust_steps = 0

    for t in range(len(ALL)):
        Xc = train.X[train.y == t]
        yc = train.y[train.y == t]
        # ── WAKE: learn the new species, replay prior (all-linear) ──
        for _ in range(WAKE_STEPS):
            X, y = Xc, yc
            if man.classes:
                xs = [Xc]; ys = [yc]
                for c in man.classes:
                    xs.append(man.sample(c, REPLAY_PER_CLASS))
                    ys.append(torch.full((REPLAY_PER_CLASS,), c))
                X = torch.cat(xs); y = torch.cat(ys)
            opt.zero_grad()
            F.cross_entropy(sub(X), y).backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            opt.step()
        man.store(t, Xc)
        # ── DREAM: co-replay all seen; spawn the boundary dendrite here ──
        for _ in range(DREAM_STEPS):
            xs, ys = [], []
            for c in man.classes:
                xs.append(man.sample(c, REPLAY_PER_CLASS))
                ys.append(torch.full((REPLAY_PER_CLASS,), c))
            X = torch.cat(xs); y = torch.cat(ys)
            opt.zero_grad()
            loss = F.cross_entropy(sub(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            opt.step()
            maybe_grow(loss.item())
        sub.end_task()

    with torch.no_grad():
        pred = sub(test.X).argmax(1)
    pc = [float((pred[test.y == c] == c).float().mean()) for c in range(len(ALL))]
    return sum(pc) / len(pc), pc, _count_quad(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    train = make_taxonomy(ALL, n_per_class=128, seed=0)
    test = make_taxonomy(ALL, n_per_class=512, seed=1)
    di, gi, du = ALL.index("dog"), ALL.index("goat"), ALL.index("duck")

    print(f"native selective-dendrite spawning in the CL loop  (seeds={args.seeds})\n")
    print(f"{'arm':12} {'overall':>8} {'dog':>6} {'goat':>6} {'duck':>6} {'quad':>5}")
    print(f"{'Bayes':12} {CEIL[0]:>8.3f} {CEIL[1]:>6.3f} {CEIL[2]:>6.3f} {CEIL[3]:>6.3f} {'-':>5}")
    for arm in ("linear", "uniform", "selective"):
        ovs, dogs, goats, ducks, quads = [], [], [], [], []
        for s in range(args.seeds):
            ov, pc, nq = run(train, test, arm, seed=s)
            ovs.append(ov); dogs.append(pc[di]); goats.append(pc[gi])
            ducks.append(pc[du]); quads.append(nq)
        def m(v):
            return statistics.mean(v)
        print(f"{arm:12} {m(ovs):>8.3f} {m(dogs):>6.3f} {m(goats):>6.3f} "
              f"{m(ducks):>6.3f} {m(quads):>5.1f}")


if __name__ == "__main__":
    main()
