"""Selective quad growth under relational frustration (Rocky, 2026-06-01).

The nonlinear-suite verdict: uniform quad regresses continual learning (−11pp),
so the substrate must be HETEROGENEOUS — mostly linear (sensing + CL stability),
quad cells GROWN only where relational computation is needed. This implements
that: the substrate starts all-linear and, under sustained frustration that
linear width can't relieve, grows QUAD cells (divide → child gets the dendrite
phenotype). "Relational frustration" is emergent — quad only grows where linear
stays stuck.

Testbed: SAME-DIFFERENT (single-pass relational). input = [stimulus_a,
stimulus_b]; label = match (same category) vs different. Needs comparison
(products of the two halves) but NO memory — isolates the quad-growth mechanism
from the satellite/memory machinery. Linear can't (the relational wall); quad
can (z² → cross-products → correlation = match).

Arms:
  no-growth      all-linear, no growth (floor).
  linear-growth  frustration → divide a LINEAR cell (more width). Control: width
                 doesn't help a relational task.
  quad-growth    frustration → divide a QUAD cell. The mechanism.

Selectivity check: the same quad-growth substrate on a LINEARLY-SEPARABLE task
should grow FEW quad cells (frustration resolves with linear capacity) — quad is
spent only where it's needed.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trioron.core import Envelope, construct
from trioron.core.epigenome import LINEAR, DENDRITE, PERCEPTION, OUTPUT, has_gene
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle import divide, GrowthConfig
from trioron.learning import FrustrationDetector, FrustrationConfig
from trioron.learning.manifold import get_interior_ids
from experiments.bench_grounded_world import N_SENSE
from experiments.bench_fine_temporal import make_prototypes


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------
def same_different(b, protos, noise, gen):
    """input=[a,b] over 2·N_SENSE; label = match(same category). Relational."""
    k = protos.shape[0]
    a = torch.randint(0, k, (b,), generator=gen)
    match = torch.rand(b, generator=gen) < 0.5
    bcat = a.clone()
    for i in range(b):
        if not match[i]:
            bcat[i] = (a[i] + torch.randint(1, k, (1,), generator=gen)) % k
    xa = protos[a] + noise * torch.randn(b, N_SENSE, generator=gen)
    xb = protos[bcat] + noise * torch.randn(b, N_SENSE, generator=gen)
    return torch.cat([xa, xb], dim=1), match.long()


def category_task(b, protos, noise, gen):
    """Linearly-separable control: classify the single stimulus's category
    (padded to 2·N_SENSE). Frustration should resolve with linear capacity."""
    k = protos.shape[0]
    a = torch.randint(0, k, (b,), generator=gen)
    xa = protos[a] + noise * torch.randn(b, N_SENSE, generator=gen)
    x = torch.cat([xa, torch.zeros(b, N_SENSE)], dim=1)
    return x, a


# ----------------------------------------------------------------------
# Substrate + selective quad growth
# ----------------------------------------------------------------------
def make_child_quad(arena, cid):
    # spec §3.6: the quad engages only with K≥2 branches, so split the child's
    # inherited fan-in into two branches (grow_branch sets the DENDRITE gene and
    # clears LINEAR). A bare gene flip is now linear (K=1) — by design.
    srcs, _ = arena.inputs_of(cid)
    if srcs.numel() >= 2:
        half = srcs.numel() // 2
        arena.grow_branch(cid, srcs[half:])
    else:
        epi = int(arena.epigenome[cid].item())
        arena.epigenome[cid] = (epi & ~(1 << LINEAR)) | (1 << DENDRITE)
        arena.refresh_phenotype(cid)


def count_quad(sub):
    a = sub.arena
    return sum(1 for c in a.alive_ids().tolist()
               if has_gene(int(a.epigenome[c].item()), DENDRITE))


def run_arm(arm, task, n_out, *, k, h_init, cap, seed, epochs, batch, lr,
            grow_budget, noise, linear_first=8):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    protos = make_prototypes(k, g)
    sub = construct(base=seeded(2 * N_SENSE, n_out, interior_cells=h_init),
                    envelope=Envelope(max_parameter_bytes=cap),
                    dispatch_table=default_dispatch_table(), capacity=2048, sparsity_k=0)
    sub.compile(); sub.prepare_training()
    frust = FrustrationDetector(FrustrationConfig())
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    grown = 0; frust_steps = 0; step = 0
    recent = deque(maxlen=20)         # recent batch losses
    prev_growth_loss = None           # recent-mean loss at the previous growth
    escalated = False                 # adaptive: have we switched to quad?
    MIN_LINEAR = 3; PLATEAU_EPS = 0.02; STUCK_LOSS = 0.4
    out_ids = torch.tensor([c for c in sub.arena.alive_ids().tolist()
                            if has_gene(int(sub.arena.epigenome[c].item()), OUTPUT)],
                           dtype=torch.int32)

    def interior_parents():
        a = sub.arena
        return [c for c in get_interior_ids(a).long().tolist()]

    def wire_to_output(cid):
        # divide() only adds INPUT edges; a grown cell must also project to the
        # output head or it's a dead-end (the bug that made growth inert).
        sub.arena.add_edges(torch.full_like(out_ids, cid), out_ids,
                            0.1 * torch.randn(out_ids.numel()))

    for ep in range(epochs):
        for _ in range(40):
            x, y = task(batch, protos, noise, g)
            logits = sub(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            recent.append(loss.item())
            m = frust.step(loss.item())
            if frust.is_frustrated:
                frust_steps += 1
            (loss * m).backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            sub.zero_dormant_grads(); opt.step(); opt.zero_grad()

            # Frustration-gated growth: sustained frustration that linear width
            # can't relieve → grow a cell. quad-growth makes the child quad.
            if (arm != "no-growth" and grown < grow_budget
                    and frust.is_frustrated and frust_steps >= 25):
                ip = interior_parents()
                if ip:
                    parent = ip[torch.randint(0, len(ip), (1,)).item()]
                    # adaptive escalation: grow linear while linear growth keeps
                    # REDUCING loss; once linear growth plateaus (loss not
                    # improving despite added width) → that's relational
                    # frustration linear can't fix → escalate to quad for good.
                    cur = statistics.mean(recent) if recent else loss.item()
                    # Escalate only when loss is STILL HIGH (stuck) AND not
                    # improving — i.e. linear is failing, not "already solved
                    # so no room to improve" (which a low loss would indicate).
                    if (arm == "adaptive" and not escalated and grown >= MIN_LINEAR
                            and prev_growth_loss is not None
                            and cur > STUCK_LOSS
                            and (prev_growth_loss - cur) < PLATEAU_EPS):
                        escalated = True
                    prev_growth_loss = cur
                    ev = divide(sub.arena, parent, GrowthConfig())
                    if ev:
                        make_quad = (arm == "quad-growth") or (
                            arm == "adaptive" and escalated)
                        if make_quad:
                            make_child_quad(sub.arena, ev.child_id)
                        wire_to_output(ev.child_id)
                        grown += 1
                        sub.arena.rank_dirty = True
                        sub.compile()
                        # NB: do NOT rebuild the optimizer — trainable_tensors()
                        # are the fixed-capacity bias/edge_weight tensors (divide
                        # only fills edge slots), so identity is stable. Rebuilding
                        # resets Adam momentum, bumps the loss, and makes growth
                        # self-sustain frustration (defeating selectivity).
                        frust_steps = 0
            step += 1
    with torch.no_grad():
        x, y = task(2000, protos, noise, g)
        acc = (sub(x).argmax(-1) == y).float().mean().item()
    return acc, grown, count_quad(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--h-init", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--grow-budget", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--cap", type=int, default=400_000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs = 1, 20

    print(f"selective_quad_growth: K={args.k} h_init={args.h_init} "
          f"grow_budget={args.grow_budget} seeds={args.seeds} epochs={args.epochs}")
    print(f"\n== RELATIONAL TASK (same-different) — needs quad ==  (chance=0.500)")
    for arm in ("no-growth", "linear-growth", "quad-growth", "adaptive"):
        accs, grs, qs = [], [], []
        for seed in range(args.seeds):
            t0 = time.time()
            acc, grown, nquad = run_arm(arm, same_different, 2, k=args.k,
                                        h_init=args.h_init, cap=args.cap, seed=seed,
                                        epochs=args.epochs, batch=args.batch, lr=args.lr,
                                        grow_budget=args.grow_budget, noise=args.noise)
            accs.append(acc); grs.append(grown); qs.append(nquad)
            print(f"  [{arm:>13s} s{seed}] acc={acc:.3f} grown={grown} quad_cells={nquad} ({time.time()-t0:.0f}s)")
        m = statistics.mean(accs); s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        v = "LEARNS" if m > 0.65 else "fails"
        print(f"  === {arm}: acc={m:.3f}±{s:.3f}  quad_cells≈{statistics.mean(qs):.1f}  [{v}]")

    print(f"\n== SELECTIVITY: ADAPTIVE on a LINEARLY-SEPARABLE task ==  (chance={1/args.k:.3f})")
    accs, qs = [], []
    for seed in range(args.seeds):
        acc, grown, nquad = run_arm("adaptive", category_task, args.k, k=args.k,
                                    h_init=args.h_init, cap=args.cap, seed=seed,
                                    epochs=args.epochs, batch=args.batch, lr=args.lr,
                                    grow_budget=args.grow_budget, noise=args.noise)
        accs.append(acc); qs.append(nquad)
        print(f"  [adaptive/linear-task s{seed}] acc={acc:.3f} quad_cells={nquad}")
    print(f"  === adaptive on linear task: acc={statistics.mean(accs):.3f}  "
          f"quad_cells≈{statistics.mean(qs):.1f}  (≈0 quad ⇒ selective)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
