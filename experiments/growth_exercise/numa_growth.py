"""Numa-gated growth — self-calibrating structural plasticity.

The design we converged on (session 022):

  * NUMA = durable HELD-OUT improvement, by ANY means (weight learning,
    reorganization, consolidation).  It measures *learning*, and learning is
    independent of resources -- a fixed-capacity net still accrues Numa by using
    what it has better.  Numa is NOT attributed to a division event (that was the
    formula error): the system can gain Numa even when the envelope is full.
  * GROWTH is a separate lever, not a Numa source.  It is fired as a *probe* only
    when intra-capacity Numa STALLS (held-out loss stops making relative progress)
    while held-out loss is still above the floor we don't yet know.
  * Numa DISPOSES: after a probe division, watch whether held-out Numa resumes
    within an integration window.  Resumes -> capacity was the wall, keep going.
    Stays flat (Mima/zero) -> it was irreducible; after a bounded number of dry
    probes, stop growing.

Why this beats the frustration trigger it replaces (`lifecycle/grow.py`): that
trigger gates on absolute loss-NOISE (`epsilon_improvement=0.00281`,
`frust_steps>=25`), so high LR manufactures spikes -> runaway growth to the cap
for zero gain.  Numa-gating keys on the system's own DURABLE held-out signal,
measured on held-out data -- so the post-divide TRAIN transient (objection 2)
never registers, and the signal never goes dark just because growth is blocked
(objection: "you should gain Numa even when resources are limited").

The scale-invariant core is `rel_eps` (relative held-out improvement, a
dimensionless ratio).  The residual knobs -- `stall_checks`, `integrate_checks`,
`max_dry_probes` -- are CHECKPOINT COUNTS over the durable signal, the honest
"still hand-set" part to self-calibrate next (they are patience/integration/
threshold-width, not absolute loss values).

Run:
  python3 -m experiments.growth_exercise.numa_growth --task rank --seeds 5
  python3 -m experiments.growth_exercise.numa_growth --task chicken-goat \
        --species chicken cat goat cow --log --seeds 5
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trioron.core import Envelope, construct
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.lifecycle import divide, GrowthConfig
from trioron.learning.manifold import get_interior_ids

from experiments.growth_exercise.calibrate_growth import lowrank_blobs
from experiments.growth_exercise.chicken_goat import make_animals, bayes_accuracy

CONTINUE, GROW, STOP = "CONTINUE", "GROW", "STOP"


@dataclass
class NumaGrowthController:
    """Decides CONTINUE / GROW / STOP from held-out loss checkpoints.

    Growth proposes; Numa disposes.  See module docstring.
    """
    rel_eps: float = 0.02          # relative held-out improvement that counts as Numa (dimensionless)
    abs_eps: float = 0.01          # ABSOLUTE significance floor (nats): improvement below this is
                                   # measurement noise, not learning.  The ONE absolute constant -- a
                                   # learning-significance floor, NOT a resource coupling.  It is what
                                   # detects the "solved" boundary (held-out ~ 0), where the relative
                                   # criterion collapses (a fraction of ~0 is ~0).  Self-referential
                                   # upgrade: replace with k * std(recent held-out) -- improvement must
                                   # exceed the held-out loss's own jitter.  TODO(self-calibrate).
    stall_checks: int = 3          # checks of no relative progress => intra-capacity learning exhausted
    integrate_checks: int = 4      # checks to let a probe division earn Numa (the integration window)
    max_dry_probes: int = 2        # consecutive dry probes tolerated before STOP (bounds threshold width)

    def __post_init__(self):
        self.best = float("inf")
        self.stall = 0
        self.mode = "learn"        # "learn" | "probe"
        self.probe_baseline = None
        self.probe_age = 0
        self.dry = 0
        self.peak_numa = 0.0

    def _significant_drop(self, heldout: float, ref: float) -> bool:
        """Did held-out drop below ref by a *meaningful* amount?  Meaningful =
        more than the relative band AND more than the absolute noise floor."""
        return heldout < ref - max(self.rel_eps * abs(ref), self.abs_eps)

    def _is_new_best(self, heldout: float) -> bool:
        ref = self.best if self.best != float("inf") else heldout + 1.0
        return self._significant_drop(heldout, ref)

    def observe(self, heldout: float) -> str:
        if self.mode == "learn":
            if self._is_new_best(heldout):
                self.peak_numa = max(self.peak_numa, self.best - heldout if self.best != float("inf") else 0.0)
                self.best = heldout
                self.stall = 0
                return CONTINUE
            self.stall += 1
            if self.stall >= self.stall_checks:
                # learning from current capacity has stalled, still above floor:
                # PROBE by growing one cell, then watch for durable Numa.
                self.probe_baseline = self.best
                self.probe_age = 0
                self.mode = "probe"
                self.stall = 0
                return GROW
            return CONTINUE

        # mode == "probe": a division just happened; is it earning Numa?
        self.probe_age += 1
        if self._significant_drop(heldout, self.probe_baseline):
            # the probe earned durable held-out improvement -> capacity WAS the wall
            self.best = heldout
            self.dry = 0
            self.mode = "learn"
            self.stall = 0
            return CONTINUE
        if self.probe_age >= self.integrate_checks:
            # probe failed to earn Numa within the integration window -> Mima/zero
            self.dry += 1
            if self.dry >= self.max_dry_probes:
                return STOP
            self.mode = "learn"     # allow another probe later (cross a wider threshold), bounded
            self.stall = 0
            return CONTINUE
        return CONTINUE


def count_cells(sub) -> int:
    return int(sub.arena.alive_ids().numel())


def build_task(task: str, seed: int, species, log: bool):
    if task == "rank":
        # Harder ladder: full rank genuinely needed and the floor CE stays > 0
        # (so the relative Numa criterion lives in its valid regime, not at 0).
        D, C, sep, noise = 8, 8, 3.0, 0.7
        xtr, ytr, _ = lowrank_blobs(D, C, sep, noise, 4096, seed, geom_seed=seed)
        xva, yva, _ = lowrank_blobs(D, C, sep, noise, 4096, seed + 1000, geom_seed=seed)
        # full-rank linear ceiling ~1.0; rank-h bottleneck caps it -> growth helps
        return xtr, ytr, xva, yva, D, C, 1, 1.0, "rank-ladder"
    elif task == "chicken-goat":
        tr = make_animals(species, seed=seed, log=log)
        va = make_animals(species, seed=seed + 1000, log=log)
        ceil = bayes_accuracy(make_animals(species, seed=0, log=log))
        return tr.x, tr.y, va.x, va.y, 1, len(species), 2, ceil, "chicken-goat"
    raise ValueError(task)


def run_seed(seed: int, *, task: str, species, log: bool, lr: float, batch: int,
             eval_every: int, max_steps: int, cap: int, cfg: NumaGrowthController):
    torch.manual_seed(seed)
    xtr, ytr, xva, yva, D, C, h_init, ceiling, name = build_task(task, seed, species, log)

    sub = construct(base=seeded(D, C, interior_cells=h_init, nonlinear=False),
                    envelope=Envelope(max_parameter_bytes=cap),
                    dispatch_table=default_dispatch_table(),
                    capacity=2048, sparsity_k=0)
    sub.compile()
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)

    ctl = NumaGrowthController(rel_eps=cfg.rel_eps, abs_eps=cfg.abs_eps,
                              stall_checks=cfg.stall_checks,
                              integrate_checks=cfg.integrate_checks,
                              max_dry_probes=cfg.max_dry_probes)
    n = len(xtr)
    g = torch.Generator().manual_seed(seed * 7919)
    grown = 0
    decisions = []
    stopped_at = None

    for step in range(max_steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        loss = F.cross_entropy(sub(xtr[idx]), ytr[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
        sub.zero_dormant_grads()
        opt.step()
        opt.zero_grad()

        if (step + 1) % eval_every == 0:
            with torch.no_grad():
                hl = F.cross_entropy(sub(xva), yva).item()
            d = ctl.observe(hl)
            decisions.append((step + 1, hl, d, count_cells(sub)))
            if d == GROW:
                ip = get_interior_ids(sub.arena).long().tolist()
                if ip:
                    parent = ip[torch.randint(0, len(ip), (1,), generator=g).item()]
                    if divide(sub.arena, parent, GrowthConfig()):
                        grown += 1
                        sub.arena.rank_dirty = True
                        sub.compile()   # optimizer identity stable; do NOT rebuild Adam
            elif d == STOP:
                stopped_at = step + 1
                break

    with torch.no_grad():
        tr_acc = (sub(xtr).argmax(-1) == ytr).float().mean().item()
        va_acc = (sub(xva).argmax(-1) == yva).float().mean().item()
    return {
        "seed": seed, "name": name, "ceiling": ceiling,
        "train_acc": tr_acc, "test_acc": va_acc,
        "grown": grown, "cells_end": count_cells(sub),
        "stopped_at": stopped_at, "decisions": decisions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["rank", "chicken-goat"], default="rank")
    ap.add_argument("--species", nargs="+", default=["chicken", "cat", "goat", "cow"])
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--cap", type=int, default=200_000)
    # controller knobs
    ap.add_argument("--rel-eps", type=float, default=0.02)
    ap.add_argument("--abs-eps", type=float, default=0.01)
    ap.add_argument("--stall-checks", type=int, default=3)
    ap.add_argument("--integrate-checks", type=int, default=4)
    ap.add_argument("--max-dry-probes", type=int, default=2)
    ap.add_argument("--trace", action="store_true", help="print per-checkpoint decision log for seed 0")
    args = ap.parse_args()

    cfg = NumaGrowthController(rel_eps=args.rel_eps, abs_eps=args.abs_eps,
                              stall_checks=args.stall_checks,
                              integrate_checks=args.integrate_checks,
                              max_dry_probes=args.max_dry_probes)
    print(f"numa_growth | task={args.task} "
          f"{'species='+str(args.species)+' log='+str(args.log) if args.task=='chicken-goat' else ''} | "
          f"seeds={args.seeds} lr={args.lr} | "
          f"rel_eps={args.rel_eps} stall={args.stall_checks} "
          f"integrate={args.integrate_checks} max_dry={args.max_dry_probes}")

    rows = []
    t0 = time.time()
    for seed in range(args.seeds):
        r = run_seed(seed, task=args.task, species=args.species, log=args.log,
                     lr=args.lr, batch=args.batch, eval_every=args.eval_every,
                     max_steps=args.max_steps, cap=args.cap, cfg=cfg)
        rows.append(r)
        sa = f"@{r['stopped_at']}" if r["stopped_at"] else "no-stop"
        print(f"  seed={seed}  grown={r['grown']:>2d}  cells->{r['cells_end']:>2d}  "
              f"train={r['train_acc']:.3f} test={r['test_acc']:.3f}  "
              f"ceiling={r['ceiling']:.3f}  stopped={sa}")
        if args.trace and seed == 0:
            print("    step   heldout  decision  cells")
            for st, hl, d, nc in r["decisions"]:
                print(f"    {st:>5d}  {hl:7.4f}  {d:<8s}  {nc}")

    elapsed = time.time() - t0
    print(f"\n--- summary (n={len(rows)}) ---  ({elapsed:.1f}s)")
    print(f"  cells grown: {statistics.mean(r['grown'] for r in rows):.1f} "
          f"(min {min(r['grown'] for r in rows)}, max {max(r['grown'] for r in rows)})")
    print(f"  train acc {statistics.mean(r['train_acc'] for r in rows):.3f}   "
          f"test acc {statistics.mean(r['test_acc'] for r in rows):.3f}   "
          f"ceiling {rows[0]['ceiling']:.3f}")
    nstop = sum(1 for r in rows if r["stopped_at"])
    print(f"  self-terminated (STOP): {nstop}/{len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
