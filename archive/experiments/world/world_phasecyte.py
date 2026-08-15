"""Phasecyte wake / trioron dream — the nested hybrid IN THE WORLD. s049.

The showcase Rocky asked for: the s049 wake/dream loop (validated on
chained-15, student>teacher 9/9) running live in the survival world,
trained EXCLUSIVELY through the released trioron v0.3.0 package API
(`trioron.pcll.PhasecyteNest` / `dream_distill`) — pip-installable code,
no experiment-only machinery in the learner.

DAY 1 (wake): five phasecyte leaves absorb the five skill masters
(WARM/FLEE/HYDRATE/FORAGE/EVADE) from banded (percept, action) pairs in
ONE gradient-free pass each. A full-cov manifold router (fitted from the
same stream) picks the skill per tick. Deployed immediately.

NIGHT (dream): each leaf dream-distills into a quad trioron leaf on
pseudo-pockets from its OWN manifold sketches — no stored data, the
per-skill action set is stationary so no CL machinery is needed.

DAY 2: the dreamed nest is deployed on the same protocol. The artifact
is the side-by-side GIF (wake vs dreamed, same unseen map) with the
routed skill overlaid live — the organism visibly sharper after sleep.

Bars (recorded, NOT rerun — Rocky's ruling): DQN 37.9 (s048 n=3),
solo linear trioron 52.4, Gaussian vocabulary nest 148.1, TD trioron
router 163.2, hand-coded arbiter 167.6.

Known risk, stated up front: single-pass imitation may inherit the s048
imitation choke; per-skill decomposition is what made imitation work
there, and if wake is modest while the dream lift is visible, that IS
the showcase (day/night learning), not a failure of it.

Run (from archive/):  python3 experiments/world/world_phasecyte.py
Env: WP_COLLECT_SEEDS (default 40), WP_EVAL_SEEDS (40), WP_PSEUDO (1500),
     WP_EPOCHS (8), WP_HIDDEN (32), WP_MAP_SEED (987654), WP_SEED (0).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math
import torch

from trioron.core.receptor import N_QUANTA
from trioron.pcll import PhasecyteNest, dream_distill

from experiments.world import fire_taming as _ft          # tamed physics
from experiments.world.fire_taming import evaluate, _near_fire  # noqa
from experiments.world.primitives import PRIMITIVES, collect
from experiments.world.vocabulary import PRIM_ORDER
from experiments.world.tile_world import N_ACTION
from experiments.world.watch_duel import record, render, RUNS

COLLECT_SEEDS = int(os.environ.get("WP_COLLECT_SEEDS", "40"))
EVAL_SEEDS = int(os.environ.get("WP_EVAL_SEEDS", "40"))
PSEUDO = int(os.environ.get("WP_PSEUDO", "1500"))
EPOCHS = int(os.environ.get("WP_EPOCHS", "8"))
HIDDEN = int(os.environ.get("WP_HIDDEN", "32"))
MAP_SEED = int(os.environ.get("WP_MAP_SEED", "987654"))
SEED = int(os.environ.get("WP_SEED", "0"))
# window=50 + action-blocked stream (the chained-15 class-sequential
# analog): probes showed window=500 forms 2 classes named the majority
# action only; 50 forms ~16 action-pure-ish classes, train-acc 0.545
# vs majority 0.423 on WARM.
WINDOW = int(os.environ.get("WP_WINDOW", "50"))
ACTION_LABELS = list(range(N_ACTION))


def sense(x: torch.Tensor) -> torch.Tensor:
    """Percept -> receptor-native: fixed affine [-1,1] -> [0,1]. The
    one-hot block is untouched up to scale; scent/predator directions
    become non-negative magnitudes the quantizer carries exactly."""
    return (x + 1.0) / 2.0


class WakeLeafPolicy:
    """Deployed phasecyte leaf: percept -> centered matched filter ->
    action. Templates precomputed once after wake (readout-only)."""

    def __init__(self, leaf) -> None:
        self.leaf = leaf
        self.names = leaf.names()
        T = leaf.mixed.templates()
        self.mu = T.mean(0, keepdim=True)
        self.Tc = T - self.mu
        self.valid = (self.names >= 0).nonzero().squeeze(1)
        if not len(self.valid):
            print("  [warn] leaf has no named classes — falls back to rest")

    @torch.no_grad()
    def act(self, p: torch.Tensor) -> int:
        if not len(self.valid):
            return 5                       # rest — degenerate fallback
        q = self.leaf.mixed.pockets_of(sense(p.unsqueeze(0)))
        Z = torch.exp(1j * 2 * math.pi * q / N_QUANTA)
        E = self.leaf.mixed._evidence(Z - self.mu, self.Tc)[0]
        c = self.valid[E[self.valid].argmax()]
        return int(self.names[c])


class DreamLeafPolicy:
    """Dreamed quad trioron leaf: percept -> pockets -> logits."""

    def __init__(self, leaf, sub) -> None:
        self.leaf = leaf
        self.sub = sub

    @torch.no_grad()
    def act(self, p: torch.Tensor) -> int:
        q = self.leaf.mixed.pockets_of(sense(p.unsqueeze(0))) / N_QUANTA
        return int(self.sub(q)[0].argmax())


class NestOrganism:
    """Recognition-routed nest over per-skill leaf policies."""

    def __init__(self, nest: PhasecyteNest, policies) -> None:
        self.nest = nest
        self.policies = policies
        self.groups = sorted(nest.leaves)
        self.route_hist = [0] * len(self.groups)

    @torch.no_grad()
    def act(self, w, p) -> tuple[int, str]:
        i = int(self.nest.router.route_class(
            sense(p.unsqueeze(0)), class_ids=self.groups,
            n_classes=len(self.groups))[0])
        self.route_hist[i] += 1
        return self.policies[i].act(p), PRIM_ORDER[i]


def build_wake_nest(seed: int = SEED,
                    collect_seeds: int = COLLECT_SEEDS) -> PhasecyteNest:
    """DAY 1: one gradient-free pass per skill master (action-blocked
    stream), router fitted; ready to deploy."""
    t0 = time.time()
    nest = PhasecyteNest(sense, seed=seed, window=WINDOW,
                         class_cap=43, manifold=True, composer=False)
    for gi, name in enumerate(PRIM_ORDER):
        spec = PRIMITIVES[name]
        P, Y = collect(spec["master"], spec["band"], seeds=collect_seeds,
                       behavior_fn=spec.get("behavior"))
        order = torch.argsort(Y, stable=True)   # action-blocked stream
        P, Y = P[order], Y[order]
        nest.enroll(gi, P)
        labs = [f"g{int(a):02d}" for a in Y]
        for w0 in range(0, len(P), WINDOW):
            nest.observe(gi, P[w0:w0 + WINDOW], labs[w0:w0 + WINDOW])
        leaf = nest.leaves[gi]
        named = int((leaf.names() >= 0).sum())
        print(f"  [wake {name}] pairs={len(P)} classes="
              f"{len(leaf.mixed.classes)} named={named} "
              f"t={time.time() - t0:.0f}s", flush=True)
    nest.fit_router()
    return nest


def main() -> None:
    t0 = time.time()
    print(f"WORLD PHASECYTE (s049 showcase) — wake/dream nested hybrid; "
          f"trained via trioron v0.3.0 package API only. "
          f"collect={COLLECT_SEEDS} eval={EVAL_SEEDS} pseudo={PSEUDO} "
          f"epochs={EPOCHS} hidden={HIDDEN} seed={SEED}")
    nest = build_wake_nest()

    wake = NestOrganism(nest, [WakeLeafPolicy(nest.leaves[g])
                               for g in sorted(nest.leaves)])
    s_wake, _, _ = evaluate(lambda w, p: wake.act(w, p)[0],
                            "phasecyte-wake (day 1)", seeds=EVAL_SEEDS)
    print(f"  wake route_hist={dict(zip(PRIM_ORDER, wake.route_hist))}")

    # ── NIGHT: dream-distill every leaf into a quad trioron leaf ──
    dreamed_policies = []
    for gi in sorted(nest.leaves):
        leaf = nest.leaves[gi]
        sub, _ = dream_distill(leaf, leaf.names(), ACTION_LABELS,
                               pseudo=PSEUDO, epochs=EPOCHS,
                               hidden=HIDDEN, seed=SEED)
        dreamed_policies.append(DreamLeafPolicy(leaf, sub))
    dreamed = NestOrganism(nest, dreamed_policies)
    s_dream, _, _ = evaluate(lambda w, p: dreamed.act(w, p)[0],
                             "trioron-dreamed (day 2)", seeds=EVAL_SEEDS)
    print(f"  dreamed route_hist="
          f"{dict(zip(PRIM_ORDER, dreamed.route_hist))}")

    print(f"\n[seed {SEED}] survival: wake={s_wake:.1f}  "
          f"dreamed={s_dream:.1f}  (recorded bars: DQN 37.9, solo 52.4, "
          f"vocab-nest 148.1, TD-router 163.2, arbiter 167.6)")

    # ── DAY 2 artifact: side-by-side GIF on the unseen map ─────────
    frames_a = record(lambda w, p: wake.act(w, p), MAP_SEED)
    frames_b = record(lambda w, p: dreamed.act(w, p), MAP_SEED)
    out = RUNS / f"phasecyte_dream_map{MAP_SEED}.gif"
    n = render(frames_a, frames_b, out,
               title_a="PHASECYTE WAKE (day 1 — one gradient-free pass)",
               title_b="TRIORON DREAMED (day 2 — distilled overnight)")
    print(f"[gif] {out} ({n} frames)  wake t={frames_a[-1]['t']} "
          f"dreamed t={frames_b[-1]['t']}  elapsed={time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
