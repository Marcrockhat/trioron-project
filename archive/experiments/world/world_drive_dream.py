"""Arm 1 (s050): structural dreaming ITERATED on top of the zero-master
drive nest — the lifecycle a pip user would actually get: declare drives
-> organism births its vocabulary by consequence -> self-reflects on its
own deaths -> births new leaves -> re-arbitrates.  No human policy code
anywhere.

Starts from the saved drive-only nest (world_drive_vocab.py: n=3
112.0±23.1 vs master-built 148.5±12.9). Each round: self_reflect on
diagnosis maps (disjoint from eval) -> top cause -> implicated drive ->
new TD leaf on that drive's delta only -> cold-start router over N+1
leaves -> 40-map eval.  ROUNDS rounds; per-round survival is printed so
the gap trajectory vs the master nest is visible.

Run: python3 <abs>/world_drive_dream.py [--seed N] [--rounds 2]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import argparse
import time

import torch

from experiments.world import fire_taming as _ft  # noqa: F401  tamed physics
from experiments.world.fire_taming import evaluate
from experiments.world.drive_common import load_drive_leaves, load_drive_router
from experiments.world.world_drive_vocab import BASELINE, save_sub
from experiments.world.world_dream_newleaf import (DRIVE_OF, Organism,
                                                   self_reflect,
                                                   train_drive_leaf,
                                                   train_router_td_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--td-episodes", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=40)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[3] / "runs" / "drive_dream"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    leaves, names = load_drive_leaves(args.seed)
    router = load_drive_router(args.seed)
    fns = [(lambda p1, l=l: l(p1)) for l in leaves]
    org = Organism(router, fns, names)
    s0, causes0, _ = evaluate(lambda w, p: org.act(w, p)[0],
                              "drive nest (round 0)", seeds=args.eval_seeds)
    hist = [s0]
    for r in range(1, args.rounds + 1):
        print(f"[round {r}] self-reflect  t={time.time() - t0:.0f}s", flush=True)
        causes = self_reflect(org, args.seed + 10 * r)
        top = causes.most_common(1)[0][0] if causes else "cold"
        drive = DRIVE_OF.get(top, "temperature")
        print(f"  diagnosis={dict(causes)} -> '{top}' -> drive '{drive}'",
              flush=True)
        leaf = train_drive_leaf(args.seed + 1000 * r, drive,
                                episodes=args.td_episodes)
        save_sub(leaf, out / f"self_{top}_r{r}_seed{args.seed}.pt")
        leaves.append(leaf); names = names + [f"SELF_{top.upper()}_R{r}"]
        fns = [(lambda p1, l=l: l(p1)) for l in leaves]
        print(f"[round {r}] router TD over {len(fns)} leaves  "
              f"t={time.time() - t0:.0f}s", flush=True)
        router = train_router_td_n(args.seed + 10 * r, fns,
                                   episodes=args.td_episodes)
        save_sub(router, out / f"router_r{r}_seed{args.seed}.pt")
        org = Organism(router, fns, names)
        s, causes_e, _ = evaluate(lambda w, p: org.act(w, p)[0],
                                  f"drive nest (round {r})",
                                  seeds=args.eval_seeds)
        hist.append(s)
        print(f"  round {r}: {hist[-2]:.1f} -> {s:.1f} ({s - hist[-2]:+.1f})  "
              f"'{top}' deaths {causes0.get(top, 0) if r == 1 else '?'}->"
              f"{causes_e.get(top, 0)}  route_hist={dict(zip(names, org.route_hist))}",
              flush=True)
        causes0 = causes_e

    base = BASELINE.get(args.seed)
    print(f"\n[seed {args.seed}] trajectory {' -> '.join(f'{h:.1f}' for h in hist)}"
          + (f"  master-built {base:.1f}  final delta {hist[-1] - base:+.1f}"
             if base is not None else "")
          + f"  elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
