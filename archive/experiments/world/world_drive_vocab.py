"""Zero-master vocabulary — can the whole primitive vocabulary be built
from DRIVES alone, with no hand-coded masters?  s050 experiment (a).

Motivation (Rocky, s050): pip users cannot write masters. The masters
turned an unsupervised organism into supervised imitation; the
accessible contract is "declare your drives" and let consequence teach.
s049 structural dreaming showed ONE leaf can be born this way
(TD, reward = implicated drive's delta only, no master: +10.4 n=3).
This script asks whether the WHOLE vocabulary can.

  1. For each drive in DRIVES, a trioron leaf is TD-trained from
     scratch with reward = that drive's delta ONLY (train_drive_leaf,
     unchanged from world_dream_newleaf). No master, no band, no
     demonstration. temperature is a setpoint drive (-2|temp-0.5|), so
     one leaf covers both warm-seeking and cooling.
  2. Each leaf is evaluated SOLO (the analog of the masters' solo rows).
  3. The consequence-taught trioron router (train_router_td_n) is
     trained over the 4 drive leaves; the nest is evaluated on the same
     40-map protocol.

Comparison: master-built 5-leaf TD nest, same seed & protocol
(s049 router_td: seed0 163.2 / seed1 143.2 / seed2 139.2; n=3
148.5±12.9; deterministic — dream_newleaf reproduced 163.2 exactly).
Success = drive-only nest within noise of the master-built nest.
Failure modes stated in advance: sparse drives (integrity) may yield a
useless leaf; a single temperature leaf may re-open the s018 overheat
ceiling that WARM/FLEE splitting closed.

Run (any cwd): python3 <abs>/world_drive_vocab.py [--seed N]
             [--td-episodes 300] [--eval-seeds 40] [--out DIR]
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
from experiments.world.world_dream_newleaf import (Organism, drive_val,  # noqa: F401
                                                   train_drive_leaf,
                                                   train_router_td_n)

DRIVES = ["temperature", "thirst", "energy", "integrity"]
BASELINE = {0: 163.2, 1: 143.2, 2: 139.2}   # s049 master-built 5-leaf nest


def save_sub(sub, path):
    """Rule (s049 ckpt bug): save EVERY tensor in trainable_tensors()."""
    torch.save([t.detach().clone() for t in sub.trainable_tensors()], path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--td-episodes", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--out", default=str(Path(__file__).resolve()
                                         .parents[3] / "runs" / "drive_vocab"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    leaves, names = [], []
    for k, drive in enumerate(DRIVES):
        print(f"[1/3] leaf {k + 1}/{len(DRIVES)}: TD on '{drive}' delta only "
              f"(no master)  t={time.time() - t0:.0f}s", flush=True)
        leaf = train_drive_leaf(args.seed + 100 * k, drive,
                                episodes=args.td_episodes)
        save_sub(leaf, out / f"{drive}_seed{args.seed}.pt")
        leaves.append(leaf); names.append(f"DRIVE_{drive.upper()}")

    print(f"[2/3] solo evals  t={time.time() - t0:.0f}s", flush=True)
    solo = {}
    for leaf, name in zip(leaves, names):
        with torch.no_grad():
            s, _, _ = evaluate(
                lambda w, p, l=leaf: int(l(p.unsqueeze(0))[0].argmax()),
                f"solo {name}", seeds=args.eval_seeds)
        solo[name] = s

    print(f"[3/3] router TD over {len(leaves)} drive leaves  "
          f"t={time.time() - t0:.0f}s", flush=True)
    fns = [(lambda p1, l=leaf: l(p1)) for leaf in leaves]
    router = train_router_td_n(args.seed, fns, episodes=args.td_episodes)
    save_sub(router, out / f"router_seed{args.seed}.pt")
    org = Organism(router, fns, names)
    s_nest, causes, _ = evaluate(
        lambda w, p: org.act(w, p)[0], "drive-only nest", seeds=args.eval_seeds)

    base = BASELINE.get(args.seed)
    print(f"\n[seed {args.seed}] drive-only nest={s_nest:.1f}"
          + (f"  master-built nest={base:.1f}  delta {s_nest - base:+.1f}"
             if base is not None else ""))
    print(f"  solo={ {k: round(v, 1) for k, v in solo.items()} }")
    print(f"  route_hist={dict(zip(names, org.route_hist))}  "
          f"causes={dict(causes)}  elapsed={time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
