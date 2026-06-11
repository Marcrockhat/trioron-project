"""Item-4 test: the one-cycle resolution trichotomy on the chicken/dog world.

Four stream types, each one period (1000 observations drawn per-observation from
a quanta schedule via feed.draw_sample), judged label-free by the Resolver:

  chicken — pure class stream      → RESOLVED, winner chicken
  dog     — pure class stream      → RESOLVED, winner dog
  mix     — 50/50 per observation  → clears the floor, gap in the noise band
                                     → FRUSTRATED (honest shared evidence, §5)
  noise   — uniform quanta         → EMPTY

PASS = every window of every type lands in its expected cell, all seeds.
Run: python3 run_resolution.py   (~seconds, CPU)
"""
from __future__ import annotations

import math
import random
from collections import Counter

import torch

from feed import CHICKEN, DOG, draw_sample
from lockin import LockIn, evidence_mask
from receptor import N_QUANTA
from resolve import EMPTY, FRUSTRATED, RESOLVED, Resolver

SEEDS = 5
WINDOWS = 10      # per type per seed
PERIOD = 1000
F = 4
CLASSES = {"chicken": CHICKEN, "dog": DOG}

EXPECT = {"chicken": (RESOLVED, "chicken"), "dog": (RESOLVED, "dog"),
          "mix": (FRUSTRATED, None), "noise": (EMPTY, None)}


def observation(kind: str, rng: random.Random) -> list:
    if kind == "chicken":
        return draw_sample(CHICKEN, F, rng)
    if kind == "dog":
        return draw_sample(DOG, F, rng)
    if kind == "mix":
        return draw_sample(CHICKEN if rng.random() < 0.5 else DOG, F, rng)
    return [rng.randint(0, N_QUANTA) for _ in range(F)]


def run_window(kind: str, rng: random.Random, resolver: Resolver):
    li = LockIn(F)
    for _ in range(PERIOD):
        q = torch.tensor(observation(kind, rng), dtype=torch.float32)
        li.step(2 * math.pi * q / N_QUANTA, mask=evidence_mask(q))
    return resolver.resolve(li)


def main() -> None:
    resolver = Resolver.from_schedules(CLASSES, F)
    ok, total = 0, 0
    stats: dict = {k: {"status": Counter(), "gaps": [], "tops": []} for k in EXPECT}

    for seed in range(SEEDS):
        rng = random.Random(seed)
        for _ in range(WINDOWS):
            for kind, (want_status, want_winner) in EXPECT.items():
                r = run_window(kind, rng, resolver)
                stats[kind]["status"][r.status] += 1
                stats[kind]["tops"].append(r.ranking[0][1] if r.ranking else 0.0)
                if r.gap_margin is not None:
                    stats[kind]["gaps"].append(r.gap_margin)
                total += 1
                ok += (r.status == want_status
                       and (want_winner is None or r.winner == want_winner))

    print(f"one-cycle resolution  seeds={SEEDS} windows={WINDOWS}/type/seed "
          f"period={PERIOD}  K_abs=K_gap=3")
    for kind in EXPECT:
        s = stats[kind]
        tops = torch.tensor(s["tops"])
        gaps = torch.tensor(s["gaps"]) if s["gaps"] else torch.tensor([float("nan")])
        print(f"{kind:8s} status={dict(s['status'])}  "
              f"top-margin med {tops.median():7.2f}  gap med {gaps.median():6.2f}")
    print(f"\n{ok}/{total} windows in their expected cell -> "
          f"{'PASS' if ok == total else 'FAIL'}")


if __name__ == "__main__":
    main()
