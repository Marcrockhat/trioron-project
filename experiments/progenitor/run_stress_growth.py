"""Item-5 test: empty-stress / ambiguity-stress growth + habituation (design §6–8).

Three scenarios, each multi-seed, growth always at the period boundary:

  A hidden signal — attached receptors see only noise; one UNATTACHED candidate
    carries a coherent signal. EMPTY → empty-stress → grow sensation (attach a
    candidate). Growth either finds the signal (stress relieved, drive resets,
    sensation rewarded) or doesn't (habituation decays). The missed-detection
    half of §6: the distinction is made OVER cycles, by the growth.

  B true void — everything is noise. Empty-stress grows fruitlessly; habituation
    decays the drive 1 → 0.5 → 0.25 → 0.125 < DRIVE_FLOOR after exactly 3 growths
    → ACCEPTED-EMPTY: empty becomes the terminal answer, growth stops, the
    sensation council has paid 3 votes for nothing.

  C ambiguity — two classes share every range the composer reads (f0–f2) and
    differ only on f3, which is SENSED but not read. Floor clears, gap = 0 →
    FRUSTRATED → ambiguity-stress → grow discrimination (the composer recruits
    f3 into its templates) → next period RESOLVED with the right winner.
    Sensation grows the receptor pool; discrimination grows the readout of it.

PASS = every assert across all seeds; vote total conserved at 20 throughout.
Run: python3 run_stress_growth.py   (~seconds, CPU)
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import torch

from feed import Schedule, draw_sample
from lockin import LockIn, evidence_mask
from receptor import N_QUANTA
from resolve import EMPTY, FRUSTRATED, RESOLVED, Resolver
from stress import DISCRIMINATION, SENSATION, StressRouter

SEEDS = 3
PERIOD = 1000
K = 3.0
MAX_PERIODS = 8

# feature spec: ("noise",) = uniform quanta; ("range", lo, hi) = coherent
Spec = Dict[int, Tuple]


def observation(spec: Spec, rng: random.Random) -> List[int]:
    out = []
    for f in range(len(spec)):
        s = spec[f]
        out.append(rng.randint(0, N_QUANTA) if s[0] == "noise"
                   else rng.randint(s[1], s[2]))
    return out


def run_period(spec: Spec, attached: List[int], rng: random.Random,
               resolver: Optional[Resolver] = None):
    """One period over the attached receptors. Feature-level (no resolver):
    RESOLVED = some receptor clears the floor, else EMPTY."""
    li = LockIn(len(attached))
    for _ in range(PERIOD):
        obs = observation(spec, rng)
        q = torch.tensor([obs[f] for f in attached], dtype=torch.float32)
        li.step(2 * math.pi * q / N_QUANTA, mask=evidence_mask(q))
    if resolver is not None:
        return resolver.resolve(li)
    status = RESOLVED if (li.margin() > K).any() else EMPTY
    return type("R", (), {"status": status, "winner": None})()


def check(router: StressRouter) -> None:
    assert abs(router.total_votes() - 20.0) < 1e-9, "vote economy not conserved"


def scenario_a(seed: int) -> None:
    rng = random.Random(seed)
    spec: Spec = {f: ("noise",) for f in range(6)}
    spec[4] = ("range", 380, 420)                      # the hidden coherent signal
    attached, pool = [0, 1, 2, 3], [4, 5]
    rng.shuffle(pool)
    router, growths = StressRouter(), 0

    r = run_period(spec, attached, rng)
    while (group := router.decide(r.status)) is not None:
        assert group == SENSATION and growths < MAX_PERIODS
        attached.append(pool.pop(0))                   # grow sensation
        growths += 1
        r = run_period(spec, attached, rng)
        router.report(group, success=r.status != EMPTY)
        check(router)
    assert r.status == RESOLVED, "hidden signal not found"
    assert growths <= 2 and router.empty_drive == 1.0
    print(f"A seed={seed}: found hidden signal after {growths} sensory growth(s); "
          f"drive reset to 1.0; votes sens={router.group_votes(SENSATION):.0f} "
          f"disc={router.group_votes(DISCRIMINATION):.0f}")


def scenario_b(seed: int) -> None:
    rng = random.Random(seed)
    spec: Spec = {f: ("noise",) for f in range(8)}     # true void
    attached, pool = [0, 1, 2, 3], [4, 5, 6, 7]
    router, growths = StressRouter(), 0

    r = run_period(spec, attached, rng)
    while (group := router.decide(r.status)) is not None:
        attached.append(pool.pop(0))
        growths += 1
        r = run_period(spec, attached, rng)
        router.report(group, success=r.status != EMPTY)
        check(router)
    assert r.status == EMPTY and growths == 3, f"expected 3 fruitless growths, got {growths}"
    assert router.empty_drive < 0.2
    assert abs(router.group_votes(SENSATION) - 7.0) < 1e-9
    print(f"B seed={seed}: accepted-empty after {growths} fruitless growths "
          f"(drive {router.empty_drive:.3f}); sensation paid -> "
          f"sens={router.group_votes(SENSATION):.0f} "
          f"disc={router.group_votes(DISCRIMINATION):.0f}")


C_CLASSES: Dict[str, Schedule] = {
    "chicken": {0: [(100, 200)], 1: [(500, 600)], 2: [(800, 900)], 3: [(250, 300)]},
    "dog":     {0: [(100, 200)], 1: [(500, 600)], 2: [(800, 900)], 3: [(650, 700)]},
}


def scenario_c(seed: int, klass: str) -> None:
    rng = random.Random(seed)
    spec: Spec = {f: ("range", lo, hi)
                  for f, [(lo, hi)] in C_CLASSES[klass].items()}
    attached = [0, 1, 2, 3]                            # f3 sensed but not read
    read = {0, 1, 2}
    router, growths = StressRouter(), 0
    resolver = Resolver.from_schedules(C_CLASSES, 4, features=read)

    r = run_period(spec, attached, rng, resolver)
    assert r.status == FRUSTRATED, "shared ranges should be ambiguous"
    while (group := router.decide(r.status)) is not None:
        assert group == DISCRIMINATION and growths < MAX_PERIODS
        read.add(3)                                    # composer recruits f3
        resolver = Resolver.from_schedules(C_CLASSES, 4, features=read)
        growths += 1
        r = run_period(spec, attached, rng, resolver)
        router.report(group, success=r.status == RESOLVED)
        check(router)
    assert r.status == RESOLVED and r.winner == klass and growths == 1
    print(f"C seed={seed} {klass}: frustrated -> recruited f3 -> resolved "
          f"({klass}, gap {r.gap_margin:.1f}); "
          f"disc={router.group_votes(DISCRIMINATION):.0f}")


def main() -> None:
    for seed in range(SEEDS):
        scenario_a(seed)
    for seed in range(SEEDS):
        scenario_b(seed)
    for seed in range(SEEDS):
        for klass in C_CLASSES:
            scenario_c(seed, klass)
    print("\nall scenarios PASS (votes conserved at 20 throughout)")


if __name__ == "__main__":
    main()
