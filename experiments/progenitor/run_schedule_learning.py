"""Unsupervised schedule-learning test (design §3) — no labels anywhere.

Three unknown classes (overlapping ranges: A/B share f0+f2, A/C share f1) plus
noise periods. Two scenarios, multi-seed:

  1 interleaved — 60 periods in random class/noise order. Assert: exactly 3
    classes born; every learned class is PURE (absorbs a single ground truth);
    noise never births and always reads empty; learned templates align ≥ 0.999
    with the true-schedule templates; and a Resolver built from the LEARNED
    templates resolves fresh periods to the right class (closes the loop with
    item 4).

  2 sequential — 8×A then 8×B then 8×C then 12 interleaved: births land exactly
    at periods 1, 9, 17; the final interleaved block assigns 100% correctly —
    learning B and C never touched A's accumulator (structural zero forgetting).

PASS = all asserts, all seeds. Run: python3 run_schedule_learning.py
"""
from __future__ import annotations

import math
import random
from typing import Dict, List

import torch

from feed import Schedule, draw_sample
from lockin import LockIn, evidence_mask
from receptor import N_QUANTA
from resolve import RESOLVED, Resolver, template
from schedule_learn import ScheduleLearner

SEEDS = 5
PERIOD = 1000
F = 4

TRUE: Dict[str, Schedule] = {
    "A": {0: [(100, 200)], 1: [(400, 500)], 2: [(700, 800)], 3: [(250, 350)]},
    "B": {0: [(100, 200)], 1: [(600, 700)], 2: [(700, 800)], 3: [(550, 650)]},
    "C": {0: [(300, 400)], 1: [(400, 500)], 2: [(100, 200)], 3: [(850, 950)]},
}


def period_lockin(truth: str, rng: random.Random) -> LockIn:
    if truth == "noise":
        qs = [[rng.randint(0, N_QUANTA) for _ in range(F)] for _ in range(PERIOD)]
    else:
        qs = [draw_sample(TRUE[truth], F, rng) for _ in range(PERIOD)]
    q = torch.tensor(qs, dtype=torch.float32)
    li = LockIn(F)
    li.absorb(2 * math.pi * q / N_QUANTA, mask=evidence_mask(q))
    return li


def purity_map(absorbed: Dict[str, List[str]]) -> Dict[str, str]:
    """learned-name → its single ground truth; asserts purity + bijection."""
    mapping = {}
    for name, truths in absorbed.items():
        assert len(set(truths)) == 1, f"{name} absorbed mixed truths {set(truths)}"
        mapping[name] = truths[0]
    assert sorted(mapping.values()) == ["A", "B", "C"], f"bad coverage {mapping}"
    return mapping


def check_alignment(learner: ScheduleLearner, mapping: Dict[str, str]) -> float:
    worst = 1.0
    for c in learner.classes:
        T_true = template(TRUE[mapping[c.name]], F)
        for f in range(F):
            if c.active[f] and T_true[f].abs() > 0:
                cos = (c.T[f] * T_true[f].conj()).real.item() / (
                    c.T[f].abs().item() * T_true[f].abs().item())
                worst = min(worst, cos)
    assert worst >= 0.999, f"template misaligned (worst cos {worst:.4f})"
    return worst


def scenario_interleaved(seed: int) -> None:
    rng = random.Random(seed)
    learner = ScheduleLearner(F)
    absorbed: Dict[str, List[str]] = {}
    births = noise_empty = noise_total = 0

    for _ in range(60):
        truth = rng.choice(["A", "B", "C", "noise"])
        event, name = learner.observe(period_lockin(truth, rng))
        if truth == "noise":
            noise_total += 1
            noise_empty += event == "empty"
            continue
        assert event != "empty", "coherent class period read empty"
        births += event == "birth"
        absorbed.setdefault(name, []).append(truth)

    assert births == 3 and len(learner.classes) == 3, f"{births} births"
    assert noise_empty == noise_total, "noise period was not empty"
    mapping = purity_map(absorbed)
    worst = check_alignment(learner, mapping)

    # close the loop: the LEARNED world model drives item-4 resolution
    resolver = Resolver(learner.templates())
    for truth in ("A", "B", "C"):
        for _ in range(2):
            r = resolver.resolve(period_lockin(truth, rng))
            assert r.status == RESOLVED and mapping[r.winner] == truth, \
                f"learned resolver failed on {truth}: {r.status}/{r.winner}"

    print(f"1 seed={seed}: 3 classes born pure of {noise_total} noise + "
          f"{60 - noise_total} class periods; worst alignment {worst:.5f}; "
          f"learned resolver 6/6")


def scenario_sequential(seed: int) -> None:
    rng = random.Random(100 + seed)
    learner = ScheduleLearner(F)
    absorbed: Dict[str, List[str]] = {}
    schedule = ["A"] * 8 + ["B"] * 8 + ["C"] * 8 + \
               [rng.choice(["A", "B", "C"]) for _ in range(12)]
    birth_at = []

    for i, truth in enumerate(schedule):
        event, name = learner.observe(period_lockin(truth, rng))
        if event == "birth":
            birth_at.append(i + 1)
        absorbed.setdefault(name, []).append(truth)

    assert birth_at == [1, 9, 17], f"births at {birth_at}"
    mapping = purity_map(absorbed)        # purity ⇒ the final mixed block was
    check_alignment(learner, mapping)     # 100% correct ⇒ zero forgetting
    print(f"2 seed={seed}: births at periods {birth_at}; "
          f"all {len(schedule)} periods pure incl. final mixed block")


def main() -> None:
    for seed in range(SEEDS):
        scenario_interleaved(seed)
    for seed in range(SEEDS):
        scenario_sequential(seed)

    # the §3 narrative, one seed: learned ranges after the runs
    rng = random.Random(0)
    learner = ScheduleLearner(F)
    for _ in range(12):
        learner.observe(period_lockin("A", rng))
    c = learner.classes[0]
    print(f"\nafter {c.m} runs, learned '{c.name}' normalised to "
          f"{c.readable_schedule()}  (truth A: {dict((f, r[0]) for f, r in TRUE['A'].items())})")
    print("\nall scenarios PASS")


if __name__ == "__main__":
    main()
