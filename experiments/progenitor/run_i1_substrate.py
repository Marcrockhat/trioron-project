"""I1 gate — run_schedule_learning.py reproduced THROUGH the Substrate
forward path (integration design, phase I1).

Same scenarios, same seeds, same RNG draw order as the standalone, but
every observation flows substrate-side: scheduler RECEPTOR injection
(spec §10.2) → arena lock-in deposits (§10.3) → end_task() boundary
meeting (§10.4: resolve → signature-learn → adapter → reset).

One translation is needed and is itself the point: the standalone test
feeds quantum values (pockets) directly, but the substrate path includes
the adaptive receptor, which renormalizes every sample by its own peak.
Feeding pocket-valued data through it therefore requires the gain
reference to be EXPLICIT — a sentinel feature pinned at 1000 — which
makes the quantizer the identity on the evidence features. The sentinel
saturates every sample (q = 1000) → masked as reference → deposits
nothing → never active in the learner. That is not a trick; it is the
receptor's design (contrast is encoded relative to the sample peak; the
peak itself is reference, not evidence — spec §10.3).

Gate asserts (PASS required before I2):
  - exact standalone parity: 3 classes born pure among noise, noise never
    births and always reads empty, worst template alignment ≥ 0.999,
    learned-template resolution 6/6 on fresh periods (interleaved);
    births at periods 1/9/17 and a 100%-pure final mixed block —
    structural zero forgetting (sequential);
  - receptor activations ARE the phase (spot-check vs 2π·q/1000);
  - the PCLLResolution adapter tracks the meetings (RESOLVED → not
    frustrated; EMPTY → frustrated at the ceiling).

Run: python3 run_i1_substrate.py   (from experiments/progenitor/)
"""
from __future__ import annotations

import math
import os
import random
import sys
from typing import Dict, List

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from feed import Schedule, draw_sample
from trioron.core import construct
from trioron.core.epigenome import PERCEPTION, RECEPTOR, set_gene
from trioron.pcll import N_QUANTA, PCLLController, RESOLVED, template

SEEDS = 5
PERIOD = 1000
F = 4                 # evidence features (the standalone's F)
SENTINEL = 1000.0     # explicit gain reference (see module doc)

TRUE: Dict[str, Schedule] = {
    "A": {0: [(100, 200)], 1: [(400, 500)], 2: [(700, 800)], 3: [(250, 350)]},
    "B": {0: [(100, 200)], 1: [(600, 700)], 2: [(700, 800)], 3: [(550, 650)]},
    "C": {0: [(300, 400)], 1: [(400, 500)], 2: [(100, 200)], 3: [(850, 950)]},
}


def make_organism():
    """F+1 RECEPTOR|PERCEPTION cells, nothing else — the minimal PCLL organism."""
    def base(sub):
        a = sub.arena
        ids = a.alloc(F + 1)
        for cid in ids.tolist():
            epi = int(a.epigenome[cid].item())
            a.epigenome[cid] = set_gene(set_gene(epi, PERCEPTION), RECEPTOR)

    sub = construct(base, capacity=16)
    return sub, PCLLController(sub)


def period_batch(truth: str, rng: random.Random) -> torch.Tensor:
    """One period of raw observations [PERIOD, F+1] — pocket values plus the
    sentinel, exactly the standalone's draws (same RNG call order)."""
    if truth == "noise":
        qs = [[rng.randint(0, N_QUANTA) for _ in range(F)] for _ in range(PERIOD)]
    else:
        qs = [draw_sample(TRUE[truth], F, rng) for _ in range(PERIOD)]
    x = torch.tensor(qs, dtype=torch.float32)
    return torch.cat([x, torch.full((PERIOD, 1), SENTINEL)], dim=1)


def feed_period(controller, sub, truth: str, rng: random.Random):
    """Stream one period through the substrate, return the MeetingReport."""
    controller.observe(period_batch(truth, rng))
    return sub.end_task()


def purity_map(absorbed: Dict[str, List[str]]) -> Dict[str, str]:
    mapping = {}
    for name, truths in absorbed.items():
        assert len(set(truths)) == 1, f"{name} absorbed mixed truths {set(truths)}"
        mapping[name] = truths[0]
    assert sorted(mapping.values()) == ["A", "B", "C"], f"bad coverage {mapping}"
    return mapping


def check_alignment(learner, mapping: Dict[str, str]) -> float:
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


def check_injection(sub) -> None:
    """Receptor activations ARE the phase: spot-check the last forward."""
    q = sub.scheduler._last_receptor_q
    ids = sub.scheduler._plan.receptor_ids.long()
    got = sub.last_activations[:, ids]
    want = 2 * math.pi * q / N_QUANTA
    assert torch.allclose(got, want), "receptor activation != 2π·q/1000"


def scenario_interleaved(seed: int) -> None:
    rng = random.Random(seed)
    sub, controller = make_organism()
    absorbed: Dict[str, List[str]] = {}
    births = noise_empty = noise_total = 0

    for _ in range(60):
        truth = rng.choice(["A", "B", "C", "noise"])
        report = feed_period(controller, sub, truth, rng)
        if truth == "noise":
            noise_total += 1
            noise_empty += report.event == "empty"
            assert controller.frustration.is_frustrated, \
                "EMPTY meeting did not register as stress"
            continue
        assert report.event != "empty", "coherent class period read empty"
        births += report.event == "birth"
        absorbed.setdefault(report.class_name, []).append(truth)

    check_injection(sub)
    assert births == 3 and len(controller.learner.classes) == 3, f"{births} births"
    assert noise_empty == noise_total, "noise period was not empty"
    mapping = purity_map(absorbed)
    worst = check_alignment(controller.learner, mapping)

    # close the loop: fresh periods resolve via the LEARNED templates at the
    # meeting itself (resolution precedes that period's learning)
    resolved = 0
    for truth in ("A", "B", "C"):
        for _ in range(2):
            report = feed_period(controller, sub, truth, rng)
            r = report.resolution
            assert r is not None and r.status == RESOLVED and \
                mapping[r.winner] == truth, \
                f"meeting resolution failed on {truth}"
            assert not controller.frustration.is_frustrated, \
                "RESOLVED meeting left the adapter frustrated"
            resolved += 1

    print(f"1 seed={seed}: 3 classes born pure of {noise_total} noise + "
          f"{60 - noise_total} class periods; worst alignment {worst:.5f}; "
          f"meeting resolution {resolved}/6; adapter tracks")


def scenario_sequential(seed: int) -> None:
    rng = random.Random(100 + seed)
    sub, controller = make_organism()
    absorbed: Dict[str, List[str]] = {}
    schedule = ["A"] * 8 + ["B"] * 8 + ["C"] * 8 + \
               [rng.choice(["A", "B", "C"]) for _ in range(12)]
    birth_at = []

    for i, truth in enumerate(schedule):
        report = feed_period(controller, sub, truth, rng)
        if report.event == "birth":
            birth_at.append(i + 1)
        absorbed.setdefault(report.class_name, []).append(truth)

    assert birth_at == [1, 9, 17], f"births at {birth_at}"
    mapping = purity_map(absorbed)            # purity ⇒ the final mixed block was
    check_alignment(controller.learner, mapping)  # 100% ⇒ zero forgetting
    print(f"2 seed={seed}: births at periods {birth_at}; "
          f"all {len(schedule)} periods pure incl. final mixed block")


def main() -> None:
    for seed in range(SEEDS):
        scenario_interleaved(seed)
    for seed in range(SEEDS):
        scenario_sequential(seed)
    print("\nI1 gate PASS — schedule learning reproduced through the "
          "Substrate forward path")


if __name__ == "__main__":
    main()
