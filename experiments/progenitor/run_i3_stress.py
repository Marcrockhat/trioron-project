"""I3 gate — the stress.py scenarios reproduced on the substrate council
(integration design, phase I3).

Same three scenarios as run_stress_growth.py, but everything runs on real
substrate machinery: the germline's conserved vote book (24 composer
seats + 4 progenitor-held perception seats, Σ=28), the StressRouter's
settle/decide at the controller's boundary meeting, sensation growth as a
REAL receptor attach (gene set + compile + controller remap), and
discrimination growth as a REAL read-set recruit.

  A hidden signal — attached receptors see only noise; one unattached
    candidate column carries a coherent signal. EMPTY → sensation → the
    progenitor attaches a candidate. Found ≤ 2 attaches; drive resets.
  B true void — everything is noise. Habituation decays the drive
    1 → 0.5 → 0.25 → 0.125 < DRIVE_FLOOR after exactly 3 fruitless
    growths → ACCEPTED-EMPTY; the perception side has paid exactly its
    3 payable votes (4 seats → floor 1.0).
  C ambiguity — two classes share every read range and differ only on a
    SENSED-but-unread feature. FRUSTRATED → discrimination → recruit it
    → next period RESOLVED with the right winner. (Known-world mode:
    resolver templates given, as in the standalone.)

PASS = every assert across all seeds; vote total conserved at 28
throughout. Run: python3 -m experiments.progenitor.run_i3_stress
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

import torch

from trioron.core import construct
from trioron.pcll import (
    EMPTY,
    FRUSTRATED,
    RESOLVED,
    SENSATION,
    DISCRIMINATION,
    Germline,
    PCLLController,
    StressRouter,
    germline_base,
    template,
)

SEEDS = 3
PERIOD = 1000
MAX_PERIODS = 8
N_QUANTA = 1000
TOTAL_VOTES = 28.0   # 24 composer + 4 perception seats

Spec = Dict[int, Tuple]


def make_organism(n_cols: int, attached: List[int], **ctrl_kw):
    """Germline + (n_cols+1)-wide perception layer (last col = the explicit
    gain-reference sentinel); receptors equipped only on *attached* + the
    sentinel — the rest are perception-only candidates."""
    sub = construct(germline_base, capacity=64)
    germ = Germline(sub)
    ids = germ.spawn_perception(n_cols + 1)
    for c in attached + [n_cols]:
        germ.equip_receptors(ids[c:c + 1])
    sub.compile()
    router = StressRouter(germ)
    ctrl = PCLLController(sub, stress=router, **ctrl_kw)
    return sub, germ, router, ctrl, ids


def period_batch(spec: Spec, rng: random.Random) -> torch.Tensor:
    rows = []
    for _ in range(PERIOD):
        obs = [rng.randint(0, N_QUANTA) if spec[f][0] == "noise"
               else rng.randint(spec[f][1], spec[f][2]) for f in range(len(spec))]
        rows.append(obs + [float(N_QUANTA)])
    return torch.tensor(rows, dtype=torch.float32)


def meeting(sub, ctrl, spec, rng):
    ctrl.observe(period_batch(spec, rng))
    return sub.end_task()


def check(router: StressRouter) -> None:
    assert abs(router.total_votes() - TOTAL_VOTES) < 1e-9, "votes not conserved"


def scenario_a(seed: int) -> None:
    rng = random.Random(seed)
    spec: Spec = {f: ("noise",) for f in range(6)}
    spec[4] = ("range", 380, 420)                  # the hidden coherent signal
    attached, pool = [0, 1, 2, 3], [4, 5]
    rng.shuffle(pool)
    sub, germ, router, ctrl, ids = make_organism(6, attached)
    growths = 0

    r = meeting(sub, ctrl, spec, rng)
    while r.grow is not None:
        assert r.grow == SENSATION and growths < MAX_PERIODS
        germ.attach_receptor(int(ids[pool.pop(0)]))     # progenitor grows sensation
        ctrl.refresh_receptors()
        growths += 1
        r = meeting(sub, ctrl, spec, rng)               # settles the pending growth
        check(router)
    assert r.status == RESOLVED, "hidden signal not found"
    assert growths <= 2 and router.empty_drive == 1.0
    print(f"A seed={seed}: found hidden signal after {growths} receptor "
          f"attach(es); drive reset to 1.0; votes sens="
          f"{router.group_votes(SENSATION):.0f} "
          f"disc={router.group_votes(DISCRIMINATION):.0f}")


def scenario_b(seed: int) -> None:
    rng = random.Random(seed)
    spec: Spec = {f: ("noise",) for f in range(8)}      # true void
    attached, pool = [0, 1, 2, 3], [4, 5, 6, 7]
    sub, germ, router, ctrl, ids = make_organism(8, attached)
    growths = 0

    r = meeting(sub, ctrl, spec, rng)
    while r.grow is not None:
        germ.attach_receptor(int(ids[pool.pop(0)]))
        ctrl.refresh_receptors()
        growths += 1
        r = meeting(sub, ctrl, spec, rng)
        check(router)
    assert r.status == EMPTY and growths == 3, f"{growths} growths"
    assert router.empty_drive < 0.2
    assert abs(router.group_votes(SENSATION) - 1.0) < 1e-9, \
        "perception side should sit at its floor (paid all 3 payable votes)"
    print(f"B seed={seed}: accepted-empty after {growths} fruitless attaches "
          f"(drive {router.empty_drive:.3f}); sensation paid to floor -> "
          f"sens={router.group_votes(SENSATION):.0f} "
          f"disc={router.group_votes(DISCRIMINATION):.0f}")


C_CLASSES = {
    "chicken": {0: [(100, 200)], 1: [(500, 600)], 2: [(800, 900)], 3: [(250, 300)]},
    "dog":     {0: [(100, 200)], 1: [(500, 600)], 2: [(800, 900)], 3: [(650, 700)]},
}


def scenario_c(seed: int, klass: str) -> None:
    rng = random.Random(seed)
    spec: Spec = {f: ("range", lo, hi)
                  for f, [(lo, hi)] in C_CLASSES[klass].items()}
    # all 4 features sensed; templates over the full width (sentinel = 0)
    templates = {name: template(s, 5) for name, s in C_CLASSES.items()}
    sub, germ, router, ctrl, ids = make_organism(
        4, [0, 1, 2, 3], resolver_templates=templates)
    ctrl.read_mask[3] = False                       # f3 sensed but NOT read
    growths = 0

    r = meeting(sub, ctrl, spec, rng)
    assert r.status == FRUSTRATED, "shared ranges should be ambiguous"
    while r.grow is not None:
        assert r.grow == DISCRIMINATION and growths < MAX_PERIODS
        ctrl.recruit(3)                             # composer recruits f3
        growths += 1
        r = meeting(sub, ctrl, spec, rng)
        check(router)
    assert r.status == RESOLVED and r.resolution.winner == klass and growths == 1
    print(f"C seed={seed} {klass}: frustrated -> recruited f3 -> resolved "
          f"({klass}, gap {r.resolution.gap_margin:.1f}); "
          f"disc={router.group_votes(DISCRIMINATION):.0f}")


def main() -> None:
    for seed in range(SEEDS):
        scenario_a(seed)
    for seed in range(SEEDS):
        scenario_b(seed)
    for seed in range(SEEDS):
        for klass in C_CLASSES:
            scenario_c(seed, klass)
    print(f"\nI3 gate PASS — stress drivers + habituation on the substrate "
          f"council (votes conserved at {TOTAL_VOTES:.0f} throughout)")


if __name__ == "__main__":
    main()
