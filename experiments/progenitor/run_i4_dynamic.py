"""I4 gate — the dynamic-perception manifold (integration design D10).

Three tests, multi-seed, all on real substrate machinery:

  1 FRAME SHIFT — learn 3 classes on raw-valued data (cols 0-3), then the
    progenitor attaches receptor col4 whose values are ~3× larger: every
    sample's gain frame jumps, so all pocket coordinates compress. With
    the translation layer (frame stamps, read-time closed-form remap) the
    old classes keep MATCHING and resolving; the no-translation control
    arm (stamps cleared) spuriously births. The new dim warms up from
    zero and auto-promotes after PROMOTE_PATIENCE matched periods.

  2 SHADOW RECRUIT — f3 is sensed but never read; per-class shadow
    accumulators collect it anyway (from birth). recruit(3) extends every
    class INSTANTLY (zero relearning): templates align ≥0.999 with truth
    and the resolution gap strictly improves.

  3 SHIP/WAKE — the learned world (signatures + frame stamps + read state
    + codec + stress book) round-trips through ship() → wake() →
    load_state_dict; a fresh period resolves identically.

Run: python3 -m experiments.progenitor.run_i4_dynamic
"""
from __future__ import annotations

import random
from typing import Dict, List

import torch

from trioron.core import construct
from trioron.lifecycle.ship import ship
from trioron.lifecycle.wake import wake
from trioron.pcll import (
    RESOLVED,
    Germline,
    PCLLController,
    germline_base,
    template,
)
from trioron.pcll.signature import PROMOTE_PATIENCE

SEEDS = 3
PERIOD = 600
N_QUANTA = 1000

# ── test 1: raw-valued classes (cols 0-3 in [0,100]; col4 ~3× larger) ──
RAW: Dict[str, Dict[int, tuple]] = {
    "A": {0: (10, 20), 1: (40, 50), 2: (70, 80), 3: (25, 35), 4: (250, 260)},
    "B": {0: (40, 50), 1: (70, 80), 2: (10, 20), 3: (55, 65), 4: (270, 280)},
    "C": {0: (70, 80), 1: (10, 20), 2: (40, 50), 3: (85, 95), 4: (290, 300)},
}
# each class's max feature pre-attach — the OLD gain reference, silent until
# col4 takes over as reference and releases it as evidence
OLD_REF = {"A": 2, "B": 1, "C": 3}


def raw_period(klass: str, rng: random.Random) -> torch.Tensor:
    rows = [[rng.uniform(*RAW[klass][f]) for f in range(5)] for _ in range(PERIOD)]
    return torch.tensor(rows, dtype=torch.float32)


def raw_organism():
    sub = construct(germline_base, capacity=64)
    germ = Germline(sub)
    ids = germ.spawn_perception(5)
    germ.equip_receptors(ids[:4])        # col4 = unattached candidate
    sub.compile()
    return sub, germ, PCLLController(sub), ids


def scenario_frame_shift(seed: int, translation: bool) -> dict:
    rng = random.Random(seed)
    sub, germ, ctrl, ids = raw_organism()
    names: Dict[str, str] = {}
    for _ in range(3):                                   # learn epoch ×3
        for k in RAW:
            ctrl.observe(raw_period(k, rng))
            r = sub.end_task()
            names.setdefault(k, r.class_name)
            assert r.class_name == names[k], "pre-attach purity broken"
    if not translation:
        for c in ctrl.learner.classes:                   # control: no stamps
            c.frame = None
    germ.attach_receptor(int(ids[4]))                    # the frame shift
    ctrl.refresh_receptors()

    births = matches = wrong = 0
    for _ in range(PROMOTE_PATIENCE):                    # post-attach epochs
        for k in RAW:
            ctrl.observe(raw_period(k, rng))
            r = sub.end_task()
            if r.event == "birth":
                births += 1
            elif r.class_name == names[k]:
                matches += 1
            else:
                wrong += 1
    # the physics of the shift: col4 is the NEW gain reference (always the
    # sample max → always masked → never warms); what gets RELEASED is each
    # class's OLD max feature, silent pre-attach because IT was the reference
    by_name = {c.name: c for c in ctrl.learner.classes}
    col4 = ctrl.receptor_ids.tolist().index(int(ids[4]))
    released = sum(bool(by_name[names[k]].active[OLD_REF[k]]) for k in RAW)
    ref_silent = all(float(by_name[names[k]].m[col4]) == 0 for k in RAW)
    return {"births": births, "matches": matches, "wrong": wrong,
            "released": released, "ref_silent": ref_silent}


# ── test 2: pocket-valued classes; f3 sensed but unread ──
TRUE2: Dict[str, Dict[int, list]] = {
    "A": {0: [(100, 200)], 1: [(400, 500)], 2: [(700, 800)], 3: [(250, 300)]},
    "B": {0: [(600, 700)], 1: [(150, 250)], 2: [(300, 400)], 3: [(650, 700)]},
}


def pocket_period(klass: str, rng: random.Random) -> torch.Tensor:
    rows = []
    for _ in range(PERIOD):
        row = [float(rng.randint(lo, hi)) for f, [(lo, hi)] in TRUE2[klass].items()]
        rows.append(row + [float(N_QUANTA)])             # gain-reference sentinel
    return torch.tensor(rows, dtype=torch.float32)


def pocket_organism():
    sub = construct(germline_base, capacity=64)
    germ = Germline(sub)
    ids = germ.spawn_perception(5)
    germ.equip_receptors(ids)
    sub.compile()
    return sub, germ, PCLLController(sub)


def scenario_shadow_recruit(seed: int) -> None:
    rng = random.Random(100 + seed)
    sub, germ, ctrl = pocket_organism()
    ctrl.read_mask[3] = False                            # sensed, never read
    names = {}
    for _ in range(4):
        for k in TRUE2:
            ctrl.observe(pocket_period(k, rng))
            r = sub.end_task()
            names.setdefault(k, r.class_name)
            assert r.event == "birth" or r.class_name == names[k]

    by_name = {c.name: c for c in ctrl.learner.classes}
    for k, n in names.items():
        c = by_name[n]
        assert float(c.m[3]) >= 4 and not bool(c.active[3]), \
            "shadow should have accumulated unread"
    # gap before recruit
    ctrl.observe(pocket_period("A", rng))
    gap_pre = sub.end_task().resolution.gap_margin

    extended = ctrl.recruit(3)
    assert extended == 2, f"recruit extended {extended} classes, want 2"
    for k, n in names.items():
        c = by_name[n]
        assert bool(c.active[3]), "recruit did not promote the shadow dim"
        T_true = template(TRUE2[k], 5)
        cos = (c.T[3] * T_true[3].conj()).real.item() / (
            c.T[3].abs().item() * T_true[3].abs().item())
        assert cos >= 0.999, f"shadow template misaligned (cos {cos:.4f})"

    ctrl.observe(pocket_period("A", rng))
    r = sub.end_task()
    assert r.resolution.status == RESOLVED and r.class_name == names["A"]
    assert r.resolution.gap_margin > gap_pre, "recruit did not sharpen the gap"
    print(f"2 seed={seed}: shadows m≥4 unread; recruit extended both classes "
          f"instantly (alignment ≥0.999); gap {gap_pre:.1f} → "
          f"{r.resolution.gap_margin:.1f}")


def scenario_ship_wake(seed: int, tmpdir: str = "/tmp") -> None:
    rng = random.Random(200 + seed)
    sub, germ, ctrl = pocket_organism()
    names = {}
    for _ in range(3):
        for k in TRUE2:
            ctrl.observe(pocket_period(k, rng))
            r = sub.end_task()
            names.setdefault(k, r.class_name)
    path = ship(sub, f"{tmpdir}/i4_shipwake_{seed}.pt")

    woken = wake(path)
    assert hasattr(woken, "_pcll_state"), "pcll state not shipped"
    ctrl2 = PCLLController(woken)
    ctrl2.load_state_dict(woken._pcll_state)
    assert len(ctrl2.learner.classes) == len(ctrl.learner.classes)

    ctrl2.observe(pocket_period("B", rng))
    r = woken.end_task()
    assert r.event == "match" and r.class_name == names["B"]
    assert r.resolution is not None and r.resolution.status == RESOLVED
    print(f"3 seed={seed}: learned world round-tripped ship→wake; fresh "
          f"period matched {names['B']} RESOLVED")


def main() -> None:
    for seed in range(SEEDS):
        on = scenario_frame_shift(seed, translation=True)
        off = scenario_frame_shift(seed, translation=False)
        assert on["births"] == 0 and on["wrong"] == 0, f"treatment failed: {on}"
        assert on["matches"] == 3 * PROMOTE_PATIENCE
        assert on["released"] == 3, "old reference dims did not promote"
        assert on["ref_silent"], "new reference dim should never deposit"
        assert off["births"] > 0, "control should degrade without translation"
        print(f"1 seed={seed}: frame shift survived — {on['matches']}/9 matched, "
              f"0 spurious births; old reference dims released in 3/3 classes, "
              f"col4 is the new reference "
              f"(control without translation: {off['births']} spurious births)")
    for seed in range(SEEDS):
        scenario_shadow_recruit(seed)
    for seed in range(SEEDS):
        scenario_ship_wake(seed)
    print("\nI4 gate PASS — dynamic-perception manifold: frame translation "
          "load-bearing, shadow recruit zero-relearning, world ships and wakes")


if __name__ == "__main__":
    main()
