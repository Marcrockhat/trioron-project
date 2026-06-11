"""M3 — per-class settlement attribution [D13] (design
mixed_stream_growth.md §3/§6).

Two scenarios, both against the s031/s032 indictments:

  1. FALSE CREDIT (the i5_diag scenario): class-sequential data_hard,
     2 epochs. The old global predicate paid DISCRIMINATION whenever the
     world changed (next period = a new class → RESOLVED → success),
     draining sensation to its floor (4→1) and inflating every phenotype
     group to a flat 4.500 — while not one growth executed. Under [D13]
     a pending decision carries the class it answered and settles only
     on that class's own testimony; periods about other classes defer.
     Gate: the 4.500 signature is gone — sensation holds ≥ 3 votes, no
     gene group at 4.5, every settled DISCRIMINATION entry carries a
     subject, Σ=28.

  2. THE DRAIN (the M2 mixed scenario): during exponential discovery
     every window is FRUSTRATED (some OTHER class is divisible), so the
     old global predicate settled every division as FAILURE and bled the
     discrimination side. Under [D13] each division settles against its
     OWN children's split-dim coherence at the next boundary. Gate:
     divisions overwhelmingly settle success (≥ 0.90), the
     discrimination side is not drained (≥ 24 votes), every settlement
     carries a subject, Σ=28.

Run: python3 -m experiments.progenitor.run_m3_settlement
"""
from __future__ import annotations

import torch

from trioron.core import construct
from trioron.pcll import (
    DISCRIMINATION,
    SENSATION,
    MixedStreamController,
    PerceptionGenesis,
    germline_base,
)
from trioron.core.epigenome import GENE_NAMES

from .data_hard import make_spec, sample
from .run_i5_architecture import S_PERIOD, SEEDS
from .run_i5_diag import discover_logged
from .run_m2_mixed import WINDOW


def book(router) -> str:
    g = router.germline
    groups = " ".join(
        f"{GENE_NAMES[p][:4]}={sum(g.votes[c] for c in ids):.3f}"
        for p, ids in g.council_ids.items())
    return (f"sens={router.group_votes(SENSATION):.3f} "
            f"disc={router.group_votes(DISCRIMINATION):.3f}  [{groups}]")


def scenario_false_credit() -> None:
    print("── 1. false credit (i5_diag scenario, class-sequential ×2 epochs) ──")
    for seed in range(SEEDS):
        spec = make_spec()
        tr, _ = sample(spec, n_per_class=S_PERIOD, seed=seed)
        _, ctrl, _, _ = discover_logged(tr.x, tr.y, len(tr.names))
        router = ctrl.stress

        assert abs(router.total_votes() - 28.0) < 1e-9, "Σ≠28"
        # every settled entry carries the class it answered [D13]
        assert all(s is not None for _, s, _ in router.settlements), \
            "a settlement was paid on global status"
        # the false-credit signature is dead: old code ended sens=1.0 and
        # every gene group at exactly 4.500
        sens = router.group_votes(SENSATION)
        g = router.germline
        groups = [sum(g.votes[c] for c in ids)
                  for ids in g.council_ids.values()]
        assert sens >= 3.0 - 1e-9, f"sensation drained to {sens}"
        assert all(v < 4.5 - 1e-9 for v in groups), f"flat-4.5 book: {groups}"

        settled = [(grp, ok) for grp, _, ok in router.settlements]
        n_ok = sum(ok for _, ok in settled)
        print(f"  seed={seed}: settled {len(settled)} "
              f"({n_ok} success / {len(settled) - n_ok} failure), "
              f"{len(router.pending)} deferred (no testimony)")
        print(f"    book: {book(router)}")


def scenario_drain(seed: int) -> float:
    spec = make_spec()
    tr, _ = sample(spec, n_per_class=S_PERIOD, seed=seed)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tr.y), generator=g)
    Xs = tr.x[order]

    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW])
    sub.end_task()
    mixed = MixedStreamController(sub, stress=pg.router, adopt=pg.controller)
    mixed.observe(Xs[:WINDOW])
    sub.end_task()

    total_div = 0
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW])
        total_div += sub.end_task().divisions

    router = pg.router
    assert abs(router.total_votes() - 28.0) < 1e-9, "Σ≠28"
    disc = [(s, ok) for grp, s, ok in router.settlements
            if grp == DISCRIMINATION]
    assert all(s is not None for s, _ in disc), \
        "a division settled on global status"
    # every division is bound and settled (the final boundary's commits
    # have not seen a next boundary yet — they sit pending)
    assert len(disc) + len(router.pending) == total_div, \
        (len(disc), len(router.pending), total_div)
    rate = sum(ok for _, ok in disc) / max(len(disc), 1)
    disc_votes = router.group_votes(DISCRIMINATION)
    print(f"  seed={seed}: {total_div} divisions → {len(disc)} settled "
          f"(success rate {rate:.3f}), {len(router.pending)} pending at end")
    print(f"    book: {book(router)}")
    assert disc_votes >= 24.0 - 1e-9, f"discrimination drained to {disc_votes}"
    return rate


def main() -> None:
    scenario_false_credit()
    print("\n── 2. the drain (M2 mixed scenario, exponential discovery) ──")
    rates = [scenario_drain(s) for s in range(SEEDS)]
    mean = sum(rates) / len(rates)
    print(f"\nmean division settlement success {mean:.3f}  (gate ≥ 0.90)")
    assert mean >= 0.90, f"divisions not holding: {mean:.3f}"
    print("M3 GATE PASS — per-class settlement attribution [D13]")


if __name__ == "__main__":
    main()
