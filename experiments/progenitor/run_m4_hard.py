"""M4 on the capacity-hard taxonomy — per-class accuracy, composer
on/off (Rocky, s033).

data_hard (32 classes × 3 modes, 12 features, ~5 disruptors) through
the M2 mixed-stream protocol with the M4 composer arm. Two things this
run probes that the relational gate cannot:

  1. data_hard is AXIS-SEPARABLE (s031 caveat on record): division +
     readout reach its ceiling without composers. The genome principle
     says the composer should stay (mostly) QUIET here — overproduction
     over a small genome is a regularizer, and a spawn needs a RELATION
     that beats its permutation null.
  2. data_hard has NO sentinel column — per-sample frames MOVE. The
     shared per-sample gain frame couples columns inside a sample, so a
     "relation" can exist in pocket space that is pure encoding
     artifact. The composer's sources are the raw forward-path phases
     (non-canonical; the frame caveat, handoff s033). Whether the
     trial machinery fires on frame-coupling is an open empirical
     question this run answers.

Run: python3 -m experiments.progenitor.run_m4_hard
"""
from __future__ import annotations

from collections import Counter

import torch

from trioron.core import census, construct
from trioron.pcll import (
    MixedStreamController,
    PerceptionGenesis,
    germline_base,
)

from .data_hard import make_spec, sample
from .run_m2_mixed import N_TEST, S_PERIOD, SEEDS, WINDOW


def run_seed(seed: int, composer: bool):
    spec = make_spec()
    tr, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
    te, _ = sample(spec, n_per_class=N_TEST, seed=100 + seed, norm=norm)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tr.y), generator=g)
    Xs, ys = tr.x[order], tr.y[order]

    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW])
    sub.end_task()
    c0 = census(sub.arena)
    mixed = MixedStreamController(sub, stress=pg.router,
                                  adopt=pg.controller, composer=composer)
    mixed.observe(Xs[:WINDOW])
    sub.end_task()

    n_spawn = n_prune = 0
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW])
        r = sub.end_task()
        n_spawn += r.spawns
        n_prune += r.pruned
    c1 = census(sub.arena)
    assert abs(sum(pg.router.germline.votes.values()) - 28.0) < 1e-9

    # frozen-world per-class readout (the M2 protocol)
    import math
    T = mixed.templates()
    Z_all = torch.exp(1j * 2 * math.pi * mixed.pockets_of(Xs) / 1000)
    member = (Z_all.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    mapping = {}
    for k in range(len(T)):
        t = ys[member == k]
        mapping[k] = Counter(t.tolist()).most_common(1)[0][0] if len(t) else -1
    Zt = torch.exp(1j * 2 * math.pi * mixed.pockets_of(te.x) / 1000)
    pred = (Zt.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    truth_of = torch.tensor([mapping[int(k)] for k in range(len(T))])
    hit = truth_of[pred] == te.y

    clean = [c for c in range(len(tr.names)) if float(spec.std[c]) <= 0.3]
    per_class = {c: float(hit[te.y == c].float().mean()) for c in clean}

    kept = [f"{s.gene}({s.i},{s.j})" for s in mixed.specs]
    print(f"  seed={seed}: {len(mixed.classes)} classes; "
          f"spawned {n_spawn}, pruned {n_prune}, kept [{', '.join(kept)}]; "
          f"{c1.delta(c0)}")
    return per_class, clean


def main() -> None:
    spec = make_spec()
    n_class = len(make_spec().std)
    tables = {}
    for arm, on in (("division-only", False), ("composer", True)):
        print(f"\n══ {arm} ══")
        acc = Counter()
        for seed in range(SEEDS):
            per_class, clean = run_seed(seed, on)
            for c, a in per_class.items():
                acc[c] += a / SEEDS
        tables[arm] = acc

    div, com = tables["division-only"], tables["composer"]
    rows = sorted(div, key=lambda c: com[c] - div[c])
    print(f"\n  {'class':<8s} {'div':>7s} {'comp':>7s} {'Δ':>7s}")
    for c in rows:
        print(f"  sp{c:03d}    {div[c]:7.3f} {com[c]:7.3f} "
              f"{com[c] - div[c]:+7.3f}")
    md, mc = (sum(t.values()) / len(t) for t in (div, com))
    print(f"  {'mean':<8s} {md:7.3f} {mc:7.3f} {mc - md:+7.3f}")


if __name__ == "__main__":
    main()
