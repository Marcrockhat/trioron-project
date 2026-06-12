"""D20 — membership quality: consolidation, the margin gate, and the
reject readout (Rocky, s033: "build all three, start with (a)").

  (a) CONSOLIDATE — one EM round per quiescent boundary: pool all
      buffered members, reassign by current templates, regroup.
      Streaming membership is one-pass greedy; settling it recovers
      the measured headroom (probe fixed point ~+0.07).
  (b) GATE — per-sample margin floor at membership: a sample whose
      best margin E/σ < gate_k joins NO buffer (it still deposits to
      lock-in). Applied in-stream AND inside consolidation (where old
      pollution is purged).
  (c) REJECT — classify(x, k_reject): the explicit "belongs to nothing
      I know" output. Characterized here as disruptor-rejection vs
      clean-coverage (the safety property: nonsense should be refused,
      not confabulated).

ATTRIBUTION (the s033 factorial, 8 arms × 3 seeds): NO single piece
lifts alone — raw-E arms all 0.81-0.82, margin-membership alone 0.806.
The pieces share one statistic (the per-observation margin) and
amplify each other: margin+gate 0.837, margin+consol 0.828, the FULL
TRIPLET 0.854. D20 ships as a package. (The margin-vs-raw-E membership
default for the M2 path is Rocky's open ruling — flag member_margin.)

Arms (data_hard, full stack, 3 seeds): baseline(raw-E) / margin+consol
/ D20-full(margin+consol+gate3). Gate asserts: the triplet lifts mean
clean accuracy ≥ +0.02; Σ=28; the reject curve separates disruptors
from clean samples.

Run: python3 -m experiments.progenitor.run_d20_membership
"""
from __future__ import annotations

import math
from collections import Counter

import torch

from trioron.core import construct
from trioron.pcll import (
    MixedStreamController,
    PerceptionGenesis,
    germline_base,
)

from .data_hard import make_spec, sample
from .run_m2_mixed import N_TEST, S_PERIOD, SEEDS, WINDOW
from .run_m5_manifold import mapping_of, read_raw

GATE_K = 3.0


def run_seed(seed: int, *, consolidate: bool, gate_k: float,
             member_margin: bool = False):
    torch.manual_seed(seed)
    spec = make_spec()
    tr, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
    te, _ = sample(spec, n_per_class=N_TEST, seed=100 + seed, norm=norm)
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tr.y), generator=g)
    Xs, ys = tr.x[order], tr.y[order]

    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW]); sub.end_task()
    mixed = MixedStreamController(sub, stress=pg.router, adopt=pg.controller,
                                  composer=True, manifold=True,
                                  divide_tries=4,
                                  consolidate=consolidate, gate_k=gate_k,
                                  member_margin=member_margin)
    mixed.observe(Xs[:WINDOW]); sub.end_task()
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW]); sub.end_task()
    assert abs(sum(pg.router.germline.votes.values()) - 28.0) < 1e-9

    truth_of = mapping_of(mixed, Xs, ys)
    clean = [c for c in range(len(tr.names)) if float(spec.std[c]) <= 0.3]
    keep = torch.isin(te.y, torch.tensor(clean))
    acc = read_raw(mixed, truth_of, te.x[keep], te.y[keep])

    # buffer purity (vs truth — eval-side only)
    q_all = mixed.pockets_of(Xs)
    Z = torch.exp(1j * 2 * math.pi * q_all / 1000)
    T = mixed.templates()
    member = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    purs = []
    for k in range(len(T)):
        t = ys[member == k]
        if len(t):
            purs.append(Counter(t.tolist()).most_common(1)[0][1] / len(t))
    return mixed, truth_of, acc, sum(purs) / len(purs), te, clean


def main() -> None:
    # the factorial (s033) showed NO single piece lifts alone — the
    # triplet (margin membership + consolidation + gate) is the
    # mechanism: 0.818 / 0.806-0.837 partials / 0.854 full
    arms = (("baseline", False, False, 0.0),
            ("margin+consol", True, True, 0.0),
            ("D20-full", True, True, GATE_K))
    results = {}
    keep_last = None
    for name, mm, cons, gk in arms:
        accs, purities, classes = [], [], []
        for seed in range(SEEDS):
            mixed, truth_of, acc, pur, te, clean = run_seed(
                seed, consolidate=cons, gate_k=gk, member_margin=mm)
            accs.append(acc)
            purities.append(pur)
            classes.append(len(mixed.classes))
            if name == "D20-full" and seed == 0:
                keep_last = (mixed, truth_of, te, clean)
        results[name] = (sum(accs) / SEEDS, sum(purities) / SEEDS,
                         sum(classes) / SEEDS)
        print(f"  {name:>13s}: acc {results[name][0]:.3f}  "
              f"purity {results[name][1]:.3f}  "
              f"classes {results[name][2]:.0f}  "
              f"(per seed: {' '.join(f'{a:.3f}' for a in accs)})")

    base, full = results["baseline"][0], results["D20-full"][0]
    assert full > base + 0.02, (
        f"D20 triplet lift {full - base:+.3f} below floor")

    # ── (c) the reject readout: nonsense vs clean ─────────────────
    mixed, truth_of, te, clean = keep_last
    keep = torch.isin(te.y, torch.tensor(clean))
    X_clean = te.x[keep]
    X_noise = te.x[~keep]                  # disruptor draws ≈ spray
    print("\n  reject curve (clean coverage vs disruptor rejection):")
    print(f"  {'k_reject':>8s} {'clean kept':>10s} {'noise rejected':>14s}")
    sep = None
    for kr in (3.0, 4.0, 5.0, 6.0, 8.0):
        pc, _ = mixed.classify(X_clean, k_reject=kr)
        pn, _ = mixed.classify(X_noise, k_reject=kr)
        cov = float((pc >= 0).float().mean())
        rej = float((pn < 0).float().mean())
        print(f"  {kr:>8.1f} {cov:>10.1%} {rej:>14.1%}")
        if sep is None or rej - (1 - cov) > sep[0]:
            sep = (rej - (1 - cov), kr, cov, rej)
    assert sep[0] > 0.10, "reject does not separate nonsense from signal"
    print(f"  best separation at k={sep[1]:.0f}: clean {sep[2]:.1%} kept, "
          f"noise {sep[3]:.1%} rejected")
    print(f"\nbaseline {base:.3f} → D20 full triplet {full:.3f} "
          f"(+{full - base:.3f})")
    print("D20 GATE PASS — membership quality")


if __name__ == "__main__":
    main()
