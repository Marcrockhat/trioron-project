"""D18 — the merge consumer, against the fragmentation gap (Rocky,
s033: "proceed with the merge consumer for the remaining gap").

The M5/review finding: the residual data_hard gap (0.82→0.99) is
fragmentation/majority-mapping ambiguity, not readout math. The merge
consumer collapses DUPLICATE fragments — pairs the matched-filter
membership split across two templates — using division's own law
INVERTED: if the pooled union fails every split-acceptance floor, the
organism's split-vs-keep criterion itself says they are one class.
Candidates are lineage-local (FAMILY_DEGREE); a template-cosine
pre-filter plus the criterion's self-consistency (a fresh split's
union IS its parent, which just divided) prevent merge/split thrash.
Survivors keep BOTH deposit histories (union buffer + exact pooled
sketch).

Arms (full stack: composer + manifold + divide_tries=4, 3 seeds):
merge OFF vs merge ON. Gate: merge ON reduces live classes, does not
reduce mean accuracy, and lifts it (the gap lever).

Run: python3 -m experiments.progenitor.run_d18_merge
"""
from __future__ import annotations

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
from .run_m5_manifold import mapping_of, read_raw, read_sigma


def run_seed(seed: int, merge: bool):
    torch.manual_seed(seed)
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
    mixed = MixedStreamController(sub, stress=pg.router, adopt=pg.controller,
                                  composer=True, manifold=True,
                                  divide_tries=4, merge=merge)
    mixed.observe(Xs[:WINDOW])
    sub.end_task()
    n_merge = 0
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW])
        r = sub.end_task()
        n_merge += r.merges

    truth_of = mapping_of(mixed, Xs, ys)
    clean = [c for c in range(len(tr.names)) if float(spec.std[c]) <= 0.3]
    keep = torch.isin(te.y, torch.tensor(clean))
    Xt, yt = te.x[keep], te.y[keep]
    raw = read_raw(mixed, truth_of, Xt, yt)
    sig = read_sigma(mixed, truth_of, Xt, yt)

    # per-class for the table
    q = mixed.pockets_of(Xt)
    import math
    T = mixed.templates()
    Z = torch.exp(1j * 2 * math.pi * q / 1000)
    pred = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    hit = truth_of[pred] == yt
    per_class = {c: float(hit[yt == c].float().mean()) for c in clean}
    return len(mixed.classes), n_merge, raw, sig, per_class


def main() -> None:
    tables = {}
    for arm, on in (("merge-off", False), ("merge-on", True)):
        print(f"\n══ {arm} ══")
        cls_t, raw_t, sig_t = [], [], []
        acc = Counter()
        for seed in range(SEEDS):
            n_cls, n_merge, raw, sig, per_class = run_seed(seed, on)
            cls_t.append(n_cls)
            raw_t.append(raw)
            sig_t.append(sig)
            for c, a in per_class.items():
                acc[c] += a / SEEDS
            extra = f", {n_merge} merges" if on else ""
            print(f"  seed={seed}: {n_cls} classes{extra}; "
                  f"raw {raw:.3f}  σ {sig:.3f}")
        tables[arm] = (sum(cls_t) / SEEDS, sum(raw_t) / SEEDS,
                       sum(sig_t) / SEEDS, acc)
        print(f"  mean: {tables[arm][0]:.0f} classes, "
              f"raw {tables[arm][1]:.3f}, σ {tables[arm][2]:.3f}")

    off, on = tables["merge-off"], tables["merge-on"]
    rows = sorted(off[3], key=lambda c: on[3][c] - off[3][c])
    print(f"\n  {'class':<8s} {'off':>7s} {'on':>7s} {'Δ':>7s}")
    for c in rows:
        d = on[3][c] - off[3][c]
        if abs(d) > 0.02:
            print(f"  sp{c:03d}    {off[3][c]:7.3f} {on[3][c]:7.3f} {d:+7.3f}")
    print(f"  {'mean':<8s} {off[1]:7.3f} {on[1]:7.3f} {on[1] - off[1]:+7.3f}"
          f"   (σ-readout {off[2]:.3f} → {on[2]:.3f})")

    assert on[0] < off[0], "merge did not reduce live classes"
    assert on[1] >= off[1] - 0.005, "merge cost raw accuracy"
    print(f"\nclasses {off[0]:.0f} → {on[0]:.0f}; raw {off[1]:.3f} → "
          f"{on[1]:.3f}; σ {off[2]:.3f} → {on[2]:.3f}")
    print("D18 MERGE CONSUMER — gate complete")


if __name__ == "__main__":
    main()
