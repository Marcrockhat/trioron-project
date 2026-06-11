"""I5 diagnostics for the Bayes-gap + frustration-trigger questions:

  A. Per-class Bayes ceiling (true generative mixture, raw space) on the
     same held-out population the I5 one-shot eval scores, vs the matched
     filter's per-class recall.
  B. Period-status census during discovery: does FRUSTRATED ever fire on
     data_hard, which classes produce it, and does the (unconsumed)
     council decide() signal line up with the worst one-shot classes?

Run: python3 -m experiments.progenitor.run_i5_diag
"""
from __future__ import annotations

from collections import Counter, defaultdict

import torch

from trioron.core import construct
from trioron.pcll import PerceptionGenesis, germline_base

from .data_hard import bayes_per_class, make_spec, sample
from .run_i5_architecture import (K_DEAD, N_TEST, S_PERIOD, SEEDS,
                                  one_shot_eval)


def discover_logged(X, y, n_class):
    """run_i5_architecture.discover with per-period status logging."""
    sub = construct(germline_base, capacity=192)
    pg = PerceptionGenesis(sub)
    log = []                                  # (epoch, truth, event, status, grow)

    pg.feed(X[y == 0])
    r = sub.end_task()                        # FirstSittingReport: no status
    absorbed = {r.class_name: [0]}
    ctrl = pg.controller

    for c in range(1, n_class):
        ctrl.observe(X[y == c])
        r = sub.end_task()
        absorbed.setdefault(r.class_name, []).append(c)
        log.append((1, c, r.event, r.status, r.grow))

    mapping = {n: t[0] for n, t in absorbed.items()}

    for c in range(n_class):
        ctrl.observe(X[y == c])
        r = sub.end_task()
        log.append((2, c, r.event, r.status, r.grow))

    return sub, ctrl, mapping, log


def main() -> None:
    bayes_acc = defaultdict(list)
    filt_acc = defaultdict(list)
    frustrated = defaultdict(int)
    status_count = Counter()
    grow_count = Counter()

    for seed in range(SEEDS):
        spec = make_spec()
        tr, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
        te, _ = sample(spec, n_per_class=N_TEST, seed=100 + seed, norm=norm)
        n_class = len(tr.names)
        clean = [c for c in range(n_class) if float(spec.std[c]) <= 0.3]

        for c, a in enumerate(bayes_per_class(te)):
            if c in clean:
                bayes_acc[c].append(a)

        sub, ctrl, mapping, log = discover_logged(tr.x, tr.y, n_class)
        for c in clean:
            keep = te.y == c
            filt_acc[c].append(
                one_shot_eval(sub, ctrl, mapping, te.x[keep], te.y[keep]))

        for epoch, c, event, status, grow in log:
            status_count[(epoch, status)] += 1
            if grow:
                grow_count[(epoch, grow)] += 1
            if status == "frustrated":
                frustrated[c] += 1

    print("── B. period-status census (3 seeds × 64 periods) ──")
    for k in sorted(status_count):
        print(f"  epoch {k[0]} {k[1]:<12s} {status_count[k]}")
    print(f"  council decide() growth signals: {dict(grow_count) or 'NONE'}")
    if frustrated:
        print("  FRUSTRATED periods by truth class (count over seeds/epochs):")
        for c in sorted(frustrated):
            print(f"    class {c:2d}  ×{frustrated[c]}")
    else:
        print("  FRUSTRATED never fired.")

    print("\n── A. per-class: matched filter vs Bayes ceiling (3-seed means) ──")
    print(f"  {'class':<10s} {'filter':>7s} {'bayes':>7s} {'gap':>7s}  frus")
    rows = []
    for c in sorted(bayes_acc):
        f = sum(filt_acc[c]) / len(filt_acc[c])
        b = sum(bayes_acc[c]) / len(bayes_acc[c])
        rows.append((c, f, b, b - f))
    for c, f, b, gap in sorted(rows, key=lambda r: -r[3]):
        print(f"  sp{c:03d}      {f:7.3f} {b:7.3f} {gap:7.3f}  "
              f"{'×' + str(frustrated[c]) if c in frustrated else ''}")
    mf = sum(r[1] for r in rows) / len(rows)
    mb = sum(r[2] for r in rows) / len(rows)
    print(f"  {'mean':<10s} {mf:7.3f} {mb:7.3f} {mb - mf:7.3f}")


if __name__ == "__main__":
    main()
