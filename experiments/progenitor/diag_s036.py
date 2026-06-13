"""s036 pre-run diagnostics (Rocky's call): before spending full-bench
compute, answer two questions with instrumentation.

A. MERGE FLOOR — is the zero-merge on 28x28 MNIST data-truth, or a
   measurement artifact (the q in {0, N} mask killing co-active
   counts)? Dump the redundancy distribution over adjacent surviving
   continuous pairs: how many pairs reach MIN_MEMBERS co-active
   evidence, and where their R_diff sits relative to the derived floor
   REDUNDANT_R = 1 - GAIN_D.

B. TILING — why 84 classes by task 2 and 0 composer spawns? Stream
   tasks 1-2 through the real controller and report, per boundary:
   divisions, class count, and the composer gate state (wired dims,
   buffers under TRIGGER_R, room).

Run: python3 -m experiments.progenitor.diag_s036
"""
from __future__ import annotations

import os

import torch

from trioron.core import construct
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)
from trioron.pcll import (MixedStreamController, PerceptionGenesis,
                          germline_base)
from trioron.pcll import composer as comp
from trioron.pcll.division import MIN_MEMBERS, MIN_CHILD

DATA_ROOT = os.environ.get(
    "AXIS6_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "outputs", "data"))


def _data():
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()[:2]
    return (build_task_views(bundle, specs, split="train"), specs)


def diag_merge_floor(window: torch.Tensor):
    """A: redundancy distribution over adjacent surviving continuous
    columns of the genesis period."""
    from trioron.pcll.retina import (REDUNDANT_R, adjacent_pairs,
                                     pair_redundancy)
    from trioron.pcll.division import GAIN_D

    sub = construct(germline_base, capacity=4096)
    pg = PerceptionGenesis(sub, input_shape=(28, 28))
    pg.feed(window)
    # census verdicts WITHOUT running the merge: replicate the kind test
    eligible = []
    for c in range(784):
        seen = pg._distinct[c]
        if seen is None:                # overflowed -> continuous
            eligible.append(c)
        elif len(seen) <= 1:            # constant -> starve
            continue
        # 2..K discrete handled as discrete; not merge-eligible
    pairs = adjacent_pairs(eligible, (28, 28))
    r_diff, cnt = pair_redundancy([window], eligible, pairs)

    print("\n=== A. MERGE FLOOR ===")
    print(f"surviving continuous cols: {len(eligible)} / 784")
    print(f"adjacent pairs: {len(pairs)}")
    print(f"derived floor REDUNDANT_R = 1 - GAIN_D = {REDUNDANT_R:.3f}")
    enough = cnt >= MIN_MEMBERS
    print(f"pairs with >= MIN_MEMBERS({MIN_MEMBERS}) co-active: "
          f"{int(enough.sum())} / {len(pairs)}")
    if int(enough.sum()):
        re = r_diff[enough]
        for p in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0):
            print(f"  R_diff q{int(p*100):>3d} = "
                  f"{float(re.quantile(p)):.3f}")
        print(f"  pairs clearing floor: "
              f"{int((re >= REDUNDANT_R).sum())} / {len(re)}")
    print(f"co-active count distribution (all pairs): "
          f"min {int(cnt.min())} med {int(cnt.median())} "
          f"max {int(cnt.max())}  (window n={len(window)})")


def diag_tiling(train_views, specs):
    """B: stream tasks 1-2, print per-boundary growth + composer gate."""
    print("\n=== B. TILING / COMPOSER GATE ===")
    sub = construct(germline_base, capacity=4096)
    pg = PerceptionGenesis(sub, input_shape=(28, 28))
    mixed = None
    WINDOW = 1000
    for ti, view in enumerate(train_views):
        g = torch.Generator().manual_seed(ti)
        order = torch.randperm(len(view.labels_global), generator=g)
        X = view.images[order]
        y = view.labels_global[order]
        labs = [f"g{int(v):02d}" for v in y]
        for w0 in range(0, len(X), WINDOW):
            xw = X[w0:w0 + WINDOW]
            lw = labs[w0:w0 + WINDOW]
            if mixed is None:
                pg.feed(xw)
                sub.end_task()
                mixed = MixedStreamController(
                    sub, stress=pg.router, adopt=pg.controller,
                    manifold=True, composer=True, class_cap=128)
            mixed.observe(xw, labels=lw)
            rep = sub.end_task()
            wired = mixed._wired()
            n_under = sum(1 for b in mixed.bufs
                          if len(b) >= 4 * MIN_CHILD
                          and float(b.mean(0).abs().mean()) < comp.TRIGGER_R)
            print(f"  t{ti+1} w{w0//WINDOW:>2d}: "
                  f"classes={len(mixed.classes):>3d} "
                  f"div={getattr(rep, 'divisions', '?')} "
                  f"spawned={len(mixed.specs)} | "
                  f"wired={len(wired)} "
                  f"bufs_under_TRIGGER_R={n_under} "
                  f"room={comp.MAX_SPAWNED - mixed._live_decisions()}",
                  flush=True)
        print(f"  -- task {ti+1} end: {len(mixed.classes)} classes, "
              f"{len(mixed.specs)} spawned --", flush=True)


def main():
    train_views, specs = _data()
    first_window = train_views[0].images[:1000]
    diag_merge_floor(first_window)
    diag_tiling(train_views, specs)


if __name__ == "__main__":
    main()
