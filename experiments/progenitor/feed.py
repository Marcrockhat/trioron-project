"""Feed model — class-as-range-schedule + the period sweep (s028 design §3).

A class is NOT a feature vector: it is a per-feature range *schedule* across the
period, in quantum units (receptor pockets, 0..1000). Data is fed as a sweep over
quantum value v = 0 → 1000 (one period). At each v, every feature slices for
whichever class's range contains v; a feature whose value-ranges contain v for no
class votes EMPTY. A class's features fire at *different* v's, so its identity is
a temporal pattern over the period — magnitude has become timing.

A feature may vote multiple classes at one v (shared/non-discriminative range,
e.g. the 0/1000 "ext" feature below), so one quantum can tally more votes than
there are features.
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Dict, Iterator, List, Sequence, Tuple

N_QUANTA = 1000
EMPTY = "empty"

Range = Tuple[int, int]            # inclusive (lo, hi) in quantum units
Schedule = Dict[int, List[Range]]  # feature index -> list of ranges

# Rocky's worked 2-class / 4-feature example (design §3). f1 is the shared
# "ext" feature — fires only at the 0/1000 extremes, votes both classes.
CHICKEN: Schedule = {0: [(1, 5)], 1: [(0, 0), (1000, 1000)], 2: [(80, 100)], 3: [(30, 40)]}
DOG: Schedule = {0: [(6, 10)], 1: [(0, 0), (1000, 1000)], 2: [(30, 50)], 3: [(35, 80)]}


def _contains(ranges: Sequence[Range], v: int) -> bool:
    return any(lo <= v <= hi for lo, hi in ranges)


def votes_at(classes: Dict[str, Schedule], n_features: int, v: int) -> Counter:
    """One quantum of the sweep: each feature votes once per class whose range
    contains v, or EMPTY if no class's range does."""
    tally: Counter = Counter()
    for f in range(n_features):
        hit = False
        for name, sched in classes.items():
            if _contains(sched.get(f, ()), v):
                tally[name] += 1
                hit = True
        if not hit:
            tally[EMPTY] += 1
    return tally


def sweep(classes: Dict[str, Schedule], n_features: int) -> Iterator[Tuple[int, Counter]]:
    """One period: v = 0 → 1000 (q=1000 wraps to 2π — the 'called off' point)."""
    for v in range(N_QUANTA + 1):
        yield v, votes_at(classes, n_features, v)


def accumulate(classes: Dict[str, Schedule], n_features: int) -> Counter:
    """Per-class evidence tallied over one full period."""
    total: Counter = Counter()
    for _, tally in sweep(classes, n_features):
        total.update(tally)
    return total


def draw_sample(schedule: Schedule, n_features: int, rng: random.Random) -> List[int]:
    """One concrete observation of a class: per feature, a quantum drawn uniformly
    from that feature's range schedule (ranges weighted by their size)."""
    q = []
    for f in range(n_features):
        ranges = schedule[f]
        weights = [hi - lo + 1 for lo, hi in ranges]
        lo, hi = rng.choices(ranges, weights=weights)[0]
        q.append(rng.randint(lo, hi))
    return q
