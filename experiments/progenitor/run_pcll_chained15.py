"""PCLL organism on bench-15 (chained-15) — FIRST CONTACT (s034).

No historical record exists: the PCLL trioron has never run on
chained-15 (the "class-sequential filter 0.386" line in run_m2_mixed
is a data_hard contrast number, not a bench-15 record). This harness
is the first.

Protocol differences vs the legacy gradient arms (deliberate — they
ARE the PCLL claims):

  * SINGLE PASS — each task's training set streams through once in
    WINDOW-sample periods (the legacy arms get 8 gradient epochs).
  * ZERO GRADIENTS in the substrate — membership, division, council
    judgment, the membership-quality stack, label supervision.
  * LABELS via the D22 annotation carrier at 100% coverage (the
    legacy arms are fully supervised through CE, so this is parity,
    not an advantage): taps + (class x label) counts + supervised
    routing — all defaults.
  * CLASS-SEQUENTIAL stream — tasks arrive in bench order, classes
    within a task shuffled. The mixed-stream regime's hard case.

Perception (Rocky's correction, s034): the GENESIS PROTOCOL faces the
raw world — 784 pixels straight into PerceptionGenesis (constant
border sensors starve and withdraw, the census types each column, the
read set is GROWN). That is the default arm (PCLL15_SENSE=raw). The
no-spatial-memory findings that argued for a projection were measured
on the GRADIENT substrate, never on PCLL matched filtering — raw-first
is the unmeasured honest experiment. PCLL15_SENSE=lcn keeps the
legacy-parity arm (frozen 16x8 retinotopic Gaussian-mask projection
784 -> 128, LINEAR — the relu's mass-spike at 0 made every dim read
as a divisible mode and division tiled to CLASS_CAP in one task;
measured). NOTE: the lcn arm also tiles to cap by task 2 — the
projection mixes class modes into every dim; recorded, not hidden.

Readout is ORGANISM-INTERNAL end to end: classes are named by their
(class x label) count majority; full = argmax over all templates ->
name; task-aware = argmax restricted to the classes named with that
task's global labels (the bench's task-aware semantics).

Eval on the held-out test split only (never streamed, never labeled).

FIRST-CONTACT VERDICT (s034, seed 0, raw arm, defaults):
task_aware 0.552, full 0.172, 128 classes (vs legacy gradient stack
0.958 / 0.551 with 8 epochs/task). Per-task: MNIST (first block)
eroded to ~0.5; Fashion (middle) held 0.68-0.98; EMNIST letters
(last) 0.50/0.00/0.50/0.00/0.50 — two tasks' labels own NO classes.
Read through the design principle (substrate adapts to input AND
resources): genesis handled raw pixels (the projection arms TILED to
cap inside 1-2 tasks; raw built sane early structure, 4 classes after
task 1) and the organism self-managed at the envelope (husk
retirement freed SOME room for late tasks) — but the allocation
policy under cap pressure starves late arrivals and erodes the
oldest tenants. That allocation law — who yields when the envelope
is full — is the measured open problem; no harness knob fixes it.

Run: python3 -m experiments.progenitor.run_pcll_chained15
Env: PCLL15_SEEDS (default 1), PCLL15_MANIFOLD (default 1),
     PCLL15_WINDOW (default 1000), PCLL15_SENSE (raw|lcn).
"""
from __future__ import annotations

import math
import os
import time
from typing import List, Optional

import torch

from trioron.core import construct
from trioron.core.receptor import N_QUANTA
from trioron.legacy.donorkit.bench_chained_15task import build_lcn_mask
from trioron.legacy.donorkit.datasets import (DatasetBundle,
                                              chained_15_specs,
                                              build_task_views)
from trioron.pcll import (MixedStreamController, PerceptionGenesis,
                          germline_base)

DATA_ROOT = os.environ.get(
    "AXIS6_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "outputs", "data"))
WINDOW = int(os.environ.get("PCLL15_WINDOW", "1000"))
SEEDS = int(os.environ.get("PCLL15_SEEDS", "1"))
MANIFOLD = os.environ.get("PCLL15_MANIFOLD", "1") == "1"
SENSE = os.environ.get("PCLL15_SENSE", "raw")
L0_WIDTH, LCN_SIGMA = 128, 0.10


def make_sense(seed: int):
    """raw: identity — genesis faces the world (default). lcn: the
    legacy-parity frozen retinotopic projection (linear; see module
    docstring for the measured relu/tiling record)."""
    if SENSE == "raw":
        return lambda x: x
    g = torch.Generator().manual_seed(10_000 + seed)
    W = torch.randn(784, L0_WIDTH, generator=g) / math.sqrt(784.0)
    W = W * build_lcn_mask(784, L0_WIDTH, LCN_SIGMA).t()
    return lambda x: x @ W


def names_of(mixed) -> torch.Tensor:
    """Per-class global label from the internal count majority
    (-1 = unnamed). Organism-internal — no eval-side mapping."""
    out = []
    for c in mixed.classes:
        comp = (mixed.label_taps.composition_of(c.name)
                if mixed.label_taps else {})
        out.append(int(max(comp, key=comp.get)[1:]) if comp else -1)
    return torch.tensor(out, dtype=torch.long)


def evidence(mixed, sense, X: torch.Tensor) -> torch.Tensor:
    T = mixed.templates()
    out = []
    for i in range(0, len(X), 2000):
        q = mixed.pockets_of(sense(X[i:i + 2000]))
        Z = torch.exp(1j * 2 * math.pi * q / N_QUANTA)
        out.append(mixed._evidence(Z, T))
    return torch.cat(out)


def run_seed(seed: int):
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"],
                           root=DATA_ROOT, n_holdout_per_dataset=0)
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    sense = make_sense(seed)

    torch.manual_seed(seed)
    sub = construct(germline_base,
                    capacity=1024 if SENSE == "raw" else 512)
    pg = PerceptionGenesis(sub)
    mixed: Optional[MixedStreamController] = None
    t0 = time.time()

    for ti, view in enumerate(train_views):
        g = torch.Generator().manual_seed(100 * seed + ti)
        order = torch.randperm(len(view.labels_global), generator=g)
        X = view.images[order]
        y = view.labels_global[order]
        labs = [f"g{int(v):02d}" for v in y]
        for w0 in range(0, len(X), WINDOW):
            xw = sense(X[w0:w0 + WINDOW])
            lw = labs[w0:w0 + WINDOW]
            if mixed is None:
                pg.feed(xw)
                sub.end_task()
                mixed = MixedStreamController(
                    sub, stress=pg.router, adopt=pg.controller,
                    manifold=MANIFOLD)
            mixed.observe(xw, labels=lw)
            sub.end_task()
        print(f"  [task {ti + 1:>2d}/15 {view.name}] "
              f"classes={len(mixed.classes)} "
              f"t={time.time() - t0:.0f}s", flush=True)

    name = names_of(mixed)
    # full: union of all 15 test sets, argmax over every template
    Xt = torch.cat([v.images for v in eval_views])
    yt = torch.cat([v.labels_global for v in eval_views])
    E = evidence(mixed, sense, Xt)
    full = float((name[E.argmax(1)] == yt).float().mean())

    # task-aware: argmax restricted to classes named with the task's
    # global labels (bench semantics); a task whose labels own no
    # class scores 0 — honest, not skipped
    per_task = []
    off = 0
    for view, spec in zip(eval_views, specs):
        n = len(view.labels_global)
        Et = E[off:off + n]
        off += n
        cand = torch.isin(name, torch.tensor(spec.global_classes)) \
            .nonzero().squeeze(1)
        if not len(cand):
            per_task.append(0.0)
            continue
        pred = name[cand[Et[:, cand].argmax(1)]]
        per_task.append(
            float((pred == view.labels_global).float().mean()))
    ta = sum(per_task) / len(per_task)
    unnamed = int((name < 0).sum())
    print(f"  per-task: " + " ".join(f"{a:.2f}" for a in per_task))
    print(f"[seed {seed}] task_aware={ta:.3f}  full={full:.3f}  "
          f"classes={len(mixed.classes)} ({unnamed} unnamed)  "
          f"elapsed={time.time() - t0:.0f}s", flush=True)
    return ta, full, len(mixed.classes)


def main() -> None:
    print(f"PCLL organism on chained-15 — single pass, no gradients, "
          f"labels at 100% (D22 defaults), manifold={'on' if MANIFOLD else 'off'}, "
          f"window={WINDOW}, seeds={SEEDS}")
    tas, fulls = [], []
    for seed in range(SEEDS):
        ta, full, _ = run_seed(seed)
        tas.append(ta)
        fulls.append(full)
    if SEEDS > 1:
        import statistics
        print(f"\nmean task_aware {statistics.mean(tas):.3f} "
              f"± {statistics.stdev(tas):.3f}  "
              f"full {statistics.mean(fulls):.3f} "
              f"± {statistics.stdev(fulls):.3f}")


if __name__ == "__main__":
    main()
