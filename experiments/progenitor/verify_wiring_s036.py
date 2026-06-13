"""s036 end-to-end WIRING verification (Rocky: the capped bench was
bit-identical to s034 — is the new machinery actually connected, or
silently bypassed?).

The bench is bit-identical BY CONFIGURATION: composer off + retina
merged 0 columns + cap 128 = every active knob identical to s034, so
identical output is the dormant-path sanity check passing, not a test
of the new code. This script forces the ACTIVE paths and asserts they
change behaviour:

  TEST 1 (retina path live): a synthetic stream on a small grid with
    adjacent columns made GENUINELY redundant (a region of pixels that
    co-vary, mid-tone so they clear the mask). Genesis MUST merge them
    into a pooled region sensor; assert (a) regions form, (b) the
    pooled sensor is in the controller's receptor set, (c) it carries
    lock-in evidence, (d) the organism discovers the classes and
    classifies the held-out set ABOVE chance — i.e. the pooled sensor
    is in the learning + readout path, not just injected.

  TEST 2 (composer path live): a synthetic 2-class stream whose two
    receptor dims are INDIVIDUALLY uninformative but jointly carry the
    class by a relation (XOR-like), so each per-dim buffer is frustrated
    (under TRIGGER_R) and the composer must spawn a real relational
    cell. Assert a cell spawns, enters the forward path, and its
    composed dim carries evidence.

Run: python3 -m experiments.progenitor.verify_wiring_s036
"""
from __future__ import annotations

import math

import torch

from trioron.core import construct
from trioron.core.epigenome import has_gene, CONV, DENDRITE, TANH
from trioron.pcll import (MixedStreamController, PerceptionGenesis,
                          germline_base)
from trioron.pcll.lockin import LockInView


def _grid_stream(n_per_class: int = 600, seed: int = 0):
    """4x4 grid, 2 classes. A 2x2 redundant region (cols 5,6,9,10) whose
    four pixels co-vary (same latent, mid-tone) -> genesis must merge
    them. The class signal lives in cols 0 and 15 (corners). Border
    cols constant -> starve. Mid-tone keeps everything off the mask."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    labels = []
    for cls in (0, 1):
        for _ in range(n_per_class):
            x = torch.full((16,), 0.30) + 0.02 * torch.randn(16, generator=g)
            latent = 0.5 + 0.3 * torch.rand(1, generator=g)
            for c in (5, 6, 9, 10):                 # the redundant region
                x[c] = latent + 0.005 * torch.randn(1, generator=g)
            # class signal in two corners (mid-tone, separable)
            x[0] = 0.7 if cls == 0 else 0.4
            x[15] = 0.4 if cls == 0 else 0.7
            x[3] = 0.0                                # constant -> starve
            x[12] = 0.0
            rows.append(x)
            labels.append(cls)
    X = torch.stack(rows)
    y = torch.tensor(labels)
    perm = torch.randperm(len(X), generator=g)
    return X[perm], y[perm]


def test_retina_path():
    print("=== TEST 1: retina path (merge -> learn -> readout) ===")
    X, y = _grid_stream()
    n = len(X)
    Xtr, ytr = X[:n * 8 // 10], y[:n * 8 // 10]
    Xte, yte = X[n * 8 // 10:], y[n * 8 // 10:]

    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub, input_shape=(4, 4))
    W = 300
    mixed = None
    for w0 in range(0, len(Xtr), W):
        xw = Xtr[w0:w0 + W]
        lw = [f"g{int(v):02d}" for v in ytr[w0:w0 + W]]
        if mixed is None:
            pg.feed(xw)
            rep = sub.end_task()
            print(f"  genesis: starved={len(rep.starved)} "
                  f"regions={len(rep.regions)} "
                  f"merged_cols={len(rep.merged)} "
                  f"receptors={int(sub.scheduler._plan.receptor_ids.numel())} "
                  f"pooled={int(sub.scheduler._plan.pooled_ids.numel())}")
            assert len(rep.regions) >= 1, "WIRING FAIL: no region merged"
            mixed = MixedStreamController(sub, stress=pg.router,
                                         adopt=pg.controller, manifold=True)
        mixed.observe(xw, labels=lw)
        sub.end_task()

    plan = sub.scheduler._plan
    pooled_in_set = bool(torch.isin(plan.pooled_ids, mixed.receptor_ids).all())
    margins = LockInView(sub.arena, mixed.receptor_ids).margin()
    pooled_cols = [i for i, c in enumerate(mixed.receptor_ids.tolist())
                   if c in set(plan.pooled_ids.tolist())]
    pooled_margin = float(margins[pooled_cols].mean()) if pooled_cols else 0.0
    print(f"  pooled sensor in receptor set : {pooled_in_set}")
    print(f"  pooled sensor lock-in margin  : {pooled_margin:.2f} "
          f"(>3 = coherent evidence)")
    assert pooled_in_set, "WIRING FAIL: pooled sensor absent from readout set"

    # readout via the organism's own templates
    name = []
    for c in mixed.classes:
        comp = (mixed.label_taps.composition_of(c.name)
                if mixed.label_taps else {})
        name.append(int(max(comp, key=comp.get)[1:]) if comp else -1)
    name = torch.tensor(name)
    T = mixed.templates()
    q = mixed.pockets_of(Xte)
    Z = torch.exp(1j * 2 * math.pi * q / 1000)
    E = mixed._evidence(Z, T)
    pred = name[E.argmax(1)]
    acc = float((pred == yte).float().mean())
    print(f"  classes discovered            : {len(mixed.classes)}")
    print(f"  held-out full accuracy        : {acc:.3f} (chance 0.5)")
    ok = acc > 0.7 and pooled_margin > 3.0
    print(f"  -> retina path {'LIVE' if ok else 'SUSPECT'}\n")
    return ok


def _relation_stream(n_per_class: int = 800, seed: int = 1):
    """2 classes, 2 informative dims. Each dim alone is ~uninformative
    (both classes span it); the class is the SIGN of their product
    (XOR-like) -> per-dim buffers frustrated -> composer must compose."""
    g = torch.Generator().manual_seed(seed)
    rows, labels = [], []
    for _ in range(2 * n_per_class):
        a = torch.rand(1, generator=g).item()
        b = torch.rand(1, generator=g).item()
        cls = int((a > 0.5) ^ (b > 0.5))
        x = torch.tensor([0.2 + 0.6 * a, 0.2 + 0.6 * b,
                          0.3 + 0.02 * torch.randn(1, generator=g).item()])
        rows.append(x)
        labels.append(cls)
    X = torch.stack(rows)
    y = torch.tensor(labels)
    perm = torch.randperm(len(X), generator=g)
    return X[perm], y[perm]


def test_composer_path():
    print("=== TEST 2: composer path (frustration -> spawn -> forward) ===")
    X, y = _relation_stream()
    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub)            # no geometry; relation, not image
    W = 400
    mixed = None
    for w0 in range(0, len(X), W):
        xw = X[w0:w0 + W]
        lw = [f"g{int(v):02d}" for v in y[w0:w0 + W]]
        if mixed is None:
            pg.feed(xw)
            sub.end_task()
            mixed = MixedStreamController(sub, stress=pg.router,
                                         adopt=pg.controller,
                                         manifold=True, composer=True)
        mixed.observe(xw, labels=lw)
        sub.end_task()

    n_spawn = len(mixed.specs)
    genes = [s.gene for s in mixed.specs]
    print(f"  composer cells spawned        : {n_spawn} {genes}")
    live = False
    if n_spawn:
        # the spawned cell is in the forward path and deposits evidence
        cell = mixed.composer_cells[0]
        in_fwd = bool(sub.arena.forward_inclusion[cell])
        sub.forward(X[:64])
        act = sub.scheduler._last_activations[:, cell]
        live = in_fwd and bool(act.abs().sum() > 0)
        print(f"  spawned cell forward-included : {in_fwd}")
        print(f"  spawned cell active in forward: "
              f"{bool(act.abs().sum() > 0)}")
    print(f"  -> composer path {'LIVE' if live else 'SUSPECT (no spawn)'}\n")
    return live


def main():
    r = test_retina_path()
    c = test_composer_path()
    print(f"WIRING VERDICT: retina={'LIVE' if r else 'SUSPECT'}  "
          f"composer={'LIVE' if c else 'SUSPECT'}")


if __name__ == "__main__":
    main()
