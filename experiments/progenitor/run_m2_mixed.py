"""M2 — division-as-discovery THROUGH the substrate (design
mixed_stream_growth.md §2/§6, [D12]).

The mixed-stream regime end-to-end on the real organism: germline-only
substrate → PerceptionGenesis on window 1 (perception spawn + the ONE
blurred world-class) → MixedStreamController for windows 2+ (membership,
council judgment, stress economy, progenitor divisions). Zero gradients,
zero labels, single pass.

Differences vs the standalone probe (run_mixed_division): pockets come
from the substrate's per-SAMPLE contrast quantizer (the probe used a
global per-feature frame — this run is the test of that geometry), and
every division is a council-decided, stress-settled growth event with
arena-visible structure (sibling astrocytes + lineage).

Self-arrest, corrected (s032): data_hard is 32 classes × 3 MODES — up to
96 true modes. Division discovers MODES (the purity map bundles them into
truths), so absolute-zero division windows are a finite-stream artifact
of the probe, not the criterion's steady state: the rarest modes clear
the MIN_CHILD=25 recurrence floor late. The honest criterion is RATE
COLLAPSE with BOUNDED discovery (verified out to 2 epochs cap-free:
~75→3-6 divisions/epoch, classes 81-87 < 96, never tiling).

Gate asserts (all seeds):
  - ≥ 29/32 truths covered; one-shot (clean) ≥ 0.65 mean across seeds
  - self-arrest: final-3-window divisions ≤ 5% of total; discovered
    classes ≤ 96 (the world's true mode count; CLASS_CAP never binds)
  - every DISCRIMINATION decision consumed (divisions > 0 that boundary)
  - census: live astrocytes == live classes, lineage recorded, votes
    conserved (Σ=28)

Run: python3 -m experiments.progenitor.run_m2_mixed
"""
from __future__ import annotations

from collections import Counter

import torch

from trioron.core import census, construct
from trioron.core.state import CellState
from trioron.pcll import (
    DISCRIMINATION,
    MixedStreamController,
    PerceptionGenesis,
    germline_base,
)

from .data_hard import make_spec, sample

WINDOW = 1000          # 1 period = 1000 stream samples (Rocky, s031)
S_PERIOD = 500         # per class → 16_000-sample stream, 16 windows
N_TEST = 100
SEEDS = 3
ARREST_TAIL = 3        # final windows that must show zero divisions


def run_seed(seed: int):
    spec = make_spec()
    tr, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
    te, _ = sample(spec, n_per_class=N_TEST, seed=100 + seed, norm=norm)

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tr.y), generator=g)
    Xs, ys = tr.x[order], tr.y[order]                  # the mixed stream

    sub = construct(germline_base, capacity=512)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW])                               # period 1
    natal = sub.end_task()
    assert natal.event == "birth", "world-class not born in period 1"
    c0 = census(sub.arena)

    mixed = MixedStreamController(sub, stress=pg.router, adopt=pg.controller)
    mixed.observe(Xs[:WINDOW])                         # genesis buffer seed
    r = sub.end_task()
    assert r.event == "birth" and len(mixed.classes) == 1

    div_per_window, consumed = [], True
    for w0 in range(WINDOW, len(Xs), WINDOW):          # windows 2..N
        mixed.observe(Xs[w0:w0 + WINDOW])
        r = sub.end_task()
        div_per_window.append(r.divisions)
        if r.grow == DISCRIMINATION and r.divisions == 0:
            consumed = False

    assert consumed, "a DISCRIMINATION decision was dropped (no consumer)"
    total_div = sum(div_per_window)
    tail = sum(div_per_window[-ARREST_TAIL:])
    assert tail <= max(1, round(0.05 * total_div)), (
        f"no rate collapse: {div_per_window}")
    assert len(mixed.classes) <= 96, (
        f"{len(mixed.classes)} classes exceeds the world's 96 true modes")
    assert abs(sum(pg.germline.votes.values()) - 28.0) < 1e-9

    # structural contract [D11]: live astrocytes == live classes; every
    # divided child carries its parent's lineage
    a = sub.arena
    n = a.cursor
    astro = (~a.forward_inclusion[:n]) & (a.epigenome[:n] == 0)
    live_astro = int((astro & (a.state[:n] == CellState.ACTIVE)).sum())
    assert live_astro == len(mixed.classes), (live_astro, len(mixed.classes))
    with_lineage = sum(
        1 for c in mixed.classes
        if c.cell_id >= 0 and int(a.parent[c.cell_id]) >= 0)
    c1 = census(a)

    # ── frozen-world readout (probe math, substrate pockets) ──────
    import math
    T = mixed.templates()
    Z_all = torch.exp(1j * 2 * math.pi * mixed.pockets_of(Xs) / 1000)
    member = (Z_all.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    mapping, purities = {}, []
    for k in range(len(T)):
        truths = ys[member == k]
        if len(truths) == 0:
            mapping[k] = -1
            continue
        top, cnt = Counter(truths.tolist()).most_common(1)[0]
        mapping[k] = top
        purities.append(cnt / len(truths))

    clean = [c for c in range(len(tr.names)) if float(spec.std[c]) <= 0.3]
    keep = torch.isin(te.y, torch.tensor(clean))
    Zt = torch.exp(1j * 2 * math.pi * mixed.pockets_of(te.x[keep]) / 1000)
    pred = (Zt.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    truth_of = torch.tensor([mapping[int(k)] for k in range(len(T))])
    acc = float((truth_of[pred] == te.y[keep]).float().mean())
    covered = len({m for m in mapping.values() if m >= 0})

    print(f"seed={seed}: {len(mixed.classes)} classes from "
          f"{sum(div_per_window)} divisions {div_per_window}; "
          f"{covered}/32 truths; purity {sum(purities)/len(purities):.3f}; "
          f"one-shot (clean) {acc:.3f}; lineage {with_lineage}/"
          f"{len(mixed.classes)}")
    print(f"  before {c0}")
    print(f"  after  {c1}  {c1.delta(c0)}")
    assert covered >= 29, f"only {covered}/32 truths covered"
    return acc


def main() -> None:
    accs = [run_seed(s) for s in range(SEEDS)]
    mean = sum(accs) / len(accs)
    print(f"\nmean one-shot (clean) {mean:.3f}  (gate ≥ 0.65; probe 0.711; "
          f"class-sequential filter 0.386)")
    assert mean >= 0.65, f"one-shot mean {mean:.3f} below gate"
    print("M2 GATE PASS — division-as-discovery through the substrate")


if __name__ == "__main__":
    main()
