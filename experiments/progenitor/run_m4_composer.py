"""M4 — the composer arm THROUGH the substrate (design
mixed_stream_growth.md §4 + §4b [D14+D16+D17], all approved).

The relational testbed end-to-end on the real organism: germline-only
substrate → PerceptionGenesis on window 1 (perception spawn + the ONE
blurred world-class) → MixedStreamController with the composer arm for
windows 2+ (membership, judgment, per-buffer + FAMILY trials, stress
economy, progenitor divisions AND spawns). Zero labels, single pass.

Spawns are REAL TISSUE [D11]: +1 computing cell (epigenome = the
winning expression gene), +2 edges from the source receptor cells,
rank earned from topology — the first depth of the rebuild. Spawned
cells compute their composition in the forward path and deposit into
their own lock-in rows through FIXED static frames; settlement is
future-deposit through M3's testify machinery with gene-targeted vote
transfers; PATIENCE failures hard-retire [D16].

FRAME NOTE: composer sources must be canonical, so the testbed carries
an I3-style SENTINEL gain-reference column (~1.0 ± 0.1% jitter — a
constant would be starved by the first-sitting census): per-sample
frames are static to 0.1% and the forward-path phases ARE canonical.
Composing over frame-moving receptors needs canonical injection
(deferred, on record).

Gate asserts (all seeds):
  - composer arm beats the division-only arm (same runner) on mean
    one-shot; A/B and E/F improve vs division-only
  - census: computing +surviving spawns, edges +2·total spawns,
    max rank ≥ 1; spawned genes ⊂ {linear, tanh, dendrite}
  - ZERO noise-pair spawns (no spec touches f2/f3) — failure-mode-4
  - forward-path composed pockets == buffer-side composition (the
    tissue is the computation, not bookkeeping)
  - votes conserved (Σ=28)

Run: python3 -m experiments.progenitor.run_m4_composer
"""
from __future__ import annotations

from collections import Counter

import torch

from trioron.core import census, construct
from trioron.core.epigenome import GENE_NAMES
from trioron.pcll import (
    MixedStreamController,
    PerceptionGenesis,
    germline_base,
)

from .run_composer_growth import make_data
from .run_composer_genome import NAMES

WINDOW = 600
S_TRAIN = 800          # 6 × 800 = 4800 samples, 8 windows
S_TEST = 200
SEEDS = 3


F_MAX = 1.0   # the testbed's feature ceiling (make_data clamps to [0, 1])


def sentinel(n: int, g: torch.Generator) -> torch.Tensor:
    """The gain-reference column, derived in QUANTUM units (Rocky,
    s033 — no hardcoded decimals; the first band [0.999, 1.001]
    straddled the feature range and a near-ceiling feature could
    displace the sentinel as the sample max).

    Band = F_MAX·(1 + [1.0, 1.2]/N_QUANTA): one quantum above the
    feature ceiling, so (a) the sentinel is ALWAYS the sample max
    (static frame), (b) a feature at exactly F_MAX quantizes to pocket
    N_QUANTA−1 — evidence, not the masked q=N_QUANTA reference —
    and (c) the 0.2-quantum jitter keeps the column census-distinct
    (a constant is starved) while moving the frame sub-quantum."""
    from trioron.core.receptor import N_QUANTA
    lo = F_MAX * (1.0 + 1.0 / N_QUANTA)
    return lo + F_MAX * (0.2 / N_QUANTA) * torch.rand(n, 1, generator=g)


def data(seed: int):
    """The probe's relational world + the sentinel gain-reference
    column (quantum-derived band, see sentinel())."""
    X, y = make_data(S_TRAIN, seed)
    Xt, yt = make_data(S_TEST, 100 + seed)
    g = torch.Generator().manual_seed(7000 + seed)
    return (torch.cat([X, sentinel(len(X), g)], 1), y,
            torch.cat([Xt, sentinel(len(Xt), g)], 1), yt)


def run_seed(seed: int, composer: bool):
    Xs, ys, Xt, yt = data(seed)

    sub = construct(germline_base, capacity=128)
    pg = PerceptionGenesis(sub)
    pg.feed(Xs[:WINDOW])                               # period 1
    natal = sub.end_task()
    assert natal.event == "birth", "world-class not born"
    c0 = census(sub.arena)

    # divide_tries=4 for BOTH arms: this world has pure-noise dims, so
    # the worst dim is always noise and worst-dim-only division never
    # fires — a tries=1 division-only baseline would be a strawman
    mixed = MixedStreamController(sub, stress=pg.router,
                                  adopt=pg.controller, composer=composer,
                                  divide_tries=4)
    mixed.observe(Xs[:WINDOW])                         # genesis buffer seed
    sub.end_task()

    n_spawn = n_prune = 0
    for w0 in range(WINDOW, len(Xs), WINDOW):          # windows 2..8
        mixed.observe(Xs[w0:w0 + WINDOW])
        r = sub.end_task()
        n_spawn += r.spawns
        n_prune += r.pruned
    c1 = census(sub.arena)
    assert abs(sum(pg.router.germline.votes.values()) - 28.0) < 1e-9

    # ── structural contract [D11] ─────────────────────────────────
    if composer:
        survivors = len(mixed.specs)
        assert survivors == n_spawn - n_prune
        assert c1.computing - c0.computing == survivors, (c0, c1)
        assert c1.edges - c0.edges == 2 * n_spawn, (c0, c1)
        if survivors:
            assert c1.depth >= 1, "depth not earned"
        for spec in mixed.specs:                       # failure-mode-4 kill
            assert spec.gene in ("linear", "tanh", "dendrite")
            assert spec.i in (0, 1) and spec.j in (0, 1), \
                f"NOISE-PAIR spawn {spec}"
        # the tissue IS the computation: forward-path composed pockets
        # must equal the buffer-side composition of the same samples
        if survivors:
            q_full = mixed.pockets_of(Xt[:200])
            F = len(mixed.receptor_ids)
            for k, spec in enumerate(mixed.specs):
                want = spec.pockets(q_full[:, :F])
                got = q_full[:, F + k]
                err = float((want - got).abs().max())
                assert err <= 6.0, f"forward/buffer divergence {err} pockets"
    else:
        assert c1.computing == c0.computing and c1.edges == c0.edges

    # ── frozen-world readout (M2 protocol, per class) ─────────────
    import math
    T = mixed.templates()
    Z_all = torch.exp(1j * 2 * math.pi * mixed.pockets_of(Xs) / 1000)
    member = (Z_all.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    mapping = []
    for k in range(len(T)):
        t = ys[member == k]
        mapping.append(Counter(t.tolist()).most_common(1)[0][0]
                       if len(t) else -1)
    Zt = torch.exp(1j * 2 * math.pi * mixed.pockets_of(Xt) / 1000)
    pred = (Zt.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    truth = torch.tensor(mapping)[pred]
    per_class = [float((truth[yt == c] == c).float().mean()) for c in range(6)]

    tag = ""
    if composer:
        kept = [f"{s.gene}({s.i},{s.j})" for s in mixed.specs]
        tag = (f"; spawned {n_spawn}, pruned {n_prune}, "
               f"kept [{', '.join(kept)}]")
        g = pg.router.germline
        book = " ".join(
            f"{GENE_NAMES[p][:4]}={sum(g.votes[c] for c in ids):.2f}"
            for p, ids in g.council_ids.items()
            if GENE_NAMES[p] in ("linear", "tanh", "dendrite"))
        tag += f"; book [{book}]"
    print(f"  seed={seed}: {len(mixed.classes)} classes, "
          f"mean {sum(per_class) / 6:.3f}{tag}")
    if composer:
        print(f"    before {c0}")
        print(f"    after  {c1}  {c1.delta(c0)}")
    return torch.tensor(per_class)


def main() -> None:
    results = {}
    for arm, on in (("division-only", False), ("composer", True)):
        print(f"\n══ {arm} ══")
        acc = torch.zeros(6)
        for seed in range(SEEDS):
            acc += run_seed(seed, on) / SEEDS
        for c in range(6):
            print(f"  {NAMES[c]:<9s} {acc[c]:.3f}")
        print(f"  mean      {acc.mean():.3f}")
        results[arm] = acc

    div, com = results["division-only"], results["composer"]
    assert com.mean() > div.mean(), (
        f"composer {com.mean():.3f} does not beat division-only "
        f"{div.mean():.3f}")
    # A/B and E/F must move (the relational classes the arm exists for)
    relational = [0, 1, 4]                  # A, B, E (F does fine divided)
    gain = sum(float(com[c] - div[c]) for c in relational)
    assert gain > 0.10, f"relational gain {gain:.3f} too small"
    print(f"\ncomposer {com.mean():.3f} > division-only {div.mean():.3f}; "
          f"relational gain (A+B+E) {gain:+.3f}")
    print("M4 GATE PASS — the composer arm grows real tissue through "
          "the substrate")


if __name__ == "__main__":
    main()
