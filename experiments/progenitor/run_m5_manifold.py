"""M5 — the manifold adapter for PCLL (design mixed_stream_growth.md
§5/§6 [D15], approved).

Three consumers, three checks, on data_hard through the M2 mixed-stream
protocol:

  A. σ-READOUT — the same trained organism read two ways: the raw
     equal-weight matched filter (the s029-comparable arm, REPORTED
     ALONGSIDE) vs the sketch likelihood (per-dim evidence weighted by
     the class's own spread). The s031 diag measured the raw readout as
     the ~0.62→0.99 half of the Bayes-ceiling gap.
  B. ANNEALING — the s031 freeze-decay scenario: discover for one
     epoch, FREEZE structure (no divisions, no trials — deployment
     mode), then stream two more epochs. Without annealing the rolling
     buffers drift with membership pollution and the raw readout
     decays; with the sketches re-anchoring the buffers each quiescent
     boundary, the plateau holds. Both arms read RAW (annealing's
     effect is structural, not a readout trick).
  C. SHIP/WAKE — the organism ships (lifecycle/ship rides
     substrate._pcll.state_dict), wakes on a fresh substrate, and the
     restored controller reproduces the σ-readout EXACTLY (sketches,
     buffers, classes, codec round-trip).

TWO FALSIFICATIONS RECORDED (s033, measured here):

  1. The s031 estimate that σ-weighting closes the raw→Bayes half of
     the gap (~0.62→0.99) is WRONG on the post-division organism:
     diagonal σ-readout gives +0.009 mean over raw (0.771→0.780), and
     full-covariance (the chained-15 pump) is within noise
     (−0.009/+0.003/+0.001). Post-division fragments are already TIGHT
     — spread-weighting has little differential signal left. The
     remaining 0.78→0.99 gap lives in fragmentation/mapping/crossing
     ambiguity (the routing story, not the readout).
  2. The s031 freeze-decay does NOT reproduce under the corrected
     mixed-stream controller: two frozen epochs show no decay in
     either arm (matched-filter membership is stable here without
     help). Annealing's measured value (+0.013 end-accuracy, every
     seed) accrues during DISCOVERY (quiescent-boundary re-anchoring),
     not as a freeze rescue. The machinery remains the lifetime-
     horizon guard the decay scenario was designed for.

Run: python3 -m experiments.progenitor.run_m5_manifold
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


def grow(seed: int, *, manifold: bool):
    torch.manual_seed(seed)        # annealing samples draw from global RNG
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
    mixed = MixedStreamController(sub, stress=pg.router,
                                  adopt=pg.controller, manifold=manifold)
    mixed.observe(Xs[:WINDOW])
    sub.end_task()
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW])
        sub.end_task()

    clean = [c for c in range(len(tr.names)) if float(spec.std[c]) <= 0.3]
    keep = torch.isin(te.y, torch.tensor(clean))
    return sub, mixed, (Xs, ys), (te.x[keep], te.y[keep]), spec


def mapping_of(mixed, Xs, ys):
    T = mixed.templates()
    Z = torch.exp(1j * 2 * math.pi * mixed.pockets_of(Xs) / 1000)
    member = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    out = []
    for k in range(len(T)):
        t = ys[member == k]
        out.append(Counter(t.tolist()).most_common(1)[0][0] if len(t) else -1)
    return torch.tensor(out)


def read_raw(mixed, truth_of, X, y) -> float:
    T = mixed.templates()
    Z = torch.exp(1j * 2 * math.pi * mixed.pockets_of(X) / 1000)
    pred = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    return float((truth_of[pred] == y).float().mean())


def read_sigma(mixed, truth_of, X, y) -> float:
    """σ-likelihood readout; classes without a consumable sketch fall
    back to their raw matched-filter score rank (−inf columns never
    win, so blend by masking)."""
    q = mixed.pockets_of(X)
    names = [c.name for c in mixed.classes]
    ll = mixed.manifold.log_likelihood(q, names)
    T = mixed.templates()
    Z = torch.exp(1j * 2 * math.pi * q / 1000)
    raw = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1)
    usable = ~torch.isinf(ll[0])
    score = torch.where(usable.unsqueeze(0), ll,
                        torch.full_like(ll, -float("inf")))
    if not bool(usable.any()):
        return read_raw(mixed, truth_of, X, y)
    pred = score.argmax(1)
    # rows where NO class had a sketch can't occur (usable is global);
    # classes lacking sketches simply can't be predicted by σ — raw
    # decides for samples whose best raw class is sketch-less
    raw_pred = raw.argmax(1)
    pred = torch.where(usable[raw_pred], pred, raw_pred)
    return float((truth_of[pred] == y).float().mean())


def part_a_and_c(seed: int):
    sub, mixed, (Xs, ys), (Xt, yt), _ = grow(seed, manifold=True)
    truth_of = mapping_of(mixed, Xs, ys)
    raw = read_raw(mixed, truth_of, Xt, yt)
    sig = read_sigma(mixed, truth_of, Xt, yt)
    n_sk = len(mixed.manifold.sketches)
    print(f"  seed={seed}: {len(mixed.classes)} classes, {n_sk} sketches; "
          f"raw {raw:.3f} → σ-readout {sig:.3f}  (Δ {sig - raw:+.3f})")

    # ── C: ship → wake → identical σ-readout ──────────────────────
    import tempfile
    from trioron.lifecycle.ship import ship
    from trioron.lifecycle.wake import wake
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = ship(sub, f.name)
    sub2 = wake(path)
    sub2.compile()
    mixed2 = MixedStreamController(sub2, stress=None, manifold=True)
    mixed2.load_state_dict(sub2._pcll_state)
    truth2 = mapping_of(mixed2, Xs, ys)
    sig2 = read_sigma(mixed2, truth2, Xt, yt)
    assert abs(sig2 - sig) < 1e-9, f"round-trip drift {sig} → {sig2}"
    return raw, sig


def part_b(seed: int, *, manifold: bool):
    """Freeze-decay: discover 1 epoch, freeze, stream 2 more epochs of
    fresh draws; raw-readout accuracy at the end of each epoch."""
    sub, mixed, (Xs, ys), (Xt, yt), spec = grow(seed, manifold=manifold)
    mixed.freeze = True
    accs = [read_raw(mixed, mapping_of(mixed, Xs, ys), Xt, yt)]
    for epoch in range(2):
        _, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
        ep, _ = sample(spec, n_per_class=S_PERIOD,
                       seed=500 + 10 * seed + epoch, norm=norm)
        g = torch.Generator().manual_seed(900 + epoch)
        order = torch.randperm(len(ep.y), generator=g)
        Xe = ep.x[order]
        for w0 in range(0, len(Xe), WINDOW):
            mixed.observe(Xe[w0:w0 + WINDOW])
            sub.end_task()
        # the truth map is INDEXED BY CLASS POSITION: consolidation can
        # retire/merge classes under freeze, so remap before every read
        # (a stale cached map collapsed a seed to ~0.24, s033)
        accs.append(read_raw(mixed, mapping_of(mixed, Xs, ys), Xt, yt))
    return accs


def main() -> None:
    print("── A+C. σ-readout (raw arm alongside) + ship/wake round-trip ──")
    raws, sigs = [], []
    for seed in range(SEEDS):
        raw, sig = part_a_and_c(seed)
        raws.append(raw)
        sigs.append(sig)
    mraw, msig = sum(raws) / SEEDS, sum(sigs) / SEEDS
    print(f"  mean: raw {mraw:.3f} → σ-readout {msig:.3f} "
          f"(Bayes ceiling 0.993)")
    assert msig > mraw, "σ-readout does not beat the raw filter"

    print("\n── B. annealing vs freeze-decay (raw readout both arms) ──")
    drops = {}
    for arm, on in (("no-anneal", False), ("anneal", True)):
        end, drop = [], []
        for seed in range(SEEDS):
            accs = part_b(seed, manifold=on)
            end.append(accs[-1])
            drop.append(accs[0] - accs[-1])
            print(f"  {arm:>9s} seed={seed}: " +
                  " → ".join(f"{a:.3f}" for a in accs))
        drops[arm] = (sum(end) / SEEDS, sum(drop) / SEEDS)
        print(f"  {arm:>9s} mean end {drops[arm][0]:.3f} "
              f"(decay {drops[arm][1]:+.3f})")
    assert drops["anneal"][0] >= drops["no-anneal"][0], (
        "annealing does not hold the plateau")

    print(f"\nσ-readout {msig:.3f} > raw {mraw:.3f}; anneal end "
          f"{drops['anneal'][0]:.3f} ≥ no-anneal {drops['no-anneal'][0]:.3f}; "
          f"ship/wake exact")
    print("M5 GATE PASS — the manifold adapter")


if __name__ == "__main__":
    main()
