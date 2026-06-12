"""D20 step-0 probe — membership headroom (Rocky, s033).

The session's measurement chain ended at: the entire remaining
data_hard gap (0.82→0.99) is SELF-LABELING quality — per-sample
membership has no "belongs to nothing I know" option, so every sample
(including the ~15%-of-stream disruptor spray) is force-routed into
some buffer (purity 0.70), and no downstream readout can undo it
(kNN-on-own-labels ≡ raw filter, D19 refuted at its step 0).

This probe measures, OFFLINE on trained organisms (no mechanism
change), what a per-sample margin floor at MEMBERSHIP time could
recover. The reject statistic is the resolver's own parameter-free
law applied per observation: a sample's evidence for class c,
E_c = Σ_d Re(z_d·T̄_cd), has exact null σ_c = √(Σ_d|T_cd|²/2);
margin = E_c/σ_c; a sample whose best margin < K joins NOTHING (it
still deposits to lock-in rows in the real mechanism — refusal is
about buffers, not evidence).

Sweep K ∈ {0..4}: refusal rate, buffer purity, starved buffers,
downstream clean-test accuracy with templates recomputed from gated
members (mapping remapped by gated majority). Plus: the disruptor
share of impurity at K=0, and an EM self-consistency pass
(reassign↔recompute to fixed point) at K=0 and the best K.

Run: python3 -m experiments.progenitor.run_d20_probe
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

KS = (0.0, 1.0, 2.0, 3.0, 4.0)
MIN_LIVE = 25          # a buffer below this after gating counts starved
EM_ROUNDS = 5


def grow(seed: int):
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
                                  divide_tries=4)
    mixed.observe(Xs[:WINDOW]); sub.end_task()
    for w0 in range(WINDOW, len(Xs), WINDOW):
        mixed.observe(Xs[w0:w0 + WINDOW]); sub.end_task()
    clean = [c for c in range(len(tr.names)) if float(spec.std[c]) <= 0.3]
    disrupt = set(range(len(tr.names))) - set(clean)
    keep = torch.isin(te.y, torch.tensor(clean))
    return (mixed, mixed.pockets_of(Xs), ys,
            mixed.pockets_of(te.x[keep]), te.y[keep], clean, disrupt)


def margins(q: torch.Tensor, T: torch.Tensor):
    """Per-sample (best class, margin) under the resolver's exact null:
    E_c = Σ Re(z·T̄_c), σ_c = √(Σ|T_c|²/2), margin = E/σ."""
    z = torch.exp(1j * 2 * math.pi * q / 1000)
    E = (z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1)     # [B, C]
    sig = ((T.abs() ** 2).sum(-1) / 2).sqrt().clamp_min(1e-9)     # [C]
    m = E / sig
    best = m.argmax(1)
    return best, m.gather(1, best.unsqueeze(1)).squeeze(1)


def assign_and_score(q_tr, ys, q_te, yt, T, K):
    """One gated assignment pass: members with margin ≥ K join their
    best class; templates + truth map recomputed; clean-test accuracy
    with the new templates (argmax, no reject at eval)."""
    best, marg = margins(q_tr, T)
    kept = marg >= K
    C = T.shape[0]
    z_tr = torch.exp(1j * 2 * math.pi * q_tr / 1000)
    T2, truth_of, purity, starved = [], [], [], 0
    for k in range(C):
        rows = kept & (best == k)
        t = ys[rows]
        if int(rows.sum()) < MIN_LIVE or len(t) == 0:
            starved += 1
            T2.append(T[k])
            truth_of.append(-1)
            continue
        T2.append(z_tr[rows].mean(0))
        maj, n_maj = Counter(t.tolist()).most_common(1)[0]
        truth_of.append(maj)
        purity.append(n_maj / len(t))
    T2 = torch.stack(T2)
    truth_of = torch.tensor(truth_of)
    z_te = torch.exp(1j * 2 * math.pi * q_te / 1000)
    pred = (z_te.unsqueeze(1) * T2.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    acc = float((truth_of[pred] == yt).float().mean())
    return {
        "refused": float((~kept).float().mean()),
        "purity": sum(purity) / max(len(purity), 1),
        "starved": starved,
        "acc": acc,
        "T": T2,
    }


def main() -> None:
    agg = {K: Counter() for K in KS}
    em_rows = []
    for seed in range(SEEDS):
        mixed, q_tr, ys, q_te, yt, clean, disrupt = grow(seed)
        T0 = mixed.templates()

        # disruptor share of impurity at K=0
        best, _ = margins(q_tr, T0)
        wrong_dis = wrong = 0
        truth0 = []
        for k in range(T0.shape[0]):
            t = ys[best == k]
            if len(t) == 0:
                truth0.append(-1)
                continue
            maj = Counter(t.tolist()).most_common(1)[0][0]
            truth0.append(maj)
            bad = t[t != maj]
            wrong += len(bad)
            wrong_dis += sum(1 for x in bad.tolist() if x in disrupt)
        print(f"seed={seed}: impurity from disruptor classes "
              f"{wrong_dis}/{wrong} = {wrong_dis / max(wrong, 1):.0%}")

        for K in KS:
            r = assign_and_score(q_tr, ys, q_te, yt, T0, K)
            agg[K]["refused"] += r["refused"] / SEEDS
            agg[K]["purity"] += r["purity"] / SEEDS
            agg[K]["starved"] += r["starved"] / SEEDS
            agg[K]["acc"] += r["acc"] / SEEDS

        # EM self-consistency at K=0 and K=2
        for K in (0.0, 2.0):
            T = T0
            accs = []
            for _ in range(EM_ROUNDS):
                r = assign_and_score(q_tr, ys, q_te, yt, T, K)
                accs.append(r["acc"])
                T = r["T"]
            em_rows.append((seed, K, accs))

    print(f"\n  {'K':>4s} {'refused':>8s} {'purity':>7s} {'starved':>8s} "
          f"{'clean acc':>10s}")
    for K in KS:
        a = agg[K]
        print(f"  {K:>4.1f} {a['refused']:>8.1%} {a['purity']:>7.3f} "
              f"{a['starved']:>8.1f} {a['acc']:>10.3f}")
    print("\n  EM self-consistency (acc per round):")
    for seed, K, accs in em_rows:
        print(f"    seed={seed} K={K:.0f}: " +
              " → ".join(f"{a:.3f}" for a in accs))


if __name__ == "__main__":
    main()
