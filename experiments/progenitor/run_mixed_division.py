"""De-risk: division as the DISCOVERY mechanism on a mixed stream (Rocky, s031).

Regime: the stream is all 32 classes shuffled together — no period/class
boundaries given. Window 1 births one blurred world-class; thereafter each
class accumulates its assigned deposits in a rolling buffer and, at every
window-boundary sitting, may DIVIDE on its least-coherent dim (circular
2-means on that dim's phases over the buffer), accepted only if BOTH
children clearly beat the parent's coherence ON THAT DIM — the local
split-vs-keep criterion (a tight unimodal dim shows no gain → rejected).
Single pass, unsupervised, gradient-free.

v0 falsified two designs: a global (12-dim mean) gain criterion can never
fire on a deep mixture (~gain/12 per split), and window-only membership
starves sittings once classes multiply. Hence per-dim gain + per-class
buffers.

Isolated from the substrate on purpose: one shared global frame (frame
translation is I4-validated; the unknown here is division). Mirrors the
lock-in phasor conventions (theta = 2*pi*q/N_QUANTA, unit phasors).

Run: python3 -m experiments.progenitor.run_mixed_division
"""
from __future__ import annotations

import math
from collections import Counter

import torch

from .data_hard import make_spec, sample

N_QUANTA = 1000
WINDOW = 1000          # samples per period (Rocky: 1 period = 1000 quanta)
S_PERIOD = 500         # per class -> 16_000-sample stream, 16 windows
N_TEST = 100
SEEDS = 3
BUF = 400              # rolling deposit buffer per class
MIN_MEMBERS = 60       # don't judge a class on fewer buffered deposits
MIN_CHILD = 25         # a mode must recur to be born
GAIN_D = 0.08          # split-dim coherence gain to accept a division
CAP = 128


def pockets(X: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    q = (X - lo) / (hi - lo) * (N_QUANTA - 2) + 1          # interior: (0, 1000)
    return q.clamp(1, N_QUANTA - 1)


def phasors(q: torch.Tensor) -> torch.Tensor:
    return torch.exp(1j * 2 * math.pi * q / N_QUANTA)


def circ_2means(phi: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """Boolean partition of angles phi by circular 2-means."""
    z = torch.exp(1j * phi)
    mu = torch.angle(z.mean())
    sd = math.sqrt(max(-2 * math.log(float(z.mean().abs().clamp(min=1e-6))), 1e-4))
    c = torch.tensor([mu - sd, mu + sd])
    for _ in range(iters):
        d = 1 - torch.cos(phi.unsqueeze(1) - c.unsqueeze(0))   # [N,2]
        side = d.argmin(1)
        for k in (0, 1):
            if (side == k).any():
                c[k] = torch.angle(z[side == k].mean())
    return side == 1


def run_seed(seed: int):
    spec = make_spec()
    tr, norm = sample(spec, n_per_class=S_PERIOD, seed=seed)
    te, _ = sample(spec, n_per_class=N_TEST, seed=100 + seed, norm=norm)
    lo = tr.x.min(0).values - 1e-3
    hi = tr.x.max(0).values + 1e-3

    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(tr), generator=g)
    Xs, ys = tr.x[order], tr.y[order]                  # the mixed stream
    Z_all = phasors(pockets(Xs, lo, hi))

    bufs: list[torch.Tensor] = []                      # per-class deposits
    n_divisions = 0

    for w0 in range(0, len(Xs), WINDOW):
        Z = Z_all[w0:w0 + WINDOW]
        if not bufs:                                   # genesis: the world-blur
            bufs = [Z[-BUF:]]
            continue
        T = torch.stack([b.mean(0) for b in bufs])
        member = (Z.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
        for k in range(len(bufs)):
            zk = Z[member == k]
            if len(zk):
                bufs[k] = torch.cat([bufs[k], zk])[-BUF:]

        new_bufs = []
        for b in bufs:                                 # the sitting: divide?
            if len(b) < MIN_MEMBERS or len(bufs) + len(new_bufs) >= CAP:
                new_bufs.append(b)
                continue
            R = b.mean(0).abs()
            d = int(R.argmin())
            side = circ_2means(torch.angle(b[:, d]))
            ba, bb = b[~side], b[side]
            if (len(ba) >= MIN_CHILD and len(bb) >= MIN_CHILD and
                    min(float(ba[:, d].mean().abs()), float(bb[:, d].mean().abs()))
                    > float(R[d]) + GAIN_D):
                new_bufs += [ba, bb]
                n_divisions += 1
            else:
                new_bufs.append(b)
        bufs = new_bufs

    # ── readout: purity map from the train stream, recall on held-out ──
    T = torch.stack([b.mean(0) for b in bufs])
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
    Zt = phasors(pockets(te.x[keep], lo, hi))
    pred = (Zt.unsqueeze(1) * T.conj().unsqueeze(0)).real.sum(-1).argmax(1)
    truth_of = torch.tensor([mapping[int(k)] for k in range(len(T))])
    y_te = te.y[keep]
    acc = float((truth_of[pred] == y_te).float().mean())
    per_class = {c: float((truth_of[pred[y_te == c]] == c).float().mean())
                 for c in clean}
    covered = len({mapping[k] for k in mapping if mapping[k] >= 0})
    return len(T), n_divisions, covered, sum(purities) / len(purities), acc, per_class


def main() -> None:
    accum = {}
    for seed in range(SEEDS):
        n_cls, n_div, covered, purity, acc, per_class = run_seed(seed)
        print(f"seed={seed}: {n_cls} classes from {n_div} divisions; "
              f"{covered}/32 truths covered; mean purity {purity:.3f}; "
              f"one-shot (clean) {acc:.3f}")
        for c, a in per_class.items():
            accum.setdefault(c, []).append(a)

    print(f"\n── per-class one-shot, 3-seed mean (filter was 0.386, "
          f"Bayes 0.993) ──")
    for c in sorted(accum):
        m = sum(accum[c]) / len(accum[c])
        print(f"  sp{c:03d}  {m:.3f}")
    overall = sum(sum(v) / len(v) for v in accum.values()) / len(accum)
    print(f"  mean   {overall:.3f}")


if __name__ == "__main__":
    main()
