"""Tiny 3-class stream simulator (s041 pedagogy) — shows the receptor's two
quantizations and the MOVING canonical frame, on a 3x5 grid.

X database = grayscale variants of three ideal digits (1, 2, 3). We stream the
samples one at a time, maintaining the running canonical frame (_flo/_fhi)
exactly as trioron/pcll/mixed.py does, and watch ONE anchor sample's pockets:
  - raw per-sample q   : history-INDEPENDENT (core/receptor.quantize)
  - canonical q        : re-mapped into the stream-wide frame -> DRIFTS as
                         brighter/more-extreme samples arrive, then FREEZES.

Run:  python3 experiments/progenitor/stream_sim_3class.py
"""
from __future__ import annotations

import torch

from trioron.core.receptor import N_QUANTA, quantize, quantize_frame

# ── ideal templates (3 wide x 5 tall, row-major) ────────────────────────
IDEAL = {
    "1": [0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0],
    "2": [1, 1, 1,  1, 0, 1,  0, 0, 1,  0, 1, 0,  1, 1, 1],
    "3": [1, 1, 1,  0, 0, 1,  1, 1, 1,  0, 0, 1,  1, 1, 1],
}


def variant(label, g):
    """One grayscale realization with a per-sample INK STRENGTH s~U(0.45,1.0):
    on-pixels = s*U(0.75,1.0), off = small. A faint sample (low s) looks
    full-contrast in isolation but only reaches part of the common dial."""
    t = torch.tensor(IDEAL[label], dtype=torch.float32)
    s = 0.45 + 0.55 * torch.rand(1, generator=g)
    on = s * (0.75 + 0.25 * torch.rand(15, generator=g))
    off = 0.12 * torch.rand(15, generator=g)
    return torch.where(t > 0.5, on, off).clamp(0, 1)


def build_X(n_per_class, seed):
    g = torch.Generator().manual_seed(seed)
    samples, labels = [], []
    for lab in IDEAL:
        for _ in range(n_per_class):
            samples.append(variant(lab, g))
            labels.append(lab)
    X = torch.stack(samples)
    order = torch.randperm(len(X), generator=g)          # shuffle into a stream
    return X[order], [labels[i] for i in order.tolist()]


def canonical(q_raw, lo_s, hi_s, FLO, FHI):
    """mixed.py _canonical: q' = a*q + b into the stream frame [FLO,FHI]."""
    span = FHI - FLO
    if span <= 1e-9:
        return q_raw.clone()
    a = (hi_s - lo_s) / span
    b = N_QUANTA * (lo_s - FLO) / span
    return (a * q_raw + b).clamp(1, N_QUANTA - 1).round()


if __name__ == "__main__":
    X, y = build_X(n_per_class=20, seed=7)
    print(f"X database: {len(X)} samples, 3 classes (1/2/3), 3x5 grayscale.\n")

    # anchor = the FAINTEST '2' in the stream (lowest own-peak); its RAW q
    # never changes, but the common frame compresses it the most
    twos = [i for i, lab in enumerate(y) if lab == "2"]
    a_idx = min(twos, key=lambda i: float(X[i].max()))
    anchor = X[a_idx]
    q_raw = quantize(anchor).to(torch.int64)
    _, alo, ahi = quantize_frame(anchor.unsqueeze(0))
    alo, ahi = float(alo), float(ahi)
    print(f"anchor = stream[{a_idx}] (class '2'), own range [{alo:.2f},{ahi:.2f}]")
    print(f"raw per-sample q (FIXED, history-independent):\n  {q_raw.tolist()}\n")

    # stream the samples, growing the canonical frame; snapshot anchor's
    # canonical q + the frame at checkpoints, then FREEZE partway
    FLO, FHI = float("inf"), float("-inf")
    frozen = False
    checkpoints = {1, 10, 30, 45, 60}
    print(f"{'after':>6} {'frame [LO,HI]':>16} {'frozen':>7}   anchor canonical q")
    for t in range(len(X)):
        if t == 30:
            frozen = True                                # freeze the gain here
        if not frozen:
            _, lo, hi = quantize_frame(X[t].unsqueeze(0))
            FLO, FHI = min(FLO, float(lo)), max(FHI, float(hi))
        if (t + 1) in checkpoints:
            qc = canonical(q_raw.float(), alo, ahi, FLO, FHI).to(torch.int64)
            print(f"{t + 1:>6} {f'[{FLO:.2f},{FHI:.2f}]':>16} {str(frozen):>7}   {qc.tolist()}")

    print("\nReading: raw q is identical no matter where the anchor sits in the")
    print("stream. Its CANONICAL q slides down as brighter samples widen HI —")
    print("until the freeze at t=30 pins the frame; after that it stops moving.")
