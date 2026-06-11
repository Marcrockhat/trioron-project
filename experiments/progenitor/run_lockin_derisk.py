"""Smallest de-risk test (s028 design §9): does a coherent stream clear the √N
noise floor while an empty one does not — WITHOUT labels?

Streams of PERIOD observations (one period = the whole budget). Two window types:
  coherent — one stable within-sample pattern, 1% feature jitter, and a random
             per-observation GLOBAL scale (the receptor must discard it);
  empty    — fresh uniform noise every observation.

Each window: receptor → phase → LockIn over the period → per-feature margin
(amplitude / √n). Decision rule (no labels): coherent iff any feature clears
K·√n. Two variants reported:
  naive  — every receptor deposits (exposes the pinned-max DC: empty ≈ N/F);
  masked — saturated (q=1000) and silent (q=0) receptors excluded (lockin.py doc).

PASS = masked variant separates perfectly across all seeds.
Run: python3 run_lockin_derisk.py   (~seconds, CPU)
"""
from __future__ import annotations

import math

import torch

from lockin import LockIn, evidence_mask
from receptor import N_QUANTA, quantize

SEEDS = 5
WINDOWS = 20      # per type per seed
PERIOD = 1000     # observations in one period
F = 4             # features
K = 3.0           # margin threshold, in noise-floor units


def coherent_window(g: torch.Generator) -> torch.Tensor:
    base = 0.1 + 0.9 * torch.rand(F, generator=g)                    # the pattern
    jitter = 1.0 + 0.01 * torch.randn(PERIOD, F, generator=g)        # 1% receptor noise
    scale = torch.exp(torch.randn(PERIOD, 1, generator=g))           # global gain, per obs
    return base * jitter * scale


def empty_window(g: torch.Generator) -> torch.Tensor:
    return torch.rand(PERIOD, F, generator=g)


def run_window(x: torch.Tensor, masked: bool) -> float:
    """Stream one window through the lock-in; return the max per-feature margin."""
    li = LockIn(F)
    for obs in x:
        q = quantize(obs)
        theta = 2 * math.pi * q / N_QUANTA
        li.step(theta, mask=evidence_mask(q) if masked else None)
    return li.margin().max().item()


def main() -> None:
    results = {("coherent", True): [], ("coherent", False): [],
               ("empty", True): [], ("empty", False): []}
    empty_dc = []  # max-feature amplitude/N on empty windows, naive (the pinned-max DC)

    for seed in range(SEEDS):
        g = torch.Generator().manual_seed(seed)
        for _ in range(WINDOWS):
            for kind, make in (("coherent", coherent_window), ("empty", empty_window)):
                x = make(g)
                for masked in (True, False):
                    m = run_window(x, masked)
                    results[(kind, masked)].append(m)
                if kind == "empty":
                    li = LockIn(F)
                    for obs in x:
                        q = quantize(obs)
                        li.step(2 * math.pi * q / N_QUANTA)
                    empty_dc.append((li.amplitude().max() / PERIOD).item())

    print(f"lock-in de-risk  seeds={SEEDS} windows={WINDOWS}/type/seed "
          f"period={PERIOD} F={F} K={K}  floor=sqrt(n)")
    print(f"{'variant':8s} {'type':9s} {'min':>8s} {'median':>8s} {'max':>8s}   margin = amp/sqrt(n)")
    for masked in (False, True):
        for kind in ("coherent", "empty"):
            v = torch.tensor(results[(kind, masked)])
            name = "masked" if masked else "naive"
            print(f"{name:8s} {kind:9s} {v.min():8.2f} {v.median():8.2f} {v.max():8.2f}")

    dc = torch.tensor(empty_dc)
    print(f"\nempty-window DC (naive, max-feature amplitude/N): "
          f"mean {dc.mean():.3f}  (predicted pinned-max bias 1/F = {1/F:.3f})")

    for masked in (False, True):
        name = "masked" if masked else "naive"
        co = torch.tensor(results[("coherent", masked)])
        em = torch.tensor(results[("empty", masked)])
        ok = (co.min() > K) and (em.max() < K)
        print(f"{name}: coherent min {co.min():.2f} vs empty max {em.max():.2f} "
              f"at K={K} -> {'PASS (perfect separation)' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
