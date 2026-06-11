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

Discrete case (s029): k-level features on fixed labeled lines θ_j = (2j+1)π/k
(no receptor, no mask — always interior). Coherent = stable level pattern with
2% flip noise; empty = fresh uniform levels each observation. Threshold is the
null-matched K (4 for binary's 1-D walk, 3 otherwise — lockin.matched_k).

PASS = masked continuous variant + every discrete k separate perfectly.
Run: python3 run_lockin_derisk.py   (~seconds, CPU)
"""
from __future__ import annotations

import math

import torch

from lockin import LockIn, evidence_mask, matched_k, theta_discrete
from receptor import N_QUANTA, quantize

SEEDS = 5
WINDOWS = 20      # per type per seed
PERIOD = 1000     # observations in one period
F = 4             # features
K = 3.0           # margin threshold for continuous channels (2-D Rayleigh null)
K_LEVELS = (2, 3, 5)   # discrete cases
FLIP = 0.02       # discrete coherent-pattern flip noise


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


def coherent_discrete_window(g: torch.Generator, k: int) -> torch.Tensor:
    base = torch.randint(0, k, (F,), generator=g)
    flip = torch.rand(PERIOD, F, generator=g) < FLIP
    rand = torch.randint(0, k, (PERIOD, F), generator=g)
    return torch.where(flip, rand, base.expand(PERIOD, F))


def empty_discrete_window(g: torch.Generator, k: int) -> torch.Tensor:
    return torch.randint(0, k, (PERIOD, F), generator=g)


def run_discrete_window(levels: torch.Tensor, k: int) -> float:
    """Labeled lines: fixed θ per level, no receptor, no mask (always interior)."""
    li = LockIn(F)
    for obs in levels:
        li.step(theta_discrete(obs, k))
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

    all_ok = True
    for masked in (False, True):
        name = "masked" if masked else "naive"
        co = torch.tensor(results[("coherent", masked)])
        em = torch.tensor(results[("empty", masked)])
        ok = (co.min() > K) and (em.max() < K)
        if masked:
            all_ok &= ok
        print(f"{name}: coherent min {co.min():.2f} vs empty max {em.max():.2f} "
              f"at K={K} -> {'PASS (perfect separation)' if ok else 'FAIL'}")

    print(f"\ndiscrete labeled lines  theta=(2j+1)*pi/k  flip={FLIP:.0%}  K=matched_k(k)")
    for k in K_LEVELS:
        co_m, em_m = [], []
        for seed in range(SEEDS):
            g = torch.Generator().manual_seed(seed)
            for _ in range(WINDOWS):
                co_m.append(run_discrete_window(coherent_discrete_window(g, k), k))
                em_m.append(run_discrete_window(empty_discrete_window(g, k), k))
        co, em, kk = torch.tensor(co_m), torch.tensor(em_m), matched_k(k)
        ok = (co.min() > kk) and (em.max() < kk)
        all_ok &= ok
        print(f"k={k} (K={kk:.0f}): coherent min {co.min():6.2f} | "
              f"empty max {em.max():5.2f} -> {'PASS' if ok else 'FAIL'}")

    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
