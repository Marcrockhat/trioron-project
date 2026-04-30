"""Step-3 calibration: capacity sweep over latent dim.

Why this exists:
    The hardened smoke experiment shows the contrastive curriculum
    plateaus at a non-trivial value when latent=2 — but we have not
    verified that the plateau is *because of* capacity tension rather
    than optimizer pathology. This sweep tests the geometric
    prediction directly:

        plateau_loss(d) should monotonically decrease in d
        and approach ~0 when d ≥ k_pairs (= 5).

    If that holds, the curriculum is a credible test bench for §4
    (growth trigger) and §5 (cellular division): when division adds a
    latent dim, the loss SHOULD drop by a predictable amount.

Also reports a "plateau slope" — a linear fit of total loss vs step
over the last 30% of training. If slope is meaningfully negative the
plateau hasn't truly arrived; that bounds the noise floor that step
4's `ε_loss` must clear.

Outputs:
- Stdout: per-(latent, seed) plateau, slope, per-pair finals.
- outputs/capacity_sweep.csv: one row per run.
"""
from __future__ import annotations
import csv
import os
import sys

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron.incubator import (
    STATE_DIM,
    ContrastiveCurriculum,
    PAIR_NAMES,
    contrastive_loss,
)


N_STEPS = 3000
BATCH = 32
HIDDEN = 16
MARGIN = 1.0
LR = 3e-3
PLATEAU_FRAC = 0.3
LATENTS = [1, 2, 3, 4, 5, 8]
SEEDS = [0, 1, 2]


def _slope(ys: list[float]) -> float:
    """Least-squares slope of ys vs index. Returned per-step."""
    n = len(ys)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(ys) / n
    num = sum((i - mean_x) * (y - mean_y) for i, y in enumerate(ys))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den != 0 else 0.0


def run_one(latent: int, seed: int) -> dict:
    torch.manual_seed(seed)
    net = TrioronNetwork(
        [
            (STATE_DIM, HIDDEN, "relu"),
            (HIDDEN, HIDDEN, "relu"),
            (HIDDEN, latent, "tanh"),
        ]
    )
    cur = ContrastiveCurriculum(seed=seed)
    opt = optim.Adam(net.parameters(), lr=LR)

    history: list[float] = []
    for _ in range(N_STEPS):
        total = 0.0
        for name in PAIR_NAMES:
            a, b = cur.sample_pair(name, batch=BATCH)
            total = total + contrastive_loss(net(a), net(b), margin=MARGIN)
        loss = total / len(PAIR_NAMES)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())

    plateau_n = max(2, int(PLATEAU_FRAC * N_STEPS))
    plateau_window = history[-plateau_n:]
    plateau_mean = sum(plateau_window) / plateau_n
    plateau_slope = _slope(plateau_window)

    final_per_pair: dict[str, float] = {}
    with torch.no_grad():
        for name in PAIR_NAMES:
            a, b = cur.sample_pair(name, batch=256)
            final_per_pair[name] = contrastive_loss(net(a), net(b), margin=MARGIN).item()

    return {
        "latent": latent,
        "seed": seed,
        "n_params": net.n_parameters(),
        "plateau_mean": plateau_mean,
        "plateau_slope": plateau_slope,
        "per_pair": final_per_pair,
    }


def main() -> int:
    print("=" * 78)
    print("Trioron — Step-3 calibration: capacity sweep over latent dim")
    print("=" * 78)
    print(f"Latents: {LATENTS}   seeds: {SEEDS}")
    print(f"Steps:   {N_STEPS}   batch/pair: {BATCH}   margin: {MARGIN}")
    print()

    rows: list[dict] = []
    for latent in LATENTS:
        for seed in SEEDS:
            r = run_one(latent, seed)
            rows.append(r)
            print(
                f"  latent={r['latent']}  seed={r['seed']}  "
                f"params={r['n_params']:4d}  "
                f"plateau={r['plateau_mean']:.4f}  "
                f"slope/step={r['plateau_slope']:+.2e}"
            )

    # Aggregate by latent.
    print()
    print("=" * 78)
    print("Aggregate (mean ± std across seeds)")
    print("=" * 78)
    print(f"  {'latent':>6s}  {'plateau':>14s}  {'slope/step':>14s}")
    for latent in LATENTS:
        plateaus = [r["plateau_mean"] for r in rows if r["latent"] == latent]
        slopes = [r["plateau_slope"] for r in rows if r["latent"] == latent]
        m = sum(plateaus) / len(plateaus)
        v = sum((x - m) ** 2 for x in plateaus) / len(plateaus)
        s = v ** 0.5
        ms = sum(slopes) / len(slopes)
        print(f"  {latent:>6d}  {m:>8.4f} ± {s:.4f}  {ms:>+14.2e}")

    print()
    print("Interpretation cues:")
    print("  - plateau should fall monotonically with latent.")
    print("  - slope/step << 0 means the plateau hasn't truly arrived;")
    print("    that magnitude is a floor on step-4's ε_loss threshold.")
    print("  - latent ≥ 5 should put plateau near zero (one dim per pair).")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "capacity_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["latent", "seed", "n_params", "plateau_mean", "plateau_slope"]
        header += [f"{p}_final" for p in PAIR_NAMES]
        w.writerow(header)
        for r in rows:
            base = [r["latent"], r["seed"], r["n_params"], r["plateau_mean"], r["plateau_slope"]]
            base += [r["per_pair"][p] for p in PAIR_NAMES]
            w.writerow(base)
    print(f"  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
