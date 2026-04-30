"""Step-3 (hardened) experiment: train a TrioronNetwork on the
contrastive curriculum with deliberate capacity tension.

The naïve form (random pair per step, 8-D latent) drives loss to zero
within ~100 steps because every pair gets its own latent direction.
That is useless as a signal for the §4 growth trigger: a loss that's
zero forever is "plateaued" but says nothing about whether capacity
is actually saturated.

Hardened design:
- LATENT = 2, with tanh on the output layer (bounded to [-1, 1]^2 so
  the network can't trivially satisfy margin by inflating magnitudes).
- Every step trains ALL 5 pairs simultaneously: the loss is the mean
  of the 5 per-pair contrastive losses. The network can no longer
  "solve one pair at a time" — it has to find a 2-D embedding that
  separates 5 independent binary concepts at once.

With 5 binary axes squeezed into a bounded 2-D plane and margin=1,
exact zero loss is geometrically impossible. The loss plateaus at a
nonzero value reflecting how well the network compromises among the
5 separation requirements. THAT plateau — distinguishable from
zero — is the load-bearing signal for the step-4 growth trigger.

Outputs:
- Stdout: per-step combined + per-pair loss, plateau summary.
- outputs/incubation_smoke_loss.csv:
    columns step, total_loss, <pair>_loss for each of the 5 pairs.
"""
from __future__ import annotations
import csv
import os
import sys

import torch
import torch.optim as optim

# Allow running as `python3 experiments/incubation_smoke.py` from repo root.
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
LATENT = 2          # deliberately starved — see module docstring
MARGIN = 1.0
LR = 3e-3
LOG_EVERY = 100
SEED = 0
PLATEAU_FRAC = 0.1  # plateau = mean of last 10% of step losses


def make_network() -> TrioronNetwork:
    return TrioronNetwork(
        [
            (STATE_DIM, HIDDEN, "relu"),
            (HIDDEN, HIDDEN, "relu"),
            (HIDDEN, LATENT, "tanh"),
        ]
    )


def main() -> int:
    torch.manual_seed(SEED)

    net = make_network()
    cur = ContrastiveCurriculum(seed=SEED)
    opt = optim.Adam(net.parameters(), lr=LR)

    log_rows: list[list[float]] = []
    initial_per_pair: dict[str, float] = {}
    final_per_pair: dict[str, float] = {}
    total_history: list[float] = []

    print("=" * 72)
    print("Trioron — Step 3 (hardened): capacity-starved contrastive curriculum")
    print("=" * 72)
    print(f"Network: {net}")
    print(f"Parameters: {net.n_parameters()}")
    print(f"Pairs:   {PAIR_NAMES}")
    print(f"Latent:  {LATENT}-D bounded by tanh   margin: {MARGIN}")
    print(f"Steps:   {N_STEPS}   batch/pair: {BATCH}   lr: {LR}")
    print()

    # Baseline (untrained) per-pair loss.
    with torch.no_grad():
        for name in PAIR_NAMES:
            a, b = cur.sample_pair(name, batch=256)
            initial_per_pair[name] = contrastive_loss(net(a), net(b), margin=MARGIN).item()

    # Train: every step samples one batch per pair and sums the 5 contrastive
    # losses. The mean is what the optimizer sees; per-pair components are
    # logged for diagnostics.
    for step in range(N_STEPS):
        per_pair_losses: list[float] = []
        total = 0.0
        for name in PAIR_NAMES:
            a, b = cur.sample_pair(name, batch=BATCH)
            l = contrastive_loss(net(a), net(b), margin=MARGIN)
            per_pair_losses.append(l.item())
            total = total + l
        loss = total / len(PAIR_NAMES)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_history.append(loss.item())
        log_rows.append([step, loss.item(), *per_pair_losses])

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            per_pair_str = "  ".join(
                f"{n[:4]}={v:.3f}" for n, v in zip(PAIR_NAMES, per_pair_losses)
            )
            print(f"    step {step:5d}: total {loss.item():.4f}   {per_pair_str}")

    # Final per-pair eval.
    with torch.no_grad():
        for name in PAIR_NAMES:
            a, b = cur.sample_pair(name, batch=256)
            final_per_pair[name] = contrastive_loss(net(a), net(b), margin=MARGIN).item()

    plateau_n = max(1, int(PLATEAU_FRAC * N_STEPS))
    plateau_loss = sum(total_history[-plateau_n:]) / plateau_n

    print()
    print("=" * 72)
    print("Per-pair contrastive loss (margin=1.0; lower = better separation)")
    print("=" * 72)
    print(f"  {'pair':28s} {'initial':>10s} {'final':>10s} {'Δ':>10s}")
    for name in PAIR_NAMES:
        i = initial_per_pair[name]
        f = final_per_pair[name]
        print(f"  {name:28s} {i:10.4f} {f:10.4f} {i - f:+10.4f}")

    print()
    print(f"  Plateau (mean over last {plateau_n} steps): {plateau_loss:.4f}")
    print(
        "  Expectation: nonzero (5 binary concepts in 2-D bounded latent "
        "cannot all reach margin=1)."
    )

    # Pass criterion: every pair improved from initial AND plateau is nonzero.
    improved = sum(1 for n in PAIR_NAMES if final_per_pair[n] < initial_per_pair[n])
    if improved == len(PAIR_NAMES) and plateau_loss > 0.01:
        print(f"  PASS: all 5 pairs improved; plateau {plateau_loss:.4f} is non-trivial")
        rc = 0
    elif plateau_loss <= 0.01:
        print(
            f"  WARN: plateau {plateau_loss:.4f} too close to zero — "
            "curriculum may still be too easy; revisit before step 4."
        )
        rc = 1
    else:
        print(f"  PARTIAL: {improved}/5 pairs improved, plateau {plateau_loss:.4f}")
        rc = 1

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "incubation_smoke_loss.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "total"] + [f"{p}_loss" for p in PAIR_NAMES])
        w.writerows(log_rows)
    print(f"  log: {csv_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
