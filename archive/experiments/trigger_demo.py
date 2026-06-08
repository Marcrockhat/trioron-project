"""§8-step-4 verification: growth trigger watched on the live curriculum.

Trains a capacity-bound TrioronNetwork (latent=2, tanh) on the
contrastive curriculum, wires up the three-condition trigger, and
logs per-condition state to CSV every step.

Reports:
- First step each individual condition went True (or never).
- First step the conjunction fired (or never).
- Steady-state behavior over the last 1000 steps:
  fraction of steps each condition held; total fires.

Outputs:
- outputs/trigger_demo_log.csv: full per-step trace of the trigger.
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
from trioron.triggers import GrowthTrigger, total_gradient_norm


N_STEPS = 8000
BATCH = 32
HIDDEN = 16
LATENT = 2
MARGIN = 1.0
LR = 3e-3
SEED = 0
LOG_EVERY = 200

# Trigger parameters — calibrated against the latent-dim sweep:
#   slope at latent=2 was -1.7e-6/step, so improvement over W=1000 is
#   ~0.0017 early in the run and decays. ε_loss=0.001 splits the noise
#   floor (~0.0004) and the early drift, so the plateau condition will
#   only become True once the slow drift has flattened further.
TRIGGER_W = 1000
TRIGGER_EPS_LOSS = 0.001
TRIGGER_EPS_RANK = 0.1
TRIGGER_G_MIN = 1e-4
TRIGGER_G_MAX = 10.0


def main() -> int:
    torch.manual_seed(SEED)

    net = TrioronNetwork(
        [
            (STATE_DIM, HIDDEN, "relu"),
            (HIDDEN, HIDDEN, "relu"),
            (HIDDEN, LATENT, "tanh"),
        ]
    )
    cur = ContrastiveCurriculum(seed=SEED)
    opt = optim.Adam(net.parameters(), lr=LR)

    trigger = GrowthTrigger(
        latent_dim=LATENT,
        window=TRIGGER_W,
        eps_loss=TRIGGER_EPS_LOSS,
        eps_rank=TRIGGER_EPS_RANK,
        g_min=TRIGGER_G_MIN,
        g_max=TRIGGER_G_MAX,
    )

    print("=" * 78)
    print("Trioron — Step 4 verification: growth trigger on capacity-bound run")
    print("=" * 78)
    print(f"Network:  {net}")
    print(f"Params:   {net.n_parameters()}")
    print(f"Trigger:  {trigger}")
    print(f"Steps:    {N_STEPS}   batch/pair: {BATCH}")
    print()

    log_rows: list[list] = []
    first: dict[str, int] = {
        "loss_plateau": -1,
        "rank_saturated": -1,
        "grad_stable": -1,
        "fire": -1,
    }
    n_fires = 0

    for step in range(N_STEPS):
        # Combined-loss training step (matches step-3 hardened recipe).
        # Hold onto one pair's hidden activations to feed the trigger:
        # we need a hidden tensor of shape (B, latent) for effective_rank.
        total = 0.0
        last_h = None
        for name in PAIR_NAMES:
            a, b = cur.sample_pair(name, batch=BATCH)
            h_a = net(a)
            h_b = net(b)
            l = contrastive_loss(h_a, h_b, margin=MARGIN)
            total = total + l
            if last_h is None:
                last_h = h_a.detach()
        loss = total / len(PAIR_NAMES)

        opt.zero_grad()
        loss.backward()
        gnorm = total_gradient_norm(net.parameters())
        opt.step()

        s = trigger.observe(loss=loss.item(), hidden=last_h, grad_norm=gnorm)

        for k in ("loss_plateau", "rank_saturated", "grad_stable", "fire"):
            if first[k] < 0 and getattr(s, k):
                first[k] = step
        if s.fire:
            n_fires += 1

        log_rows.append([
            s.step, s.loss, s.effective_rank, s.grad_norm,
            int(s.loss_plateau), int(s.rank_saturated), int(s.grad_stable),
            int(s.fire), s.loss_improvement, s.rank_recent_mean,
            s.grad_recent_median, int(s.warmup),
        ])

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            print(
                f"  step {step:5d}  loss {s.loss:.4f}  "
                f"rank {s.effective_rank:.3f}  gnorm {s.grad_norm:.3f}  "
                f"L={int(s.loss_plateau)} R={int(s.rank_saturated)} "
                f"G={int(s.grad_stable)} → fire={int(s.fire)}"
                + ("  [warmup]" if s.warmup else "")
            )

    # Steady-state analysis over last 1000 steps.
    tail = log_rows[-1000:]
    tail_loss_plateau = sum(r[4] for r in tail) / len(tail)
    tail_rank_saturated = sum(r[5] for r in tail) / len(tail)
    tail_grad_stable = sum(r[6] for r in tail) / len(tail)
    tail_fire = sum(r[7] for r in tail) / len(tail)

    print()
    print("=" * 78)
    print("First-True step per condition (−1 = never within run)")
    print("=" * 78)
    for k in ("loss_plateau", "rank_saturated", "grad_stable", "fire"):
        v = first[k]
        print(f"  {k:18s}: {v}")

    print()
    print(f"Total fires across run: {n_fires}")
    print()
    print("Steady-state (last 1000 steps, fraction of steps True):")
    print(f"  loss_plateau:    {tail_loss_plateau:.3f}")
    print(f"  rank_saturated:  {tail_rank_saturated:.3f}")
    print(f"  grad_stable:     {tail_grad_stable:.3f}")
    print(f"  fire (conjunction): {tail_fire:.3f}")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "trigger_demo_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "step", "loss", "effective_rank", "grad_norm",
            "loss_plateau", "rank_saturated", "grad_stable", "fire",
            "loss_improvement", "rank_recent_mean", "grad_recent_median",
            "warmup",
        ])
        w.writerows(log_rows)
    print(f"  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
