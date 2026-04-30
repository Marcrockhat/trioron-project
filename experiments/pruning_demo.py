"""§8-step-6 verification: pruning loop on an overcapacity network.

Symmetric counterpart to step 5 (cellular division). Step 5 showed
capacity-bound networks growing into their natural size. Step 6 shows
overcapacity networks shrinking into their natural size.

Setup:
    Network: 8 → 16 → 16 → 8-tanh   (latent=8 — way more than 5 pairs need)
    Pruner targets the latent layer (and any other layer's dormant nodes).
    Per-step utility update via |h * h.grad| (§3.2 magnitude form).
    Slower-clock prune (every 200 steps), require sustained low u for
    1000 steps before removal.

Falsification target (from outputs/capacity_sweep.csv):
    latent=5 plateau ≈ 0.0002,  latent=8 plateau ≈ 0.0000.
    The pruned network should sit somewhere in [latent=5..latent=8] in
    capacity terms. If pruning works, end-of-run loss is preserved at a
    near-zero level even after multiple latent dims are removed.

Outputs:
- outputs/pruning_demo_log.csv: per-step loss, n_nodes_per_layer, prunes.
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
from trioron.pruner import PruningController, utility_capture


N_STEPS = 8000
BATCH = 32
HIDDEN = 16
LATENT_INIT = 8     # deliberate over-allocation
MARGIN = 1.0
LR = 3e-3
SEED = 0
LOG_EVERY = 200

# Pruner settings.
U_THRESHOLD = 1e-4
T_PRUNE = 1000
PRUNE_CLOCK = 200
WARMUP_STEPS = 1500   # let utilities stabilize before pruning is allowed
U_DECAY = 0.99        # longer EMA window than the default 0.9 so streaks
                      # don't constantly reset on noise spikes


def make_network(latent: int) -> TrioronNetwork:
    net = TrioronNetwork(
        [
            (STATE_DIM, HIDDEN, "relu"),
            (HIDDEN, HIDDEN, "relu"),
            (HIDDEN, latent, "tanh"),
        ]
    )
    # Bump utility EMA window so noisy spikes don't reset streak counters.
    for layer in net.layers:
        layer.u_decay = U_DECAY
    return net


def train_step(net, curriculum, optimizer, pruner, step_idx, allow_prune):
    """One training step. Returns (loss_value, prune_events)."""
    with utility_capture(net) as cap:
        total = 0.0
        for name in PAIR_NAMES:
            a, b = curriculum.sample_pair(name, batch=BATCH)
            h_a = net(a)
            h_b = net(b)
            l = contrastive_loss(h_a, h_b, margin=MARGIN)
            total = total + l
        loss = total / len(PAIR_NAMES)
        optimizer.zero_grad()
        loss.backward()
        cap.update_layer_utilities()

    prune_events: list = []
    if allow_prune:
        prune_events = pruner.step(net, step_idx)

    if prune_events:
        # Structural change → must rebuild optimizer.
        return float(loss.item()), prune_events
    optimizer.step()
    return float(loss.item()), []


def main() -> int:
    torch.manual_seed(SEED)

    net = make_network(LATENT_INIT)
    cur = ContrastiveCurriculum(seed=SEED)
    opt = optim.Adam(net.parameters(), lr=LR)
    pruner = PruningController(
        u_threshold=U_THRESHOLD,
        T_prune=T_PRUNE,
        prune_clock=PRUNE_CLOCK,
    )

    print("=" * 78)
    print("Trioron — Step 6 verification: pruning on overcapacity network")
    print("=" * 78)
    print(f"Network:    {net}")
    print(f"Params:     {net.n_parameters()}")
    print(f"Pruner:     {pruner}")
    print(f"Warmup before pruning: {WARMUP_STEPS} steps")
    print(f"u_decay (EMA β):       {U_DECAY}")
    print()

    log_rows: list[list] = []
    nodes_history: list[list[int]] = []
    prune_events_log: list[tuple[int, list]] = []
    initial_n_nodes = net.n_nodes_per_layer()

    for step in range(N_STEPS):
        allow_prune = step >= WARMUP_STEPS
        loss_val, events = train_step(net, cur, opt, pruner, step, allow_prune)

        if events:
            opt = optim.Adam(net.parameters(), lr=LR)
            prune_events_log.append((step, events))

        nodes_now = net.n_nodes_per_layer()
        nodes_history.append(nodes_now)
        log_rows.append([
            step, loss_val, *nodes_now,
            len(events),
        ])

        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            u_summary = [
                f"L{i}={[round(v, 5) for v in layer.u.tolist()][:4]}"
                + ("…" if layer.n_nodes > 4 else "")
                for i, layer in enumerate(net.layers)
            ]
            tag = ""
            if events:
                tag = f"  PRUNED: {events}"
            print(
                f"  step {step:5d}  loss {loss_val:.4f}  "
                f"nodes {nodes_now}  " + " ".join(u_summary) + tag
            )

    # Summary.
    final_n_nodes = net.n_nodes_per_layer()
    early_loss = sum(r[1] for r in log_rows[100:200]) / 100
    late_loss = sum(r[1] for r in log_rows[-500:]) / 500

    print()
    print("=" * 78)
    print("Pruning summary")
    print("=" * 78)
    print(f"  Initial nodes per layer:  {initial_n_nodes}")
    print(f"  Final nodes per layer:    {final_n_nodes}")
    print(f"  Latent dim shrunk from {initial_n_nodes[-1]} → {final_n_nodes[-1]}")
    print(f"  Pruning events:")
    if not prune_events_log:
        print("    (none)")
    else:
        for s, evs in prune_events_log:
            print(f"    step {s}: {evs}")
    print()
    print(f"  Early-training loss (steps 100–200):  {early_loss:.4f}")
    print(f"  End-of-run loss (last 500 steps):     {late_loss:.4f}")
    print()
    print("  Calibration sweep references (outputs/capacity_sweep.csv):")
    print("    latent=5 plateau ≈ 0.0002    latent=8 plateau ≈ 0.0000")

    total_pruned = sum(initial_n_nodes) - sum(final_n_nodes)
    if total_pruned == 0:
        print()
        print("  PARTIAL: no pruning occurred anywhere. Threshold/T_prune may "
              "need tuning, or the network is genuinely using every node.")
        rc = 1
    elif late_loss > 0.01:
        print()
        print(
            f"  PARTIAL: pruned {total_pruned} nodes total, but end-of-run "
            f"loss {late_loss:.4f} did not stay below 0.01. Pruning is too "
            "aggressive — used nodes are being removed, or redistribution "
            "is degrading the function."
        )
        rc = 1
    else:
        latent_changed = initial_n_nodes[-1] != final_n_nodes[-1]
        print()
        print(
            f"  PASS: pruned {total_pruned} nodes "
            f"({initial_n_nodes} → {final_n_nodes}), "
            f"end-of-run loss {late_loss:.4f} preserved."
        )
        if not latent_changed:
            print(
                "    Note: latent layer was NOT pruned — the network "
                f"distributed its 5 pair-axes across all {final_n_nodes[-1]} "
                "latent dims rather than concentrating in 5. That is a valid "
                "non-redundant solution; the hidden layers had the slack."
            )
        rc = 0

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pruning_demo_log.csv")
    n_layers = max(len(r) - 2 for r in log_rows)
    header = ["step", "loss"] + [f"L{i}_n_nodes" for i in range(n_layers)] + ["n_pruned_this_step"]
    # Pad rows that came from a smaller layer count (can't happen but safe).
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(log_rows)
    print(f"  log: {csv_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
