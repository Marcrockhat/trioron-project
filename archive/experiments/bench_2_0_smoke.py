"""Trioron 2.0 — end-to-end smoke bench for the new edge-level machinery.

Goal: exercise the four 2.0 primitives on a real training run and
confirm each integrates cleanly with the existing forward/backward/
optimizer/EWC stack. This is a *smoke* — not a scientific delta.
Pass criterion: each arm trains, converges, and the 2.0 machinery
doesn't error out. Reported numbers are descriptive, not headline.

Three arms, all share the same seed/data/baseline architecture:

  baseline       — vanilla TrioronNetwork at sequential default
  long_range     — one hand-added long-range edge from L0 → L2 before
                   Task B; column init = zeros, src node = arbitrary
                   pick (node 0 of L0). Tests that grow_input(source=)
                   plumbing reaches all the way through backward.
  insert_layer   — `insert_layer` between L0 and L1 with identity init
                   + relu activation. Tests that the pseudo-block
                   primitive integrates with subsequent training and
                   gradient flow lands cleanly on the new layer.

The task is a synthetic 4-class XOR-style classification (input
dim 6, 4 classes, ~3-sample-per-class quirks) to keep the bench under
5 seconds end-to-end. 3 seeds.

Output written to outputs/bench_2_0_smoke.csv.
"""

from __future__ import annotations
import csv
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trioron.network import TrioronNetwork
from trioron.node import _EwcZeroWarning


# Pre-disarm the silent-zero EWC warning — first-task training has
# all-zero lambda by construction (consolidation hasn't run yet) and
# the warning would dump noise into the smoke log.
_EwcZeroWarning._warned = True


N_SAMPLES = 200
N_CLASSES = 2
INPUT_DIM = 6
N_STEPS_TASK = 400
LR = 5e-3
EWC_BETA = 2.0
SEEDS = (0, 1, 2)

# Task A asks "is input feature 0 positive?" — uses dim 0 only.
# Task B asks "is input feature 3 positive?" — uses dim 3 only.
# Features 0 and 3 are independent, so the second task requires
# learning a new direction in input space while EWC tries to
# preserve task A. This is a clean continual-learning setup.
TASK_A_DIM = 0
TASK_B_DIM = 3


def gen_data(seed: int, label_dim: int, n_samples: int = N_SAMPLES):
    """Binary classification: y = (x[label_dim] > 0).long(). Inputs are
    Gaussian on the unit sphere with all dims independent."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, INPUT_DIM, generator=g)
    y = (x[:, label_dim] > 0).long()
    return x, y


def train_phase(net, x, y, n_steps: int, ewc_beta: float = 0.0):
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for step in range(n_steps):
        opt.zero_grad()
        logits = net(x)
        loss = F.cross_entropy(logits, y)
        if ewc_beta > 0:
            loss = loss + ewc_beta * net.ewc_penalty()
        loss.backward()
        opt.step()


def consolidate(net, x, y):
    net.reset_fisher_all()
    opt = torch.optim.Adam(net.parameters(), lr=LR * 0.2)
    for _ in range(40):
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
        loss.backward()
        net.update_fisher_all()
        opt.step()
    net.update_lambda_all()
    net.anchor_all()


def accuracy(net, x, y) -> float:
    with torch.no_grad():
        return float((net(x).argmax(dim=1) == y).float().mean().item())


def build_baseline_net():
    return TrioronNetwork(
        [(INPUT_DIM, 8, "relu"), (8, 6, "relu"), (6, N_CLASSES, "linear")]
    )


def run_seed(seed: int, arm: str) -> dict:
    torch.manual_seed(seed)
    net = build_baseline_net()

    # Task A: is input feature 0 positive?
    x_a, y_a = gen_data(seed, label_dim=TASK_A_DIM)
    train_phase(net, x_a, y_a, n_steps=N_STEPS_TASK, ewc_beta=0.0)
    consolidate(net, x_a, y_a)
    acc_a_post = accuracy(net, *gen_data(seed + 1000, label_dim=TASK_A_DIM))

    # ---- Arm-specific mutation between tasks ----
    if arm == "long_range":
        # Add a long-range edge from L0 directly into L2 (the head).
        # We pick the source node greedily as L0's most active node on
        # task B's input — a coarse proxy for "the node that already
        # carries useful Task B signal." init_col=None → zeros → the
        # edge contributes 0 initially; backprop trains the new column
        # under no EWC stiffness (fresh column, fisher_W column = 0).
        x_b_probe, _ = gen_data(seed + 1, label_dim=TASK_B_DIM)
        with torch.no_grad():
            l0_out = net.layers[0](x_b_probe)
            src_node = int(l0_out.abs().mean(dim=0).argmax().item())
        net.layers[2].grow_input(init_col=None, source=(0, src_node))
    elif arm == "insert_layer":
        # Insert a relu identity layer between L0 and L1. With relu,
        # the layer clips negative pre-activations, materializing a
        # genuine new nonlinearity in the sequential path. Identity
        # init means post-insertion forward ≈ pre-insertion on the
        # positive orthant — gradient descent then specializes.
        net.insert_layer(
            between=(0, 1), activation="relu", init_mode="identity",
        )
    elif arm != "baseline":
        raise ValueError(f"unknown arm {arm}")

    # Task B: is input feature 3 positive? Different feature dim than A
    # → the network must learn a new direction while EWC pulls toward
    # the Task A solution.
    x_b, y_b = gen_data(seed + 1, label_dim=TASK_B_DIM)
    train_phase(net, x_b, y_b, n_steps=N_STEPS_TASK, ewc_beta=EWC_BETA)
    acc_a_final = accuracy(net, *gen_data(seed + 1000, label_dim=TASK_A_DIM))
    acc_b_final = accuracy(net, *gen_data(seed + 1001, label_dim=TASK_B_DIM))

    return {
        "seed": seed,
        "arm": arm,
        "acc_a_post_task_a": acc_a_post,
        "acc_a_after_task_b": acc_a_final,
        "acc_b": acc_b_final,
        "forgetting": acc_a_post - acc_a_final,
        "n_params": net.n_parameters(),
        "n_nodes_per_layer": tuple(net.n_nodes_per_layer()),
    }


def main():
    out_path = Path(__file__).resolve().parent.parent / "outputs" / "bench_2_0_smoke.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arms = ("baseline", "long_range", "insert_layer")
    rows: list[dict] = []

    t0 = time.time()
    for arm in arms:
        for seed in SEEDS:
            r = run_seed(seed, arm)
            rows.append(r)
            print(
                f"  {arm:>12} seed={seed}  "
                f"task_A_post={r['acc_a_post_task_a']:.3f}  "
                f"task_A_final={r['acc_a_after_task_b']:.3f}  "
                f"task_B={r['acc_b']:.3f}  "
                f"forgetting={r['forgetting']:+.3f}  "
                f"params={r['n_params']}  arch={r['n_nodes_per_layer']}"
            )
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")

    print("\n--- per-arm summary (mean over seeds) ---")
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        acc_b_mean = statistics.mean(r["acc_b"] for r in arm_rows)
        forget_mean = statistics.mean(r["forgetting"] for r in arm_rows)
        params_mean = statistics.mean(r["n_params"] for r in arm_rows)
        print(
            f"  {arm:>12}  task_B mean={acc_b_mean:.3f}  "
            f"forgetting mean={forget_mean:+.3f}  params mean={params_mean:.0f}"
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            r2 = {**r, "n_nodes_per_layer": "|".join(str(n) for n in r["n_nodes_per_layer"])}
            writer.writerow(r2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
