"""Trioron 2.0 — regime map across EWC strength.

Follow-up to bench_2_0_delta. The n=2 insert_layer arm showed best
Task B mean at high variance and worst Task A retention. Hypothesis:
each arm's value depends on the EWC-pressure regime —

  β=0   no retention pressure; pure capacity test; best acquirer wins
  β=5   moderate; standard continual setup; balanced arm wins
  β=25  strong retention pressure; pure preservation test; best
        retainer wins

Five arms × three β × five seeds = 75 cells. ~5 min.
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
from trioron.growth_direction import (
    features_at_growth_point,
    from_per_class_scatter,
)

_EwcZeroWarning._warned = True


N_SAMPLES = 400
INPUT_DIM = 6
N_CLASSES = 2
N_STEPS_TASK = 600
LR = 5e-3
SEEDS = (0, 1, 2, 3, 4)
BETAS = (0.0, 5.0, 25.0)
ARMS = (
    "baseline",
    "grow_node",
    "long_range",
    "insert_identity",
    "insert_n2_identity",
)
TASK_A_DIMS = (0, 1)
TASK_B_DIMS = (3, 4)


def gen_xor_data(seed: int, dims: tuple, n_samples: int = N_SAMPLES):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, INPUT_DIM, generator=g)
    y = (x[:, dims[0]] * x[:, dims[1]] > 0).long()
    return x, y


def train_phase(net, x, y, n_steps, ewc_beta=0.0):
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
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


def build_net():
    return TrioronNetwork(
        [(INPUT_DIM, 4, "relu"), (4, 3, "relu"), (3, N_CLASSES, "linear")]
    )


def apply_intervention(net, arm, x_b, y_b):
    if arm == "baseline":
        return
    if arm == "grow_node":
        net.grow_layer(layer_idx=1, init_vec=None, task_idx=1)
        return
    if arm == "long_range":
        with torch.no_grad():
            l0_out = net.layers[0](x_b)
            src_node = int(l0_out.abs().mean(dim=0).argmax().item())
        net.layers[2].grow_input(init_col=None, source=(0, src_node))
        return
    if arm == "insert_identity":
        net.insert_layer(
            between=(0, 1), activation="relu", init_mode="identity",
        )
        return
    if arm == "insert_n2_identity":
        net.insert_layer(
            between=(0, 1), n_nodes=2, activation="relu",
            init_mode="identity",
        )
        return
    raise ValueError(arm)


def run_cell(seed, arm, beta):
    torch.manual_seed(seed)
    net = build_net()
    x_a, y_a = gen_xor_data(seed, TASK_A_DIMS)
    train_phase(net, x_a, y_a, n_steps=N_STEPS_TASK, ewc_beta=0.0)
    consolidate(net, x_a, y_a)
    acc_a_post = accuracy(net, *gen_xor_data(seed + 1000, TASK_A_DIMS))

    x_b, y_b = gen_xor_data(seed + 1, TASK_B_DIMS)
    apply_intervention(net, arm, x_b, y_b)

    train_phase(net, x_b, y_b, n_steps=N_STEPS_TASK, ewc_beta=beta)
    acc_a_final = accuracy(net, *gen_xor_data(seed + 1000, TASK_A_DIMS))
    acc_b_final = accuracy(net, *gen_xor_data(seed + 1001, TASK_B_DIMS))
    return {
        "seed": seed, "arm": arm, "beta": beta,
        "acc_a_post": acc_a_post,
        "acc_a_final": acc_a_final,
        "acc_b": acc_b_final,
        "forgetting": acc_a_post - acc_a_final,
        "avg_final": (acc_a_final + acc_b_final) / 2.0,
        "n_params": net.n_parameters(),
    }


def main():
    out_path = Path(__file__).resolve().parent.parent / "outputs" / "bench_2_0_regime.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for beta in BETAS:
        print(f"\n=== β = {beta} ===")
        for arm in ARMS:
            arm_rows = []
            for seed in SEEDS:
                r = run_cell(seed, arm, beta)
                arm_rows.append(r)
                rows.append(r)
            mean_a = statistics.mean(r["acc_a_final"] for r in arm_rows)
            mean_b = statistics.mean(r["acc_b"] for r in arm_rows)
            mean_avg = statistics.mean(r["avg_final"] for r in arm_rows)
            mean_forget = statistics.mean(r["forgetting"] for r in arm_rows)
            std_avg = statistics.stdev(r["avg_final"] for r in arm_rows)
            print(
                f"  {arm:>22}  A={mean_a:.3f}  B={mean_b:.3f}  "
                f"forget={mean_forget:+.3f}  avg={mean_avg:.3f}±{std_avg:.2f}  "
                f"params={arm_rows[0]['n_params']}"
            )
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")

    print("\n=== regime map: avg_final mean by (arm × β) ===")
    print(f"  {'arm':>22}  " + "  ".join(f"β={b:>4.1f}" for b in BETAS) + "    best β")
    for arm in ARMS:
        by_beta = {}
        for beta in BETAS:
            cells = [r["avg_final"] for r in rows if r["arm"] == arm and r["beta"] == beta]
            by_beta[beta] = statistics.mean(cells)
        best_b = max(by_beta, key=lambda b: by_beta[b])
        line = f"  {arm:>22}  " + "  ".join(f"{by_beta[b]:.3f}" for b in BETAS)
        line += f"    β={best_b}"
        print(line)

    print("\n=== winner by β ===")
    for beta in BETAS:
        scores = {arm: statistics.mean(r["avg_final"] for r in rows if r["arm"] == arm and r["beta"] == beta) for arm in ARMS}
        winner = max(scores, key=lambda a: scores[a])
        print(f"  β={beta:>4.1f}:  {winner} (avg_final={scores[winner]:.3f})")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
