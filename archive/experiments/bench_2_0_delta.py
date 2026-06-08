"""Trioron 2.0 — sharper delta bench against 1.0 capacity-growth controls.

Sharper version of bench_2_0_smoke. Designed so the baseline cannot
trivially saturate; the 2.0 primitives are compared against a 1.0
`grow_node` control under the same capacity-stress regime.

Hypothesis: when a continual-learning task pair has strongly
conflicting feature requirements AND the network is capacity-tight
AND EWC pulls hard on Task A, both `long_range` and `insert_layer`
should help Task B acquisition (and possibly Task A retention)
beyond what `grow_node` alone provides at comparable training cost.

Setup
-----
Task A: y = sign(x[0] * x[1]) on 6-d input → XOR-on-2-dims.
        Non-linearly separable; requires hidden nonlinearity.
Task B: y = sign(x[3] * x[4]) — same structure on different dims.

Network: 6 → 4 → 3 → 2. Tight middle layer (only 3 hidden units)
forces capacity competition between tasks.

Procedure per (seed, arm):
  1. Train Task A for N_STEPS (no EWC).
  2. Consolidate: estimate Fisher, update lambda, anchor weights.
  3. Apply arm-specific structural intervention.
  4. Train Task B for N_STEPS with EWC β=10 pulling toward Task A.
  5. Report:
     - acc_a_post   — Task A held-out after Task A training only.
     - acc_a_final  — Task A held-out after Task B training (retention).
     - acc_b        — Task B held-out after Task B training.
     - forgetting   — acc_a_post - acc_a_final.

Arms
----
  baseline       — no intervention (1.0 baseline)
  grow_node      — add 1 node to L1 (the bottleneck) — 1.0 capability
  long_range     — grow_input(source=(0, k)) at L2 head, k greedy
                    from L0 activity on Task B input
  insert_layer   — insert relu layer between L0 and L1 with
                    growth_direction init (top-K right singular
                    vectors of per-class scatter on Task B features)

5 seeds, ~2 min total.
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
EWC_BETA = 10.0
SEEDS = (0, 1, 2, 3, 4)

TASK_A_DIMS = (0, 1)
TASK_B_DIMS = (3, 4)


def gen_xor_data(seed: int, dims: tuple, n_samples: int = N_SAMPLES):
    """XOR-on-two-dims binary task. y = (x[dims[0]] * x[dims[1]] > 0).
    Non-linearly separable — needs hidden nonlinearity to learn."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, INPUT_DIM, generator=g)
    prod = x[:, dims[0]] * x[:, dims[1]]
    y = (prod > 0).long()
    return x, y


def train_phase(net, x, y, n_steps: int, ewc_beta: float = 0.0):
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
    """6 → 4 → 3 → 2. Tight middle layer (only 3 hidden) forces
    capacity competition between Tasks A and B."""
    return TrioronNetwork(
        [(INPUT_DIM, 4, "relu"), (4, 3, "relu"), (3, N_CLASSES, "linear")]
    )


def apply_intervention(net, arm: str, x_b: torch.Tensor, y_b: torch.Tensor):
    """Apply the arm-specific structural mutation between Tasks A and B."""
    if arm == "baseline":
        return
    if arm == "grow_node":
        # 1.0 control: add 1 node to L1 (the bottleneck). grow_layer
        # also extends L2's fan_in by 1. init_vec=None → Kaiming.
        net.grow_layer(layer_idx=1, init_vec=None, task_idx=1)
        return
    if arm == "long_range":
        # 2.0 axis 1+2: grow_input at the head (L2) reading from L0,
        # source node picked greedily as L0's most-active node on
        # Task B's input. init_col=None → zeros → the new edge
        # contributes 0 initially; backprop trains it under no EWC
        # stiffness (fresh column, fisher_W column = 0).
        with torch.no_grad():
            l0_out = net.layers[0](x_b)
            src_node = int(l0_out.abs().mean(dim=0).argmax().item())
        net.layers[2].grow_input(init_col=None, source=(0, src_node))
        return
    if arm == "insert_layer_growth":
        # 2.0 axis 3: insert a new layer between L0 and L1 with
        # growth_direction init. Width matches L0 (v1 constraint).
        # Init vectors = top-K right singular vectors of per-class
        # scatter at L1's input (= L0's output), computed on Task B
        # data — biases the new layer to discriminate Task B classes
        # from the moment of insertion.
        features = features_at_growth_point(net, x_b, dest_layer_idx=1)
        k = net.layers[0].n_nodes
        init_vecs = from_per_class_scatter(features, y_b, k=k)
        net.insert_layer(
            between=(0, 1), activation="relu",
            init_mode="growth_direction", init_vecs=init_vecs,
        )
        return
    if arm == "insert_layer_identity":
        # 2.0 axis 3 with identity init: post-insertion forward is
        # byte-identical to pre-insertion on the positive orthant
        # (relu identity preserves x for x>=0, clips x<0). Task A
        # retention is preserved at insertion; the new layer then
        # specializes during Task B training.
        net.insert_layer(
            between=(0, 1), activation="relu", init_mode="identity",
        )
        return
    if arm == "insert_layer_n2_identity":
        # 2.0 axis 3 with n_nodes=2 (narrower than prev_layer's 4)
        # and identity init. The relaxed insert_layer shrinks
        # next_layer.fan_in to 2 by pruning the two lowest-Fisher
        # columns. Total network params ≈ 55 (vs 58 grow_node),
        # closest param-match to the 1.0 control. Identity init
        # only partially preserves Task A — the dropped L1 columns
        # are gone.
        net.insert_layer(
            between=(0, 1), n_nodes=2, activation="relu",
            init_mode="identity",
        )
        return
    if arm == "insert_layer_n1_identity":
        # 2.0 axis 3 with n_nodes=1 (severe bottleneck). next_layer
        # shrinks to fan_in=1, dropping 3 of 4 trained Task-A columns.
        # Total network params ≈ 47 (less than baseline). Expected
        # to collapse Task A retention.
        net.insert_layer(
            between=(0, 1), n_nodes=1, activation="relu",
            init_mode="identity",
        )
        return
    raise ValueError(f"unknown arm {arm}")


def run_seed(seed: int, arm: str) -> dict:
    torch.manual_seed(seed)
    net = build_net()

    # ---- Task A ----
    x_a, y_a = gen_xor_data(seed, TASK_A_DIMS)
    train_phase(net, x_a, y_a, n_steps=N_STEPS_TASK, ewc_beta=0.0)
    consolidate(net, x_a, y_a)
    acc_a_post = accuracy(net, *gen_xor_data(seed + 1000, TASK_A_DIMS))

    # ---- Intervention ----
    x_b, y_b = gen_xor_data(seed + 1, TASK_B_DIMS)
    apply_intervention(net, arm, x_b, y_b)

    # ---- Task B ----
    train_phase(net, x_b, y_b, n_steps=N_STEPS_TASK, ewc_beta=EWC_BETA)
    acc_a_final = accuracy(net, *gen_xor_data(seed + 1000, TASK_A_DIMS))
    acc_b_final = accuracy(net, *gen_xor_data(seed + 1001, TASK_B_DIMS))

    return {
        "seed": seed,
        "arm": arm,
        "acc_a_post": acc_a_post,
        "acc_a_final": acc_a_final,
        "acc_b": acc_b_final,
        "forgetting": acc_a_post - acc_a_final,
        "avg_final": (acc_a_final + acc_b_final) / 2.0,
        "n_params": net.n_parameters(),
        "arch": tuple(net.n_nodes_per_layer()),
    }


def main():
    out_path = Path(__file__).resolve().parent.parent / "outputs" / "bench_2_0_delta.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arms = ("baseline", "grow_node", "long_range",
            "insert_layer_growth", "insert_layer_identity",
            "insert_layer_n2_identity", "insert_layer_n1_identity")
    rows: list[dict] = []

    t0 = time.time()
    for arm in arms:
        for seed in SEEDS:
            r = run_seed(seed, arm)
            rows.append(r)
            print(
                f"  {arm:>12} seed={seed}  "
                f"A_post={r['acc_a_post']:.3f}  "
                f"A_final={r['acc_a_final']:.3f}  "
                f"B={r['acc_b']:.3f}  "
                f"forget={r['forgetting']:+.3f}  "
                f"avg_final={r['avg_final']:.3f}  "
                f"params={r['n_params']}"
            )
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")

    print("\n--- per-arm summary (mean ± std over seeds, n={}) ---".format(len(SEEDS)))
    print(f"  {'arm':>12}  {'A_post':>10}  {'A_final':>11}  {'B':>10}  "
          f"{'forget':>10}  {'avg_final':>11}  {'params':>6}")
    summary_rows = []
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        ms = {}
        for k in ("acc_a_post", "acc_a_final", "acc_b", "forgetting", "avg_final"):
            vals = [r[k] for r in arm_rows]
            ms[k] = (statistics.mean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0)
        params = int(statistics.mean(r["n_params"] for r in arm_rows))
        summary_rows.append({"arm": arm, **{k: ms[k] for k in ms}, "params": params})
        print(
            f"  {arm:>12}  "
            f"{ms['acc_a_post'][0]:.3f}±{ms['acc_a_post'][1]:.2f}  "
            f"{ms['acc_a_final'][0]:.3f}±{ms['acc_a_final'][1]:.2f}  "
            f"{ms['acc_b'][0]:.3f}±{ms['acc_b'][1]:.2f}  "
            f"{ms['forgetting'][0]:+.3f}±{ms['forgetting'][1]:.2f}  "
            f"{ms['avg_final'][0]:.3f}±{ms['avg_final'][1]:.2f}  "
            f"{params:>6}"
        )

    # Compare 2.0 arms against the closest 1.0 control (grow_node).
    print("\n--- 2.0 vs grow_node (Δ mean) ---")
    grow_node_means = {k: statistics.mean(r[k] for r in rows if r["arm"] == "grow_node")
                       for k in ("acc_b", "forgetting", "avg_final")}
    for arm in ("long_range", "insert_layer_growth", "insert_layer_identity",
                "insert_layer_n2_identity", "insert_layer_n1_identity"):
        arm_means = {k: statistics.mean(r[k] for r in rows if r["arm"] == arm)
                     for k in ("acc_b", "forgetting", "avg_final")}
        print(
            f"  {arm:>22}:  "
            f"Δacc_B={arm_means['acc_b'] - grow_node_means['acc_b']:+.3f}  "
            f"Δforget={arm_means['forgetting'] - grow_node_means['forgetting']:+.3f}  "
            f"Δavg_final={arm_means['avg_final'] - grow_node_means['avg_final']:+.3f}"
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "arm", "acc_a_post", "acc_a_final",
                           "acc_b", "forgetting", "avg_final", "n_params", "arch"],
        )
        writer.writeheader()
        for r in rows:
            r2 = {**r, "arch": "|".join(str(n) for n in r["arch"])}
            writer.writerow(r2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
