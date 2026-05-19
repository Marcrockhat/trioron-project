"""Trioron 2.0 — single-task depth-required probe.

INCONCLUSIVE RESULT — see "diagnostic" section at the end.

The continual-learning benches (bench_2_0_delta, bench_2_0_regime)
showed no 2.0 arm strictly dominates the 1.0 grow_node control on
that workload. The proposal's original motivation was pneuma's
single-task capacity-fitting plateau on TinyStories — a workload
structurally different from CL. This probe tested the matching
hypothesis on a trioron-scale task:

  Hypothesis (NOT CONFIRMED): when a single task requires depth (not
  just width), insert_layer can escape a plateau that grow_node
  cannot.

Task: 4-bit product XOR.
    y = (x[0] * x[1] * x[2] * x[3] > 0).long()
This is the XOR-of-two-XORs: theoretically, depth-2 networks fit it
with constant width; depth-1 networks need width 2^(n-1) to fit
n-bit parity exactly.

Result at 5 seeds:
    baseline   acc_final=0.537±0.03    params=38   arch=(4, 2)
    widen_L0   acc_final=0.566±0.04    params=47   arch=(5, 2)
    insert_n4  acc_final=0.560±0.03    params=58   arch=(4, 4, 2)
    insert_n6  acc_final=0.561±0.03    params=72   arch=(4, 6, 2)

All capacity arms are statistically tied at ~0.56. The 1.0 widen
control gets +0.029 over baseline; the 2.0 depth arms get +0.023
to +0.024 — slightly LESS than widening, despite using more
parameters.

Diagnostic (param-matched depth-vs-width, 3000 training steps):
    arch=(6, 16, 2)     params=146   acc=0.839    wide shallow
    arch=(6, 8, 8, 2)   params=146   acc=0.669    deep narrow
    arch=(6, 16, 16, 2) params=418   acc=0.863    deep AND wide

At matched param budget, wide-shallow BEATS deep-narrow by 17
points on this task. The theoretical depth requirement for exact
parity representation doesn't translate to gradient-descent
optimization at this scale — width substitutes for depth in
practice. Depth-requiring optimization landscapes likely appear at
larger scale (vision, language, pneuma transformer-FFN) where
compositional structure compounds across many layers.

Honest takeaway: insert_layer's empirical advantage isn't
demonstrable at trioron substrate scale on synthetic compositional
tasks. The primitive remains mathematically sound and integration-
tested; demonstrating its value requires a workload aligned with
the original proposal's premise (pneuma transformer FFN, not
trioron-scale benches).

Procedure:
  1. Train baseline 6→4→2 to plateau (800 steps).
  2. Apply arm-specific intervention.
  3. Train another 800 steps.
  4. Measure held-out accuracy.

Arms:
  baseline       — train more, no intervention
  widen_L0       — grow_layer(0): +1 node on L0 (extends L1 fan_in)
  insert_n4      — insert 4-node layer between L0 and L1 (identity)
  insert_n6      — insert 6-node layer (uses Phase 2 width relaxation)

5 seeds, ~2 min.
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

_EwcZeroWarning._warned = True


N_SAMPLES = 800
N_TEST = 1000
INPUT_DIM = 6
N_CLASSES = 2
TASK_DIMS = (0, 1, 2, 3)
N_STEPS_PRE = 800
N_STEPS_POST = 800
LR = 5e-3
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("baseline", "widen_L0", "insert_n4", "insert_n6")


def gen_parity(seed: int, n_samples: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_samples, INPUT_DIM, generator=g)
    prod = torch.ones(n_samples)
    for d in TASK_DIMS:
        prod = prod * x[:, d]
    y = (prod > 0).long()
    return x, y


def train_phase(net, x, y, n_steps):
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = F.cross_entropy(net(x), y)
        loss.backward()
        opt.step()


def accuracy(net, x, y):
    with torch.no_grad():
        return float((net(x).argmax(dim=1) == y).float().mean().item())


def build_net():
    return TrioronNetwork(
        [(INPUT_DIM, 4, "relu"), (4, N_CLASSES, "linear")]
    )


def apply_intervention(net, arm):
    if arm == "baseline":
        return
    if arm == "widen_L0":
        # grow_layer(0) adds one node to L0 and extends layer 1's fan_in.
        net.grow_layer(layer_idx=0, init_vec=None, task_idx=0)
        return
    if arm == "insert_n4":
        # Insert a 4-node layer between L0 and head, identity init.
        net.insert_layer(
            between=(0, 1), n_nodes=4, activation="relu",
            init_mode="identity",
        )
        return
    if arm == "insert_n6":
        # Insert wider — 6 nodes. Uses Phase 2 width relaxation to grow
        # the head's fan_in from 4 to 6.
        net.insert_layer(
            between=(0, 1), n_nodes=6, activation="relu",
            init_mode="identity",
        )
        return
    raise ValueError(arm)


def run_cell(seed, arm):
    torch.manual_seed(seed)
    net = build_net()
    x_train, y_train = gen_parity(seed, N_SAMPLES)
    x_test, y_test = gen_parity(seed + 1000, N_TEST)

    train_phase(net, x_train, y_train, N_STEPS_PRE)
    acc_pre = accuracy(net, x_test, y_test)

    apply_intervention(net, arm)
    train_phase(net, x_train, y_train, N_STEPS_POST)
    acc_post = accuracy(net, x_test, y_test)
    return {
        "seed": seed, "arm": arm,
        "acc_pre_intervention": acc_pre,
        "acc_final": acc_post,
        "delta": acc_post - acc_pre,
        "n_params": net.n_parameters(),
        "arch": tuple(net.n_nodes_per_layer()),
    }


def main():
    out_path = Path(__file__).resolve().parent.parent / "outputs" / "bench_2_0_single_task.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for arm in ARMS:
        for seed in SEEDS:
            r = run_cell(seed, arm)
            rows.append(r)
            print(
                f"  {arm:>12} seed={seed}  "
                f"pre={r['acc_pre_intervention']:.3f}  "
                f"final={r['acc_final']:.3f}  "
                f"Δ={r['delta']:+.3f}  "
                f"params={r['n_params']}  arch={r['arch']}"
            )
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.1f}s")

    print("\n--- per-arm summary (n={}) ---".format(len(SEEDS)))
    print(f"  {'arm':>12}  {'acc_pre':>10}  {'acc_final':>11}  {'Δ':>10}  {'params':>8}  {'arch':>14}")
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        pre_m = statistics.mean(r["acc_pre_intervention"] for r in arm_rows)
        pre_s = statistics.stdev(r["acc_pre_intervention"] for r in arm_rows)
        final_m = statistics.mean(r["acc_final"] for r in arm_rows)
        final_s = statistics.stdev(r["acc_final"] for r in arm_rows)
        delta_m = statistics.mean(r["delta"] for r in arm_rows)
        delta_s = statistics.stdev(r["delta"] for r in arm_rows)
        params_m = int(statistics.mean(r["n_params"] for r in arm_rows))
        arch = arm_rows[0]["arch"]
        print(
            f"  {arm:>12}  {pre_m:.3f}±{pre_s:.2f}  "
            f"{final_m:.3f}±{final_s:.2f}  "
            f"{delta_m:+.3f}±{delta_s:.2f}  "
            f"{params_m:>8}  {str(arch):>14}"
        )

    print("\n--- Δ vs widen_L0 (the 1.0 control) ---")
    widen_delta = statistics.mean(r["delta"] for r in rows if r["arm"] == "widen_L0")
    widen_final = statistics.mean(r["acc_final"] for r in rows if r["arm"] == "widen_L0")
    for arm in ARMS:
        if arm == "widen_L0":
            continue
        arm_delta = statistics.mean(r["delta"] for r in rows if r["arm"] == arm)
        arm_final = statistics.mean(r["acc_final"] for r in rows if r["arm"] == arm)
        print(
            f"  {arm:>12}:  Δ(post-pre)={arm_delta - widen_delta:+.3f}  "
            f"Δ(final)={arm_final - widen_final:+.3f}"
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["seed", "arm", "acc_pre_intervention",
                           "acc_final", "delta", "n_params", "arch"],
        )
        writer.writeheader()
        for r in rows:
            r2 = {**r, "arch": "|".join(str(n) for n in r["arch"])}
            writer.writerow(r2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
