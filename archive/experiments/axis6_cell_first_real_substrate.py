"""Trioron 2.0 Axis 6 — real-substrate test, three arms.

Two-layer net: TrioronLayer(2 → H, relu) → TrioronLayer(H → 5, linear),
multi-head (one output cell per task). H starts at AXIS6_H_INIT.
Curriculum = linear → XOR → rings → diamond → stripe.

Arms:
    BASELINE_frozen — Axis 6 off, no growth.
    AXIS6_emerge    — Axis 6 on: field-conditional grow_node fires when
                      a cell's internal_stress trips epi_A above floor.
                      Plasticity only; no anchoring.
    AXIS6_credit    — Axis 6 on PLUS per-cell-row credit-based freezing.
                      At end of each task, any cell whose EMA
                      internal_stress > AXIS6_CREDIT_THR is granted
                      credit on that task. Cells with credit on any
                      prior task have their L0 input-row (W[i,:]) and
                      bias (b[i]) gradients zeroed during subsequent
                      tasks. Freshly-spawned cells start with no credit
                      → fully plastic.

The two questions:
  1. Does Axis 6 spawning fire on demand? (Arm 2 vs 1)
  2. Does cell-ownership-tied freezing rescue retention without
     killing plasticity? (Arm 3 vs 2)
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F

from trioron.network import TrioronNetwork


def make_curriculum(n: int = 300, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * 2.0 - 1.0
    tasks = [
        ("linear", ((X[:, 0] + X[:, 1]) > 0).float() * 2.0 - 1.0),
        ("XOR",    ((X[:, 0] > 0) ^ (X[:, 1] > 0)).float() * 2.0 - 1.0),
        ("rings",  ((X[:, 0] ** 2 + X[:, 1] ** 2) > 0.5).float() * 2.0 - 1.0),
        ("diamond", ((X[:, 0].abs() + X[:, 1].abs()) > 0.7).float() * 2.0 - 1.0),
        ("stripe",  (torch.sin(math.pi * X[:, 0]) > 0).float() * 2.0 - 1.0),
    ]
    return X, tasks


def fresh_net(H: int, n_outputs: int = 5) -> TrioronNetwork:
    """Multi-head topology: one output cell per task, gradient-path-
    isolated. Mirrors the toy cl5 PoC's 5-output layout so the cell-first
    CL property can be tested on a real substrate."""
    net = TrioronNetwork([(2, H, "relu"), (H, n_outputs, "linear")])
    L0 = net.layers[0]
    with torch.no_grad():
        for i in range(L0.n_nodes):
            L0.cell_position[i] = torch.tensor([float(i), 0.0, 0.0])
    return net


def train_axis6(
    net: TrioronNetwork,
    X: torch.Tensor,
    y: torch.Tensor,
    task_id: int,
    n_epochs: int,
    lr: float,
    label: str,
    axis6: bool,
    diff_mode: str,
    diff_k: float,
    diff_floor: float,
    diff_b_thr: float,
    field_sigma: float,
    field_dt: float,
    stress_tolerance: float,
    spawn_cap: int,
    cooldown_steps: int,
    credit_mask: torch.Tensor | None = None,
    track_engagement: bool = False,
) -> Tuple[List[Tuple[int, str]], torch.Tensor | None]:
    """Returns (events, engagement_frac).

    If credit_mask is provided (bool tensor of cells with credit on
    prior tasks), L0.W and L0.b gradients on those cells' rows are
    zeroed before each opt.step. Mask is padded with False to cover
    any cells spawned during the current task.

    If track_engagement is True, returns a per-cell engagement fraction
    (mean over epochs of per-input activation rate of y_i > 0 for ReLU
    layers). Cells that spawn mid-task only accumulate engagement from
    their spawn point forward — engagement is divided by the number of
    epochs the cell existed, so the fraction is comparable across cells.
    """
    L0 = net.layers[0]
    if axis6:
        L0.enable_axis6_field(field_sigma=field_sigma)
        L0.reset_epi_field()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    events: List[Tuple[int, str]] = []
    spawns_this_task = 0
    last_spawn_epoch = -10**9
    engagement_sum: torch.Tensor | None = (
        torch.zeros(L0.n_nodes) if track_engagement else None
    )
    engagement_count: torch.Tensor | None = (
        torch.zeros(L0.n_nodes) if track_engagement else None
    )
    for epoch in range(n_epochs):
        opt.zero_grad()
        outs = net(X)
        pred = outs[:, task_id]
        loss = F.mse_loss(pred, y)
        loss.backward()
        if track_engagement:
            with torch.no_grad():
                last_y = L0._last_y
                if last_y is not None:
                    if L0.activation == "relu":
                        active = (last_y > 0).float()
                    else:
                        active = (last_y.abs() > 0.05).float()
                    per_cell_rate = active.mean(dim=0).cpu()
                    n_cells = per_cell_rate.numel()
                    if engagement_sum is None or engagement_sum.numel() < n_cells:
                        pad_n = n_cells - (
                            0 if engagement_sum is None else engagement_sum.numel()
                        )
                        zeros = torch.zeros(pad_n)
                        engagement_sum = (
                            zeros if engagement_sum is None
                            else torch.cat([engagement_sum, zeros])
                        )
                        engagement_count = (
                            torch.zeros_like(engagement_sum)
                            if engagement_count is None
                            else torch.cat([engagement_count, zeros])
                        )
                    engagement_sum[:n_cells] += per_cell_rate
                    engagement_count[:n_cells] += 1.0
        # Stress + field updates must be inside the backward window
        # because update_internal_stress reads cached _last_y +
        # _last_upstream.
        L0.update_internal_stress()
        if credit_mask is not None and credit_mask.any():
            with torch.no_grad():
                cm = credit_mask
                n_cells = L0.W.shape[0]
                if cm.numel() < n_cells:
                    pad = torch.zeros(
                        n_cells - cm.numel(),
                        dtype=torch.bool,
                        device=cm.device,
                    )
                    cm = torch.cat([cm, pad])
                if L0.W.grad is not None:
                    L0.W.grad[cm] = 0.0
                if L0.b.grad is not None:
                    L0.b.grad[cm] = 0.0
        opt.step()
        if axis6:
            L0.update_epi_field(dt=field_dt, stress_tolerance=stress_tolerance)
            if (spawns_this_task < spawn_cap
                    and epoch - last_spawn_epoch >= cooldown_steps):
                cand = L0.field_conditional_growth_candidate(
                    mode=diff_mode,
                    k=diff_k,
                    stress_floor=diff_floor,
                    b_threshold=diff_b_thr,
                )
                if cand is not None:
                    new_idx = net.axis6_spawn(0, cand, position_jitter=0.3)
                    spawns_this_task += 1
                    last_spawn_epoch = epoch
                    events.append(
                        (epoch,
                         f"AXIS6 spawn: candidate cell {cand} → new cell {new_idx} "
                         f"@ {L0.cell_position[new_idx].tolist()}")
                    )
                    # Optimizer must rebuild after structural change.
                    opt = torch.optim.Adam(net.parameters(), lr=lr)
        if epoch % 100 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                acc = ((pred.sign() == y.sign()).float().mean().item())
            print(
                f"  [{label}] epoch {epoch:>3}: "
                f"loss={loss.item():.4f}  acc={acc:.3f}  "
                f"H={L0.n_nodes}  "
                f"epi_A_max={L0.epi_A.max().item():.3f}  "
                f"epi_B_max={L0.epi_B.max().item():.3f}"
            )
    engagement_frac: torch.Tensor | None = None
    if track_engagement and engagement_sum is not None and engagement_count is not None:
        n_cells = L0.n_nodes
        if engagement_sum.numel() < n_cells:
            pad = torch.zeros(n_cells - engagement_sum.numel())
            engagement_sum = torch.cat([engagement_sum, pad])
            engagement_count = torch.cat([engagement_count, torch.zeros_like(pad)])
        engagement_frac = engagement_sum / engagement_count.clamp(min=1.0)
    return events, engagement_frac


def evaluate(net: TrioronNetwork, X: torch.Tensor, y: torch.Tensor, task_id: int) -> float:
    with torch.no_grad():
        outs = net(X)
        pred = outs[:, task_id]
    return (pred.sign() == y.sign()).float().mean().item()


def run_one_arm(
    seed: int,
    arm_label: str,
    axis6: bool,
    credit_enabled: bool,
    X: torch.Tensor,
    tasks: List[Tuple[str, torch.Tensor]],
    epochs_per_task: List[int],
    lr: float,
    H_init: int,
    hp: dict,
    verbose: bool = True,
) -> dict:
    """Run one (seed, arm) trial. Returns dict of summary stats."""
    if verbose:
        print(f"\n========== SEED {seed}  ARM: {arm_label} ==========")
    torch.manual_seed(seed)
    net = fresh_net(H=H_init)
    all_events: List[Tuple[str, int, str]] = []
    retention: List[List[float]] = []
    cell_credit: torch.Tensor | None = (
        torch.zeros(net.layers[0].n_nodes, dtype=torch.bool)
        if credit_enabled else None
    )
    for task_id, ((name, y), n_ep) in enumerate(zip(tasks, epochs_per_task)):
        if verbose:
            print(f"\n[Phase {task_id+1}: training task '{name}']")
            if credit_enabled and cell_credit is not None:
                n_frozen = int(cell_credit.sum().item())
                print(f"  [credit] entering with {n_frozen}/{cell_credit.numel()} frozen")
        events, engagement_frac = train_axis6(
            net, X, y, task_id=task_id,
            n_epochs=n_ep, lr=lr, label=name,
            axis6=axis6,
            diff_mode=hp["diff_mode"],
            diff_k=hp["diff_k"], diff_floor=hp["diff_floor"],
            diff_b_thr=hp["diff_b_thr"],
            field_sigma=hp["field_sigma"], field_dt=hp["field_dt"],
            stress_tolerance=hp["stress_tolerance"],
            spawn_cap=hp["spawn_cap"], cooldown_steps=hp["cooldown_steps"],
            credit_mask=cell_credit,
            track_engagement=credit_enabled,
        )
        if verbose:
            for ev in events:
                print(f"    >> {ev[1]}  (epoch {ev[0]})")
        for ev in events:
            all_events.append((name, ev[0], ev[1]))
        row = [evaluate(net, X, tasks[k][1], task_id=k) for k in range(len(tasks))]
        retention.append(row)
        if verbose:
            print(f"  >> {name} final acc = {row[task_id]:.3f}  "
                  f"H={net.layers[0].n_nodes}")
        if credit_enabled and engagement_frac is not None:
            with torch.no_grad():
                L0 = net.layers[0]
                if cell_credit is None:
                    cell_credit = torch.zeros(L0.n_nodes, dtype=torch.bool)
                elif cell_credit.numel() < L0.n_nodes:
                    pad = torch.zeros(
                        L0.n_nodes - cell_credit.numel(), dtype=torch.bool,
                    )
                    cell_credit = torch.cat([cell_credit, pad])
                if engagement_frac.numel() < L0.n_nodes:
                    pad_e = torch.zeros(L0.n_nodes - engagement_frac.numel())
                    engagement_frac = torch.cat([engagement_frac, pad_e])
                earned = engagement_frac > hp["credit_thr"]
                cell_credit = cell_credit | earned
                if verbose:
                    print(f"  [credit] +{int((earned & ~cell_credit).sum().item())} new; "
                          f"total frozen = {int(cell_credit.sum().item())}/{cell_credit.numel()}")

    first_acc = [retention[k][k] for k in range(len(tasks))]
    final_acc = retention[-1]
    drift = [first_acc[k] - final_acc[k] for k in range(len(tasks))]
    max_drift = max(abs(d) for d in drift)
    return {
        "seed": seed,
        "arm": arm_label,
        "retention": retention,
        "first_acc": first_acc,
        "final_acc": final_acc,
        "drift": drift,
        "max_drift": max_drift,
        "n_hidden": net.layers[0].n_nodes,
        "n_spawns": len(all_events),
        "n_frozen": int(cell_credit.sum().item()) if cell_credit is not None else 0,
        "sum_final_acc": sum(final_acc),
    }


def mean_std(xs: List[float]) -> Tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return m, v ** 0.5


def main() -> int:
    diff_mode = os.environ.get("AXIS6_MODE", "absolute")
    diff_k = float(os.environ.get("AXIS6_K", "1.0"))
    diff_floor = float(os.environ.get("AXIS6_FLOOR", "0.005"))
    diff_b_thr = float(os.environ.get("AXIS6_B_THR", "1.0"))
    field_sigma = float(os.environ.get("AXIS6_SIGMA", "1.5"))
    field_dt = float(os.environ.get("AXIS6_DT", "0.1"))
    stress_tolerance = float(os.environ.get("AXIS6_TOL", "0.0"))
    spawn_cap = int(os.environ.get("AXIS6_SPAWN_CAP", "3"))
    cooldown_steps = int(os.environ.get("AXIS6_COOLDOWN", "50"))
    H_init = int(os.environ.get("AXIS6_H_INIT", "4"))
    # Credit threshold is on engagement-fraction (mean activation rate
    # over a task's inputs). 0.1 = "cell fires on at least 10% of inputs".
    credit_thr = float(os.environ.get("AXIS6_CREDIT_THR", "0.1"))
    n_seeds = int(os.environ.get("AXIS6_N_SEEDS", "3"))
    quiet = os.environ.get("AXIS6_QUIET", "1") != "0"
    epochs_per_task = [200, 400, 400, 400, 400]
    lr = 0.05

    hp = dict(
        diff_mode=diff_mode, diff_k=diff_k, diff_floor=diff_floor,
        diff_b_thr=diff_b_thr, field_sigma=field_sigma, field_dt=field_dt,
        stress_tolerance=stress_tolerance, spawn_cap=spawn_cap,
        cooldown_steps=cooldown_steps, credit_thr=credit_thr,
    )

    X, tasks = make_curriculum(n=300, seed=0)
    print("=" * 88)
    print(f"Trioron 2.0 Axis 6 — real-substrate, n={n_seeds} seeds")
    print(f"H_init={H_init}  mode={diff_mode}  k={diff_k}  floor={diff_floor}  "
          f"b_thr={diff_b_thr}  sigma={field_sigma}  dt={field_dt}  "
          f"tol={stress_tolerance}  spawn_cap={spawn_cap}  cooldown={cooldown_steps}  "
          f"credit_thr={credit_thr}")
    print("=" * 88)

    arms = [
        ("BASELINE_frozen", False, False),
        ("AXIS6_emerge", True, False),
        ("AXIS6_credit", True, True),
    ]
    results: dict = {a[0]: [] for a in arms}
    for seed in range(n_seeds):
        for arm_label, axis6, credit_enabled in arms:
            r = run_one_arm(
                seed=seed,
                arm_label=arm_label,
                axis6=axis6,
                credit_enabled=credit_enabled,
                X=X, tasks=tasks,
                epochs_per_task=epochs_per_task, lr=lr, H_init=H_init,
                hp=hp, verbose=not quiet,
            )
            results[arm_label].append(r)
            print(
                f"[seed {seed}  arm {arm_label:<16}]  "
                f"max|drift|={r['max_drift']:.3f}  "
                f"sum_final={r['sum_final_acc']:.3f}  "
                f"H={r['n_hidden']}  spawns={r['n_spawns']}  "
                f"frozen={r['n_frozen']}"
            )

    print("\n" + "=" * 88)
    print(f"AGGREGATE (n={n_seeds} seeds, mean ± std)")
    print("=" * 88)
    task_names = [t[0] for t in tasks]
    header = (
        f"  {'arm':<16}  {'max|drift|':>14}  {'sum_final':>14}  "
        f"{'H_final':>10}  {'spawns':>8}  {'frozen':>8}"
    )
    print(header)
    for arm_label, _, _ in arms:
        rs = results[arm_label]
        md_m, md_s = mean_std([r["max_drift"] for r in rs])
        sf_m, sf_s = mean_std([r["sum_final_acc"] for r in rs])
        h_m, h_s = mean_std([float(r["n_hidden"]) for r in rs])
        sp_m, sp_s = mean_std([float(r["n_spawns"]) for r in rs])
        fr_m, fr_s = mean_std([float(r["n_frozen"]) for r in rs])
        print(
            f"  {arm_label:<16}  {md_m:>6.3f} ± {md_s:>5.3f}  "
            f"{sf_m:>6.3f} ± {sf_s:>5.3f}  "
            f"{h_m:>4.1f} ± {h_s:>3.1f}  "
            f"{sp_m:>3.1f} ± {sp_s:>2.1f}  "
            f"{fr_m:>3.1f} ± {fr_s:>2.1f}"
        )
    print(f"\n  per-task final acc (mean ± std across {n_seeds} seeds):")
    print(f"  {'arm':<16}  " + "  ".join(f"{n:>14}" for n in task_names))
    for arm_label, _, _ in arms:
        rs = results[arm_label]
        cells = []
        for k in range(len(tasks)):
            vals = [r["final_acc"][k] for r in rs]
            m, s = mean_std(vals)
            cells.append(f"{m:>6.3f} ± {s:>5.3f}")
        print(f"  {arm_label:<16}  " + "  ".join(cells))

    # CSV for later analysis
    csv_path = os.environ.get(
        "AXIS6_CSV", "outputs/axis6_real_substrate_credit_n3.csv"
    )
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w") as f:
            f.write("seed,arm,task,first_acc,final_acc,drift,max_drift,sum_final,n_hidden,n_spawns,n_frozen\n")
            for arm_label, _, _ in arms:
                for r in results[arm_label]:
                    for k, name in enumerate(task_names):
                        f.write(
                            f"{r['seed']},{arm_label},{name},"
                            f"{r['first_acc'][k]:.4f},{r['final_acc'][k]:.4f},"
                            f"{r['drift'][k]:.4f},{r['max_drift']:.4f},"
                            f"{r['sum_final_acc']:.4f},{r['n_hidden']},"
                            f"{r['n_spawns']},{r['n_frozen']}\n"
                        )
        print(f"\n  CSV → {csv_path}")
    except Exception as e:
        print(f"  CSV write failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
