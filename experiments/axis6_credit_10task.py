"""Trioron 2.0 Axis 6 + credit-based freezing — 10-task scaling stress test.

Doubles the 5-task curriculum from axis6_cell_first_real_substrate.py to 10
tasks on the same 2D substrate. Same 3 arms (BASELINE_frozen, AXIS6_emerge,
AXIS6_credit) at n=3 seeds. Tests whether the credit mechanism scales:

  1. Does credit-budget saturate the substrate (everything frozen, no
     plasticity left for new tasks)?
  2. Does spawn-cap × n_tasks keep growing the substrate without runaway?
  3. Does engagement_frac stay informative as cells overlap in 2D
     feature space (10 tasks competing for the same 2D area)?

Curriculum (each task is a binary classification on 2D points in [-1,1]^2):
   1. linear        sign(x0 + x1)
   2. XOR           (x0>0) ^ (x1>0)
   3. rings         x0² + x1² > 0.5
   4. diamond       |x0| + |x1| > 0.7
   5. stripe        sin(πx0) > 0
   6. anti_linear   sign(x0 - x1)        (orthogonal axis to #1)
   7. half_plane    x0 > 0
   8. checker       sign(sin(2π x0) · sin(2π x1))
   9. annulus       0.2 < x0² + x1² < 0.6
  10. y_stripe      sin(πx1) > 0          (orthogonal stripe to #5)
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F

from trioron.network import TrioronNetwork


def make_curriculum_10(n: int = 300, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 2, generator=g) * 2.0 - 1.0
    r2 = X[:, 0] ** 2 + X[:, 1] ** 2
    tasks = [
        ("linear",      ((X[:, 0] + X[:, 1]) > 0).float() * 2.0 - 1.0),
        ("XOR",         ((X[:, 0] > 0) ^ (X[:, 1] > 0)).float() * 2.0 - 1.0),
        ("rings",       (r2 > 0.5).float() * 2.0 - 1.0),
        ("diamond",     ((X[:, 0].abs() + X[:, 1].abs()) > 0.7).float() * 2.0 - 1.0),
        ("stripe",      (torch.sin(math.pi * X[:, 0]) > 0).float() * 2.0 - 1.0),
        ("anti_linear", ((X[:, 0] - X[:, 1]) > 0).float() * 2.0 - 1.0),
        ("half_plane",  (X[:, 0] > 0).float() * 2.0 - 1.0),
        ("checker",     (torch.sin(2 * math.pi * X[:, 0])
                         * torch.sin(2 * math.pi * X[:, 1]) > 0).float() * 2.0 - 1.0),
        ("annulus",     ((r2 > 0.2) & (r2 < 0.6)).float() * 2.0 - 1.0),
        ("y_stripe",    (torch.sin(math.pi * X[:, 1]) > 0).float() * 2.0 - 1.0),
    ]
    return X, tasks


def fresh_net(H: int, n_outputs: int = 10) -> TrioronNetwork:
    """One output cell per task; multi-head with gradient-path isolation
    via outs[:, task_id] indexing in train_axis6."""
    net = TrioronNetwork([(2, H, "relu"), (H, n_outputs, "linear")])
    L0 = net.layers[0]
    with torch.no_grad():
        for i in range(L0.n_nodes):
            L0.cell_position[i] = torch.tensor([float(i), 0.0, 0.0])
    return net


def train_one_task(
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
        L0.update_internal_stress()
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
        if credit_mask is not None and credit_mask.any():
            with torch.no_grad():
                cm = credit_mask
                n_cells = L0.W.shape[0]
                if cm.numel() < n_cells:
                    pad = torch.zeros(
                        n_cells - cm.numel(), dtype=torch.bool, device=cm.device,
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
                    mode=diff_mode, k=diff_k,
                    stress_floor=diff_floor, b_threshold=diff_b_thr,
                )
                if cand is not None:
                    new_idx = net.axis6_spawn(0, cand, position_jitter=0.3)
                    spawns_this_task += 1
                    last_spawn_epoch = epoch
                    events.append(
                        (epoch,
                         f"AXIS6 spawn: candidate cell {cand} → new cell {new_idx}")
                    )
                    opt = torch.optim.Adam(net.parameters(), lr=lr)
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
    verbose: bool = False,
) -> dict:
    torch.manual_seed(seed)
    net = fresh_net(H=H_init, n_outputs=len(tasks))
    all_events: List[Tuple[str, int, str]] = []
    retention: List[List[float]] = []
    cell_credit: torch.Tensor | None = (
        torch.zeros(net.layers[0].n_nodes, dtype=torch.bool)
        if credit_enabled else None
    )
    for task_id, ((name, y), n_ep) in enumerate(zip(tasks, epochs_per_task)):
        events, engagement_frac = train_one_task(
            net, X, y, task_id=task_id,
            n_epochs=n_ep, lr=lr, label=name,
            axis6=axis6,
            diff_mode=hp["diff_mode"], diff_k=hp["diff_k"],
            diff_floor=hp["diff_floor"], diff_b_thr=hp["diff_b_thr"],
            field_sigma=hp["field_sigma"], field_dt=hp["field_dt"],
            stress_tolerance=hp["stress_tolerance"],
            spawn_cap=hp["spawn_cap"], cooldown_steps=hp["cooldown_steps"],
            credit_mask=cell_credit,
            track_engagement=credit_enabled,
        )
        for ev in events:
            all_events.append((name, ev[0], ev[1]))
        row = [evaluate(net, X, tasks[k][1], task_id=k) for k in range(len(tasks))]
        retention.append(row)
        if verbose:
            print(f"    [{name}] H={net.layers[0].n_nodes} acc={row[task_id]:.3f}")
        if credit_enabled and engagement_frac is not None:
            with torch.no_grad():
                L0 = net.layers[0]
                if cell_credit is None:
                    cell_credit = torch.zeros(L0.n_nodes, dtype=torch.bool)
                elif cell_credit.numel() < L0.n_nodes:
                    pad = torch.zeros(L0.n_nodes - cell_credit.numel(), dtype=torch.bool)
                    cell_credit = torch.cat([cell_credit, pad])
                if engagement_frac.numel() < L0.n_nodes:
                    pad_e = torch.zeros(L0.n_nodes - engagement_frac.numel())
                    engagement_frac = torch.cat([engagement_frac, pad_e])
                earned = engagement_frac > hp["credit_thr"]
                cell_credit = cell_credit | earned

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
        "mean_final_acc": sum(final_acc) / len(final_acc),
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
    credit_thr = float(os.environ.get("AXIS6_CREDIT_THR", "0.1"))
    n_seeds = int(os.environ.get("AXIS6_N_SEEDS", "3"))
    # epochs: same schedule for first 5, plus 400 each for the new 5.
    epochs_per_task = [200, 400, 400, 400, 400, 400, 400, 400, 400, 400]
    lr = 0.05

    hp = dict(
        diff_mode=diff_mode, diff_k=diff_k, diff_floor=diff_floor,
        diff_b_thr=diff_b_thr, field_sigma=field_sigma, field_dt=field_dt,
        stress_tolerance=stress_tolerance, spawn_cap=spawn_cap,
        cooldown_steps=cooldown_steps, credit_thr=credit_thr,
    )

    X, tasks = make_curriculum_10(n=300, seed=0)
    print("=" * 88)
    print(f"Trioron 2.0 Axis 6 + credit — 10-task curriculum, n={n_seeds} seeds")
    print(f"H_init={H_init}  mode={diff_mode}  k={diff_k}  floor={diff_floor}  "
          f"b_thr={diff_b_thr}  sigma={field_sigma}  dt={field_dt}  "
          f"spawn_cap={spawn_cap}  cooldown={cooldown_steps}  credit_thr={credit_thr}")
    print(f"tasks: {[t[0] for t in tasks]}")
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
                seed=seed, arm_label=arm_label, axis6=axis6,
                credit_enabled=credit_enabled,
                X=X, tasks=tasks, epochs_per_task=epochs_per_task,
                lr=lr, H_init=H_init, hp=hp, verbose=False,
            )
            results[arm_label].append(r)
            print(
                f"[seed {seed}  arm {arm_label:<16}]  "
                f"max|drift|={r['max_drift']:.3f}  "
                f"sum_final={r['sum_final_acc']:.3f}  "
                f"mean_final={r['mean_final_acc']:.3f}  "
                f"H={r['n_hidden']}  spawns={r['n_spawns']}  "
                f"frozen={r['n_frozen']}"
            )

    print("\n" + "=" * 88)
    print(f"AGGREGATE (n={n_seeds} seeds, mean ± std) — 10-task curriculum")
    print("=" * 88)
    task_names = [t[0] for t in tasks]
    print(
        f"  {'arm':<16}  {'max|drift|':>14}  {'sum_final':>14}  "
        f"{'mean_final':>14}  {'H_final':>10}  {'spawns':>8}  {'frozen':>8}"
    )
    for arm_label, _, _ in arms:
        rs = results[arm_label]
        md_m, md_s = mean_std([r["max_drift"] for r in rs])
        sf_m, sf_s = mean_std([r["sum_final_acc"] for r in rs])
        mf_m, mf_s = mean_std([r["mean_final_acc"] for r in rs])
        h_m, h_s = mean_std([float(r["n_hidden"]) for r in rs])
        sp_m, sp_s = mean_std([float(r["n_spawns"]) for r in rs])
        fr_m, fr_s = mean_std([float(r["n_frozen"]) for r in rs])
        print(
            f"  {arm_label:<16}  {md_m:>6.3f} ± {md_s:>5.3f}  "
            f"{sf_m:>6.3f} ± {sf_s:>5.3f}  "
            f"{mf_m:>6.3f} ± {mf_s:>5.3f}  "
            f"{h_m:>4.1f} ± {h_s:>3.1f}  "
            f"{sp_m:>3.1f} ± {sp_s:>2.1f}  "
            f"{fr_m:>3.1f} ± {fr_s:>2.1f}"
        )
    print(f"\n  per-task final acc (mean ± std across {n_seeds} seeds):")
    for k, name in enumerate(task_names):
        line = f"  {name:<14}  "
        for arm_label, _, _ in arms:
            vals = [r["final_acc"][k] for r in results[arm_label]]
            m, s = mean_std(vals)
            line += f"{arm_label[:8]}:{m:.3f}±{s:.3f}   "
        print(line)

    csv_path = os.environ.get(
        "AXIS6_CSV", "outputs/axis6_credit_10task_n3.csv"
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
