"""PackNet head-to-head: §11 risk-register comparison.

The §11 risk register flags *"It's basically Cascade Correlation / DEN
with extra steps"* — bench_step8 and bench_harder didn't address this.
Both use a simple fixed-MLP-with-EWC baseline, which is structurally
weaker than dedicated continual-learning algorithms.

PackNet (Mallya & Lazebnik 2018) is the most directly comparable
published baseline: per-task disjoint subnets carved out by magnitude
pruning, with task-aware inference. If grown beats PackNet at the same
parameter budget on bench_harder's curriculum, that's a substantively
stronger claim than the EWC comparison.

Asymmetry to call out: PackNet uses task-ID at inference time (it picks
which mask to apply). Grown and fixed-EWC don't — they use the same
weights for all tasks. This is a structural advantage for PackNet that
the literature standardly accepts; we report it explicitly rather than
hide it.

Same harness and curriculum as bench_harder.py — 16-dim state, 20-task
progressive curriculum (12 single + 8 compound). EWC posture identical
for grown and fixed-EWC. PackNet uses uniform per-task allocation
(1/n_total_tasks of free weights) instead of EWC.
"""
from __future__ import annotations
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron.incubator import contrastive_loss
from trioron.curriculum import (
    ParameterizedContrastiveCurriculum,
    build_progressive_pairs,
)
from trioron.triggers import GrowthTrigger, total_gradient_norm
from trioron.ceilings import CeilingsController
from trioron.pruner import PruningController, utility_capture
from trioron.packnet import PackNetController


# Curriculum + EWC config — identical to bench_harder for apples-to-apples.
STATE_DIM = 16
N_SINGLE = 12
N_COMPOUND = 8
N_PAIRS = N_SINGLE + N_COMPOUND
N_STEPS_PER_TASK = 1500
BATCH = 32
HIDDEN = 16
LATENT_INIT_GROWN = 1
MARGIN = 1.0
LR = 3e-3
SEED = 0

TRIGGER_W = 400
TRIGGER_EPS_LOSS = 0.001
TRIGGER_EPS_RANK = 0.1
TRIGGER_G_MIN = 1e-4
TRIGGER_G_MAX = 10.0

T_STABILIZE = 300
EWC_STAB_BOOST = 5000.0
LAMBDA_FLOOR = 0.1
EWC_INTERTASK = 1000.0

M_MAX_BYTES = 2 * 1024 ** 3
T_DIV_MAX_SECONDS = 60.0

PRUNE_U_THRESHOLD = 1e-3
PRUNE_T = 2000
PRUNE_CLOCK = 500

# PackNet-only width — single point for now. Pick H=16 to match the matched-
# fixed comparison point in bench_harder (612 params).
PACKNET_HIDDEN_SIZES = [12, 16, 20]

EVAL_BATCH = 512
LOG_EVERY = 500
EVAL_SEED = 42


def make_network(state_dim: int, hidden: int, latent: int) -> TrioronNetwork:
    return TrioronNetwork(
        [
            (state_dim, hidden, "relu"),
            (hidden, hidden, "relu"),
            (hidden, latent, "tanh"),
        ]
    )


def make_fixed_eval_batches(pair_names, cur_factory, seed=EVAL_SEED, batch=EVAL_BATCH):
    cur = cur_factory(seed=seed)
    fixed: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name in pair_names:
        fixed[name] = cur.sample_pair(name, batch=batch)
    return fixed


def evaluate_all_pairs(net, eval_batches, pair_names):
    losses: Dict[str, float] = {}
    with torch.no_grad():
        for name in pair_names:
            a, b = eval_batches[name]
            losses[name] = float(contrastive_loss(net(a), net(b), MARGIN).item())
    return losses


# ---------------------------------------------------------------------
# Grown / fixed-EWC path (carried over verbatim from bench_harder)
# ---------------------------------------------------------------------


def estimate_fisher_for_pair(net, train_cur, pair_name, batch=BATCH, n_batches=20):
    def batches():
        for _ in range(n_batches):
            a, b = train_cur.sample_pair(pair_name, batch=batch)
            yield a, b

    def loss_fn(pred_a, b_input):
        h_b = net(b_input)
        return contrastive_loss(pred_a, h_b, MARGIN)

    net.estimate_fisher(batches(), loss_fn, n_batches=n_batches)


def consolidate_task(net, train_cur, pair_name):
    estimate_fisher_for_pair(net, train_cur, pair_name)
    net.update_lambda_all()
    with torch.no_grad():
        for layer in net.layers:
            layer.lam.clamp_(min=LAMBDA_FLOOR)
    net.anchor_all()


def compute_growth_direction(net, train_cur, pair_name, batch=128):
    with torch.no_grad():
        a, b = train_cur.sample_pair(pair_name, batch=batch)
        f_a, f_b = a, b
        for layer in net.layers[:-1]:
            f_a = layer(f_a)
            f_b = layer(f_b)
        D = f_a - f_b
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    v = Vh[0]
    return v / (v.norm() + 1e-12)


def train_one_task_ewc(
    net, task_idx, pair_name, n_steps, opt, train_cur,
    *, ewc_baseline, trigger, ceilings, pruner,
    do_growth, do_pruning, label, global_step_offset, n_total_pairs,
):
    fires: List[dict] = []
    prunes: List[dict] = []
    pathology_steps = 0
    stab_remaining = 0
    ewc_now = ewc_baseline

    for step in range(n_steps):
        a, b = train_cur.sample_pair(pair_name, batch=BATCH)
        if do_pruning and pruner is not None:
            with utility_capture(net, mode="combined") as cap:
                h_a = net(a); h_b = net(b)
                l_task = contrastive_loss(h_a, h_b, MARGIN)
                l = l_task + ewc_now * net.ewc_penalty() if ewc_now > 0 else l_task
                opt.zero_grad(); l.backward()
                gnorm = total_gradient_norm(net.parameters())
                cap.update_layer_utilities()
        else:
            h_a = net(a); h_b = net(b)
            l_task = contrastive_loss(h_a, h_b, MARGIN)
            l = l_task + ewc_now * net.ewc_penalty() if ewc_now > 0 else l_task
            opt.zero_grad(); l.backward()
            gnorm = total_gradient_norm(net.parameters())
        opt.step()

        if stab_remaining > 0:
            stab_remaining -= 1
            if stab_remaining == 0:
                ewc_now = ewc_baseline
                if ceilings is not None:
                    ceilings.mark_stabilization_end()

        if trigger is not None:
            s = trigger.observe(loss=l_task.item(), hidden=h_a.detach(), grad_norm=gnorm)
            if s.loss_plateau and s.rank_saturated and not s.grad_stable:
                pathology_steps += 1
            if (s.fire and do_growth and ceilings is not None
                    and not ceilings.arrested and stab_remaining == 0):
                target_idx = len(net.layers) - 1
                decision = ceilings.preflight(net, target_idx)
                fire_record = {
                    "task_idx": task_idx, "pair_name": pair_name,
                    "global_step": global_step_offset + step, "task_step": step,
                    "allowed": decision.allowed, "reason": decision.reason,
                    "effective_rank": s.effective_rank, "grad_norm": s.grad_norm,
                    "loss_at_fire": l_task.item(),
                }
                if decision.allowed:
                    consolidate_task(net, train_cur, pair_name)
                    v = compute_growth_direction(net, train_cur, pair_name)
                    new_idx = net.grow_layer(target_idx, init_vec=v)
                    fire_record["new_node_idx"] = new_idx
                    fire_record["latent_after"] = net.layers[-1].n_nodes
                    trigger.set_latent_dim(net.layers[-1].n_nodes)
                    trigger.reset()
                    opt = optim.Adam(net.parameters(), lr=LR)
                    ewc_now = EWC_STAB_BOOST
                    stab_remaining = T_STABILIZE
                    ceilings.mark_stabilization_start()
                    print(f"  [{label}] *** FIRE @ task {task_idx+1}/{n_total_pairs} step {step}: "
                          f"ALLOW; latent {net.layers[-1].n_nodes-1}→{net.layers[-1].n_nodes}")
                else:
                    print(f"  [{label}] *** FIRE @ task {task_idx+1}/{n_total_pairs} step {step}: "
                          f"DENY ({decision.reason})")
                fires.append(fire_record)

        if do_pruning and pruner is not None:
            pruned = pruner.step(net, global_step_offset + step)
            if pruned:
                for L_idx, n_idx in pruned:
                    prunes.append({
                        "task_idx": task_idx, "pair_name": pair_name,
                        "global_step": global_step_offset + step,
                        "layer_idx": L_idx, "node_idx": n_idx,
                        "arch_after": tuple(net.n_nodes_per_layer()),
                    })
                opt = optim.Adam(net.parameters(), lr=LR)
                print(f"  [{label}] PRUNE @ task {task_idx+1}/{n_total_pairs} step {step}: "
                      f"{pruned} → arch {net.n_nodes_per_layer()}")

        if step == 0 or (step + 1) % LOG_EVERY == 0 or step == n_steps - 1:
            print(f"  [{label}] task {task_idx+1}/{n_total_pairs} ({pair_name}) "
                  f"step {step:5d}  loss {l_task.item():.4f}  "
                  f"arch {net.n_nodes_per_layer()}  params {net.n_parameters()}")

    return {"opt": opt, "fires": fires, "prunes": prunes,
            "pathology_steps": pathology_steps, "ewc_now": ewc_now}


def run_ewc_curriculum(net, label, *, do_growth, do_pruning,
                       train_cur, eval_batches, pair_names):
    print(f"\n[{label}] curriculum start — arch {net.n_nodes_per_layer()}  "
          f"params {net.n_parameters()}  growth={do_growth} pruning={do_pruning}")

    opt = optim.Adam(net.parameters(), lr=LR)
    trigger = (GrowthTrigger(latent_dim=net.layers[-1].n_nodes, window=TRIGGER_W,
                             eps_loss=TRIGGER_EPS_LOSS, eps_rank=TRIGGER_EPS_RANK,
                             g_min=TRIGGER_G_MIN, g_max=TRIGGER_G_MAX)
               if do_growth else None)
    ceilings = (CeilingsController(M_max_bytes=M_MAX_BYTES, T_div_max_seconds=T_DIV_MAX_SECONDS)
                if do_growth else None)
    pruner = (PruningController(u_threshold=PRUNE_U_THRESHOLD, T_prune=PRUNE_T,
                                prune_clock=PRUNE_CLOCK)
              if do_pruning else None)

    K = len(pair_names)
    loss_matrix: List[List[float]] = [[float("nan")] * K for _ in range(K)]
    initial_n_params = net.n_parameters()
    initial_arch = tuple(net.n_nodes_per_layer())
    all_fires: List[dict] = []
    all_prunes: List[dict] = []
    total_pathology = 0
    cumulative_step = 0
    ewc_baseline = 0.0

    t0 = time.monotonic()
    for task_idx, pair_name in enumerate(pair_names):
        print(f"\n[{label}] === Task {task_idx+1}/{K}: {pair_name} ===")
        if trigger is not None:
            trigger.reset()
        result = train_one_task_ewc(
            net, task_idx, pair_name, n_steps=N_STEPS_PER_TASK, opt=opt,
            train_cur=train_cur, ewc_baseline=ewc_baseline, trigger=trigger,
            ceilings=ceilings, pruner=pruner, do_growth=do_growth,
            do_pruning=do_pruning, label=label,
            global_step_offset=cumulative_step, n_total_pairs=K,
        )
        opt = result["opt"]; all_fires.extend(result["fires"])
        all_prunes.extend(result["prunes"])
        total_pathology += result["pathology_steps"]
        cumulative_step += N_STEPS_PER_TASK

        consolidate_task(net, train_cur, pair_name)
        ewc_baseline = EWC_INTERTASK

        per_pair = evaluate_all_pairs(net, eval_batches, pair_names)
        for j, pname in enumerate(pair_names):
            loss_matrix[task_idx][j] = per_pair[pname]
        avg_so_far = sum(loss_matrix[task_idx][: task_idx + 1]) / (task_idx + 1)
        print(f"[{label}] After task {task_idx+1}: own={per_pair[pair_name]:.3f}  "
              f"avg over tasks 1..{task_idx+1}={avg_so_far:.3f}  "
              f"arch={net.n_nodes_per_layer()} params={net.n_parameters()}")

    elapsed = time.monotonic() - t0
    return _summarize(label, do_growth, do_pruning, initial_arch,
                      tuple(net.n_nodes_per_layer()), initial_n_params,
                      net.n_parameters(), net.layers[-1].n_nodes,
                      loss_matrix, all_fires, all_prunes, total_pathology,
                      elapsed)


# ---------------------------------------------------------------------
# PackNet path
# ---------------------------------------------------------------------


def run_packnet_curriculum(net, label, *, train_cur, eval_batches, pair_names):
    print(f"\n[{label}] curriculum start — arch {net.n_nodes_per_layer()}  "
          f"params {net.n_parameters()}  PackNet, n_tasks={len(pair_names)}")

    K = len(pair_names)
    ctrl = PackNetController(net, n_total_tasks=K)
    loss_matrix: List[List[float]] = [[float("nan")] * K for _ in range(K)]
    initial_n_params = net.n_parameters()
    initial_arch = tuple(net.n_nodes_per_layer())

    t0 = time.monotonic()
    for task_idx, pair_name in enumerate(pair_names):
        print(f"\n[{label}] === Task {task_idx+1}/{K}: {pair_name} ===")
        ctrl.begin_task(task_idx + 1)
        # Adam state is stale across the re-init; rebuild.
        opt = optim.Adam(net.parameters(), lr=LR)

        for step in range(N_STEPS_PER_TASK):
            a, b = train_cur.sample_pair(pair_name, batch=BATCH)
            h_a = net(a); h_b = net(b)
            l = contrastive_loss(h_a, h_b, MARGIN)
            opt.zero_grad()
            l.backward()
            ctrl.freeze_grads()
            opt.step()
            if step == 0 or (step + 1) % LOG_EVERY == 0 or step == N_STEPS_PER_TASK - 1:
                print(f"  [{label}] task {task_idx+1}/{K} ({pair_name}) "
                      f"step {step:5d}  loss {l.item():.4f}  "
                      f"frozen={ctrl.cumulative_frozen_count()}/{initial_n_params}")

        ctrl.end_task(task_idx + 1)

        # Eval all pairs already-learned (j ≤ task_idx) using their own masks.
        # For j > task_idx: those tasks haven't been trained yet. Per the
        # PackNet protocol we just leave loss_matrix[task_idx][j] = NaN, which
        # avg_final won't see (avg_final is computed from the LAST row).
        for j, pname in enumerate(pair_names):
            if j <= task_idx:
                snap = ctrl.apply_inference_mask(j + 1)
                with torch.no_grad():
                    a, b = eval_batches[pname]
                    loss_j = float(contrastive_loss(net(a), net(b), MARGIN).item())
                ctrl.restore(snap)
                loss_matrix[task_idx][j] = loss_j
            else:
                loss_matrix[task_idx][j] = float("nan")

        own = loss_matrix[task_idx][task_idx]
        avg_so_far = (
            sum(loss_matrix[task_idx][: task_idx + 1]) / (task_idx + 1)
        )
        print(f"[{label}] After task {task_idx+1}: own={own:.3f}  "
              f"avg over tasks 1..{task_idx+1}={avg_so_far:.3f}  "
              f"frozen_total={ctrl.cumulative_frozen_count()}")

    elapsed = time.monotonic() - t0
    return _summarize(label, False, False, initial_arch,
                      tuple(net.n_nodes_per_layer()), initial_n_params,
                      net.n_parameters(), net.layers[-1].n_nodes,
                      loss_matrix, [], [], 0, elapsed,
                      packnet_capacity=ctrl.per_task_capacity())


# ---------------------------------------------------------------------
# Result summary helper
# ---------------------------------------------------------------------


def _summarize(label, do_growth, do_pruning, initial_arch, final_arch,
               initial_n_params, final_n_params, final_latent,
               loss_matrix, fires, prunes, pathology_steps, elapsed,
               packnet_capacity=None):
    K = len(loss_matrix)
    final_eval = loss_matrix[K - 1]
    avg_final = sum(final_eval) / K
    forgetting_per_task: List[float] = []
    for j in range(K - 1):
        diag = loss_matrix[j][j]
        end = loss_matrix[K - 1][j]
        forgetting_per_task.append(end - diag)
    avg_forgetting = (sum(forgetting_per_task) / len(forgetting_per_task)
                      if forgetting_per_task else float("nan"))
    return {
        "label": label, "do_growth": do_growth, "do_pruning": do_pruning,
        "initial_arch": initial_arch, "final_arch": final_arch,
        "initial_n_params": initial_n_params, "final_n_params": final_n_params,
        "final_latent": final_latent, "loss_matrix": loss_matrix,
        "avg_final_loss": avg_final, "forgetting_per_task": forgetting_per_task,
        "avg_forgetting": avg_forgetting, "fires": fires, "prunes": prunes,
        "pathology_steps": pathology_steps, "wall_clock_seconds": elapsed,
        "packnet_capacity": packnet_capacity,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    torch.manual_seed(SEED)
    pair_specs = build_progressive_pairs(
        state_dim=STATE_DIM, n_single=N_SINGLE, n_compound=N_COMPOUND, seed=SEED,
    )
    pair_names = [p.name for p in pair_specs]

    def cur_factory(seed):
        return ParameterizedContrastiveCurriculum(
            state_dim=STATE_DIM, pair_specs=pair_specs, seed=seed)

    eval_batches = make_fixed_eval_batches(pair_names, cur_factory)

    print("=" * 78)
    print("Trioron — bench_packnet: PackNet head-to-head on bench_harder curriculum")
    print("=" * 78)
    print(f"Curriculum:        16-dim state, 12 single + 8 compound = 20 tasks")
    print(f"Per-task budget:   {N_STEPS_PER_TASK} steps")
    print(f"PackNet widths:    {PACKNET_HIDDEN_SIZES}")
    print()

    # --- Grown ---
    torch.manual_seed(SEED)
    grown_net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
    grown_result = run_ewc_curriculum(
        grown_net, label="grown", do_growth=True, do_pruning=True,
        train_cur=cur_factory(seed=SEED), eval_batches=eval_batches,
        pair_names=pair_names,
    )

    # --- Fixed-EWC at H=16 (the param-matched baseline from bench_harder) ---
    torch.manual_seed(SEED + 16)
    fixed_ewc_net = make_network(STATE_DIM, 16, grown_result["final_latent"])
    fixed_ewc_result = run_ewc_curriculum(
        fixed_ewc_net, label="fixed_ewc_H16", do_growth=False, do_pruning=False,
        train_cur=cur_factory(seed=SEED + 7 * 16), eval_batches=eval_batches,
        pair_names=pair_names,
    )

    # --- PackNet sweep ---
    packnet_results: List[dict] = []
    for H in PACKNET_HIDDEN_SIZES:
        torch.manual_seed(SEED + H)
        pn_net = make_network(STATE_DIM, H, grown_result["final_latent"])
        result = run_packnet_curriculum(
            pn_net, label=f"packnet_H{H}",
            train_cur=cur_factory(seed=SEED + 13 * H),
            eval_batches=eval_batches, pair_names=pair_names,
        )
        packnet_results.append(result)

    target_params = grown_result["final_n_params"]
    matched_packnet = min(
        packnet_results, key=lambda r: abs(r["final_n_params"] - target_params)
    )

    # --- Final report ---
    print()
    print("=" * 78)
    print("bench_packnet — Final Report")
    print("=" * 78)
    print()
    print(f"Grown:")
    print(f"  arch:         {grown_result['initial_arch']} → {grown_result['final_arch']}")
    print(f"  params:       {grown_result['initial_n_params']} → {grown_result['final_n_params']}")
    print(f"  divisions:    {len([f for f in grown_result['fires'] if f.get('allowed')])} allowed")
    print(f"  avg final loss:    {grown_result['avg_final_loss']:.4f}")
    print(f"  avg forgetting:    {grown_result['avg_forgetting']:.4f}")
    print(f"  wall-clock:        {grown_result['wall_clock_seconds']:.1f}s")
    print()
    print(f"Fixed-EWC H=16 (param-matched):")
    print(f"  params:       {fixed_ewc_result['final_n_params']}")
    print(f"  avg final loss:    {fixed_ewc_result['avg_final_loss']:.4f}")
    print(f"  avg forgetting:    {fixed_ewc_result['avg_forgetting']:.4f}")
    print(f"  wall-clock:        {fixed_ewc_result['wall_clock_seconds']:.1f}s")
    print()
    print(f"PackNet sweep:")
    print(f"  {'H':>4} {'params':>8} {'avg_final':>10} {'avg_forget':>11} {'seconds':>9}")
    for r in packnet_results:
        marker = "  *" if r is matched_packnet else ""
        H = r["initial_arch"][1]
        cap_str = (
            f"  per-task: min={min(r['packnet_capacity'])} max={max(r['packnet_capacity'])}"
            if r["packnet_capacity"] else ""
        )
        print(f"  {H:>4d} {r['final_n_params']:>8d} "
              f"{r['avg_final_loss']:>10.4f} "
              f"{r['avg_forgetting']:>11.4f} "
              f"{r['wall_clock_seconds']:>9.1f}{marker}{cap_str}")
    print(f"  (* = closest in n_params to grown final = {target_params})")
    print()

    # --- Verdict ---
    print("Comparison vs PackNet:")
    g_vs_pn_match = (grown_result["avg_final_loss"], matched_packnet["avg_final_loss"])
    g_vs_pn_winner = "grown" if g_vs_pn_match[0] < g_vs_pn_match[1] else "PackNet"
    print(f"  param-matched: grown {g_vs_pn_match[0]:.4f} vs PackNet "
          f"{g_vs_pn_match[1]:.4f} — winner: {g_vs_pn_winner}")

    best_pn = min(packnet_results, key=lambda r: r["avg_final_loss"])
    print(f"  best PackNet:  H={best_pn['initial_arch'][1]} "
          f"({best_pn['final_n_params']} params) avg_final={best_pn['avg_final_loss']:.4f}")
    print(f"  vs grown ({grown_result['final_n_params']} params, "
          f"avg_final={grown_result['avg_final_loss']:.4f})")
    print()
    print("Comparison vs Fixed-EWC H=16:")
    print(f"  grown {grown_result['avg_final_loss']:.4f} vs "
          f"fixed-EWC {fixed_ewc_result['avg_final_loss']:.4f}")
    print()
    print("Caveat: PackNet uses task-ID at inference time (selects per-task mask).")
    print("Grown and fixed-EWC don't. This is a structural advantage for PackNet")
    print("that the literature standardly accepts; we report it explicitly.")
    print()

    # --- CSV ---
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_packnet_log.csv")
    all_results = [grown_result, fixed_ewc_result, *packnet_results]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = [
            "label", "initial_arch", "final_arch",
            "initial_n_params", "final_n_params", "final_latent",
            "wall_clock_seconds", "avg_final_loss", "avg_forgetting",
            "n_divisions_allowed", "n_prunes", "pathology_steps",
        ]
        for i in range(N_PAIRS):
            for j in range(N_PAIRS):
                header.append(f"M[{i+1}][{j+1}]")
        w.writerow(header)
        for r in all_results:
            n_alw = len([f for f in r["fires"] if f.get("allowed")])
            row = [
                r["label"], str(r["initial_arch"]), str(r["final_arch"]),
                r["initial_n_params"], r["final_n_params"], r["final_latent"],
                f"{r['wall_clock_seconds']:.2f}",
                f"{r['avg_final_loss']:.6f}", f"{r['avg_forgetting']:.6f}",
                n_alw, len(r["prunes"]), r["pathology_steps"],
            ]
            for i in range(N_PAIRS):
                for j in range(N_PAIRS):
                    v = r["loss_matrix"][i][j]
                    row.append("" if v != v else f"{v:.6f}")  # NaN → ""
            w.writerow(row)
    print(f"  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
