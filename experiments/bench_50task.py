"""Capacity-stress revisit: 50-task curriculum on 12-dim state.

bench_packnet showed grown beating PackNet ~3× at matched params on 20
tasks. The PackNet failure mode was clear: at 612 params / 20 tasks,
PackNet allocated ~30 weights per task — too sparse for a 16-dim
contrastive task. The architecture's continuous-allocation advantage
showed up when task count was high relative to network width.

This bench tightens that screw further to test consistency:

  - State dim: 12 (down from 16) — narrower input.
  - Tasks: 12 single-dim + 38 compound XOR pairs = 50 total. The 38
    compound pairs are sampled WITHOUT replacement from the C(12,2)=66
    possible (d1,d2) combinations, so every compound task is a unique
    XOR over a unique dim-pair. Compound pairs share dims with single
    pairs by construction (since both draw from the same 12-dim space).
  - Hidden width sweep: H ∈ {8, 12, 16}. H=8 is below grown's likely
    final params; H=16 is above. Brackets the comparison.
  - Per-task budget: 1500 steps (same as bench_packnet so the
    apples-to-apples comparison stays direct on the per-task budget).
  - Grown starts at HIDDEN=12 (matches state dim), latent=1.

Question this answers: does grown's PackNet-beating advantage
hold/grow/collapse at higher task density? And does the fixed-EWC
margin (which was 6% at 20 tasks) widen or narrow?
"""
from __future__ import annotations
import csv
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron.incubator import contrastive_loss
from trioron.curriculum import (
    ContrastivePairSpec,
    ParameterizedContrastiveCurriculum,
)
from trioron.triggers import GrowthTrigger, total_gradient_norm
from trioron.ceilings import CeilingsController
from trioron.pruner import PruningController, utility_capture
from trioron.packnet import PackNetController


# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------

STATE_DIM = 12
N_SINGLE = 12
N_COMPOUND = 38
N_PAIRS = N_SINGLE + N_COMPOUND  # 50
N_STEPS_PER_TASK = 1500
BATCH = 32
HIDDEN = 12
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

SWEEP_HIDDEN_SIZES = [8, 12, 16]

EVAL_BATCH = 512
LOG_EVERY = 750
EVAL_SEED = 42


# ---------------------------------------------------------------------
# Curriculum builder — unique compound pairs by sorted-tuple
# ---------------------------------------------------------------------


def build_50task_pairs(
    state_dim: int = STATE_DIM,
    n_single: int = N_SINGLE,
    n_compound: int = N_COMPOUND,
    seed: int = SEED,
    low: float = 0.1,
    high: float = 0.9,
) -> List[ContrastivePairSpec]:
    """50-task curriculum builder. Compound pairs are unique XOR pairs
    sampled without replacement from C(state_dim, 2) possible (d1<d2)
    combinations. Different from build_progressive_pairs in curriculum.py
    which samples with replacement and may produce duplicates."""
    if n_single > state_dim:
        raise ValueError(
            f"n_single={n_single} > state_dim={state_dim}"
        )
    all_compound: List[Tuple[int, int]] = []
    for d1 in range(state_dim):
        for d2 in range(d1 + 1, state_dim):
            all_compound.append((d1, d2))
    if n_compound > len(all_compound):
        raise ValueError(
            f"n_compound={n_compound} > unique pairs available "
            f"({len(all_compound)} for state_dim={state_dim})"
        )

    rng = random.Random(seed)
    rng.shuffle(all_compound)
    chosen_compound = all_compound[:n_compound]

    specs: List[ContrastivePairSpec] = []
    for d in range(n_single):
        specs.append(
            ContrastivePairSpec(
                name=f"single_{d:02d}",
                a_settings=[(d, low)],
                b_settings=[(d, high)],
            )
        )
    for i, (d1, d2) in enumerate(chosen_compound):
        specs.append(
            ContrastivePairSpec(
                name=f"compound_{i:02d}",
                a_settings=[(d1, low), (d2, high)],
                b_settings=[(d1, high), (d2, low)],
            )
        )
    return specs


# ---------------------------------------------------------------------
# Eval batches
# ---------------------------------------------------------------------


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
# Network construction
# ---------------------------------------------------------------------


def make_network(state_dim: int, hidden: int, latent: int) -> TrioronNetwork:
    return TrioronNetwork(
        [
            (state_dim, hidden, "relu"),
            (hidden, hidden, "relu"),
            (hidden, latent, "tanh"),
        ]
    )


# ---------------------------------------------------------------------
# Fisher refresh + λ floor
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


# ---------------------------------------------------------------------
# EWC training (grown + fixed-EWC)
# ---------------------------------------------------------------------


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
        # Compress per-task summary to one line for 50-task scrolling.
        print(f"[{label}] After task {task_idx+1}: own={per_pair[pair_name]:.3f}  "
              f"avg_to_date={avg_so_far:.3f}  arch={net.n_nodes_per_layer()} "
              f"params={net.n_parameters()}")

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
        avg_so_far = sum(loss_matrix[task_idx][: task_idx + 1]) / (task_idx + 1)
        print(f"[{label}] After task {task_idx+1}: own={own:.3f}  "
              f"avg_to_date={avg_so_far:.3f}  "
              f"frozen={ctrl.cumulative_frozen_count()}")

    elapsed = time.monotonic() - t0
    return _summarize(label, False, False, initial_arch,
                      tuple(net.n_nodes_per_layer()), initial_n_params,
                      net.n_parameters(), net.layers[-1].n_nodes,
                      loss_matrix, [], [], 0, elapsed,
                      packnet_capacity=ctrl.per_task_capacity())


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


def _print_loss_summary(label: str, M: List[List[float]]) -> None:
    K = len(M)
    diag = [M[i][i] for i in range(K)]
    final_row = M[K - 1]
    avg_diag = sum(diag) / K
    avg_final = sum(final_row) / K
    pre_diag = sum(M[i][i] for i in range(N_SINGLE)) / N_SINGLE
    post_diag = (
        sum(M[i][i] for i in range(N_SINGLE, K)) / (K - N_SINGLE)
        if K > N_SINGLE else float("nan")
    )
    pre_final = sum(M[K - 1][:N_SINGLE]) / N_SINGLE
    post_final = (
        sum(M[K - 1][N_SINGLE:]) / (K - N_SINGLE)
        if K > N_SINGLE else float("nan")
    )
    print(f"  [{label}] loss summary:")
    print(f"     diagonal mean (own-task):       {avg_diag:.4f}")
    print(f"     final-row mean (end-of-cur):    {avg_final:.4f}")
    print(f"     single-pair phase: own={pre_diag:.4f}  end={pre_final:.4f}")
    print(f"     compound-pair phase: own={post_diag:.4f}  end={post_final:.4f}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> int:
    torch.manual_seed(SEED)
    pair_specs = build_50task_pairs(
        state_dim=STATE_DIM, n_single=N_SINGLE, n_compound=N_COMPOUND, seed=SEED,
    )
    pair_names = [p.name for p in pair_specs]

    def cur_factory(seed):
        return ParameterizedContrastiveCurriculum(
            state_dim=STATE_DIM, pair_specs=pair_specs, seed=seed)

    eval_batches = make_fixed_eval_batches(pair_names, cur_factory)

    print("=" * 78)
    print("Trioron — bench_50task: capacity stress at 50-task density")
    print("=" * 78)
    print(f"State dim:         {STATE_DIM}")
    print(f"Tasks:             {N_SINGLE} single + {N_COMPOUND} compound = {N_PAIRS}")
    print(f"Per-task budget:   {N_STEPS_PER_TASK} steps   "
          f"(total {N_STEPS_PER_TASK * N_PAIRS} steps per network)")
    print(f"Sweep H ∈          {SWEEP_HIDDEN_SIZES}")
    print(f"Grown init:        H={HIDDEN}, latent={LATENT_INIT_GROWN}")
    print()

    # --- Grown ---
    torch.manual_seed(SEED)
    grown_net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
    grown_result = run_ewc_curriculum(
        grown_net, label="grown", do_growth=True, do_pruning=True,
        train_cur=cur_factory(seed=SEED), eval_batches=eval_batches,
        pair_names=pair_names,
    )

    # --- Fixed-EWC sweep ---
    fixed_results: List[dict] = []
    target_latent = grown_result["final_latent"]
    for H in SWEEP_HIDDEN_SIZES:
        torch.manual_seed(SEED + H)
        net = make_network(STATE_DIM, H, target_latent)
        result = run_ewc_curriculum(
            net, label=f"fixed_ewc_H{H}", do_growth=False, do_pruning=False,
            train_cur=cur_factory(seed=SEED + 7 * H),
            eval_batches=eval_batches, pair_names=pair_names,
        )
        fixed_results.append(result)

    # --- PackNet sweep ---
    packnet_results: List[dict] = []
    for H in SWEEP_HIDDEN_SIZES:
        torch.manual_seed(SEED + 100 + H)
        net = make_network(STATE_DIM, H, target_latent)
        result = run_packnet_curriculum(
            net, label=f"packnet_H{H}",
            train_cur=cur_factory(seed=SEED + 13 * H),
            eval_batches=eval_batches, pair_names=pair_names,
        )
        packnet_results.append(result)

    target_params = grown_result["final_n_params"]
    matched_fixed = min(
        fixed_results, key=lambda r: abs(r["final_n_params"] - target_params)
    )
    matched_packnet = min(
        packnet_results, key=lambda r: abs(r["final_n_params"] - target_params)
    )

    # --- Final report ---
    print()
    print("=" * 78)
    print("bench_50task — Final Report")
    print("=" * 78)
    print()
    print(f"Grown:")
    print(f"  arch:         {grown_result['initial_arch']} → {grown_result['final_arch']}")
    print(f"  params:       {grown_result['initial_n_params']} → {grown_result['final_n_params']}")
    n_alw = len([f for f in grown_result["fires"] if f.get("allowed")])
    fires_phase1 = [f for f in grown_result["fires"]
                    if f.get("allowed") and f["task_idx"] < N_SINGLE]
    fires_phase2 = [f for f in grown_result["fires"]
                    if f.get("allowed") and f["task_idx"] >= N_SINGLE]
    print(f"  divisions:    {n_alw} allowed "
          f"(single phase: {len(fires_phase1)}, compound phase: {len(fires_phase2)})")
    print(f"  prunings:     {len(grown_result['prunes'])}")
    print(f"  avg final loss:    {grown_result['avg_final_loss']:.4f}")
    print(f"  avg forgetting:    {grown_result['avg_forgetting']:.4f}")
    print(f"  wall-clock:        {grown_result['wall_clock_seconds']:.1f}s")
    _print_loss_summary("grown", grown_result["loss_matrix"])
    print()
    print(f"Fixed-EWC sweep:")
    print(f"  {'H':>4} {'params':>8} {'avg_final':>10} "
          f"{'avg_forget':>11} {'seconds':>9}")
    for r in fixed_results:
        marker = "  *" if r is matched_fixed else ""
        H = r["initial_arch"][1]
        print(f"  {H:>4d} {r['final_n_params']:>8d} "
              f"{r['avg_final_loss']:>10.4f} "
              f"{r['avg_forgetting']:>11.4f} "
              f"{r['wall_clock_seconds']:>9.1f}{marker}")
    print()
    print(f"PackNet sweep:")
    print(f"  {'H':>4} {'params':>8} {'avg_final':>10} "
          f"{'avg_forget':>11} {'seconds':>9}")
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
    print("Headline comparisons:")
    print(f"  grown {grown_result['avg_final_loss']:.4f}  vs  "
          f"fixed-EWC matched (H={matched_fixed['initial_arch'][1]}) "
          f"{matched_fixed['avg_final_loss']:.4f}")
    print(f"  grown {grown_result['avg_final_loss']:.4f}  vs  "
          f"PackNet matched (H={matched_packnet['initial_arch'][1]}) "
          f"{matched_packnet['avg_final_loss']:.4f}")
    print()
    print("Caveat: PackNet uses task-ID at inference time (per-task mask).")
    print("Grown / fixed-EWC don't. Standard literature asymmetry, kept in.")
    print()

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_50task_log.csv")
    all_results = [grown_result, *fixed_results, *packnet_results]
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
            n_alw_r = len([f for f in r["fires"] if f.get("allowed")])
            row = [
                r["label"], str(r["initial_arch"]), str(r["final_arch"]),
                r["initial_n_params"], r["final_n_params"], r["final_latent"],
                f"{r['wall_clock_seconds']:.2f}",
                f"{r['avg_final_loss']:.6f}", f"{r['avg_forgetting']:.6f}",
                n_alw_r, len(r["prunes"]), r["pathology_steps"],
            ]
            for i in range(N_PAIRS):
                for j in range(N_PAIRS):
                    v = r["loss_matrix"][i][j]
                    row.append("" if v != v else f"{v:.6f}")
            w.writerow(row)
    print(f"  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
