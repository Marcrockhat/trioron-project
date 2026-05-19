"""Harder-curriculum revisit of the §13 falsification gates.

Step 8 (bench_step8.py) cleared §13 on a 5-task curriculum drawn from
the original blueprint §5.3 pairs. That benchmark didn't answer two
questions:

  - Does the architecture keep working past 5 tasks? Step 8 ended with
    grown at latent=4 after 3 fires. Does it scale to 20 tasks?
  - Does the saturation evidence survive a structural distribution
    shift? In §5.3 every pair is single-dim and orthogonal. Compound
    pairs (two dims jointly) cannot be solved by any single-dim
    projection and force the network to compose latent directions —
    a different kind of capacity tension.

This bench answers both with the same harness:

  - 16-dim state (twice the §5.1 default).
  - 20-task curriculum: 12 single-dim pairs (dims 0..11) followed by
    8 compound (two-dim XOR) pairs over randomly-chosen dim pairs.
    The single → compound boundary at task 13 is the structural shift.
  - 1500 steps per task (50% of step 8's budget — total 30 000 steps
    per network, 6 networks ⇒ ~50 min on CPU).
  - Same EWC posture as bench_step8 (LAMBDA_FLOOR=0.1, baseline=1000,
    boost=5000), same trigger / ceilings / pruner machinery.
  - Trigger window W=400 (warmup 800 < 1500-step task — fits in one
    task with margin).

§13 verdict logic carried over from bench_step8: same three gates,
same decision rules.
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
from trioron.incubator import contrastive_loss  # loss is curriculum-agnostic
from trioron.curriculum import (
    ParameterizedContrastiveCurriculum,
    build_progressive_pairs,
)
from trioron.triggers import GrowthTrigger, total_gradient_norm
from trioron.ceilings import CeilingsController
from trioron.pruner import PruningController, utility_capture


# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------

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

FIXED_HIDDEN_SIZES = [12, 16, 20, 24]

EVAL_BATCH = 512
LOG_EVERY = 500
EVAL_SEED = 42


# ---------------------------------------------------------------------
# Eval batches — fixed per pair, shared across all networks.
# ---------------------------------------------------------------------


def make_fixed_eval_batches(
    pair_names: List[str], curriculum_factory, seed: int = EVAL_SEED, batch: int = EVAL_BATCH
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    cur = curriculum_factory(seed=seed)
    fixed: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name in pair_names:
        fixed[name] = cur.sample_pair(name, batch=batch)
    return fixed


def evaluate_all_pairs(
    net: TrioronNetwork,
    eval_batches: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    pair_names: List[str],
) -> Dict[str, float]:
    losses: Dict[str, float] = {}
    with torch.no_grad():
        for name in pair_names:
            a, b = eval_batches[name]
            h_a = net(a)
            h_b = net(b)
            losses[name] = float(contrastive_loss(h_a, h_b, MARGIN).item())
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


def estimate_fisher_for_pair(
    net: TrioronNetwork,
    train_cur: ParameterizedContrastiveCurriculum,
    pair_name: str,
    batch: int = BATCH,
    n_batches: int = 20,
) -> None:
    def batches():
        for _ in range(n_batches):
            a, b = train_cur.sample_pair(pair_name, batch=batch)
            yield a, b

    def loss_fn(pred_a: torch.Tensor, b_input: torch.Tensor) -> torch.Tensor:
        h_b = net(b_input)
        return contrastive_loss(pred_a, h_b, MARGIN)

    net.estimate_fisher(batches(), loss_fn, n_batches=n_batches)


def consolidate_task(
    net: TrioronNetwork,
    train_cur: ParameterizedContrastiveCurriculum,
    pair_name: str,
) -> None:
    estimate_fisher_for_pair(net, train_cur, pair_name)
    net.update_lambda_all()
    with torch.no_grad():
        for layer in net.layers:
            layer.lam.clamp_(min=LAMBDA_FLOOR)
    net.anchor_all()


# ---------------------------------------------------------------------
# Growth-direction PCA
# ---------------------------------------------------------------------


def compute_growth_direction(
    net: TrioronNetwork,
    train_cur: ParameterizedContrastiveCurriculum,
    pair_name: str,
    batch: int = 128,
) -> torch.Tensor:
    """Thin wrapper around `trioron.growth_direction.from_contrastive_pair`."""
    from trioron.growth_direction import from_contrastive_pair
    a, b = train_cur.sample_pair(pair_name, batch=batch)
    dest_idx = len(net.layers) - 1
    return from_contrastive_pair(net, a, b, dest_layer_idx=dest_idx, k=1)[0]


# ---------------------------------------------------------------------
# Single-task training loop
# ---------------------------------------------------------------------


def train_one_task(
    net: TrioronNetwork,
    task_idx: int,
    pair_name: str,
    n_steps: int,
    opt: optim.Optimizer,
    train_cur: ParameterizedContrastiveCurriculum,
    *,
    ewc_baseline: float = 0.0,
    trigger: Optional[GrowthTrigger] = None,
    ceilings: Optional[CeilingsController] = None,
    pruner: Optional[PruningController] = None,
    do_growth: bool = False,
    do_pruning: bool = False,
    label: str = "",
    global_step_offset: int = 0,
    log_every: int = LOG_EVERY,
    n_total_pairs: int = N_PAIRS,
) -> dict:
    fires: List[dict] = []
    prunes: List[dict] = []
    pathology_steps = 0
    stab_remaining = 0
    ewc_now = ewc_baseline

    for step in range(n_steps):
        a, b = train_cur.sample_pair(pair_name, batch=BATCH)

        if do_pruning and pruner is not None:
            with utility_capture(net, mode="combined") as cap:
                h_a = net(a)
                h_b = net(b)
                l_task = contrastive_loss(h_a, h_b, MARGIN)
                l = l_task + ewc_now * net.ewc_penalty() if ewc_now > 0 else l_task
                opt.zero_grad()
                l.backward()
                gnorm = total_gradient_norm(net.parameters())
                cap.update_layer_utilities()
        else:
            h_a = net(a)
            h_b = net(b)
            l_task = contrastive_loss(h_a, h_b, MARGIN)
            l = l_task + ewc_now * net.ewc_penalty() if ewc_now > 0 else l_task
            opt.zero_grad()
            l.backward()
            gnorm = total_gradient_norm(net.parameters())
        opt.step()

        if stab_remaining > 0:
            stab_remaining -= 1
            if stab_remaining == 0:
                ewc_now = ewc_baseline
                if ceilings is not None:
                    ceilings.mark_stabilization_end()

        if trigger is not None:
            s = trigger.observe(
                loss=l_task.item(), hidden=h_a.detach(), grad_norm=gnorm
            )
            if s.loss_plateau and s.rank_saturated and not s.grad_stable:
                pathology_steps += 1

            if (
                s.fire
                and do_growth
                and ceilings is not None
                and not ceilings.arrested
                and stab_remaining == 0
            ):
                target_idx = len(net.layers) - 1
                decision = ceilings.preflight(net, target_idx)
                fire_record = {
                    "task_idx": task_idx,
                    "pair_name": pair_name,
                    "global_step": global_step_offset + step,
                    "task_step": step,
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    "effective_rank": s.effective_rank,
                    "grad_norm": s.grad_norm,
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
                    print(
                        f"  [{label}] *** FIRE @ task {task_idx+1}/{n_total_pairs} "
                        f"({pair_name}) step {step}: ALLOW; latent "
                        f"{net.layers[-1].n_nodes-1}→{net.layers[-1].n_nodes}, "
                        f"rank {s.effective_rank:.2f}, grad {s.grad_norm:.3f}"
                    )
                else:
                    print(
                        f"  [{label}] *** FIRE @ task {task_idx+1}/{n_total_pairs} "
                        f"({pair_name}) step {step}: DENY ({decision.reason})"
                    )
                fires.append(fire_record)

        if do_pruning and pruner is not None:
            pruned = pruner.step(net, global_step_offset + step)
            if pruned:
                for L_idx, n_idx in pruned:
                    prunes.append(
                        {
                            "task_idx": task_idx,
                            "pair_name": pair_name,
                            "global_step": global_step_offset + step,
                            "layer_idx": L_idx,
                            "node_idx": n_idx,
                            "arch_after": tuple(net.n_nodes_per_layer()),
                        }
                    )
                opt = optim.Adam(net.parameters(), lr=LR)
                print(
                    f"  [{label}] PRUNE @ task {task_idx+1}/{n_total_pairs} step {step}: "
                    f"{pruned} → arch {net.n_nodes_per_layer()}"
                )

        if step == 0 or (step + 1) % log_every == 0 or step == n_steps - 1:
            print(
                f"  [{label}] task {task_idx+1}/{n_total_pairs} ({pair_name}) "
                f"step {step:5d}  loss {l_task.item():.4f}  "
                f"arch {net.n_nodes_per_layer()}  params {net.n_parameters()}"
            )

    return {
        "opt": opt,
        "fires": fires,
        "prunes": prunes,
        "pathology_steps": pathology_steps,
        "ewc_now": ewc_now,
    }


# ---------------------------------------------------------------------
# Full N-task curriculum
# ---------------------------------------------------------------------


def run_n_task_curriculum(
    net: TrioronNetwork,
    label: str,
    *,
    do_growth: bool,
    do_pruning: bool,
    train_cur: ParameterizedContrastiveCurriculum,
    eval_batches: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    pair_names: List[str],
    n_steps_per_task: int = N_STEPS_PER_TASK,
) -> dict:
    print()
    print("-" * 78)
    print(
        f"[{label}] curriculum start — arch {net.n_nodes_per_layer()}  "
        f"params {net.n_parameters()}  growth={do_growth} pruning={do_pruning}"
    )
    print("-" * 78)

    opt = optim.Adam(net.parameters(), lr=LR)
    trigger = (
        GrowthTrigger(
            latent_dim=net.layers[-1].n_nodes,
            window=TRIGGER_W,
            eps_loss=TRIGGER_EPS_LOSS,
            eps_rank=TRIGGER_EPS_RANK,
            g_min=TRIGGER_G_MIN,
            g_max=TRIGGER_G_MAX,
        )
        if do_growth
        else None
    )
    ceilings = (
        CeilingsController(
            M_max_bytes=M_MAX_BYTES, T_div_max_seconds=T_DIV_MAX_SECONDS
        )
        if do_growth
        else None
    )
    pruner = (
        PruningController(
            u_threshold=PRUNE_U_THRESHOLD,
            T_prune=PRUNE_T,
            prune_clock=PRUNE_CLOCK,
        )
        if do_pruning
        else None
    )

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
        print()
        print(f"[{label}] === Task {task_idx+1}/{K}: {pair_name} ===")
        if trigger is not None:
            trigger.reset()

        result = train_one_task(
            net,
            task_idx,
            pair_name,
            n_steps=n_steps_per_task,
            opt=opt,
            train_cur=train_cur,
            ewc_baseline=ewc_baseline,
            trigger=trigger,
            ceilings=ceilings,
            pruner=pruner,
            do_growth=do_growth,
            do_pruning=do_pruning,
            label=label,
            global_step_offset=cumulative_step,
            n_total_pairs=K,
        )
        opt = result["opt"]
        all_fires.extend(result["fires"])
        all_prunes.extend(result["prunes"])
        total_pathology += result["pathology_steps"]
        cumulative_step += n_steps_per_task

        consolidate_task(net, train_cur, pair_name)
        ewc_baseline = EWC_INTERTASK

        per_pair = evaluate_all_pairs(net, eval_batches, pair_names)
        for j, pname in enumerate(pair_names):
            loss_matrix[task_idx][j] = per_pair[pname]
        # Compact per-task summary: avg loss across all K pairs and the
        # diagonal (this task's own loss right after training it).
        avg_so_far = sum(loss_matrix[task_idx][: task_idx + 1]) / (task_idx + 1)
        print(
            f"[{label}] After task {task_idx+1}: own={per_pair[pair_name]:.3f}  "
            f"avg over tasks 1..{task_idx+1}={avg_so_far:.3f}  "
            f"arch={net.n_nodes_per_layer()} params={net.n_parameters()}"
        )

    elapsed = time.monotonic() - t0
    final_arch = tuple(net.n_nodes_per_layer())
    final_params = net.n_parameters()
    final_latent = net.layers[-1].n_nodes
    final_eval = loss_matrix[K - 1]
    avg_final = sum(final_eval) / K

    forgetting_per_task: List[float] = []
    for j in range(K - 1):
        diag = loss_matrix[j][j]
        end = loss_matrix[K - 1][j]
        forgetting_per_task.append(end - diag)
    avg_forgetting = (
        sum(forgetting_per_task) / len(forgetting_per_task)
        if forgetting_per_task
        else float("nan")
    )

    return {
        "label": label,
        "do_growth": do_growth,
        "do_pruning": do_pruning,
        "initial_arch": initial_arch,
        "final_arch": final_arch,
        "initial_n_params": initial_n_params,
        "final_n_params": final_params,
        "final_latent": final_latent,
        "loss_matrix": loss_matrix,
        "avg_final_loss": avg_final,
        "forgetting_per_task": forgetting_per_task,
        "avg_forgetting": avg_forgetting,
        "fires": all_fires,
        "prunes": all_prunes,
        "pathology_steps": total_pathology,
        "wall_clock_seconds": elapsed,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def _print_loss_summary(label: str, M: List[List[float]], pair_names: List[str]) -> None:
    K = len(pair_names)
    diag = [M[i][i] for i in range(K)]
    final_row = M[K - 1]
    pre = M[:N_SINGLE]
    post = M[N_SINGLE:]
    avg_diag = sum(diag) / K
    avg_final = sum(final_row) / K
    avg_pre_diag = sum(M[i][i] for i in range(N_SINGLE)) / N_SINGLE if N_SINGLE > 0 else float("nan")
    avg_post_diag = (
        sum(M[i][i] for i in range(N_SINGLE, K)) / N_COMPOUND
        if N_COMPOUND > 0
        else float("nan")
    )
    avg_pre_final = sum(M[K - 1][:N_SINGLE]) / N_SINGLE if N_SINGLE > 0 else float("nan")
    avg_post_final = (
        sum(M[K - 1][N_SINGLE:]) / N_COMPOUND if N_COMPOUND > 0 else float("nan")
    )
    print(f"  [{label}] loss summary:")
    print(f"     diagonal mean (own-task):       {avg_diag:.4f}")
    print(f"     final-row mean (end-of-cur):    {avg_final:.4f}")
    print(f"     single-pair phase: own={avg_pre_diag:.4f}  end={avg_pre_final:.4f}")
    print(f"     compound-pair phase: own={avg_post_diag:.4f}  end={avg_post_final:.4f}")


def main() -> int:
    torch.manual_seed(SEED)

    pair_specs = build_progressive_pairs(
        state_dim=STATE_DIM,
        n_single=N_SINGLE,
        n_compound=N_COMPOUND,
        seed=SEED,
    )
    pair_names = [p.name for p in pair_specs]

    def cur_factory(seed):
        return ParameterizedContrastiveCurriculum(
            state_dim=STATE_DIM, pair_specs=pair_specs, seed=seed
        )

    eval_batches = make_fixed_eval_batches(pair_names, cur_factory)

    print("=" * 78)
    print("Trioron — bench_harder: §13 revisit on a 20-task / 16-dim curriculum")
    print("=" * 78)
    print(f"State dim:         {STATE_DIM}")
    print(f"Pairs:             {N_SINGLE} single-dim + {N_COMPOUND} compound = {N_PAIRS}")
    print(f"Per-task budget:   {N_STEPS_PER_TASK} steps   "
          f"(total {N_STEPS_PER_TASK * N_PAIRS} steps per network)")
    print(
        f"EWC schedule:      baseline {EWC_INTERTASK:.0f}, "
        f"stab boost {EWC_STAB_BOOST:.0f}, stab steps {T_STABILIZE}, "
        f"λ-floor {LAMBDA_FLOOR}"
    )
    print(f"Trigger:           W={TRIGGER_W}, eps_loss={TRIGGER_EPS_LOSS}, "
          f"eps_rank={TRIGGER_EPS_RANK}, g∈[{TRIGGER_G_MIN}, {TRIGGER_G_MAX}]")
    print(f"Fixed-sweep H ∈    {FIXED_HIDDEN_SIZES}")
    print(f"Eval:              fixed batches, seed={EVAL_SEED}, batch={EVAL_BATCH}")
    print()

    # --- Grown ---
    torch.manual_seed(SEED)
    train_cur_grown = cur_factory(seed=SEED)
    grown_net = make_network(STATE_DIM, HIDDEN, LATENT_INIT_GROWN)
    grown_result = run_n_task_curriculum(
        grown_net,
        label="grown",
        do_growth=True,
        do_pruning=True,
        train_cur=train_cur_grown,
        eval_batches=eval_batches,
        pair_names=pair_names,
    )

    # --- Fixed-MLP sweep ---
    target_latent = grown_result["final_latent"]
    fixed_results: List[dict] = []
    for H in FIXED_HIDDEN_SIZES:
        torch.manual_seed(SEED + H)
        train_cur_fixed = cur_factory(seed=SEED + 7 * H)
        fixed_net = make_network(STATE_DIM, H, target_latent)
        result = run_n_task_curriculum(
            fixed_net,
            label=f"fixed_H{H}",
            do_growth=False,
            do_pruning=False,
            train_cur=train_cur_fixed,
            eval_batches=eval_batches,
            pair_names=pair_names,
        )
        fixed_results.append(result)

    target_params = grown_result["final_n_params"]
    matched = min(
        fixed_results, key=lambda r: abs(r["final_n_params"] - target_params)
    )

    # --- Final report ---
    print()
    print("=" * 78)
    print("bench_harder — Final Report")
    print("=" * 78)
    print()
    print(f"Grown:")
    print(
        f"  arch:         {grown_result['initial_arch']} → "
        f"{grown_result['final_arch']}"
    )
    print(
        f"  params:       {grown_result['initial_n_params']} → "
        f"{grown_result['final_n_params']}"
    )
    n_alw = len([f for f in grown_result["fires"] if f.get("allowed")])
    n_att = len(grown_result["fires"])
    print(f"  divisions:    {n_alw} allowed / {n_att} attempted")
    print(f"  prunings:     {len(grown_result['prunes'])}")
    print(f"  pathology:    {grown_result['pathology_steps']} steps")
    print(f"  avg final loss ({N_PAIRS} pairs):  {grown_result['avg_final_loss']:.4f}")
    print(f"  avg forgetting ({N_PAIRS-1} pairs): {grown_result['avg_forgetting']:.4f}")
    print(f"  wall-clock:                {grown_result['wall_clock_seconds']:.1f}s")

    # When-fires-fired report: how many fires before vs after the structural
    # boundary (single → compound at task N_SINGLE).
    fires_phase1 = [
        f for f in grown_result["fires"] if f.get("allowed") and f["task_idx"] < N_SINGLE
    ]
    fires_phase2 = [
        f for f in grown_result["fires"] if f.get("allowed") and f["task_idx"] >= N_SINGLE
    ]
    print(
        f"  fire timing:  {len(fires_phase1)} during single-pair phase "
        f"(tasks 1..{N_SINGLE}), {len(fires_phase2)} during compound phase "
        f"(tasks {N_SINGLE+1}..{N_PAIRS})"
    )

    _print_loss_summary("grown", grown_result["loss_matrix"], pair_names)
    print()
    print("Fixed-MLP sweep:")
    print(
        f"  {'H':>4} {'params':>8} {'avg_final':>10} "
        f"{'avg_forget':>11} {'seconds':>9}"
    )
    for r in fixed_results:
        marker = "  *" if r is matched else ""
        H = r["initial_arch"][1]
        print(
            f"  {H:>4d} {r['final_n_params']:>8d} "
            f"{r['avg_final_loss']:>10.4f} "
            f"{r['avg_forgetting']:>11.4f} "
            f"{r['wall_clock_seconds']:>9.1f}{marker}"
        )
    print(f"  (* = closest in n_params to grown final = {target_params})")
    print()

    # --- §13 verdict ---
    print("§13 falsification gates (revisited on harder curriculum):")
    g1_pass = grown_result["avg_final_loss"] <= matched["avg_final_loss"]
    print(
        f"  (1) Grown matches param-matched fixed: "
        f"{'PASS' if g1_pass else 'FAIL'}  "
        f"(grown {grown_result['avg_final_loss']:.4f} vs "
        f"matched {matched['avg_final_loss']:.4f})"
    )
    g2_ratio = grown_result["wall_clock_seconds"] / max(
        matched["wall_clock_seconds"], 1e-9
    )
    g2_pass = g2_ratio <= 2.0
    print(
        f"  (2) Compute ≤ 2x fixed equivalent: "
        f"{'PASS' if g2_pass else 'FAIL'}  "
        f"(grown / matched = {g2_ratio:.2f}x)"
    )
    fires_allowed = [f for f in grown_result["fires"] if f.get("allowed")]
    fires_transient = [f for f in fires_allowed if f.get("task_step", 0) < 200]
    g3_pass = len(fires_allowed) >= 1 and len(fires_transient) == 0
    print(
        f"  (3) Trigger fires only on saturation, not pathology: "
        f"{'PASS' if g3_pass else 'FAIL'}  "
        f"(allowed-fires={len(fires_allowed)}, "
        f"transient-fires={len(fires_transient)}, "
        f"pathology-steps={grown_result['pathology_steps']})"
    )

    overall = g1_pass and g2_pass and g3_pass
    print()
    if overall:
        print("  OVERALL: PASS — architecture survives the harder curriculum.")
        rc = 0
    else:
        print(
            "  OVERALL: FAIL — at least one gate failed on the harder "
            "curriculum; revisit before any further scope expansion."
        )
        rc = 1

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_harder_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = [
            "label",
            "do_growth",
            "do_pruning",
            "initial_arch",
            "final_arch",
            "initial_n_params",
            "final_n_params",
            "final_latent",
            "wall_clock_seconds",
            "avg_final_loss",
            "avg_forgetting",
            "n_divisions_allowed",
            "n_divisions_attempted",
            "n_prunes",
            "pathology_steps",
        ]
        for i in range(N_PAIRS):
            for j in range(N_PAIRS):
                header.append(f"M[{i+1}][{j+1}]")
        w.writerow(header)
        for r in [grown_result, *fixed_results]:
            n_alw_r = len([f for f in r["fires"] if f.get("allowed")])
            n_att_r = len(r["fires"])
            row = [
                r["label"],
                int(r["do_growth"]),
                int(r["do_pruning"]),
                str(r["initial_arch"]),
                str(r["final_arch"]),
                r["initial_n_params"],
                r["final_n_params"],
                r["final_latent"],
                f"{r['wall_clock_seconds']:.2f}",
                f"{r['avg_final_loss']:.6f}",
                f"{r['avg_forgetting']:.6f}",
                n_alw_r,
                n_att_r,
                len(r["prunes"]),
                r["pathology_steps"],
            ]
            for i in range(N_PAIRS):
                for j in range(N_PAIRS):
                    row.append(f"{r['loss_matrix'][i][j]:.6f}")
            w.writerow(row)
    print(f"  log: {csv_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
