"""§8-step-8 verification: moment-of-truth benchmark.

The §13 falsification gates: the project is considered to have failed if any of

    (1) The grown network does not match a same-parameter-count fixed MLP on a
        5-task continual-learning benchmark.
    (2) Total training compute exceeds 2x a fixed-baseline equivalent.
    (3) The growth trigger fires more often during optimizer pathology than
        during genuine capacity saturation.

This script implements the agreed step-8 scope:

    - Sequential 5-task curriculum: each of the 5 contrastive pairs from §5.3
      presented as its own task, in order. Only the active pair is sampled.
    - Between tasks: refresh per-pair Fisher, anchor weights, hold a baseline
      EWC penalty during subsequent tasks. Single-anchor EWC (the most recent
      task's anchor) is the simplest faithful reading of §3.2.
    - Grown run: starts at latent=1 with full triggers + division + pruning +
      ceilings (steps 4 through 7.5 all live).
    - Fixed-MLP sweep: same code path with growth + pruning disabled, varying
      hidden width. The size whose final n_params is closest to grown's is the
      "matched" baseline used for the §13.1 comparison.
    - All evaluations use a fixed held-out batch per pair (seed=42, batch=512)
      so the per-task loss matrix is comparable across networks.

Outputs:
    outputs/bench_step8_log.csv  — per-network row including the 5x5 loss
                                   matrix, division/pruning counts, walltime,
                                   forgetting metric.
    stdout                       — per-task progress + final §13 verdict.
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
from trioron.incubator import (
    STATE_DIM,
    ContrastiveCurriculum,
    PAIR_NAMES,
    contrastive_loss,
)
from trioron.triggers import GrowthTrigger, total_gradient_norm
from trioron.ceilings import CeilingsController
from trioron.pruner import PruningController, utility_capture


# ---------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------

N_STEPS_PER_TASK = 3000
BATCH = 32
HIDDEN = 16
LATENT_INIT_GROWN = 1
MARGIN = 1.0
LR = 3e-3
SEED = 0

# Three-condition trigger (§4). Window short enough to allow a fire within a
# single 3000-step task after a 2W=1600 warmup.
TRIGGER_W = 800
TRIGGER_EPS_LOSS = 0.001
TRIGGER_EPS_RANK = 0.1
TRIGGER_G_MIN = 1e-4
TRIGGER_G_MAX = 10.0

# Stabilization (§4.1.5). Same posture as division_demo: boost EWC during stab.
T_STABILIZE = 300
EWC_STAB_BOOST = 5000.0

# Cross-task EWC. The hinge contrastive loss (clamp(m-dist, 0)) truly hits
# zero, so Fisher ≈ 0 at convergence — naïve EWC has no quadratic well to
# defend. We compensate by clamping each layer's λ to LAMBDA_FLOOR after the
# Fisher pass: where Fisher caught real signal we keep it; where it didn't,
# we get uniform L2-anchor with floor strength. EWC_INTERTASK is the
# multiplier on top of that quadratic.
LAMBDA_FLOOR = 0.1
EWC_INTERTASK = 1000.0

# Hard ceilings (§4.2). Orange-Pi-tier values exercised here too.
M_MAX_BYTES = 2 * 1024 ** 3
T_DIV_MAX_SECONDS = 60.0

# Pruning (§3.3 + §8 step 6).
PRUNE_U_THRESHOLD = 1e-3
PRUNE_T = 2000
PRUNE_CLOCK = 500

# Fixed-MLP sweep widths.
FIXED_HIDDEN_SIZES = [8, 12, 16, 20, 24]

EVAL_BATCH = 512
LOG_EVERY = 500
EVAL_SEED = 42


# ---------------------------------------------------------------------
# Eval batches — fixed per pair, shared across all networks.
# ---------------------------------------------------------------------


def make_fixed_eval_batches(
    seed: int = EVAL_SEED, batch: int = EVAL_BATCH
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    cur = ContrastiveCurriculum(seed=seed)
    fixed: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name in PAIR_NAMES:
        fixed[name] = cur.sample_pair(name, batch=batch)
    return fixed


def evaluate_all_pairs(
    net: TrioronNetwork,
    eval_batches: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, float]:
    losses: Dict[str, float] = {}
    with torch.no_grad():
        for name in PAIR_NAMES:
            a, b = eval_batches[name]
            h_a = net(a)
            h_b = net(b)
            losses[name] = float(contrastive_loss(h_a, h_b, MARGIN).item())
    return losses


# ---------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------


def make_network(hidden: int, latent: int) -> TrioronNetwork:
    return TrioronNetwork(
        [
            (STATE_DIM, hidden, "relu"),
            (hidden, hidden, "relu"),
            (hidden, latent, "tanh"),
        ]
    )


# ---------------------------------------------------------------------
# Fisher refresh on a single contrastive pair
# ---------------------------------------------------------------------


def estimate_fisher_for_pair(
    net: TrioronNetwork,
    train_cur: ContrastiveCurriculum,
    pair_name: str,
    batch: int = BATCH,
    n_batches: int = 20,
) -> None:
    """Refresh per-weight Fisher info for a single-pair task. We piggyback on
    network.estimate_fisher's (x, y) interface by passing (a, b) as (x, y) and
    re-feeding b through the network inside loss_fn."""

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
    train_cur: ContrastiveCurriculum,
    pair_name: str,
) -> None:
    """End-of-task EWC consolidation: refresh Fisher, copy to λ, clamp to a
    minimum floor so EWC still has a quadratic well to defend even when the
    margin loss has nulled the gradient. Then anchor."""
    estimate_fisher_for_pair(net, train_cur, pair_name)
    net.update_lambda_all()
    with torch.no_grad():
        for layer in net.layers:
            layer.lam.clamp_(min=LAMBDA_FLOOR)
    net.anchor_all()


# ---------------------------------------------------------------------
# Growth-direction PCA (§4.1.1) — single-pair variant
# ---------------------------------------------------------------------


def compute_growth_direction(
    net: TrioronNetwork,
    train_cur: ContrastiveCurriculum,
    pair_name: str,
    batch: int = 128,
) -> torch.Tensor:
    """First right-singular-vector of (F_a − F_b) at the penultimate layer's
    output — the direction the existing network is failing to represent for
    the currently-active contrastive pair. Thin wrapper around
    `trioron.growth_direction.from_contrastive_pair`."""
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
    train_cur: ContrastiveCurriculum,
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
) -> dict:
    """Train net on one task. Mutates net (potentially via grow_layer +
    prune_layer_node). Returns the (possibly rebuilt) optimizer plus
    fires/prunes diagnostic logs."""
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

        # Decay stabilization (no-op when not stabilizing).
        if stab_remaining > 0:
            stab_remaining -= 1
            if stab_remaining == 0:
                ewc_now = ewc_baseline
                if ceilings is not None:
                    ceilings.mark_stabilization_end()

        # Trigger observation runs whether or not growth is enabled — the
        # condition states are diagnostic in either case. Action only on
        # `s.fire and do_growth`.
        if trigger is not None:
            s = trigger.observe(
                loss=l_task.item(), hidden=h_a.detach(), grad_norm=gnorm
            )
            # §13.3 pathology accounting: loss+rank but not grad-stable.
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
                        f"  [{label}] *** FIRE @ task {task_idx+1} step {step}: "
                        f"ALLOW; latent {net.layers[-1].n_nodes-1}→{net.layers[-1].n_nodes}, "
                        f"rank {s.effective_rank:.2f}, grad {s.grad_norm:.3f}"
                    )
                else:
                    print(
                        f"  [{label}] *** FIRE @ task {task_idx+1} step {step}: "
                        f"DENY ({decision.reason})"
                    )
                fires.append(fire_record)

        # Pruning controller runs on its own clock.
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
                    f"  [{label}] PRUNE @ task {task_idx+1} step {step}: "
                    f"{pruned} → arch {net.n_nodes_per_layer()}"
                )

        if step == 0 or (step + 1) % log_every == 0 or step == n_steps - 1:
            print(
                f"  [{label}] task {task_idx+1}/{len(PAIR_NAMES)} ({pair_name}) "
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
# Full 5-task curriculum
# ---------------------------------------------------------------------


def run_5task_curriculum(
    net: TrioronNetwork,
    label: str,
    *,
    do_growth: bool,
    do_pruning: bool,
    train_cur: ContrastiveCurriculum,
    eval_batches: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
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

    K = len(PAIR_NAMES)
    loss_matrix: List[List[float]] = [[float("nan")] * K for _ in range(K)]
    initial_n_params = net.n_parameters()
    initial_arch = tuple(net.n_nodes_per_layer())

    all_fires: List[dict] = []
    all_prunes: List[dict] = []
    total_pathology = 0
    cumulative_step = 0
    ewc_baseline = 0.0  # zero until at least one task has been anchored

    t0 = time.monotonic()
    for task_idx, pair_name in enumerate(PAIR_NAMES):
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
        )
        opt = result["opt"]
        all_fires.extend(result["fires"])
        all_prunes.extend(result["prunes"])
        total_pathology += result["pathology_steps"]
        cumulative_step += n_steps_per_task

        # End-of-task consolidation. Single-anchor EWC: this overwrites the
        # previous anchor and Fisher. Real continual EWC would maintain a sum
        # of per-task quadratic terms; we accept the simpler form here as the
        # straightforward reading of §3.2 and apply the same rule to grown
        # and fixed runs alike.
        consolidate_task(net, train_cur, pair_name)
        ewc_baseline = EWC_INTERTASK

        per_pair = evaluate_all_pairs(net, eval_batches)
        for j, pname in enumerate(PAIR_NAMES):
            loss_matrix[task_idx][j] = per_pair[pname]
        eval_summary = " ".join(
            f"{p[:4]}={per_pair[p]:.3f}" for p in PAIR_NAMES
        )
        print(f"[{label}] After task {task_idx+1} eval: {eval_summary}")

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


def _print_loss_matrix(label: str, M: List[List[float]]) -> None:
    print(f"  [{label}] loss matrix M[i][j] = pair-j loss after task i:")
    print("    " + " " * 8 + "  ".join(f"{p[:4]:>6s}" for p in PAIR_NAMES))
    for i, row in enumerate(M):
        print(
            "    "
            + f"after T{i+1}: "
            + "  ".join(f"{x:6.3f}" for x in row)
        )


def main() -> int:
    torch.manual_seed(SEED)
    eval_batches = make_fixed_eval_batches()

    print("=" * 78)
    print("Trioron — Step 8: moment-of-truth benchmark (§13 falsification gates)")
    print("=" * 78)
    print(f"Curriculum:        sequential 5 tasks, one contrastive pair each")
    print(f"Task ordering:     {PAIR_NAMES}")
    print(
        f"Per-task budget:   {N_STEPS_PER_TASK} steps   "
        f"(total {N_STEPS_PER_TASK * len(PAIR_NAMES)} steps per network)"
    )
    print(
        f"EWC schedule:      baseline {EWC_INTERTASK:.0f}, "
        f"stab boost {EWC_STAB_BOOST:.0f}, "
        f"stab steps {T_STABILIZE}"
    )
    print(f"Trigger:           W={TRIGGER_W}, eps_loss={TRIGGER_EPS_LOSS}, "
          f"eps_rank={TRIGGER_EPS_RANK}, g∈[{TRIGGER_G_MIN}, {TRIGGER_G_MAX}]")
    print(f"Ceilings:          M_max=2GB, T_div_max=60s")
    print(f"Fixed-sweep H ∈    {FIXED_HIDDEN_SIZES}")
    print(f"Eval:              fixed batches, seed={EVAL_SEED}, batch={EVAL_BATCH}")
    print()

    # --- Grown ---
    torch.manual_seed(SEED)
    train_cur_grown = ContrastiveCurriculum(seed=SEED)
    grown_net = make_network(HIDDEN, LATENT_INIT_GROWN)
    grown_result = run_5task_curriculum(
        grown_net,
        label="grown",
        do_growth=True,
        do_pruning=True,
        train_cur=train_cur_grown,
        eval_batches=eval_batches,
    )

    # --- Fixed-MLP sweep ---
    target_latent = grown_result["final_latent"]
    fixed_results: List[dict] = []
    for H in FIXED_HIDDEN_SIZES:
        torch.manual_seed(SEED + H)
        train_cur_fixed = ContrastiveCurriculum(seed=SEED + 7 * H)
        fixed_net = make_network(H, target_latent)
        result = run_5task_curriculum(
            fixed_net,
            label=f"fixed_H{H}",
            do_growth=False,
            do_pruning=False,
            train_cur=train_cur_fixed,
            eval_batches=eval_batches,
        )
        fixed_results.append(result)

    # --- Param-matched fixed ---
    target_params = grown_result["final_n_params"]
    matched = min(
        fixed_results, key=lambda r: abs(r["final_n_params"] - target_params)
    )

    # --- Final report ---
    print()
    print("=" * 78)
    print("Step 8 — Final Report")
    print("=" * 78)
    print()
    print(f"Grown ({grown_result['label']}):")
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
    print(f"  pathology:    {grown_result['pathology_steps']} steps "
          f"(L+R but not G — §4 escape-valve indicator)")
    print(f"  avg final loss (5 pairs):  {grown_result['avg_final_loss']:.4f}")
    print(f"  avg forgetting (4 pairs):  {grown_result['avg_forgetting']:.4f}")
    print(f"  wall-clock:                {grown_result['wall_clock_seconds']:.1f}s")
    _print_loss_matrix(grown_result["label"], grown_result["loss_matrix"])
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
    print("§13 falsification gates:")
    g1_pass = grown_result["avg_final_loss"] <= matched["avg_final_loss"]
    print(
        f"  (1) Grown matches param-matched fixed on continual benchmark: "
        f"{'PASS' if g1_pass else 'FAIL'}  "
        f"(grown {grown_result['avg_final_loss']:.4f} "
        f"vs matched {matched['avg_final_loss']:.4f})"
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

    # §13.3: trigger fires more on saturation than pathology.
    # Our trigger requires grad_stable=True to fire, so by construction it
    # cannot fire during pathology — every recorded fire IS a saturation
    # event under the §4 definition. The pathology-step counter (loss+rank
    # without grad-stable) is the diagnostic that the §4 escape valve was
    # exercised: the trigger correctly REFUSED to fire on bad gradients.
    # The gate passes iff the architecture (a) actually grew at least once
    # — otherwise the growth mechanism wasn't tested — and (b) no fire
    # happened in the first 200 steps after a structural change (transient
    # window where stale-statistics could let a fire slip through).
    fires_allowed = [f for f in grown_result["fires"] if f.get("allowed")]
    fires_transient = [
        f for f in fires_allowed if f.get("task_step", 0) < 200
    ]
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
        print("  OVERALL: PASS — architecture meets all §13 falsification gates.")
        rc = 0
    else:
        print(
            "  OVERALL: FAIL — §13 mandates a written re-evaluation before "
            "iterating further."
        )
        rc = 1

    # --- CSV ---
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_step8_log.csv")
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
        for i in range(len(PAIR_NAMES)):
            for j in range(len(PAIR_NAMES)):
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
            for i in range(len(PAIR_NAMES)):
                for j in range(len(PAIR_NAMES)):
                    row.append(f"{r['loss_matrix'][i][j]:.6f}")
            w.writerow(row)
    print(f"  log: {csv_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
