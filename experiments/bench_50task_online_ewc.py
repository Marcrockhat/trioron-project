"""Online EWC baseline (Schwarz et al. 2018) — multi-seed, 50-task curriculum.

The standard fixed-EWC baseline used in bench_step8 / bench_harder /
bench_packnet / bench_50task is *naive single-anchor EWC*: at each
task boundary, Fisher is reset and re-estimated on the new task only,
the anchor is overwritten to the current weights. The literature
considers this naive — it's what the original Kirkpatrick 2017 paper
called "uninformed about prior tasks beyond the most recent."

Online EWC (Schwarz et al. 2018, "Progress & Compress: A scalable
framework for continual learning") improves this by accumulating
Fisher information across tasks via a discounted EMA:

    F̃_new = γ · F̃_old + F_t

where F_t is the just-estimated Fisher for the current task and γ ∈
(0, 1] is a discount factor (Schwarz et al. use γ ≈ 1 in practice;
slightly lower values guard against unbounded growth). The anchor is
still single (current weights), but the importance weights remember
prior tasks.

Predicted behavior in our regime:

  - Our hinge contrastive loss has a zero-gradient region; raw Fisher
    estimates are near-zero at convergence. Standard EWC compensates
    via LAMBDA_FLOOR (a uniform floor applied after each estimate).
  - Online EWC's accumulation across tasks should partially substitute
    for the floor — even if F_t ≈ 0 at convergence, F̃ accumulates
    contributions from non-converged steps across many tasks, drifting
    upward.
  - Whether this beats grown is the question.

This bench: 3 seeds × 3 widths {H=8, 12, 16}, same curriculum and
trigger/EWC hyperparameters as bench_50task_seeds. Compares against
grown's already-collected seed distribution from bench_50task_seeds.
"""
from __future__ import annotations
import csv
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron.incubator import contrastive_loss
from trioron.curriculum import ParameterizedContrastiveCurriculum

from experiments.bench_50task import (
    STATE_DIM,
    N_SINGLE,
    N_COMPOUND,
    N_PAIRS,
    N_STEPS_PER_TASK,
    BATCH,
    MARGIN,
    LR,
    LAMBDA_FLOOR,
    EWC_INTERTASK,
    LOG_EVERY,
    build_50task_pairs,
    make_fixed_eval_batches,
    make_network,
    estimate_fisher_for_pair,
    evaluate_all_pairs,
)


SEEDS = [0, 1, 2]
ONLINE_EWC_HIDDEN_SIZES = [8, 12, 16]
GAMMA_ONLINE = 0.95  # Online-EWC discount factor

# Grown's already-collected distribution from bench_50task_seeds_run1.
GROWN_AVG_FINALS = [0.0297, 0.0689, 0.0603]
GROWN_FINAL_PARAMS = [247, 264, 252]
GROWN_FINAL_LATENTS = [4, 3, 3]
TARGET_LATENT = int(round(sum(GROWN_FINAL_LATENTS) / len(GROWN_FINAL_LATENTS)))


def _agg(xs: List[float]) -> Tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else float("nan"), 0.0)
    return statistics.mean(xs), statistics.stdev(xs)


def consolidate_task_online(
    net: TrioronNetwork,
    train_cur: ParameterizedContrastiveCurriculum,
    pair_name: str,
    task_idx: int,
    gamma: float = GAMMA_ONLINE,
) -> None:
    """Online-EWC consolidation. On task 0 this is identical to the
    standard consolidate_task (no prior λ to accumulate). On task ≥ 1
    it discounts the prior λ and adds the current task's Fisher
    estimate.

    Pseudocode:
        if task_idx == 0:
            standard consolidate (estimate, λ ← Fisher mean, λ floor, anchor)
        else:
            prev_λ = current λ.clone()
            estimate Fisher (resets fisher_W, populates with current task)
            new_F  = mean of fresh fisher_W across incoming weights
            online_λ = γ · prev_λ + new_F
            λ ← clamp(online_λ, min=LAMBDA_FLOOR)
            anchor at current weights
    """
    if task_idx == 0:
        estimate_fisher_for_pair(net, train_cur, pair_name)
        net.update_lambda_all()
        with torch.no_grad():
            for layer in net.layers:
                layer.lam.clamp_(min=LAMBDA_FLOOR)
        net.anchor_all()
        return

    prev_lambdas = [layer.lam.detach().clone() for layer in net.layers]
    estimate_fisher_for_pair(net, train_cur, pair_name)
    new_fishers = [layer.fisher_W.mean(dim=1).clone() for layer in net.layers]

    with torch.no_grad():
        for li, layer in enumerate(net.layers):
            online_lam = gamma * prev_lambdas[li] + new_fishers[li]
            online_lam.clamp_(min=LAMBDA_FLOOR)
            layer.lam.copy_(online_lam)
    net.anchor_all()


def train_one_task_online_ewc(
    net, task_idx, pair_name, n_steps, opt, train_cur,
    *, ewc_baseline, label, n_total_pairs,
):
    ewc_now = ewc_baseline
    for step in range(n_steps):
        a, b = train_cur.sample_pair(pair_name, batch=BATCH)
        h_a = net(a); h_b = net(b)
        l_task = contrastive_loss(h_a, h_b, MARGIN)
        l = l_task + ewc_now * net.ewc_penalty() if ewc_now > 0 else l_task
        opt.zero_grad(); l.backward()
        opt.step()
        if step == 0 or (step + 1) % LOG_EVERY == 0 or step == n_steps - 1:
            print(f"  [{label}] task {task_idx+1}/{n_total_pairs} ({pair_name}) "
                  f"step {step:5d}  loss {l_task.item():.4f}")
    return {"opt": opt}


def run_online_ewc_curriculum(
    net, label, *, train_cur, eval_batches, pair_names,
):
    print(f"\n[{label}] Online-EWC curriculum start — arch {net.n_nodes_per_layer()}  "
          f"params {net.n_parameters()}  γ={GAMMA_ONLINE}")

    opt = optim.Adam(net.parameters(), lr=LR)
    K = len(pair_names)
    loss_matrix: List[List[float]] = [[float("nan")] * K for _ in range(K)]
    initial_n_params = net.n_parameters()
    initial_arch = tuple(net.n_nodes_per_layer())
    ewc_baseline = 0.0  # zero before any anchor

    t0 = time.monotonic()
    for task_idx, pair_name in enumerate(pair_names):
        print(f"\n[{label}] === Task {task_idx+1}/{K}: {pair_name} ===")
        train_one_task_online_ewc(
            net, task_idx, pair_name, n_steps=N_STEPS_PER_TASK, opt=opt,
            train_cur=train_cur, ewc_baseline=ewc_baseline,
            label=label, n_total_pairs=K,
        )

        consolidate_task_online(net, train_cur, pair_name, task_idx)
        ewc_baseline = EWC_INTERTASK

        per_pair = evaluate_all_pairs(net, eval_batches, pair_names)
        for j, pname in enumerate(pair_names):
            loss_matrix[task_idx][j] = per_pair[pname]
        avg_so_far = sum(loss_matrix[task_idx][: task_idx + 1]) / (task_idx + 1)
        # Per-layer λ summary so we can see accumulation in the log.
        lam_means = [f"{layer.lam.mean().item():.3f}" for layer in net.layers]
        print(f"[{label}] After task {task_idx+1}: own={per_pair[pair_name]:.3f}  "
              f"avg_to_date={avg_so_far:.3f}  λ-means={lam_means}")

    elapsed = time.monotonic() - t0
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
        "label": label, "initial_arch": initial_arch,
        "final_arch": tuple(net.n_nodes_per_layer()),
        "initial_n_params": initial_n_params,
        "final_n_params": net.n_parameters(),
        "final_latent": net.layers[-1].n_nodes,
        "loss_matrix": loss_matrix,
        "avg_final_loss": avg_final,
        "avg_forgetting": avg_forgetting,
        "wall_clock_seconds": elapsed,
    }


def main() -> int:
    pair_specs = build_50task_pairs(
        state_dim=STATE_DIM, n_single=N_SINGLE, n_compound=N_COMPOUND, seed=0,
    )
    pair_names = [p.name for p in pair_specs]

    def cur_factory(seed):
        return ParameterizedContrastiveCurriculum(
            state_dim=STATE_DIM, pair_specs=pair_specs, seed=seed)

    eval_batches = make_fixed_eval_batches(pair_names, cur_factory)

    print("=" * 78)
    print("bench_50task_online_ewc — Online EWC × 3 widths × 3 seeds")
    print("=" * 78)
    print(f"Seeds:  {SEEDS}")
    print(f"Widths: {ONLINE_EWC_HIDDEN_SIZES}")
    print(f"γ (online discount): {GAMMA_ONLINE}")
    print(f"Target latent: {TARGET_LATENT}")
    print()

    rows: List[dict] = []
    results_by_H: Dict[int, List[dict]] = {H: [] for H in ONLINE_EWC_HIDDEN_SIZES}
    t_start = time.monotonic()

    for H in ONLINE_EWC_HIDDEN_SIZES:
        for s in SEEDS:
            torch.manual_seed(s + 3000 + H)
            net = make_network(STATE_DIM, H, TARGET_LATENT)
            result = run_online_ewc_curriculum(
                net, label=f"online_ewc_H{H}_seed{s}",
                train_cur=cur_factory(seed=s + 7 * H + 100),
                eval_batches=eval_batches, pair_names=pair_names,
            )
            rows.append({
                "arch": "online_ewc",
                "H": H,
                "seed": s,
                "n_params_final": result["final_n_params"],
                "final_latent": result["final_latent"],
                "avg_final_loss": result["avg_final_loss"],
                "avg_forgetting": result["avg_forgetting"],
                "wall_clock_seconds": result["wall_clock_seconds"],
            })
            results_by_H[H].append(result)

    elapsed_total = time.monotonic() - t_start

    # ----- Report -----
    print()
    print("=" * 78)
    print(f"bench_50task_online_ewc — Final Report  "
          f"(total wall-clock {elapsed_total/60:.1f} min)")
    print("=" * 78)
    print()
    print(f"Online-EWC ({len(SEEDS)} seeds × {len(ONLINE_EWC_HIDDEN_SIZES)} widths):")
    print(f"  {'H':>4} {'params':>8} {'avg_final mean±std':>22} "
          f"{'forget mean±std':>20} {'per-seed avg_final':>40}")
    for H in ONLINE_EWC_HIDDEN_SIZES:
        results = results_by_H[H]
        finals = [r["avg_final_loss"] for r in results]
        forgets = [r["avg_forgetting"] for r in results]
        params = results[0]["final_n_params"]
        f_mean, f_std = _agg(finals)
        fg_mean, fg_std = _agg(forgets)
        finals_str = "[" + ", ".join(f"{x:.4f}" for x in finals) + "]"
        print(f"  {H:>4d} {params:>8d}    {f_mean:.4f} ± {f_std:.4f}"
              f"    {fg_mean:+.4f} ± {fg_std:.4f}    {finals_str}")
    print()

    g_mean, g_std = _agg(GROWN_AVG_FINALS)
    print(f"Grown distribution (from bench_50task_seeds_run1):")
    print(f"  per-seed: {GROWN_AVG_FINALS} → mean ± std: {g_mean:.4f} ± {g_std:.4f}")
    print()

    matched_H = min(
        ONLINE_EWC_HIDDEN_SIZES,
        key=lambda H: abs(results_by_H[H][0]["final_n_params"] -
                          (sum(GROWN_FINAL_PARAMS) / len(GROWN_FINAL_PARAMS))),
    )
    matched_finals = [r["avg_final_loss"] for r in results_by_H[matched_H]]
    m_mean, m_std = _agg(matched_finals)
    print(f"Headline: grown vs matched Online-EWC (H={matched_H}, "
          f"~{results_by_H[matched_H][0]['final_n_params']} params):")
    print(f"  grown:        {g_mean:.4f} ± {g_std:.4f}")
    print(f"  Online EWC:   {m_mean:.4f} ± {m_std:.4f}")
    rel = (m_mean - g_mean) / g_mean if g_mean > 0 else 0
    print(f"  grown is {rel*100:.1f}% lower than Online-EWC mean")
    if g_std + m_std > 0:
        gap_in_std = (m_mean - g_mean) / max(g_std + m_std, 1e-9)
        print(f"  gap is {gap_in_std:.1f}× the combined std — "
              f"{'robust' if gap_in_std > 2 else 'borderline' if gap_in_std > 1 else 'within noise'}")

    best_oe_H = min(
        ONLINE_EWC_HIDDEN_SIZES,
        key=lambda H: statistics.mean([r["avg_final_loss"] for r in results_by_H[H]]),
    )
    best_finals = [r["avg_final_loss"] for r in results_by_H[best_oe_H]]
    bo_mean, bo_std = _agg(best_finals)
    bo_params = results_by_H[best_oe_H][0]["final_n_params"]
    print()
    print(f"Best Online-EWC width: H={best_oe_H} ({bo_params} params, "
          f"{bo_params / (sum(GROWN_FINAL_PARAMS) / len(GROWN_FINAL_PARAMS)):.2f}× "
          f"grown's avg params):")
    print(f"  grown:      {g_mean:.4f} ± {g_std:.4f}")
    print(f"  Online EWC: {bo_mean:.4f} ± {bo_std:.4f}")
    if g_mean > 0:
        rel = (bo_mean - g_mean) / g_mean
        print(f"  grown is {rel*100:.1f}% lower than best Online-EWC")
    if g_std + bo_std > 0:
        gap_in_std = (bo_mean - g_mean) / max(g_std + bo_std, 1e-9)
        print(f"  gap is {gap_in_std:.1f}× the combined std — "
              f"{'robust' if gap_in_std > 2 else 'borderline' if gap_in_std > 1 else 'within noise'}")

    # Quick lookup vs the bench_50task_seeds fixed-EWC numbers (single-anchor)
    # so we can tell if Online EWC actually moved the needle.
    print()
    print("Cross-reference (from bench_50task_seeds_run1, single-anchor EWC):")
    print("  fixed-EWC H=8  (203p): 0.0950 ± 0.0147")
    print("  fixed-EWC H=12 (351p): 0.0730 ± 0.0250  (best)")
    print("  fixed-EWC H=16 (531p): 0.1021 ± 0.0240")

    # CSV
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_50task_online_ewc_log.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arch", "H", "seed", "n_params_final", "final_latent",
                    "avg_final_loss", "avg_forgetting", "wall_clock_seconds"])
        for r in rows:
            w.writerow([r["arch"], r["H"], r["seed"], r["n_params_final"],
                        r["final_latent"], f"{r['avg_final_loss']:.6f}",
                        f"{r['avg_forgetting']:.6f}",
                        f"{r['wall_clock_seconds']:.2f}"])
    print(f"\n  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
