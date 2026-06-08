"""Split-MNIST canary — single seed, validates fan_in=784 path.

Pre-empts the obvious reviewer question ("does this work on real
images?") and de-risks the chained-15 headline bench. Two arms, no
dreaming, no pruning, no PackNet/HAT — that machinery comes later.

Architecture:
    784 → H_init=32 (grown) → H_init=32 (grown) → head (grows 2 → 10)

Key divergence from bench_50task: growth fires on layer 1 (the second
hidden), NOT on the head. The head is dictated by the curriculum (each
task adds 2 outputs deterministically via extend_output_head). The
growth trigger watches layer 1's post-relu activations for plateau +
rank-saturation + grad-stable.

Run:
    python3 -m experiments.bench_split_mnist
    python3 -m experiments.bench_split_mnist --smoke   # tiny budget
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron.classification import (
    accuracy,
    extend_output_head,
    masked_cross_entropy,
    summarize,
)
from trioron.triggers import GrowthTrigger, total_gradient_norm
from trioron.ceilings import CeilingsController

from experiments.datasets import (
    DEFAULT_DATA_ROOT,
    DatasetBundle,
    TaskDataView,
    build_task_views,
    split_mnist_specs,
)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

INPUT_DIM = 784
H_INIT_GROWN = 32       # grown starts narrower so the trigger has room to fire
H_FIXED = 64            # fixed-MLP baseline width (the "matched-fixed" point)
INIT_CLASSES = 2        # head starts at 2 outputs (task 1's class count)
N_STEPS_PER_TASK = 1500
BATCH = 64
LR = 1e-3
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

# No-cap for the canary (matches bench_50task) — capacity-stress is the
# headline bench's job. This is just a fan_in=784 sanity check.
M_MAX_BYTES_UNCAPPED = 2 * 1024 ** 3
T_DIV_MAX_SECONDS = 60.0

GROWTH_TARGET_LAYER_IDX = 1  # second hidden — NOT the head

LOG_EVERY = 250


# ---------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------


def make_classifier(input_dim: int, hidden: int, init_classes: int) -> TrioronNetwork:
    """Three-layer MLP classifier: relu → relu → linear logits."""
    return TrioronNetwork(
        [
            (input_dim, hidden, "relu"),
            (hidden, hidden, "relu"),
            (hidden, init_classes, "linear"),
        ]
    )


# ---------------------------------------------------------------------
# Forward helpers — split so we can grab layer-1 activations for the trigger
# ---------------------------------------------------------------------


def forward_with_hidden(net: TrioronNetwork, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (layer_1_post_relu_activation, logits). Used during training
    so the growth trigger can observe the second-hidden layer's rank."""
    h0 = net.layers[0](x)
    h1 = net.layers[1](h0)
    logits = net.layers[2](h1)
    return h1, logits


# ---------------------------------------------------------------------
# Fisher / consolidation for classification
# ---------------------------------------------------------------------


def estimate_fisher_for_task(
    net: TrioronNetwork,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    batch: int = BATCH,
    n_batches: int = 20,
) -> None:
    def batches():
        for _ in range(n_batches):
            x, y = train_view.sample(batch=batch)
            yield x, y

    active = list(active_classes)

    def loss_fn(pred_logits, y):
        return masked_cross_entropy(pred_logits, y, active_classes=active)

    net.estimate_fisher(batches(), loss_fn, n_batches=n_batches)


def consolidate_task(
    net: TrioronNetwork,
    train_view: TaskDataView,
    active_classes: Sequence[int],
) -> None:
    estimate_fisher_for_task(net, train_view, active_classes)
    net.update_lambda_all()
    with torch.no_grad():
        for layer in net.layers:
            layer.lam.clamp_(min=LAMBDA_FLOOR)
    net.anchor_all()


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------


def evaluate_all_tasks(
    net: TrioronNetwork,
    eval_views: Sequence[TaskDataView],
) -> List[float]:
    accs: List[float] = []
    with torch.no_grad():
        for v in eval_views:
            x, y = v.all_examples()
            logits = net(x)
            accs.append(accuracy(logits, y))
    return accs


# ---------------------------------------------------------------------
# One-task training loop (shared between grown + fixed-MLP arms)
# ---------------------------------------------------------------------


def train_one_classification_task(
    net: TrioronNetwork,
    task_idx: int,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    n_steps: int,
    opt: optim.Optimizer,
    *,
    ewc_baseline: float,
    trigger: Optional[GrowthTrigger],
    ceilings: Optional[CeilingsController],
    do_growth: bool,
    label: str,
    n_total_tasks: int,
    global_step_offset: int,
) -> Dict[str, object]:
    fires: List[dict] = []
    pathology_steps = 0
    stab_remaining = 0
    ewc_now = ewc_baseline
    active = list(active_classes)

    for step in range(n_steps):
        x, y_global = train_view.sample(BATCH)
        h1, logits = forward_with_hidden(net, x)
        l_task = masked_cross_entropy(logits, y_global, active_classes=active)
        l = (l_task + ewc_now * net.ewc_penalty()
             if ewc_now > 0 else l_task)
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
            s = trigger.observe(loss=l_task.item(), hidden=h1.detach(), grad_norm=gnorm)
            if s.loss_plateau and s.rank_saturated and not s.grad_stable:
                pathology_steps += 1
            if (s.fire and do_growth and ceilings is not None
                    and not ceilings.arrested and stab_remaining == 0):
                target_idx = GROWTH_TARGET_LAYER_IDX
                decision = ceilings.preflight(net, target_idx)
                fire_record = {
                    "task_idx": task_idx,
                    "global_step": global_step_offset + step,
                    "task_step": step,
                    "allowed": decision.allowed,
                    "reason": decision.reason,
                    "effective_rank": s.effective_rank,
                    "grad_norm": s.grad_norm,
                    "loss_at_fire": l_task.item(),
                }
                if decision.allowed:
                    consolidate_task(net, train_view, active)
                    new_idx = net.grow_layer(
                        target_idx, init_vec=None, task_idx=task_idx,
                    )
                    fire_record["new_node_idx"] = new_idx
                    fire_record["hidden_after"] = net.layers[target_idx].n_nodes
                    trigger.set_latent_dim(net.layers[target_idx].n_nodes)
                    trigger.reset()
                    opt = optim.Adam(net.parameters(), lr=LR)
                    ewc_now = EWC_STAB_BOOST
                    stab_remaining = T_STABILIZE
                    ceilings.mark_stabilization_start()
                    print(f"  [{label}] *** FIRE @ task {task_idx+1}/{n_total_tasks} step {step}: "
                          f"ALLOW; layer{target_idx} "
                          f"{net.layers[target_idx].n_nodes-1}→{net.layers[target_idx].n_nodes}")
                else:
                    print(f"  [{label}] *** FIRE @ task {task_idx+1}/{n_total_tasks} step {step}: "
                          f"DENY ({decision.reason})")
                fires.append(fire_record)

        if step == 0 or (step + 1) % LOG_EVERY == 0 or step == n_steps - 1:
            print(f"  [{label}] task {task_idx+1}/{n_total_tasks} ({train_view.name}) "
                  f"step {step:5d}  loss {l_task.item():.4f}  "
                  f"arch {net.n_nodes_per_layer()}  params {net.n_parameters()}")

    return {
        "opt": opt,
        "fires": fires,
        "pathology_steps": pathology_steps,
        "ewc_now": ewc_now,
    }


# ---------------------------------------------------------------------
# Whole-curriculum runner
# ---------------------------------------------------------------------


def run_classification_curriculum(
    net: TrioronNetwork,
    label: str,
    *,
    do_growth: bool,
    train_views: Sequence[TaskDataView],
    eval_views: Sequence[TaskDataView],
    task_class_lists: Sequence[Sequence[int]],
    n_steps_per_task: int,
    m_max_bytes: int = M_MAX_BYTES_UNCAPPED,
) -> Dict[str, object]:
    K = len(train_views)
    initial_n_params = net.n_parameters()
    initial_arch = tuple(net.n_nodes_per_layer())
    print(f"\n[{label}] curriculum start — arch {net.n_nodes_per_layer()}  "
          f"params {net.n_parameters()}  growth={do_growth}  "
          f"K={K} steps/task={n_steps_per_task}")

    opt = optim.Adam(net.parameters(), lr=LR)
    trigger = (
        GrowthTrigger(
            latent_dim=net.layers[GROWTH_TARGET_LAYER_IDX].n_nodes,
            window=TRIGGER_W, eps_loss=TRIGGER_EPS_LOSS,
            eps_rank=TRIGGER_EPS_RANK,
            g_min=TRIGGER_G_MIN, g_max=TRIGGER_G_MAX,
        )
        if do_growth else None
    )
    ceilings = (
        CeilingsController(M_max_bytes=m_max_bytes, T_div_max_seconds=T_DIV_MAX_SECONDS)
        if do_growth else None
    )

    accuracy_matrix: List[List[float]] = [
        [float("nan")] * K for _ in range(K)
    ]
    all_fires: List[dict] = []
    n_params_per_task: List[int] = []
    total_pathology = 0
    cumulative_step = 0
    ewc_baseline = 0.0

    t0 = time.monotonic()
    for task_idx, train_view in enumerate(train_views):
        active = list(task_class_lists[task_idx])
        # Extend the head to accommodate this task's classes (idempotent
        # for task 0 since the head was already built at init_classes).
        head_size = net.layers[-1].n_nodes
        max_active = max(active)
        if max_active >= head_size:
            n_new = max_active - head_size + 1
            extend_output_head(net, n_new)
            opt = optim.Adam(net.parameters(), lr=LR)
            print(f"[{label}] head: extend by {n_new} → "
                  f"{net.layers[-1].n_nodes} outputs")

        print(f"\n[{label}] === Task {task_idx+1}/{K}: {train_view.name} "
              f"(active classes {active}) ===")
        if trigger is not None:
            trigger.reset()

        result = train_one_classification_task(
            net, task_idx, train_view, active,
            n_steps=n_steps_per_task, opt=opt,
            ewc_baseline=ewc_baseline, trigger=trigger,
            ceilings=ceilings, do_growth=do_growth,
            label=label, n_total_tasks=K,
            global_step_offset=cumulative_step,
        )
        opt = result["opt"]
        all_fires.extend(result["fires"])
        total_pathology += result["pathology_steps"]
        cumulative_step += n_steps_per_task

        consolidate_task(net, train_view, active)
        ewc_baseline = EWC_INTERTASK

        per_task_acc = evaluate_all_tasks(net, eval_views)
        for j in range(K):
            if j <= task_idx:
                accuracy_matrix[task_idx][j] = per_task_acc[j]
            else:
                accuracy_matrix[task_idx][j] = float("nan")
        avg_so_far = sum(accuracy_matrix[task_idx][: task_idx + 1]) / (task_idx + 1)
        n_params_per_task.append(net.n_parameters())
        print(f"[{label}] After task {task_idx+1}: own={per_task_acc[task_idx]:.4f}  "
              f"avg_to_date={avg_so_far:.4f}  arch={net.n_nodes_per_layer()} "
              f"params={net.n_parameters()}")

    elapsed = time.monotonic() - t0
    rep = summarize(accuracy_matrix, [v.name for v in eval_views])
    return {
        "label": label,
        "do_growth": do_growth,
        "initial_arch": initial_arch,
        "final_arch": tuple(net.n_nodes_per_layer()),
        "initial_n_params": initial_n_params,
        "final_n_params": net.n_parameters(),
        "accuracy_matrix": accuracy_matrix,
        "final_accuracy": rep.final_accuracy,
        "avg_forgetting": rep.avg_forgetting,
        "fires": all_fires,
        "n_params_per_task": n_params_per_task,
        "pathology_steps": total_pathology,
        "wall_clock_seconds": elapsed,
        "task_names": [v.name for v in eval_views],
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny budget for fast smoke test (200 steps/task).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--ewc-intertask", type=float, default=EWC_INTERTASK,
        help=f"EWC strength used between tasks (default {EWC_INTERTASK}). "
             f"Tune this for fan_in=784 + CE-loss; the bench_50task value "
             f"(1000) is too stiff and prevents new-task learning.",
    )
    parser.add_argument(
        "--ewc-stab-boost", type=float, default=EWC_STAB_BOOST,
        help="EWC strength used briefly after a division (irrelevant when "
             "do_growth=False).",
    )
    parser.add_argument(
        "--no-grown", action="store_true",
        help="Skip the grown arm (used by the EWC-strength sweep — only "
             "fixed_ewc behavior depends on the strength).",
    )
    parser.add_argument(
        "--label-suffix", default="",
        help="Appended to the per-arm label so multiple sweep runs can "
             "share an output CSV without clobbering.",
    )
    args = parser.parse_args(argv)

    # Mutate module-level EWC constants so the helpers (which read these
    # at call time) see the override. Cleaner than threading every
    # function through a config object.
    global EWC_INTERTASK, EWC_STAB_BOOST
    EWC_INTERTASK = float(args.ewc_intertask)
    EWC_STAB_BOOST = float(args.ewc_stab_boost)

    n_steps = 200 if args.smoke else N_STEPS_PER_TASK

    torch.manual_seed(args.seed)

    print("=" * 78)
    print("Trioron — bench_split_mnist (canary, fan_in=784)")
    print("=" * 78)
    print(f"Steps/task:        {n_steps}{' [SMOKE]' if args.smoke else ''}")
    print(f"H_init grown:      {H_INIT_GROWN}")
    print(f"H fixed:           {H_FIXED}")
    print(f"Growth target:     layer {GROWTH_TARGET_LAYER_IDX}")
    print(f"Seed:              {args.seed}")
    print()

    bundle = DatasetBundle(["mnist"], root=args.data_root)
    specs = split_mnist_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    # --- Grown ---
    torch.manual_seed(args.seed)
    grown_net = make_classifier(INPUT_DIM, H_INIT_GROWN, INIT_CLASSES)
    grown_result = run_classification_curriculum(
        grown_net, label="grown",
        do_growth=True,
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_steps_per_task=n_steps,
    )

    # --- Fixed-MLP + EWC ---
    torch.manual_seed(args.seed + 1)
    fixed_net = make_classifier(INPUT_DIM, H_FIXED, INIT_CLASSES)
    fixed_result = run_classification_curriculum(
        fixed_net, label=f"fixed_ewc_H{H_FIXED}",
        do_growth=False,
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists, n_steps_per_task=n_steps,
    )

    # --- Final report ---
    print()
    print("=" * 78)
    print("bench_split_mnist — Final Report")
    print("=" * 78)
    for r in (grown_result, fixed_result):
        print(f"\n[{r['label']}]")
        print(f"  arch:               {r['initial_arch']} → {r['final_arch']}")
        print(f"  params:             {r['initial_n_params']} → {r['final_n_params']}")
        n_alw = len([f for f in r["fires"] if f.get("allowed")])
        n_den = len([f for f in r["fires"] if not f.get("allowed")])
        print(f"  divisions:          {n_alw} allowed, {n_den} denied")
        print(f"  pathology steps:    {r['pathology_steps']}")
        print(f"  final accuracy:     {r['final_accuracy']:.4f}")
        print(f"  avg forgetting:     {r['avg_forgetting']:.4f}")
        print(f"  wall-clock:         {r['wall_clock_seconds']:.1f}s")
        # Per-task final accuracy row
        K = len(r["task_names"])
        final_row = r["accuracy_matrix"][K - 1]
        for nm, a in zip(r["task_names"], final_row):
            print(f"     {nm:<16s}  {a:.4f}")

    print()
    print("Headline:")
    print(f"  grown {grown_result['final_accuracy']:.4f}  vs  "
          f"fixed_ewc_H{H_FIXED} {fixed_result['final_accuracy']:.4f}")
    print()

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_split_mnist_log.csv")
    all_results = [grown_result, fixed_result]
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        K = len(specs)
        header = [
            "label", "initial_arch", "final_arch",
            "initial_n_params", "final_n_params",
            "wall_clock_seconds", "final_accuracy", "avg_forgetting",
            "n_divisions_allowed", "n_divisions_denied", "pathology_steps",
        ]
        for i in range(K):
            for j in range(K):
                header.append(f"A[{i+1}][{j+1}]")
        w.writerow(header)
        for r in all_results:
            n_alw_r = len([f for f in r["fires"] if f.get("allowed")])
            n_den_r = len([f for f in r["fires"] if not f.get("allowed")])
            row = [
                r["label"], str(r["initial_arch"]), str(r["final_arch"]),
                r["initial_n_params"], r["final_n_params"],
                f"{r['wall_clock_seconds']:.2f}",
                f"{r['final_accuracy']:.6f}", f"{r['avg_forgetting']:.6f}",
                n_alw_r, n_den_r, r["pathology_steps"],
            ]
            for i in range(K):
                for j in range(K):
                    v = r["accuracy_matrix"][i][j]
                    row.append("" if v != v else f"{v:.6f}")
            w.writerow(row)
    print(f"  log: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
