"""Chained-15 headline: MNIST → FashionMNIST → EMNIST-letters, hard param cap.

Tests the apoptosis claim: "trioron survives streams the baselines weren't
designed for, because dreaming reclaims substrate when capacity binds."

Curriculum:
    15 binary tasks total (5 per dataset). Global classes 0..29.
    Tasks 0-4   use MNIST classes 0..9               → global 0..9
    Tasks 5-9   use FashionMNIST classes 0..9        → global 10..19
    Tasks 10-14 use EMNIST-letters A..J (local 0..9) → global 20..29

KMNIST was planned for the third block but its torchvision mirror is dead;
EMNIST-letters fills the same role (different glyph distribution).

Architecture:
    784 → H_init=32 (grown) → H_init=32 (grown) → head (grows 2 → 30)

Trigger choice (per session decision: "Option B"):
    Trigger-driven growth is OFF. Each task tries to deterministically
    grow N_GROW_PER_TASK hidden nodes into layer 1 BEFORE training. The
    growth happens iff projected params after grow are <= cap; otherwise
    it's denied. This isolates the apoptosis-reclaim claim from the
    trigger-calibration question.

Arms:
    1. fixed_ewc           — H=64, no growth, no dream, EWC. The
                             matched-fixed baseline.
    2. grown_capped_no_dream  — start H=32, deterministic grow N=4 per
                             task, hard cap. Once cap binds, can't grow
                             more. No dreaming. Control for "what does
                             pure growth-under-cap look like".
    3. grown_capped_dream  — same growth + cap as (2), with dreaming
                             (replay → starve+apoptosis → purge) called
                             on every task end + IMMEDIATELY when growth
                             is denied. The protagonist.
    4. grown_uncapped_dream — same growth + dream as (3), no cap.
                             Capacity-control: shows what's possible
                             when substrate is unlimited.

Headline metric:
    Final accuracy + accuracy on tasks 10-14 (the late-stream KMNIST
    block where the cap should be binding). Side panel: per-task
    purge_count + apoptosis_event_count for arm (3).

Run:
    python3 -m experiments.bench_chained_15task               # full budget
    python3 -m experiments.bench_chained_15task --smoke       # 200 steps/task
    python3 -m experiments.bench_chained_15task --arms grown_capped_dream,fixed_ewc
"""
from __future__ import annotations
import argparse
import csv
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
from trioron.dreaming import (
    PurgeEvent,
    MergeEvent,
    apoptosis_decay,
    compress,
    purge,
)

from experiments.datasets import (
    DEFAULT_DATA_ROOT,
    DatasetBundle,
    TaskDataView,
    build_task_views,
    chained_15_specs,
)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

INPUT_DIM = 784
H_INIT_GROWN = 32
H_FIXED = 64
INIT_CLASSES = 2
GROWTH_TARGET_LAYER_IDX = 1   # second hidden — NOT the head
N_GROW_PER_TASK = 4           # deterministic per-task hidden growth

N_STEPS_PER_TASK = 1500
BATCH = 64
LR = 1e-3
SEED = 0

LAMBDA_FLOOR = 0.1
EWC_INTERTASK = 30.0          # tuned for fan_in=784 + CE; bench_50task used 1000
EWC_DREAM_STRENGTH = 30.0     # match intertask strength inside dreaming

# Cap calibration: fixed_ewc at H=64 with full head (30 outputs) sits at
# ~57k params. We pick the cap a touch above that so deterministic grown
# can actually exercise growth before binding, but well below where it
# could grow uncapped (which would be 92² + 815*92 + 30 ≈ 83k for
# H_final=92 = H_INIT + 15 tasks * N_GROW_PER_TASK).
# Cap calibration: tighter than the fixed_ewc baseline (~55k params at
# H=64) so growth saturates partway through MNIST and the dream-rescue
# path is actually exercised. ~35k params ≈ H=40 hidden in layer 1.
M_MAX_BYTES_CAPPED = 35_000 * 4     # ≈ 35k params at FP32 → 140 KB
M_MAX_BYTES_UNCAPPED = 2 * 1024 ** 3

# Dreaming-block configuration — substrate-preserving compression with
# apoptosis spike, plus aggressive purge so room actually frees.
DREAM_REPLAY_FRACTION = 0.25
DREAM_REPLAY_STEPS = 50       # smaller than bench_50task's 200 because the
                               # task data here is bigger and replay is
                               # called more often (post-task + on-deny)
DREAM_REPLAY_BATCH = BATCH
DREAM_AC_THRESHOLD = 0.85
DREAM_PROBE_BATCH_SIZE = 256
DREAM_COMPRESSION_ACTION = "starve"
DREAM_MAX_DOWNSCALES_PER_LAYER = 1
DREAM_STARVATION_ALPHA = 0.7
DREAM_STARVATION_FLOOR = 1e-3
DREAM_APOPTOSIS_ON = True
DREAM_APOPTOSIS_SPIKE_INIT = 0.8
DREAM_APOPTOSIS_DECAY_RATE = 0.7

# Purge needs a usable utility threshold. The contrastive benches kept
# this at 1e-3 because contributions there were small. For CE-on-MNIST
# the per-batch contributions are larger; raising the threshold means
# starved units (whose effective contribution decays toward 0) will
# actually be reclaimed.
DREAM_U_THRESHOLD = 0.01
DREAM_PURGE_SKIP_OUTPUT = True

LOG_EVERY = 500


# ---------------------------------------------------------------------
# Network construction + forward helpers
# ---------------------------------------------------------------------


def make_classifier(input_dim: int, hidden: int, init_classes: int) -> TrioronNetwork:
    return TrioronNetwork(
        [
            (input_dim, hidden, "relu"),
            (hidden, hidden, "relu"),
            (hidden, init_classes, "linear"),
        ]
    )


# ---------------------------------------------------------------------
# Cap math (inline — bypasses CeilingsController whose arrest is sticky)
# ---------------------------------------------------------------------


def projected_params_after_grow(net: TrioronNetwork, target_layer_idx: int) -> int:
    """Predict net.n_parameters() after one grow_layer(target_layer_idx)."""
    target = net.layers[target_layer_idx]
    delta = target.fan_in + 1   # +1 W row + +1 b
    if target_layer_idx + 1 < len(net.layers):
        delta += net.layers[target_layer_idx + 1].n_nodes  # +1 W col on next
    return net.n_parameters() + delta


def try_grow_one(
    net: TrioronNetwork,
    target_layer_idx: int,
    cap_bytes: int,
    task_idx: int,
    bytes_per_param: int = 4,
) -> Tuple[bool, str]:
    """Attempt one grow_layer call iff projected params * 4 <= cap_bytes.
    Returns (allowed, reason). Bypasses CeilingsController whose arrest
    flag prevents resumed growth after dreaming-driven reclaim.
    """
    projected_bytes = projected_params_after_grow(
        net, target_layer_idx,
    ) * bytes_per_param
    if projected_bytes > cap_bytes:
        return False, f"cap_exceeded(projected={projected_bytes}B > cap={cap_bytes}B)"
    net.grow_layer(target_layer_idx, init_vec=None, task_idx=task_idx)
    return True, "ok"


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
# Utility-update during training (needed so purge has a real signal)
# ---------------------------------------------------------------------


def utility_contributions(net: TrioronNetwork, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """|activation * grad| averaged over batch for layer `layer_idx`.

    We hook into the forward pass after .backward() — this is similar
    to the pruner's combined-mode signal. For dreaming-phase purge the
    distinction (vs strict |a·g|) doesn't matter because we just need
    "low utility ⇒ unused".
    """
    layer = net.layers[layer_idx]
    # Layer's outputs are not retained on default; instead approximate
    # contribution as |W * grad_W| row-sum, which is a per-node summary
    # of "how much do we depend on this node's incoming weights now".
    if layer.W.grad is None:
        return torch.zeros(layer.n_nodes, device=layer.W.device)
    contrib = (layer.W.detach().abs() * layer.W.grad.detach().abs()).sum(dim=1)
    return contrib


def update_layer_utilities(net: TrioronNetwork) -> None:
    """Capture a per-node utility update on every layer based on the
    current gradients. Call after .backward(), before optimizer.step().
    Cheap — just a row-sum on each layer's W * W.grad.
    """
    for L in range(len(net.layers)):
        layer = net.layers[L]
        if layer.W.grad is None:
            continue
        contrib = (layer.W.detach().abs() * layer.W.grad.detach().abs()).sum(dim=1)
        layer.update_utility(contrib)


# ---------------------------------------------------------------------
# Classification-shaped dreaming block
# ---------------------------------------------------------------------


def _classification_replay(
    net: TrioronNetwork,
    past_views: Sequence[TaskDataView],
    past_active_classes: Sequence[Sequence[int]],
    *,
    fraction: float,
    n_steps_per_task: int,
    batch: int,
    lr: float,
    ewc_strength: float,
    rng: random.Random,
) -> Tuple[float, float, int, int]:
    """CE-shaped analog of dreaming.replay. Returns
    (avg_loss_before, avg_loss_after, n_tasks_sampled, total_steps)."""
    if not past_views:
        return (0.0, 0.0, 0, 0)
    n = len(past_views)
    k = max(1, int(round(fraction * n)))
    idxs = rng.sample(range(n), k=min(k, n))

    def _avg_loss() -> float:
        net.eval()
        total = 0.0
        with torch.no_grad():
            for i in idxs:
                v = past_views[i]
                active = list(past_active_classes[i])
                x, y = v.sample(batch)
                total += float(masked_cross_entropy(net(x), y, active).item())
        net.train()
        return total / len(idxs)

    loss_before = _avg_loss()
    opt = optim.Adam(net.parameters(), lr=lr)
    total_steps = 0
    for i in idxs:
        v = past_views[i]
        active = list(past_active_classes[i])
        for _ in range(n_steps_per_task):
            x, y = v.sample(batch)
            l_task = masked_cross_entropy(net(x), y, active)
            l = (l_task + ewc_strength * net.ewc_penalty()
                 if ewc_strength > 0 else l_task)
            opt.zero_grad()
            l.backward()
            update_layer_utilities(net)  # keep purge signal warm
            opt.step()
            total_steps += 1
    loss_after = _avg_loss()
    return (loss_before, loss_after, len(idxs), total_steps)


def _build_classification_probe(
    past_views: Sequence[TaskDataView],
    probe_batch_size: int,
    rng: random.Random,
) -> Optional[torch.Tensor]:
    if not past_views:
        return None
    per = max(1, probe_batch_size // len(past_views))
    chunks: List[torch.Tensor] = []
    for v in past_views:
        x, _ = v.sample(per, generator=None)
        chunks.append(x)
    out = torch.cat(chunks, dim=0)
    if out.shape[0] > probe_batch_size:
        out = out[:probe_batch_size]
    return out


def classification_dreaming_block(
    net: TrioronNetwork,
    past_views: Sequence[TaskDataView],
    past_active_classes: Sequence[Sequence[int]],
    *,
    rng: random.Random,
    mode: str,
) -> Dict[str, object]:
    """CE-shaped dreaming. Two modes:

    mode='replay_only' — used post-task to keep prior memories warm.
        Runs apoptosis_decay (so any spike from a prior block fades)
        then replay. NO compress, NO purge — substrate is unchanged.
        This is the "consolidation rest" mode.

    mode='reclaim' — used on growth-denial to free substrate.
        replay + compress(starve+apoptosis) + purge restricted to the
        growth-target layer. Purge u_threshold is high enough that
        starvation-decayed units actually vacate; layer 0 (the 784-fan
        adapter) is NEVER purged because dropping a layer-0 unit
        wipes 784 weights of feature-detector capacity for prior tasks.

    Returns a flat dict. Caller MUST rebuild the optimizer if
    `n_purges > 0` (purge replaces Parameter objects).
    """
    if mode not in ("replay_only", "reclaim"):
        raise ValueError(f"mode must be 'replay_only' or 'reclaim', got {mode!r}")

    n_before = net.n_parameters()
    arch_before = tuple(net.n_nodes_per_layer())

    if DREAM_APOPTOSIS_ON:
        apoptosis_decay(net, decay_rate=DREAM_APOPTOSIS_DECAY_RATE)

    loss_before, loss_after, n_tasks, n_steps = _classification_replay(
        net, past_views, past_active_classes,
        fraction=DREAM_REPLAY_FRACTION,
        n_steps_per_task=DREAM_REPLAY_STEPS,
        batch=DREAM_REPLAY_BATCH,
        lr=LR,
        ewc_strength=EWC_DREAM_STRENGTH,
        rng=rng,
    )

    merges: List[MergeEvent] = []
    purges: List[PurgeEvent] = []

    if mode == "reclaim":
        probe = _build_classification_probe(
            past_views, DREAM_PROBE_BATCH_SIZE, rng,
        )
        if probe is not None:
            merges = compress(
                net,
                layer_idxs=[GROWTH_TARGET_LAYER_IDX],   # only the growth target
                redundancy_signal="activation",
                probe_batch=probe,
                ac_threshold=DREAM_AC_THRESHOLD,
                compression_action=DREAM_COMPRESSION_ACTION,
                max_downscales_per_layer=DREAM_MAX_DOWNSCALES_PER_LAYER,
                starvation_alpha=DREAM_STARVATION_ALPHA,
                starvation_floor=DREAM_STARVATION_FLOOR,
                apoptosis_on=DREAM_APOPTOSIS_ON,
                apoptosis_spike_init=DREAM_APOPTOSIS_SPIKE_INIT,
                skip_output_layer=True,
            )
        # Restrict purge to the growth-target layer ONLY. Layer 0
        # (the input adapter) and the head (output) stay untouched.
        purges = purge(
            net,
            layer_idxs=[GROWTH_TARGET_LAYER_IDX],
            u_threshold=DREAM_U_THRESHOLD,
            skip_output_layer=False,  # we already constrain via layer_idxs
        )

    return {
        "n_params_before": n_before,
        "n_params_after": net.n_parameters(),
        "arch_before": arch_before,
        "arch_after": tuple(net.n_nodes_per_layer()),
        "replay_loss_before": loss_before,
        "replay_loss_after": loss_after,
        "n_replay_tasks": n_tasks,
        "n_replay_steps": n_steps,
        "n_merges": len(merges),
        "n_purges": len(purges),
        "n_latched": sum(1 for m in merges if m.victim_latched),
        "mode": mode,
    }


# ---------------------------------------------------------------------
# Per-task training loop
# ---------------------------------------------------------------------


def train_one_task(
    net: TrioronNetwork,
    task_idx: int,
    train_view: TaskDataView,
    active_classes: Sequence[int],
    n_steps: int,
    opt: optim.Optimizer,
    *,
    ewc_baseline: float,
    label: str,
    n_total_tasks: int,
) -> optim.Optimizer:
    active = list(active_classes)
    for step in range(n_steps):
        x, y_global = train_view.sample(BATCH)
        logits = net(x)
        l_task = masked_cross_entropy(logits, y_global, active_classes=active)
        l = (l_task + ewc_baseline * net.ewc_penalty()
             if ewc_baseline > 0 else l_task)
        opt.zero_grad()
        l.backward()
        update_layer_utilities(net)
        opt.step()
        if step == 0 or (step + 1) % LOG_EVERY == 0 or step == n_steps - 1:
            print(f"  [{label}] task {task_idx+1}/{n_total_tasks} ({train_view.name}) "
                  f"step {step:5d}  loss {l_task.item():.4f}  "
                  f"arch {net.n_nodes_per_layer()}  params {net.n_parameters()}")
    return opt


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
# Whole-curriculum runner
# ---------------------------------------------------------------------


def run_chained_curriculum(
    net: TrioronNetwork,
    label: str,
    *,
    do_growth: bool,
    do_dream: bool,
    cap_bytes: int,
    n_grow_per_task: int,
    train_views: Sequence[TaskDataView],
    eval_views: Sequence[TaskDataView],
    task_class_lists: Sequence[Sequence[int]],
    n_steps_per_task: int,
    rng_seed: int,
) -> Dict[str, object]:
    K = len(train_views)
    initial_n_params = net.n_parameters()
    initial_arch = tuple(net.n_nodes_per_layer())
    rng = random.Random(rng_seed)
    print(f"\n[{label}] start — arch {initial_arch}  params {initial_n_params}  "
          f"growth={do_growth} dream={do_dream}  "
          f"cap_bytes={cap_bytes}  K={K}  steps/task={n_steps_per_task}")

    opt = optim.Adam(net.parameters(), lr=LR)
    accuracy_matrix: List[List[float]] = [[float("nan")] * K for _ in range(K)]
    per_task_log: List[Dict[str, object]] = []
    n_params_per_task: List[int] = []
    cumulative_grows = 0
    cumulative_grows_denied = 0
    cumulative_purges = 0
    cumulative_latched = 0
    ewc_baseline = 0.0

    t0 = time.monotonic()
    for task_idx, train_view in enumerate(train_views):
        active = list(task_class_lists[task_idx])

        # 1. Extend output head to fit this task's classes.
        head_size = net.layers[-1].n_nodes
        max_active = max(active)
        if max_active >= head_size:
            n_new_head = max_active - head_size + 1
            extend_output_head(net, n_new_head)
            opt = optim.Adam(net.parameters(), lr=LR)

        # 2. Deterministic per-task hidden growth (with optional dream-rescue).
        attempted = 0
        allowed = 0
        denied = 0
        if do_growth and n_grow_per_task > 0:
            for _ in range(n_grow_per_task):
                attempted += 1
                ok, reason = try_grow_one(
                    net, GROWTH_TARGET_LAYER_IDX, cap_bytes, task_idx,
                )
                if ok:
                    allowed += 1
                else:
                    denied += 1
                    if do_dream:
                        # Dream-rescue: try to free room then retry once.
                        past_views = train_views[:task_idx]
                        past_actives = task_class_lists[:task_idx]
                        rescue = classification_dreaming_block(
                            net, past_views, past_actives,
                            rng=rng, mode="reclaim",
                        )
                        cumulative_purges += rescue["n_purges"]
                        cumulative_latched += rescue["n_latched"]
                        opt = optim.Adam(net.parameters(), lr=LR)
                        ok2, reason2 = try_grow_one(
                            net, GROWTH_TARGET_LAYER_IDX, cap_bytes, task_idx,
                        )
                        if ok2:
                            allowed += 1
                            denied -= 1   # not actually denied
                            print(f"  [{label}] dream-rescue freed room: "
                                  f"purges={rescue['n_purges']} → grow OK")
                        else:
                            print(f"  [{label}] dream-rescue insufficient: "
                                  f"purges={rescue['n_purges']}; growth still denied "
                                  f"({reason2})")
                            break  # no point trying more grows this task
                    else:
                        # No dreaming → cap binds, accept partial growth.
                        break
        cumulative_grows += allowed
        cumulative_grows_denied += denied

        if allowed > 0:
            opt = optim.Adam(net.parameters(), lr=LR)

        print(f"\n[{label}] === Task {task_idx+1}/{K}: {train_view.name} "
              f"(active {active}) growth: {allowed}/{attempted} allowed, "
              f"{denied} denied  arch={net.n_nodes_per_layer()} "
              f"params={net.n_parameters()} ===")

        # 3. Train.
        opt = train_one_task(
            net, task_idx, train_view, active,
            n_steps=n_steps_per_task, opt=opt,
            ewc_baseline=ewc_baseline,
            label=label, n_total_tasks=K,
        )

        # 4. Consolidate.
        consolidate_task(net, train_view, active)
        ewc_baseline = EWC_INTERTASK

        # 5. Post-task dreaming = REPLAY ONLY (keeps memories warm; does
        #    NOT touch substrate). Structural reclamation is reserved
        #    for the on-deny dream-rescue above.
        dream_rep = {"n_merges": 0, "n_purges": 0, "n_latched": 0,
                     "n_params_before": net.n_parameters(),
                     "n_params_after": net.n_parameters(),
                     "replay_loss_before": 0.0, "replay_loss_after": 0.0,
                     "n_replay_tasks": 0}
        if do_dream:
            past_views = train_views[: task_idx + 1]
            past_actives = task_class_lists[: task_idx + 1]
            dream_rep = classification_dreaming_block(
                net, past_views, past_actives,
                rng=rng, mode="replay_only",
            )
            # No purges in replay_only mode, but rebuild optimizer
            # defensively in case future modes add structural changes.
            opt = optim.Adam(net.parameters(), lr=LR)
            print(f"  [{label}] post-task DREAM: replay "
                  f"{dream_rep['replay_loss_before']:.4f}→"
                  f"{dream_rep['replay_loss_after']:.4f} on "
                  f"{dream_rep['n_replay_tasks']}p; "
                  f"merges={dream_rep['n_merges']} purges={dream_rep['n_purges']} "
                  f"latched={dream_rep['n_latched']} → "
                  f"arch {net.n_nodes_per_layer()} "
                  f"({dream_rep['n_params_before']}→{dream_rep['n_params_after']} params)")

        # 6. Eval all completed tasks.
        per_task_acc = evaluate_all_tasks(net, eval_views)
        for j in range(K):
            if j <= task_idx:
                accuracy_matrix[task_idx][j] = per_task_acc[j]
            else:
                accuracy_matrix[task_idx][j] = float("nan")
        avg_so_far = sum(accuracy_matrix[task_idx][: task_idx + 1]) / (task_idx + 1)
        n_params_per_task.append(net.n_parameters())
        per_task_log.append({
            "task_idx": task_idx,
            "task_name": train_view.name,
            "active_classes": active,
            "n_params_after": net.n_parameters(),
            "arch_after": tuple(net.n_nodes_per_layer()),
            "grows_allowed": allowed,
            "grows_denied": denied,
            "dream_merges": dream_rep["n_merges"],
            "dream_purges": dream_rep["n_purges"],
            "dream_latched": dream_rep["n_latched"],
            "own_acc": per_task_acc[task_idx],
            "avg_to_date": avg_so_far,
        })
        print(f"[{label}] After task {task_idx+1}: own={per_task_acc[task_idx]:.4f}  "
              f"avg_to_date={avg_so_far:.4f}  arch={net.n_nodes_per_layer()} "
              f"params={net.n_parameters()}  cum_grows={cumulative_grows} "
              f"cum_denied={cumulative_grows_denied} cum_purges={cumulative_purges} "
              f"cum_latched={cumulative_latched}")

    elapsed = time.monotonic() - t0
    rep = summarize(accuracy_matrix, [v.name for v in eval_views])
    return {
        "label": label,
        "do_growth": do_growth,
        "do_dream": do_dream,
        "cap_bytes": cap_bytes,
        "initial_arch": initial_arch,
        "final_arch": tuple(net.n_nodes_per_layer()),
        "initial_n_params": initial_n_params,
        "final_n_params": net.n_parameters(),
        "accuracy_matrix": accuracy_matrix,
        "final_accuracy": rep.final_accuracy,
        "avg_forgetting": rep.avg_forgetting,
        "n_params_per_task": n_params_per_task,
        "cumulative_grows_allowed": cumulative_grows,
        "cumulative_grows_denied": cumulative_grows_denied,
        "cumulative_purges": cumulative_purges,
        "cumulative_latched": cumulative_latched,
        "per_task_log": per_task_log,
        "task_names": [v.name for v in eval_views],
        "wall_clock_seconds": elapsed,
    }


# ---------------------------------------------------------------------
# Arm dispatch
# ---------------------------------------------------------------------


ARM_DEFINITIONS = {
    "fixed_ewc": {
        "h_init": H_FIXED, "do_growth": False, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_UNCAPPED,
    },
    "grown_capped_no_dream": {
        "h_init": H_INIT_GROWN, "do_growth": True, "do_dream": False,
        "cap_bytes": M_MAX_BYTES_CAPPED,
    },
    "grown_capped_dream": {
        "h_init": H_INIT_GROWN, "do_growth": True, "do_dream": True,
        "cap_bytes": M_MAX_BYTES_CAPPED,
    },
    "grown_uncapped_dream": {
        "h_init": H_INIT_GROWN, "do_growth": True, "do_dream": True,
        "cap_bytes": M_MAX_BYTES_UNCAPPED,
    },
}

DEFAULT_ARMS = list(ARM_DEFINITIONS.keys())


def run_arm(
    arm: str,
    *,
    seed: int,
    n_steps_per_task: int,
    train_views,
    eval_views,
    task_class_lists,
) -> Dict[str, object]:
    cfg = ARM_DEFINITIONS[arm]
    torch.manual_seed(seed)
    net = make_classifier(INPUT_DIM, cfg["h_init"], INIT_CLASSES)
    return run_chained_curriculum(
        net, label=arm,
        do_growth=cfg["do_growth"], do_dream=cfg["do_dream"],
        cap_bytes=cfg["cap_bytes"], n_grow_per_task=N_GROW_PER_TASK,
        train_views=train_views, eval_views=eval_views,
        task_class_lists=task_class_lists,
        n_steps_per_task=n_steps_per_task,
        rng_seed=seed + 7919,
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def _phase_means(M: List[List[float]], task_names: Sequence[str]) -> Dict[str, float]:
    """Final-row accuracy averaged within each chained block."""
    K = len(M)
    if K == 0:
        return {}
    final_row = M[K - 1]
    out: Dict[str, float] = {}
    for prefix, block_label in [
        ("mnist", "phase1_mnist"),
        ("fashion_mnist", "phase2_fashion"),
        ("emnist_letters", "phase3_emnist"),
    ]:
        block_idxs = [j for j, nm in enumerate(task_names) if nm.startswith(prefix)]
        if block_idxs:
            out[block_label] = sum(final_row[j] for j in block_idxs) / len(block_idxs)
    return out


def report(results: Sequence[Dict[str, object]]) -> None:
    print()
    print("=" * 78)
    print("bench_chained_15task — Final Report")
    print("=" * 78)
    for r in results:
        K = len(r["task_names"])
        print(f"\n[{r['label']}]")
        print(f"  arch:               {r['initial_arch']} → {r['final_arch']}")
        print(f"  params:             {r['initial_n_params']} → {r['final_n_params']}")
        print(f"  cap_bytes:          {r['cap_bytes']:_}")
        print(f"  cum grows allowed:  {r['cumulative_grows_allowed']}")
        print(f"  cum grows denied:   {r['cumulative_grows_denied']}")
        print(f"  cum dream purges:   {r['cumulative_purges']}")
        print(f"  cum dream latched:  {r['cumulative_latched']}")
        print(f"  final accuracy:     {r['final_accuracy']:.4f}")
        print(f"  avg forgetting:     {r['avg_forgetting']:.4f}")
        print(f"  wall-clock:         {r['wall_clock_seconds']:.1f}s")
        phase_means = _phase_means(r["accuracy_matrix"], r["task_names"])
        for nm, v in phase_means.items():
            print(f"     {nm:<20s} {v:.4f}")

    print()
    print("Headline (final accuracy across arms):")
    for r in results:
        print(f"  {r['label']:<28s}  {r['final_accuracy']:.4f}  "
              f"(forgetting {r['avg_forgetting']:+.4f})")
    print()


def write_csv(results: Sequence[Dict[str, object]], csv_path: str) -> None:
    K = len(results[0]["task_names"]) if results else 0
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        header = [
            "label", "do_growth", "do_dream", "cap_bytes",
            "initial_arch", "final_arch",
            "initial_n_params", "final_n_params",
            "wall_clock_seconds", "final_accuracy", "avg_forgetting",
            "cum_grows_allowed", "cum_grows_denied",
            "cum_purges", "cum_latched",
        ]
        for i in range(K):
            for j in range(K):
                header.append(f"A[{i+1}][{j+1}]")
        w.writerow(header)
        for r in results:
            row = [
                r["label"], r["do_growth"], r["do_dream"], r["cap_bytes"],
                str(r["initial_arch"]), str(r["final_arch"]),
                r["initial_n_params"], r["final_n_params"],
                f"{r['wall_clock_seconds']:.2f}",
                f"{r['final_accuracy']:.6f}", f"{r['avg_forgetting']:.6f}",
                r["cumulative_grows_allowed"], r["cumulative_grows_denied"],
                r["cumulative_purges"], r["cumulative_latched"],
            ]
            for i in range(K):
                for j in range(K):
                    v = r["accuracy_matrix"][i][j]
                    row.append("" if v != v else f"{v:.6f}")
            w.writerow(row)
    print(f"  log: {csv_path}")


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
        "--arms", default=",".join(DEFAULT_ARMS),
        help=f"Comma-separated subset of {DEFAULT_ARMS}",
    )
    parser.add_argument(
        "--csv", default="bench_chained_15task_log.csv",
        help="Output CSV filename (under outputs/).",
    )
    args = parser.parse_args(argv)

    n_steps = 200 if args.smoke else N_STEPS_PER_TASK
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_DEFINITIONS:
            raise SystemExit(
                f"Unknown arm {a!r}. Available: {list(ARM_DEFINITIONS)}"
            )

    print("=" * 78)
    print("Trioron — bench_chained_15task: MNIST → FashionMNIST → EMNIST-letters")
    print("=" * 78)
    print(f"Steps/task:         {n_steps}{' [SMOKE]' if args.smoke else ''}")
    print(f"H_init grown:       {H_INIT_GROWN}")
    print(f"H fixed:            {H_FIXED}")
    print(f"N_grow_per_task:    {N_GROW_PER_TASK}")
    print(f"M_max_bytes capped: {M_MAX_BYTES_CAPPED:_}")
    print(f"EWC intertask:      {EWC_INTERTASK}")
    print(f"Arms:               {arms}")
    print(f"Seed:               {args.seed}")
    print()

    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"], root=args.data_root)
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]

    results: List[Dict[str, object]] = []
    for arm in arms:
        r = run_arm(
            arm,
            seed=args.seed + (hash(arm) % 7919),
            n_steps_per_task=n_steps,
            train_views=train_views, eval_views=eval_views,
            task_class_lists=task_class_lists,
        )
        results.append(r)

    report(results)

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.csv)
    write_csv(results, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
