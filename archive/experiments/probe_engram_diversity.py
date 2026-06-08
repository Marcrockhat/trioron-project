"""Probe: why does raw rehearsal beat engrams 5x?

Hypothesis space:
  H1 (diversity)     — single engram per class is too narrow; K=100 should
                       close the gap to rehearsal-100.
  H2 (prototype kind) — gradient-ascent prototypes are adversarial /
                       degenerate; even at K=100, engrams won't match
                       real samples.
  H3 (loss form)     — hard CE on real samples works because it covers
                       all output dims; LwF distillation alone is weaker.

Experimental matrix (9 conditions on a chained-5-task MNIST):
  Real-K   (K=1, 10, 100)  — store K random training samples per class,
                             mix into batches, hard CE on all-classes-seen
  Engram-K (K=1, 10, 100)  — gradient-ascent K prototypes per class
                             through anchored network, mix into batches,
                             hard CE on all-classes-seen
  Engram-K-LwF (K=1,10,100) — same engrams, but KL-distill live↔anchor
                             on old-class columns instead of hard CE

Same training budget, network, optimizer, seed across all conditions.
Only the rehearsal mechanism varies.

Reads ~5 min on a CPU. No external dependencies beyond what bench already
uses. Run from project root:

    PYTHONPATH=$(pwd) python3 experiments/probe_engram_diversity.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from trioron.network import TrioronNetwork
from trioron.classification import masked_cross_entropy
from experiments.datasets import (
    DatasetBundle, build_task_views, TaskDataView,
    split_mnist_specs,
)


# --------------------------------------------------------------------- config
INPUT_DIM = 784
HIDDEN1 = 64           # smaller than chained-15 to keep probe fast
HIDDEN2 = 32
N_TASKS = 5
EPOCHS_PER_TASK = 4
BATCH = 64
LR = 1e-3
EWC_INTERTASK = 30.0
LAMBDA_FLOOR = 1e-3
SEED = 0

# Engram gradient-ascent hyperparams (same as bench)
GA_STEPS = 80
GA_LR = 0.05
GA_L2 = 1e-3
GA_INIT_NOISE_SCALE = 0.1
GA_CLIP = (0.0, 1.0)

# LwF hyperparams (same as bench)
LWF_T = 2.0


# --------------------------------------------------------------------- helpers


def make_net() -> TrioronNetwork:
    return TrioronNetwork([
        (INPUT_DIM, HIDDEN1, "relu"),
        (HIDDEN1, HIDDEN2, "relu"),
        (HIDDEN2, 2, "linear"),
    ])


def consolidate(net: TrioronNetwork, view: TaskDataView, active: List[int]) -> None:
    """Quick consolidate: estimate Fisher on this task, refresh λ, anchor."""
    # one-batch Fisher proxy is enough for a probe
    net.zero_grad()
    x, y = view.sample(256)
    logits = net(x)
    l = masked_cross_entropy(logits, y, active_classes=active)
    l.backward()
    net.update_fisher_all()
    net.update_lambda_all()
    with torch.no_grad():
        for layer in net.layers:
            layer.lam.clamp_(min=LAMBDA_FLOOR)
    net.anchor_all()


def extend_head(net: TrioronNetwork, n_new: int) -> None:
    head = net.layers[-1]
    for _ in range(n_new):
        head.grow_node(init_vec=None, task_idx=0)


def build_engrams(
    net: TrioronNetwork,
    classes: List[int],
    K: int,
) -> Dict[int, torch.Tensor]:
    """Return {class_idx -> tensor of shape (K, INPUT_DIM)} via grad ascent
    through forward_with_anchors_grad. Each of the K prototypes per class
    starts from a different random initialization."""
    out: Dict[int, torch.Tensor] = {}
    for c in classes:
        rows = []
        for _k in range(K):
            x = torch.empty(INPUT_DIM).uniform_(
                GA_CLIP[0],
                GA_CLIP[0] + GA_INIT_NOISE_SCALE * (GA_CLIP[1] - GA_CLIP[0]),
            )
            x.requires_grad_(True)
            for _step in range(GA_STEPS):
                logits = net.forward_with_anchors_grad(x.unsqueeze(0))
                if c >= logits.shape[1]:
                    break
                loss = -logits[0, c] + GA_L2 * x.pow(2).sum()
                if x.grad is not None:
                    x.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    x.data = (
                        (x.data - GA_LR * x.grad).clamp_(*GA_CLIP)
                    )
            rows.append(x.detach().clone())
        out[int(c)] = torch.stack(rows, dim=0)
    return out


def sample_real(view: TaskDataView, classes: List[int], K: int) -> Dict[int, torch.Tensor]:
    """Return {class_idx -> tensor of shape (K, INPUT_DIM)} of real samples."""
    out: Dict[int, torch.Tensor] = {}
    x_all, y_all = view.all_examples()
    for c in classes:
        mask = (y_all == c)
        x_c = x_all[mask]
        if x_c.shape[0] == 0:
            continue
        idx = torch.randperm(x_c.shape[0])[: K]
        out[int(c)] = x_c[idx].clone()
    return out


def merge_buffers(buffers: List[Dict[int, torch.Tensor]]) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for b in buffers:
        for c, x in b.items():
            out[c] = x
    return out


def sample_from_buffer(
    buf: Dict[int, torch.Tensor], n: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample `n` rows uniformly across all classes in buf."""
    if not buf:
        return None, None
    classes = list(buf.keys())
    cls_choice = torch.randint(0, len(classes), (n,))
    rows = []; ys = []
    for i in range(n):
        c = classes[int(cls_choice[i])]
        bank = buf[c]
        ridx = int(torch.randint(0, bank.shape[0], (1,)).item())
        rows.append(bank[ridx])
        ys.append(c)
    return torch.stack(rows, dim=0), torch.tensor(ys, dtype=torch.long)


# --------------------------------------------------------------------- training


def train_one_task(
    net: TrioronNetwork,
    view: TaskDataView,
    active: List[int],
    *,
    n_epochs: int,
    opt: optim.Optimizer,
    ewc_baseline: float,
    rehearsal_buf: Optional[Dict[int, torch.Tensor]] = None,
    rehearsal_loss: str = "ce",   # "ce" or "lwf"
    old_classes: Optional[Sequence[int]] = None,
) -> None:
    head_size = net.layers[-1].n_nodes
    all_seen = list(range(head_size))
    for _epoch in range(n_epochs):
        for x, y in view.iter_epoch(BATCH):
            logits = net(x)
            l_task = masked_cross_entropy(logits, y, active_classes=active)
            l_total = l_task

            if rehearsal_buf is not None and old_classes:
                x_r, y_r = sample_from_buffer(rehearsal_buf, BATCH)
                if x_r is not None:
                    if rehearsal_loss == "ce":
                        logits_r = net(x_r)
                        l_r = masked_cross_entropy(
                            logits_r, y_r, active_classes=all_seen,
                        )
                        l_total = l_total + l_r
                    elif rehearsal_loss == "lwf":
                        logits_r_live = net(x_r)
                        with torch.no_grad():
                            logits_r_anchor = net.forward_with_anchors(x_r)
                        old_idx = torch.as_tensor(
                            list(old_classes), dtype=torch.long,
                        )
                        in_range = old_idx[old_idx < logits_r_live.shape[1]]
                        if in_range.numel() > 1:
                            T = LWF_T
                            z_a = logits_r_anchor.index_select(1, in_range)
                            z_l = logits_r_live.index_select(1, in_range)
                            p_a = F.softmax(z_a / T, dim=1)
                            log_p_l = F.log_softmax(z_l / T, dim=1)
                            l_r = F.kl_div(
                                log_p_l, p_a, reduction="batchmean",
                            ) * (T * T)
                            l_total = l_total + l_r
                    else:
                        raise ValueError(f"unknown rehearsal_loss={rehearsal_loss}")

            if ewc_baseline > 0:
                l_total = l_total + ewc_baseline * net.ewc_penalty()
            opt.zero_grad()
            l_total.backward()
            opt.step()


def evaluate_full(
    net: TrioronNetwork,
    eval_views: List[TaskDataView],
    seen_classes: List[int],
) -> Dict[str, float]:
    net.eval()
    full_correct = 0; full_total = 0
    task_correct = 0; task_total = 0
    seen_t = torch.tensor(seen_classes, dtype=torch.long)
    with torch.no_grad():
        for tv in eval_views:
            tv_classes = list(tv.global_classes)
            tv_t = torch.tensor(tv_classes, dtype=torch.long)
            x, y = tv.all_examples()
            logits = net(x)
            # full-softmax over all seen
            logits_seen = logits.index_select(1, seen_t)
            pred_full = seen_t[logits_seen.argmax(dim=1)]
            full_correct += int((pred_full == y).sum().item())
            full_total += int(y.shape[0])
            # task-aware
            logits_task = logits.index_select(1, tv_t)
            pred_task = tv_t[logits_task.argmax(dim=1)]
            task_correct += int((pred_task == y).sum().item())
            task_total += int(y.shape[0])
    net.train()
    return {
        "full": full_correct / max(1, full_total),
        "task": task_correct / max(1, task_total),
    }


# --------------------------------------------------------------------- driver


@dataclass
class Condition:
    name: str
    kind: str       # "real" or "engram"
    K: int
    loss: str       # "ce" or "lwf"


def run_condition(
    cond: Condition,
    train_views: List[TaskDataView],
    eval_views: List[TaskDataView],
    task_class_lists: List[List[int]],
) -> Dict[str, float]:
    torch.manual_seed(SEED)
    net = make_net()
    opt = optim.Adam(net.parameters(), lr=LR)
    buffers: List[Dict[int, torch.Tensor]] = []
    ewc_baseline = 0.0
    seen: List[int] = []
    for k in range(N_TASKS):
        active = list(task_class_lists[k])
        # extend head if needed
        if max(active) >= net.layers[-1].n_nodes:
            extend_head(net, max(active) - net.layers[-1].n_nodes + 1)
            opt = optim.Adam(net.parameters(), lr=LR)
        old_classes = sorted(set(seen))
        rehearsal_buf = merge_buffers(buffers) if buffers else None

        train_one_task(
            net, train_views[k], active,
            n_epochs=EPOCHS_PER_TASK, opt=opt,
            ewc_baseline=ewc_baseline,
            rehearsal_buf=rehearsal_buf,
            rehearsal_loss=cond.loss,
            old_classes=old_classes,
        )

        consolidate(net, train_views[k], active)
        ewc_baseline = EWC_INTERTASK

        # build buffer from this task using the *consolidated* network
        if cond.kind == "real":
            new_buf = sample_real(train_views[k], active, cond.K)
        elif cond.kind == "engram":
            new_buf = build_engrams(net, active, cond.K)
        else:
            raise ValueError(f"unknown kind={cond.kind}")
        buffers.append(new_buf)
        seen.extend(active)

    return evaluate_full(net, eval_views, sorted(set(seen)))


def main() -> None:
    print(f"[probe] loading datasets ...")
    bundle = DatasetBundle(["mnist"], n_holdout_per_dataset=0)
    specs = split_mnist_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [list(s.global_classes) for s in specs]

    conditions = [
        Condition("real-K=1",      "real",   1,   "ce"),
        Condition("real-K=10",     "real",   10,  "ce"),
        Condition("real-K=100",    "real",   100, "ce"),
        Condition("engram-K=1-CE", "engram", 1,   "ce"),
        Condition("engram-K=10-CE","engram", 10,  "ce"),
        Condition("engram-K=100-CE","engram",100, "ce"),
        Condition("engram-K=1-LwF", "engram", 1,   "lwf"),
        Condition("engram-K=10-LwF","engram", 10,  "lwf"),
        Condition("engram-K=100-LwF","engram",100, "lwf"),
        # zero-rehearsal control
        Condition("none",          "real",   0,   "ce"),
    ]

    results: List[Tuple[str, float, float]] = []
    for cond in conditions:
        if cond.K == 0:
            torch.manual_seed(SEED)
            net = make_net()
            opt = optim.Adam(net.parameters(), lr=LR)
            ewc_baseline = 0.0
            seen: List[int] = []
            for k in range(N_TASKS):
                active = list(task_class_lists[k])
                if max(active) >= net.layers[-1].n_nodes:
                    extend_head(net, max(active) - net.layers[-1].n_nodes + 1)
                    opt = optim.Adam(net.parameters(), lr=LR)
                train_one_task(
                    net, train_views[k], active,
                    n_epochs=EPOCHS_PER_TASK, opt=opt,
                    ewc_baseline=ewc_baseline,
                )
                consolidate(net, train_views[k], active)
                ewc_baseline = EWC_INTERTASK
                seen.extend(active)
            metrics = evaluate_full(net, eval_views, sorted(set(seen)))
        else:
            metrics = run_condition(cond, train_views, eval_views, task_class_lists)
        print(f"[probe] {cond.name:25s} full={metrics['full']:.4f} task={metrics['task']:.4f}")
        results.append((cond.name, metrics["full"], metrics["task"]))

    print("\n=== Summary (chained-5 MNIST, single seed) ===")
    print(f"{'condition':25s} {'full':>8s} {'task':>8s}")
    for name, f, t in results:
        print(f"{name:25s} {f:>8.4f} {t:>8.4f}")


if __name__ == "__main__":
    main()
