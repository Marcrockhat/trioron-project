"""Chained-15 + LCN perception + Axis 6 emergence + credit-based freezing.

Three modes (controlled by AXIS6_MODE_RUN env var):
  - native  (default): train all 15 tasks from scratch with credit-based freezing
  - extend            : train tasks 0..9 → freeze L1 substrate → extend with 10..14
  - absorb            : train two donors on overlapping subsets, api.absorb fuse

Architecture:
  784  → L0 (128, LCN-masked random projection, FROZEN)
       → L1 (16, growable via Axis 6, credit-frozen per cell)
       → head (30, gradient-isolated per task via outs[:, task.global_classes])

Perception (L0): 16×8 retinotopic grid on image [0,1]² with Gaussian-overlap
receptive fields (σ=0.10, ~50% overlap between adjacent cells). L0 outputs
have meaningful positional identity — cell i = "weighted activity in region
X of the image".

Substrate (L1): credit on a cell = (engagement_frac > AXIS6_CREDIT_THR)
where engagement_frac is the mean per-input activation rate of the cell
over the task's training data. Credited cells get L1.W[i,:] and L1.b[i]
gradient masks for all subsequent tasks. Axis 6 field_conditional_growth
spawns new cells where epi_A is hot; fresh cells start uncredited =
fully plastic.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.network import TrioronNetwork
from trioron import masked_cross_entropy
from experiments.datasets import (
    DatasetBundle, TaskDataView, chained_15_specs, build_task_views,
    DEFAULT_DATA_ROOT,
)
from experiments.bench_chained_15task import build_lcn_mask, _lcn_cell_centers


# ----------------------------------------------------------------------
# Architecture constants
# ----------------------------------------------------------------------
INPUT_DIM = 784
L0_WIDTH = 128
LCN_GRID_X = 16
LCN_GRID_Y = 8
LCN_SIGMA = 0.10
H_INIT = int(os.environ.get("AXIS6_H_INIT", "16"))
N_CLASSES = 30
# Cosine-head temperature: scales the [-1, 1] cosine similarities into
# CE-friendly logits. Higher = sharper softmax. 16 is a common choice
# (matches CosFace / ArcFace conventions for shallow heads).
COSINE_TEMPERATURE = float(os.environ.get("AXIS6_COS_TEMP", "16.0"))


def cosine_logits(
    l1_features: torch.Tensor,
    head_W: torch.Tensor,
    temperature: float = COSINE_TEMPERATURE,
) -> torch.Tensor:
    """Cosine-similarity head: outs[b, k] = τ · cos(l1_features[b], head_W[k]).

    No bias term — each class's head row IS its prototype vector in L1
    feature space. Both l1_features and head_W are L2-normalized so the
    inner product gives bounded similarities in [-1, 1]. Temperature τ
    scales for cross-entropy softmax sharpness.

    This removes the bias-accumulation pathology of linear+softmax heads
    in the streamlined-arch setup (no EWC, fixed-size head, credit-frozen
    L1): synth replay can no longer pump stored-class logits to large
    positive bias values, because the cosine output is bounded.
    """
    h_norm = F.normalize(l1_features, dim=-1, eps=1e-8)
    w_norm = F.normalize(head_W, dim=-1, eps=1e-8)
    return temperature * (h_norm @ w_norm.t())


# ----------------------------------------------------------------------
# Build a 3-layer trioron with LCN-masked frozen L0 + growable L1 + head
# ----------------------------------------------------------------------
def build_net(seed: int) -> TrioronNetwork:
    torch.manual_seed(seed)
    net = TrioronNetwork([
        (INPUT_DIM, L0_WIDTH, "relu"),    # L0: perception
        (L0_WIDTH, H_INIT, "relu"),       # L1: growable substrate
        (H_INIT, N_CLASSES, "linear"),    # Head
    ])
    L0 = net.layers[0]
    L1 = net.layers[1]

    # Install LCN mask on L0 (multiplicative), then freeze L0.
    mask = build_lcn_mask(INPUT_DIM, L0_WIDTH, LCN_SIGMA).to(
        device=L0.W.device, dtype=L0.W.dtype,
    )
    L0.register_buffer("W_lcn_mask", mask)
    with torch.no_grad():
        L0.W.data.mul_(mask)
        if hasattr(L0, "W_anchor"):
            L0.W_anchor.data.mul_(mask.to(L0.W_anchor.dtype))
    L0.W.requires_grad_(False)
    L0.b.requires_grad_(False)
    if hasattr(L0, "branch_weight"):
        L0.branch_weight.requires_grad_(False)

    # L0 cell positions live in image-space [0,1]² (third dim padded to 0).
    with torch.no_grad():
        l0_xy = _lcn_cell_centers()  # (128, 2)
        L0.cell_position[:, 0] = l0_xy[:, 0]
        L0.cell_position[:, 1] = l0_xy[:, 1]
        L0.cell_position[:, 2] = 0.0

    # Cosine head: head.b is unused (cosine logits have no bias term).
    # Freeze it at zero so it can't drift via any leftover gradient path.
    head = net.layers[2]
    with torch.no_grad():
        head.b.data.zero_()
    head.b.requires_grad_(False)

    # L1 cells start at uniform random positions in image-space [0,1]² so
    # the Axis 6 field diffuses over image coordinates, not arbitrary
    # indices. New spawned L1 cells will appear near stressed parents
    # (with jitter), letting the substrate self-organize a locus.
    g = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        L1.cell_position[:, 0] = torch.rand(L1.n_nodes, generator=g)
        L1.cell_position[:, 1] = torch.rand(L1.n_nodes, generator=g)
        L1.cell_position[:, 2] = 0.0

    return net


# ----------------------------------------------------------------------
# Manifold replay store — per-class μ/σ of L0 (frozen perception) codes
# ----------------------------------------------------------------------
@dataclass
class ManifoldStore:
    """Stores per-class (μ, σ) over L0 OUTPUT codes (post-LCN-perception).

    Why L0 not L1: L0 is the frozen perception adapter (LCN-masked, no
    backward). Its output for a given image is deterministic and stable
    across the entire curriculum. Storing μ/σ at L0 means synthetic
    samples drawn now will be statistically equivalent to L0 outputs
    drawn at eval time on the same class.

    During training of subsequent tasks, synthetic L0 codes are forwarded
    through the CURRENT L1 + head (forward_from_layer(start_layer=1)).
    This means synthetic samples pass through the same L1 cells the real
    eval-time data will pass through — including new cells that spawned
    after the class was stored. Head learns a consistent decision over
    all classes' representations under the current L1.

    Storage: ~128 × 4 × 2 = 1 KB/class. 30 classes = ~30 KB total.
    """
    mu_per_class: Dict[int, torch.Tensor] = field(default_factory=dict)
    sigma_per_class: Dict[int, torch.Tensor] = field(default_factory=dict)
    n_l0: int = 0
    sigma_floor: float = 1e-3

    def has_classes(self) -> bool:
        return len(self.mu_per_class) > 0

    def store_task(
        self,
        net: TrioronNetwork,
        view: TaskDataView,
        task_global_classes: Sequence[int],
        batch_size: int = 1024,
    ) -> None:
        """Compute per-class μ/σ of L0 codes on this task's training set.

        L0 codes = post-activation output of layer 0 (frozen LCN-masked
        perception). Stable across the curriculum (L0 has requires_grad
        False). Stored AFTER credit-freezing for symmetry with the
        rest of the pipeline.
        """
        L0 = net.layers[0]
        with torch.no_grad():
            x, y = view.all_examples()
            n = x.shape[0]
            feats_chunks: List[torch.Tensor] = []
            label_chunks: List[torch.Tensor] = []
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                h0 = L0(x[start:end])
                feats_chunks.append(h0.detach().cpu())
                label_chunks.append(y[start:end])
            feats = torch.cat(feats_chunks, dim=0)
            labels = torch.cat(label_chunks, dim=0)
            for c in task_global_classes:
                m = (labels == c)
                if int(m.sum().item()) > 0:
                    cf = feats[m]
                    self.mu_per_class[int(c)] = cf.mean(dim=0)
                    self.sigma_per_class[int(c)] = (
                        cf.std(dim=0, unbiased=False).clamp(min=self.sigma_floor)
                    )
            self.n_l0 = L0.n_nodes

    def sample_synthetic(
        self,
        n_total: int,
        generator: Optional[torch.Generator] = None,
        noise_scale: float = 1.0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Draw n_total synthetic L0 codes, each from a uniformly-chosen
        stored class. Matches bench's ManifoldBuffer.sample pattern: fixed
        total budget, sparse-per-class as the curriculum grows.

        Returns (codes [n_total, n_l0], labels [n_total]) or None if no
        classes stored. Codes are clamped to >=0 since L0 uses ReLU.
        """
        if not self.mu_per_class:
            return None
        classes_sorted = sorted(self.mu_per_class.keys())
        choice = torch.randint(
            0, len(classes_sorted), (n_total,), generator=generator,
        )
        d = self.mu_per_class[classes_sorted[0]].shape[0]
        feats = torch.zeros(n_total, d)
        labels = torch.zeros(n_total, dtype=torch.long)
        for i in range(n_total):
            c = classes_sorted[int(choice[i])]
            mu = self.mu_per_class[c]
            sigma = self.sigma_per_class[c] * noise_scale
            noise = torch.randn(d, generator=generator)
            feats[i] = (mu + sigma * noise).clamp(min=0.0)
            labels[i] = c
        return feats, labels


# ----------------------------------------------------------------------
# Train one task
# ----------------------------------------------------------------------
@dataclass
class TaskTrainResult:
    events: List[Tuple[int, str]]
    engagement_frac: Optional[torch.Tensor]
    train_loss_final: float
    train_acc_final: float


def train_one_task(
    net: TrioronNetwork,
    view: TaskDataView,
    task_global_classes: List[int],
    n_epochs: int,
    batch_size: int,
    lr: float,
    axis6: bool,
    diff_floor: float,
    diff_b_thr: float,
    field_sigma: float,
    field_dt: float,
    stress_tolerance: float,
    spawn_cap: int,
    cooldown_steps: int,
    l1_credit_mask: Optional[torch.Tensor],
    track_engagement: bool,
    manifold: Optional[ManifoldStore] = None,
    lambda_manifold: float = 1.0,
    manifold_total_batch: int = 64,
    manifold_noise_scale: float = 1.0,
    seen_classes_global: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> TaskTrainResult:
    L1 = net.layers[1]
    if axis6:
        L1.enable_axis6_field(field_sigma=field_sigma)
        L1.reset_epi_field()

    # Optimizer: only L1 + head (L0 was set to requires_grad=False).
    opt = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=lr,
    )

    events: List[Tuple[int, str]] = []
    spawns_this_task = 0
    last_spawn_step = -10**9
    step = 0
    engagement_sum: Optional[torch.Tensor] = (
        torch.zeros(L1.n_nodes) if track_engagement else None
    )
    engagement_count: Optional[torch.Tensor] = (
        torch.zeros(L1.n_nodes) if track_engagement else None
    )
    last_loss = float("nan")
    last_acc = float("nan")

    head_class_idx = torch.tensor(task_global_classes, dtype=torch.long)
    gen = torch.Generator().manual_seed(int(time.time() * 1000) & 0xFFFFFFFF)

    for epoch in range(n_epochs):
        for xb, yb_global in view.iter_epoch(batch_size, generator=gen):
            opt.zero_grad()
            # Explicit L0 → L1 forward (so we can apply cosine head and
            # also have L1's _last_y / hook properly cached for stress).
            h0 = net.layers[0](xb)
            h1 = net.layers[1](h0)
            outs = cosine_logits(h1, net.layers[2].W)
            # Task-local labels: map global → local-in-task ({0, 1})
            yb_local = torch.zeros_like(yb_global)
            for i, g in enumerate(task_global_classes):
                yb_local = torch.where(yb_global == g,
                                       torch.tensor(i, dtype=yb_local.dtype),
                                       yb_local)
            task_logits = outs[:, head_class_idx]
            real_loss = F.cross_entropy(task_logits, yb_local)
            # Manifold replay: head-only forward of synthetic L1 features
            # for every previously-stored class. Calibrates head logits
            # across all classes seen so the full-softmax metric stays
            # meaningful, independent of credit-mask + spawning on L1.
            synth_loss = torch.tensor(0.0)
            if (manifold is not None
                    and manifold.has_classes()
                    and seen_classes_global is not None):
                sample = manifold.sample_synthetic(
                    manifold_total_batch, noise_scale=manifold_noise_scale,
                )
                if sample is not None:
                    synth_feats, synth_labels = sample
                    # Forward synth L0 → L1 (using CURRENT L1 so the
                    # representation matches eval-time), THEN DETACH
                    # before head. The detach ensures the synth loss
                    # only updates the head — L1 stays purely a function
                    # of real data. Without this, synth pulls new
                    # (uncredited) cells toward predicting OLD classes,
                    # fighting the credit anchor and degrading task_aware.
                    with torch.no_grad():
                        synth_h1 = net.layers[1](synth_feats)
                    synth_logits = cosine_logits(synth_h1, net.layers[2].W)
                    synth_loss = masked_cross_entropy(
                        synth_logits, synth_labels,
                        active_classes=list(seen_classes_global),
                    )
            loss = real_loss + lambda_manifold * synth_loss
            loss.backward()
            L1.update_internal_stress()

            # Track L1 engagement BEFORE the optimizer step (need the
            # cached _last_y from the just-completed forward pass).
            if track_engagement:
                with torch.no_grad():
                    last_y = L1._last_y
                    if last_y is not None:
                        if L1.activation == "relu":
                            active = (last_y > 0).float()
                        else:
                            active = (last_y.abs() > 0.05).float()
                        per_cell_rate = active.mean(dim=0).cpu()
                        n_cells = per_cell_rate.numel()
                        if engagement_sum is None or engagement_sum.numel() < n_cells:
                            pad_n = n_cells - (
                                0 if engagement_sum is None
                                else engagement_sum.numel()
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

            # Apply credit-based gradient mask on L1 (NOT L0 — L0 is frozen).
            if l1_credit_mask is not None and l1_credit_mask.any():
                with torch.no_grad():
                    cm = l1_credit_mask
                    n_cells = L1.W.shape[0]
                    if cm.numel() < n_cells:
                        pad = torch.zeros(
                            n_cells - cm.numel(),
                            dtype=torch.bool, device=cm.device,
                        )
                        cm = torch.cat([cm, pad])
                    if L1.W.grad is not None:
                        L1.W.grad[cm] = 0.0
                    if L1.b.grad is not None:
                        L1.b.grad[cm] = 0.0

            opt.step()
            step += 1

            if axis6:
                L1.update_epi_field(dt=field_dt, stress_tolerance=stress_tolerance)
                if (spawns_this_task < spawn_cap
                        and step - last_spawn_step >= cooldown_steps):
                    cand = L1.field_conditional_growth_candidate(
                        mode="absolute", k=1.0,
                        stress_floor=diff_floor, b_threshold=diff_b_thr,
                    )
                    if cand is not None:
                        new_idx = net.axis6_spawn(1, cand, position_jitter=0.05)
                        spawns_this_task += 1
                        last_spawn_step = step
                        events.append(
                            (step,
                             f"AXIS6 spawn: parent {cand} → new {new_idx} "
                             f"@ {L1.cell_position[new_idx][:2].tolist()}")
                        )
                        # Rebuild opt to include new parameter slots.
                        opt = torch.optim.Adam(
                            [p for p in net.parameters() if p.requires_grad],
                            lr=lr,
                        )

            with torch.no_grad():
                pred = task_logits.argmax(dim=1)
                last_loss = float(loss.item())
                last_acc = float((pred == yb_local).float().mean().item())

        if verbose:
            extras = ""
            if axis6:
                extras = (
                    f" stress_max={float(L1.internal_stress.max().item()):.4f}"
                    f" epi_A_max={float(L1.epi_A.max().item()):.4f}"
                    f" epi_B_max={float(L1.epi_B.max().item()):.4f}"
                )
            print(f"    [epoch {epoch}] loss={last_loss:.4f} acc={last_acc:.3f} "
                  f"H={L1.n_nodes} spawns={spawns_this_task}{extras}")

    engagement_frac: Optional[torch.Tensor] = None
    if track_engagement and engagement_sum is not None and engagement_count is not None:
        n_cells = L1.n_nodes
        if engagement_sum.numel() < n_cells:
            pad = torch.zeros(n_cells - engagement_sum.numel())
            engagement_sum = torch.cat([engagement_sum, pad])
            engagement_count = torch.cat([engagement_count, torch.zeros_like(pad)])
        engagement_frac = engagement_sum / engagement_count.clamp(min=1.0)

    return TaskTrainResult(
        events=events,
        engagement_frac=engagement_frac,
        train_loss_final=last_loss,
        train_acc_final=last_acc,
    )


# ----------------------------------------------------------------------
# Evaluate a single task in the GLOBAL-class space (task-aware metric:
# argmax restricted to task's 2 class outputs).
# ----------------------------------------------------------------------
def _forward_logits(net: TrioronNetwork, x: torch.Tensor) -> torch.Tensor:
    h0 = net.layers[0](x)
    h1 = net.layers[1](h0)
    return cosine_logits(h1, net.layers[2].W)


def evaluate_task_aware(
    net: TrioronNetwork,
    view: TaskDataView,
    task_global_classes: List[int],
) -> float:
    head_class_idx = torch.tensor(task_global_classes, dtype=torch.long)
    with torch.no_grad():
        x, y_global = view.all_examples()
        outs = _forward_logits(net, x)
        task_logits = outs[:, head_class_idx]
        pred_local = task_logits.argmax(dim=1)
        pred_global = head_class_idx[pred_local]
    return float((pred_global == y_global).float().mean().item())


def evaluate_full(
    net: TrioronNetwork,
    view: TaskDataView,
) -> float:
    with torch.no_grad():
        x, y_global = view.all_examples()
        outs = _forward_logits(net, x)
        pred_global = outs.argmax(dim=1)
    return float((pred_global == y_global).float().mean().item())


# ----------------------------------------------------------------------
# Run a single arm
# ----------------------------------------------------------------------
def run_arm(
    seed: int,
    arm_label: str,
    axis6: bool,
    credit_enabled: bool,
    manifold_enabled: bool,
    train_views: Sequence[TaskDataView],
    eval_views: Sequence[TaskDataView],
    task_class_lists: Sequence[List[int]],
    n_epochs: int,
    batch_size: int,
    lr: float,
    hp: dict,
    starting_task_idx: int = 0,
    starting_net: Optional[TrioronNetwork] = None,
    starting_credit: Optional[torch.Tensor] = None,
    verbose: bool = False,
) -> dict:
    if starting_net is None:
        net = build_net(seed=seed)
        cell_credit = (
            torch.zeros(net.layers[1].n_nodes, dtype=torch.bool)
            if credit_enabled else None
        )
    else:
        net = starting_net
        cell_credit = starting_credit if starting_credit is not None else (
            torch.zeros(net.layers[1].n_nodes, dtype=torch.bool)
            if credit_enabled else None
        )

    manifold = (
        ManifoldStore(n_l0=net.layers[0].n_nodes)
        if manifold_enabled else None
    )

    K = len(train_views)
    retention_task_aware: List[List[float]] = []
    retention_full: List[List[float]] = []
    all_events: List[Tuple[str, int, str]] = []
    seen_classes: List[int] = []
    t0 = time.time()
    for task_idx in range(starting_task_idx, K):
        # Track cumulative seen classes (current task included from the
        # start of its training — this gates the masked-CE active set).
        for c in task_class_lists[task_idx]:
            if c not in seen_classes:
                seen_classes.append(c)
        name = train_views[task_idx].name
        if verbose:
            print(f"  [phase {task_idx+1}/{K}: {name}] L1.H={net.layers[1].n_nodes}  "
                  f"frozen={int(cell_credit.sum().item()) if cell_credit is not None else 'n/a'}")
        res = train_one_task(
            net=net, view=train_views[task_idx],
            task_global_classes=task_class_lists[task_idx],
            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
            axis6=axis6,
            diff_floor=hp["diff_floor"], diff_b_thr=hp["diff_b_thr"],
            field_sigma=hp["field_sigma"], field_dt=hp["field_dt"],
            stress_tolerance=hp["stress_tolerance"],
            spawn_cap=hp["spawn_cap"], cooldown_steps=hp["cooldown_steps"],
            l1_credit_mask=cell_credit,
            track_engagement=credit_enabled,
            manifold=manifold,
            lambda_manifold=hp["lambda_manifold"],
            manifold_total_batch=hp["manifold_total_batch"],
            manifold_noise_scale=hp["manifold_noise_scale"],
            seen_classes_global=list(seen_classes),
            verbose=verbose,
        )
        for step, msg in res.events:
            all_events.append((name, step, msg))
        # End-of-task: snapshot retention on every task seen so far.
        row_ta = [
            evaluate_task_aware(net, eval_views[k], task_class_lists[k])
            for k in range(K)
        ]
        row_full = [evaluate_full(net, eval_views[k]) for k in range(K)]
        retention_task_aware.append(row_ta)
        retention_full.append(row_full)
        if verbose:
            print(f"    task_aware(this)={row_ta[task_idx]:.3f}  "
                  f"task_aware(mean_seen)={sum(row_ta[:task_idx+1])/(task_idx+1):.3f}  "
                  f"full(this)={row_full[task_idx]:.3f}  "
                  f"spawn_events={len(res.events)}")
        # Update L1 cell_credit by engagement.
        if credit_enabled and res.engagement_frac is not None:
            with torch.no_grad():
                L1 = net.layers[1]
                if cell_credit is None:
                    cell_credit = torch.zeros(L1.n_nodes, dtype=torch.bool)
                elif cell_credit.numel() < L1.n_nodes:
                    pad = torch.zeros(L1.n_nodes - cell_credit.numel(),
                                      dtype=torch.bool)
                    cell_credit = torch.cat([cell_credit, pad])
                ef = res.engagement_frac
                if ef.numel() < L1.n_nodes:
                    pad_e = torch.zeros(L1.n_nodes - ef.numel())
                    ef = torch.cat([ef, pad_e])
                earned = ef > hp["credit_thr"]
                newly = earned & (~cell_credit)
                cell_credit = cell_credit | earned
                if verbose:
                    print(f"    [credit] +{int(newly.sum().item())} new; "
                          f"total frozen = {int(cell_credit.sum().item())}/"
                          f"{cell_credit.numel()}")
        # Snapshot manifold per-class μ/σ AFTER credit is applied so the
        # stored features reflect the frozen-from-now-on representation.
        if manifold is not None:
            manifold.store_task(
                net=net, view=train_views[task_idx],
                task_global_classes=task_class_lists[task_idx],
            )
            if verbose:
                print(f"    [manifold] stored {len(task_class_lists[task_idx])} classes; "
                      f"total stored = {len(manifold.mu_per_class)} classes "
                      f"(n_l0={manifold.n_l0})")

    elapsed = time.time() - t0
    # Final pass: per-task task-aware + full accuracies
    final_task_aware = retention_task_aware[-1] if retention_task_aware else []
    final_full = retention_full[-1] if retention_full else []
    first_task_aware = [retention_task_aware[k][k] for k in range(len(retention_task_aware))]
    drift_task_aware = [
        first_task_aware[k] - final_task_aware[k]
        for k in range(len(first_task_aware))
    ]
    max_drift_ta = max((abs(d) for d in drift_task_aware), default=0.0)

    return {
        "seed": seed,
        "arm": arm_label,
        "elapsed_s": elapsed,
        "n_hidden_L1": net.layers[1].n_nodes,
        "n_spawns": len(all_events),
        "n_frozen": int(cell_credit.sum().item()) if cell_credit is not None else 0,
        "retention_task_aware": retention_task_aware,
        "retention_full": retention_full,
        "final_task_aware": final_task_aware,
        "final_full": final_full,
        "mean_final_task_aware": (
            sum(final_task_aware) / len(final_task_aware)
            if final_task_aware else 0.0
        ),
        "mean_final_full": (
            sum(final_full) / len(final_full)
            if final_full else 0.0
        ),
        "max_drift_task_aware": max_drift_ta,
        "events": all_events,
        "net": net,
        "cell_credit": cell_credit,
    }


def mean_std(xs):
    n = len(xs)
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / max(1, n - 1)
    return m, v ** 0.5


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> int:
    n_epochs = int(os.environ.get("AXIS6_EPOCHS", "4"))
    batch_size = int(os.environ.get("AXIS6_BATCH", "256"))
    lr = float(os.environ.get("AXIS6_LR", "0.01"))
    n_seeds = int(os.environ.get("AXIS6_N_SEEDS", "1"))
    diff_floor = float(os.environ.get("AXIS6_FLOOR", "0.005"))
    diff_b_thr = float(os.environ.get("AXIS6_B_THR", "1.0"))
    field_sigma = float(os.environ.get("AXIS6_SIGMA", "0.2"))
    field_dt = float(os.environ.get("AXIS6_DT", "0.1"))
    stress_tolerance = float(os.environ.get("AXIS6_TOL", "0.0"))
    spawn_cap = int(os.environ.get("AXIS6_SPAWN_CAP", "5"))
    cooldown_steps = int(os.environ.get("AXIS6_COOLDOWN", "50"))
    credit_thr = float(os.environ.get("AXIS6_CREDIT_THR", "0.1"))
    lambda_manifold = float(os.environ.get("AXIS6_LAMBDA_MANIFOLD", "1.0"))
    manifold_total_batch = int(os.environ.get("AXIS6_MANIFOLD_BATCH", "64"))
    manifold_noise_scale = float(os.environ.get("AXIS6_MANIFOLD_NOISE", "1.0"))
    data_root = os.environ.get("AXIS6_DATA_ROOT", DEFAULT_DATA_ROOT)
    arms_env = os.environ.get(
        "AXIS6_ARMS",
        "BASELINE_frozen,AXIS6_emerge,AXIS6_credit,AXIS6_credit_manifold",
    )

    hp = dict(
        diff_floor=diff_floor, diff_b_thr=diff_b_thr,
        field_sigma=field_sigma, field_dt=field_dt,
        stress_tolerance=stress_tolerance,
        spawn_cap=spawn_cap, cooldown_steps=cooldown_steps,
        credit_thr=credit_thr,
        lambda_manifold=lambda_manifold,
        manifold_total_batch=manifold_total_batch,
        manifold_noise_scale=manifold_noise_scale,
    )

    print("=" * 88)
    print(f"axis6_credit_chained15 — LCN perception + L1 substrate")
    print(f"epochs/task={n_epochs}  batch={batch_size}  lr={lr}  "
          f"n_seeds={n_seeds}  spawn_cap={spawn_cap}  credit_thr={credit_thr}")
    print(f"H_init(L1)={H_INIT}  L0_width(frozen,LCN)={L0_WIDTH}  "
          f"LCN_grid={LCN_GRID_X}×{LCN_GRID_Y}  LCN_σ={LCN_SIGMA}")
    print(f"field_sigma={field_sigma}  diff_floor={diff_floor}  "
          f"cooldown={cooldown_steps}")
    print("=" * 88)

    bundle = DatasetBundle(
        ["mnist", "fashion_mnist", "emnist_letters"],
        root=data_root,
        n_holdout_per_dataset=0,
    )
    specs = chained_15_specs()
    train_views = build_task_views(bundle, specs, split="train")
    eval_views = build_task_views(bundle, specs, split="test")
    task_class_lists = [s.global_classes for s in specs]
    print(f"Curriculum: {[s.name for s in specs]}")
    print(f"Train sizes: {[v.n_examples() for v in train_views]}")
    print(f"Eval  sizes: {[v.n_examples() for v in eval_views]}")
    print()

    # Each arm = (label, axis6_on, credit_on, manifold_on)
    arms = [
        ("BASELINE_frozen", False, False, False),
        ("AXIS6_emerge", True, False, False),
        ("AXIS6_credit", True, True, False),
        ("AXIS6_credit_manifold", True, True, True),
    ]
    arms_set = {s.strip() for s in arms_env.split(",")}
    selected = [a for a in arms if a[0] in arms_set]

    results: dict = {a[0]: [] for a in selected}
    for seed in range(n_seeds):
        for arm_label, axis6, credit_enabled, manifold_enabled in selected:
            print(f"\n========== SEED {seed}  ARM: {arm_label} ==========")
            r = run_arm(
                seed=seed, arm_label=arm_label,
                axis6=axis6, credit_enabled=credit_enabled,
                manifold_enabled=manifold_enabled,
                train_views=train_views, eval_views=eval_views,
                task_class_lists=task_class_lists,
                n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                hp=hp, verbose=True,
            )
            results[arm_label].append(r)
            print(
                f"[seed {seed}  arm {arm_label}]  "
                f"task_aware_mean={r['mean_final_task_aware']:.3f}  "
                f"full_mean={r['mean_final_full']:.3f}  "
                f"max|drift|_ta={r['max_drift_task_aware']:.3f}  "
                f"H={r['n_hidden_L1']}  spawns={r['n_spawns']}  "
                f"frozen={r['n_frozen']}  elapsed={r['elapsed_s']:.1f}s"
            )

    print("\n" + "=" * 88)
    print(f"AGGREGATE (n={n_seeds} seeds, mean ± std)")
    print("=" * 88)
    print(
        f"  {'arm':<16}  {'task_aware':>14}  {'full':>14}  "
        f"{'max|drift|_ta':>14}  {'H_L1':>10}  {'spawns':>8}  {'frozen':>8}"
    )
    for arm_label, *_ in selected:
        rs = results[arm_label]
        ta_m, ta_s = mean_std([r["mean_final_task_aware"] for r in rs])
        f_m, f_s = mean_std([r["mean_final_full"] for r in rs])
        md_m, md_s = mean_std([r["max_drift_task_aware"] for r in rs])
        h_m, h_s = mean_std([float(r["n_hidden_L1"]) for r in rs])
        sp_m, sp_s = mean_std([float(r["n_spawns"]) for r in rs])
        fr_m, fr_s = mean_std([float(r["n_frozen"]) for r in rs])
        print(
            f"  {arm_label:<16}  {ta_m:>6.3f} ± {ta_s:>5.3f}  "
            f"{f_m:>6.3f} ± {f_s:>5.3f}  "
            f"{md_m:>6.3f} ± {md_s:>5.3f}  "
            f"{h_m:>4.1f} ± {h_s:>3.1f}  "
            f"{sp_m:>3.1f} ± {sp_s:>2.1f}  "
            f"{fr_m:>3.1f} ± {fr_s:>2.1f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
