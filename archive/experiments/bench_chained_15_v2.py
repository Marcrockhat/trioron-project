"""Chained-15 continual learning benchmark on the v2.0 substrate.

MNIST → FashionMNIST → EMNIST-letters, 15 binary tasks, 30 global classes.
Tests credit-based locking + manifold replay + frustration-gated growth.

Usage:
    python3 -m experiments.bench_chained_15_v2 [--seed 42] [--epochs 4] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

_NONLINEAR = os.environ.get("TRIORON_NONLINEAR", "0") == "1"

import torch
import torch.nn as nn

from experiments.datasets import (
    DatasetBundle, ChainedTaskSpec, chained_15_specs, IMAGE_DIM,
)
from trioron.core import Envelope, construct
from trioron.core.epigenome import OUTPUT, PERCEPTION, has_gene
from trioron.core.state import CellState
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
from trioron.learning import (
    CreditTracker, FrustrationDetector, ManifoldArchive,
    dream_cycle, DreamConfig, interleaved_replay_batch,
)
from trioron.learning.dream import cluster_replay_batch, kibra_tag
from trioron.learning.manifold import get_interior_ids
from trioron.lifecycle import divide, GrowthConfig
from trioron.viz import Recorder, export_html
from trioron.viz.detect import detect_from_directory


# ── Config ───────────────────────────────────────────────────────

N_GLOBAL_CLASSES = 30
H_INIT = 55
BATCH = 30
LR = 6.68e-4
N_GROW_PER_TASK = 9
PARAM_CAP_BYTES = 300_000
SPARSITY_LAMBDA = 0.01  # L1 penalty on H-cell activations (soft sparsity)
PRIVATE_FAN_IN = 784  # perception edges per task-detector cell
ANCHOR_LAMBDA = 100.0  # EWC-like penalty on H→output edges
ANCHOR_FISHER_BATCHES = 15  # batches for Fisher estimation
ANCHOR_GAMMA = 0.9  # Online EWC decay for cumulative Fisher
N_DET_CELLS = 4  # detector cells per task (multi-neuron)
CALIBRATE_EPOCHS = 3  # joint detector calibration after all tasks


# ── Output-edge anchoring ────────────────────────────────────────

@dataclass
class OutputAnchor:
    """Online EWC anchor for H→output edges only."""
    edge_idx: torch.Tensor      # indices into arena.edge_weight
    weight_snap: torch.Tensor   # θ* at last task boundary
    fisher: torch.Tensor        # cumulative Fisher diagonal


def find_output_edge_indices(arena) -> torch.Tensor:
    """Return edge indices whose dst is an OUTPUT cell."""
    out_mask = torch.zeros(arena.capacity, dtype=torch.bool, device=arena.device)
    for cid in range(arena.cursor):
        if arena.alive[cid] and has_gene(int(arena.epigenome[cid].item()), OUTPUT):
            out_mask[cid] = True
    dst = arena.edge_dst[:arena.edge_cursor].long()
    is_output_edge = out_mask[dst]
    return is_output_edge.nonzero(as_tuple=False).squeeze(-1)


def compute_output_fisher(sub, train_view, output_edge_idx: torch.Tensor,
                          n_batches: int = ANCHOR_FISHER_BATCHES) -> torch.Tensor:
    """Estimate diagonal Fisher for output edges via gradient squared."""
    fisher = torch.zeros(output_edge_idx.shape[0], device=sub.arena.device)
    sub.prepare_training()
    count = 0
    for x_batch, y_batch in train_view.iter_epoch(BATCH):
        logits = sub(x_batch)
        loss = torch.nn.functional.cross_entropy(logits, y_batch)
        loss.backward()
        if sub.arena.edge_weight.grad is not None:
            g = sub.arena.edge_weight.grad[output_edge_idx]
            fisher += g * g
        sub.arena.edge_weight.grad = None
        sub.arena.bias.grad = None
        count += 1
        if count >= n_batches:
            break
    fisher /= max(count, 1)
    return fisher.detach()


def update_anchor(anchor: OutputAnchor | None, sub, train_view,
                  gamma: float = ANCHOR_GAMMA) -> OutputAnchor:
    """Compute Fisher for current task and update the cumulative anchor."""
    edge_idx = find_output_edge_indices(sub.arena)
    fisher_new = compute_output_fisher(sub, train_view, edge_idx)
    w_snap = sub.arena.edge_weight[edge_idx].detach().clone()

    if anchor is None:
        return OutputAnchor(edge_idx, w_snap, fisher_new)

    if anchor.edge_idx.shape[0] != edge_idx.shape[0]:
        old_fisher = torch.zeros(edge_idx.shape[0], device=sub.arena.device)
        n_old = min(anchor.edge_idx.shape[0], edge_idx.shape[0])
        old_fisher[:n_old] = anchor.fisher[:n_old]
        anchor.fisher = old_fisher
        anchor.edge_idx = edge_idx

    anchor.fisher = gamma * anchor.fisher + fisher_new
    anchor.weight_snap = w_snap
    anchor.edge_idx = edge_idx
    return anchor


def anchor_penalty(anchor: OutputAnchor | None, arena, lam: float = ANCHOR_LAMBDA) -> torch.Tensor:
    """Quadratic penalty: λ * Σ F_i * (w_i - w_i*)²."""
    if anchor is None:
        return torch.tensor(0.0, device=arena.device)
    w = arena.edge_weight[anchor.edge_idx]
    diff = w - anchor.weight_snap
    return lam * (anchor.fisher * diff * diff).sum()


# ── Substrate construction ───────────────────────────────────────

def build_substrate(seed: int = 42, h_init: int = H_INIT, interior_layers: int = 1):
    torch.manual_seed(seed)
    import os
    nonlinear = os.environ.get("TRIORON_NONLINEAR", "0") == "1"
    sub = construct(
        base=seeded(IMAGE_DIM, N_GLOBAL_CLASSES,
                    interior_cells=h_init, interior_layers=interior_layers,
                    nonlinear=nonlinear),
        envelope=Envelope(max_parameter_bytes=PARAM_CAP_BYTES),
        dispatch_table=default_dispatch_table(),
        capacity=2048,
        sparsity_k=0,
    )
    return sub


def add_task_detectors(sub, n_tasks: int = 15, n_det_cells: int = N_DET_CELLS, seed: int = 42):
    """Allocate n_det_cells detector cells per task with full perception input.

    Returns list of lists: detectors[t] = [cell_id_0, ..., cell_id_{n-1}].
    At eval, mean activation over cells gives the task score.
    """
    a = sub.arena

    perc_ids = [cid for cid in range(a.cursor)
                if a.alive[cid] and has_gene(int(a.epigenome[cid].item()), PERCEPTION)]

    detectors = []
    for t in range(n_tasks):
        task_dets = []
        for d in range(n_det_cells):
            pid = int(a.alloc(1)[0].item())
            a.rank[pid] = 1
            a.position[pid] = torch.tensor([
                0.5,
                t / max(n_tasks - 1, 1),
                0.85 + 0.1 * d / max(n_det_cells - 1, 1),
            ])
            a.state[pid] = CellState.DORMANT
            a.forward_inclusion[pid] = True

            src = torch.tensor(perc_ids[:PRIVATE_FAN_IN], dtype=torch.int32)
            dst = torch.full((len(src),), pid, dtype=torch.int32)
            a.add_edges(src, dst)
            task_dets.append(pid)
        detectors.append(task_dets)

    sub.compile()
    return detectors


# ── Head extension ───────────────────────────────────────────────

def extend_head(sub, new_global_classes: list[int]):
    """Add output cells for any global classes not yet in the substrate."""
    a = sub.arena
    existing_out = []
    for cid in a.alive_ids().tolist():
        if has_gene(int(a.epigenome[cid].item()), OUTPUT):
            existing_out.append(cid)

    n_existing = len(existing_out)
    n_needed = max(0, max(new_global_classes) + 1 - n_existing)
    if n_needed <= 0:
        return

    # Already allocated at construction for N_GLOBAL_CLASSES — no extension needed
    # for chained-15 since we pre-allocate all 30 output cells


# ── Evaluation ───────────────────────────────────────────────────

@dataclass
class TaskResult:
    task_idx: int
    task_name: str
    full_acc: float = 0.0
    task_acc: float = 0.0


@dataclass
class EvalResult:
    after_task: int
    per_task: list[TaskResult] = field(default_factory=list)

    @property
    def mean_full(self) -> float:
        if not self.per_task:
            return 0.0
        return sum(t.full_acc for t in self.per_task) / len(self.per_task)

    @property
    def mean_task(self) -> float:
        if not self.per_task:
            return 0.0
        return sum(t.task_acc for t in self.per_task) / len(self.per_task)


def evaluate_all_tasks(
    sub,
    bundle: DatasetBundle,
    specs: list[ChainedTaskSpec],
    tasks_seen: int,
    archive: ManifoldArchive | None = None,
    detectors: list[list[int]] | None = None,
    h_archive: ManifoldArchive | None = None,
    h_interior_ids: torch.Tensor | None = None,
    h_route_mode: str = "task",
    h_full_cov: bool = False,
) -> EvalResult:
    """Evaluate full-softmax (with optional routing) and task-aware accuracy."""
    result = EvalResult(after_task=tasks_seen - 1)

    n_perc = 0
    if h_interior_ids is None:
        h_interior_ids = get_interior_ids(sub.arena).long()
    for cid in range(sub.arena.cursor):
        if sub.arena.alive[cid] and has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION):
            n_perc += 1

    for t_idx in range(tasks_seen):
        spec = specs[t_idx]
        test_view = bundle.task_view(
            spec.dataset_name, spec.local_classes, spec.global_classes,
            split="test", task_name=spec.name,
        )
        x, y = test_view.all_examples()

        with torch.no_grad():
            logits = sub(x)
            act = sub.last_activations

            # Routing priority: detectors > H-space manifold > perception manifold > raw

            if detectors and len(detectors) >= tasks_seen:
                # Detector-based routing: mean-pool multi-cell detectors
                det_scores = torch.zeros(x.shape[0], tasks_seen, device=x.device)
                for tidx in range(tasks_seen):
                    cell_ids = detectors[tidx]
                    cell_acts = act[:, cell_ids]
                    det_scores[:, tidx] = cell_acts.mean(dim=1)
                best_task = det_scores.argmax(dim=1)

                pred_full = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                for tidx in range(tasks_seen):
                    mask = best_task == tidx
                    if not mask.any():
                        continue
                    tspec = specs[tidx]
                    tlogits = logits[mask][:, tspec.global_classes]
                    local_pred = tlogits.argmax(dim=1)
                    gc_t = torch.tensor(tspec.global_classes, dtype=torch.long)
                    pred_full[mask] = gc_t[local_pred]

            elif h_archive is not None and h_archive.n_classes > 0:
                # H-space manifold routing in the learned representation
                h_act = act[:, h_interior_ids]
                ll_fn = (lambda astro: astro.log_likelihood_full(h_act)) if h_full_cov \
                    else (lambda astro: astro.log_likelihood(h_act))
                if h_route_mode == "class":
                    # Pure QDA: per-class log-likelihood, bypasses output projection
                    all_seen_classes = []
                    for tidx in range(tasks_seen):
                        all_seen_classes.extend(specs[tidx].global_classes)
                    class_ll = torch.full((x.shape[0], N_GLOBAL_CLASSES), -1e12, device=x.device)
                    for gc in all_seen_classes:
                        astro = h_archive.get(gc)
                        if astro is not None:
                            class_ll[:, gc] = ll_fn(astro)
                    pred_full = class_ll.argmax(dim=1)
                else:
                    # Task-level: H-manifold picks task (robust), logits pick class (accurate)
                    task_ll = torch.zeros(x.shape[0], tasks_seen, device=x.device)
                    for tidx in range(tasks_seen):
                        tspec = specs[tidx]
                        ll_max = torch.full((x.shape[0],), -1e12, device=x.device)
                        for gc in tspec.global_classes:
                            astro = h_archive.get(gc)
                            if astro is not None:
                                ll_max = torch.maximum(ll_max, ll_fn(astro))
                        task_ll[:, tidx] = ll_max
                    best_task = task_ll.argmax(dim=1)
                    pred_full = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                    for tidx in range(tasks_seen):
                        mask = best_task == tidx
                        if not mask.any():
                            continue
                        tspec = specs[tidx]
                        tlogits = logits[mask][:, tspec.global_classes]
                        local_pred = tlogits.argmax(dim=1)
                        gc_t = torch.tensor(tspec.global_classes, dtype=torch.long)
                        pred_full[mask] = gc_t[local_pred]

            elif archive is not None and archive.n_classes > 0:
                # Perception-space manifold routing (fallback)
                code = x[:, :n_perc]
                task_ll = torch.zeros(x.shape[0], tasks_seen, device=x.device)
                for tidx in range(tasks_seen):
                    tspec = specs[tidx]
                    ll_sum = torch.zeros(x.shape[0], device=x.device)
                    for gc in tspec.global_classes:
                        astro = archive.get(gc)
                        if astro is not None:
                            ll_sum = ll_sum + astro.log_likelihood(code)
                    task_ll[:, tidx] = ll_sum / len(tspec.global_classes)

                best_task = task_ll.argmax(dim=1)
                pred_full = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                for tidx in range(tasks_seen):
                    mask = best_task == tidx
                    if not mask.any():
                        continue
                    tspec = specs[tidx]
                    tlogits = logits[mask][:, tspec.global_classes]
                    local_pred = tlogits.argmax(dim=1)
                    gc_t = torch.tensor(tspec.global_classes, dtype=torch.long)
                    pred_full[mask] = gc_t[local_pred]
            else:
                pred_full = logits.argmax(dim=1)

            full_acc = (pred_full == y).float().mean().item()

            # Task-aware accuracy (argmax restricted to this task's classes)
            task_logits = logits[:, spec.global_classes]
            pred_task_local = task_logits.argmax(dim=1)
            gc_tensor = torch.tensor(spec.global_classes, dtype=torch.long)
            pred_task_global = gc_tensor[pred_task_local]
            task_acc = (pred_task_global == y).float().mean().item()

        result.per_task.append(TaskResult(
            task_idx=t_idx,
            task_name=spec.name,
            full_acc=full_acc,
            task_acc=task_acc,
        ))

    return result


# ── Training loop ────────────────────────────────────────────────

def train_one_task(
    sub,
    credit: CreditTracker,
    frust: FrustrationDetector,
    archive: ManifoldArchive,
    train_view,
    spec: ChainedTaskSpec,
    task_idx: int,
    epochs: int,
    recorder: Recorder | None = None,
    *,
    use_sparsity: bool = True,
    detector_id: int = -1,
    use_cluster_replay: bool = True,
    use_task_replay: bool = True,
    output_anchor: OutputAnchor | None = None,
    anchor_lambda: float = ANCHOR_LAMBDA,
    h_archive: ManifoldArchive | None = None,
    h_interior_ids: torch.Tensor | None = None,
    lr: float = LR,
    train_input_noise: float = 0.0,
    growth_cfg: GrowthConfig | None = None,
):
    """Train one task with frustration-gated growth and manifold collection."""
    growth_cfg = growth_cfg or GrowthConfig()
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)

    if recorder:
        recorder.on_task_start(sub.arena, task_idx)

    code_boundary = []
    for cid in sub.arena.alive_ids().tolist():
        if has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION):
            code_boundary.append(cid)

    interior_ids = get_interior_ids(sub.arena).long()

    growth_budget = N_GROW_PER_TASK
    growth_count = 0
    frust_steps = 0

    for epoch in range(epochs):
        for x_batch, y_batch in train_view.iter_epoch(BATCH):
            if train_input_noise > 0:
                x_in = x_batch + train_input_noise * torch.randn_like(x_batch)
            else:
                x_in = x_batch
            logits = sub(x_in)
            loss = torch.nn.functional.cross_entropy(logits, y_batch)

            # Capture task H-activations now, before replay overwrites last_activations
            task_h_act = None
            if h_archive is not None and h_interior_ids is not None and sub.last_activations is not None:
                task_h_act = sub.last_activations[:, h_interior_ids].detach().clone()

            # Output-edge anchoring (Online EWC on H→output edges)
            if output_anchor is not None and anchor_lambda > 0:
                loss = loss + anchor_penalty(output_anchor, sub.arena, anchor_lambda)

            # Soft sparsity: L1 penalty on interior H-cell activations
            if use_sparsity and SPARSITY_LAMBDA > 0 and sub.live_activations is not None and interior_ids.numel() > 0:
                h_live = sub.live_activations[:, interior_ids]
                loss = loss + SPARSITY_LAMBDA * h_live.abs().mean()

            m = frust.step(loss.item())
            if frust.is_frustrated:
                frust_steps += 1

            scaled_loss = loss * m
            scaled_loss.backward()
            if _NONLINEAR:  # quad z² needs grad clipping for stability
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            sub.zero_dormant_grads()
            credit.update_utility()

            if sub.last_activations is not None:
                credit.update_engagement(sub.last_activations)

            opt.step()
            opt.zero_grad()

            # Replay of at-risk past tasks
            code = x_batch[:, :len(code_boundary)]
            if use_task_replay:
                if use_cluster_replay:
                    replay = cluster_replay_batch(
                        archive, spec.global_classes,
                        batch_size=BATCH, n_perc=len(code_boundary),
                        code_batch=code,
                    )
                else:
                    replay = interleaved_replay_batch(
                        archive, spec.global_classes,
                        batch_size=BATCH, n_perc=len(code_boundary),
                        code_batch=code,
                    )
                if replay is not None:
                    rx, ry = replay
                    r_logits = sub(rx)
                    r_loss = torch.nn.functional.cross_entropy(r_logits, ry)
                    r_loss.backward()
                    if _NONLINEAR:
                        torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                    sub.zero_dormant_grads()
                    opt.step()
                    opt.zero_grad()

            # Manifold collection — perception-space (for replay)
            for gc in spec.global_classes:
                mask = y_batch == gc
                if mask.any():
                    archive.update_class(gc, code[mask])

            # H-space manifold collection (for routing) — from captured task activations
            if task_h_act is not None:
                for gc in spec.global_classes:
                    mask = y_batch == gc
                    if mask.any():
                        h_archive.update_class(gc, task_h_act[mask])

            # Frustration-gated growth
            if growth_budget > 0 and frust.is_frustrated and frust_steps >= 25:
                interior = [cid for cid in sub.arena.alive_ids().tolist()
                            if not has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION)
                            and not has_gene(int(sub.arena.epigenome[cid].item()), OUTPUT)
                            and sub.arena.state[cid] == CellState.ACTIVE]
                if interior:
                    parent = interior[torch.randint(0, len(interior), (1,)).item()]
                    event = divide(sub.arena, parent, growth_cfg)
                    if event:
                        growth_budget -= 1
                        growth_count += 1
                        if recorder:
                            recorder.on_growth(sub.arena, task_idx, event.child_id)
                        sub.compile()
                        interior_ids = get_interior_ids(sub.arena).long()
                        opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
                        frust_steps = 0

    if recorder:
        recorder.on_task_end(sub.arena, task_idx)

    return growth_count


def train_detector(sub, detector_ids: list[int], train_view, archive, n_perc: int, epochs: int = 2):
    """Train multi-cell task detector: high for this task, low for past tasks."""
    a = sub.arena
    det_edges = []
    for ei in range(a.edge_cursor):
        if int(a.edge_dst[ei].item()) in detector_ids:
            det_edges.append(ei)

    if not det_edges:
        return

    det_weight_idx = torch.tensor(det_edges, dtype=torch.long)
    det_id_set = set(detector_ids)
    opt = torch.optim.Adam([a.bias, a.edge_weight], lr=1e-3)

    for epoch in range(epochs):
        for x_batch, y_batch in train_view.iter_epoch(BATCH):
            _ = sub(x_batch)
            det_acts = sub.live_activations[:, detector_ids]  # [B, n_det_cells]
            det_mean = det_acts.mean(dim=1)  # [B]
            pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                det_mean, torch.ones_like(det_mean))

            neg_loss = torch.tensor(0.0, device=a.device)
            past_classes = [c for c in archive.class_ids if archive.get(c) is not None]
            if past_classes:
                neg_samples = []
                for cid in past_classes[:8]:
                    astro = archive.get(cid)
                    s = astro.sample(4)
                    neg_samples.append(s[:, :n_perc] if s.shape[1] >= n_perc else
                                       torch.zeros(4, n_perc, device=a.device))
                neg_x = torch.cat(neg_samples)
                _ = sub(neg_x)
                neg_acts = sub.live_activations[:, detector_ids]
                neg_mean = neg_acts.mean(dim=1)
                neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    neg_mean, torch.zeros_like(neg_mean))

            loss = pos_loss + neg_loss
            loss.backward()

            if a.edge_weight.grad is not None:
                mask = torch.ones(a.edge_weight.shape[0], dtype=torch.bool)
                mask[det_weight_idx] = False
                a.edge_weight.grad[mask] = 0.0
            if a.bias.grad is not None:
                bias_mask = torch.ones(a.bias.shape[0], dtype=torch.bool)
                for did in detector_ids:
                    bias_mask[did] = False
                a.bias.grad[bias_mask] = 0.0

            opt.step()
            opt.zero_grad()


def calibrate_detectors(sub, detectors: list[list[int]], archive, specs, n_perc: int,
                        tasks_seen: int, epochs: int = CALIBRATE_EPOCHS):
    """Joint calibration: present examples from ALL tasks, train all detectors together.

    Each detector bank should fire high for its own task and low for all others.
    """
    a = sub.arena

    all_det_edges = []
    all_det_cell_ids = []
    for t_dets in detectors[:tasks_seen]:
        all_det_cell_ids.extend(t_dets)
        for ei in range(a.edge_cursor):
            if int(a.edge_dst[ei].item()) in t_dets:
                all_det_edges.append(ei)

    if not all_det_edges:
        return

    det_weight_idx = torch.tensor(list(set(all_det_edges)), dtype=torch.long)
    det_cell_set = set(all_det_cell_ids)

    for did in all_det_cell_ids:
        a.state[did] = CellState.ACTIVE
    sub.compile()

    opt = torch.optim.Adam([a.bias, a.edge_weight], lr=5e-4)

    for epoch in range(epochs):
        for target_task in range(tasks_seen):
            spec = specs[target_task]
            samples = []
            for gc in spec.global_classes:
                astro = archive.get(gc)
                if astro is not None:
                    s = astro.sample(8)
                    if s.shape[1] >= n_perc:
                        samples.append(s[:, :n_perc])
                    else:
                        padded = torch.zeros(8, n_perc, device=a.device)
                        padded[:, :s.shape[1]] = s
                        samples.append(padded)
            if not samples:
                continue
            x_batch = torch.cat(samples)

            _ = sub(x_batch)

            total_loss = torch.tensor(0.0, device=a.device)
            for tidx in range(tasks_seen):
                cell_ids = detectors[tidx]
                det_acts = sub.live_activations[:, cell_ids].mean(dim=1)
                target = 1.0 if tidx == target_task else 0.0
                total_loss = total_loss + torch.nn.functional.binary_cross_entropy_with_logits(
                    det_acts, torch.full_like(det_acts, target))

            total_loss.backward()

            if a.edge_weight.grad is not None:
                mask = torch.ones(a.edge_weight.shape[0], dtype=torch.bool)
                mask[det_weight_idx] = False
                a.edge_weight.grad[mask] = 0.0
            if a.bias.grad is not None:
                bias_mask = torch.ones(a.bias.shape[0], dtype=torch.bool)
                for did in all_det_cell_ids:
                    bias_mask[did] = False
                a.bias.grad[bias_mask] = 0.0

            opt.step()
            opt.zero_grad()

    for did in all_det_cell_ids:
        a.state[did] = CellState.DORMANT
    sub.compile()


def refresh_h_archive(sub, bundle, specs, tasks_seen, h_interior_ids, full_cov=False,
                      source="real", perc_archive=None, n_perc=0, samples_per_class=400,
                      perc_full_sample=False, perc_sample_rank=None, perc_jitter=0.0):
    """Rebuild the H-space manifold from a fresh pass through the CURRENT substrate.

    Fixes stale statistics: H-cell activations drift as later tasks train, so
    statistics collected during each task become outdated. This re-collects
    per-class (mu, sigma[, Σ]) using the final substrate's representation.

    source="real":     forward real training data (oracle ceiling — needs past data).
    source="manifold": sample synthetic inputs from the storage-free perception
                       manifold, forward through the current substrate. No past
                       data retained — the honest continual-learning path.
    """
    fresh = ManifoldArchive(sub.arena, full_cov=full_cov)

    if source == "manifold":
        with torch.no_grad():
            for t_idx in range(tasks_seen):
                spec = specs[t_idx]
                for gc in spec.global_classes:
                    astro = perc_archive.get(gc) if perc_archive is not None else None
                    if astro is None:
                        continue
                    syn = None
                    if perc_archive is not None and perc_archive.mixture_k > 0:
                        syn = perc_archive.sample_mixture(gc, samples_per_class)
                    if syn is None:
                        if perc_full_sample:
                            syn = astro.sample_full(samples_per_class, rank=perc_sample_rank)
                        else:
                            syn = astro.sample(samples_per_class)  # [N, perc_dim] in pixel space
                    if perc_jitter > 0:
                        syn = syn + perc_jitter * torch.randn_like(syn)
                    x_in = torch.zeros(syn.shape[0], n_perc, device=sub.arena.device)
                    w = min(syn.shape[1], n_perc)
                    x_in[:, :w] = syn[:, :w]
                    _ = sub(x_in)
                    h_act = sub.last_activations[:, h_interior_ids]
                    fresh.update_class(gc, h_act)
        fresh.finalize_all()
        return fresh

    for t_idx in range(tasks_seen):
        spec = specs[t_idx]
        train_view = bundle.task_view(
            spec.dataset_name, spec.local_classes, spec.global_classes,
            split="train", task_name=spec.name,
        )
        with torch.no_grad():
            for x_batch, y_batch in train_view.iter_epoch(BATCH):
                _ = sub(x_batch)
                h_act = sub.last_activations[:, h_interior_ids]
                for gc in spec.global_classes:
                    mask = y_batch == gc
                    if mask.any():
                        fresh.update_class(gc, h_act[mask])
    fresh.finalize_all()
    return fresh


# ── Discriminative H-space router ─────────────────────────────────


class HRouter(nn.Module):
    """Discriminative classifier over forwarded H-vectors.

    The generative QDA router keeps only per-class (mu, Sigma); this keeps the
    full nonlinear sample distribution. Trained on the same forwarded H-vectors
    the refresh collects, so it is storage-free under source=manifold.
    """

    def __init__(self, in_dim: int, n_out: int, kind: str = "mlp", hidden: int = 64):
        super().__init__()
        if kind == "logistic":
            self.net = nn.Linear(in_dim, n_out)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, n_out),
            )

    def forward(self, x):
        return self.net(x)


def collect_h_samples(sub, bundle, specs, tasks_seen, h_interior_ids, source,
                      perc_archive=None, n_perc=0, samples_per_class=400,
                      perc_full_sample=False, perc_sample_rank=None, perc_jitter=0.0):
    """Forward synthetic (storage-free) or real (oracle) inputs through the CURRENT
    substrate and collect H-vectors with class+task labels — the training set for a
    discriminative router. Mirrors refresh_h_archive's sampling so the router and the
    QDA manifold see identical data.
    """
    Xs, ycs, yts = [], [], []
    dev = sub.arena.device
    with torch.no_grad():
        if source == "manifold":
            for t_idx in range(tasks_seen):
                for gc in specs[t_idx].global_classes:
                    astro = perc_archive.get(gc) if perc_archive is not None else None
                    if astro is None:
                        continue
                    syn = None
                    if perc_archive is not None and perc_archive.mixture_k > 0:
                        syn = perc_archive.sample_mixture(gc, samples_per_class)
                    if syn is None:
                        if perc_full_sample:
                            syn = astro.sample_full(samples_per_class, rank=perc_sample_rank)
                        else:
                            syn = astro.sample(samples_per_class)
                    if perc_jitter > 0:
                        syn = syn + perc_jitter * torch.randn_like(syn)
                    x_in = torch.zeros(syn.shape[0], n_perc, device=dev)
                    w = min(syn.shape[1], n_perc)
                    x_in[:, :w] = syn[:, :w]
                    _ = sub(x_in)
                    h = sub.last_activations[:, h_interior_ids]
                    Xs.append(h)
                    ycs.append(torch.full((h.shape[0],), gc, dtype=torch.long, device=dev))
                    yts.append(torch.full((h.shape[0],), t_idx, dtype=torch.long, device=dev))
        else:  # real — oracle ceiling
            for t_idx in range(tasks_seen):
                spec = specs[t_idx]
                train_view = bundle.task_view(
                    spec.dataset_name, spec.local_classes, spec.global_classes,
                    split="train", task_name=spec.name,
                )
                for x_batch, y_batch in train_view.iter_epoch(BATCH):
                    _ = sub(x_batch)
                    h = sub.last_activations[:, h_interior_ids]
                    for gc in spec.global_classes:
                        mask = y_batch == gc
                        if mask.any():
                            nk = int(mask.sum())
                            Xs.append(h[mask])
                            ycs.append(torch.full((nk,), gc, dtype=torch.long, device=dev))
                            yts.append(torch.full((nk,), t_idx, dtype=torch.long, device=dev))
    return torch.cat(Xs), torch.cat(ycs), torch.cat(yts)


def train_h_router(X, y, n_out, kind="mlp", hidden=64, epochs=60, lr=1e-2, batch=256, seed=0):
    """Train a discriminative router on standardized H-vectors. Returns (model, mu, sd)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu = X.mean(dim=0)
    sd = X.std(dim=0) + 1e-6
    Xn = (X - mu) / sd
    model = HRouter(X.shape[1], n_out, kind=kind, hidden=hidden).to(X.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = Xn.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(X.device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Xn[idx]), y[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model, mu, sd


def evaluate_router(sub, bundle, specs, tasks_seen, h_interior_ids,
                    router, mu, sd, granularity):
    """Full + task-aware accuracy using a discriminative H-router for routing.

    granularity='class': router argmax over seen global classes (bypasses head).
    granularity='task' : router picks the task, output head picks class within it.
    """
    result = EvalResult(after_task=tasks_seen - 1)
    seen_classes = []
    for ti in range(tasks_seen):
        seen_classes.extend(specs[ti].global_classes)
    for t_idx in range(tasks_seen):
        spec = specs[t_idx]
        test_view = bundle.task_view(
            spec.dataset_name, spec.local_classes, spec.global_classes,
            split="test", task_name=spec.name,
        )
        x, y = test_view.all_examples()
        with torch.no_grad():
            logits = sub(x)
            h = (sub.last_activations[:, h_interior_ids] - mu) / sd
            r = router(h)
            if granularity == "class":
                colmask = torch.full((r.shape[1],), float("-inf"), device=r.device)
                colmask[seen_classes] = 0.0
                pred_full = (r + colmask).argmax(dim=1)
            else:
                best_task = r[:, :tasks_seen].argmax(dim=1)
                pred_full = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
                for ti in range(tasks_seen):
                    m = best_task == ti
                    if not m.any():
                        continue
                    tspec = specs[ti]
                    gc_t = torch.tensor(tspec.global_classes, dtype=torch.long)
                    pred_full[m] = gc_t[logits[m][:, tspec.global_classes].argmax(dim=1)]
            full_acc = (pred_full == y).float().mean().item()
            gc_tensor = torch.tensor(spec.global_classes, dtype=torch.long)
            pred_task = gc_tensor[logits[:, spec.global_classes].argmax(dim=1)]
            task_acc = (pred_task == y).float().mean().item()
        result.per_task.append(TaskResult(
            task_idx=t_idx, task_name=spec.name,
            full_acc=full_acc, task_acc=task_acc,
        ))
    return result


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chained-15 v2.0 bench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="2 epochs, fast")
    parser.add_argument("--viz", action="store_true", help="generate HTML viewer")
    parser.add_argument("--no-sparsity", action="store_true", help="disable L1 sparsity penalty")
    parser.add_argument("--no-kibra", action="store_true", help="disable KIBRA edge tagging")
    parser.add_argument("--no-routing", action="store_true", help="disable manifold routing at eval")
    parser.add_argument("--legacy-replay", action="store_true", help="use interleaved replay instead of cluster replay")
    parser.add_argument("--no-dream-replay", action="store_true", help="skip replay stage in dream cycle")
    parser.add_argument("--no-private-cells", action="store_true", help="disable per-class private cells")
    parser.add_argument("--anchor-lambda", type=float, default=ANCHOR_LAMBDA,
                        help=f"output-edge anchoring strength (0=off, default={ANCHOR_LAMBDA})")
    parser.add_argument("--no-anchor", action="store_true", help="disable output-edge anchoring")
    parser.add_argument("--det-cells", type=int, default=N_DET_CELLS,
                        help=f"detector cells per task (default={N_DET_CELLS})")
    parser.add_argument("--no-calibrate", action="store_true", help="skip joint detector calibration")
    parser.add_argument("--no-task-replay", action="store_true", help="disable in-task replay (manifold collection continues)")
    parser.add_argument("--h-routing", action="store_true", help="use H-space manifold for routing instead of perception-space")
    parser.add_argument("--refresh-h", action="store_true", help="rebuild H-manifold from current substrate after training (fixes stale stats)")
    parser.add_argument("--full-cov", action="store_true", help="full-covariance Mahalanobis H-routing instead of diagonal Gaussian")
    parser.add_argument("--h-route-mode", choices=["task", "class"], default="task", help="H-routing granularity (default task)")
    parser.add_argument("--refresh-source", choices=["real", "manifold"], default="real",
                        help="refresh H-stats from real data (oracle ceiling) or perception manifold (storage-free)")
    parser.add_argument("--perc-full-cov", action="store_true",
                        help="accumulate full pixel covariance in the perception manifold and sample correlated pixels during storage-free refresh")
    parser.add_argument("--perc-sample-rank", type=int, default=0,
                        help="rank-k truncation for correlated pixel sampling (0=full covariance)")
    parser.add_argument("--perc-mixture-k", type=int, default=0,
                        help="K diag-Gaussian sub-clusters per class in the perception archive (0=single Gaussian)")
    parser.add_argument("--perc-jitter", type=float, default=0.0,
                        help="Gaussian noise sigma added to synthetic pixels before forwarding (storage-free aug)")
    parser.add_argument("--h-router", choices=["qda", "logistic", "mlp"], default="qda",
                        help="discriminative H-router probe after refresh (qda=generative only, no router)")
    parser.add_argument("--router-samples", type=int, default=400,
                        help="synthetic H-vectors per class for storage-free router training (manifold source)")
    parser.add_argument("--router-hidden", type=int, default=64, help="router MLP hidden width")
    parser.add_argument("--router-epochs", type=int, default=60, help="router training epochs")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"substrate optimizer LR (default {LR}); 2x = ~1.34e-3 for less-precise training probe")
    parser.add_argument("--train-input-noise", type=float, default=0.0,
                        help="Gaussian noise sigma added to training inputs only (eval/replay unaffected) — upstream regularization probe")
    parser.add_argument("--h-init", type=int, default=H_INIT,
                        help=f"interior (H-space) cells at substrate construction (default {H_INIT})")
    parser.add_argument("--self-arrange", action="store_true",
                        help="relax divide() rank policy so the substrate self-organizes depth "
                             "(interior↔interior edges); the trioron-native depth path")
    parser.add_argument("--interior-layers", type=int, default=1,
                        help="number of stacked bipartite interior layers (1=current, 2+ adds compositional depth)")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 2

    torch.manual_seed(args.seed)

    use_sparsity = not args.no_sparsity
    use_kibra = not args.no_kibra
    use_routing = not args.no_routing
    use_cluster_replay = not args.legacy_replay
    use_dream_replay = not args.no_dream_replay
    use_private_cells = not args.no_private_cells
    use_anchor = not args.no_anchor
    anchor_lambda = args.anchor_lambda if use_anchor else 0.0
    use_calibrate = not args.no_calibrate and use_private_cells
    use_task_replay = not args.no_task_replay
    use_h_routing = args.h_routing
    use_refresh_h = args.refresh_h and use_h_routing
    use_full_cov = args.full_cov and use_h_routing
    h_route_mode = args.h_route_mode
    refresh_source = args.refresh_source
    use_perc_full_cov = args.perc_full_cov
    perc_sample_rank = args.perc_sample_rank if args.perc_sample_rank > 0 else None
    perc_mixture_k = args.perc_mixture_k
    perc_jitter = args.perc_jitter
    h_router_kind = args.h_router
    growth_cfg = GrowthConfig(same_rank_edges=args.self_arrange)

    flags = []
    if args.self_arrange:
        flags.append("self-arrange")
    if not use_sparsity:
        flags.append("no-sparsity")
    if not use_kibra:
        flags.append("no-kibra")
    if not use_routing:
        flags.append("no-routing")
    if not use_cluster_replay:
        flags.append("legacy-replay")
    if not use_dream_replay:
        flags.append("no-dream-replay")
    if not use_task_replay:
        flags.append("no-task-replay")
    if not use_private_cells:
        flags.append("no-private-cells")
    if use_anchor:
        flags.append(f"anchor-λ={anchor_lambda}")
    else:
        flags.append("no-anchor")
    if use_private_cells:
        flags.append(f"det-cells={args.det_cells}")
        if use_calibrate:
            flags.append("calibrate")
    if use_h_routing:
        flags.append(f"h-routing({h_route_mode})")
    if use_full_cov:
        flags.append("full-cov")
    if use_refresh_h:
        flags.append(f"refresh-h({refresh_source})")
    if use_perc_full_cov:
        rank_lbl = "full" if perc_sample_rank is None else f"rank{perc_sample_rank}"
        flags.append(f"perc-cov({rank_lbl})")
    if perc_mixture_k > 0:
        flags.append(f"perc-mix(K={perc_mixture_k})")
    if perc_jitter > 0:
        flags.append(f"perc-jit({perc_jitter})")
    if args.lr != LR:
        flags.append(f"lr={args.lr:g}")
    if args.train_input_noise > 0:
        flags.append(f"train-noise({args.train_input_noise})")
    if args.h_init != H_INIT:
        flags.append(f"h-init={args.h_init}")
    if args.interior_layers != 1:
        flags.append(f"interior-layers={args.interior_layers}")
    if h_router_kind != "qda":
        flags.append(f"h-router({h_router_kind})")
    flag_str = f" [{', '.join(flags)}]"

    print("=" * 60)
    print(f"Chained-15 v2.0 — seed={args.seed}, epochs={args.epochs}{flag_str}")
    print("=" * 60)

    # Load data
    specs = chained_15_specs()
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"])

    # Build substrate
    sub = build_substrate(args.seed, h_init=args.h_init, interior_layers=args.interior_layers)
    detectors = []
    if use_private_cells:
        detectors = add_task_detectors(sub, n_tasks=len(specs),
                                       n_det_cells=args.det_cells, seed=args.seed)
    print(f"Initial substrate: {sub.n_cells} cells, {sub.n_edges} edges")

    # Fixed interior cell set for H-space routing (captured before any growth)
    from trioron.learning.manifold import get_interior_ids as _gii
    h_interior_ids = _gii(sub.arena).long().clone() if use_h_routing else None

    # Learning components
    credit = CreditTracker(sub.arena)
    frust = FrustrationDetector()
    archive = ManifoldArchive(sub.arena, full_cov=use_perc_full_cov, mixture_k=perc_mixture_k)
    h_archive = ManifoldArchive(sub.arena, full_cov=use_full_cov) if use_h_routing else None
    output_anchor: OutputAnchor | None = None

    # Recorder
    recorder = None
    if args.viz:
        recorder = Recorder("runs/chained15/snapshots/", sample_growth_every_n=1)

    # Train
    eval_history: list[EvalResult] = []
    t_start = time.time()

    for task_idx, spec in enumerate(specs):
        t_task = time.time()

        train_view = bundle.task_view(
            spec.dataset_name, spec.local_classes, spec.global_classes,
            split="train", task_name=spec.name,
        )

        print(f"\n--- Task {task_idx}: {spec.name} "
              f"(classes {spec.global_classes}) ---")

        # Activate this task's detector cells
        if use_private_cells and detectors:
            for did in detectors[task_idx]:
                sub.arena.state[did] = CellState.ACTIVE
            sub.compile()

        grown = train_one_task(
            sub, credit, frust, archive, train_view, spec,
            task_idx, args.epochs, recorder,
            use_sparsity=use_sparsity,
            use_cluster_replay=use_cluster_replay,
            use_task_replay=use_task_replay,
            output_anchor=output_anchor,
            anchor_lambda=anchor_lambda,
            h_archive=h_archive,
            h_interior_ids=h_interior_ids,
            lr=args.lr,
            train_input_noise=args.train_input_noise,
            growth_cfg=growth_cfg,
        )

        # Dream cycle
        archive.finalize_all()
        if h_archive is not None:
            h_archive.finalize_all()
        dream_cfg = DreamConfig()
        if not use_dream_replay:
            dream_cfg.replay_steps_per_class = 0
        dream_result = dream_cycle(
            sub, credit, archive,
            current_classes=spec.global_classes,
            cfg=dream_cfg,
        )

        # KIBRA: one-shot edge tagging for cluster-level protection
        n_tagged = kibra_tag(sub, archive) if use_kibra else 0

        if recorder:
            recorder.on_dream(sub.arena, task_idx)

        # Update output-edge anchor (Online EWC on H→output edges)
        if use_anchor:
            output_anchor = update_anchor(output_anchor, sub, train_view)
            print(f"  anchor: {output_anchor.edge_idx.shape[0]} edges, "
                  f"fisher max={output_anchor.fisher.max():.4g}, "
                  f"mean={output_anchor.fisher.mean():.4g}")

        # Train and freeze this task's detector cells
        if use_private_cells and detectors:
            n_perc = sum(1 for cid in range(sub.arena.cursor)
                         if sub.arena.alive[cid] and has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION))
            train_detector(sub, detectors[task_idx], train_view, archive, n_perc)
            for did in detectors[task_idx]:
                sub.arena.state[did] = CellState.DORMANT

        frust.reset()
        sub.end_task()
        sub.compile()

        # Re-open manifold astrocytes for future collection
        for gc in spec.global_classes:
            astro = archive.get(gc)
            if astro and sub.arena.state[astro.cell_id] == CellState.DORMANT:
                sub.arena.state[astro.cell_id] = CellState.ACTIVE
            if h_archive is not None:
                h_astro = h_archive.get(gc)
                if h_astro and sub.arena.state[h_astro.cell_id] == CellState.DORMANT:
                    sub.arena.state[h_astro.cell_id] = CellState.ACTIVE

        # Evaluate (detector routing > H-space routing > perception routing > raw)
        ev = evaluate_all_tasks(
            sub, bundle, specs, task_idx + 1,
            archive if use_routing else None,
            detectors if use_private_cells else None,
            h_archive=h_archive if use_h_routing else None,
            h_interior_ids=h_interior_ids,
            h_route_mode=h_route_mode,
            h_full_cov=use_full_cov,
        )
        eval_history.append(ev)

        n_protected = int(sub.arena.edge_protected[:sub.arena.edge_cursor].sum().item())
        n_clusters = len(archive.clusters)
        elapsed = time.time() - t_task
        print(f"  cells={sub.n_cells}, edges={sub.n_edges}, "
              f"locked={dream_result.n_locked}, grown={grown}, "
              f"tagged={n_tagged}, protected={n_protected}, clusters={n_clusters}")
        print(f"  mean full={ev.mean_full:.4f}, "
              f"mean task-aware={ev.mean_task:.4f}  ({elapsed:.1f}s)")

    # Refresh H-manifold from current substrate (fixes stale statistics)
    if use_refresh_h:
        print(f"\n--- Refreshing H-manifold ({refresh_source}) ---")
        n_perc_r = sum(1 for cid in range(sub.arena.cursor)
                       if sub.arena.alive[cid] and has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION))
        h_archive = refresh_h_archive(sub, bundle, specs, len(specs), h_interior_ids,
                                      full_cov=use_full_cov, source=refresh_source,
                                      perc_archive=archive, n_perc=n_perc_r,
                                      perc_full_sample=use_perc_full_cov,
                                      perc_sample_rank=perc_sample_rank,
                                      perc_jitter=perc_jitter)
        ev_refresh = evaluate_all_tasks(
            sub, bundle, specs, len(specs),
            None, None,
            h_archive=h_archive,
            h_interior_ids=h_interior_ids,
            h_route_mode=h_route_mode,
            h_full_cov=use_full_cov,
        )
        eval_history.append(ev_refresh)
        print(f"  POST-REFRESH (QDA): mean full={ev_refresh.mean_full:.4f}, "
              f"mean task-aware={ev_refresh.mean_task:.4f}")

        # Discriminative H-router probe: trains on the same forwarded H-vectors,
        # compares task-level (route 15-way, head classifies) vs class-level (30-way).
        if h_router_kind != "qda":
            print(f"\n--- Discriminative H-router probe ({h_router_kind}, source={refresh_source}) ---")
            X, y_class, y_task = collect_h_samples(
                sub, bundle, specs, len(specs), h_interior_ids, refresh_source,
                perc_archive=archive, n_perc=n_perc_r, samples_per_class=args.router_samples,
                perc_full_sample=use_perc_full_cov, perc_sample_rank=perc_sample_rank,
                perc_jitter=perc_jitter,
            )
            print(f"  collected {X.shape[0]} H-vectors ({X.shape[1]}-d)")
            if perc_mixture_k > 0:
                print(f"  [mixture diagnostics K={perc_mixture_k}] per-class cluster n:")
                for ti in range(len(specs)):
                    for gc in specs[ti].global_classes:
                        mx = archive._mixtures.get(gc)
                        if mx is None:
                            continue
                        ns = mx.n.tolist()
                        wt = [n / max(1.0, sum(ns)) for n in ns]
                        print(f"    cls {gc:2d}: n={ns}  w=[{', '.join('%.2f' % w for w in wt)}]")
            for gran, n_out, labels in (("task", len(specs), y_task),
                                        ("class", N_GLOBAL_CLASSES, y_class)):
                model, mu, sd = train_h_router(X, labels, n_out, kind=h_router_kind,
                                               hidden=args.router_hidden, epochs=args.router_epochs,
                                               seed=args.seed)
                ev_r = evaluate_router(sub, bundle, specs, len(specs), h_interior_ids,
                                       model, mu, sd, gran)
                print(f"  POST-ROUTER ({gran}): mean full={ev_r.mean_full:.4f}, "
                      f"mean task-aware={ev_r.mean_task:.4f}")

    # Joint detector calibration after all tasks
    if use_calibrate and detectors:
        print("\n--- Joint detector calibration ---")
        n_perc = sum(1 for cid in range(sub.arena.cursor)
                     if sub.arena.alive[cid] and has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION))
        calibrate_detectors(sub, detectors, archive, specs, n_perc, len(specs))

        # Re-evaluate after calibration
        ev_cal = evaluate_all_tasks(
            sub, bundle, specs, len(specs),
            archive if use_routing else None,
            detectors,
            h_archive=h_archive if use_h_routing else None,
            h_interior_ids=h_interior_ids,
            h_route_mode=h_route_mode,
            h_full_cov=use_full_cov,
        )
        eval_history.append(ev_cal)
        print(f"  POST-CALIBRATION: mean full={ev_cal.mean_full:.4f}, "
              f"mean task-aware={ev_cal.mean_task:.4f}")

    total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"FINAL after all 15 tasks ({total:.1f}s total)")
    print("=" * 60)

    final = eval_history[-1]
    print(f"\n{'Task':<25} {'Full':>8} {'Task-Aware':>10}")
    print("-" * 45)
    for tr in final.per_task:
        print(f"{tr.task_name:<25} {tr.full_acc:>8.4f} {tr.task_acc:>10.4f}")
    print("-" * 45)
    print(f"{'MEAN':<25} {final.mean_full:>8.4f} {final.mean_task:>10.4f}")
    print(f"\nSubstrate: {sub.n_cells} cells, {sub.n_edges} edges, "
          f"{sub.arena.param_bytes} param bytes")

    # Export viewer
    if args.viz and recorder:
        structs = detect_from_directory("runs/chained15/snapshots/")
        path = export_html("runs/chained15/snapshots/", "runs/chained15/viewer.html", structs)
        print(f"\nViewer exported: {path}")


if __name__ == "__main__":
    main()
