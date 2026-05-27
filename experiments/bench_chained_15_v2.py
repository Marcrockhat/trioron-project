"""Chained-15 continual learning benchmark on the v2.0 substrate.

MNIST → FashionMNIST → EMNIST-letters, 15 binary tasks, 30 global classes.
Tests credit-based locking + manifold replay + frustration-gated growth.

Usage:
    python3 -m experiments.bench_chained_15_v2 [--seed 42] [--epochs 4] [--smoke]
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field

import torch

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


# ── Substrate construction ───────────────────────────────────────

def build_substrate(seed: int = 42):
    torch.manual_seed(seed)
    sub = construct(
        base=seeded(IMAGE_DIM, N_GLOBAL_CLASSES, interior_cells=H_INIT),
        envelope=Envelope(max_parameter_bytes=PARAM_CAP_BYTES),
        dispatch_table=default_dispatch_table(),
        capacity=2048,
        sparsity_k=0,
    )
    return sub


def add_task_detectors(sub, n_tasks: int = 15, seed: int = 42):
    """Allocate one task-detector cell per task with full perception input.

    Each detector is a linear neuron that learns to fire HIGH for its
    task's inputs. Starts DORMANT; activated during training, then
    re-frozen. At eval, the highest-scoring detector selects the task,
    and task-aware logits decide the class.

    Returns list of detector cell ids (one per task).
    """
    a = sub.arena

    perc_ids = [cid for cid in range(a.cursor)
                if a.alive[cid] and has_gene(int(a.epigenome[cid].item()), PERCEPTION)]

    detectors = []
    for t in range(n_tasks):
        pid = int(a.alloc(1)[0].item())
        a.rank[pid] = 1
        a.position[pid] = torch.tensor([0.5, t / max(n_tasks - 1, 1), 0.9])
        a.state[pid] = CellState.DORMANT
        a.forward_inclusion[pid] = True

        src = torch.tensor(perc_ids[:PRIVATE_FAN_IN], dtype=torch.int32)
        dst = torch.full((len(src),), pid, dtype=torch.int32)
        a.add_edges(src, dst)
        detectors.append(pid)

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
    detectors: list[int] | None = None,
) -> EvalResult:
    """Evaluate full-softmax (with optional routing) and task-aware accuracy."""
    result = EvalResult(after_task=tasks_seen - 1)

    n_perc = 0
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

            # Detector-based routing: pick the task whose detector fires highest
            if detectors and len(detectors) >= tasks_seen:
                det_ids = [detectors[tidx] for tidx in range(tasks_seen)]
                det_scores = act[:, det_ids]  # [B, tasks_seen]
                best_task = det_scores.argmax(dim=1)  # [B]

                # For each sample, use task-aware logits from the detected task
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

                task_gates = torch.nn.functional.softmax(task_ll, dim=1)
                routed = logits.clone()
                for tidx in range(tasks_seen):
                    tspec = specs[tidx]
                    for gc in tspec.global_classes:
                        routed[:, gc] = logits[:, gc] + torch.log(task_gates[:, tidx] + 1e-10)

                pred_full = routed.argmax(dim=1)
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
):
    """Train one task with frustration-gated growth and manifold collection."""
    sub.prepare_training()
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)

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
            logits = sub(x_batch)
            loss = torch.nn.functional.cross_entropy(logits, y_batch)

            # Soft sparsity: L1 penalty on interior H-cell activations
            if use_sparsity and SPARSITY_LAMBDA > 0 and sub.live_activations is not None and interior_ids.numel() > 0:
                h_live = sub.live_activations[:, interior_ids]
                loss = loss + SPARSITY_LAMBDA * h_live.abs().mean()

            m = frust.step(loss.item())
            if frust.is_frustrated:
                frust_steps += 1

            scaled_loss = loss * m
            scaled_loss.backward()
            sub.zero_dormant_grads()
            credit.update_utility()

            if sub.last_activations is not None:
                credit.update_engagement(sub.last_activations)

            opt.step()
            opt.zero_grad()

            # Replay of at-risk past tasks
            code = x_batch[:, :len(code_boundary)]
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
                sub.zero_dormant_grads()
                opt.step()
                opt.zero_grad()

            # Manifold collection (code already computed above)
            for gc in spec.global_classes:
                mask = y_batch == gc
                if mask.any():
                    archive.update_class(gc, code[mask])

            # Frustration-gated growth
            if growth_budget > 0 and frust.is_frustrated and frust_steps >= 25:
                interior = [cid for cid in sub.arena.alive_ids().tolist()
                            if not has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION)
                            and not has_gene(int(sub.arena.epigenome[cid].item()), OUTPUT)
                            and sub.arena.state[cid] == CellState.ACTIVE]
                if interior:
                    parent = interior[torch.randint(0, len(interior), (1,)).item()]
                    event = divide(sub.arena, parent)
                    if event:
                        growth_budget -= 1
                        growth_count += 1
                        if recorder:
                            recorder.on_growth(sub.arena, task_idx, event.child_id)
                        sub.compile()
                        interior_ids = get_interior_ids(sub.arena).long()
                        opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)
                        frust_steps = 0

    if recorder:
        recorder.on_task_end(sub.arena, task_idx)

    return growth_count


def train_detector(sub, detector_id: int, train_view, archive, n_perc: int, epochs: int = 2):
    """Train a task detector cell: high for this task, low for past tasks.

    Only the detector cell's input edges are updated (everything else frozen
    by zero_dormant_grads since only the detector is ACTIVE among private cells).
    """
    a = sub.arena
    det_edges = []
    for ei in range(a.edge_cursor):
        if int(a.edge_dst[ei].item()) == detector_id:
            det_edges.append(ei)

    if not det_edges:
        return

    det_weight_idx = torch.tensor(det_edges, dtype=torch.long)
    opt = torch.optim.Adam([a.bias, a.edge_weight], lr=1e-3)

    for epoch in range(epochs):
        for x_batch, y_batch in train_view.iter_epoch(BATCH):
            _ = sub(x_batch)
            det_act = sub.live_activations[:, detector_id]
            pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                det_act, torch.ones_like(det_act))

            neg_loss = torch.tensor(0.0, device=a.device)
            past_classes = [c for c in archive.class_ids if archive.arena.state[archive.get(c).cell_id] == CellState.DORMANT]
            if past_classes:
                neg_samples = []
                for cid in past_classes[:8]:
                    astro = archive.get(cid)
                    s = astro.sample(4)
                    neg_samples.append(s[:, :n_perc] if s.shape[1] >= n_perc else
                                       torch.zeros(4, n_perc, device=a.device))
                neg_x = torch.cat(neg_samples)
                _ = sub(neg_x)
                neg_det = sub.live_activations[:, detector_id]
                neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    neg_det, torch.zeros_like(neg_det))

            loss = pos_loss + neg_loss
            loss.backward()

            # Zero all grads except detector edges and bias
            if a.edge_weight.grad is not None:
                mask = torch.ones(a.edge_weight.shape[0], dtype=torch.bool)
                mask[det_weight_idx] = False
                a.edge_weight.grad[mask] = 0.0
            if a.bias.grad is not None:
                bias_mask = torch.ones(a.bias.shape[0], dtype=torch.bool)
                bias_mask[detector_id] = False
                a.bias.grad[bias_mask] = 0.0

            opt.step()
            opt.zero_grad()


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

    flags = []
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
    if not use_private_cells:
        flags.append("no-private-cells")
    flag_str = f" [{', '.join(flags)}]" if flags else " [full stack]"

    print("=" * 60)
    print(f"Chained-15 v2.0 — seed={args.seed}, epochs={args.epochs}{flag_str}")
    print("=" * 60)

    # Load data
    specs = chained_15_specs()
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"])

    # Build substrate
    sub = build_substrate(args.seed)
    detectors = []
    if use_private_cells:
        detectors = add_task_detectors(sub, n_tasks=len(chained_15_specs()), seed=args.seed)
    print(f"Initial substrate: {sub.n_cells} cells, {sub.n_edges} edges")

    # Learning components
    credit = CreditTracker(sub.arena)
    frust = FrustrationDetector()
    archive = ManifoldArchive(sub.arena)

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

        # Activate this task's detector cell
        if use_private_cells and detectors:
            sub.arena.state[detectors[task_idx]] = CellState.ACTIVE
            sub.compile()

        det_id = detectors[task_idx] if (use_private_cells and detectors) else -1
        grown = train_one_task(
            sub, credit, frust, archive, train_view, spec,
            task_idx, args.epochs, recorder,
            use_sparsity=use_sparsity,
            use_cluster_replay=use_cluster_replay,
            detector_id=det_id,
        )

        # Dream cycle
        archive.finalize_all()
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

        # Train and freeze this task's detector cell
        if use_private_cells and detectors:
            n_perc = sum(1 for cid in range(sub.arena.cursor)
                         if sub.arena.alive[cid] and has_gene(int(sub.arena.epigenome[cid].item()), PERCEPTION))
            train_detector(sub, detectors[task_idx], train_view, archive, n_perc)
            sub.arena.state[detectors[task_idx]] = CellState.DORMANT

        frust.reset()
        sub.end_task()
        sub.compile()

        # Re-open manifold astrocytes for future collection
        for gc in spec.global_classes:
            astro = archive.get(gc)
            if astro and sub.arena.state[astro.cell_id] == CellState.DORMANT:
                sub.arena.state[astro.cell_id] = CellState.ACTIVE

        # Evaluate (detector routing > manifold routing > raw logits)
        ev = evaluate_all_tasks(
            sub, bundle, specs, task_idx + 1,
            archive if use_routing else None,
            detectors if use_private_cells else None,
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
