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
from trioron.lifecycle import divide, GrowthConfig
from trioron.viz import Recorder, export_html
from trioron.viz.detect import detect_from_directory


# ── Config ───────────────────────────────────────────────────────

N_GLOBAL_CLASSES = 30
H_INIT = 55
BATCH = 30
LR = 6.68e-4
N_GROW_PER_TASK = 9
PARAM_CAP_BYTES = 200_000


# ── Substrate construction ───────────────────────────────────────

def build_substrate(seed: int = 42):
    torch.manual_seed(seed)
    sub = construct(
        base=seeded(IMAGE_DIM, N_GLOBAL_CLASSES, interior_cells=H_INIT),
        envelope=Envelope(max_parameter_bytes=PARAM_CAP_BYTES),
        dispatch_table=default_dispatch_table(),
        capacity=2048,
    )
    return sub


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
) -> EvalResult:
    """Evaluate full-softmax and task-aware accuracy on all tasks seen so far."""
    result = EvalResult(after_task=tasks_seen - 1)

    for t_idx in range(tasks_seen):
        spec = specs[t_idx]
        test_view = bundle.task_view(
            spec.dataset_name, spec.local_classes, spec.global_classes,
            split="test", task_name=spec.name,
        )
        x, y = test_view.all_examples()

        with torch.no_grad():
            logits = sub(x)

        # Full-softmax accuracy (argmax over all 30 classes)
        pred_full = logits.argmax(dim=1)
        full_acc = (pred_full == y).float().mean().item()

        # Task-aware accuracy (argmax restricted to this task's classes)
        task_logits = logits[:, spec.global_classes]
        pred_task_local = task_logits.argmax(dim=1)
        # Map local prediction back to global
        gc = torch.tensor(spec.global_classes, dtype=torch.long)
        pred_task_global = gc[pred_task_local]
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

    growth_budget = N_GROW_PER_TASK
    growth_count = 0
    frust_steps = 0

    for epoch in range(epochs):
        for x_batch, y_batch in train_view.iter_epoch(BATCH):
            logits = sub(x_batch)
            loss = torch.nn.functional.cross_entropy(logits, y_batch)

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

            # Astrocyte-gated replay of at-risk past tasks
            code = x_batch[:, :len(code_boundary)]
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
                        # Re-create optimizer to include new parameters
                        opt = torch.optim.Adam(sub.trainable_tensors(), lr=LR)
                        frust_steps = 0

    if recorder:
        recorder.on_task_end(sub.arena, task_idx)

    return growth_count


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chained-15 v2.0 bench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--smoke", action="store_true", help="2 epochs, fast")
    parser.add_argument("--viz", action="store_true", help="generate HTML viewer")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 2

    torch.manual_seed(args.seed)

    print("=" * 60)
    print(f"Chained-15 v2.0 — seed={args.seed}, epochs={args.epochs}")
    print("=" * 60)

    # Load data
    specs = chained_15_specs()
    bundle = DatasetBundle(["mnist", "fashion_mnist", "emnist_letters"])

    # Build substrate
    sub = build_substrate(args.seed)
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

        grown = train_one_task(
            sub, credit, frust, archive, train_view, spec,
            task_idx, args.epochs, recorder,
        )

        # Dream cycle
        archive.finalize_all()
        dream_cfg = DreamConfig(replay_batch_size=BATCH, replay_lr=LR * 0.1)
        dream_result = dream_cycle(
            sub, credit, archive,
            current_classes=spec.global_classes,
            cfg=dream_cfg,
        )
        if recorder:
            recorder.on_dream(sub.arena, task_idx)

        frust.reset()
        sub.end_task()
        sub.compile()

        # Re-open manifold astrocytes for future collection
        for gc in spec.global_classes:
            astro = archive.get(gc)
            if astro and sub.arena.state[astro.cell_id] == CellState.DORMANT:
                sub.arena.state[astro.cell_id] = CellState.ACTIVE

        # Evaluate
        ev = evaluate_all_tasks(sub, bundle, specs, task_idx + 1)
        eval_history.append(ev)

        elapsed = time.time() - t_task
        print(f"  cells={sub.n_cells}, edges={sub.n_edges}, "
              f"locked={dream_result.n_locked}, grown={grown}")
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
