"""Train a single donor on the full Mode-E curriculum:
8 primitive groups + 1 Pong-action task.

This is the proper Mode-E shape (Shape A in docs/learning_methods.md
§3): one build_donor call with the full task sequence. Continual
learning preserves earlier primitives as the Pong-action task is
added at the end.

Output: outputs/atari_primitive_donors/pong_curriculum_donor.pt

Usage:
    python3 -m experiments.atari_trioron.primitives.train_curriculum_donor
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import (   # noqa: E402
    TaskData, TrioronConfig, AdvancedConfig, build_donor, load_organism,
)
from experiments.atari_trioron.primitives.synthetic_env import (   # noqa: E402
    generate_dataset, CLASS_NAMES,
    PONG_ACTION_UP, PONG_ACTION_DOWN, PONG_ACTION_HOLD,
)
from experiments.atari_trioron.primitives.train_donors import (   # noqa: E402
    DONOR_GROUPS, OUT_ROOT, make_task,
)


PONG_TASK_CLASSES: List[int] = [PONG_ACTION_UP, PONG_ACTION_DOWN, PONG_ACTION_HOLD]
PONG_TASK_NAME = "PONG_ACTION"

CURRICULUM_PATH = OUT_ROOT / "pong_curriculum_donor.pt"


def build_full_curriculum(
    *,
    n_train_per_class: int = 1500,
    n_test_per_class: int = 400,
    seed: int = 0,
) -> List[TaskData]:
    """8 primitive tasks + 1 Pong-action task, in curriculum order."""
    tasks: List[TaskData] = []
    for i, (group_name, class_ids) in enumerate(DONOR_GROUPS.items()):
        tasks.append(make_task(group_name, class_ids,
                               n_train_per_class=n_train_per_class,
                               n_test_per_class=n_test_per_class,
                               seed=seed + 1000 * i))
    tasks.append(make_task(PONG_TASK_NAME, PONG_TASK_CLASSES,
                           n_train_per_class=n_train_per_class,
                           n_test_per_class=n_test_per_class,
                           seed=seed + 1000 * len(DONOR_GROUPS)))
    return tasks


def train(
    *,
    cap_bytes: int = 64_000,
    epochs_per_task: int = 8,
    seed: int = 42,
) -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tasks = build_full_curriculum(seed=0)
    print(f"Curriculum: {len(tasks)} tasks")
    for t in tasks:
        names = [CLASS_NAMES[c] for c in t.classes]
        print(f"  {t.name:25s}  classes={t.classes}  ({names})")

    cfg = TrioronConfig(
        cap_bytes=cap_bytes,
        dream_replay_steps=80,
        advanced=AdvancedConfig(
            h_init=32,
            n_grow_per_task=4,
            l0_width=64,
            freeze_l0=True,
        ),
    )
    print(f"\nbuild_donor — cap_bytes={cap_bytes}, "
          f"epochs/task={epochs_per_task}, seed={seed}")
    t0 = time.time()
    out = build_donor(
        label="pong_curriculum",
        tasks=tasks,
        seed=seed,
        epochs_per_task=epochs_per_task,
        config=cfg,
        out_path=CURRICULUM_PATH,
    )
    print(f"built in {(time.time()-t0)/60:.1f} min -> {out}")
    return out


@torch.no_grad()
def evaluate(
    organism_path: Path,
    *,
    n_per_class: int = 500,
    seed: int = 9999,
) -> Dict[str, float]:
    """Per-task held-out eval through the trained donor."""
    organism = load_organism(organism_path)
    summary: Dict[str, float] = {}
    all_tasks: List[tuple] = list(DONOR_GROUPS.items()) + [
        (PONG_TASK_NAME, PONG_TASK_CLASSES),
    ]
    for group_name, class_ids in all_tasks:
        eligible = set(class_ids)
        total_correct = 0
        total = 0
        per_class: Dict[int, float] = {}
        for i, cid in enumerate(class_ids):
            X = generate_dataset(cid, n_per_class, seed=seed + 100 * i)
            logits = organism(X, routing="soft")
            if isinstance(logits, tuple):
                logits = logits[0]
            union = list(organism.union_classes)
            masked = torch.full_like(logits, float("-inf"))
            for j, c in enumerate(union):
                if int(c) in eligible:
                    masked[:, j] = logits[:, j]
            pred_idx = masked.argmax(dim=-1)
            pred_class = torch.tensor([int(union[int(k)]) for k in pred_idx])
            correct = int((pred_class == cid).sum())
            total_correct += correct
            total += n_per_class
            per_class[cid] = correct / n_per_class
        overall = total_correct / total
        summary[group_name] = overall
        print(f"  {group_name:25s}  task-aware {overall:.3f}")
        for cid in class_ids:
            print(f"      {CLASS_NAMES[cid]:>26s}: {per_class[cid]:.3f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap-bytes", type=int, default=64_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    out = train(cap_bytes=args.cap_bytes,
                epochs_per_task=args.epochs,
                seed=args.seed)
    if not args.no_eval:
        print(f"\n=== held-out eval on {out.name} ===")
        summary = evaluate(out)
        print(f"\n=== summary ===")
        for g, acc in summary.items():
            verdict = ("OK" if acc >= 0.95
                       else ("YELLOW" if acc >= 0.85 else "RED"))
            print(f"  {g:25s}  {acc:.3f}  [{verdict}]")


if __name__ == "__main__":
    main()
