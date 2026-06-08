"""Train a single multi-task Pong donor (all 8 skill classes).

Replaces the 3 separate per-skill donors (SKILL_CATCH/SMASH/PREPOS.pt)
with one donor that knows all skill classes. This is the canonical
"trioron-P" artifact for the cross-game extend/absorb experiment —
api.extend requires a single donor checkpoint and will not accept an
absorbed multi-branch organism.

Tasks (curriculum order):
    1. SKILL_CATCH  — SKILL_CATCH_UP / DOWN / HOLD
    2. SKILL_SMASH  — SKILL_SMASH_UP / DOWN
    3. SKILL_PREPOS — SKILL_PREPOS_UP / DOWN / HOLD

Output: outputs/atari_primitive_donors/trioron_P_skill.pt

Usage:
    python3 -m experiments.atari_trioron.primitives.train_pong_single_donor
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import torch

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import (  # noqa: E402
    TrioronConfig, AdvancedConfig, build_donor, load_organism,
)
from experiments.atari_trioron.primitives.train_donors import (  # noqa: E402
    make_task as make_pong_task, OUT_ROOT, L0_SEED,
)
from experiments.atari_trioron.primitives.train_skill_donors import (  # noqa: E402
    SKILL_GROUPS as PONG_SKILL_GROUPS,
)
from experiments.atari_trioron.primitives.synthetic_env import (  # noqa: E402
    generate_dataset, CLASS_NAMES,
)


PONG_DONOR_PATH = OUT_ROOT / "trioron_P_skill.pt"
PONG_TASK_ORDER = ["SKILL_CATCH", "SKILL_SMASH", "SKILL_PREPOS"]


def build_pong_single_donor(
    *,
    cap_bytes: int = 8_000,
    h_init: int = 32,
    l0_width: int = 64,
    epochs_per_task: int = 6,
    seed: int = L0_SEED,
    out_path: Path = PONG_DONOR_PATH,
) -> Path:
    """Train one donor on all three Pong skill tasks in sequence."""
    print(f"\n=== train_pong_single_donor → {out_path.name} ===")
    tasks = []
    for tname in PONG_TASK_ORDER:
        cls = PONG_SKILL_GROUPS[tname]
        task = make_pong_task(tname, cls, seed=seed)
        print(f"  task {tname}: classes={cls} → "
              f"{[CLASS_NAMES[c] for c in cls]} "
              f"train={tuple(task.X_train.shape)} test={tuple(task.X_test.shape)}")
        tasks.append(task)

    cfg = TrioronConfig(
        cap_bytes=cap_bytes,
        dream_replay_steps=50,
        advanced=AdvancedConfig(
            h_init=h_init, n_grow_per_task=4, l0_width=l0_width,
            freeze_l0=True,
        ),
    )
    t0 = time.time()
    donor_path = build_donor(
        label="trioron_P_skill",
        tasks=tasks, seed=seed,
        epochs_per_task=epochs_per_task,
        config=cfg, out_path=out_path,
    )
    elapsed = time.time() - t0
    print(f"  built in {elapsed:.1f}s -> {donor_path}")
    return donor_path


@torch.no_grad()
def evaluate_pong_single_donor(
    donor_path: Path = PONG_DONOR_PATH,
    n_per_class: int = 500, seed: int = 7777,
) -> dict:
    """Per-class held-out eval through the single donor."""
    organism = load_organism(donor_path)
    union = list(organism.union_classes)

    overall_correct = 0
    overall_total = 0
    per_skill: dict = {}
    for tname in PONG_TASK_ORDER:
        cls = PONG_SKILL_GROUPS[tname]
        eligible = set(cls)
        skill_correct = 0
        skill_total = 0
        for i, cid in enumerate(cls):
            X = generate_dataset(cid, n_per_class, seed=seed + 100 * i)
            logits = organism(X, routing="soft")
            if isinstance(logits, tuple):
                logits = logits[0]
            masked = torch.full_like(logits, float("-inf"))
            for j, c in enumerate(union):
                if int(c) in eligible:
                    masked[:, j] = logits[:, j]
            pred_idx = masked.argmax(dim=-1)
            pred_class = torch.tensor([int(union[int(k)]) for k in pred_idx])
            correct = int((pred_class == cid).sum())
            skill_correct += correct
            skill_total += n_per_class
        skill_acc = skill_correct / skill_total
        per_skill[tname] = skill_acc
        overall_correct += skill_correct
        overall_total += skill_total
        verdict = ("OK" if skill_acc >= 0.95
                   else ("YELLOW" if skill_acc >= 0.85 else "RED"))
        print(f"  {tname:15s}  {skill_acc:.3f}  [{verdict}]")
    overall = overall_correct / overall_total
    print(f"\n  OVERALL: {overall:.3f}")
    return {"overall": overall, "per_skill": per_skill}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap-bytes", type=int, default=8_000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    build_pong_single_donor(cap_bytes=args.cap_bytes,
                            epochs_per_task=args.epochs)
    if not args.no_eval:
        evaluate_pong_single_donor()


if __name__ == "__main__":
    main()
