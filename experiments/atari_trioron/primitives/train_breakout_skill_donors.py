"""Train Breakout's 2 multi-skill action donors.

Mirrors `train_skill_donors.py` (Pong's recipe) but for Breakout's
8-d state, 6-class action space. LAUNCH is single-action — there's
nothing to learn — so it has no donor; the inference loop emits FIRE
out-of-band whenever ball_in_play=False.

  RECEIVE — ball_dy > 0 (approaching paddle row from above)
            classes: BO_RECEIVE_LEFT / RIGHT / HOLD
  SETTLE  — ball_dy ≤ 0 (receding upward toward bricks)
            classes: BO_SETTLE_LEFT / RIGHT / HOLD

Donors share `l0_seed=42`. The L0 random projection is shape-distinct
from Pong's (STATE_DIM=8 vs Pong's 9), so a single absorbed organism
hosting both games' donors is not yet feasible without retraining the
L0 layer — flagged as a future-work item rather than something this
script needs to solve.

Usage:
    python3 -m experiments.atari_trioron.primitives.train_breakout_skill_donors
    python3 -m experiments.atari_trioron.primitives.train_breakout_skill_donors --skill SKILL_RECEIVE
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import (  # noqa: E402
    TaskData, TrioronConfig, AdvancedConfig, build_donor, load_organism,
)
from experiments.atari_trioron.primitives.breakout_state import (  # noqa: E402
    STATE_DIM, generate_dataset, CLASS_NAMES_BREAKOUT,
    SKILL_RECEIVE_LEFT, SKILL_RECEIVE_RIGHT, SKILL_RECEIVE_HOLD,
    SKILL_SETTLE_LEFT, SKILL_SETTLE_RIGHT, SKILL_SETTLE_HOLD,
)


SKILL_GROUPS: Dict[str, List[int]] = {
    "BO_SKILL_RECEIVE": [SKILL_RECEIVE_LEFT, SKILL_RECEIVE_RIGHT,
                         SKILL_RECEIVE_HOLD],
    "BO_SKILL_SETTLE":  [SKILL_SETTLE_LEFT, SKILL_SETTLE_RIGHT,
                         SKILL_SETTLE_HOLD],
}

OUT_ROOT = PROJ / "outputs" / "atari_primitive_donors"
L0_SEED = 42


def make_task(
    group_name: str,
    class_ids: List[int],
    *,
    n_train_per_class: int = 1500,
    n_test_per_class: int = 400,
    seed: int = 0,
) -> TaskData:
    Xtr_list, ytr_list = [], []
    Xte_list, yte_list = [], []
    for i, cid in enumerate(class_ids):
        Xtr = generate_dataset(cid, n_train_per_class, seed=seed + 100 * i)
        Xte = generate_dataset(cid, n_test_per_class,  seed=seed + 100 * i + 1)
        ytr = torch.full((n_train_per_class,), cid, dtype=torch.long)
        yte = torch.full((n_test_per_class,),  cid, dtype=torch.long)
        Xtr_list.append(Xtr); ytr_list.append(ytr)
        Xte_list.append(Xte); yte_list.append(yte)
    Xtr = torch.cat(Xtr_list, dim=0)
    ytr = torch.cat(ytr_list, dim=0)
    Xte = torch.cat(Xte_list, dim=0)
    yte = torch.cat(yte_list, dim=0)
    perm = torch.randperm(Xtr.shape[0],
                          generator=torch.Generator().manual_seed(seed + 9999))
    Xtr = Xtr[perm]
    ytr = ytr[perm]
    return TaskData(
        name=group_name,
        X_train=Xtr, y_train=ytr,
        X_test=Xte,  y_test=yte,
        classes=list(class_ids),
    )


def train_donor(
    group_name: str,
    class_ids: List[int],
    *,
    cap_bytes: int = 4_000,
    l0_width: int = 64,
    h_init: int = 32,
    epochs_per_task: int = 6,
    seed: int = L0_SEED,
    out_dir: Path = OUT_ROOT,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group_name}.pt"

    print(f"\n=== train_donor({group_name}) ===")
    print(f"  classes: {class_ids} -> "
          f"{[CLASS_NAMES_BREAKOUT[c] for c in class_ids]}")
    task = make_task(group_name, class_ids, seed=seed)
    print(f"  data: train {tuple(task.X_train.shape)}  "
          f"test {tuple(task.X_test.shape)}")

    cfg = TrioronConfig(
        cap_bytes=cap_bytes,
        dream_replay_steps=50,
        advanced=AdvancedConfig(
            h_init=h_init,
            n_grow_per_task=4,
            l0_width=l0_width,
            freeze_l0=True,
        ),
    )
    t0 = time.time()
    donor_path = build_donor(
        label=group_name,
        tasks=[task],
        seed=seed,
        epochs_per_task=epochs_per_task,
        config=cfg,
        out_path=out_path,
    )
    elapsed = time.time() - t0
    print(f"  built in {elapsed:.1f}s -> {donor_path}")
    return donor_path


@torch.no_grad()
def evaluate_donor(
    group_name: str,
    class_ids: List[int],
    *,
    n_per_class: int = 500,
    seed: int = 7777,
) -> Dict[str, float]:
    donor_path = OUT_ROOT / f"{group_name}.pt"
    organism = load_organism(donor_path)

    correct_per_class: Dict[int, Tuple[int, int]] = {
        c: (0, 0) for c in class_ids
    }
    for i, cid in enumerate(class_ids):
        X = generate_dataset(cid, n_per_class, seed=seed + 100 * i)
        logits = organism(X, routing="soft")
        if isinstance(logits, tuple):
            logits = logits[0]
        union = list(organism.union_classes)
        eligible = set(class_ids)
        masked = torch.full_like(logits, float("-inf"))
        for j, c in enumerate(union):
            if int(c) in eligible:
                masked[:, j] = logits[:, j]
        pred_idx = masked.argmax(dim=-1)
        pred_class = torch.tensor(
            [int(union[int(k)]) for k in pred_idx]
        )
        correct = int((pred_class == cid).sum())
        correct_per_class[cid] = (correct, n_per_class)

    total_correct = sum(c for c, _ in correct_per_class.values())
    total = sum(t for _, t in correct_per_class.values())
    overall = total_correct / total

    print(f"  eval {group_name}: task-aware {overall:.3f}")
    for cid in class_ids:
        c, t = correct_per_class[cid]
        print(f"    {CLASS_NAMES_BREAKOUT[cid]:>22s}: "
              f"{c}/{t} = {c/t:.3f}")
    return {
        "overall": overall,
        "per_class": {
            CLASS_NAMES_BREAKOUT[cid]:
                correct_per_class[cid][0] / correct_per_class[cid][1]
            for cid in class_ids
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skill", choices=list(SKILL_GROUPS.keys()) + ["all"],
                    default="all")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--cap-bytes", type=int, default=4_000)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    skills = (list(SKILL_GROUPS.keys())
              if args.skill == "all" else [args.skill])

    summary: Dict[str, float] = {}
    for s in skills:
        train_donor(s, SKILL_GROUPS[s],
                    cap_bytes=args.cap_bytes,
                    epochs_per_task=args.epochs)
        if not args.no_eval:
            r = evaluate_donor(s, SKILL_GROUPS[s])
            summary[s] = r["overall"]

    if summary:
        print("\n=== Breakout skill donor task-aware accuracy ===")
        for s, acc in summary.items():
            verdict = ("OK" if acc >= 0.95
                       else ("YELLOW" if acc >= 0.85 else "RED"))
            print(f"  {s:18s}  {acc:.3f}  [{verdict}]")


if __name__ == "__main__":
    main()
