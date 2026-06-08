"""Train the 7 primitive donors for the Pong vocabulary.

Each donor covers one concept group (e.g. ball-vertical-position).
All donors share `l0_seed=42` so absorb (task #9) composes them
without retraining. Trained donors land at
`outputs/atari_primitive_donors/{group}.pt`.

Usage:
    python3 -m experiments.atari_trioron.primitives.train_donors
    python3 -m experiments.atari_trioron.primitives.train_donors --group BALL_SPEED

The script also runs a held-out evaluation per donor and prints task-
aware accuracy. Expected post-training: ≥ 0.95 on every donor (the
clustering probe at l0=64 already saturates at probe_acc 0.96-1.00).
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

from trioron.api import (   # noqa: E402
    TaskData, TrioronConfig, AdvancedConfig, build_donor, load_organism,
)
from experiments.atari_trioron.primitives.synthetic_env import (   # noqa: E402
    STATE_DIM, generate_dataset, CLASS_NAMES,
    BALL_HIGH, BALL_MID, BALL_LOW,
    BALL_LEFT, BALL_CENTER, BALL_RIGHT,
    PADDLE_HIGH, PADDLE_MID, PADDLE_LOW,
    BALL_GOING_UP, BALL_GOING_DOWN, BALL_GOING_LEFT, BALL_GOING_RIGHT,
    BALL_FAST, BALL_SLOW,
    BALL_ABOVE_PADDLE, BALL_ALIGNED_WITH_PADDLE, BALL_BELOW_PADDLE,
    BALL_APPROACHING_PADDLE, BALL_RECEDING_FROM_PADDLE,
)


# ---------------------------------------------------------------------
# Donor groups (one Mode-A donor per group)
# ---------------------------------------------------------------------

DONOR_GROUPS: Dict[str, List[int]] = {
    "BALL_VERTICAL":         [BALL_HIGH, BALL_MID, BALL_LOW],
    "BALL_HORIZONTAL":       [BALL_LEFT, BALL_CENTER, BALL_RIGHT],
    "PADDLE_VERTICAL":       [PADDLE_HIGH, PADDLE_MID, PADDLE_LOW],
    # Motion-direction split into two 2-class donors.
    # The original 4-class formulation was non-exclusive (dx>0 ∧ dy>0
    # is both RIGHT and DOWN); a 4-way CE can't choose between them.
    # Split per axis-of-motion makes each donor mutually exclusive:
    # sign(dy) is one or the other, never both.
    "BALL_MOTION_VERTICAL":   [BALL_GOING_UP, BALL_GOING_DOWN],
    "BALL_MOTION_HORIZONTAL": [BALL_GOING_LEFT, BALL_GOING_RIGHT],
    "BALL_SPEED":            [BALL_FAST, BALL_SLOW],
    "BALL_PADDLE_VERTICAL":  [BALL_ABOVE_PADDLE, BALL_ALIGNED_WITH_PADDLE,
                              BALL_BELOW_PADDLE],
    "BALL_PADDLE_APPROACH":  [BALL_APPROACHING_PADDLE,
                              BALL_RECEDING_FROM_PADDLE],
}

OUT_ROOT = PROJ / "outputs" / "atari_primitive_donors"

# Shared L0 seed across all donors — invariant for absorb (#9).
L0_SEED = 42


# ---------------------------------------------------------------------
# Build TaskData for a donor group
# ---------------------------------------------------------------------

def make_task(
    group_name: str,
    class_ids: List[int],
    *,
    n_train_per_class: int = 1500,
    n_test_per_class: int = 400,
    seed: int = 0,
) -> TaskData:
    """Generate synthetic standardized state-vectors for every class
    in the group, stack into a single TaskData."""
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
    # Shuffle train set (test set order doesn't matter for eval).
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


# ---------------------------------------------------------------------
# Train one donor
# ---------------------------------------------------------------------

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
    print(f"  classes: {class_ids} -> {[CLASS_NAMES[c] for c in class_ids]}")
    task = make_task(group_name, class_ids, seed=seed)
    print(f"  data: train {tuple(task.X_train.shape)}  test {tuple(task.X_test.shape)}")

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


# ---------------------------------------------------------------------
# Evaluate one donor on held-out data
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate_donor(
    group_name: str,
    class_ids: List[int],
    *,
    n_per_class: int = 500,
    seed: int = 7777,
) -> Dict[str, float]:
    """Run held-out eval on the trained donor: per-class accuracy +
    overall task-aware accuracy."""
    donor_path = OUT_ROOT / f"{group_name}.pt"
    organism = load_organism(donor_path)

    correct_per_class: Dict[int, Tuple[int, int]] = {c: (0, 0) for c in class_ids}
    for i, cid in enumerate(class_ids):
        X = generate_dataset(cid, n_per_class, seed=seed + 100 * i)
        logits = organism(X, routing="soft")
        if isinstance(logits, tuple):
            logits = logits[0]
        # union_classes order; mask to this group's classes only.
        union = list(organism.union_classes)
        eligible = set(class_ids)
        masked = torch.full_like(logits, float("-inf"))
        for j, c in enumerate(union):
            if int(c) in eligible:
                masked[:, j] = logits[:, j]
        pred_idx = masked.argmax(dim=-1)
        pred_class = torch.tensor([int(union[int(k)]) for k in pred_idx])
        correct = int((pred_class == cid).sum())
        correct_per_class[cid] = (correct, n_per_class)

    total_correct = sum(c for c, _ in correct_per_class.values())
    total = sum(t for _, t in correct_per_class.values())
    overall = total_correct / total

    print(f"  eval {group_name}: task-aware {overall:.3f}")
    for cid in class_ids:
        c, t = correct_per_class[cid]
        print(f"    {CLASS_NAMES[cid]:>26s}: {c}/{t} = {c/t:.3f}")
    return {
        "overall": overall,
        "per_class": {CLASS_NAMES[cid]: correct_per_class[cid][0] / correct_per_class[cid][1]
                      for cid in class_ids},
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=list(DONOR_GROUPS.keys()) + ["all"],
                    default="all")
    ap.add_argument("--no-eval", action="store_true",
                    help="Skip held-out eval after training")
    ap.add_argument("--cap-bytes", type=int, default=4_000)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    groups: List[str]
    if args.group == "all":
        groups = list(DONOR_GROUPS.keys())
    else:
        groups = [args.group]

    summary: Dict[str, float] = {}
    for g in groups:
        train_donor(g, DONOR_GROUPS[g],
                    cap_bytes=args.cap_bytes,
                    epochs_per_task=args.epochs)
        if not args.no_eval:
            r = evaluate_donor(g, DONOR_GROUPS[g])
            summary[g] = r["overall"]

    if summary:
        print("\n=== Donor task-aware accuracy summary ===")
        for g, acc in summary.items():
            verdict = "OK" if acc >= 0.95 else ("YELLOW" if acc >= 0.85 else "RED")
            print(f"  {g:25s}  {acc:.3f}  [{verdict}]")


if __name__ == "__main__":
    main()
