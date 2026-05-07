"""Train the 3 multi-skill action donors for Pong.

Each skill is a separate primitive donor over a disjoint region of
state space. A hand-coded gate at inference selects the active skill
per frame; the trioron donor learns the action policy within its
region.

  CATCH  — bdx > 0 and bx84 < SMASH_TRIGGER_X
           classes: SKILL_CATCH_UP / DOWN / HOLD
  SMASH  — bdx > 0 and bx84 ≥ SMASH_TRIGGER_X
           classes: SKILL_SMASH_UP / DOWN
  PREPOS — bdx ≤ 0
           classes: SKILL_PREPOS_UP / DOWN / HOLD

All donors share `l0_seed=42` so they can be absorbed into one
organism without retraining (per absorption_mechanism_design.md).

This script is separate from `train_donors.py` because the multi-skill
work bumped STATE_DIM 8 → 9 (added opp_y for SMASH); the existing
8-d primitive donors at outputs/atari_primitive_donors/*.pt are stale
under STATE_DIM=9 and would be silently retrained at the wrong width
if added to DONOR_GROUPS in train_donors.py.

Usage:
    python3 -m experiments.atari_trioron.primitives.train_skill_donors
    python3 -m experiments.atari_trioron.primitives.train_skill_donors --skill CATCH
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from experiments.atari_trioron.primitives.train_donors import (  # noqa: E402
    train_donor, evaluate_donor, OUT_ROOT,
)
from experiments.atari_trioron.primitives.synthetic_env import (  # noqa: E402
    SKILL_CATCH_UP, SKILL_CATCH_DOWN, SKILL_CATCH_HOLD,
    SKILL_SMASH_UP, SKILL_SMASH_DOWN,
    SKILL_PREPOS_UP, SKILL_PREPOS_DOWN, SKILL_PREPOS_HOLD,
)


SKILL_GROUPS: Dict[str, List[int]] = {
    "SKILL_CATCH":  [SKILL_CATCH_UP, SKILL_CATCH_DOWN, SKILL_CATCH_HOLD],
    "SKILL_SMASH":  [SKILL_SMASH_UP, SKILL_SMASH_DOWN],
    "SKILL_PREPOS": [SKILL_PREPOS_UP, SKILL_PREPOS_DOWN, SKILL_PREPOS_HOLD],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skill", choices=list(SKILL_GROUPS.keys()) + ["all"],
                    default="all")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--cap-bytes", type=int, default=4_000)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    if args.skill == "all":
        skills = list(SKILL_GROUPS.keys())
    else:
        skills = [args.skill]

    summary: Dict[str, float] = {}
    for s in skills:
        train_donor(s, SKILL_GROUPS[s],
                    cap_bytes=args.cap_bytes,
                    epochs_per_task=args.epochs)
        if not args.no_eval:
            r = evaluate_donor(s, SKILL_GROUPS[s])
            summary[s] = r["overall"]

    if summary:
        print("\n=== Skill donor task-aware accuracy ===")
        for s, acc in summary.items():
            verdict = "OK" if acc >= 0.95 else ("YELLOW" if acc >= 0.85 else "RED")
            print(f"  {s:15s}  {acc:.3f}  [{verdict}]")


if __name__ == "__main__":
    main()
