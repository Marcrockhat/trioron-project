"""Build the 5-organism cross-game matrix.

  trioron-P          = single donor, Pong only          (already exists)
  trioron-B          = single donor, Breakout only      (already exists)
  trioron-PB         = api.extend(trioron-P, base=Pong, new=Breakout)
  trioron-BP         = api.extend(trioron-B, base=Breakout, new=Pong)
  trioron-PB-absorb  = absorb(trioron-P, trioron-B)

All five share L0 seed=42 (the shared-seed invariant — see
absorption_mechanism_design.md). State_dim is 9 across both games
(Breakout pads opp_y to a constant per breakout_state.py).

Pong skill class IDs: 210..232 (8 classes across 3 skills).
Breakout skill class IDs: 300..312 (6 classes across 2 skills).
Disjoint by construction so head-extension is automatic.

Usage:
    python3 -m experiments.atari_trioron.primitives.build_cross_game_organisms
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import extend, absorb  # noqa: E402
from experiments.atari_trioron.primitives.train_pong_single_donor import (  # noqa: E402
    PONG_TASK_ORDER, PONG_DONOR_PATH,
)
from experiments.atari_trioron.primitives.train_breakout_single_donor import (  # noqa: E402
    BREAKOUT_TASK_ORDER, BREAKOUT_DONOR_PATH,
)
from experiments.atari_trioron.primitives.train_donors import (  # noqa: E402
    OUT_ROOT, make_task as make_pong_task,
)
from experiments.atari_trioron.primitives.train_skill_donors import (  # noqa: E402
    SKILL_GROUPS as PONG_SKILL_GROUPS,
)
from experiments.atari_trioron.primitives.train_breakout_skill_donors import (  # noqa: E402
    make_task as make_breakout_task,
    SKILL_GROUPS as BREAKOUT_SKILL_GROUPS,
)


PB_DONOR_PATH        = OUT_ROOT / "trioron_PB_skill.pt"
BP_DONOR_PATH        = OUT_ROOT / "trioron_BP_skill.pt"
PB_ABSORB_PATH       = OUT_ROOT / "trioron_PB_absorb.pt"


def _build_pong_tasks(seed: int = 42):
    return [make_pong_task(t, PONG_SKILL_GROUPS[t], seed=seed)
            for t in PONG_TASK_ORDER]


def _build_breakout_tasks(seed: int = 42):
    return [make_breakout_task(t, BREAKOUT_SKILL_GROUPS[t], seed=seed)
            for t in BREAKOUT_TASK_ORDER]


def build_PB(extension_cap_bytes: int = 16_000) -> Path:
    """trioron-PB = trioron-P extended with Breakout tasks.

    api.extend resumes from trioron-P's substrate, replays Pong base
    classes via boundary dream → archive lock, then trains Breakout
    classes on top. Pong skills survive via archive-lock; Breakout
    skills trained on the same substrate."""
    print("\n=== build trioron-PB (extend Pong → +Breakout) ===")
    out = extend(
        donor_path=PONG_DONOR_PATH,
        base_tasks=_build_pong_tasks(),
        new_tasks=_build_breakout_tasks(),
        out_path=PB_DONOR_PATH,
        extension_cap_bytes=extension_cap_bytes,
        epochs_per_task=6,
        permanent_int8=False,
    )
    print(f"  → {out}")
    return out


def build_BP(extension_cap_bytes: int = 16_000) -> Path:
    """trioron-BP = trioron-B extended with Pong tasks (reverse order)."""
    print("\n=== build trioron-BP (extend Breakout → +Pong) ===")
    out = extend(
        donor_path=BREAKOUT_DONOR_PATH,
        base_tasks=_build_breakout_tasks(),
        new_tasks=_build_pong_tasks(),
        out_path=BP_DONOR_PATH,
        extension_cap_bytes=extension_cap_bytes,
        epochs_per_task=6,
        permanent_int8=False,
    )
    print(f"  → {out}")
    return out


def build_PB_absorb() -> Path:
    """trioron-PB-absorb = absorb(trioron-P, trioron-B). Concurrent
    graft via shared L0; no new training."""
    print("\n=== build trioron-PB-absorb (absorb Pong + Breakout) ===")
    out = absorb(
        donor_paths=[PONG_DONOR_PATH, BREAKOUT_DONOR_PATH],
        out_path=PB_ABSORB_PATH,
    )
    print(f"  → {out}")
    return out


def build_all():
    paths = {
        "trioron-PB":        build_PB(),
        "trioron-BP":        build_BP(),
        "trioron-PB-absorb": build_PB_absorb(),
    }
    print("\n=== artifacts ===")
    for name, p in paths.items():
        size_kb = p.stat().st_size / 1024
        print(f"  {name:20s}  {p}  ({size_kb:.1f} KB)")
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["PB", "BP", "absorb", "all"],
                    default="all")
    ap.add_argument("--extension-cap-bytes", type=int, default=16_000)
    args = ap.parse_args()
    if args.only == "all":
        build_all()
    elif args.only == "PB":
        build_PB(args.extension_cap_bytes)
    elif args.only == "BP":
        build_BP(args.extension_cap_bytes)
    elif args.only == "absorb":
        build_PB_absorb()


if __name__ == "__main__":
    main()
