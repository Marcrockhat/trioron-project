"""Single-donor Pong dream loop.

Mirrors `play_with_dreams.py` (the per-skill loop) but operates on a
single multi-task donor (`trioron_P_skill.pt`) instead of three
separate skill donors. This is the canonical "trioron-P" loop for
the cross-game extend/absorb experiment.

Key differences from per-skill:
  - Loading: one donor backs all three skills (CATCH/SMASH/PREPOS),
    with per-skill class masking happening at action selection.
  - Refresh: rebuild the single donor on the AUGMENTED MULTI-TASK
    curriculum — each skill's synthetic data is concatenated with
    THAT skill's corrections + reinforcements, then build_donor
    runs the multi-task pass again. This gives the dream loop a way
    to update all three skills coherently within one substrate.

The play_episode and credit-assignment helpers are reused unchanged
from `play_with_dreams.py` — the difference is purely in donor
loading and refresh.

Usage:
    python3 -m experiments.atari_trioron.primitives.play_with_dreams_pong_single \\
        --episodes 6 --seed 1 --target-return 0
"""
from __future__ import annotations
import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import torch
import gymnasium as gym
import ale_py

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import (  # noqa: E402
    TrioronConfig, AdvancedConfig, build_donor, load_organism,
)
from experiments.atari_trioron.primitives.play_with_dreams import (  # noqa: E402
    play_episode, collect_frustrations, collect_celebrations,
    build_correction_tasks, build_celebration_tasks,
    collect_episode_level, oracle_action_for_state, SKILL_ACTION_TO_CLASS,
    DONOR_ROOT,
)
from experiments.atari_trioron.primitives.train_pong_single_donor import (  # noqa: E402
    PONG_DONOR_PATH, PONG_TASK_ORDER,
)
from experiments.atari_trioron.primitives.train_donors import (  # noqa: E402
    make_task as make_pong_task, L0_SEED,
)
from experiments.atari_trioron.primitives.train_skill_donors import (  # noqa: E402
    SKILL_GROUPS as PONG_SKILL_GROUPS,
)
from experiments.atari_trioron.primitives.adaptive_target import (  # noqa: E402
    AdaptiveTarget, AdaptiveTargetConfig,
)

gym.register_envs(ale_py)


def _skill_to_group(skill: str) -> str:
    """Map play-loop skill names ('CATCH','SMASH','PREPOS') to
    train_donors' group names ('SKILL_CATCH', etc.)."""
    return f"SKILL_{skill}"


def refresh_pong_single_donor(
    corrections_per_skill: Dict[str, List[Tuple[torch.Tensor, int]]],
    out_path: Path,
    *,
    live_multiplier: int = 1,
    n_synthetic_per_class: int = 1500,
    cap_bytes: int = 16_000,
    epochs_per_task: int = 6,
) -> None:
    """Rebuild the single Pong donor on the multi-task augmented
    curriculum. Each task is the synthetic baseline for its skill
    plus that skill's corrections+reinforcements (oversampled by
    live_multiplier)."""
    if not any(corrections_per_skill.values()):
        return
    n_total = sum(len(v) for v in corrections_per_skill.values())
    print(f"  refresh trioron-P-single with {n_total} live examples "
          f"(synth/class={n_synthetic_per_class}, live×={live_multiplier})...")
    augmented_tasks = []
    for tname in PONG_TASK_ORDER:
        cls = PONG_SKILL_GROUPS[tname]
        base_task = make_pong_task(tname, cls,
                                   n_train_per_class=n_synthetic_per_class)
        skill_key = tname.replace("SKILL_", "")
        corr = corrections_per_skill.get(skill_key, [])
        if corr:
            X_corr_one = torch.cat([sv for sv, _ in corr], dim=0)
            y_corr_one = torch.tensor([c for _, c in corr], dtype=torch.long)
            if live_multiplier > 1:
                X_corr = X_corr_one.repeat(live_multiplier, 1)
                y_corr = y_corr_one.repeat(live_multiplier)
            else:
                X_corr, y_corr = X_corr_one, y_corr_one
            Xtr = torch.cat([base_task.X_train, X_corr], dim=0)
            ytr = torch.cat([base_task.y_train, y_corr], dim=0)
            from trioron.api import TaskData
            augmented = TaskData(
                name=tname, X_train=Xtr, y_train=ytr,
                X_test=base_task.X_test, y_test=base_task.y_test,
                classes=base_task.classes,
            )
            print(f"    {tname:15s}: synth={base_task.X_train.shape[0]} + "
                  f"live×{live_multiplier}={X_corr.shape[0]}")
            augmented_tasks.append(augmented)
        else:
            augmented_tasks.append(base_task)

    cfg = TrioronConfig(
        cap_bytes=cap_bytes,
        dream_replay_steps=50,
        advanced=AdvancedConfig(
            h_init=32, n_grow_per_task=4, l0_width=64, freeze_l0=True,
        ),
    )
    build_donor(
        label="trioron_P_skill",
        tasks=augmented_tasks, seed=L0_SEED,
        epochs_per_task=epochs_per_task,
        config=cfg, out_path=out_path,
    )


def run_dream_loop_single(
    *, seed: int, n_episodes: int, max_steps: int,
    frustration_window: int,
    donor_path: Path, out_dir: Path,
    vary_seed: bool = False,
    min_dream_examples: int = 20,
    positive_buffer_size: int = 500,
    target_return: Optional[float] = None,
    live_multiplier: int = 1,
    n_synthetic_per_class: int = 1500,
    adaptive_target: Optional[AdaptiveTarget] = None,
) -> List[float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    returns: List[float] = []
    positive_buffer: Dict[str, Deque[Tuple[torch.Tensor, int]]] = {
        skill: deque(maxlen=positive_buffer_size)
        for skill in ("CATCH", "SMASH", "PREPOS")
    }
    for ep in range(n_episodes):
        single = load_organism(donor_path)
        donors = {s: single for s in ("CATCH", "SMASH", "PREPOS")}
        episode_seed = seed + ep if vary_seed else seed
        t0 = time.time()
        ret, n_steps, frame_log = play_episode(
            seed=episode_seed,
            max_steps=max_steps, donors=donors,
            out_dir=out_dir, record=(ep == 0 or ep == n_episodes - 1),
            name=f"pong_single_dream_ep{ep}",
        )
        elapsed = time.time() - t0
        print(f"\n[ep {ep+1}/{n_episodes} seed={episode_seed}] "
              f"return={ret:+.1f} length={n_steps} ({elapsed:.1f}s)")
        returns.append(ret)
        eff_target = target_return
        if adaptive_target is not None:
            update = adaptive_target.update_after_episode(ret)
            eff_target = update["target"]
            print(f"  adaptive target: ret={ret:+.1f} target={eff_target:+.1f} "
                  f"best={update['running_best']:+.1f} "
                  f"plateau={update['plateau_count']} "
                  f"strain={update['strain_count']} "
                  f"[{update['transition']}]")
        if eff_target is not None:
            flagged, celebrated = collect_episode_level(
                frame_log, ret, eff_target,
            )
            mode = ("FRUSTRATED" if flagged else "SATISFIED")
            print(f"  episode-level: ret={ret:+.1f} vs "
                  f"target={eff_target:+.1f} → {mode} "
                  f"(frustrated={len(flagged)}, celebrated={len(celebrated)})")
        else:
            celebrated = collect_celebrations(frame_log,
                                              window=frustration_window)
            flagged = collect_frustrations(frame_log,
                                           window=frustration_window)
        reinforcements = build_celebration_tasks(frame_log, celebrated)
        for skill in positive_buffer:
            for sv, cls in reinforcements[skill]:
                positive_buffer[skill].append((sv, cls))
        buf = ", ".join(
            f"{s}={len(positive_buffer[s])}" for s in positive_buffer
        )
        print(f"  positive buffer: {buf}")
        satisfied = (eff_target is not None and ret >= eff_target)
        legacy_win = (eff_target is None and ret > 0)
        if satisfied or legacy_win:
            tag = "satisfied" if satisfied else f"net win ({ret:+.1f})"
            print(f"  {tag} — buffer updated, skipping retrain")
            continue
        if not flagged and not celebrated:
            print("  no frustrations or celebrations — skipping dream")
            continue
        corrections = build_correction_tasks(frame_log, flagged)
        # Combine corrections + reinforcements + buffer per skill
        combined: Dict[str, List[Tuple[torch.Tensor, int]]] = {}
        total = 0
        for skill in ("CATCH", "SMASH", "PREPOS"):
            n_corr = len(corrections[skill])
            n_reinf = len(reinforcements[skill])
            if n_corr + n_reinf < min_dream_examples:
                combined[skill] = []
                continue
            combined[skill] = (corrections[skill]
                               + reinforcements[skill]
                               + list(positive_buffer[skill]))
            total += len(combined[skill])
        if total == 0:
            print("  not enough examples in any skill — skipping dream")
            continue
        for skill, items in combined.items():
            if items:
                print(f"  retrain set [{skill}]: {len(items)} examples")
        refresh_pong_single_donor(
            combined, donor_path,
            live_multiplier=live_multiplier,
            n_synthetic_per_class=n_synthetic_per_class,
        )
    return returns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--frustration-window", type=int, default=30)
    ap.add_argument("--donor-path", type=str, default=str(PONG_DONOR_PATH))
    ap.add_argument("--out-dir", type=str,
                    default=str(PROJ / "outputs" / "atari_primitive_donors"
                                / "pong_single_dream_loop"))
    ap.add_argument("--vary-seed", action="store_true")
    ap.add_argument("--min-dream-examples", type=int, default=20)
    ap.add_argument("--positive-buffer-size", type=int, default=500)
    ap.add_argument("--target-return", type=float, default=None)
    ap.add_argument("--live-multiplier", type=int, default=1)
    ap.add_argument("--n-synthetic-per-class", type=int, default=1500)
    ap.add_argument("--adaptive-target", action="store_true")
    ap.add_argument("--at-stretch", type=float, default=2.0)
    ap.add_argument("--at-raise-delta", type=float, default=2.0)
    ap.add_argument("--at-cool-delta", type=float, default=2.0)
    ap.add_argument("--at-plateau-k", type=int, default=2)
    ap.add_argument("--at-strain-m", type=int, default=3)
    args = ap.parse_args()
    at = None
    if args.adaptive_target:
        at = AdaptiveTarget(config=AdaptiveTargetConfig(
            stretch=args.at_stretch, raise_delta=args.at_raise_delta,
            cool_delta=args.at_cool_delta, plateau_K=args.at_plateau_k,
            strain_M=args.at_strain_m,
        ))
    returns = run_dream_loop_single(
        seed=args.seed, n_episodes=args.episodes,
        max_steps=args.max_steps,
        frustration_window=args.frustration_window,
        donor_path=Path(args.donor_path),
        out_dir=Path(args.out_dir),
        vary_seed=args.vary_seed,
        min_dream_examples=args.min_dream_examples,
        positive_buffer_size=args.positive_buffer_size,
        target_return=args.target_return,
        live_multiplier=args.live_multiplier,
        n_synthetic_per_class=args.n_synthetic_per_class,
        adaptive_target=at,
    )
    print("\n=== Dream-loop trajectory ===")
    for i, r in enumerate(returns):
        print(f"  ep {i+1}: {r:+.1f}")
    if len(returns) >= 2:
        print(f"  delta (last - first): {returns[-1] - returns[0]:+.1f}")


if __name__ == "__main__":
    main()
