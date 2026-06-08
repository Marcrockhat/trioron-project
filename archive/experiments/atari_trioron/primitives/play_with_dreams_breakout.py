"""Frustration-driven dream loop for the Breakout skill organism.

Direct port of `play_with_dreams.py` (Pong) to Breakout's geometry.
Same architectural arc — frustration on rally-loss, celebration on
rally-win, rally-based credit assignment, cross-episode positive
buffer to anchor against drift.

What's different from Pong:
  - The "rally end" is not just a reward sign-flip. Breakout has only
    positive rewards (per brick), so frustration must trigger off
    LIFE-LOSS (RAM byte 57 dropping). Celebration triggers off
    reward > 0 as in Pong.
  - Two skills (RECEIVE, SETTLE) instead of three. LAUNCH is single-
    action and out-of-band, no donor refresh path.
  - `min_dream_examples` defaults lower (5 vs 20) — frustration is
    sparser in Breakout (5 lives total per episode), so we need to
    let smaller correction batches trigger a refresh.

Per-episode loop:
  1. Play one Breakout episode; record (raw_state, std_state, skill,
     action, reward, lives) for every in-play frame.
  2. Detect rally events: reward > 0 (brick-break) OR lives drop
     (life-loss). A rally is the run from the previous event (or
     episode start) to the current event.
  3. For each life-loss rally: at every frame, compute oracle action;
     if agent disagreed, flag as frustration correction.
  4. For each brick-break rally: at every frame, the agent's own
     action is reinforced as celebration (positive replay).
  5. Group corrections + reinforcements by which skill was active.
  6. Per affected skill: rebuild donor on synthetic + corrections +
     reinforcements + cross-episode positive buffer (full retrain).
  7. Continue to the next episode with the updated donors.

Usage:
    python3 -m experiments.atari_trioron.primitives.play_with_dreams_breakout \\
        --episodes 5 --seed 0
"""
from __future__ import annotations
import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
import ale_py
from gymnasium.wrappers import (
    AtariPreprocessing, FrameStackObservation, RecordVideo,
)

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import load_organism  # noqa: E402

from experiments.atari_trioron.features import extract_breakout, get_ale  # noqa: E402
from experiments.atari_trioron.eval_render import _FlatRGBObsWrapper, _resolve_env_id  # noqa: E402
from experiments.atari_trioron.env import FRAME_HW, FRAME_STACK  # noqa: E402
from experiments.atari_trioron.primitives.breakout_state import (  # noqa: E402
    BreakoutStateBuilder, select_skill_breakout,
    PADDLE_LOOKAHEAD, ACTION_DEAD_ZONE, SETTLE_ANCHOR_X,
    _PRED_DY_EPS,
    SKILL_RECEIVE_LEFT, SKILL_RECEIVE_RIGHT, SKILL_RECEIVE_HOLD,
    SKILL_SETTLE_LEFT, SKILL_SETTLE_RIGHT, SKILL_SETTLE_HOLD,
)
from experiments.atari_trioron.primitives.breakout_skill_inference import (  # noqa: E402
    select_action_for_skill, SKILL_CLASS_TO_ALE, SKILL_CLASS_SETS,
    DONOR_ROOT, ALE_NOOP, ALE_FIRE, ALE_RIGHT, ALE_LEFT,
    PADDLE_LEFT_84, PADDLE_RIGHT_84,
)
from experiments.atari_trioron.primitives.breakout_oracle import (  # noqa: E402
    _track_target_x,
)
from experiments.atari_trioron.primitives.train_breakout_skill_donors import (  # noqa: E402
    SKILL_GROUPS,
)
from experiments.atari_trioron.primitives.adaptive_target import (  # noqa: E402
    AdaptiveTarget, AdaptiveTargetConfig,
)

gym.register_envs(ale_py)


# ---------------------------------------------------------------------
# Oracle — same logic as breakout_oracle, applied per-frame to a
# logged raw state vector.
# ---------------------------------------------------------------------


def oracle_action_for_state(raw: np.ndarray) -> int:
    """Compute the oracle's action for a raw 8-d Breakout state."""
    ball_dy = float(raw[3])
    paddle_x84 = float(raw[4])
    paddle_dx = float(raw[5])
    pred_x = float(raw[7])
    if ball_dy > _PRED_DY_EPS:
        target = pred_x
    else:
        target = SETTLE_ANCHOR_X
    return _track_target_x(target, paddle_x84, paddle_dx)


# Per-skill ALE-action → class-id map. Used to convert agent/oracle
# actions back into trainable supervision targets.
SKILL_ACTION_TO_CLASS: Dict[str, Dict[int, int]] = {
    "RECEIVE": {ALE_LEFT:  SKILL_RECEIVE_LEFT,
                ALE_RIGHT: SKILL_RECEIVE_RIGHT,
                ALE_NOOP:  SKILL_RECEIVE_HOLD},
    "SETTLE":  {ALE_LEFT:  SKILL_SETTLE_LEFT,
                ALE_RIGHT: SKILL_SETTLE_RIGHT,
                ALE_NOOP:  SKILL_SETTLE_HOLD},
}


# ---------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------


def make_breakout_env(seed: int, out_dir: Optional[Path], record: bool,
                      name: str):
    env_id = _resolve_env_id("Breakout")
    env = gym.make(env_id, render_mode="rgb_array",
                   frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False)
    if record and out_dir is not None:
        env = RecordVideo(env, video_folder=str(out_dir),
                          episode_trigger=lambda i: i == 0,
                          name_prefix=name, disable_logger=True)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4,
                             screen_size=FRAME_HW,
                             terminal_on_life_loss=False,
                             grayscale_obs=True, grayscale_newaxis=False,
                             scale_obs=False)
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)
    return env


def play_episode(*, seed: int, max_steps: int, donors: dict,
                 out_dir: Optional[Path] = None, record: bool = False,
                 name: str = "breakout_dream_eval"
                 ) -> Tuple[float, int, List[dict]]:
    env = make_breakout_env(seed, out_dir, record, name)
    ale = get_ale(env)
    state_builder = BreakoutStateBuilder()

    obs, _info = env.reset(seed=seed)
    state_builder.reset()
    ret = 0.0
    n_steps = 0
    frame_log: List[dict] = []
    for _ in range(max_steps):
        sv_raw = state_builder.step(ale)
        s = extract_breakout(ale)
        if sv_raw[0] is None:
            if not s.ball_in_play:
                action = ALE_FIRE
            else:
                action = ALE_NOOP
            sv = raw = skill = None
        else:
            sv, raw = sv_raw
            paddle_x84 = float(raw[4])
            skill = select_skill_breakout(raw)
            action = select_action_for_skill(donors[skill], sv, skill,
                                             paddle_x84=paddle_x84)
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r)
        n_steps += 1
        frame_log.append({
            "raw":    None if raw is None else raw.copy(),
            "std":    None if sv is None else sv.clone(),
            "skill":  skill,
            "action": action,
            "reward": float(r),
            "lives":  int(s.lives),
        })
        if term or trunc:
            break
    env.close()
    return ret, n_steps, frame_log


# ---------------------------------------------------------------------
# Credit assignment — two strategies
#
# 1. **Rally-based** (transferred from Pong): a rally runs from the
#    previous reward/life event up to the current one. Frustration
#    rallies end in life-loss, celebration rallies end in brick-break.
# 2. **Episode-level** (Breakout-specific reformulation, 2026-05-08):
#    treat the whole episode as either frustrated (return below
#    target) or satisfied (at-or-above target). The trigger isn't
#    "did I lose this life" — that fires 5×/ep regardless of skill —
#    it's "did I achieve a high score". A child playing Breakout
#    isn't upset about losing one life; they're upset when they keep
#    getting stuck at the same low score. The episode-level signal
#    captures that: every in-play frame becomes either a correction
#    (if disagreed with oracle and ep was frustrated) or a
#    reinforcement (if ep was satisfied).
#
# The episode-level mode is enabled by passing a `target_return`;
# rally-based mode is the default for backwards compat with the Pong
# port shape.
# ---------------------------------------------------------------------


def collect_frustrations(frame_log: List[dict]) -> List[int]:
    """Return indices of frames in life-loss rallies. The rally start
    is the previous reward/life event (or 0); the rally end is the
    frame where lives decreased from the previous frame's lives."""
    flagged: List[int] = []
    rally_start = 0
    prev_lives = frame_log[0]["lives"] if frame_log else 0
    for i, f in enumerate(frame_log):
        lost_life = (f["lives"] < prev_lives)
        scored = (f["reward"] > 0)
        if lost_life:
            for j in range(rally_start, i):
                if frame_log[j]["raw"] is not None:
                    flagged.append(j)
            rally_start = i + 1
        elif scored:
            rally_start = i + 1
        prev_lives = f["lives"]
    return sorted(set(flagged))


def collect_celebrations(frame_log: List[dict]) -> List[int]:
    """Return indices of frames in brick-break rallies. Same shape as
    collect_frustrations but inverted: rallies that end with reward>0
    are the wins."""
    flagged: List[int] = []
    rally_start = 0
    prev_lives = frame_log[0]["lives"] if frame_log else 0
    for i, f in enumerate(frame_log):
        lost_life = (f["lives"] < prev_lives)
        scored = (f["reward"] > 0)
        if scored:
            for j in range(rally_start, i):
                if frame_log[j]["raw"] is not None:
                    flagged.append(j)
            rally_start = i + 1
        elif lost_life:
            rally_start = i + 1
        prev_lives = f["lives"]
    return sorted(set(flagged))


def collect_episode_level(
    frame_log: List[dict], episode_return: float, target_return: float,
) -> Tuple[List[int], List[int]]:
    """Episode-level credit assignment.

    Returns (frustrated_indices, celebrated_indices). If
    episode_return < target_return, ALL in-play frames are flagged
    as frustrated. Otherwise, ALL in-play frames are flagged as
    celebrated. The two lists are disjoint by construction — every
    episode is either frustrated or satisfied as a whole, not both.
    """
    in_play = [i for i, f in enumerate(frame_log) if f["raw"] is not None]
    if episode_return < target_return:
        return in_play, []
    return [], in_play


def build_correction_tasks(frame_log: List[dict], flagged: List[int]
                           ) -> Dict[str, List[Tuple[torch.Tensor, int]]]:
    """At each frustration frame, label with oracle's action class."""
    out: Dict[str, List[Tuple[torch.Tensor, int]]] = {
        "RECEIVE": [], "SETTLE": [],
    }
    n_disagree = 0
    for j in flagged:
        f = frame_log[j]
        skill = f["skill"]
        if skill is None or skill not in out:
            continue
        oracle_act = oracle_action_for_state(f["raw"])
        agent_act = f["action"]
        if oracle_act == agent_act:
            continue
        if oracle_act not in SKILL_ACTION_TO_CLASS[skill]:
            continue
        cls = SKILL_ACTION_TO_CLASS[skill][oracle_act]
        out[skill].append((f["std"], cls))
        n_disagree += 1
    print(f"  frustration: {len(flagged)} flagged frames, "
          f"{n_disagree} agent-vs-oracle disagreements")
    for s in out:
        print(f"    {s}: {len(out[s])} corrections")
    return out


def build_celebration_tasks(frame_log: List[dict], celebrated: List[int]
                            ) -> Dict[str, List[Tuple[torch.Tensor, int]]]:
    """At each celebration frame, store (state, agent_action_class)."""
    out: Dict[str, List[Tuple[torch.Tensor, int]]] = {
        "RECEIVE": [], "SETTLE": [],
    }
    n_kept = 0
    for j in celebrated:
        f = frame_log[j]
        skill = f["skill"]
        if skill is None or skill not in out:
            continue
        agent_act = f["action"]
        if agent_act not in SKILL_ACTION_TO_CLASS[skill]:
            continue
        cls = SKILL_ACTION_TO_CLASS[skill][agent_act]
        out[skill].append((f["std"], cls))
        n_kept += 1
    print(f"  celebration: {len(celebrated)} flagged frames, "
          f"{n_kept} positive-replay examples")
    for s in out:
        print(f"    {s}: {len(out[s])} reinforcements")
    return out


# ---------------------------------------------------------------------
# Per-skill donor refresh — retrain on synthetic + corrections.
# ---------------------------------------------------------------------


def refresh_donor(skill_name: str, group_name: str,
                  corrections: List[Tuple[torch.Tensor, int]],
                  out_dir: Path,
                  live_multiplier: int = 1,
                  n_synthetic_per_class: int = 1500) -> None:
    if not corrections:
        return
    print(f"  retraining {skill_name} with {len(corrections)} corrections "
          f"(synth/class={n_synthetic_per_class}, live×={live_multiplier})...")
    from experiments.atari_trioron.primitives.train_breakout_skill_donors import (
        make_task,
    )
    base_task = make_task(group_name, SKILL_GROUPS[group_name],
                          n_train_per_class=n_synthetic_per_class)
    X_corr_one = torch.cat([sv for sv, _ in corrections], dim=0)
    y_corr_one = torch.tensor([cls for _, cls in corrections], dtype=torch.long)
    if live_multiplier > 1:
        X_corr = X_corr_one.repeat(live_multiplier, 1)
        y_corr = y_corr_one.repeat(live_multiplier)
    else:
        X_corr = X_corr_one
        y_corr = y_corr_one
    Xtr = torch.cat([base_task.X_train, X_corr], dim=0)
    ytr = torch.cat([base_task.y_train, y_corr], dim=0)
    print(f"    augmented set: synth={base_task.X_train.shape[0]} + "
          f"live×{live_multiplier}={X_corr.shape[0]} = "
          f"{Xtr.shape[0]} ({100*X_corr.shape[0]/Xtr.shape[0]:.0f}% live)")
    perm = torch.randperm(Xtr.shape[0],
                          generator=torch.Generator().manual_seed(0))
    Xtr = Xtr[perm]
    ytr = ytr[perm]
    from trioron.api import (
        TaskData, TrioronConfig, AdvancedConfig, build_donor,
    )
    augmented_task = TaskData(
        name=group_name,
        X_train=Xtr, y_train=ytr,
        X_test=base_task.X_test, y_test=base_task.y_test,
        classes=base_task.classes,
    )
    cfg = TrioronConfig(
        cap_bytes=4_000,
        dream_replay_steps=50,
        advanced=AdvancedConfig(
            h_init=32, n_grow_per_task=4, l0_width=64, freeze_l0=True,
        ),
    )
    out_path = out_dir / f"{group_name}.pt"
    build_donor(label=group_name, tasks=[augmented_task],
                seed=42, epochs_per_task=6,
                config=cfg, out_path=out_path)


# ---------------------------------------------------------------------
# Multi-episode loop
# ---------------------------------------------------------------------


def run_dream_loop(*, seed: int, n_episodes: int, max_steps: int,
                   donor_dir: Path, out_dir: Path,
                   vary_seed: bool = False,
                   min_dream_examples: int = 5,
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
        for skill in ("RECEIVE", "SETTLE")
    }
    for ep in range(n_episodes):
        donors = {
            s: load_organism(donor_dir / f"{name}.pt")
            for s, name in [("RECEIVE", "BO_SKILL_RECEIVE"),
                            ("SETTLE",  "BO_SKILL_SETTLE")]
        }
        episode_seed = seed + ep if vary_seed else seed
        t0 = time.time()
        ret, n_steps, frame_log = play_episode(
            seed=episode_seed,
            max_steps=max_steps, donors=donors,
            out_dir=out_dir, record=(ep == 0 or ep == n_episodes - 1),
            name=f"breakout_dream_ep{ep}",
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
        satisfied = (eff_target is not None and ret >= eff_target)
        if eff_target is not None:
            flagged, celebrated = collect_episode_level(
                frame_log, ret, eff_target,
            )
            mode = ("FRUSTRATED" if flagged else "SATISFIED")
            print(f"  episode-level: ret={ret:+.1f} vs "
                  f"target={eff_target:+.1f} → {mode} "
                  f"(frustrated={len(flagged)}, celebrated={len(celebrated)})")
        else:
            celebrated = collect_celebrations(frame_log)
            flagged = collect_frustrations(frame_log)
        reinforcements = build_celebration_tasks(frame_log, celebrated)
        for skill in positive_buffer:
            for sv, cls in reinforcements[skill]:
                positive_buffer[skill].append((sv, cls))
        buf_summary = ", ".join(
            f"{s}={len(positive_buffer[s])}" for s in positive_buffer)
        print(f"  positive buffer: {buf_summary}")
        # Ratchet: when episode-level mode says SATISFIED (ret≥target),
        # populate the buffer (which we just did) but DON'T retrain.
        # Reinforcing the agent's already-good play at high live-
        # multipliers over-fits to the specific trajectory and disturbs
        # subsequent episodes (seed=0 regression in C-multiseed,
        # 2026-05-08). Pong's analog: `if ret > 0: continue`.
        if satisfied:
            print(f"  satisfied (ret≥target) — buffer updated, "
                  f"skipping retrain")
            continue
        if not flagged and not celebrated:
            print("  no frustrations or celebrations — skipping dream")
            continue
        corrections = build_correction_tasks(frame_log, flagged)
        for skill, group_name in [("RECEIVE", "BO_SKILL_RECEIVE"),
                                  ("SETTLE",  "BO_SKILL_SETTLE")]:
            n_corr = len(corrections[skill])
            n_reinf = len(reinforcements[skill])
            if n_corr + n_reinf < min_dream_examples:
                continue
            combined = (corrections[skill]
                        + reinforcements[skill]
                        + list(positive_buffer[skill]))
            print(f"  retrain set [{skill}]: corr={n_corr} "
                  f"reinf={n_reinf} buffer={len(positive_buffer[skill])}")
            refresh_donor(skill, group_name, combined, donor_dir,
                          live_multiplier=live_multiplier,
                          n_synthetic_per_class=n_synthetic_per_class)
    return returns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--donor-dir", type=str, default=str(DONOR_ROOT))
    ap.add_argument("--out-dir", type=str,
                    default=str(PROJ / "outputs" / "atari_primitive_donors"
                                / "breakout_dream_loop"))
    ap.add_argument("--vary-seed", action="store_true")
    ap.add_argument("--min-dream-examples", type=int, default=5,
                    help="Lower than Pong's 20 — Breakout frustration is "
                         "sparser (5 lives total).")
    ap.add_argument("--positive-buffer-size", type=int, default=500)
    ap.add_argument("--target-return", type=float, default=None,
                    help="Episode-level frustration mode. If set, every "
                         "in-play frame in an episode below this return "
                         "becomes a frustration correction; episodes "
                         "at-or-above become celebrations. Replaces the "
                         "rally-based per-life-loss / per-brick triggers. "
                         "Reframes frustration as 'unable to achieve a "
                         "high score' rather than 'lost this rally'.")
    ap.add_argument("--live-multiplier", type=int, default=1,
                    help="Oversample live corrections+reinforcements K× "
                         "in the augmented retrain set. Default 1 keeps "
                         "the synthetic prior dominant (~87%% synth at "
                         "K=1). K=5 shifts to ~40%% live.")
    ap.add_argument("--n-synthetic-per-class", type=int, default=1500,
                    help="Synthetic samples per class in the augmented "
                         "retrain set. Lower → less synthetic anchor → "
                         "more weight on live corrections.")
    ap.add_argument("--adaptive-target", action="store_true",
                    help="Vygotsky-style curriculum: initial target = "
                         "ep1_return + stretch, heats after K satisfied "
                         "eps, cools after M strained eps. Overrides "
                         "--target-return if both are set.")
    ap.add_argument("--at-stretch", type=float, default=2.0,
                    help="Initial stretch above ep1's return.")
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
    returns = run_dream_loop(
        seed=args.seed, n_episodes=args.episodes,
        max_steps=args.max_steps,
        donor_dir=Path(args.donor_dir),
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
        delta = returns[-1] - returns[0]
        print(f"  delta (last - first): {delta:+.1f}")


if __name__ == "__main__":
    main()
