"""Frustration-driven dream loop for the Pong skill organism.

Architectural arc (Rocky's framing): the agent should get FRUSTRATED
when it loses a rally, replay what just happened, and IMPROVE — like
a child practicing.

Per-episode loop:
  1. Play one Pong episode; record (raw_state, std_state, skill_used,
     action_taken, reward) for every in-play frame.
  2. Detect frustrations — every opp-score event (reward = -1) marks
     the last FRUSTRATION_WINDOW frames before it as "things I should
     have done differently."
  3. Compute the oracle's action at each frustration frame (the same
     phase-locked-SMASH rule used by pong_oracle).
  4. Keep only frames where the agent's action ≠ oracle's action —
     these are the actual mistakes worth correcting.
  5. Group disagreements by which skill was active (via the gate).
  6. Per affected skill: rebuild the donor on the union of original
     synthetic data + frustration corrections (full retrain, not
     api.extend, because corrections share class IDs with base data).
  7. Continue to the next episode with the updated donors.

The expectation: returns trend upward over episodes as the donors
absorb the corrections. If they don't, the architectural story is
weaker — and we'd need to look at oracle teacher capacity, dream
budget, or class-rebalancing as next levers.

Usage:
    python3 -m experiments.atari_trioron.primitives.play_with_dreams \\
        --episodes 5 --seed 0 --frustration-window 30
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import load_organism  # noqa: E402

from experiments.atari_trioron.features import get_ale  # noqa: E402
from experiments.atari_trioron.eval_render import _FlatRGBObsWrapper, _resolve_env_id  # noqa: E402
from experiments.atari_trioron.env import FRAME_HW, FRAME_STACK  # noqa: E402
from experiments.atari_trioron.primitives.synthetic_env import (  # noqa: E402
    standardize, predict_ball_y_at_impact,
    PADDLE_HEIGHT_84, SMASH_TRIGGER_X, _PRED_DX_EPS,
    PONG_ACTION_DEAD_ZONE, PADDLE_LOOKAHEAD,
    SKILL_CATCH_UP, SKILL_CATCH_DOWN, SKILL_CATCH_HOLD,
    SKILL_SMASH_UP, SKILL_SMASH_DOWN,
    SKILL_PREPOS_UP, SKILL_PREPOS_DOWN, SKILL_PREPOS_HOLD,
    PREPOS_ANCHOR_Y,
)
from experiments.atari_trioron.primitives.pong_skill_inference import (  # noqa: E402
    PongSkillStateBuilder, select_skill, select_action_for_skill,
    SKILL_CLASS_TO_ALE, SKILL_CLASS_SETS, DONOR_ROOT,
    ALE_HOLD, ALE_UP, ALE_DOWN,
    PADDLE_TOP_84, PADDLE_BOTTOM_84,
)
from experiments.atari_trioron.primitives.train_donors import (  # noqa: E402
    train_donor,
)
from experiments.atari_trioron.primitives.train_skill_donors import (  # noqa: E402
    SKILL_GROUPS,
)

gym.register_envs(ale_py)


# ---------------------------------------------------------------------
# Oracle action — same logic as pong_oracle (phase-locked SMASH @ 70).
# Reads the raw 9-d state and emits an ALE action class.
# ---------------------------------------------------------------------


def oracle_action_for_state(raw: np.ndarray, paddle_dy: float = 0.0) -> int:
    """Compute the oracle's chosen action for a raw state vector.

    Mirrors pong_oracle.evaluate_oracle's per-frame rule:
      - bdx ≤ 0 → HOLD
      - bx84 ≥ SMASH_TRIGGER_X (impact phase) → DOWN if opp_y > my_y else UP
      - else (catch phase): UP/DOWN/HOLD by pred_y vs eff_paddle_y
    """
    bx84 = float(raw[0])
    bdx = float(raw[2])
    py84 = float(raw[4])
    pdy = float(raw[5])
    pred_y = float(raw[7])
    opp_y84 = float(raw[8])
    if bdx <= _PRED_DX_EPS:
        return ALE_HOLD
    if bx84 >= SMASH_TRIGGER_X:
        if opp_y84 == py84:
            return ALE_HOLD
        return ALE_DOWN if opp_y84 > py84 else ALE_UP
    eff_py = py84 + pdy * PADDLE_LOOKAHEAD
    if pred_y < eff_py - PONG_ACTION_DEAD_ZONE:
        return ALE_UP
    if pred_y > eff_py + PONG_ACTION_DEAD_ZONE:
        return ALE_DOWN
    return ALE_HOLD


# ALE action → corresponding skill class ID (per skill).
SKILL_ACTION_TO_CLASS: Dict[str, Dict[int, int]] = {
    "CATCH": {ALE_UP: SKILL_CATCH_UP, ALE_DOWN: SKILL_CATCH_DOWN,
              ALE_HOLD: SKILL_CATCH_HOLD},
    "SMASH": {ALE_UP: SKILL_SMASH_UP, ALE_DOWN: SKILL_SMASH_DOWN},
    "PREPOS": {ALE_UP: SKILL_PREPOS_UP, ALE_DOWN: SKILL_PREPOS_DOWN,
               ALE_HOLD: SKILL_PREPOS_HOLD},
}


# ---------------------------------------------------------------------
# Episode runner — plays one game, returns frame buffer + return.
# ---------------------------------------------------------------------


def make_pong_env(seed: int, out_dir: Optional[Path], record: bool, name: str):
    env_id = _resolve_env_id("Pong")
    env = gym.make(
        env_id, render_mode="rgb_array",
        frameskip=1, repeat_action_probability=0.0,
        full_action_space=False,
    )
    if record and out_dir is not None:
        env = RecordVideo(env, video_folder=str(out_dir),
                          episode_trigger=lambda i: i == 0,
                          name_prefix=name, disable_logger=True)
    env = AtariPreprocessing(
        env, noop_max=30, frame_skip=4, screen_size=FRAME_HW,
        terminal_on_life_loss=False,
        grayscale_obs=True, grayscale_newaxis=False, scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)
    return env


def play_episode(*, seed: int, max_steps: int, donors: dict,
                 out_dir: Optional[Path] = None, record: bool = False,
                 name: str = "pong_dream_eval") -> Tuple[float, int, List[dict]]:
    env = make_pong_env(seed, out_dir, record, name)
    ale = get_ale(env)
    state_builder = PongSkillStateBuilder()

    obs, _info = env.reset(seed=seed)
    state_builder.reset()
    ret = 0.0
    n_steps = 0
    frame_log: List[dict] = []
    prev_skill = "CATCH"
    for _ in range(max_steps):
        sv_raw = state_builder.step(ale)
        if sv_raw[0] is None:
            action = ALE_HOLD
            sv = raw = skill = None
        else:
            sv, raw = sv_raw
            skill = select_skill(raw, prev_skill)
            paddle_y84 = float(raw[4])
            action = select_action_for_skill(donors[skill], sv, skill,
                                             paddle_y84=paddle_y84)
            prev_skill = skill
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r)
        n_steps += 1
        frame_log.append({
            "raw": None if raw is None else raw.copy(),
            "std": None if sv is None else sv.clone(),
            "skill": skill,
            "action": action,
            "reward": float(r),
        })
        if term or trunc:
            break
    env.close()
    return ret, n_steps, frame_log


# ---------------------------------------------------------------------
# Frustration extraction — flag the K frames before every opp-score.
# ---------------------------------------------------------------------


def collect_frustrations(frame_log: List[dict], window: int = 30) -> List[int]:
    """Return indices of frames in the K-frame window before each
    opp-score event (reward = -1)."""
    flagged: List[int] = []
    for i, f in enumerate(frame_log):
        if f["reward"] < 0:
            lo = max(0, i - window)
            for j in range(lo, i):
                if frame_log[j]["raw"] is not None:
                    flagged.append(j)
    # Deduplicate (overlapping windows on consecutive losses)
    return sorted(set(flagged))


def collect_celebrations(frame_log: List[dict], window: int = 30) -> List[int]:
    """Symmetric to collect_frustrations — frames before each AGENT
    score (reward = +1). These are positive examples: agent's action
    led to a winning rally, so reinforce them by training the
    responsible skill donor on (state, agent_action_class)."""
    flagged: List[int] = []
    for i, f in enumerate(frame_log):
        if f["reward"] > 0:
            lo = max(0, i - window)
            for j in range(lo, i):
                if frame_log[j]["raw"] is not None:
                    flagged.append(j)
    return sorted(set(flagged))


def build_correction_tasks(frame_log: List[dict], flagged: List[int]
                           ) -> Dict[str, List[Tuple[torch.Tensor, int]]]:
    """Frustration: oracle-action labels at flagged frames where agent
    disagreed with oracle. Returns per-skill list of corrections."""
    out: Dict[str, List[Tuple[torch.Tensor, int]]] = {
        "CATCH": [], "SMASH": [], "PREPOS": [],
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
    """Positive replay: at each celebration frame, store (state,
    agent_action_class) — reinforce what worked. We only keep frames
    where the agent action is representable in its skill's class set
    (filters out edge cases where override fired)."""
    out: Dict[str, List[Tuple[torch.Tensor, int]]] = {
        "CATCH": [], "SMASH": [], "PREPOS": [],
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
                  out_dir: Path) -> None:
    """Rebuild donor with augmented dataset. Calls build_donor via the
    train_donor helper to keep training config consistent.

    Augmentation: the original synthetic dataset is regenerated by
    train_donor (it owns make_task), then we monkey-patch make_task
    to append our corrections. Simpler: pre-generate corrections and
    pass via a custom TaskData."""
    if not corrections:
        return
    print(f"  retraining {skill_name} with {len(corrections)} corrections...")
    # Stack corrections into tensors and build a TaskData via the
    # standard make_task, then concatenate. We replicate make_task's
    # logic here to inject corrections cleanly.
    from experiments.atari_trioron.primitives.train_donors import make_task
    base_task = make_task(group_name, SKILL_GROUPS[group_name])
    X_corr = torch.cat([sv for sv, _ in corrections], dim=0)
    y_corr = torch.tensor([cls for _, cls in corrections], dtype=torch.long)
    # Append corrections to training set; leave test set untouched
    # (we want held-out eval on synthetic, not on the corrections).
    Xtr = torch.cat([base_task.X_train, X_corr], dim=0)
    ytr = torch.cat([base_task.y_train, y_corr], dim=0)
    perm = torch.randperm(Xtr.shape[0],
                          generator=torch.Generator().manual_seed(0))
    Xtr = Xtr[perm]
    ytr = ytr[perm]
    from trioron.api import TaskData, TrioronConfig, AdvancedConfig, build_donor
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
                   frustration_window: int, donor_dir: Path,
                   out_dir: Path, vary_seed: bool = False,
                   min_dream_examples: int = 20) -> List[float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    returns: List[float] = []
    for ep in range(n_episodes):
        donors = {
            s: load_organism(donor_dir / f"{name}.pt")
            for s, name in [("CATCH", "SKILL_CATCH"),
                            ("SMASH", "SKILL_SMASH"),
                            ("PREPOS", "SKILL_PREPOS")]
        }
        episode_seed = seed + ep if vary_seed else seed
        t0 = time.time()
        ret, n_steps, frame_log = play_episode(
            seed=episode_seed,
            max_steps=max_steps, donors=donors,
            out_dir=out_dir, record=(ep == 0 or ep == n_episodes - 1),
            name=f"pong_dream_ep{ep}",
        )
        elapsed = time.time() - t0
        print(f"\n[ep {ep+1}/{n_episodes} seed={episode_seed}] "
              f"return={ret:+.1f} length={n_steps} ({elapsed:.1f}s)")
        returns.append(ret)
        # FIX 1 — skip-dream-on-wins. If the agent net-won, don't
        # retrain. Over-correcting after wins erased prior beneficial
        # drift in the previous run (ep 6 +20 → ep 7 -19 regression).
        if ret > 0:
            print(f"  net win (return={ret:+.1f}) — skipping dream")
            continue
        flagged = collect_frustrations(frame_log, window=frustration_window)
        celebrated = collect_celebrations(frame_log, window=frustration_window)
        if not flagged and not celebrated:
            print("  no frustrations or celebrations — skipping dream")
            continue
        corrections = build_correction_tasks(frame_log, flagged)
        reinforcements = build_celebration_tasks(frame_log, celebrated)
        for skill, group_name in [("CATCH", "SKILL_CATCH"),
                                  ("SMASH", "SKILL_SMASH"),
                                  ("PREPOS", "SKILL_PREPOS")]:
            # FIX 2 — minimum disagreement threshold. Tiny corrections
            # get drowned by 4500 synthetic samples; the retrain ends
            # up near a fresh fit, erasing prior drift.
            n_corr = len(corrections[skill])
            n_reinf = len(reinforcements[skill])
            if n_corr + n_reinf < min_dream_examples:
                continue
            # FIX 3 — positive replay merged in alongside corrections.
            combined = corrections[skill] + reinforcements[skill]
            refresh_donor(skill, group_name, combined, donor_dir)
    return returns


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--frustration-window", type=int, default=30,
                    help="Frames before each opp-score to flag as mistakes.")
    ap.add_argument("--donor-dir", type=str, default=str(DONOR_ROOT))
    ap.add_argument("--out-dir", type=str,
                    default=str(PROJ / "outputs" / "atari_primitive_donors"
                                / "dream_loop"))
    ap.add_argument("--vary-seed", action="store_true",
                    help="Use seed+ep instead of fixed seed each episode "
                         "(tests generalization; default = same seed for "
                         "clean improvement signal).")
    ap.add_argument("--min-dream-examples", type=int, default=20,
                    help="Skip per-skill dream when (corrections + "
                         "reinforcements) is below this. Avoids tiny-batch "
                         "retrains that erase prior beneficial drift.")
    args = ap.parse_args()
    returns = run_dream_loop(
        seed=args.seed, n_episodes=args.episodes,
        max_steps=args.max_steps,
        frustration_window=args.frustration_window,
        donor_dir=Path(args.donor_dir),
        out_dir=Path(args.out_dir),
        vary_seed=args.vary_seed,
        min_dream_examples=args.min_dream_examples,
    )
    print("\n=== Dream-loop trajectory ===")
    for i, r in enumerate(returns):
        print(f"  ep {i+1}: {r:+.1f}")
    if len(returns) >= 2:
        delta = returns[-1] - returns[0]
        print(f"  delta (last - first): {delta:+.1f}")


if __name__ == "__main__":
    main()
