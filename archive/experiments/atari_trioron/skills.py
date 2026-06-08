"""Skill rules + skill-driven data collection.

A "skill" is a deterministic state→action rule. The skill labeler runs
the rule and emits the action to take. We use the skill labeler as a
**teacher policy**: roll out in ALE with the skill labeler driving,
log (frame_stack, action) pairs, group by action, hand the groups to
trioron.api as a sequenced curriculum.

The frame_stack the trioron sees is the standard 84×84×4 preprocessed
input — same as a vanilla Atari DQN. The skill labeler peeks at RAM,
which the trioron does **not** see at any point.

Skill list:
  Breakout
    SERVE     — no ball in play          → FIRE (1)
    LEFT      — ball is left of paddle   → LEFT (3)
    RIGHT     — ball is right of paddle  → RIGHT (2)
    HOLD      — ball is overhead         → NOOP (0)
  Pong
    UP        — ball is above paddle     → action 2 (UP, paddle moves toward small y)
    DOWN      — ball is below paddle     → action 3 (DOWN, paddle moves toward large y)
    HOLD      — ball roughly aligned     → NOOP (0)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .features import (
    BreakoutState, PongState, extract_breakout, extract_pong, get_ale,
)


# ------ skill-action constants ------
SKILL_LABELS_BREAKOUT = ["SERVE", "LEFT", "RIGHT", "HOLD"]
SKILL_LABELS_PONG = ["UP", "DOWN", "HOLD"]

# Hysteresis margin: if |ball_x - paddle_x| < this, call it HOLD instead
# of LEFT/RIGHT. Without it the agent jitters around the center pixel.
PADDLE_BREAKOUT_DEAD_ZONE = 4    # RAM-units (each = 1 raw pixel)
PADDLE_PONG_DEAD_ZONE = 6


def skill_for_breakout(state: BreakoutState) -> Tuple[str, int]:
    """Returns (skill_name, action_class)."""
    if state.is_serve:
        return ("SERVE", 1)        # FIRE
    dx = state.ball_x - state.paddle_x
    if dx < -PADDLE_BREAKOUT_DEAD_ZONE:
        return ("LEFT", 3)
    if dx > PADDLE_BREAKOUT_DEAD_ZONE:
        return ("RIGHT", 2)
    return ("HOLD", 0)


def skill_for_pong(state: PongState) -> Tuple[str, int]:
    """Returns (skill_name, action_class)."""
    if state.is_serve:
        # Pong serves automatically; press FIRE only briefly out of
        # safety. Treat serve frames as HOLD so we don't spam FIRE.
        return ("HOLD", 0)
    dy = state.ball_y - state.my_paddle_y
    if dy < -PADDLE_PONG_DEAD_ZONE:
        return ("UP", 2)
    if dy > PADDLE_PONG_DEAD_ZONE:
        return ("DOWN", 3)
    return ("HOLD", 0)


# ------ Data-collection rollout driven by skill labeler ------

@dataclass
class SkillSample:
    frame_stack: np.ndarray      # (OBS_DIM,) float32 in [0,1]
    skill: str
    action: int
    return_so_far: float


def collect_skill_data(
    *,
    game: str,
    n_episodes: int = 16,
    seed: int = 0,
    max_steps_per_episode: int = 5_000,
    eps_explore: float = 0.05,
    verbose: bool = True,
) -> List[SkillSample]:
    """Roll out in ALE driven by the skill labeler. At each step:
      - read RAM via get_ale(env).getRAM() and extract state
      - apply the skill rule to pick action
      - with probability eps_explore, take a random action (still a
        valid action in the game's action set) — gives the trioron
        edge-case state coverage
      - log the standard preprocessed framestack the trioron will see
        plus the skill label and action
    """
    from .env import make_env, GAME_ACTION_MASK

    rng = np.random.default_rng(seed)
    env = make_env(game, seed=seed, terminal_on_life_loss=True)
    ale = get_ale(env)
    extract = extract_breakout if game == "Breakout" else extract_pong
    skill_fn = (skill_for_breakout if game == "Breakout"
                else skill_for_pong)
    valid_actions = np.flatnonzero(GAME_ACTION_MASK[game])

    samples: List[SkillSample] = []
    skill_counts: Dict[str, int] = {}
    rets: List[float] = []
    for ep_i in range(n_episodes):
        obs, _info = env.reset(seed=seed + ep_i)
        ret = 0.0
        for _t in range(max_steps_per_episode):
            state = extract(ale)
            skill, action = skill_fn(state)
            if rng.random() < eps_explore:
                action = int(rng.choice(valid_actions))
                skill = f"{skill}*explore"
            samples.append(SkillSample(
                frame_stack=obs.astype(np.float32),
                skill=skill, action=int(action), return_so_far=ret,
            ))
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
            obs, r, term, trunc, _info = env.step(action)
            ret += float(r)
            if term or trunc:
                break
        rets.append(ret)
        if verbose:
            print(f"  [skill-rollout] ep {ep_i+1}/{n_episodes}: "
                  f"return={ret:+.1f}")
    env.close()
    if verbose:
        rets_np = np.array(rets)
        print(f"  [skill-rollout] {n_episodes} episodes: "
              f"return mean={rets_np.mean():+.2f} max={rets_np.max():+.0f}")
        total = sum(skill_counts.values())
        for k, v in sorted(skill_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {k:18s}: {v:5d} ({100*v/total:.1f}%)")
    return samples


def group_by_action(samples: List[SkillSample]) -> Dict[int, Dict]:
    """Group skill-labeled samples by action. Each action's group is
    a separate trioron task (same y across the group; the L0-code
    cluster is what the manifold archive captures)."""
    out: Dict[int, Dict[str, List]] = {}
    for s in samples:
        a = int(s.action)
        if a not in out:
            out[a] = {"X": [], "y": [], "skills": []}
        out[a]["X"].append(s.frame_stack)
        out[a]["y"].append(a)
        out[a]["skills"].append(s.skill)
    final: Dict[int, Dict] = {}
    for a, group in out.items():
        final[a] = {
            "X": torch.from_numpy(np.stack(group["X"])).float(),
            "y": torch.tensor(group["y"], dtype=torch.long),
            "n": len(group["X"]),
            "skill_examples": group["skills"][:5],
        }
    return final
