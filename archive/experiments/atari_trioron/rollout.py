"""Rollout: run an organism (or random policy) in ALE, collect
(state, action, return) per episode.

Two modes:
  - organism is None → uniform random sampling over the env's action
    space. The naive bootstrap; trioron has no opinion yet.
  - organism is loaded → sample action from softmax(per_class_log_lik)
    over the action axis, with optional ε-greedy exploration mixed
    in. Action-space mismatch (Pong's 6 vs Breakout's 4) is handled
    by masking out-of-game logits to -inf before softmax.

Every state-action emitted carries the full episode's return — the
return-filter in `filter.py` decides which episodes survive into
training data.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .env import GAME_ACTION_MASK, N_ACTIONS, make_env, obs_to_tensor


@dataclass
class Episode:
    """One episode's worth of (state, action, return) tuples."""
    states: torch.Tensor          # (T, OBS_DIM) float32
    actions: torch.Tensor         # (T,) int64
    return_: float                # episode return (sum of clipped rewards)
    length: int                   # T


def _select_action(
    organism,
    obs_tensor: torch.Tensor,
    game_mask: np.ndarray,
    eps: float,
    rng: np.random.Generator,
) -> int:
    """Pick an action from the organism's union-class logits, masked
    to the game's valid action subset, with ε-greedy exploration.

    Uses MultiBranchOrganism.forward(routing="soft") so this path
    works for:
      - single-branch organisms (arm1, arm2, arm3) — soft-routing
        over one branch is a no-op pass-through;
      - multi-branch organisms (arm4) — the gate picks which branch
        contributes per-frame, based on archive log-likelihood of
        the L0-projected frame.

    For zero-shot transfer (Pong→eval Breakout), the organism's
    archive only covers Pong-seen classes; classes outside that
    coverage stay at -inf, and the game mask further restricts to
    Breakout's valid 4-action subset.
    """
    valid_idx = np.flatnonzero(game_mask)
    # ε-greedy: uniform over game's valid actions.
    if rng.random() < eps:
        return int(rng.choice(valid_idx))
    if organism is None:
        return int(rng.choice(valid_idx))

    with torch.no_grad():
        logits = organism(obs_tensor, routing="soft").squeeze(0)
    # Map union-class logits onto N_ACTIONS slots; absent classes -inf.
    full = torch.full((N_ACTIONS,), float("-inf"))
    for i, c in enumerate(organism.union_classes):
        c_int = int(c)
        if 0 <= c_int < N_ACTIONS:
            full[c_int] = logits[i]
    # Mask out-of-game actions.
    mask_t = torch.from_numpy(game_mask).bool()
    full = torch.where(mask_t, full, torch.full_like(full, float("-inf")))
    # No class overlap with the game's actions → uniform fallback.
    if torch.isinf(full).all():
        return int(rng.choice(valid_idx))
    # Softmax-sample so rollout stays stochastic after the organism has
    # opinions — the self-imitation filter needs occasional exploration
    # to discover non-greedy good moves.
    probs = torch.softmax(full, dim=0).numpy()
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        return int(rng.choice(valid_idx))
    return int(rng.choice(N_ACTIONS, p=probs / probs.sum()))


def collect_episodes(
    *,
    game: str,
    organism=None,
    n_episodes: int = 16,
    eps: float = 0.05,
    seed: int = 0,
    max_steps_per_episode: int = 10_000,
    terminal_on_life_loss: bool = True,
    verbose: bool = True,
) -> List[Episode]:
    """Run `n_episodes` episodes; return per-episode (state, action,
    return) records.

    With life-loss-as-terminal on (default), each life is its own
    "episode" — that's the standard Atari preprocessing trick that
    gives the agent denser feedback. Pass False for true-score eval.
    """
    rng = np.random.default_rng(seed)
    game_mask = GAME_ACTION_MASK[game]
    env = make_env(game, seed=seed,
                   terminal_on_life_loss=terminal_on_life_loss)
    out: List[Episode] = []
    for ep_i in range(n_episodes):
        obs, _info = env.reset(seed=seed + ep_i)
        states: List[np.ndarray] = []
        actions: List[int] = []
        ret = 0.0
        for _t in range(max_steps_per_episode):
            obs_t = obs_to_tensor(obs)
            a = _select_action(organism, obs_t, game_mask, eps, rng)
            states.append(obs)
            actions.append(a)
            obs, r, term, trunc, _info = env.step(a)
            ret += float(r)
            if term or trunc:
                break
        ep = Episode(
            states=torch.from_numpy(np.stack(states)).float(),
            actions=torch.tensor(actions, dtype=torch.long),
            return_=ret,
            length=len(actions),
        )
        out.append(ep)
        if verbose:
            print(f"  [rollout] ep {ep_i+1}/{n_episodes}: "
                  f"len={ep.length} return={ret:+.1f}")
    env.close()
    if verbose:
        rets = np.array([e.return_ for e in out])
        print(f"  [rollout] {len(out)} episodes: "
              f"return mean={rets.mean():+.2f} median={np.median(rets):+.2f} "
              f"max={rets.max():+.1f}")
    return out
