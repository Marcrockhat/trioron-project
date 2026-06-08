"""Skill-curriculum training: distill the skill labeler's policy
into trioron via the api.

Each game's skills get a disjoint slice of the global class-ID space
so api.absorb works on graft (arm 4):

    Pong skills      (UP, DOWN, HOLD)             classes 10, 11, 12
    Breakout skills  (FIRE, LEFT, RIGHT, HOLD)    classes 20, 21, 22, 23

Disjoint class IDs preserve the canonical shared-L0 invariant for
absorb: each donor covers a different class subset, so the union
construction has no collisions. A side benefit is interpretability —
class 21 means "Breakout LEFT-skill" specifically, not just "action 3".

At eval time, per-class log-likelihood gives the most-likely skill
class; SKILL_TO_ACTION maps it to the actual game action.
"""
from __future__ import annotations
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from trioron.api import (
    TaskData, TrioronConfig, AdvancedConfig,
    build_donor, extend, absorb, load_organism,
)

from .env import make_env, GAME_ACTION_MASK
from .skills import (
    SkillSample, collect_skill_data, group_by_action,
    skill_for_breakout, skill_for_pong,
)
from .features import extract_breakout, extract_pong, get_ale


# --- Disjoint skill class-ID space ---
SKILL_CLASS_PONG_UP = 10
SKILL_CLASS_PONG_DOWN = 11
SKILL_CLASS_PONG_HOLD = 12
SKILL_CLASS_BR_FIRE = 20
SKILL_CLASS_BR_LEFT = 21
SKILL_CLASS_BR_RIGHT = 22
SKILL_CLASS_BR_HOLD = 23

# Map skill names → global class IDs.
PONG_SKILL_TO_CLASS = {
    "UP": SKILL_CLASS_PONG_UP,
    "DOWN": SKILL_CLASS_PONG_DOWN,
    "HOLD": SKILL_CLASS_PONG_HOLD,
}
BREAKOUT_SKILL_TO_CLASS = {
    "SERVE": SKILL_CLASS_BR_FIRE,
    "LEFT":  SKILL_CLASS_BR_LEFT,
    "RIGHT": SKILL_CLASS_BR_RIGHT,
    "HOLD":  SKILL_CLASS_BR_HOLD,
}

# Map global class IDs → game action.
SKILL_TO_ACTION = {
    SKILL_CLASS_PONG_UP:    2,
    SKILL_CLASS_PONG_DOWN:  3,
    SKILL_CLASS_PONG_HOLD:  0,
    SKILL_CLASS_BR_FIRE:    1,
    SKILL_CLASS_BR_LEFT:    3,
    SKILL_CLASS_BR_RIGHT:   2,
    SKILL_CLASS_BR_HOLD:    0,
}

# Per-game eligible skill classes (used to mask non-game classes at
# inference: don't pick "Pong-UP" while playing Breakout, even if the
# absorbed organism's archive happens to vote for it).
PONG_SKILL_CLASSES = list(PONG_SKILL_TO_CLASS.values())
BREAKOUT_SKILL_CLASSES = list(BREAKOUT_SKILL_TO_CLASS.values())


def _samples_to_taskdata(
    samples: List[SkillSample],
    skill_to_class: Dict[str, int],
    name: str,
    per_skill_cap: Optional[int] = 800,
    train_split: float = 0.85,
) -> TaskData:
    """Convert skill samples → one TaskData covering all the game's
    skills as classes. Caps per-skill sample count so HOLD doesn't
    swamp the scarcer SERVE/LEFT/RIGHT.

    Filters out '*explore' skills (random exploration steps) — those
    are noise on the labeler signal and would teach the trioron to do
    random things in clean states."""
    rng = np.random.default_rng(0)
    by_skill: Dict[str, List[SkillSample]] = {}
    for s in samples:
        if "*explore" in s.skill:
            continue
        by_skill.setdefault(s.skill, []).append(s)
    keep_X: List[np.ndarray] = []
    keep_y: List[int] = []
    for skill, group in by_skill.items():
        if skill not in skill_to_class:
            continue
        if per_skill_cap and len(group) > per_skill_cap:
            idx = rng.choice(len(group), size=per_skill_cap, replace=False)
            group = [group[i] for i in idx]
        cls = skill_to_class[skill]
        for s in group:
            keep_X.append(s.frame_stack)
            keep_y.append(cls)
    if not keep_X:
        raise RuntimeError(f"no usable skill samples in {len(samples)} samples")
    X = torch.from_numpy(np.stack(keep_X)).float()
    y = torch.tensor(keep_y, dtype=torch.long)
    classes = sorted(set(keep_y))

    n = X.shape[0]
    n_tr = max(1, int(train_split * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    X_tr, X_te = X[perm[:n_tr]], X[perm[n_tr:]]
    y_tr, y_te = y[perm[:n_tr]], y[perm[n_tr:]]
    if X_te.shape[0] == 0:
        X_te, y_te = X_tr.clone(), y_tr.clone()
    return TaskData(
        name=name,
        X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        classes=classes,
    )


# ----- Per-arm builders -----

def build_skill_donor(
    *,
    game: str,
    out_path: Path,
    n_episodes: int = 16,
    eps_explore: float = 0.05,
    epochs_per_task: int = 8,
    cap_bytes: int = 64_000,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """Cold-start: collect skill rollouts for `game`, build a donor
    on the (frame_stack, skill_class) tuples."""
    skill_to_class = (PONG_SKILL_TO_CLASS if game == "Pong"
                      else BREAKOUT_SKILL_TO_CLASS)
    if verbose:
        print(f"\n=== build_skill_donor({game}) ===")
        print(f"[1/2] collecting {n_episodes} skill-driven rollouts")
    samples = collect_skill_data(
        game=game, n_episodes=n_episodes, seed=seed,
        eps_explore=eps_explore, verbose=verbose,
    )
    task = _samples_to_taskdata(
        samples, skill_to_class, name=f"{game}_skills",
    )
    if verbose:
        print(f"[2/2] training donor (input_dim={task.X_train.shape[1]}, "
              f"classes={task.classes})")
    cfg = TrioronConfig(
        cap_bytes=cap_bytes,
        dream_replay_steps=50,
        advanced=AdvancedConfig(
            h_init=32, n_grow_per_task=4,
            l0_width=128, freeze_l0=True,
        ),
    )
    return build_donor(
        label=f"{game}_skills",
        tasks=[task],
        seed=seed, epochs_per_task=epochs_per_task,
        config=cfg, out_path=out_path,
    )


def extend_skill_donor(
    *,
    base_donor: Path,
    new_game: str,
    out_path: Path,
    n_episodes: int = 16,
    eps_explore: float = 0.05,
    epochs_per_task: int = 8,
    extension_cap_bytes: int = 128_000,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """Extension: take an existing skill donor and learn a new game's
    skills on top via api.extend. Used for arm 2.

    Note: api.extend's `base_tasks` parameter is required. We re-collect
    a small batch of base-game samples to feed the consolidation dream's
    real-data replay over past skills.
    """
    if verbose:
        print(f"\n=== extend_skill_donor(base={base_donor.name}, "
              f"new={new_game}) ===")

    # Figure out base game from class IDs in donor payload
    payload = torch.load(str(base_donor), map_location="cpu",
                         weights_only=False)
    base_classes = payload.get("classes_covered", [])
    if any(c in PONG_SKILL_CLASSES for c in base_classes):
        base_game = "Pong"
        base_skill_to_class = PONG_SKILL_TO_CLASS
    elif any(c in BREAKOUT_SKILL_CLASSES for c in base_classes):
        base_game = "Breakout"
        base_skill_to_class = BREAKOUT_SKILL_TO_CLASS
    else:
        raise RuntimeError(
            f"base donor's classes {base_classes} don't match any "
            f"known skill class set"
        )

    new_skill_to_class = (PONG_SKILL_TO_CLASS if new_game == "Pong"
                          else BREAKOUT_SKILL_TO_CLASS)

    if verbose:
        print(f"[1/3] base game inferred as {base_game}; collecting "
              f"replay data for consolidation dream")
    base_samples = collect_skill_data(
        game=base_game, n_episodes=max(2, n_episodes // 4),
        seed=seed + 9999, eps_explore=eps_explore, verbose=False,
    )
    base_task = _samples_to_taskdata(
        base_samples, base_skill_to_class,
        name=f"{base_game}_skills_replay",
    )

    if verbose:
        print(f"[2/3] collecting {n_episodes} new-game ({new_game}) rollouts")
    new_samples = collect_skill_data(
        game=new_game, n_episodes=n_episodes, seed=seed,
        eps_explore=eps_explore, verbose=verbose,
    )
    new_task = _samples_to_taskdata(
        new_samples, new_skill_to_class, name=f"{new_game}_skills",
    )

    if verbose:
        print(f"[3/3] api.extend (base classes={base_task.classes}, "
              f"new classes={new_task.classes})")
    return extend(
        donor_path=base_donor,
        base_tasks=[base_task],
        new_tasks=[new_task],
        out_path=out_path,
        extension_cap_bytes=extension_cap_bytes,
        epochs_per_task=epochs_per_task,
        permanent_int8=False,
    )


def absorb_skill_donors(
    *,
    pong_donor: Path,
    breakout_donor: Path,
    out_path: Path,
    verbose: bool = True,
) -> Path:
    """Graft: independent Pong + Breakout skill donors → one
    multi-branch organism. The two donors cover disjoint class sets
    (Pong: {10,11,12}, Breakout: {20,21,22,23}), so absorb's class-
    namespace check passes."""
    if verbose:
        print(f"\n=== absorb_skill_donors({pong_donor.name} ⊕ "
              f"{breakout_donor.name}) ===")
    return absorb(
        donor_paths=[pong_donor, breakout_donor],
        out_path=out_path,
    )
