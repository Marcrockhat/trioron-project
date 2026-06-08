"""Evaluate the 5-organism × 2-game cross-game matrix.

For each organism in {trioron-P, trioron-B, trioron-PB, trioron-BP,
trioron-PB-absorb}, evaluate on Pong and Breakout. Multi-seed
inference (no dream loop). Reports the full 5×2 return matrix.

Each evaluation:
  - Load the organism via load_organism
  - Use it as the backing for ALL skills of the active game
  - Per-frame: state builder → skill gate → mask logits to skill's
    classes → argmax → ALE action

If an organism doesn't know a game's classes (e.g., trioron-P on
Breakout), the masked-argmax falls back to ALE_NOOP (or ALE_FIRE
on serve for Breakout). This is the "no transfer baseline" for
catastrophic-forgetting-style measurement.

Usage:
    python3 -m experiments.atari_trioron.primitives.eval_cross_game_matrix
    python3 -m experiments.atari_trioron.primitives.eval_cross_game_matrix --seeds 0 1 2 3 4
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from trioron.api import load_organism  # noqa: E402

from experiments.atari_trioron.features import (  # noqa: E402
    extract_pong, extract_breakout, get_ale,
)
from experiments.atari_trioron.eval_render import _FlatRGBObsWrapper, _resolve_env_id  # noqa: E402
from experiments.atari_trioron.env import FRAME_HW, FRAME_STACK  # noqa: E402

# Pong inference imports
from experiments.atari_trioron.primitives.pong_skill_inference import (  # noqa: E402
    PongSkillStateBuilder, select_skill as select_skill_pong,
    select_action_for_skill as pong_action,
    ALE_HOLD as PONG_HOLD,
)

# Breakout inference imports
from experiments.atari_trioron.primitives.breakout_state import (  # noqa: E402
    BreakoutStateBuilder, select_skill_breakout,
)
from experiments.atari_trioron.primitives.breakout_skill_inference import (  # noqa: E402
    select_action_for_skill as breakout_action,
    ALE_NOOP as BO_NOOP, ALE_FIRE as BO_FIRE,
)


OUT_ROOT = PROJ / "outputs" / "atari_primitive_donors"

ORGANISMS = {
    "trioron-P":         OUT_ROOT / "trioron_P_skill.pt",
    "trioron-B":         OUT_ROOT / "trioron_B_skill.pt",
    "trioron-PB":        OUT_ROOT / "trioron_PB_skill.pt",
    "trioron-BP":        OUT_ROOT / "trioron_BP_skill.pt",
    "trioron-PB-absorb": OUT_ROOT / "trioron_PB_absorb.pt",
}


def play_pong(organism, *, seed: int, max_steps: int = 8000) -> dict:
    env_id = _resolve_env_id("Pong")
    env = gym.make(env_id, render_mode="rgb_array",
                   frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4,
                             screen_size=FRAME_HW,
                             terminal_on_life_loss=False,
                             grayscale_obs=True, grayscale_newaxis=False,
                             scale_obs=False)
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)
    ale = get_ale(env)
    sb = PongSkillStateBuilder()
    obs, _ = env.reset(seed=seed)
    sb.reset()
    ret = 0.0
    n_steps = 0
    prev_skill = "CATCH"
    donors = {"CATCH": organism, "SMASH": organism, "PREPOS": organism}
    for _ in range(max_steps):
        sv_raw = sb.step(ale)
        if sv_raw[0] is None:
            action = PONG_HOLD
        else:
            sv, raw = sv_raw
            skill = select_skill_pong(raw, prev_skill)
            paddle_y84 = float(raw[4])
            action = pong_action(donors[skill], sv, skill,
                                 paddle_y84=paddle_y84,
                                 fallback_action=PONG_HOLD)
            prev_skill = skill
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r)
        n_steps += 1
        if term or trunc:
            break
    env.close()
    return {"return": ret, "length": n_steps}


def play_breakout(organism, *, seed: int, max_steps: int = 8000) -> dict:
    env_id = _resolve_env_id("Breakout")
    env = gym.make(env_id, render_mode="rgb_array",
                   frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4,
                             screen_size=FRAME_HW,
                             terminal_on_life_loss=False,
                             grayscale_obs=True, grayscale_newaxis=False,
                             scale_obs=False)
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)
    ale = get_ale(env)
    sb = BreakoutStateBuilder()
    obs, _ = env.reset(seed=seed)
    sb.reset()
    ret = 0.0
    n_steps = 0
    donors = {"RECEIVE": organism, "SETTLE": organism}
    for _ in range(max_steps):
        sv_raw = sb.step(ale)
        s = extract_breakout(ale)
        if sv_raw[0] is None:
            action = BO_FIRE if not s.ball_in_play else BO_NOOP
        else:
            sv, raw = sv_raw
            paddle_x84 = float(raw[4])
            skill = select_skill_breakout(raw)
            action = breakout_action(donors[skill], sv, skill,
                                     paddle_x84=paddle_x84,
                                     fallback_action=BO_NOOP)
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r)
        n_steps += 1
        if term or trunc:
            break
    env.close()
    return {"return": ret, "length": n_steps}


def evaluate(seeds: List[int]) -> Dict:
    matrix: Dict[str, Dict[str, dict]] = {}
    for org_name, org_path in ORGANISMS.items():
        if not org_path.exists():
            print(f"  SKIP {org_name}: missing {org_path.name}")
            continue
        organism = load_organism(org_path)
        org_results: Dict[str, dict] = {}
        for game in ("Pong", "Breakout"):
            returns = []
            lengths = []
            play = play_pong if game == "Pong" else play_breakout
            for s in seeds:
                r = play(organism, seed=s)
                returns.append(r["return"])
                lengths.append(r["length"])
            org_results[game] = {
                "returns": returns,
                "mean": float(np.mean(returns)),
                "std": float(np.std(returns)),
                "lengths": lengths,
            }
            print(f"  {org_name:20s} on {game:10s}: "
                  f"{[f'{r:+.0f}' for r in returns]}  "
                  f"mean={np.mean(returns):+.1f} ± {np.std(returns):.1f}")
        matrix[org_name] = org_results
    return matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    print(f"\n=== Cross-game eval matrix (seeds={args.seeds}) ===\n")
    matrix = evaluate(args.seeds)

    print("\n=== SUMMARY ===")
    print(f"{'organism':22s}  {'Pong mean':>12s}  {'Breakout mean':>14s}")
    print("-" * 55)
    for org_name in ORGANISMS:
        if org_name not in matrix:
            continue
        pm = matrix[org_name]["Pong"]["mean"]
        bm = matrix[org_name]["Breakout"]["mean"]
        print(f"{org_name:22s}  {pm:+12.1f}  {bm:+14.1f}")


if __name__ == "__main__":
    main()
