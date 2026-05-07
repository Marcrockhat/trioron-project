"""Predict-y oracle for Pong — no donor, hand-coded policy.

Edge-hit variant. The action targets a paddle position OFFSET from the
ball's predicted landing y, so the ball strikes the paddle's far edge
relative to the opponent — deflecting away from where the opponent is.

    target_paddle_y = pred_y + sign(opp_y - my_y) * EDGE_OFFSET

EDGE_OFFSET = 0 reproduces the prior catching oracle (defense-only).

Usage:
    python3 -m experiments.atari_trioron.primitives.pong_oracle
    python3 -m experiments.atari_trioron.primitives.pong_oracle --edge-offset 3.0
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo

PROJ = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from experiments.atari_trioron.features import extract_pong, get_ale  # noqa: E402
from experiments.atari_trioron.eval_render import _FlatRGBObsWrapper, _resolve_env_id  # noqa: E402
from experiments.atari_trioron.env import FRAME_HW, FRAME_STACK  # noqa: E402
from experiments.atari_trioron.primitives.synthetic_env import (  # noqa: E402
    predict_ball_y_at_impact, PONG_ACTION_DEAD_ZONE, PADDLE_LOOKAHEAD,
    PADDLE_HEIGHT_84,
)
from experiments.atari_trioron.primitives.pong_inference import (  # noqa: E402
    _ram_to_84, RAM_BALL_X_RANGE, RAM_BALL_Y_RANGE, RAM_PADDLE_Y_RANGE,
    ALE_FRAME_SKIP,
)

gym.register_envs(ale_py)

ALE_HOLD, ALE_UP, ALE_DOWN = 0, 2, 3


def evaluate_oracle(*, out_dir: Path, seed: int = 0, max_steps: int = 5000,
                    name: str = "pong_oracle_eval", verbose: bool = True,
                    edge_offset: float = 0.0, smash: bool = False,
                    smash_trigger_x: float = 0.0,
                    anti_anticipate_w: float = 0.0,
                    pre_position: bool = False,
                    pre_position_y: float = 42.0) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_id = _resolve_env_id("Pong")
    env = gym.make(env_id, render_mode="rgb_array",
                   frameskip=1, repeat_action_probability=0.0,
                   full_action_space=False)
    env = RecordVideo(env, video_folder=str(out_dir),
                      episode_trigger=lambda i: i == 0,
                      name_prefix=name, disable_logger=True)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=FRAME_HW,
                             terminal_on_life_loss=False,
                             grayscale_obs=True, grayscale_newaxis=False,
                             scale_obs=False)
    env = FrameStackObservation(env, stack_size=FRAME_STACK)
    env = _FlatRGBObsWrapper(env)

    ale = get_ale(env)
    obs, _ = env.reset(seed=seed)
    prev_x = prev_y = prev_py = None
    prev_opp_py = None
    ret = 0.0
    n_steps = 0
    action_counts = {ALE_HOLD: 0, ALE_UP: 0, ALE_DOWN: 0}
    t0 = time.time()
    for _ in range(max_steps):
        s = extract_pong(ale)
        if not s.ball_in_play:
            prev_x = prev_y = prev_py = None
            prev_opp_py = None
            action = ALE_HOLD
        else:
            bx84 = _ram_to_84(s.ball_x, *RAM_BALL_X_RANGE)
            by84 = _ram_to_84(s.ball_y, *RAM_BALL_Y_RANGE)
            # Paddle byte = top edge; shift to center for both paddles.
            py84 = _ram_to_84(s.my_paddle_y, *RAM_PADDLE_Y_RANGE) + PADDLE_HEIGHT_84 / 2.0
            opp_py84 = _ram_to_84(s.opp_paddle_y, *RAM_PADDLE_Y_RANGE) + PADDLE_HEIGHT_84 / 2.0
            if prev_x is None:
                action = ALE_HOLD
            else:
                bdx = (bx84 - prev_x) / ALE_FRAME_SKIP
                bdy = (by84 - prev_y) / ALE_FRAME_SKIP
                paddle_dy = (py84 - prev_py) / ALE_FRAME_SKIP if prev_py is not None else 0.0
                opp_dy = (opp_py84 - prev_opp_py) / ALE_FRAME_SKIP if prev_opp_py is not None else 0.0
                pred_y = predict_ball_y_at_impact(bx84, by84, bdx, bdy)
                eff_py = py84 + paddle_dy * PADDLE_LOOKAHEAD
                # ANTI-ANTICIPATE: factor opp's velocity into the deflection
                # direction. opp moving DOWN → aim ball UP (away from where
                # opp is heading), and vice versa. Reduces to position-only
                # when anti_anticipate_w == 0.
                opp_signal = (opp_py84 - py84) + anti_anticipate_w * opp_dy
                # Edge-hit offset: deflect ball away from opponent's side.
                if edge_offset != 0.0 and abs(opp_signal) > 1e-3:
                    target_y = pred_y + (1.0 if opp_signal > 0 else -1.0) * edge_offset
                else:
                    target_y = pred_y
                if bdx <= 0:
                    if pre_position:
                        # PRE-POSITION: while ball recedes, drift toward
                        # mid-screen (or configured anchor) to minimize
                        # worst-case distance to next pred_y.
                        if py84 < pre_position_y - PONG_ACTION_DEAD_ZONE:
                            action = ALE_DOWN
                        elif py84 > pre_position_y + PONG_ACTION_DEAD_ZONE:
                            action = ALE_UP
                        else:
                            action = ALE_HOLD
                    else:
                        action = ALE_HOLD
                else:
                    # PHASE-LOCKED SMASH. During approach (bx84 below
                    # trigger), track normally — HOLD inside DZ. During
                    # impact phase (ball close to paddle column), commit
                    # to deflection-direction motion regardless of DZ;
                    # this prevents the SMASH-vs-corrective oscillation
                    # that washes out paddle_dy at impact.
                    in_impact_phase = (smash and bx84 >= smash_trigger_x
                                       and edge_offset != 0.0
                                       and abs(opp_signal) > 1e-3)
                    if in_impact_phase:
                        action = ALE_DOWN if opp_signal > 0 else ALE_UP
                    elif target_y < eff_py - PONG_ACTION_DEAD_ZONE:
                        action = ALE_UP
                    elif target_y > eff_py + PONG_ACTION_DEAD_ZONE:
                        action = ALE_DOWN
                    else:
                        action = ALE_HOLD
            prev_x, prev_y, prev_py = bx84, by84, py84
            prev_opp_py = opp_py84
        action_counts[action] = action_counts.get(action, 0) + 1
        obs, r, term, trunc, _ = env.step(action)
        ret += float(r)
        n_steps += 1
        if term or trunc:
            break
    env.close()
    elapsed = time.time() - t0
    video_path = out_dir / f"{name}-episode-0.mp4"
    if verbose:
        print(f"[oracle] return={ret:+.1f}  length={n_steps}  ({elapsed:.1f}s)")
        print(f"  action counts: {action_counts}")
        print(f"  video: {video_path}")
    return {"return": ret, "length": n_steps, "action_counts": action_counts,
            "video_path": str(video_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=str,
                    default=str(PROJ / "outputs" / "atari_primitive_donors"
                                / "pong_eval"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--edge-offset", type=float, default=0.0,
                    help="Edge-hit offset in 84-coords. 0 = catching baseline. "
                         "~3.0 aims for paddle edge (paddle half-height ≈ 4.0).")
    ap.add_argument("--smash", action="store_true",
                    help="Maintain paddle motion in deflection direction at "
                         "impact instead of HOLDing once aligned.")
    ap.add_argument("--smash-trigger-x", type=float, default=0.0,
                    help="Phase-lock SMASH: only fire when bx84 >= this. "
                         "Paddle is at x84≈84; trigger_x=70 ≈ last 14 frames "
                         "before impact, =78 ≈ last 6 frames. 0 = always-on "
                         "(legacy oscillating behavior).")
    ap.add_argument("--anti-anticipate-w", type=float, default=0.0,
                    help="Weight on opp_dy in deflection-sign decision. "
                         "0 = position only; ~1-3 = anticipate opp's motion.")
    ap.add_argument("--pre-position", action="store_true",
                    help="Drift toward mid-screen while ball recedes.")
    args = ap.parse_args()
    res = evaluate_oracle(out_dir=Path(args.out_dir), seed=args.seed,
                          max_steps=args.max_steps,
                          edge_offset=args.edge_offset,
                          smash=args.smash,
                          smash_trigger_x=args.smash_trigger_x,
                          anti_anticipate_w=args.anti_anticipate_w,
                          pre_position=args.pre_position)
    flags = []
    if args.smash:
        if args.smash_trigger_x > 0:
            flags.append(f"smash@x≥{args.smash_trigger_x}")
        else:
            flags.append("smash")
    if args.anti_anticipate_w: flags.append(f"anti_w={args.anti_anticipate_w}")
    if args.pre_position: flags.append("preposition")
    flag_str = " ".join(flags) if flags else "none"
    print(f"\nfinal: return={res['return']:+.1f} length={res['length']} "
          f"edge_offset={args.edge_offset} skills=[{flag_str}]")


if __name__ == "__main__":
    main()
