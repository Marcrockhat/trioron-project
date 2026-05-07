"""Skill-curriculum runner for the four-arm Atari setup.

Arms (Rocky's spec):
  arm1: Breakout-only.    build skill donor on Breakout rollouts.
  arm2: Pong → extend(Breakout). Pong donor + api.extend on Breakout.
  arm3: Pong-only.        skill donor on Pong rollouts.
  arm4: Pong ⊕ Breakout (graft). absorb arm3.final + arm1.final.
        Disjoint class IDs (Pong: {10,11,12}, Breakout: {20-23})
        make absorb work cleanly.

All four arms produce an MP4 of the trained organism playing
**Breakout** (Rocky's display target).

Usage:
  python3 experiments/atari_trioron/run_skill.py --arm arm1 [--n-episodes 16]
  python3 experiments/atari_trioron/run_skill.py --arm all
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from experiments.atari_trioron.skill_curriculum import (   # noqa: E402
    build_skill_donor, extend_skill_donor, absorb_skill_donors,
)
from experiments.atari_trioron.skill_eval import (         # noqa: E402
    evaluate_skill_organism,
)


OUT_ROOT = PROJ / "outputs" / "atari_trioron_skill"
EVAL_GAME = "Breakout"           # all four arms display Breakout


def run_arm1(args):
    out_dir = OUT_ROOT / "arm1"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "final.pt"
    t0 = time.time()
    build_skill_donor(
        game="Breakout", out_path=final,
        n_episodes=args.n_episodes,
        eps_explore=args.eps_explore,
        epochs_per_task=args.epochs_per_task,
        cap_bytes=args.cap_bytes, seed=args.seed,
    )
    train_s = time.time() - t0
    print(f"[arm1] training done in {train_s/60:.1f} min")
    return final, train_s


def run_arm3(args):
    out_dir = OUT_ROOT / "arm3"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "final.pt"
    t0 = time.time()
    build_skill_donor(
        game="Pong", out_path=final,
        n_episodes=args.n_episodes,
        eps_explore=args.eps_explore,
        epochs_per_task=args.epochs_per_task,
        cap_bytes=args.cap_bytes, seed=args.seed,
    )
    train_s = time.time() - t0
    print(f"[arm3] training done in {train_s/60:.1f} min")
    return final, train_s


def run_arm2(args, pong_donor: Path):
    out_dir = OUT_ROOT / "arm2"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "final.pt"
    t0 = time.time()
    extend_skill_donor(
        base_donor=pong_donor, new_game="Breakout",
        out_path=final,
        n_episodes=args.n_episodes,
        eps_explore=args.eps_explore,
        epochs_per_task=args.epochs_per_task,
        extension_cap_bytes=args.cap_bytes * 2, seed=args.seed,
    )
    train_s = time.time() - t0
    print(f"[arm2] extension done in {train_s/60:.1f} min")
    return final, train_s


def run_arm4(args, pong_donor: Path, breakout_donor: Path):
    out_dir = OUT_ROOT / "arm4"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "final.pt"
    t0 = time.time()
    absorb_skill_donors(
        pong_donor=pong_donor, breakout_donor=breakout_donor,
        out_path=final,
    )
    train_s = time.time() - t0
    print(f"[arm4] graft done in {train_s:.1f}s")
    return final, train_s


def evaluate_arm(arm: str, donor: Path, args, train_s: float):
    out_dir = OUT_ROOT / arm
    print(f"\n=== {arm} eval on {EVAL_GAME} ===")
    res = evaluate_skill_organism(
        organism_path=donor, game=EVAL_GAME,
        out_dir=out_dir, name="eval",
        seed=args.seed, eps=args.eval_eps,
        max_steps=args.eval_max_steps,
    )
    log = {
        "arm": arm,
        "eval_game": EVAL_GAME,
        "training_wallclock_s": train_s,
        "eval": res,
    }
    (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True,
                   choices=["arm1", "arm2", "arm3", "arm4", "all"])
    p.add_argument("--n-episodes", type=int, default=16)
    p.add_argument("--eps-explore", type=float, default=0.05,
                   help="ε on the skill labeler at data collection time")
    p.add_argument("--epochs-per-task", type=int, default=8)
    p.add_argument("--cap-bytes", type=int, default=64_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-eps", type=float, default=0.0)
    p.add_argument("--eval-max-steps", type=int, default=10_000)
    args = p.parse_args()

    if args.arm == "all":
        # Order: arm3 (Pong donor first) → arm1 (Breakout donor) →
        # arm2 (extend Pong with Breakout) → arm4 (graft arm3+arm1).
        donor3, t3 = run_arm3(args)
        evaluate_arm("arm3", donor3, args, t3)
        donor1, t1 = run_arm1(args)
        evaluate_arm("arm1", donor1, args, t1)
        donor2, t2 = run_arm2(args, donor3)
        evaluate_arm("arm2", donor2, args, t2)
        donor4, t4 = run_arm4(args, donor3, donor1)
        evaluate_arm("arm4", donor4, args, t4)
    elif args.arm == "arm1":
        d, t = run_arm1(args)
        evaluate_arm("arm1", d, args, t)
    elif args.arm == "arm3":
        d, t = run_arm3(args)
        evaluate_arm("arm3", d, args, t)
    elif args.arm == "arm2":
        pong_donor = OUT_ROOT / "arm3" / "final.pt"
        if not pong_donor.exists():
            print("[run_skill] arm2 needs arm3 first", file=sys.stderr)
            return 2
        d, t = run_arm2(args, pong_donor)
        evaluate_arm("arm2", d, args, t)
    elif args.arm == "arm4":
        pong_donor = OUT_ROOT / "arm3" / "final.pt"
        breakout_donor = OUT_ROOT / "arm1" / "final.pt"
        for need in (pong_donor, breakout_donor):
            if not need.exists():
                print(f"[run_skill] arm4 needs {need}", file=sys.stderr)
                return 2
        d, t = run_arm4(args, pong_donor, breakout_donor)
        evaluate_arm("arm4", d, args, t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
