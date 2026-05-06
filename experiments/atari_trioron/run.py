"""Four-arm self-imitation runner.

Arms (Rocky's spec):
  arm1: Breakout-only. Cold-start trioron, self-imitation on Breakout.
  arm2: Pong → Breakout. Train Pong first, api.extend into Breakout
        self-imitation. Continual-learning path.
  arm3: Pong-only, eval on Breakout. Trains Pong, no Breakout training.
        Eval plays Breakout zero-shot through the (single) Pong branch.
  arm4: Pong + Breakout independently → graft. Two donors trained
        cold-start (re-uses arm1.final + arm3.final, no extra training),
        then api.absorb composes them into one multi-branch organism.
        The manifold archive routes per-frame to the right branch.

Usage:
    python3 experiments/atari_trioron/run.py --arm arm1 --train-iters 8
    python3 experiments/atari_trioron/run.py --arm arm2 --train-iters 8
    python3 experiments/atari_trioron/run.py --arm arm3 --train-iters 8
    python3 experiments/atari_trioron/run.py --arm arm4
        # arm4 depends on arm1 and arm3 having been run first.

The --eval-only flag skips training and just renders an MP4 from an
existing donor (for re-rendering after a tweak).

Output layout:
    outputs/atari_trioron/{arm}/donor_iter{i}.pt    (per-iter donors)
    outputs/atari_trioron/{arm}/final.pt            (symlink to last)
    outputs/atari_trioron/{arm}/eval-episode-0.mp4  (render)
    outputs/atari_trioron/{arm}/log.json            (training metrics)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from experiments.atari_trioron.train import self_imitation_train  # noqa: E402
from experiments.atari_trioron.eval_render import (              # noqa: E402
    evaluate_and_record,
)
from trioron.api import absorb                                   # noqa: E402


OUT_ROOT = PROJ / "outputs" / "atari_trioron"


def _arm_train_game(arm: str) -> str:
    return {"arm1": "Breakout", "arm2_pong": "Pong",
            "arm2_breakout": "Breakout", "arm3": "Pong"}[arm]


def _arm_eval_game(arm: str) -> str:
    """All three arms display Breakout — that's Rocky's spec."""
    return "Breakout"


def _resolve_eps_schedule(n_iters: int) -> list:
    """Linear decay from 0.8 → 0.1. Early iters need lots of
    exploration since the policy is essentially random; later iters
    should be near-greedy to exploit what was learned."""
    if n_iters == 1:
        return [0.5]
    return [0.8 - (0.7 * i / (n_iters - 1)) for i in range(n_iters)]


def _save_log(out_dir: Path, result, train_game: str, arm: str,
              wallclock_s: float, eval_result: dict):
    log_path = out_dir / "log.json"
    payload = {
        "arm": arm,
        "train_game": train_game,
        "eval_game": "Breakout",
        "wallclock_s": wallclock_s,
        "iterations": [asdict(it) for it in result.iterations],
        "final_donor": str(result.final_donor),
        "eval": eval_result,
    }
    log_path.write_text(json.dumps(payload, indent=2))
    print(f"[run] wrote {log_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True,
                   choices=["arm1", "arm2", "arm3", "arm4"])
    p.add_argument("--train-iters", type=int, default=8)
    p.add_argument("--episodes-per-iter", type=int, default=16)
    p.add_argument("--epochs-per-task", type=int, default=4)
    p.add_argument("--cap-bytes", type=int, default=32_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; render from existing final.pt")
    p.add_argument("--eval-eps", type=float, default=0.05,
                   help="ε for stochasticity in the rendered episode")
    p.add_argument("--eval-max-steps", type=int, default=10_000)
    args = p.parse_args()

    arm = args.arm
    out_dir = OUT_ROOT / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_game = _arm_eval_game(arm)

    if args.eval_only:
        donor = out_dir / "final.pt"
        if not donor.exists():
            print(f"[run] --eval-only but {donor} does not exist", file=sys.stderr)
            sys.exit(2)
        evaluate_and_record(
            organism_path=donor, game=eval_game,
            out_dir=out_dir, name="eval",
            seed=args.seed, eps=args.eval_eps,
            max_steps=args.eval_max_steps,
        )
        return 0

    t0 = time.time()
    eps_schedule = _resolve_eps_schedule(args.train_iters)

    if arm == "arm1":
        # Breakout-only, cold start.
        result = self_imitation_train(
            game="Breakout", out_dir=out_dir,
            n_iterations=args.train_iters,
            n_episodes_per_iter=args.episodes_per_iter,
            eps_schedule=eps_schedule,
            epochs_per_task=args.epochs_per_task,
            cap_bytes=args.cap_bytes,
            seed=args.seed,
        )
        train_game = "Breakout"

    elif arm == "arm2":
        # Pong first, then Breakout via api.extend on iter 0.
        # Phase A: Pong from cold start.
        pong_result = self_imitation_train(
            game="Pong", out_dir=out_dir / "pong_phase",
            n_iterations=args.train_iters,
            n_episodes_per_iter=args.episodes_per_iter,
            eps_schedule=eps_schedule,
            epochs_per_task=args.epochs_per_task,
            cap_bytes=args.cap_bytes,
            seed=args.seed,
        )
        # Phase B: Breakout extension, starting from Pong donor.
        breakout_result = self_imitation_train(
            game="Breakout", out_dir=out_dir / "breakout_phase",
            n_iterations=args.train_iters,
            n_episodes_per_iter=args.episodes_per_iter,
            eps_schedule=eps_schedule,
            epochs_per_task=args.epochs_per_task,
            cap_bytes=args.cap_bytes * 2,
            seed=args.seed + 1,
            initial_donor=pong_result.final_donor,
        )
        result = breakout_result
        train_game = "Pong→Breakout"
        # Persist combined log fields by stuffing both phases' iters.
        result.iterations = (pong_result.iterations
                             + breakout_result.iterations)

    elif arm == "arm3":
        # Pong only — never sees Breakout in training.
        result = self_imitation_train(
            game="Pong", out_dir=out_dir,
            n_iterations=args.train_iters,
            n_episodes_per_iter=args.episodes_per_iter,
            eps_schedule=eps_schedule,
            epochs_per_task=args.epochs_per_task,
            cap_bytes=args.cap_bytes,
            seed=args.seed,
        )
        train_game = "Pong"

    elif arm == "arm4":
        # Independent training + graft. Reuses arm1.final (Breakout
        # cold-start) and arm3.final (Pong cold-start). The two
        # donors must share the L0 seed — both arms default to
        # seed=42, so this falls into the canonical shared-L0 path.
        arm1_donor = OUT_ROOT / "arm1" / "final.pt"
        arm3_donor = OUT_ROOT / "arm3" / "final.pt"
        for needed in (arm1_donor, arm3_donor):
            if not needed.exists():
                print(f"[run] arm4 needs {needed} — run "
                      f"{'arm1' if 'arm1' in str(needed) else 'arm3'} first",
                      file=sys.stderr)
                sys.exit(2)
        organism_path = out_dir / "organism.pt"
        absorb(
            donor_paths=[arm3_donor.resolve(), arm1_donor.resolve()],
            out_path=organism_path,
        )
        # Mirror the train.py output contract so eval reuses one path.
        from dataclasses import dataclass

        @dataclass
        class _Stub:
            final_donor: Path
            iterations: list
        result = _Stub(final_donor=organism_path, iterations=[])
        train_game = "Pong⊕Breakout (grafted)"

    else:
        raise ValueError(arm)

    # Symlink final.pt for stable eval path.
    final_link = out_dir / "final.pt"
    if final_link.exists() or final_link.is_symlink():
        final_link.unlink()
    final_link.symlink_to(Path(result.final_donor).resolve())

    wallclock_s = time.time() - t0
    print(f"\n[run] {arm} training done in {wallclock_s/60:.1f} min")

    print(f"\n[run] === eval on {eval_game} ===")
    eval_result = evaluate_and_record(
        organism_path=final_link, game=eval_game,
        out_dir=out_dir, name="eval",
        seed=args.seed, eps=args.eval_eps,
        max_steps=args.eval_max_steps,
    )
    _save_log(out_dir, result, train_game, arm, wallclock_s, eval_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
