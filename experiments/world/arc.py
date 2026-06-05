"""Reflex-vs-wisdom arc — the common Numa harness + Arm A (Reflex).

The skill-acquisition head-to-head designed in session 018. All arms are scored
on ONE axis — **net Numa** (consolidated, dream-validated world-model learning;
numa.py NumaLedger) — with **survival as the precondition gate**. The arms differ
ONLY in the POLICY DRIVER; the Numa measurement is identical across all four:

  A Reflex    — imitation: the vocabulary organism (5 donors + manifold router) acts.
  B Wisdom    — curiosity / learning-progress (NEXT build).
  C Reward    — TD value head (organism_v2 scaffold).
  D Synthesis — reflex -> consolidate+lock -> grow wisdom on top (after A/B/C).

Net Numa is a MEASUREMENT axis (world-model consolidation), ORTHOGONAL to the
policy driver. So we score every arm the same way: an EXTERNAL **observer** — a
small trioron substrate with only the 5-d contrastive-pair head — rides along the
arm's life. It never acts; it learns to predict the next-step pair labels from the
states the arm's policy visits. The held-out (dream) re-test turns those loss-drops
into Numa (persists) or Mima (reverts). This resolves the s018 wrinkle: arm A is a
composite with nowhere to hang a pair head, so we hang it OUTSIDE, on the observer,
and let it measure the learnability of arm A's experience stream — uniformly with
the other arms.

Hypothesis (s018): A high survival / low Numa (the reflex keeps the world in a
narrow, already-predictable homeostatic band — little reducible surprise to consume).

Usage:
    python3 -m experiments.world.arc --smoke
    python3 -m experiments.world.arc --arm A --episodes 40
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from trioron.core import Envelope, construct
from trioron.bases import seeded
from trioron.phenotype import default_dispatch_table
# NB: importing vocabulary pulls in fire_taming, which sets the TAMED temperature
# physics (WARM_RATE=0.04, TEMP_LOW=0.02, TEMP_HIGH=0.99) on the TileWorld class.
# Every world created below therefore matches the regime the donors were built in.
from experiments.world.vocabulary import build_vocabulary
from experiments.world.tile_world import TileWorld
from experiments.world.numa import contrast_targets, NumaLedger, N_PAIR
from experiments.world.fire_taming import _cause

PERCEPT_DIM = 77


# ----------------------------------------------------------------------
# The common observer — a small substrate with ONLY the 5-d pair head.
# Identical build across every arm; only the policy that feeds it differs.
# ----------------------------------------------------------------------
def build_observer(seed, *, nonlinear=False):
    torch.manual_seed(seed)
    sub = construct(base=seeded(PERCEPT_DIM, N_PAIR, interior_cells=32, nonlinear=nonlinear),
                    envelope=Envelope(max_parameter_bytes=500_000),
                    dispatch_table=default_dispatch_table(), capacity=2048, sparsity_k=0)
    sub.compile(); sub.prepare_training()
    return sub


def _pair_loss(observer, batch):
    """Per-pair MSE [N_PAIR] on a list of (percept, pair_target) samples."""
    bp = torch.stack([b[0] for b in batch])
    tgt = torch.stack([b[1] for b in batch])
    with torch.no_grad():
        pred = observer(bp)
    return ((pred - tgt) ** 2).mean(dim=0)


# ----------------------------------------------------------------------
# The harness — roll out a policy, train the observer on what it visits,
# accrue Numa. Survival = episode lengths (same seed set as fire_taming.evaluate).
# ----------------------------------------------------------------------
def run_arm(policy, label, *, episodes=40, max_steps=300, lr=3e-3, batch=64,
            numa_every=200, obs_seed=0, nonlinear=False):
    """policy: act_fn(w, p) -> action int. Returns (lengths, observer, ledger)."""
    observer = build_observer(obs_seed, nonlinear=nonlinear)
    opt = torch.optim.Adam(observer.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    ledger = NumaLedger()
    g = torch.Generator().manual_seed(obs_seed + 31)
    lengths = []
    causes = {}
    gstep = 0

    for ep in range(episodes):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)   # fire_taming seed set
        p = w.percept(); done = False
        while not done:
            a = policy(w, p)
            p2, r, done, info = w.step(a)
            pair_tgt = contrast_targets(w)              # predict the NEXT-step pairs
            buf.append((p, pair_tgt))
            p = p2
            if len(buf) >= 2 * batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[i] for i in idx]
                bp = torch.stack([b[0] for b in bs])
                btgt = torch.stack([b[1] for b in bs])
                out = observer(bp)
                loss = torch.nn.functional.mse_loss(out, btgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(observer.trainable_tensors(), 1.0)
                observer.zero_dormant_grads(); opt.step()
                gstep += 1
                # Numa/Mima checkpoint: train batch vs a held-out "dream" batch.
                if gstep % numa_every == 0 and len(buf) >= 4 * batch:
                    didx = torch.randint(0, len(buf), (batch,), generator=g)
                    dream = [buf[i] for i in didx]
                    ledger.checkpoint(_pair_loss(observer, bs), _pair_loss(observer, dream))
        lengths.append(info["t"])
        causes[_cause(w)] = causes.get(_cause(w), 0) + 1

    surv = statistics.mean(lengths)
    cause_str = ", ".join(f"{k}:{v}" for k, v in
                          sorted(causes.items(), key=lambda kv: -kv[1]))
    print(f"  {label:>16s}: survival {surv:6.1f}  {ledger.summary()}  [{cause_str}]")
    return lengths, observer, ledger


# ----------------------------------------------------------------------
# Arm A — Reflex (imitation): the frozen vocabulary organism is the policy.
# ----------------------------------------------------------------------
def arm_a_policy(*, router_seeds=120, per_prim_cap=1500):
    org, _ = build_vocabulary(full_cov=True, router_seeds=router_seeds,
                              per_prim_cap=per_prim_cap)
    return org.act       # act(w, p) -> action int (stateful route_hist on org)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", choices=["A"],
                    help="B/C/D land in later sessions")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--router-seeds", type=int, default=120)
    ap.add_argument("--per-prim-cap", type=int, default=1500)
    ap.add_argument("--obs-seed", type=int, default=0)
    ap.add_argument("--nonlinear", action="store_true",
                    help="nonlinear observer substrate")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.episodes, args.router_seeds, args.per_prim_cap = 8, 30, 300

    print("=== REFLEX-vs-WISDOM ARC — common Numa harness ===")
    print("axis: net Numa (dream-validated world-model learning); survival = gate.")
    print(f"observer: 5-d pair head, identical across arms (obs_seed={args.obs_seed}, "
          f"nonlinear={args.nonlinear})\n")

    if args.arm == "A":
        print(f"Arm A (Reflex): vocabulary organism acts; observer learns its stream "
              f"(episodes={args.episodes}).")
        policy = arm_a_policy(router_seeds=args.router_seeds,
                              per_prim_cap=args.per_prim_cap)
        run_arm(policy, "A:Reflex", episodes=args.episodes, obs_seed=args.obs_seed,
                nonlinear=args.nonlinear)
        print("\n  Expected signature: high survival, LOW net Numa (the reflex keeps "
              "the world\n  in a predictable homeostatic band — little reducible "
              "surprise to consume).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
