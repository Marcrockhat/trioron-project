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
def run_arm(agent, label, *, episodes=40, max_steps=300, lr=3e-3, batch=64,
            numa_every=200, obs_seed=0, nonlinear=False):
    """agent: object with .act(w, p) -> action int, and OPTIONAL .learn(p, a, r,
    p2, done, pair_tgt) (a learning policy driver) and .start_episode(). The
    observer is identical regardless of which arm drives. Returns (lengths,
    observer, ledger)."""
    observer = build_observer(obs_seed, nonlinear=nonlinear)
    opt = torch.optim.Adam(observer.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    ledger = NumaLedger()
    g = torch.Generator().manual_seed(obs_seed + 31)
    learn = getattr(agent, "learn", None)
    start_ep = getattr(agent, "start_episode", None)
    lengths = []
    causes = {}
    gstep = 0
    steps = 0

    for ep in range(episodes):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)   # fire_taming seed set
        if start_ep is not None:
            start_ep()
        p = w.percept(); done = False
        while not done:
            a = agent.act(w, p)
            p2, r, done, info = w.step(a)
            pair_tgt = contrast_targets(w)              # predict the NEXT-step pairs
            if learn is not None:                       # learning driver (arm B/C/D)
                learn(p, a, r, p2, float(done), pair_tgt)
            buf.append((p, pair_tgt))
            steps += 1
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
    last30 = statistics.mean(lengths[-30:])      # end-state competence (learning arms)
    # DENSITY is the canonical Numa axis (Rocky, s019): net Numa per 1k env steps,
    # so survival (the gate) is not also baked into the learning number. Raw total
    # is survival-confounded — a longer life feeds the observer more experience.
    density = ledger.net() / (steps / 1000.0) if steps else 0.0
    cause_str = ", ".join(f"{k}:{v}" for k, v in
                          sorted(causes.items(), key=lambda kv: -kv[1]))
    diag = agent.diagnostics() if hasattr(agent, "diagnostics") else ""
    print(f"  {label:>16s}: survival {surv:6.1f} (last30 {last30:5.1f})  "
          f"net-Numa/1k {density:+.3f}  ({ledger.summary()}, {steps} steps)  "
          f"[{cause_str}]{('  ' + diag) if diag else ''}")
    return lengths, observer, ledger, density


# ----------------------------------------------------------------------
# Arm A — Reflex (imitation): the frozen vocabulary organism is the policy.
# ----------------------------------------------------------------------
class FrozenPolicy:
    """Adapter: a frozen act_fn(w, p) as an agent with no .learn (arm A)."""
    def __init__(self, act_fn):
        self._act = act_fn

    def act(self, w, p):
        return self._act(w, p)


def arm_a_agent(*, router_seeds=120, per_prim_cap=1500):
    org, _ = build_vocabulary(full_cov=True, router_seeds=router_seeds,
                              per_prim_cap=per_prim_cap)
    return FrozenPolicy(org.act)     # org.act stateful on its own route_hist


# ----------------------------------------------------------------------
# Arms B & C — a shared TD driver; B and C differ ONLY in the reward source
# (the clean reward-source ablation). One driver substrate carries a VALUE head
# (acts, ε-greedy argmax) and its OWN pair world-model (separate from the neutral
# observer, so Numa stays orthogonal to the driver). The pair head is the
# curiosity machinery for B and a trained passenger for C (matching organism_v2).
#
#   B Wisdom — reward = LEARNING PROGRESS, the per-step drop in the driver
#     world-model's prediction error on the just-visited state:
#         r_int(s_t) = relu( err_before(s_t) - err_after(s_t) )   (Δerror)
#     measured around one world-model update — rewards regions where a step makes
#     the model MORE predictive (reducible surprise), ignoring both mastered
#     (err flat-low) and irreducible-noise (err flat-high) regions (noisy-TV
#     immune). Reward-free AND label-free (never sees world reward r).
#     r_int is NORMALISED by its running scale so it is O(1) and actually steers
#     the value head (raw Δerror ~1e-3 is too weak — s019 it left B ≈ random).
#
#   C Reward — reward = the world reward r (TD value head on extrinsic reward;
#     organism_v2's scaffold). The pair head still trains (fair architecture) but
#     does not feed the reward.
# ----------------------------------------------------------------------
class TDAgent:
    def __init__(self, seed, *, episodes, reward_mode, lr=3e-3, gamma=0.95,
                 batch=64, nonlinear=False, eps_start=1.0, eps_end=0.1):
        assert reward_mode in ("curiosity", "world")
        from experiments.world.tile_world import N_ACTION
        self.N_ACTION = N_ACTION
        self.reward_mode = reward_mode
        torch.manual_seed(seed)
        self.sub = construct(
            base=seeded(PERCEPT_DIM, N_ACTION + N_PAIR, interior_cells=32,
                        nonlinear=nonlinear),
            envelope=Envelope(max_parameter_bytes=500_000),
            dispatch_table=default_dispatch_table(), capacity=2048, sparsity_k=0)
        self.sub.compile(); self.sub.prepare_training()
        self.opt = torch.optim.Adam(self.sub.trainable_tensors(), lr=lr)
        self.buf = deque(maxlen=20000)
        self.g = torch.Generator().manual_seed(seed + 101)
        self.gamma = gamma; self.batch = batch
        self.eps_start = eps_start; self.eps_end = eps_end; self.episodes = episodes
        self._ep = 0; self.eps = eps_start
        self.ri_scale = None                        # running scale of raw Δerror
        self.rew_sum = 0.0; self.rew_n = 0          # diagnostics: TD-reward flow

    def start_episode(self):
        self._ep += 1
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
            max(0.0, 1 - self._ep / (0.7 * self.episodes))

    def act(self, w, p):
        if torch.rand(1, generator=self.g).item() < self.eps:
            return int(torch.randint(0, self.N_ACTION, (1,), generator=self.g))
        with torch.no_grad():
            return int(self.sub(p.unsqueeze(0))[0, :self.N_ACTION].argmax())

    def _wm_err(self, p, pair_tgt):
        with torch.no_grad():
            pred = self.sub(p.unsqueeze(0))[0, self.N_ACTION:]
        return float(((pred - pair_tgt) ** 2).mean())

    def _norm_lp(self, raw):
        """Normalise raw learning-progress by its running scale → O(1) reward that
        steers (above-baseline LP > 1, below < 1). EMA of the magnitude."""
        if self.ri_scale is None:
            self.ri_scale = raw + 1e-6
        else:
            self.ri_scale = 0.99 * self.ri_scale + 0.01 * raw
        return raw / (self.ri_scale + 1e-8)

    def learn(self, p, a, r_world, p2, done, pair_tgt):
        na = self.N_ACTION
        if self.reward_mode == "curiosity":
            err_before = self._wm_err(p, pair_tgt)
        # one combined update on a minibatch of PAST transitions (each carries its
        # own stored TD reward); world-model MSE + TD on that reward.
        if len(self.buf) >= 2 * self.batch:
            idx = torch.randint(0, len(self.buf), (self.batch,), generator=self.g)
            bs = [self.buf[i] for i in idx]
            bp = torch.stack([b[0] for b in bs]); ba = torch.tensor([b[1] for b in bs])
            brw = torch.tensor([b[2] for b in bs]); bp2 = torch.stack([b[3] for b in bs])
            bd = torch.tensor([b[4] for b in bs]); btgt = torch.stack([b[5] for b in bs])
            out = self.sub(bp)
            q = out[torch.arange(self.batch), ba]
            with torch.no_grad():
                q_next = self.sub(bp2)[:, :na].max(dim=1).values
                target = brw + self.gamma * q_next * (1 - bd)
            loss_v = torch.nn.functional.mse_loss(q, target)
            loss_wm = torch.nn.functional.mse_loss(out[:, na:], btgt)
            self.opt.zero_grad(); (loss_v + loss_wm).backward()
            torch.nn.utils.clip_grad_norm_(self.sub.trainable_tensors(), 1.0)
            self.sub.zero_dormant_grads(); self.opt.step()
        if self.reward_mode == "curiosity":
            err_after = self._wm_err(p, pair_tgt)
            reward = self._norm_lp(max(0.0, err_before - err_after))    # learning progress
        else:
            reward = float(r_world)                                    # extrinsic world reward
        self.rew_sum += reward; self.rew_n += 1
        self.buf.append((p, a, reward, p2, done, pair_tgt))

    def diagnostics(self):
        mean_r = self.rew_sum / max(self.rew_n, 1)
        tag = "r_int" if self.reward_mode == "curiosity" else "r_world"
        extra = (f" rawLP~{self.ri_scale:.2e}" if self.reward_mode == "curiosity"
                 and self.ri_scale is not None else "")
        return f"{tag} mean={mean_r:.3f}{extra}"


def arm_b_agent(seed, *, episodes, nonlinear=False):
    return TDAgent(seed, episodes=episodes, reward_mode="curiosity", nonlinear=nonlinear)


def arm_c_agent(seed, *, episodes, nonlinear=False):
    return TDAgent(seed, episodes=episodes, reward_mode="world", nonlinear=nonlinear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", choices=["A", "B", "C", "AB", "ABC"],
                    help="AB/ABC = run arms at matched episodes; D lands later")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--router-seeds", type=int, default=120)
    ap.add_argument("--per-prim-cap", type=int, default=1500)
    ap.add_argument("--obs-seed", type=int, default=0)
    ap.add_argument("--driver-seed", type=int, default=0,
                    help="seed for the arm-B/C/D learning driver")
    ap.add_argument("--nonlinear", action="store_true",
                    help="nonlinear observer + driver substrate")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.router_seeds, args.per_prim_cap = 30, 300
        if args.episodes == 40:                  # leave an explicit --episodes alone
            args.episodes = 8 if args.arm == "A" else 60

    def _run_a():
        agent = arm_a_agent(router_seeds=args.router_seeds, per_prim_cap=args.per_prim_cap)
        return run_arm(agent, "A:Reflex", episodes=args.episodes,
                       obs_seed=args.obs_seed, nonlinear=args.nonlinear)

    def _run_b():
        agent = arm_b_agent(args.driver_seed, episodes=args.episodes,
                            nonlinear=args.nonlinear)
        return run_arm(agent, "B:Wisdom", episodes=args.episodes,
                       obs_seed=args.obs_seed, nonlinear=args.nonlinear)

    def _run_c():
        agent = arm_c_agent(args.driver_seed, episodes=args.episodes,
                            nonlinear=args.nonlinear)
        return run_arm(agent, "C:Reward", episodes=args.episodes,
                       obs_seed=args.obs_seed, nonlinear=args.nonlinear)

    print("=== REFLEX-vs-WISDOM ARC — common Numa harness ===")
    print("axis: net Numa DENSITY (per 1k env steps); survival = gate (separate).")
    print(f"observer: 5-d pair head, identical across arms (obs_seed={args.obs_seed}, "
          f"nonlinear={args.nonlinear})\n")

    if args.arm == "A":
        print(f"Arm A (Reflex): vocabulary organism acts (episodes={args.episodes}).")
        _run_a()
        print("\n  Expected: high survival; net-Numa density the bar for B/C.")
    elif args.arm == "B":
        print(f"Arm B (Wisdom): learning-progress driver, reward-free "
              f"(episodes={args.episodes}, driver_seed={args.driver_seed}).")
        _run_b()
        print("\n  Expected: lower survival (curiosity doesn't eat); the Numa-density "
              "question.")
    elif args.arm == "C":
        print(f"Arm C (Reward): TD on world reward r (episodes={args.episodes}, "
              f"driver_seed={args.driver_seed}).")
        _run_c()
        print("\n  Expected: reward-acts (survival up vs B); moderate Numa density.")
    elif args.arm in ("AB", "ABC"):
        arms = {"AB": "A (Reflex) vs B (Wisdom)",
                "ABC": "A (Reflex) vs B (Wisdom) vs C (Reward)"}[args.arm]
        print(f"Comparison: {arms} at matched episodes={args.episodes}, "
              f"obs_seed={args.obs_seed}.\n")
        _run_a(); _run_b()
        if args.arm == "ABC":
            _run_c()
        print("\n  Read net-Numa/1k as the axis (survival-decoupled); survival is the "
              "gate.\n  Arc hypothesis: no single arm wins BOTH — the opening D "
              "(reflex->wisdom) must fill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
