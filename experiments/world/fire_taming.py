"""Fire-taming by apprenticeship — the real test of mirror cells (2026-06-02).

The solo organism learns to FEAR fire (pyrophobia is emergent: fire burns, so
the value learner avoids it → it never warms → freezes to death; see
probe_temp_conflict.py). A lone learner can't discover "stay NEAR the fire, not
ON it, and leave before you overheat" — every attempt is punished by the burn
before the trick is found. This is the explore-vs-exploit case where apprenticing
should win big: independent search is COSTLY, so copying a competent master wins.

Setup:
  1. A hand-coded FIRE ORACLE that has the wisdom — warm at the fire when cold,
     step away before overheating, forage otherwise. (The reactive baseline lacks
     the "leave" rule and dies 30/30 of overheat; the oracle fixes exactly that.)
  2. A mirror-apprentice that learns from the oracle via the master-avatar channel
     (avatar CE + internalization CE, credit gated to mirror cells).
  3. Compare apprentice vs learn-from-scratch on: survival, COLD-death rate, and
     fire-USE (does it approach fire when cold instead of freezing?).

Win = the apprentice overcomes pyrophobia (cold-deaths drop, fire-use rises) that
the scratch learner cannot.
"""
from __future__ import annotations

import sys
import statistics
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.tile_world import (
    TileWorld, N_ACTION, ACTIONS, FIRE, WATER, FOOD, BERRY, _DXY,
)
from experiments.world.mirror_cells import (
    build_mirror, _solo, train_mirror_student, eval_avatar_greedy,
    keep_only_mirror_grads, obs_onehot,
)
from experiments.world.render_organism import train_solo
from collections import deque

# Soften the temperature physics so fire is TAMABLE (Rocky 2026-06-02): the
# canonical +0.15 warm-rate vs ~0.01 cooling is uncontrollable even for a
# perfect-info oracle (overheats 37/40). Gentler warming + a wider lethal band
# gives a controllable equilibrium an oracle can demonstrate and an apprentice
# can learn. Set on the class so worlds created INSIDE train_solo /
# train_mirror_student inherit it without kwarg threading.
TileWorld.WARM_RATE = 0.04
TileWorld.TEMP_LOW = 0.02
TileWorld.TEMP_HIGH = 0.99

_OPP_ACT = {ACTIONS.index("N"): "S", ACTIONS.index("S"): "N",
            ACTIONS.index("E"): "W", ACTIONS.index("W"): "E"}


# ----------------------------------------------------------------------
# The fire oracle — the master with the wisdom
# ----------------------------------------------------------------------
# Default-off switch (session 015): when True, the masters' random-explore fallback
# becomes a DETERMINISTIC, learnable default (top up water, else rest). Used to confirm
# that the imitation/survival ceiling is the TEACHER'S stochasticity — not the substrate
# or the percept (both cleared by imitation_ceiling.py). consolidate_base unaffected.
EXPLORE_DETERMINISTIC = False


def _explore(w):
    if EXPLORE_DETERMINISTIC:
        t = _toward(w, WATER)
        return t if t is not None else ACTIONS.index("rest")
    return int(torch.randint(0, 4, (1,)))


def _near_fire(w):
    s = w.size
    return any(int(w.grid[(w.py + dy) % s, (w.px + dx) % s]) == FIRE
               for dy in (-1, 0, 1) for dx in (-1, 0, 1))


def _toward(w, tile):
    d = w._scent(tile)
    if float(d.abs().sum()) == 0:
        return None
    if abs(float(d[0])) >= abs(float(d[1])):
        return ACTIONS.index("E" if d[0] > 0 else "W")
    return ACTIONS.index("S" if d[1] > 0 else "N")


def _away_from_fire(w):
    t = _toward(w, FIRE)
    if t is None:
        return int(torch.randint(0, 4, (1,)))
    return ACTIONS.index(_OPP_ACT[t])


def fire_oracle(w):
    """The wisdom: TAP the stove — warming is violent (+0.15/step), so soak only
    when very cold and step away the instant you're no longer freezing; never
    stand on the flame. This holds temp ~0.5 without freezing OR overheating."""
    here = int(w.grid[w.py, w.px])
    # --- temperature management (the skill the solo learner can't find) ---
    if here == FIRE:                         # never stand ON the flame (it burns)
        return _away_from_fire(w)
    if _near_fire(w):
        # in the warmth: soak one step only if still freezing, else leave NOW
        if w.temp < 0.3:
            return ACTIONS.index("rest")
        return _away_from_fire(w)
    if w.temp < 0.5:                         # cool and away from fire → go warm up,
        if w.thirst > 0.25 and w.energy > 0.25:   # unless something is dire
            t = _toward(w, FIRE)
            if t is not None:
                return t
    # --- foraging ---
    if here == WATER and w.thirst < 0.8:
        return ACTIONS.index("consume")
    if here in (FOOD, BERRY) and w.energy < 0.8:
        return ACTIONS.index("consume")
    if w.thirst <= w.energy and w.thirst < 0.6:
        t = _toward(w, WATER)
        if t is not None:
            return t
    if w.energy < 0.6:
        t = _toward(w, FOOD)
        if t is not None:
            return t
    return _explore(w)                        # random (default) or deterministic fallback


# ----------------------------------------------------------------------
# Demos + instrumented evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_oracle_demos(n_ep, seed, max_steps=300):
    demos = []
    for ep in range(n_ep):
        w = TileWorld(seed=seed * 13000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            a = fire_oracle(w)
            demos.append((p, a))
            p, _, done, _ = w.step(a)
    return demos


def _cause(w):
    if w.alive:
        return "timeout"
    if w.temp <= 0.05:
        return "cold"
    if w.temp >= 0.98:
        return "overheat"
    if w.thirst <= 0:
        return "thirst"
    if w.energy <= 0:
        return "energy"
    return "integrity"


def train_dagger_student(seed, episodes, *, gamma=0.95, lr=3e-3, batch=64,
                         imit_w=0.5, max_steps=300, seed_demos=None):
    """DAgger apprentice: the oracle labels the STUDENT's own visited states
    (not just its own trajectory), fixing behavioural-cloning distribution shift.
    Imitation does NOT anneal — the master keeps coaching where the student
    actually goes. Credit still gated to mirror cells (master-avatar mechanism)."""
    sub = build_mirror(seed, n_mirror=8)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    a = sub.arena
    buf = deque(maxlen=20000)
    demos = deque(maxlen=20000)
    if seed_demos:
        demos.extend(seed_demos)
    g = torch.Generator().manual_seed(seed + 31)
    ce = torch.nn.functional.cross_entropy
    for ep in range(episodes):
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            demos.append((p, fire_oracle(w)))    # DAgger: oracle labels THIS state
            if torch.rand(1, generator=g).item() < eps:
                act = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    act = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            p2, r, done, info = w.step(act)
            buf.append((p, act, r, p2, float(done))); p = p2
            if len(buf) >= batch and len(demos) >= batch:
                idx = torch.randint(0, len(buf), (batch,), generator=g)
                bs = [buf[i] for i in idx]
                bp = _solo(torch.stack([b[0] for b in bs]))
                ba = torch.tensor([b[1] for b in bs])
                br = torch.tensor([b[2] for b in bs])
                bp2 = _solo(torch.stack([b[3] for b in bs]))
                bd = torch.tensor([b[4] for b in bs])
                q = sub(bp)[torch.arange(batch), ba]
                with torch.no_grad():
                    tgt = br + gamma * sub(bp2).max(dim=1).values * (1 - bd)
                opt.zero_grad()
                torch.nn.functional.mse_loss(q, tgt).backward()
                td_b = a.bias.grad.clone(); td_w = a.edge_weight.grad.clone()
                didx = torch.randint(0, len(demos), (batch,), generator=g)
                ds = torch.stack([demos[i][0] for i in didx])
                da = torch.tensor([demos[i][1] for i in didx])
                avatar = torch.cat([ds, obs_onehot(da, batch)], dim=1)
                opt.zero_grad()
                (imit_w * (ce(sub(avatar), da) + ce(sub(_solo(ds)), da))).backward()
                keep_only_mirror_grads(sub)
                a.bias.grad.add_(td_b); a.edge_weight.grad.add_(td_w)
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
    return sub


@torch.no_grad()
def eval_avatar_oracle(sub, seed, episodes=30, max_steps=300):
    """The fire-oracle pilots the body through the observation channel."""
    from experiments.world.mirror_cells import obs_onehot
    out = []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 7000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            m = fire_oracle(w)
            x = torch.cat([p.unsqueeze(0), obs_onehot(torch.tensor([m]), 1)], dim=1)
            a = int(sub(x)[0].argmax()); p, _, done, info = w.step(a)
        out.append(info["t"])
    return statistics.mean(out)


@torch.no_grad()
def evaluate(act_fn, label, seeds=40, max_steps=300):
    """Survival + cause-of-death + fire-use (near-fire steps while cold)."""
    surv, near_steps, tot_steps = [], 0, 0
    causes = Counter()
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            a = act_fn(w, p)
            tot_steps += 1
            if _near_fire(w):                # fire occupancy = the fire-USE signal
                near_steps += 1              # (phobic organism ≈ 0; a fire-user > 0)
            p, _, done, info = w.step(a)
        surv.append(info["t"]); causes[_cause(w)] += 1
    fire_use = (100 * near_steps / tot_steps) if tot_steps else 0.0
    print(f"  {label:>22s}: survival {statistics.mean(surv):5.1f}  "
          f"cold-deaths {causes['cold']:2d}/{seeds}  "
          f"fire-occupancy {fire_use:4.1f}%  "
          f"[{', '.join(f'{k}:{v}' for k,v in causes.most_common())}]")
    return statistics.mean(surv), causes, fire_use


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-only", action="store_true")
    ap.add_argument("--dagger", action="store_true",
                    help="DAgger arm (oracle labels student-visited states)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=300)
    args = ap.parse_args()

    print("fire-taming: can apprenticing teach the organism to USE fire?\n")
    print("oracle sanity (the teacher must itself be fire-competent):")
    evaluate(lambda w, p: fire_oracle(w), "fire-oracle")
    if args.oracle_only:
        return 0

    if args.dagger:
        demos = collect_oracle_demos(40, 99)
        print(f"\nDAgger: {len(demos)} seed demos; oracle labels student states "
              f"on the fly (n={args.seeds}, {args.episodes} ep)...\n")
        scr_surv, scr_cold, scr_fire = [], [], []
        dag_surv, dag_cold, dag_fire, dag_av = [], [], [], []
        for seed in range(args.seeds):
            scr = train_solo(seed=seed, episodes=args.episodes)
            s, c, f = evaluate(lambda w, p, m=scr: int(m(_solo(p.unsqueeze(0)))[0].argmax()),
                               f"scratch[seed{seed}]")
            scr_surv.append(s); scr_cold.append(c["cold"]); scr_fire.append(f)
            dag = train_dagger_student(seed, args.episodes, seed_demos=demos)
            s, c, f = evaluate(lambda w, p, m=dag: int(m(_solo(p.unsqueeze(0)))[0].argmax()),
                               f"dagger[seed{seed}]")
            dag_surv.append(s); dag_cold.append(c["cold"]); dag_fire.append(f)
            dag_av.append(eval_avatar_oracle(dag, seed))
        print("\n=== DAgger VERDICT (means) ===")
        print(f"  scratch : survival {statistics.mean(scr_surv):.1f}  "
              f"cold-deaths {statistics.mean(scr_cold):.1f}/40  "
              f"fire-occupancy {statistics.mean(scr_fire):.1f}%")
        print(f"  dagger  : survival {statistics.mean(dag_surv):.1f}  "
              f"cold-deaths {statistics.mean(dag_cold):.1f}/40  "
              f"fire-occupancy {statistics.mean(dag_fire):.1f}%  "
              f"avatar-piloted {statistics.mean(dag_av):.1f}")
        print("  (BC apprentice from run1: cold 24.0/40, fire-occupancy 9.2%)")
        print("  win = dagger fire-occupancy UP and cold-deaths DOWN vs scratch+BC")
        return 0

    demos = collect_oracle_demos(40, 99)
    print(f"\ncollected {len(demos)} oracle demo steps; training students "
          f"(n={args.seeds}, {args.episodes} ep each)...\n")

    scratch_surv, scratch_cold, scratch_fire = [], [], []
    appr_surv, appr_cold, appr_fire, appr_avatar = [], [], [], []
    for seed in range(args.seeds):
        scr = train_solo(seed=seed, episodes=args.episodes)
        s, c, f = evaluate(lambda w, p, m=scr: int(m(_solo(p.unsqueeze(0)))[0].argmax()),
                           f"scratch[seed{seed}]")
        scratch_surv.append(s); scratch_cold.append(c["cold"]); scratch_fire.append(f)

        curve, appr = train_mirror_student(demos, seed=seed, episodes=args.episodes)
        s, c, f = evaluate(lambda w, p, m=appr: int(m(_solo(p.unsqueeze(0)))[0].argmax()),
                           f"apprentice[seed{seed}]")
        appr_surv.append(s); appr_cold.append(c["cold"]); appr_fire.append(f)
        appr_avatar.append(eval_avatar_oracle(appr, seed))

    print("\n=== VERDICT (means) ===")
    print(f"  scratch    : survival {statistics.mean(scratch_surv):.1f}  "
          f"cold-deaths {statistics.mean(scratch_cold):.1f}/40  "
          f"fire-occupancy {statistics.mean(scratch_fire):.1f}%")
    print(f"  apprentice : survival {statistics.mean(appr_surv):.1f}  "
          f"cold-deaths {statistics.mean(appr_cold):.1f}/40  "
          f"fire-occupancy {statistics.mean(appr_fire):.1f}%  "
          f"avatar-piloted {statistics.mean(appr_avatar):.1f}")
    print("  win = apprentice fire-occupancy UP and cold-deaths DOWN vs scratch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
