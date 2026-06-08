"""Consolidation done right — anchor the shared BASE policy, not just mirror cells.

Session 013 diagnosed the leak (HANDOFF open-item #2): the quest's consolidate-and-
lock froze the fire MIRROR cells but left the shared `perception->interior->output`
base policy free. Every dagger chapter's TD loss reshapes that base for ALL trainable
tensors (mirror_cells.py / quest_consolidate.py: `td_b/td_w` are added back AFTER the
mirror-gated imitation grads). So when the organism learns water, the base drifts and
the *frozen* fire skill erodes anyway — the output cell sums `interior->output` (base,
unfrozen) + `mirror->output` (frozen). Freezing the readout can't save a skill whose
substrate underneath keeps moving.

THE FIX (this file): at the fire->water boundary, anchor the base. We freeze every
edge whose BOTH endpoints predate water (perception->interior, interior->output,
interior->fire_mirror, fire_mirror->output) plus all old biases — but NOT the new
water mirror cells' edges. Water then learns purely as an additive readout off the
FROZEN interior representation (interior->water_mirror->output). Modularity: one frozen
shared representation, one fresh per-skill readout.

THE MEASUREMENT (this file, and the reason it's in the same module): the tile-world's
whole-episode occupancy swings +/-15 (predator/spawn luck) — session 013 hit a wall
trying to read 2-5pp retention deltas off it. We can't validate consolidation on a
metric that noisy. So retention is measured on a CONTROLLED PROBE: a fixed battery of
cold states (collected once from the fire master), scored by greedy policy-agreement.
Deterministic given the trained net — no episode noise. That's what makes the leak
falsifiable at small n.

Five capacity-matched arms (all end at 16 mirror cells), differing ONLY in the lock:
  PLAIN        : grow 8, water trains everything (16 mirror + base). Max forgetting.
  MIRROR-LOCK  : freeze fire's 8 mirror cells, grow 8, water trains fresh + base.
                 (the current quest_consolidate lock — the base leaks through.)
  FULL-LOCK    : anchor base + fire mirror, grow 8, water trains ONLY the fresh 8 (HARD).
  LAMBDA-FISHER: the namesake epigenetic lock λ, SOFT. Model-dist Fisher over the fire
                 decision states → per-node λ → EWC pull toward the fire anchor.
  LAMBDA-REWARD: λ driven by reward (activation of cells that handle fire correctly),
                 not Fisher — λ as an environment/reward sense. SOFT EWC pull.

Hard freeze (FULL) holds a skill STATIC; soft λ lets the shared base keep refining the
old skill while learning the new one (competence can RISE, not just hold). The reward
vs Fisher driver is the open question (HANDOFF 0b: reward likely the real world lever).

Prediction if the leak is real:
  fire-retention  PLAIN < MIRROR-LOCK < FULL-LOCK  (MIRROR-LOCK still drops = the leak)
  water-acquired  all three rise (FULL-LOCK is the open question: does anchoring the
                  base cost plasticity? if yes -> motivates SOFT anchoring next.)
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import fire_oracle, evaluate, TileWorld  # tamed-fire physics
from experiments.world.quest import water_master
from experiments.world.tile_world import N_ACTION
from experiments.world.mirror_cells import (
    _solo, keep_only_mirror_grads, obs_onehot, mirror_ids, add_mirror_cell, OBS_DIM,
)
from experiments.world.render_organism import train_solo
from trioron.core.epigenome import (
    PERCEPTION, OUTPUT, CREDIT_ELIGIBLE, MIRROR, has_gene,
)
from trioron.learning.epigenetic_lock import (
    accumulate_fisher, accumulate_saliency, refresh_lambda, anchor, ewc_penalty,
    set_lambda, fisher_loss,
)


def _p(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# Cell-type masks (epigenome-driven, capacity-length bool tensors)
# ----------------------------------------------------------------------
def _gene_mask(sub, gene, exclude=()):
    a = sub.arena
    m = torch.zeros(a.capacity, dtype=torch.bool)
    for c in a.alive_ids().tolist():
        epi = int(a.epigenome[c].item())
        if has_gene(epi, gene) and not any(has_gene(epi, g) for g in exclude):
            m[c] = True
    return m


def _alive_mask(sub):
    a = sub.arena
    m = torch.zeros(a.capacity, dtype=torch.bool)
    m[a.alive_ids()] = True
    return m


# ----------------------------------------------------------------------
# Growth (fresh mirror cells off the seeded interior + obs channel)
# ----------------------------------------------------------------------
def grow_mirror(sub, n):
    """Add n fresh mirror cells (interior efference + obs -> output)."""
    a = sub.arena
    perc = sorted(torch.nonzero(_gene_mask(sub, PERCEPTION)).flatten().tolist())
    obs_ids = torch.tensor(perc[-OBS_DIM:], dtype=torch.long)
    interior = torch.nonzero(_gene_mask(sub, CREDIT_ELIGIBLE, exclude=(OUTPUT, MIRROR))).flatten()
    out_ids = torch.nonzero(_gene_mask(sub, OUTPUT)).flatten()
    new = [add_mirror_cell(a, interior, obs_ids, out_ids) for _ in range(n)]
    a.rank_dirty = True
    sub.compile()
    return new


# ----------------------------------------------------------------------
# The lock — applied each batch, vectorized, position-independent.
# We zero grads from edge_src/edge_dst (NOT positional edge_protected) so a
# compile() reorder between consolidate and training can't misalign the mask.
# ----------------------------------------------------------------------
def make_lock(sub, mode, *, fire_mirror_mask=None, old_mask=None):
    """Return a closure lock(sub) that zeros the frozen grads for the given mode.

      none   : no-op (PLAIN).
      mirror : freeze every edge incident to a fire-mirror cell + their biases
               (the current quest_consolidate lock; the base leaks).
      full   : freeze every edge whose BOTH endpoints predate water + old biases;
               water's fresh-mirror edges (one endpoint new) stay trainable.
    """
    if mode == "none":
        return lambda s: None

    def lock(s):
        a = s.arena
        if a.edge_cursor == 0:
            return
        src = a.edge_src[: a.edge_cursor].long()
        dst = a.edge_dst[: a.edge_cursor].long()
        if mode == "mirror":
            edge_frozen = fire_mirror_mask[src] | fire_mirror_mask[dst]
            bias_frozen = fire_mirror_mask
        elif mode == "full":
            edge_frozen = old_mask[src] & old_mask[dst]   # old<->old only
            bias_frozen = old_mask                        # incl. output bias
        else:
            raise ValueError(mode)
        if a.bias.grad is not None:
            a.bias.grad[bias_frozen] = 0.0
        if a.edge_weight.grad is not None:
            a.edge_weight.grad[: a.edge_cursor][edge_frozen] = 0.0

    return lock


# ----------------------------------------------------------------------
# The epigenetic lock λ — SOFT anchoring (the namesake third node variable).
# Where make_lock zeros grads (hard freeze), λ adds a per-node-gated EWC pull
# toward the fire-competent anchor. Two drivers, normalized to max=1 so a single
# strength compares the PATTERN of protection (which cells), not its scale.
# ----------------------------------------------------------------------
@torch.no_grad()
def _normalize_lambda(a):
    """Scale λ so the stiffest cell = 1.0 (others ∈ (0,1]); makes ewc_strength
    comparable across the Fisher and reward drivers."""
    m = float(a.node_lambda.max())
    if m > 0:
        a.node_lambda.div_(m)


def setup_lambda_fisher(sub, states, *, batches=15, batch=64, seed=7):
    """Fisher driver: model-distribution Fisher over the fire decision states →
    row-sum → λ. fisher_loss samples the target from softmax so λ doesn't wash out
    at convergence (the calibration fix). Kept for reference; the swapped-in default
    is the native |w·g| driver below."""
    a = sub.arena
    g = torch.Generator().manual_seed(seed)
    n = states.shape[0]
    for _ in range(batches):
        idx = torch.randint(0, n, (batch,), generator=g)
        a.bias.grad = None; a.edge_weight.grad = None
        fisher_loss(sub(_solo(states[idx]))).backward()
        accumulate_fisher(a)
    refresh_lambda(a)
    _normalize_lambda(a)


def setup_lambda_wg(sub, states, acts, *, batches=15, batch=64, seed=7):
    """|w·g| saliency driver — trioron's NATIVE importance signal (KIBRA / utility /
    pruner dialect), not Fisher's grad². Rolls per-edge |w·g| over the oracle-labeled
    fire decision states into node_lambda. The |w| factor keeps settled weights
    important at convergence (no Fisher floor-collapse)."""
    a = sub.arena
    g = torch.Generator().manual_seed(seed)
    n = states.shape[0]
    ce = torch.nn.functional.cross_entropy
    for _ in range(batches):
        idx = torch.randint(0, n, (batch,), generator=g)
        a.bias.grad = None; a.edge_weight.grad = None
        ce(sub(_solo(states[idx])), acts[idx]).backward()      # oracle labels (KIBRA-style)
        accumulate_saliency(a)
    refresh_lambda(a)
    _normalize_lambda(a)


@torch.no_grad()
def setup_lambda_reward(sub, states, oracle_acts):
    """Reward driver: λ_i ∝ mean activation of cell i over the cold states the fire
    policy handles CORRECTLY (agreement with the oracle = the survival reward proxy).
    Protects the cells that implement the rewarded fire behavior — λ as an
    'environment/reward sense', the intrinsic-value path (no Fisher pass)."""
    a = sub.arena
    logits = sub(_solo(states))
    acts = sub.last_activations                         # [N, capacity], detached
    correct = (logits.argmax(dim=1) == oracle_acts).float()
    if float(correct.sum()) > 0:
        sig = (acts.abs() * correct[:, None]).sum(0) / correct.sum()
    else:
        sig = acts.abs().mean(0)
    set_lambda(a, sig, mode="absolute")
    _normalize_lambda(a)


# ----------------------------------------------------------------------
# DAgger chapter (master labels student-visited states; lock applied per batch)
# ----------------------------------------------------------------------
def dagger(sub, master_fn, *, seed, episodes, lock=lambda s: None, ewc_strength=0.0,
           gamma=0.95, lr=2e-3, batch=64, imit_w=0.5, max_steps=300):
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    a = sub.arena
    buf = deque(maxlen=20000)
    demos = deque(maxlen=20000)
    for ep in range(20):                                  # seed demos (master trajectory)
        w = TileWorld(seed=seed * 13000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            demos.append((p, master_fn(w)))
            p, _, done, _ = w.step(master_fn(w))
    g = torch.Generator().manual_seed(seed + 71)
    ce = torch.nn.functional.cross_entropy
    for ep in range(episodes):
        eps = 0.05 + 0.25 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep + 5000, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            demos.append((p, master_fn(w)))               # DAgger: label THIS state
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
                td_loss = torch.nn.functional.mse_loss(q, tgt)
                if ewc_strength > 0:                       # soft λ-EWC pull (not mirror-gated)
                    td_loss = td_loss + ewc_strength * ewc_penalty(a)
                td_loss.backward()
                td_b = a.bias.grad.clone(); td_w = a.edge_weight.grad.clone()
                didx = torch.randint(0, len(demos), (batch,), generator=g)
                ds = torch.stack([demos[i][0] for i in didx])
                da = torch.tensor([demos[i][1] for i in didx])
                avatar = torch.cat([ds, obs_onehot(da, batch)], dim=1)
                opt.zero_grad()
                (imit_w * (ce(sub(avatar), da) + ce(sub(_solo(ds)), da))).backward()
                keep_only_mirror_grads(sub)
                a.bias.grad.add_(td_b); a.edge_weight.grad.add_(td_w)
                lock(sub)                                  # <-- anchor the frozen substrate
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads()
                opt.step()
    return sub


# ----------------------------------------------------------------------
# Controlled probe — a FIXED battery of decision states; deterministic agreement
# ----------------------------------------------------------------------
@torch.no_grad()
def build_battery(master_fn, *, which, seeds=15, max_steps=300, cap=500):
    """Roll the master and collect (percept, master_action) at decision-relevant
    states. `which='cold'`: temp<0.5 (the fire oracle's warm-up decision band).
    `which='thirsty'`: thirst<0.9 (the water master's hydration-management band).
    A competent master keeps its drive satisfied, so these bands ARE the skill's
    decision region. Fixed seeds -> identical battery every run."""
    perc, acts = [], []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 31 + 3, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            keep = (w.temp < 0.5) if which == "cold" else (w.thirst < 0.9)
            a = master_fn(w)
            if keep:
                perc.append(p); acts.append(a)
            p, _, done, _ = w.step(a)
            if len(perc) >= cap:
                break
        if len(perc) >= cap:
            break
    return torch.stack(perc), torch.tensor(acts)


@torch.no_grad()
def policy_on(sub, battery_perc):
    """Greedy solo action on each battery state -> [N] int."""
    return sub(_solo(battery_perc)).argmax(dim=1)


# ----------------------------------------------------------------------
# One arm: solo base -> dagger fire -> [lock] -> grow -> dagger water
# ----------------------------------------------------------------------
def run_arm(mode, seed, *, solo_ep, fire_ep, water_ep, cold_bat, warm_bat,
            thirst_bat, ewc_strength=0.0, keep_sub=False):
    sub = train_solo(seed, solo_ep, n_mirror=8)
    sub = dagger(sub, fire_oracle, seed=seed, episodes=fire_ep)

    fire_mirror_mask = _gene_mask(sub, MIRROR)            # the 8 fire cells
    old_mask = _alive_mask(sub)                           # everything pre-water

    # snapshot the fire-competent policy (after fire, before water)
    pi_fire_cold = policy_on(sub, cold_bat[0])
    comp_fire = (pi_fire_cold == cold_bat[1]).float().mean().item()
    water_pre = (policy_on(sub, thirst_bat[0]) == thirst_bat[1]).float().mean().item()

    # λ arms: set the per-node lock on the FIRE-competent net (before growth)
    is_lambda = mode in ("lambda_fisher", "lambda_wg", "lambda_reward")
    if mode == "lambda_fisher":
        setup_lambda_fisher(sub, cold_bat[0])
    elif mode == "lambda_wg":
        setup_lambda_wg(sub, cold_bat[0], cold_bat[1])
    elif mode == "lambda_reward":
        setup_lambda_reward(sub, cold_bat[0], cold_bat[1])

    grow_mirror(sub, 8)                                   # fresh water capacity
    if is_lambda:
        anchor(sub.arena)                                 # defend fire weights (fresh water cells: λ≈0)
        lock, ewc = (lambda s: None), ewc_strength
    else:
        lock = make_lock(sub, mode, fire_mirror_mask=fire_mirror_mask, old_mask=old_mask)
        ewc = 0.0
    sub = dagger(sub, water_master, seed=seed, episodes=water_ep, lock=lock,
                 ewc_strength=ewc)

    pi_water_cold = policy_on(sub, cold_bat[0])
    retain = (pi_water_cold == pi_fire_cold).float().mean().item()       # self-consistency
    comp_water = (pi_water_cold == cold_bat[1]).float().mean().item()    # vs fire master
    water_post = (policy_on(sub, thirst_bat[0]) == thirst_bat[1]).float().mean().item()
    out = dict(retain=retain, comp_fire=comp_fire, comp_water=comp_water,
               water_pre=water_pre, water_post=water_post)
    if keep_sub:
        out["_sub"] = sub
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--solo-ep", type=int, default=120)
    ap.add_argument("--fire-ep", type=int, default=150)
    ap.add_argument("--water-ep", type=int, default=150)
    ap.add_argument("--ewc-strength", type=float, default=1.0,
                    help="soft λ-EWC pull strength for the LAMBDA arms (λ normalized max=1)")
    ap.add_argument("--quick", action="store_true",
                    help="tiny budget to prove the pipeline runs end-to-end")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.solo_ep, args.fire_ep, args.water_ep = 1, 20, 25, 25

    _p("=== CONSOLIDATION DONE RIGHT: anchor the base, not just the readout ===")
    _p("    retention measured on a FIXED cold-state battery (deterministic;")
    _p("    no episode noise) — the leak made falsifiable at small n.\n")

    cold_bat = build_battery(fire_oracle, which="cold")
    warm_bat = None
    thirst_bat = build_battery(water_master, which="thirsty")
    _p(f"battery: {cold_bat[0].shape[0]} cold states (fire decision), "
       f"{thirst_bat[0].shape[0]} thirsty states (water decision)\n")

    seeds = list(range(args.seeds))
    arms = {}
    for mode, label in (("none", "PLAIN"), ("mirror", "MIRROR-LOCK"),
                        ("full", "FULL-LOCK"),
                        ("lambda_wg", "LAMBDA-WG"),
                        ("lambda_reward", "LAMBDA-REWARD")):
        tag = f"lock={mode}" + (f", ewc={args.ewc_strength:g}"
                                if mode.startswith("lambda") else "")
        _p(f"########## {label} ({tag}) ##########")
        rows = []
        for s in seeds:
            r = run_arm(mode, s, solo_ep=args.solo_ep, fire_ep=args.fire_ep,
                        water_ep=args.water_ep, cold_bat=cold_bat, warm_bat=warm_bat,
                        thirst_bat=thirst_bat, ewc_strength=args.ewc_strength)
            rows.append(r)
            _p(f"  seed{s}: fire-retain {r['retain']:.3f}  "
               f"fire-comp(after-water) {r['comp_water']:.3f} "
               f"(was {r['comp_fire']:.3f})  "
               f"water {r['water_pre']:.3f}->{r['water_post']:.3f}")
        arms[label] = rows

    def mean(rows, k):
        return statistics.mean(r[k] for r in rows)

    def std(rows, k):
        xs = [r[k] for r in rows]
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    _p("\n=== SUMMARY (means +/- pstd, n=%d) ===" % args.seeds)
    _p(f"  {'arm':>12s} | {'fire-retain':>18s} | {'fire-comp f->w':>16s} | "
       f"{'water-acq pre->post':>20s}")
    for label, rows in arms.items():
        _p(f"  {label:>12s} | {mean(rows,'retain'):.3f} +/- {std(rows,'retain'):.3f}"
           f"      | {mean(rows,'comp_fire'):.3f}->{mean(rows,'comp_water'):.3f}"
           f"   | {mean(rows,'water_pre'):.3f}->{mean(rows,'water_post'):.3f}")
    _p("\n  leak confirmed if  retain(FULL) > retain(MIRROR) > retain(PLAIN).")
    _p("  hard freeze (FULL) maximizes raw retention but holds fire competence STATIC;")
    _p("  soft λ (LAMBDA-*) trades some retention to let water training IMPROVE fire")
    _p("  competence (f->w rises). reward-driven λ vs Fisher-driven λ: which lever wins?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
