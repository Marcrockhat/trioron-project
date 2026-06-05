"""Redesigned WARM curriculum — staged skills, taught one at a time.

The temporal-WARM experiment (temporal_warm.py) was confounded: it asked the
substrate to learn navigate-to-random-fire AND the leave-timing latch at once,
from scratch. The fire is re-scattered every spawn (TileWorld.reset), so "warm
up" is scent-following navigation to a random location that must generalize.
Navigation dominated and failed -> cold-collapse -> the timing skill was never
exercised. Two skills entangled in one curriculum.

Fix (Rocky, 2026-06-05): stage the curriculum (Mode-E's primitive-curriculum
doctrine, applied INSIDE the WARM skill):

  Stage 0 — NAVIGATE : follow the fire-scent to a randomly-placed fire so temp
                       reliably climbs. No temporal demand. (THIS FILE.)
  Stage 1 — REGULATE : reactive leave/return, hold temp in the band. Reactive;
                       test whether feed-forward suffices.
  Stage 2 — ANTICIPATE: leave EARLY under actuation lag + the 0.04-heat/0.008-cool
                       asymmetry, to kill the overshoot a reactive policy can't.
                       This is the only stage that needs memory — the clean
                       paradigm test (recurrent ON vs OFF), on top of working
                       navigation.

Stage 0 here establishes the floor: can a plain feed-forward donor, trained on
clean approach demos, navigate to fire on HELD-OUT randomized worlds and raise
temperature? If yes, the cold-collapse was the entanglement, and we proceed to
Stage 1. If no, navigation itself is the wall and the whole WARM line needs a
perception/scent fix first.

Usage:
    python3 -m experiments.world.warm_curriculum --smoke
    python3 -m experiments.world.warm_curriculum            # Stage 0 verdict
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import experiments.world.fire_taming as _ft               # noqa: E402  (pins WARM_RATE=0.04)
from experiments.world.fire_taming import _toward, _near_fire, _away_from_fire  # noqa: E402
from experiments.world.tile_world import TileWorld, ACTIONS, FIRE  # noqa: E402
from experiments.world.mirror_cells import _solo, build_mirror  # noqa: E402
from experiments.world.tile_world import N_ACTION            # noqa: E402
from experiments.world.primitives import (                 # noqa: E402
    collect, train_donor,
)
from collections import deque                               # noqa: E402
import statistics as _stats                                 # noqa: E402
from trioron.core.epigenome import LINEAR, DENDRITE, OUTPUT, has_gene  # noqa: E402
from trioron.lifecycle import divide, GrowthConfig          # noqa: E402
from trioron.learning import FrustrationDetector, FrustrationConfig  # noqa: E402
from trioron.learning.manifold import get_interior_ids      # noqa: E402

_REST = ACTIONS.index("rest")


def _p(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# Stage 0 master + band — pure scent-following navigation TO fire
# ----------------------------------------------------------------------
def nav_master(w):
    """Walk toward the nearest fire by its scent. (No regulation — Stage 0 only
    teaches REACHING the fire; leaving is Stage 1+.)"""
    t = _toward(w, FIRE)
    return t if t is not None else _REST


def _band_navigate(w):
    """Keep the clean APPROACH steps: the agent is still navigating (not yet
    adjacent to the fire). Excludes the at-fire steps where the skill is no
    longer 'get there' but 'what now' (Stage 1's job)."""
    return not _near_fire(w)


# ----------------------------------------------------------------------
# Evaluate navigation: does the donor REACH fire (temp climbs) on held-out
# randomized worlds? We never test on a training seed.
# ----------------------------------------------------------------------
@torch.no_grad()
def eval_navigation(sub, *, seeds=40, max_steps=120, reach_temp=0.8):
    """Run the donor as sole controller from the (fixed-centre) start on fresh
    worlds. Report: fraction of episodes whose temp ever reaches `reach_temp`
    (reached the fire), median steps-to-reach, and median peak temp."""
    reached, steps_to, peaks = 0, [], []
    for ep in range(seeds):
        w = TileWorld(seed=ep * 9001 + 3, max_steps=max_steps)   # held-out seeds
        p = w.percept(); done = False
        peak, hit = w.temp, None
        while not done:
            a = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            p, _, done, info = w.step(a)
            peak = max(peak, w.temp)
            if hit is None and w.temp >= reach_temp:
                hit = info["t"]
        peaks.append(peak)
        if hit is not None:
            reached += 1; steps_to.append(hit)
    reach_rate = reached / seeds
    med_steps = statistics.median(steps_to) if steps_to else float("nan")
    return reach_rate, med_steps, statistics.median(peaks)


@torch.no_grad()
def nav_fidelity(sub, *, seeds=20, max_steps=120):
    """Held-out imitation fidelity of the toward-fire action on navigation
    states (fresh worlds), and the per-action confusion the donor makes."""
    correct = total = 0
    conf = Counter()
    for ep in range(seeds):
        w = TileWorld(seed=ep * 4242 + 19, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            if _band_navigate(w):
                want = nav_master(w)
                got = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
                total += 1
                if got == want:
                    correct += 1
                else:
                    conf[f"{ACTIONS[want]}->{ACTIONS[got]}"] += 1
            p, _, done, _ = w.step(nav_master(w))   # follow master to stay on-policy
    return (correct / max(total, 1)), conf.most_common(4)


# ----------------------------------------------------------------------
# Stage 0 run
# ----------------------------------------------------------------------
def stage0(*, seed=0, collect_seeds=80, cap=1500, epochs=300, eval_seeds=40):
    _ft.EXPLORE_DETERMINISTIC = False
    _p("=== STAGE 0: navigate to fire (feed-forward) ===\n")
    P, Y = collect(nav_master, _band_navigate, seeds=collect_seeds, cap=cap)
    _p(f"collected {P.shape[0]} clean approach frames "
       f"({int(Y.unique().numel())} distinct actions)")
    sub, acc, chance, _ = train_donor(P, Y, seed=seed, epochs=epochs)
    _p(f"held-out imitation (i.i.d. split): fidelity {acc:.3f}  chance {chance:.3f}\n")

    fid, conf = nav_fidelity(sub, seeds=20)
    reach_rate, med_steps, med_peak = eval_navigation(sub, seeds=eval_seeds)
    _p(f"on-policy nav fidelity (held-out worlds): {fid:.3f}")
    if conf:
        _p("  top confusions: " + "  ".join(f"{k}:{v}" for k, v in conf))
    _p(f"\nNAVIGATION (held-out, sole controller, max_steps 120):")
    _p(f"  reached fire (temp>=0.8): {reach_rate*100:.0f}%   "
       f"median steps-to-reach {med_steps}   median peak temp {med_peak:.2f}")
    verdict = ("PASS — navigation generalizes; proceed to Stage 1 (regulate)"
               if reach_rate >= 0.8 and med_peak >= 0.8
               else "FAIL — navigation is the wall; fix scent/perception before timing")
    _p(f"\n=== STAGE 0 VERDICT: {verdict} ===")
    return sub, reach_rate, med_peak


# ----------------------------------------------------------------------
# Stage 1 — REACTIVE regulation (feed-forward). Locate the overheat wall:
# does a standalone reactive donor hold the band, or does it overshoot?
# ----------------------------------------------------------------------
def regulate_master(w):
    """Pure reactive threshold: warm up below target, cool down above it. No
    hysteresis band, no memory — the honest reactive baseline. Overshoot (if
    any) comes from the 0.04-heat/0.008-cool asymmetry + the lag of physically
    moving clear of the fire — exactly what Stage 2 anticipation would target."""
    if w.temp < 0.5:
        t = _toward(w, FIRE)
        return t if t is not None else _REST
    return _away_from_fire(w)


def _cause(w):
    if w.alive:
        return "timeout"
    if w.temp <= w.temp_low:
        return "cold"
    if w.temp >= w.temp_high:
        return "overheat"
    if w.thirst <= 0:
        return "thirst"
    if w.energy <= 0:
        return "energy"
    return "integrity"


@torch.no_grad()
def eval_regulate(sub, *, seeds=40, max_steps=300):
    """Solo controller. The locus test: a reactive regulator that HOLDS the band
    dies of thirst/energy (it ignores them), NOT thermally. Count thermal deaths
    and the temperature profile (mean + fraction of steps overshooting 0.8)."""
    surv, causes, temps, hot = [], Counter(), [], 0
    n_steps = 0
    for ep in range(seeds):
        w = TileWorld(seed=ep * 7000 + 7, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            a = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            p, _, done, info = w.step(a)
            temps.append(float(w.temp)); n_steps += 1
            if w.temp >= 0.8:
                hot += 1
        surv.append(info["t"]); causes[_cause(w)] += 1
    thermal = causes["overheat"] + causes["cold"]
    return (statistics.mean(surv), causes, thermal,
            statistics.mean(temps), hot / max(n_steps, 1))


def _train_donor_nl(P, Y, *, seed, epochs, nonlinear, lr=3e-3, batch=128):
    """train_donor, but with the substrate's nonlinearity toggleable. The
    regulation policy is a temp x fire-direction interaction (sign-flip of the
    action conditioned on temp) — not linearly separable, so this is the lever
    the linear default can't reach."""
    torch.manual_seed(seed)
    n = P.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    ntr = int(n * 0.8)
    tr, te = perm[:ntr], perm[ntr:]
    Ptr, Ytr, Pte, Yte = P[tr], Y[tr], P[te], Y[te]
    sub = build_mirror(seed, n_mirror=8, nonlinear=nonlinear)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    ce = torch.nn.functional.cross_entropy
    ntr = Ptr.shape[0]
    for _ in range(epochs):
        ep_perm = torch.randperm(ntr)
        for i in range(0, ntr, batch):
            idx = ep_perm[i:i + batch]
            opt.zero_grad()
            ce(sub(_solo(Ptr[idx])), Ytr[idx]).backward()
            sub.zero_dormant_grads(); opt.step()
    with torch.no_grad():
        acc = (sub(_solo(Pte)).argmax(1) == Yte).float().mean().item()
        maj = int(torch.bincount(Ytr, minlength=N_ACTION).argmax())
        chance = (Yte == maj).float().mean().item()
    return sub, acc, chance


# ----------------------------------------------------------------------
# NATIVE rebuild — the substrate GROWS its own quad cells under regulation
# frustration (selective_quad_growth machinery), instead of a hand-set flag.
# ----------------------------------------------------------------------
def _make_child_quad(arena, cid):
    epi = int(arena.epigenome[cid].item())
    arena.epigenome[cid] = (epi & ~(1 << LINEAR)) | (1 << DENDRITE)
    arena.refresh_phenotype(cid)


def _count_quad(sub):
    a = sub.arena
    return sum(1 for c in a.alive_ids().tolist()
               if has_gene(int(a.epigenome[c].item()), DENDRITE))


def train_donor_native(P, Y, *, seed=0, epochs=300, lr=3e-3, batch=128, grow=True,
                       grow_budget=None, min_linear=3, stuck_loss=0.7,
                       plateau_eps=0.02, cap_bytes=50_000_000, capacity=8192):
    """Adaptive selective-quad growth on the regulation demos. Start ALL-LINEAR;
    under sustained imitation frustration, divide interior cells (linear width);
    once linear growth plateaus while loss is STILL STUCK (the relational wall),
    escalate the child phenotype to DENDRITE (quad). Returns (sub, acc, grown,
    n_quad). grow=False is the all-linear floor.

    grow_budget=None => UNCAPPED: growth is bounded only by the substrate's own
    frustration self-throttling (grow under frustration; stop when the added
    capacity relieves it). cap_bytes/capacity are raised so the envelope is not
    the limiter. (NOTE: this deliberately exceeds the Phase-1 50K-param contract —
    it is an exploration of how much nonlinearity the task demands, not a
    deployable donor; a shipped donor would compact back under the envelope.)"""
    torch.manual_seed(seed)
    n = P.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    ntr = int(n * 0.8)
    tr, te = perm[:ntr], perm[ntr:]
    Ptr, Ytr, Pte, Yte = P[tr], Y[tr], P[te], Y[te]
    budget = float("inf") if grow_budget is None else grow_budget
    sub = build_mirror(seed, n_mirror=8, nonlinear=False,       # all-linear start
                       capacity=capacity, cap_bytes=cap_bytes)
    a = sub.arena
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    ce = torch.nn.functional.cross_entropy
    frust = FrustrationDetector(FrustrationConfig())
    out_ids = torch.tensor([c for c in a.alive_ids().tolist()
                            if has_gene(int(a.epigenome[c].item()), OUTPUT)],
                           dtype=torch.int32)

    def wire_to_output(cid):                                    # divide adds only inputs
        a.add_edges(torch.full_like(out_ids, cid), out_ids,
                    0.1 * torch.randn(out_ids.numel()))

    grown = frust_steps = 0
    escalated = False
    prev_growth_loss = None
    recent = deque(maxlen=20)
    ntr2 = Ptr.shape[0]
    for _ in range(epochs):
        ep_perm = torch.randperm(ntr2)
        for i in range(0, ntr2, batch):
            idx = ep_perm[i:i + batch]
            loss = ce(sub(_solo(Ptr[idx])), Ytr[idx])
            recent.append(loss.item())
            m = frust.step(loss.item())
            if frust.is_frustrated:
                frust_steps += 1
            opt.zero_grad(); (loss * m).backward()
            torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
            sub.zero_dormant_grads(); opt.step()

            if (grow and grown < budget and frust.is_frustrated
                    and frust_steps >= 25):
                ip = get_interior_ids(a).long().tolist()
                if ip:
                    parent = ip[torch.randint(0, len(ip), (1,)).item()]
                    cur = _stats.mean(recent) if recent else loss.item()
                    if (not escalated and grown >= min_linear
                            and prev_growth_loss is not None
                            and cur > stuck_loss
                            and (prev_growth_loss - cur) < plateau_eps):
                        escalated = True            # relational wall -> go quad
                    prev_growth_loss = cur
                    ev = divide(a, parent, GrowthConfig())
                    if ev:
                        if escalated:
                            _make_child_quad(a, ev.child_id)
                        wire_to_output(ev.child_id)
                        grown += 1
                        a.rank_dirty = True; sub.compile()
                        frust_steps = 0
    with torch.no_grad():
        acc = (sub(_solo(Pte)).argmax(1) == Yte).float().mean().item()
    return sub, acc, grown, _count_quad(sub)


def stage1_native(*, seed=0, collect_seeds=80, cap=2000, epochs=300, eval_seeds=40):
    _ft.EXPLORE_DETERMINISTIC = False
    _p("\n=== STAGE 1 (NATIVE): regulation via selective quad GROWTH ===")
    _p("    all-linear start -> frustration grows quad where temp x direction demands")
    _p("    ceilings: reactive master 3/40 thermal | hand-set nonlinear 15/40\n")
    P, Y = collect(regulate_master, lambda w: True, seeds=collect_seeds, cap=cap)
    _p(f"collected {P.shape[0]} regulation frames\n")
    for grow, name in [(False, "all-linear FLOOR (no growth)"),
                       (True,  "NATIVE adaptive quad-growth")]:
        sub, acc, grown, nquad = train_donor_native(
            P, Y, seed=seed, epochs=epochs, grow=grow)
        surv, causes, thermal, mean_temp, hot_frac = eval_regulate(
            sub, seeds=eval_seeds)
        _p(f"[{name}]")
        _p(f"  grown {grown} cells, {nquad} QUAD   fidelity {acc:.3f}   "
           f"survival {surv:.1f}   THERMAL {thermal}/{eval_seeds}")
        _p(f"  mean temp {mean_temp:.2f}  deaths {dict(causes.most_common())}\n")
    _p("=== NATIVE VERDICT: growth should spawn quad cells and pull THERMAL")
    _p("    toward the nonlinear result — the substrate discovering it needs")
    _p("    nonlinearity from regulation frustration alone. ===")
    return None


def stage1(*, seed=0, collect_seeds=80, cap=2000, epochs=300, eval_seeds=40):
    _ft.EXPLORE_DETERMINISTIC = False
    _p("\n=== STAGE 1: reactive regulation — LINEAR vs NONLINEAR donor ===")
    _p("    master ceilings: reactive 3/40 thermal, hysteretic 6/40 (both HOLD)\n")
    P, Y = collect(regulate_master, lambda w: True, seeds=collect_seeds, cap=cap)
    _p(f"collected {P.shape[0]} regulation frames "
       f"({int(Y.unique().numel())} distinct actions)\n")

    for nonlinear, name in [(False, "LINEAR (default)"), (True, "NONLINEAR (quad)")]:
        sub, acc, chance = _train_donor_nl(P, Y, seed=seed, epochs=epochs,
                                           nonlinear=nonlinear)
        surv, causes, thermal, mean_temp, hot_frac = eval_regulate(
            sub, seeds=eval_seeds)
        _p(f"[{name}]")
        _p(f"  fidelity {acc:.3f} (chance {chance:.3f})   survival {surv:.1f}   "
           f"THERMAL {thermal}/{eval_seeds}")
        _p(f"  mean temp {mean_temp:.2f}  steps hot(>=0.8) {hot_frac*100:.1f}%  "
           f"deaths {dict(causes.most_common())}\n")
    _p("=== STAGE 1: if NONLINEAR slashes thermal toward the 3/40 master ceiling,")
    _p("    the wall is NONLINEARITY (temp x direction), not memory. ===")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--collect-seeds", type=int, default=80)
    ap.add_argument("--cap", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-seeds", type=int, default=40)
    ap.add_argument("--stage", type=int, default=None, help="run only this stage")
    args = ap.parse_args()
    if args.smoke:
        args.collect_seeds, args.cap, args.epochs, args.eval_seeds = 20, 400, 40, 10

    if args.stage in (None, 0):
        stage0(seed=args.seed, collect_seeds=args.collect_seeds, cap=args.cap,
               epochs=args.epochs, eval_seeds=args.eval_seeds)
    if args.stage in (None, 1):
        stage1(seed=args.seed, collect_seeds=args.collect_seeds,
               cap=max(args.cap, 2000), epochs=args.epochs, eval_seeds=args.eval_seeds)
    if args.stage in (None, 2):
        stage1_native(seed=args.seed, collect_seeds=args.collect_seeds,
                      cap=max(args.cap, 2000), epochs=args.epochs,
                      eval_seeds=args.eval_seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
