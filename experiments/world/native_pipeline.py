"""Full NATIVE consolidation pipeline — use trioron's own machinery, not hand-rolls.

Rocky (2026-06-04): "utilize all the trioron advantages instead of re-inventing the
wheels." The world experiments ran a raw Adam loop over sub.trainable_tensors() and
bypassed the substrate's designed continual-learning machinery — then hand-rolled crude
versions (freeze_grads, grow_mirror, an almost-written EWC penalty). This module wires
the NATIVE mechanisms into the fire→water apprenticeship, end to end:

  CreditTracker      — credit-based DORMANT locking ("the primary anchoring mechanism",
                       §4.1). Driven each batch (update_engagement + update_utility);
                       consolidate() at the task boundary locks the stabilised cells.
  ManifoldArchive    — per-ACTION (μ,σ) sketches of the fire policy's input distribution
                       (action-as-class: CE on policy logits = BC rehearsal). This is the
                       mechanism that beat EWC in our classification benches, finally used
                       in the world. Replay re-runs the FULL forward from synthesised
                       inputs (the new substrate stores input-space codes, dream._run_replay
                       slices x = batch[:, :n_perc]) — so NO bottleneck-injection needed.
  dream_cycle        — at the boundary: replay fire (rehearsal) → consolidate (lock) →
                       rejuvenate. The substrate's offline-consolidation orchestrator.
  FrustrationDetector→ divide  — native frustration→growth: when water stalls, grow fresh
                       capacity by mitosis instead of pre-allocating mirror cells.

Sequencing gotchas found while wiring (verified against source):
  * CreditConfig.consecutive_tasks defaults to 4 (assumes a long curriculum); a single
    fire→water boundary locks NOTHING unless set to 1.
  * dream_cycle does NOT call archive.finalize_all(); replay_batches() only returns
    DORMANT astrocytes — so finalize_all() must be called at the boundary FIRST, else
    fire's still-ACTIVE astrocytes replay nothing.
  * credit.consolidate() requires utility < g_min to lock; the default g_min (3.92e-6) is
    calibrated for classification and may be too strict for the world — exposed as a knob.

Measured on the SAME deterministic cold/thirsty battery as consolidate_base.py, against
PLAIN and the hand-rolled FULL-LOCK, so the question is sharp: does trioron's OWN
machinery match or beat the hand-rolled freeze?
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import fire_oracle, TileWorld
from experiments.world.quest import water_master
from experiments.world.tile_world import N_ACTION
from experiments.world.mirror_cells import (
    _solo, keep_only_mirror_grads, obs_onehot, mirror_ids,
)
from experiments.world.render_organism import train_solo
from experiments.world.consolidate_base import (
    build_battery, policy_on, _gene_mask, _alive_mask, run_arm,
)
from trioron.core.epigenome import OUTPUT, CREDIT_ELIGIBLE, MIRROR
from trioron.core.state import CellState
from trioron.learning.credit import CreditTracker, CreditConfig
from trioron.learning.manifold import ManifoldArchive, ManifoldConfig
from trioron.learning.dream import dream_cycle, DreamConfig
from trioron.learning.frustration import FrustrationDetector
from trioron.lifecycle.grow import divide, check_growth_trigger, GrowthConfig
from trioron.lifecycle.saliency import compute_saliency


def _p(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# Native DAgger chapter — drives credit + archive + frustration each batch
# ----------------------------------------------------------------------
def dagger_native(sub, master_fn, *, seed, episodes, credit, archive, frust,
                  archive_fire=False, replay_fire=False, grow=False, imit_gated=True,
                  gamma=0.95, lr=2e-3, batch=64, imit_w=0.5, max_steps=300,
                  grow_cfg=None, diag=None):
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    a = sub.arena
    buf = deque(maxlen=20000)
    demos = deque(maxlen=20000)
    for ep in range(20):
        w = TileWorld(seed=seed * 13000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            demos.append((p, master_fn(w)))
            p, _, done, _ = w.step(master_fn(w))
    g = torch.Generator().manual_seed(seed + 71)
    ce = torch.nn.functional.cross_entropy
    n_grow = 0
    for ep in range(episodes):
        eps = 0.05 + 0.25 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep + 5000, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            demos.append((p, master_fn(w)))
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
                td_loss.backward()
                td_b = a.bias.grad.clone(); td_w = a.edge_weight.grad.clone()
                # fire rehearsal — WHOLE-NET pseudo-replay (protects the BASE policy,
                # not just mirror cells), sampled from the z2 MIXTURE (not the single
                # Gaussian, which smears the per-action input distribution).
                rb = rw = None
                if replay_fire:
                    rloss = None; nrep = 0
                    for cid in archive.class_ids:
                        astro = archive.get(cid)
                        if astro is None or a.state[astro.cell_id] != CellState.DORMANT:
                            continue
                        samples = archive.sample_mixture(cid, batch)
                        if samples is None or samples.shape[0] == 0:
                            continue
                        tgt_a = torch.full((samples.shape[0],), cid, dtype=torch.long)
                        term = ce(sub(samples), tgt_a)
                        rloss = term if rloss is None else rloss + term
                        nrep += 1
                    if nrep:
                        opt.zero_grad(); (imit_w * rloss).backward()
                        rb = a.bias.grad.clone(); rw = a.edge_weight.grad.clone()
                # water-master imitation — mirror-gated (the apprenticing design)
                didx = torch.randint(0, len(demos), (batch,), generator=g)
                ds = torch.stack([demos[i][0] for i in didx])
                da = torch.tensor([demos[i][1] for i in didx])
                avatar = torch.cat([ds, obs_onehot(da, batch)], dim=1)
                opt.zero_grad()
                imit = imit_w * (ce(sub(avatar), da) + ce(sub(_solo(ds)), da))
                imit.backward()
                if imit_gated:                              # apprenticing convention
                    keep_only_mirror_grads(sub)             # (off → whole-net imitation)
                a.bias.grad.add_(td_b); a.edge_weight.grad.add_(td_w)
                if rb is not None:                          # whole-net fire rehearsal
                    a.bias.grad.add_(rb); a.edge_weight.grad.add_(rw)
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads()                 # native protection of locked cells
                opt.step()

                # ── native bookkeeping ──
                credit.update_engagement(sub.last_activations)
                credit.update_utility()
                frust.step(float(td_loss.item()))
                if archive_fire:                          # archive fire policy per action
                    for act_id in da.unique().tolist():
                        m = da == act_id
                        archive.update_class(int(act_id), _solo(ds[m]))
                if grow and check_growth_trigger(frust.multiplier, frust._step_count,
                                                 grow_cfg) and n_grow < 8:
                    pid = _most_salient_active(sub)
                    if pid is not None and divide(a, pid, grow_cfg) is not None:
                        a.rank_dirty = True; sub.compile()
                        opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
                        n_grow += 1
    if diag is not None:
        diag["n_grow"] = n_grow
    return sub


def _most_salient_active(sub):
    """Pick the highest-saliency ACTIVE, credit-eligible, non-output cell to divide."""
    a = sub.arena
    sal = compute_saliency(a)
    elig = _gene_mask(sub, CREDIT_ELIGIBLE, exclude=(OUTPUT,))
    elig &= (a.state == CellState.ACTIVE)
    if not elig.any():
        return None
    masked = torch.where(elig, sal, torch.full_like(sal, -1.0))
    return int(masked.argmax())


# ----------------------------------------------------------------------
# The native arm — fire chapter → dream cycle → water chapter
# ----------------------------------------------------------------------
def run_native(seed, *, solo_ep, fire_ep, water_ep, cold_bat, thirst_bat,
               g_min=1e-3, mixture_k=2, grow=False, use_dream=False,
               imit_gated=False, verbose=False):
    sub = train_solo(seed, solo_ep, n_mirror=8)
    a = sub.arena
    credit = CreditTracker(a, CreditConfig(consecutive_tasks=1, g_min=g_min))
    archive = ManifoldArchive(a, ManifoldConfig(), mixture_k=mixture_k)
    frust = FrustrationDetector()
    grow_cfg = GrowthConfig()

    # FIRE chapter: learn + drive credit + archive the fire policy (z2 mixture)
    dagger_native(sub, fire_oracle, seed=seed, episodes=fire_ep, credit=credit,
                  archive=archive, frust=frust, archive_fire=True)

    pi_fire_cold = policy_on(sub, cold_bat[0])
    comp_fire = (pi_fire_cold == cold_bat[1]).float().mean().item()
    water_pre = (policy_on(sub, thirst_bat[0]) == thirst_bat[1]).float().mean().item()

    # BOUNDARY: freeze fire archive (→DORMANT, replayable) then native credit-lock.
    # CLEAN mode skips dream_cycle's replay-SGD + rejuvenation churn — retention is
    # carried by interleaved whole-net replay during water, not by re-fitting here.
    archive.finalize_all()
    if use_dream:
        fpres = frust.multiplier if frust.is_frustrated else 0.0
        res = dream_cycle(sub, credit, archive, frustration_pressure=fpres,
                          growth_rate=0.0, cfg=DreamConfig())
        n_locked = res.n_locked
    else:
        n_locked = len(credit.consolidate(stability_factor=1.0))
        sub.compile()
    frust.reset()

    # WATER chapter: learn + interleaved whole-net fire replay (+ optional growth)
    diag = {}
    dagger_native(sub, water_master, seed=seed, episodes=water_ep, credit=credit,
                  archive=archive, frust=frust, replay_fire=True, grow=grow,
                  imit_gated=imit_gated, grow_cfg=grow_cfg, diag=diag)

    pi_water_cold = policy_on(sub, cold_bat[0])
    retain = (pi_water_cold == pi_fire_cold).float().mean().item()
    comp_water = (pi_water_cold == cold_bat[1]).float().mean().item()
    water_post = (policy_on(sub, thirst_bat[0]) == thirst_bat[1]).float().mean().item()
    if verbose:
        _p(f"    [native diag] n_locked={n_locked} mixture_k={mixture_k} "
           f"grow={diag.get('n_grow', 0)} archive_classes={archive.n_classes}")
    return dict(retain=retain, comp_fire=comp_fire, comp_water=comp_water,
                water_pre=water_pre, water_post=water_post,
                n_locked=n_locked, n_grow=diag.get("n_grow", 0))


# ----------------------------------------------------------------------
# Smoke — prove each native piece actually fires before the comparison
# ----------------------------------------------------------------------
def smoke():
    _p("native_pipeline smoke — does each native mechanism actually fire?\n")
    torch.manual_seed(20260604)              # masters' explore uses GLOBAL rng -> seed it
    cold_bat = build_battery(fire_oracle, which="cold", seeds=4)
    thirst_bat = build_battery(water_master, which="thirsty", seeds=4)
    r = run_native(0, solo_ep=15, fire_ep=20, water_ep=20, cold_bat=cold_bat,
                   thirst_bat=thirst_bat, verbose=True)
    _p(f"\n  result: retain={r['retain']:.3f} comp {r['comp_fire']:.3f}->{r['comp_water']:.3f} "
       f"water {r['water_pre']:.3f}->{r['water_post']:.3f}")
    _p(f"  native fired: locked={r['n_locked']} grow={r['n_grow']}")
    if r["n_locked"] == 0:
        _p("  WARNING: nothing locked — raise g_min (utility never dipped below it).")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--solo-ep", type=int, default=120)
    ap.add_argument("--fire-ep", type=int, default=150)
    ap.add_argument("--water-ep", type=int, default=150)
    ap.add_argument("--g-min", type=float, default=1e-3)
    args = ap.parse_args()
    if args.smoke:
        return smoke()

    _p("=== NATIVE PIPELINE vs PLAIN vs hand-rolled FULL-LOCK (n=%d) ===" % args.seeds)
    _p("    does trioron's OWN machinery (credit-lock + manifold replay + dream +")
    _p("    frustration-growth) match/beat the hand-rolled freeze?\n")
    torch.manual_seed(20260604)              # masters' explore uses GLOBAL rng -> seed it
    cold_bat = build_battery(fire_oracle, which="cold")
    thirst_bat = build_battery(water_master, which="thirsty")
    _p(f"battery: {cold_bat[0].shape[0]} cold / {thirst_bat[0].shape[0]} thirsty\n")

    seeds = list(range(args.seeds))
    arms = {}
    for label in ("PLAIN", "FULL-LOCK", "NATIVE-free", "NATIVE-gate"):
        _p(f"########## {label} ##########")
        rows = []
        for s in seeds:
            if label.startswith("NATIVE"):
                r = run_native(s, solo_ep=args.solo_ep, fire_ep=args.fire_ep,
                               water_ep=args.water_ep, cold_bat=cold_bat,
                               thirst_bat=thirst_bat, g_min=args.g_min,
                               imit_gated=(label == "NATIVE-gate"), verbose=True)
                extra = f"  [lock {r['n_locked']}]"
            else:
                mode = "none" if label == "PLAIN" else "full"
                r = run_arm(mode, s, solo_ep=args.solo_ep, fire_ep=args.fire_ep,
                            water_ep=args.water_ep, cold_bat=cold_bat, warm_bat=None,
                            thirst_bat=thirst_bat)
                extra = ""
            rows.append(r)
            _p(f"  seed{s}: retain {r['retain']:.3f}  comp {r['comp_fire']:.3f}->"
               f"{r['comp_water']:.3f}  water {r['water_pre']:.3f}->{r['water_post']:.3f}{extra}")
        arms[label] = rows

    def m(rows, k):
        return statistics.mean(r[k] for r in rows)

    def sd(rows, k):
        xs = [r[k] for r in rows]
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    _p("\n=== SUMMARY (means +/- pstd, n=%d) ===" % args.seeds)
    _p(f"  {'arm':>10s} | {'fire-retain':>16s} | {'fire-comp f->w':>15s} | {'water pre->post':>16s}")
    for label, rows in arms.items():
        _p(f"  {label:>10s} | {m(rows,'retain'):.3f} +/- {sd(rows,'retain'):.3f}    | "
           f"{m(rows,'comp_fire'):.3f}->{m(rows,'comp_water'):.3f}   | "
           f"{m(rows,'water_pre'):.3f}->{m(rows,'water_post'):.3f}")
    _p("\n  native wins if it matches/beats FULL-LOCK retain AND keeps water rising,")
    _p("  using trioron's OWN consolidation rather than a hand-rolled freeze.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
