"""Consolidate-and-lock — does protecting a learned skill stop the next one
erasing it? (Rocky 2026-06-03: the quest skipped Numa/Mima consolidation, so new
skills overwrite old ones. Wire the lock and re-test, multi-seed.)

Controlled, capacity-matched comparison (both arms end with 16 mirror cells):
  PLAIN  : 16 mirror cells, no locking. Fire trains all 16, water trains all 16.
  LOCKED : 8 cells learn fire → Numa-check (held-out re-test) → consolidate (lock
           those 8: edge_protected + frozen grads) → grow 8 fresh → water trains
           ONLY the new 8. Fire is physically protected.

Win for LOCKED: fire-occupancy is RETAINED from after-fire → after-water (no
regression) while water is still acquired (thirst-deaths fall) — i.e. two skills
coexist, which plain SGD couldn't manage.
"""
from __future__ import annotations

import sys
import statistics
from collections import deque, Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import evaluate, TileWorld     # sets tamed-fire physics
from experiments.world.tile_world import N_ACTION
from experiments.world.mirror_cells import (
    build_mirror, _solo, keep_only_mirror_grads, obs_onehot, mirror_ids,
    add_mirror_cell, OBS_DIM,
)
from experiments.world.render_organism import train_solo
from experiments.world.quest import MASTERS
from trioron.core.epigenome import PERCEPTION, OUTPUT, CREDIT_ELIGIBLE, MIRROR, has_gene


def _p(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------
# Growth + consolidation primitives
# ----------------------------------------------------------------------
def _gene_ids(a, gene, exclude=()):
    return [c for c in a.alive_ids().tolist()
            if has_gene(int(a.epigenome[c].item()), gene)
            and not any(has_gene(int(a.epigenome[c].item()), g) for g in exclude)]


def grow_mirror(sub, n):
    """Add n fresh mirror cells (interior+obs → output), like the founders."""
    a = sub.arena
    perc = sorted(_gene_ids(a, PERCEPTION))
    obs_ids = torch.tensor(perc[-OBS_DIM:], dtype=torch.long)
    interior = torch.tensor(_gene_ids(a, CREDIT_ELIGIBLE, exclude=(OUTPUT, MIRROR)),
                            dtype=torch.long)
    out_ids = torch.tensor(_gene_ids(a, OUTPUT), dtype=torch.long)
    new = [add_mirror_cell(a, interior, obs_ids, out_ids) for _ in range(n)]
    a.rank_dirty = True
    sub.compile()
    return new


def consolidate(sub, cell_ids):
    """Lock a learned skill: protect every edge incident to these mirror cells
    (KIBRA edge_protected — zero_dormant_grads already refuses to update them)."""
    a = sub.arena
    ids = set(int(c) for c in cell_ids)
    src = a.edge_src[: a.edge_cursor]
    dst = a.edge_dst[: a.edge_cursor]
    for e in range(a.edge_cursor):
        if int(src[e].item()) in ids or int(dst[e].item()) in ids:
            a.edge_protected[e] = True


def freeze_grads(sub, cell_ids):
    """Belt-and-suspenders: zero grads on locked cells' edges AND biases."""
    if not cell_ids:
        return
    a = sub.arena
    ids = set(int(c) for c in cell_ids)
    if a.bias.grad is not None:
        for c in ids:
            a.bias.grad[c] = 0.0
    if a.edge_weight.grad is not None and a.edge_cursor > 0:
        src = a.edge_src[: a.edge_cursor]
        dst = a.edge_dst[: a.edge_cursor]
        mask = torch.tensor([(int(src[e].item()) in ids or int(dst[e].item()) in ids)
                             for e in range(a.edge_cursor)])
        a.edge_weight.grad[: a.edge_cursor][mask] = 0.0


# ----------------------------------------------------------------------
# DAgger chapter (resume an existing sub; optionally freeze locked cells)
# ----------------------------------------------------------------------
def dagger_chapter(sub, master_fn, *, seed, episodes, frozen_cells=(),
                   gamma=0.95, lr=2e-3, batch=64, imit_w=0.5, max_steps=300):
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    a = sub.arena
    buf = deque(maxlen=20000)
    # seed demos
    demos = deque(maxlen=20000)
    for ep in range(20):
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
                freeze_grads(sub, frozen_cells)             # lock prior skills
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads()                    # also honours edge_protected
                opt.step()
    return sub


def _eval(sub, label):
    return evaluate(lambda w, p: int(sub(_solo(p.unsqueeze(0)))[0].argmax()),
                    label, seeds=40)


# ----------------------------------------------------------------------
# The two arms
# ----------------------------------------------------------------------
def run_plain(seed):
    sub = train_solo(seed, 120, n_mirror=16)
    sub = dagger_chapter(sub, MASTERS["fire"], seed=seed, episodes=200)
    _, fc, ff = _eval(sub, f"  PLAIN seed{seed} after-FIRE ")
    sub = dagger_chapter(sub, MASTERS["water"], seed=seed, episodes=200)
    _, wc, wf = _eval(sub, f"  PLAIN seed{seed} after-WATER")
    return ff, fc, wf, wc


def run_locked(seed):
    sub = train_solo(seed, 120, n_mirror=8)
    sub = dagger_chapter(sub, MASTERS["fire"], seed=seed, episodes=200)
    _, fc, ff = _eval(sub, f"  LOCKED seed{seed} after-FIRE ")
    fire_cells = mirror_ids(sub).tolist()             # Numa-validated by the eval above
    consolidate(sub, fire_cells)                      # lock fire
    grow_mirror(sub, 8)                               # fresh capacity for water
    sub = dagger_chapter(sub, MASTERS["water"], seed=seed, episodes=200,
                         frozen_cells=fire_cells)     # water trains only new cells
    _, wc, wf = _eval(sub, f"  LOCKED seed{seed} after-WATER")
    return ff, fc, wf, wc


def main():
    seeds = [2, 7, 13]
    _p("=== CONSOLIDATE-AND-LOCK: can fire survive learning water? (n=3) ===")
    _p("    both arms end at 16 mirror cells — only the lock differs\n")
    arms = {}
    for name, runner in (("PLAIN", run_plain), ("LOCKED", run_locked)):
        _p(f"\n########## {name} ##########")
        rows = [runner(s) for s in seeds]
        # rows: (fire_occ, fire_causes, water_occ, water_causes)
        f_occ = statistics.mean(r[0] for r in rows)
        w_occ = statistics.mean(r[2] for r in rows)
        f_cold = statistics.mean(r[1]["cold"] for r in rows)
        w_cold = statistics.mean(r[3]["cold"] for r in rows)
        f_thirst = statistics.mean(r[1]["thirst"] for r in rows)
        w_thirst = statistics.mean(r[3]["thirst"] for r in rows)
        arms[name] = dict(f_occ=f_occ, w_occ=w_occ, f_cold=f_cold, w_cold=w_cold,
                          f_thirst=f_thirst, w_thirst=w_thirst)

    _p("\n=== SUMMARY (means, n=3) ===")
    _p(f"  {'arm':>7s} | {'fire-occ: after-fire→water':>26s} | "
       f"{'cold: fire→water':>17s} | {'thirst: fire→water':>19s}")
    for name, m in arms.items():
        ret = "RETAINED" if m["w_occ"] >= m["f_occ"] - 2 else "forgot"
        learned = "learned" if m["w_thirst"] < m["f_thirst"] - 2 else "no"
        _p(f"  {name:>7s} | {m['f_occ']:>9.1f} → {m['w_occ']:<9.1f} {ret:>6s} | "
           f"{m['f_cold']:>5.1f} → {m['w_cold']:<5.1f} | "
           f"{m['f_thirst']:>6.1f} → {m['w_thirst']:<6.1f} {learned}")
    _p("\n  win: LOCKED keeps fire-occ (RETAINED) AND drops thirst (water learned).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
