"""One watchable test case (2026-08-15): nested trioron vs flat DQN, same map.

Trains (or loads from runs/duel/) the consequence-taught trioron router over the
5 leaf donors and a flat 2x128 DQN — the two headline agents of s048 — then runs
both on the SAME unseen random map and renders a side-by-side GIF:
left = nested organism (with its live leaf choice shown), right = DQN, drive
bars under each, until both are dead (a dead panel freezes, greyed).

Usage:
    python3 experiments/world/watch_duel.py                 # train/load + render
    python3 experiments/world/watch_duel.py --map-seed 7    # another map
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.fire_taming import evaluate            # tamed physics
from experiments.world.tile_world import TileWorld, ACTIONS
from experiments.world.mirror_cells import _solo
from experiments.world.vocabulary import PRIM_ORDER
from experiments.world.router_trioron import (
    build_router_sub, load_leaves, train_router_td,
)
from experiments.world.dqn_baseline import QNet, train_dqn

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"
DUEL = RUNS / "duel"

TILE_COLORS = ["#161b22", "#3fb950", "#58a6ff", "#f85149", "#bc8cff", "#db61a2"]
_CMAP = ListedColormap(TILE_COLORS)
ACT_NAMES = ["N", "S", "E", "W", "eat", "rest"]


# ----------------------------------------------------------------------
# Get the two agents (train once, checkpoint, reload thereafter)
# ----------------------------------------------------------------------
def get_agents(seed=0, episodes=300):
    DUEL.mkdir(parents=True, exist_ok=True)
    donors = load_leaves()

    r_ck = DUEL / "router_td.pt"
    router = build_router_sub(seed)
    if r_ck.exists():
        state = torch.load(r_ck)
        with torch.no_grad():
            router.arena.bias.copy_(state["bias"])
            router.arena.edge_weight.copy_(state["edge_weight"])
            # branch_alpha is TRAINABLE on the quad-dendrite router;
            # omitting it lobotomizes the reload (s049: 89.7 vs 163.2)
            if "branch_alpha" in state:
                router.arena.branch_alpha.copy_(state["branch_alpha"])
            else:
                print("WARN: legacy ckpt without branch_alpha — "
                      "router behavior will NOT match training")
        print(f"loaded router <- {r_ck}")
    else:
        print(f"training TD trioron router ({episodes} eps)...", flush=True)
        router, _ = train_router_td(seed, donors, episodes=episodes)
        torch.save({"bias": router.arena.bias.detach().clone(),
                    "edge_weight": router.arena.edge_weight.detach().clone(),
                    "branch_alpha":
                        router.arena.branch_alpha.detach().clone()},
                   r_ck)
        print(f"saved router -> {r_ck}")

    d_ck = DUEL / "dqn.pt"
    if d_ck.exists():
        q = QNet()
        q.load_state_dict(torch.load(d_ck))
        print(f"loaded DQN <- {d_ck}")
    else:
        print(f"training DQN ({episodes} eps)...", flush=True)
        q, _ = train_dqn(seed, episodes=episodes)
        torch.save(q.state_dict(), d_ck)
        print(f"saved DQN -> {d_ck}")
    return router, donors, q


# ----------------------------------------------------------------------
# Record one episode per agent on the same map seed
# ----------------------------------------------------------------------
@torch.no_grad()
def record(act_fn, map_seed, max_steps=300):
    """act_fn(w, p) -> (action, note). Returns list of per-step snapshots."""
    w = TileWorld(seed=map_seed, max_steps=max_steps)
    p = w.percept()
    frames = [snap(w, "start", "-")]
    done = False
    while not done:
        a, note = act_fn(w, p)
        p, _, done, info = w.step(a)
        frames.append(snap(w, ACT_NAMES[a], note))
    return frames


def snap(w, act, note):
    return dict(grid=w.grid.clone().numpy(), px=w.px, py=w.py,
                pred=tuple(w.pred), night=w.is_night, t=w.t, alive=w.alive,
                drives=(w.energy, w.thirst, w.integrity, w.temp),
                act=act, note=note)


# ----------------------------------------------------------------------
# Side-by-side rendering
# ----------------------------------------------------------------------
# tile glyphs: shape + color both encode the type (readable in gray too)
_TILE_MARKS = {1: ("o", "#3fb950", 110),   # FOOD   — circle, green
               2: ("v", "#58a6ff", 110),   # WATER  — down-triangle, blue
               3: ("^", "#f85149", 150),   # FIRE   — up-triangle, red
               4: ("D", "#bc8cff", 80),    # BERRY  — diamond, purple
               5: ("P", "#db61a2", 110)}   # POISON — plus, pink


def _panel(axg, axd, s, title, dead_now):
    n = s["grid"].shape[0]
    axg.set_facecolor("#0a0d12" if s["night"] else "#161b22")
    axg.set_xlim(-0.5, n - 0.5); axg.set_ylim(n - 0.5, -0.5)
    axg.set_aspect("equal")
    for k in range(n + 1):                       # faint grid lines
        axg.axhline(k - 0.5, color="#21262d", lw=0.5, zorder=0)
        axg.axvline(k - 0.5, color="#21262d", lw=0.5, zorder=0)
    alpha = 0.35 if dead_now else 1.0
    for t, (mark, col, size) in _TILE_MARKS.items():
        ys, xs = np.where(s["grid"] == t)
        if len(xs):
            axg.scatter(xs, ys, marker=mark, color=col, s=size,
                        alpha=alpha, zorder=2)
    face = "#666666" if dead_now else "white"
    axg.scatter([s["px"]], [s["py"]], s=260, marker="o", facecolors=face,
                edgecolors="black", linewidths=2, zorder=3)
    axg.scatter([s["pred"][0]], [s["pred"][1]], s=200, marker="X",
                color="#ff4d4d", edgecolors="black", linewidths=1.5, zorder=3)
    status = "DEAD" if dead_now else f"act={s['act']}  [{s['note']}]"
    axg.set_title(f"{title}\n t={s['t']}  {'night' if s['night'] else 'day'}  "
                  f"{status}", color="#c9d1d9", fontsize=10)
    axg.set_xticks([]); axg.set_yticks([])

    names = ["energy", "thirst", "integrity", "temp"]
    cols = ["#3fb950", "#58a6ff", "#f0883e", "#bc8cff"]
    axd.set_facecolor("#0d1117")
    for i, (name, val, col) in enumerate(zip(names, s["drives"], cols)):
        y = len(names) - 1 - i
        axd.barh(y, 1.0, color="#21262d", height=0.55)
        axd.barh(y, max(0.0, min(1.0, val)),
                 color="#444444" if dead_now else col, height=0.55)
        axd.text(0.02, y, f"{name} {val:.2f}", va="center", fontsize=8,
                 color="#c9d1d9")
    axd.axvline(0.5, color="#8b949e", ls=":", lw=0.8)
    axd.set_xlim(0, 1); axd.set_ylim(-0.6, len(names) - 0.4)
    axd.set_xticks([]); axd.set_yticks([])
    for sp in axd.spines.values():
        sp.set_visible(False)


def render(frames_a, frames_b, out_path, fps=10,
           title_a="NESTED TRIORON (router + 5 leaves)",
           title_b="FLAT DQN (2x128 MLP)"):
    n = max(len(frames_a), len(frames_b))
    imgs = []
    for t in range(n):
        sa = frames_a[min(t, len(frames_a) - 1)]
        sb = frames_b[min(t, len(frames_b) - 1)]
        dead_a = t >= len(frames_a) - 1 and not sa["alive"]
        dead_b = t >= len(frames_b) - 1 and not sb["alive"]
        fig = plt.figure(figsize=(11, 6.4))
        fig.patch.set_facecolor("#0d1117")
        gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.1], hspace=0.16,
                              wspace=0.08)
        _panel(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), sa,
               title_a, dead_a)
        _panel(fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1]), sb,
               title_b, dead_b)
        legend = [("● food", "#3fb950"), ("▼ water", "#58a6ff"),
                  ("▲ fire", "#f85149"), ("◆ berry", "#bc8cff"),
                  ("✚ poison", "#db61a2"), ("○ agent", "#e6edf3"),
                  ("✖ predator", "#ff4d4d")]
        x = 0.5 - 0.095 * len(legend) / 2
        for txt, col in legend:
            fig.text(x, 0.015, txt, ha="left", color=col, fontsize=9)
            x += 0.095
        fig.canvas.draw()
        imgs.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, imgs, fps=fps, loop=0)
    return len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-seed", type=int, default=987654,
                    help="unseen world seed (train used 0..299000 ranges)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    router, donors, q = get_agents(args.seed, args.episodes)

    def act_nest(w, p):
        i = int(router(p.unsqueeze(0))[0].argmax())
        leaf = donors[PRIM_ORDER[i]]
        return (int(leaf(_solo(p.unsqueeze(0)))[0].argmax()),
                PRIM_ORDER[i])

    def act_dqn(w, p):
        return int(q(p.unsqueeze(0))[0].argmax()), "-"

    # sanity: both agents' held-out means, so the duel map is in context
    evaluate(lambda w, p: act_nest(w, p)[0], "nest (held-out x40)", seeds=40)
    evaluate(lambda w, p: act_dqn(w, p)[0], "dqn  (held-out x40)", seeds=40)

    fa = record(act_nest, args.map_seed)
    fb = record(act_dqn, args.map_seed)
    print(f"duel on map {args.map_seed}: nest survived {fa[-1]['t']} steps, "
          f"DQN survived {fb[-1]['t']} steps", flush=True)

    out = RUNS / f"duel_map{args.map_seed}.gif"
    nf = render(fa, fb, out, fps=args.fps)
    print(f"rendered -> {out}  ({nf} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
