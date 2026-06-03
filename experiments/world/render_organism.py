"""Render an organism — body (substrate anatomy) and behaviour (world GIF).

Two artifacts, both of a MIRROR organism (so the anatomy shows the new cell type):
  1. runs/organism_world.gif  — a greedy survival episode in the tile world:
     the 12×12 grid, the organism foraging, the predator, drive bars, day/night.
  2. runs/organism_anatomy.html — the substrate as an interactive graph
     (interior / MIRROR / output cells, weighted edges) via viz/export.py.
     MIRROR cells are gold-highlighted by post-processing the generated HTML
     (export.py is a do-not-touch carried file, so we patch the artifact, not it).

The organism is trained with pure solo TD self-RL (no teacher) — a competent
creature to watch, carrying the mirror cells for the anatomy view.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.world.tile_world import (
    TileWorld, N_ACTION, EMPTY, FOOD, WATER, FIRE, BERRY, POISON,
)
from experiments.world.mirror_cells import build_mirror, _solo, mirror_ids
from trioron.viz.snapshot import capture_full
from trioron.viz.export import export_html

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"

# tile palette (index = tile id)
TILE_COLORS = ["#161b22",  # EMPTY  (dark)
               "#3fb950",  # FOOD   (green)
               "#58a6ff",  # WATER  (blue)
               "#f85149",  # FIRE   (red)
               "#bc8cff",  # BERRY  (violet)
               "#db61a2"]  # POISON (magenta)
_CMAP = ListedColormap(TILE_COLORS)


# ----------------------------------------------------------------------
# Train a competent solo organism (TD self-RL, obs channel held at 0)
# ----------------------------------------------------------------------
def train_solo(seed, episodes, *, gamma=0.95, lr=3e-3, batch=64, max_steps=300,
               n_mirror=8):
    sub = build_mirror(seed, n_mirror=n_mirror)
    opt = torch.optim.Adam(sub.trainable_tensors(), lr=lr)
    buf = deque(maxlen=20000)
    g = torch.Generator().manual_seed(seed + 31)
    for ep in range(episodes):
        eps = 0.1 + 0.9 * max(0.0, 1 - ep / (0.7 * episodes))
        w = TileWorld(seed=seed * 1000 + ep, max_steps=max_steps)
        p = w.percept(); done = False
        while not done:
            if torch.rand(1, generator=g).item() < eps:
                a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            else:
                with torch.no_grad():
                    a = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
            p2, r, done, info = w.step(a)
            buf.append((p, a, r, p2, float(done))); p = p2
            if len(buf) >= batch:
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
                loss = torch.nn.functional.mse_loss(q, tgt)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(sub.trainable_tensors(), 1.0)
                sub.zero_dormant_grads(); opt.step()
    return sub


# ----------------------------------------------------------------------
# 1. Behaviour — greedy episode → GIF
# ----------------------------------------------------------------------
def _frame(w, step, action):
    fig, (axg, axd) = plt.subplots(
        1, 2, figsize=(8, 4.2), gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor("#0d1117")

    axg.imshow(w.grid.numpy(), cmap=_CMAP, vmin=0, vmax=5)
    axg.scatter([w.px], [w.py], s=300, marker="o",
                facecolors="white", edgecolors="black", linewidths=2, zorder=3)
    axg.scatter([w.pred[0]], [w.pred[1]], s=220, marker="X",
                color="#ff4d4d", edgecolors="black", linewidths=1.5, zorder=3)
    phase = "night" if w.is_night else "day"
    axg.set_title(f"t={step}  [{phase}]  action={['N','S','E','W','eat','rest'][action]}",
                  color="#c9d1d9", fontsize=11)
    axg.set_xticks([]); axg.set_yticks([])

    drives = [("energy", w.energy, "#3fb950"), ("thirst", w.thirst, "#58a6ff"),
              ("integrity", w.integrity, "#f0883e"), ("temp", w.temp, "#bc8cff")]
    axd.set_facecolor("#0d1117")
    for i, (name, val, col) in enumerate(drives):
        y = len(drives) - 1 - i
        axd.barh(y, 1.0, color="#21262d", height=0.55)
        axd.barh(y, max(0.0, min(1.0, val)), color=col, height=0.55)
        axd.text(0.02, y, f"{name} {val:.2f}", va="center", fontsize=10,
                 color="#c9d1d9")
    axd.axvline(0.5, color="#8b949e", ls=":", lw=0.8)   # temp target
    axd.set_xlim(0, 1); axd.set_ylim(-0.6, len(drives) - 0.4)
    axd.set_xticks([]); axd.set_yticks([])
    for sp in axd.spines.values():
        sp.set_visible(False)

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


@torch.no_grad()
def render_world_gif(sub, seed, out_path, max_steps=300, fps=12):
    w = TileWorld(seed=seed, max_steps=max_steps)
    p = w.percept(); done = False
    frames = [_frame(w, 0, 5)]
    while not done:
        a = int(sub(_solo(p.unsqueeze(0)))[0].argmax())
        p, _, done, info = w.step(a)
        frames.append(_frame(w, info["t"], a))
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return len(frames)


# ----------------------------------------------------------------------
# 2. Anatomy — substrate graph (MIRROR cells gold-highlighted)
# ----------------------------------------------------------------------
_MIRROR_RULE = "if (epi & (1 << 10)) return '#ffd33d'; // mirror cell\n  "


def render_anatomy(sub, out_html):
    snap_dir = RUNS / "_anatomy_snap"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for f in snap_dir.glob("*.json"):
        f.unlink()
    capture_full(sub.arena, task_index=0).write(snap_dir / "000.json")
    export_html(snap_dir, out_html, filter_perception=True)
    # inject a distinct colour for the new MIRROR cell type into the generated
    # artifact (we do NOT edit viz/export.py — a carried do-not-touch file).
    html = Path(out_html).read_text()
    anchor = "if (epi & (1 << 6)) return '#f0883e'; // output\n  "
    if anchor in html and "1 << 10" not in html:
        html = html.replace(anchor, anchor + _MIRROR_RULE)
        Path(out_html).write_text(html)
    return out_html


def main():
    RUNS.mkdir(exist_ok=True)
    print("training a solo mirror organism for the render (~250 eps)...")
    sub = train_solo(seed=0, episodes=250)
    mids = mirror_ids(sub)
    print(f"  organism: {sub.n_cells} cells ({len(mids)} mirror), {sub.n_edges} edges")

    gif = RUNS / "organism_world.gif"
    n = render_world_gif(sub, seed=7, out_path=gif)
    print(f"behaviour → {gif}  ({n} frames, survived {n-1} steps)")

    html = RUNS / "organism_anatomy.html"
    render_anatomy(sub, html)
    print(f"anatomy   → {html}  (open in a browser; MIRROR cells are gold)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
