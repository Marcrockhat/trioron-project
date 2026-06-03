"""Side-by-side render: solo apprentice (freezes) vs master-piloted (uses fire).

The honest visual of the fire-taming finding — the master-avatar channel lets the
oracle DRIVE the body to fire competence, but the organism doesn't internalise it
and freezes when alone. Left panel = the apprentice acting solo; right panel = the
same apprentice with the fire-oracle piloting through the observation channel.
Same world seed; the policies diverge from there.
"""
from __future__ import annotations

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

# importing fire_taming sets the tamed-fire physics on TileWorld
from experiments.world.fire_taming import (
    fire_oracle, collect_oracle_demos, TileWorld,
)
from experiments.world.mirror_cells import train_mirror_student, _solo, obs_onehot
from experiments.world.render_organism import TILE_COLORS

RUNS = Path(__file__).resolve().parent.parent.parent / "runs"
_CMAP = ListedColormap(TILE_COLORS)
_ACT = ["N", "S", "E", "W", "eat", "rest"]


@torch.no_grad()
def rollout(sub, seed, piloted, max_steps=300):
    """Record per-step world state. piloted=True → oracle drives via obs channel."""
    w = TileWorld(seed=seed, max_steps=max_steps)
    p = w.percept(); done = False
    frames = []
    def snap(step, act):
        frames.append((w.grid.clone(), w.px, w.py, list(w.pred),
                       w.temp, w.is_night, step, act, w.alive))
    snap(0, 5)
    while not done:
        if piloted:
            m = fire_oracle(w)
            x = torch.cat([p.unsqueeze(0), obs_onehot(torch.tensor([m]), 1)], dim=1)
        else:
            x = _solo(p.unsqueeze(0))
        a = int(sub(x)[0].argmax())
        p, _, done, info = w.step(a)
        snap(info["t"], a)
    return frames


def _panel(ax, st, title):
    grid, px, py, pred, temp, night, step, act, alive = st
    ax.imshow(grid.numpy(), cmap=_CMAP, vmin=0, vmax=5)
    ax.scatter([px], [py], s=240, marker="o", facecolors="white",
               edgecolors="black", linewidths=2, zorder=3)
    ax.scatter([pred[0]], [pred[1]], s=180, marker="X", color="#ff4d4d",
               edgecolors="black", linewidths=1.3, zorder=3)
    # temp bar under the title
    bar_c = "#bc8cff" if 0.1 < temp < 0.9 else "#ff4d4d"
    ax.add_patch(plt.Rectangle((0, -1.4), 11, 0.5, color="#21262d", clip_on=False))
    ax.add_patch(plt.Rectangle((0, -1.4), 11 * max(0.0, min(1.0, temp)), 0.5,
                               color=bar_c, clip_on=False))
    state = "FROZEN" if (not alive and temp <= 0.05) else ("dead" if not alive else "")
    ax.set_title(f"{title}\nt={step} temp={temp:.2f} {state}", color="#c9d1d9",
                 fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])


def render(sub, seed, out_path, fps=12):
    solo = rollout(sub, seed, piloted=False)
    pilot = rollout(sub, seed, piloted=True)
    n = max(len(solo), len(pilot))
    frames = []
    for i in range(n):
        s = solo[min(i, len(solo) - 1)]
        m = pilot[min(i, len(pilot) - 1)]
        fig, (axl, axr) = plt.subplots(1, 2, figsize=(9, 4.8))
        fig.patch.set_facecolor("#0d1117")
        _panel(axl, s, "SOLO apprentice (alone)")
        _panel(axr, m, "MASTER-piloted (oracle drives via obs)")
        fig.suptitle("Mirror-cell apprentice: pilots fine, hasn't learned it alone",
                     color="#58a6ff", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return len(solo) - 1, len(pilot) - 1, len(frames)


def main():
    RUNS.mkdir(exist_ok=True)
    print("training BC apprentice (250 ep) for the render...")
    demos = collect_oracle_demos(40, 99)
    _, sub = train_mirror_student(demos, seed=0, episodes=250)
    out = RUNS / "fire_taming_sidebyside.gif"
    solo_t, pilot_t, nf = render(sub, seed=7, out_path=out)
    print(f"solo survived {solo_t} steps, master-piloted {pilot_t} steps "
          f"({nf} frames) → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
