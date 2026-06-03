"""Mini-Minecraft tile world — an open survival world for the organism.

The "alive" test (per the discipline established 2026-06-01): an embodied,
explorable, persistent world with INDEPENDENT physics (we design the dynamics,
never the solution), NO labels (the organism learns from drive-consequence),
honest BASELINES, and "alive" = survival over time (not accuracy).

This module is the WORLD only — physics + percepts + a random and a reactive
baseline. The organism (predict-then-act over the trioron/satellite/quad
substrate) plugs in separately. Establishing baselines first is deliberate:
"alive" only means something measured against random and reactive floors.

World (Minecraft-like, state-vector percepts — NOT pixels, per the substrate's
pixel dead end):
  - Grid of tiles: EMPTY, FOOD, WATER, FIRE (warmth), BERRY, POISON.
  - Drives (homeostatic, decay each step): energy, thirst, temperature, integrity.
  - Day/night cycle = the wake/dream phase signal; night is colder + a predator
    roams (moves toward the organism, damages integrity on contact).
  - Actions: move N/S/E/W, consume (eat/drink the current tile), rest.
  - Death: any drive leaves its safe band → episode ends. Survival = steps alive.

Relational hook: BERRY (safe, +energy) and POISON (−integrity) share their
visible features and differ only by a CONTEXT bit (e.g. smell) — so telling
them apart needs comparison, not just perception.
Temporal hook: drives decay, the predator moves, night recurs — surviving needs
memory + anticipation, not pure reaction.
"""
from __future__ import annotations

import torch

EMPTY, FOOD, WATER, FIRE, BERRY, POISON = range(6)
N_TILE = 6
ACTIONS = ["N", "S", "E", "W", "consume", "rest"]
N_ACTION = len(ACTIONS)
_DXY = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}


class TileWorld:
    # Tunable physics (class-level so an experiment can soften them once without
    # threading kwargs through every training helper). Defaults = the canonical
    # difficulty-reduced world; do NOT change these — override per experiment,
    # e.g. fire_taming sets WARM_RATE=0.04, TEMP_LOW=0.02, TEMP_HIGH=0.99.
    WARM_RATE = 0.15      # temp gain/step when near fire (vs ~0.01 cooling)
    TEMP_LOW = 0.05       # lethal cold threshold
    TEMP_HIGH = 0.98      # lethal heat threshold

    def __init__(self, size=12, day_len=20, max_steps=300, seed=0,
                 warm_rate=None, temp_low=None, temp_high=None):
        self.warm_rate = self.WARM_RATE if warm_rate is None else warm_rate
        self.temp_low = self.TEMP_LOW if temp_low is None else temp_low
        self.temp_high = self.TEMP_HIGH if temp_high is None else temp_high
        self.size = size
        self.day_len = day_len
        self.max_steps = max_steps
        self.g = torch.Generator().manual_seed(seed)
        self.reset()

    # ── world setup ───────────────────────────────────────────────
    def reset(self):
        s = self.size
        self.grid = torch.zeros(s, s, dtype=torch.long)
        # scatter resources (fixed physics, independent of any solution)
        def scatter(tile, n):
            for _ in range(n):
                x = int(torch.randint(0, s, (1,), generator=self.g))
                y = int(torch.randint(0, s, (1,), generator=self.g))
                self.grid[y, x] = tile
        scatter(FOOD, s); scatter(WATER, s); scatter(FIRE, max(2, s // 3))
        scatter(BERRY, s); scatter(POISON, s)
        # poison carries a hidden context bit that BERRY lacks (smell) — the
        # only signal distinguishing the look-alikes. stored per-cell.
        self.smell = (self.grid == POISON).float()  # 1 on poison, 0 elsewhere
        self.px = s // 2; self.py = s // 2
        self.pred = [int(torch.randint(0, s, (1,), generator=self.g)),
                     int(torch.randint(0, s, (1,), generator=self.g))]
        # drives in [0,1]; safe band roughly (0, 1] for energy/thirst/integrity,
        # temperature target ~0.5 (both extremes bad).
        self.energy = 1.0; self.thirst = 1.0; self.integrity = 1.0; self.temp = 0.5
        self.t = 0
        self.alive = True
        return self.percept()

    @property
    def is_night(self):
        return (self.t // self.day_len) % 2 == 1

    def _scent(self, tile):
        """Normalized (dx,dy) toward the nearest cell of `tile` (toroidal),
        scaled by closeness (stronger when near). (0,0) if none exist.
        A distance sense — the organism must still learn to follow it."""
        s = self.size
        ys, xs = torch.where(self.grid == tile)
        if xs.numel() == 0:
            return torch.zeros(2)
        dx = ((xs - self.px + s // 2) % s) - s // 2
        dy = ((ys - self.py + s // 2) % s) - s // 2
        d2 = dx * dx + dy * dy
        i = int(torch.argmin(d2))
        dist = max(1.0, float(d2[i]) ** 0.5)
        return torch.tensor([float(dx[i]) / dist, float(dy[i]) / dist]) / dist

    # ── percept (state vector, no pixels) ─────────────────────────
    def percept(self):
        """Local 3×3 view (63) + scent gradients to food/water/fire (6) +
        interoception drives (4) + night flag (1) = 74-d. No pixels."""
        s = self.size
        feats = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x = (self.px + dx) % s
                y = (self.py + dy) % s
                oh = torch.zeros(N_TILE)
                oh[int(self.grid[y, x])] = 1.0
                feats.append(torch.cat([oh, self.smell[y, x].view(1)]))
        local = torch.cat(feats)                       # 63
        scent = torch.cat([self._scent(FOOD), self._scent(WATER), self._scent(FIRE)])
        intero = torch.tensor([self.energy, self.thirst, self.integrity, self.temp])
        phase = torch.tensor([1.0 if self.is_night else 0.0])
        return torch.cat([local, scent, intero, phase])  # 74

    # ── dynamics ──────────────────────────────────────────────────
    def step(self, action_idx):
        """Apply action, drive decay, predator, day/night. Returns
        (percept, reward_drive_delta, done, info). reward = total drive change
        (the consequence signal; no external label)."""
        if not self.alive:
            return self.percept(), 0.0, True, {}
        a = ACTIONS[action_idx]
        s = self.size
        before = self._drive_sum()

        if a in _DXY:
            dx, dy = _DXY[a]
            self.px = (self.px + dx) % s; self.py = (self.py + dy) % s
        elif a == "consume":
            tile = int(self.grid[self.py, self.px])
            if tile == FOOD:
                self.energy = min(1.0, self.energy + 0.5); self.grid[self.py, self.px] = EMPTY
            elif tile == WATER:
                self.thirst = min(1.0, self.thirst + 0.5)
            elif tile == BERRY:
                self.energy = min(1.0, self.energy + 0.25); self.grid[self.py, self.px] = EMPTY
            elif tile == POISON:
                self.integrity = max(0.0, self.integrity - 0.5); self.grid[self.py, self.px] = EMPTY
        elif a == "rest":
            pass

        # warmth from being on/adjacent to FIRE; else drift cold (more at night)
        near_fire = any(int(self.grid[(self.py + dy) % s, (self.px + dx) % s]) == FIRE
                        for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        if near_fire:
            self.temp = min(1.0, self.temp + self.warm_rate)
            if int(self.grid[self.py, self.px]) == FIRE:
                self.integrity = max(0.0, self.integrity - 0.08)   # too close burns
        else:
            self.temp = max(0.0, self.temp - (0.015 if self.is_night else 0.008))

        # metabolic decay (DIFFICULTY-REDUCED 2026-06-01: slower clocks give
        # time to ACT, so survival rewards skillful seeking — but you still
        # starve/dehydrate if you don't seek. Widens the skill band.)
        self.energy = max(0.0, self.energy - 0.012)
        self.thirst = max(0.0, self.thirst - 0.015)

        # predator: a MINOR nuisance (slow, light hit) — deliberately not the
        # dominant killer, so the binding survival challenge is resource-seeking
        # (a skill random can't fake), not a predator-luck lottery.
        if self.t % 4 == 0:
            self.pred[0] = (self.pred[0] + (1 if self.px > self.pred[0] else -1)) % s
            self.pred[1] = (self.pred[1] + (1 if self.py > self.pred[1] else -1)) % s
        if self.pred[0] == self.px and self.pred[1] == self.py:
            self.integrity = max(0.0, self.integrity - 0.1)

        self.t += 1
        # death: any drive depleted, or temperature out of the safe band
        if (self.energy <= 0 or self.thirst <= 0 or self.integrity <= 0
                or self.temp <= self.temp_low or self.temp >= self.temp_high):
            self.alive = False
        done = (not self.alive) or self.t >= self.max_steps
        reward = self._drive_sum() - before
        return self.percept(), reward, done, {"t": self.t, "alive": self.alive}

    def _drive_sum(self):
        # distance of temp from its 0.5 target counts as a drive too
        return self.energy + self.thirst + self.integrity - 2 * abs(self.temp - 0.5)


# ── baselines ─────────────────────────────────────────────────────
def run_random(seed, episodes=20):
    g = torch.Generator().manual_seed(seed + 7)
    lengths = []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 100 + ep)
        done = False
        while not done:
            a = int(torch.randint(0, N_ACTION, (1,), generator=g))
            _, _, done, info = w.step(a)
        lengths.append(info["t"])
    return lengths


def run_reactive(seed, episodes=20):
    """Competent hand-coded floor: address the most urgent drive — flee a near
    predator, warm at fire when cold, drink/eat when low — using the 3×3 view.
    No learning, no memory; this is the skill floor the organism must beat."""
    lengths = []
    for ep in range(episodes):
        w = TileWorld(seed=seed * 100 + ep)
        done = False
        while not done:
            s = w.size
            def find(tile):  # direction to a tile in the 3×3 view, or None
                for ai, mv in _DXY.items():
                    if int(w.grid[(w.py + mv[1]) % s, (w.px + mv[0]) % s]) == tile:
                        return ACTIONS.index(ai)
                return None
            def toward(tile):  # move along the dominant axis of the scent
                d = w._scent(tile)
                if float(d.abs().sum()) == 0:
                    return None
                if abs(float(d[0])) >= abs(float(d[1])):
                    return ACTIONS.index("E" if d[0] > 0 else "W")
                return ACTIONS.index("S" if d[1] > 0 else "N")
            here = int(w.grid[w.py, w.px])
            pred_adj = abs((w.pred[0] - w.px)) <= 1 and abs((w.pred[1] - w.py)) <= 1
            a = None
            # 1) flee an adjacent predator
            if pred_adj and w.integrity < 0.8:
                ax = -1 if w.pred[0] > w.px else 1
                a = ACTIONS.index("E" if ax > 0 else "W")
            # 2) warm up when cold (head to the nearest fire)
            elif w.temp < 0.3:
                a = toward(FIRE)
            # 3) drink / eat on the spot
            elif here == WATER and w.thirst < 0.7:
                a = ACTIONS.index("consume")
            elif here in (FOOD, BERRY) and w.energy < 0.7:
                a = ACTIONS.index("consume")
            # 4) head to the most-needed resource (whole-map scent)
            if a is None:
                if w.thirst <= w.energy and w.thirst < 0.6:
                    a = toward(WATER)
                elif w.energy < 0.6:
                    a = toward(FOOD)
            if a is None:
                a = int(torch.randint(0, 4, (1,)))   # explore
            _, _, done, info = w.step(a)
        lengths.append(info["t"])
    return lengths


if __name__ == "__main__":
    import statistics
    print("Mini-Minecraft tile world — baseline survival (steps alive, max 300)")
    for name, fn in [("random", run_random), ("reactive", run_reactive)]:
        allL = []
        for seed in range(3):
            allL += fn(seed, episodes=20)
        print(f"  {name:>9s}: mean={statistics.mean(allL):.1f}  "
              f"median={statistics.median(allL):.0f}  "
              f"max={max(allL)}  (n={len(allL)} episodes)")
    print("\n(organism must beat these floors to count as 'alive'.)")
