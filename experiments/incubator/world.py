"""Vee's world, scripted: the simulator that drives her environment sense
(docs/design/vee_development.md §7.1).  Fully simulated and reproducible; it
never reads the real machine.

One moment = 15 simulated minutes (96 a day).  Hidden states change with
realistic persistence and are returned separately — they are for the gates
and must NEVER enter her input:
    night      the clock (18:00-06:00)
    busy       the machine: CPU, processes and traffic move together
    full       memory nearly full
    elsewhere  she is on another machine for a while (rare)
    window     the window to the outside is open

    w = World(seed);  raw, hidden = w.step();  x = environment.vector(raw)

Stage 2 (fetal, plan §8.2): ``step(action)`` lets her kick.  The world answers in the
readings she already has; ``raw["held"]`` and ``raw["marks"]`` feed her echo block.
    still    nothing
    work     CPU busy now and load 1 min rise (not processes, not traffic); fades next moment
    hold     takes a little memory (her own memory, RAM used); until let go; at most 4
    let go   releases one hold
    mark     leaves a mark; stays; at most 8
    wipe     removes one mark
An action that cannot happen (let go with nothing held, …) does nothing.  Actions use no
random numbers, so the world's own course is the same with or without her.
"""
from __future__ import annotations
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

PER_DAY = 96
ACTIONS = ("still", "work", "hold", "let go", "mark", "wipe")
MAX_HELD, MAX_MARKS = 4, 8
HIDDEN = ("night", "busy", "full", "elsewhere", "window")


class World:
    held = marks = 0            # defaults for a world saved before she could kick (the embryo's)
    worked = 0.0

    def __init__(self, seed: int = 0, noise: float = 0.05):
        self.r, self.noise, self.t = np.random.default_rng(seed), noise, 0
        self.busy = self.full = self.elsewhere = self.window = self.rain = self.someone = False
        self.load5 = 0.1
        self.held = self.marks = 0
        self.worked = 0.0                                         # what is left of her last burst

    def _flip(self, state: bool, mean_on: float, mean_off: float) -> bool:
        """Markov switch with mean dwell times in moments."""
        return (not state) if self.r.random() < 1 / (mean_on if state else mean_off) else state

    def can(self, action: int) -> bool:
        a = ACTIONS[action]
        return {"hold": self.held < MAX_HELD, "let go": self.held > 0, "mark": self.marks < MAX_MARKS,
                "wipe": self.marks > 0}.get(a, True)

    def step(self, action: int = 0) -> tuple[dict, dict]:
        r, n = self.r, lambda v, s=1.0: float(v + self.noise * s * self.r.standard_normal())
        hour = (self.t % PER_DAY) / 4
        night = hour < 6 or hour >= 18
        self.busy = self._flip(self.busy, 6, 14)                  # ~1.5 h busy, ~3.5 h quiet
        self.full = self._flip(self.full, 30, 50)
        self.elsewhere = self._flip(self.elsewhere, 2 * PER_DAY, 10 * PER_DAY)
        self.window = self._flip(self.window, 8, 24)
        self.rain = self._flip(self.rain, 12, 60)
        self.someone = self._flip(self.someone, 10, 40 if not night else 200)
        light = max(0.0, math.sin(math.pi * (hour - 6) / 12))
        load = 0.9 if self.busy else 0.1
        self.load5 += 0.3 * (load - self.load5)
        w = self.window
        raw = {
            "hour": hour, "weekday": (self.t // PER_DAY) % 7, "awake_hours": self.t / 4,
            "ram_used": min(max(n(0.85 if self.full else 0.30), 0), 1), "swap_used": max(n(0.3 if self.full else 0.0), 0),
            "load1": max(n(load, 2), 0), "load5": max(n(self.load5, 2), 0), "busy": min(max(n(0.85 if self.busy else 0.05, 2), 0), 1),
            "running": min(max(n(0.6 if self.busy else 0.1, 2), 0), 1), "alive": max(n(190 if self.busy else 150, 200), 1),
            "disk_free": 0.9 - 1e-5 * self.t,
            "links": 1, "route": True, "traffic_in": max(n(2e5 if self.busy else 2e3, 2e4), 0), "traffic_out": max(n(5e4 if self.busy else 5e2, 5e3), 0),
            "machine": "elsewhere" if self.elsewhere else "home", "address": "10.9.9.9" if self.elsewhere else "10.1.1.1",
            "own_mb": max(n(300, 400), 1), "reflex": True,
            "daylight": n(light) if w else None, "temperature_c": n(22 + 8 * light, 20) if w else None,
            "rain": self.rain if w else None, "someone_here": self.someone if w else None,
        }
        a = ACTIONS[action] if self.can(action) else "still"
        self.held += (a == "hold") - (a == "let go")
        self.marks += (a == "mark") - (a == "wipe")
        self.worked = 0.5 if a == "work" else 0.4 * self.worked       # a burst, then it fades
        if self.worked > 0.05:                                        # never above what a busy world gives
            raw["busy"] = min(raw["busy"] + self.worked, 0.85); raw["load1"] = min(raw["load1"] + self.worked, 0.9)
        raw["ram_used"] = min(raw["ram_used"] + 0.05 * self.held, 1.0)
        raw["own_mb"] += 300 * self.held
        raw["held"], raw["marks"] = self.held, self.marks
        hidden = {"night": night, "busy": self.busy, "full": self.full, "elsewhere": self.elsewhere, "window": w}
        self.t += 1
        return raw, hidden
