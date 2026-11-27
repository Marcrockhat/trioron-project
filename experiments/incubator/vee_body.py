"""The body's side of the world (plan §6): world readings placed into the layout, the pools
senses form on, and the one-time fitting of the language sense.

The body itself lives in ``trioron/organs/`` since s072 (spec §9.17): ``layout.py`` (the 148
numbers), ``home.py`` (HOME, the stage file, the RAM ceiling), ``body.py`` (``Body``).  They
are re-exported here under their old names, so the drivers and every body saved as
``vee_body.Body`` read as before.

The language sense is fitted ONCE (Wikipedia sentence readings + stage-2 question readings,
§6.2), stored in HOME/sense.pt and never refitted.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import environment
from world import PER_DAY, World
from trioron.organs.blur import Blur
from trioron.organs.body import PATIENCE_BASE, PATIENCE_PER, RECENT, WINDOW, Body
from trioron.organs.home import HOME, OUT, RAM_CEILING, ram_used, require_stage, stage
from trioron.organs.layout import (LANG, LANG_SCALE, LAYOUT, MATRON, NOBODY, RES, ROCKY, WIDTH,
                                   can_be_silent, changes)
from trioron.senses.hidden_state import HiddenStateSense

# ---- the language sense (germline) -------------------------------------------
def wiki_states() -> torch.Tensor:
    z = np.load(f"{OUT}/wiki_sent.npz")
    return torch.cat([torch.tensor(z[p].astype(np.float32)) for p in ("h_last", "h_mean")], 1)


def language_sense() -> HiddenStateSense:
    """The fixed sense.  Fitted once on sentences + questions, then only loaded."""
    p = f"{HOME}/sense.pt"
    if os.path.exists(p):
        return torch.load(p, weights_only=False)
    from stage2_memory import load
    sense = HiddenStateSense(torch.cat([wiki_states(), load("both")[0]]), LANG, dup=False)
    os.makedirs(HOME, exist_ok=True)
    torch.save(sense, p)
    return sense


# ---- one moment ----------------------------------------------------------------
def moment(lang: torch.Tensor | None, raw: dict, presence=NOBODY, action: int | None = None) -> torch.Tensor:
    """Her input at one moment.  ``lang`` = sensed text (64, unscaled) or None; ``raw`` = environment
    readings; ``action`` = what she did in the moment just past (None = the echo is still silent)."""
    x = torch.zeros(WIDTH)
    if lang is not None: x[:LANG] = LANG_SCALE * lang
    x[64:97] = torch.tensor(environment.vector(raw))
    x[97:100] = torch.tensor(presence)
    if action is not None:
        x[100:106] = -1.0; x[100 + action] = 1.0
        x[106] = 2 * raw.get("held", 0) / 4 - 1; x[107] = 2 * raw.get("marks", 0) / 8 - 1
    return x


def kin_pool(seed: int, n: int = 10 * WINDOW, corridor: float = 0.0) -> torch.Tensor:
    """What the kin organ's senses form on: a stretch of world in which kicks fire in half the
    moments and the world's switches flip often (spontaneous activity, as at genesis)."""
    from world import ACTIONS
    w, g = World(600 + seed, corridor=corridor), np.random.default_rng(3600 + seed)
    rows = []
    for _ in range(n + 1):
        if g.random() < 0.2: w.busy, w.full, w.window = (bool(g.random() < 0.5) for _ in range(3))
        ok = [a for a in range(1, len(ACTIONS)) if w.can(a)]
        a = int(g.choice(ok)) if g.random() < 0.5 else 0
        rows.append(moment(None, w.step(a)[0], action=a))
    X = torch.stack(rows)
    return changes(X[1:], X[0])


def genesis_pool(seed: int, ambient: torch.Tensor, n: int = 10 * WINDOW, p_text: float = 0.5, corridor: float = 0.0) -> torch.Tensor:
    """What her senses form on: every state of her world toggled freely, any presence,
    spontaneous activity on the reserved channels (they fall silent once she lives)."""
    w, g = World(500 + seed, corridor=corridor), np.random.default_rng(3000 + seed)
    rows = []
    for _ in range(n):
        w.busy, w.full, w.elsewhere, w.window = (bool(g.random() < 0.5) for _ in range(4))
        w.t = int(g.integers(PER_DAY * 7))
        lang = ambient[int(g.integers(len(ambient)))] if g.random() < p_text else None
        x = moment(lang, w.step()[0], g.choice([-1.0, 1.0], 3).tolist())
        x[WIDTH - RES:] = torch.tensor(g.standard_normal(RES)).float()
        rows.append(x)
    return torch.stack(rows)
