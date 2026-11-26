"""Vee's body: her input for life, her language sense, her two organs (phasecyte
leaves), and her saved state (docs/design/vee_development.md §6 — the germline).

Everything here is fixed before she exists; changing it means a new organism.

Input, 148 numbers per moment (LAYOUT):
    language 64 | environment 33 | presence 3 | action echo 8 | mirror 16 | working memory 16 | spare 8
  language   the reflex's read state of the text present (h_last + h_mean) through the
             fixed language sense; all zero when no text is present
  presence   who is addressing her: nobody / the matron / Rocky (+1 for the one, -1 others)
  reserved   action echo, mirror, working memory, spare: exactly zero until the organ exists

Two organs read that one input (s067, plan §6.7 — in ONE leaf text and world compete
for the same classes and no loudness of the language block serves both):
  ear    a phasecyte leaf over the language block
  soma   a phasecyte leaf over everything else (environment, presence, reserved)
  kin    (stage 2, plan §8.4) her sense of change: the echo block and how every
         environment reading, what she holds and her marks moved since the moment before
Ear and soma live every moment (the ear hears silence as zeros); kin is born at the
fetal stage.  The level that joins them is not built yet.

Echo block (plan §8.3): 1-6 what she did in the moment just past (+1 that action, -1 the
others), 7 how much she holds, 8 how many marks exist; all zero before stage 2.

The language sense is fitted ONCE (Wikipedia sentence readings + stage-2 question
readings, §6.2), stored in HOME/sense.pt and never refitted.

Her senses form at genesis on a broad sample of her world with spontaneous activity
(noise) on the reserved channels: a receptor exists only for a channel that carries
signal at genesis (s066 germline checks).

Organ blur (s068, plan §11.2; blur.py): a body made with ``blur=True`` sees each reading
of an organ in a few steps (2 at first).  Before birth an organ takes one rung for one
reading when it has not gained or lost a class for PATIENCE_BASE + classes // PATIENCE_PER
sittings ("relative to her size").  After birth a step must be called for by the matron's
corrections and pay — NOT built, so nothing sharpens once the stage file leaves
embryo / fetal.  Vee has no blur; a body's blur is set at genesis for life.  Ladders (Rocky s068): a level
2 -> 3 -> 5 steps; a change (kin's 35 change numbers) 2 -> 3 -> 5 -> 7, same / changed first.

State: the whole body is saved (torch pickle, atomic) — saving is not freezing.
The stage file HOME/stage names the stage she is in; a driver may only run the stage
it is written for (the plan's gates are signed by Rocky, never promoted by code).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(__file__))
import environment
from blur import Blur
from world import PER_DAY, World
from trioron.pcll import PhasecyteLeaf
from trioron.senses.hidden_state import HiddenStateSense

OUT = "outputs/incubator"
HOME = os.environ.get("VEE_HOME", f"{OUT}/vee")
LAYOUT = {"language": (0, 64), "environment": (64, 97), "presence": (97, 100),
          "echo": (100, 108), "mirror": (108, 124), "memory": (124, 140), "spare": (140, 148)}
WIDTH, LANG, RES = 148, 64, 48
LANG_SCALE = 1.0            # the ear has the language block to itself, so it is not turned down (§6.7)
WINDOW = int(os.environ.get("VEE_WINDOW", 48))   # moments per sitting (boundary); 48 = half a simulated day
NOBODY, MATRON, ROCKY = ([1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0])
RAM_CEILING = 0.85          # of host RAM, swap never counted (incubator design §5.1)
PATIENCE_BASE, PATIENCE_PER = 3, 8   # sittings settled before an organ may sharpen: BASE + classes // PER (the s067 probe's values)
RECENT = 10                 # sittings of what the world showed, kept to choose which reading sharpens


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


def can_be_silent(organ: str) -> torch.Tensor:
    """Which of an organ's numbers read exactly zero when there is nothing to read ("none"):
    the language block, the four readings through the window, the echo flags and the reserved
    channels.  Not what she holds or her marks (zero there is a real amount)."""
    x = torch.zeros(WIDTH, dtype=torch.bool)
    x[:LANG] = True; x[92:96] = True; x[100:106] = True; x[108:] = True
    if organ == "kin": return torch.cat([x[100:108], torch.zeros(35, dtype=torch.bool)])
    return x[:LANG] if organ == "ear" else x[LANG:]


def changes(X: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
    """What the kin organ reads from a run of consecutive moments: echo (8) | change of the 33
    environment readings | change of held and marks.  [N, 43].  ``prev`` = the moment before X[0]."""
    P = torch.cat([X[:1] if prev is None else prev[None], X[:-1]])
    return torch.cat([X[:, 100:108], X[:, 64:97] - P[:, 64:97], X[:, 106:108] - P[:, 106:108]], 1)


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


# ---- guards -----------------------------------------------------------------------
def stage() -> str:
    p = f"{HOME}/stage"
    return open(p).read().strip() if os.path.exists(p) else ""


def require_stage(name: str) -> None:
    if stage() != name:
        raise SystemExit(f"stage file {HOME}/stage says '{stage()}', this driver runs '{name}' only — "
                         f"a stage is entered by Rocky's sign-off, not by code")


def ram_used() -> float:
    m = {l.split(":")[0]: float(l.split()[1]) for l in open("/proc/meminfo")}
    return 1 - m["MemAvailable"] / m["MemTotal"]


# ---- the body ----------------------------------------------------------------------
class Body:
    blur = mature = None        # a body saved before organ blur existed (Vee) has none
    still_share = None          # opt-in (Rocky s070, option 1; tried on a copy): a sitting is still when the class count moved by
                                # no more than this share of the organ's classes since the sitting before. None = no change at all.

    def __init__(self, seed: int, pool: torch.Tensor, blur: bool | str = False):
        """``blur``: False = sharp organs (Vee); True = blurred (the first sister); "none" = blurred,
        and a silent reading is a state of its own (Rocky s068)."""
        self.seed, self.moments, self.born = seed, 0, time.time()
        mk = lambda P, k: PhasecyteLeaf(lambda x: x, P, seed, k, window=WINDOW, class_cap=None,
                                        manifold=True, composer=True, verbose=False)
        self.ear, self.soma = mk(pool[:, :LANG], 0), mk(pool[:, LANG:], 1)
        self.receptors = {k: int(l.sub.scheduler._plan.receptor_ids.numel()) for k, l in self.organs().items()}
        self.extra: dict = {}                       # what a driver must carry across saves (world, logs, …)
        if blur:                                    # senses formed on the full readings above; she sees them blurred from now on
            self.blur = {k: Blur(self.part(k, pool), none=can_be_silent(k) if blur == "none" else None) for k in ("ear", "soma")}
            self.mature = {k: {"quiet": 0, "n": 0, "sittings": 0, "recent": []} for k in self.blur}

    def organs(self) -> dict:
        o = {"ear": self.ear, "soma": self.soma}
        if getattr(self, "kin", None) is not None: o["kin"] = self.kin
        return o

    def grow_kin(self, pool: torch.Tensor) -> None:
        """Stage 2: the sense of change is born (a third leaf)."""
        self.kin = PhasecyteLeaf(lambda x: x, pool, self.seed, 2, window=WINDOW, class_cap=None,
                                 manifold=True, composer=True, verbose=False)
        self.receptors["kin"] = int(self.kin.sub.scheduler._plan.receptor_ids.numel())
        if self.blur is not None:                   # the echo (8) is levels; the 35 changes start at same / changed (Rocky s068)
            self.blur["kin"] = Blur(pool, change=torch.arange(pool.shape[1]) >= 8,
                                    none=can_be_silent("kin") if self.blur["soma"].none is not None else None)
            self.mature["kin"] = {"quiet": 0, "n": 0, "sittings": 0, "recent": []}
        self.prev = None                            # the moment before the next one she lives

    def part(self, organ: str, X: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
        """The numbers one organ reads.  For kin, X must be consecutive moments."""
        if organ == "kin": return changes(X, prev)
        return X[:, :LANG] if organ == "ear" else X[:, LANG:]

    @property
    def classes(self) -> dict:
        return {k: len(l.mixed.classes) for k, l in self.organs().items()}

    def live(self, X: torch.Tensor, labels=None) -> None:
        """One sitting's worth of moments, absorbed as a stream (never batch-trained).
        Labels (what the matron says a wording is) go to the ear only."""
        if ram_used() > RAM_CEILING:
            raise SystemExit(f"memory ceiling: host RAM {ram_used():.0%} used > {RAM_CEILING:.0%}; she is not fed")
        for k, l in self.organs().items():
            P = self.part(k, X, getattr(self, "prev", None))
            l.mixed.observe(self.seen(k, P), labels=labels if k == "ear" else None)
            l.sub.end_task()
            if self.blur is not None and k in self.blur: self._mature(k, P, len(l.mixed.classes))
        self.prev = X[-1].clone()
        self.moments += len(X)

    def seen(self, organ: str, P: torch.Tensor) -> torch.Tensor:
        """What the organ sees of its numbers ``P``: blurred if this body has blur."""
        return self.blur[organ].see(P) if self.blur is not None and organ in self.blur else P

    def _mature(self, organ: str, P: torch.Tensor, n: int) -> None:
        """Before birth: settled long enough for her size -> one reading one step sharper (no trial)."""
        m = self.mature[organ]
        m["sittings"] += 1
        m["recent"] = (m["recent"] + [P])[-RECENT:]
        allowed = int(self.still_share * n) if self.still_share else 0
        m["quiet"] = m["quiet"] + 1 if abs(n - m["n"]) <= allowed else 0
        m["n"] = n
        if m["quiet"] >= PATIENCE_BASE + n // PATIENCE_PER and stage() in ("embryo", "fetal"):
            self.blur[organ].sharpen(torch.cat(m["recent"]), m["sittings"])
            m["quiet"] = 0

    def read(self, X: torch.Tensor, organ: str = "soma") -> tuple[torch.Tensor, torch.Tensor]:
        """(class index, margin) per moment from one organ — no learning."""
        return self.organs()[organ].mixed.classify(self.seen(organ, self.part(organ, X)))

    def class_names(self, organ: str = "soma") -> list[str]:
        return [c.name for c in self.organs()[organ].mixed.classes]

    def save(self, name: str = "body") -> str:
        os.makedirs(HOME, exist_ok=True)
        p = f"{HOME}/{name}.pt"
        torch.save(self, p + ".tmp"); os.replace(p + ".tmp", p)
        return p

    @staticmethod
    def load(name: str = "body") -> "Body":
        return torch.load(f"{HOME}/{name}.pt", weights_only=False)
