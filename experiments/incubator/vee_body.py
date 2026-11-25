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
Both live every moment (the ear hears silence as zeros).  The level that joins them
is not built yet.

The language sense is fitted ONCE (Wikipedia sentence readings + stage-2 question
readings, §6.2), stored in HOME/sense.pt and never refitted.

Her senses form at genesis on a broad sample of her world with spontaneous activity
(noise) on the reserved channels: a receptor exists only for a channel that carries
signal at genesis (s066 germline checks).

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
def moment(lang: torch.Tensor | None, raw: dict, presence=NOBODY) -> torch.Tensor:
    """Her input at one moment.  ``lang`` = sensed text (64, unscaled) or None; ``raw`` = environment readings."""
    x = torch.zeros(WIDTH)
    if lang is not None: x[:LANG] = LANG_SCALE * lang
    x[64:97] = torch.tensor(environment.vector(raw))
    x[97:100] = torch.tensor(presence)
    return x


def genesis_pool(seed: int, ambient: torch.Tensor, n: int = 10 * WINDOW, p_text: float = 0.5) -> torch.Tensor:
    """What her senses form on: every state of her world toggled freely, any presence,
    spontaneous activity on the reserved channels (they fall silent once she lives)."""
    w, g = World(500 + seed), np.random.default_rng(3000 + seed)
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
    def __init__(self, seed: int, pool: torch.Tensor):
        self.seed, self.moments, self.born = seed, 0, time.time()
        mk = lambda P, k: PhasecyteLeaf(lambda x: x, P, seed, k, window=WINDOW, class_cap=None,
                                        manifold=True, composer=True, verbose=False)
        self.ear, self.soma = mk(pool[:, :LANG], 0), mk(pool[:, LANG:], 1)
        self.receptors = {k: int(l.sub.scheduler._plan.receptor_ids.numel()) for k, l in self.organs().items()}
        self.extra: dict = {}                       # what a driver must carry across saves (world, logs, …)

    def organs(self) -> dict:
        return {"ear": self.ear, "soma": self.soma}

    @staticmethod
    def part(organ: str, X: torch.Tensor) -> torch.Tensor:
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
            l.mixed.observe(self.part(k, X), labels=labels if k == "ear" else None)
            l.sub.end_task()
        self.moments += len(X)

    def read(self, X: torch.Tensor, organ: str = "soma") -> tuple[torch.Tensor, torch.Tensor]:
        """(class index, margin) per moment from one organ — no learning."""
        return self.organs()[organ].mixed.classify(self.part(organ, X))

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
