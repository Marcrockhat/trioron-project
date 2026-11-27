"""A body: organs that read one input, lived as a stream, saved whole (plan §6, §8.4, §11.2).

Organs (phasecyte leaves) over the 148-number input (layout.py):
  ear    over the language block (hears silence as zeros)
  soma   the body sense, over everything else (environment, presence, reserved)
  kin    the sense of change, born at the fetal stage (plan §8.4): the echo block and how
         every environment reading, what it holds and its marks moved since the moment before
In ONE leaf text and world compete for the same classes (s067, plan §6.7), hence organs.
The level that joins them is not built yet.

Senses form at genesis on a broad sample of the world with spontaneous activity on the
reserved channels: a receptor exists only for a channel that carries signal at genesis.

Organ blur (s068, plan §11.2; blur.py): a body made with ``blur=True`` sees each reading of
an organ in a few steps (2 at first).  Before birth an organ takes one rung for one reading
when it has not gained or lost a class for PATIENCE_BASE + classes // PATIENCE_PER sittings.
After birth a step must be called for by the matron's corrections and pay — NOT built, so
nothing sharpens once the stage file leaves embryo / fetal.  A body's blur is set at genesis
for life.

What an organ may carry beside this (sitting cap, Numa rule, fourth organ) is set on its
leaf (``leaf.mixed.sitting / numa / court``), not here.

State: the whole body is saved (torch pickle, atomic) — saving is not freezing.
"""
from __future__ import annotations
import os, time
import torch
from ..pcll import PhasecyteLeaf
from .blur import Blur
from .home import HOME, RAM_CEILING, ram_used, stage
from .layout import LANG, can_be_silent, changes

WINDOW = int(os.environ.get("VEE_WINDOW", 48))   # moments per sitting (boundary); 48 = half a simulated day
PATIENCE_BASE, PATIENCE_PER = 3, 8   # sittings settled before an organ may sharpen: BASE + classes // PER (the s067 probe's values)
RECENT = 10                 # sittings of what the world showed, kept to choose which reading sharpens


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
