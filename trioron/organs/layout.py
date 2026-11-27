"""The input a body reads, fixed before it exists (plan §6 — the germline; rule 24).

148 numbers per moment (LAYOUT):
    language 64 | environment 33 | presence 3 | action echo 8 | mirror 16 | working memory 16 | spare 8
  language   the reflex's read state of the text present, through the fixed language sense;
             all zero when no text is present
  presence   who is addressing it: nobody / the matron / Rocky (+1 for the one, -1 others)
  echo       (plan §8.3) 1-6 what it did in the moment just past (+1 that action, -1 the
             others), 7 how much it holds, 8 how many marks exist; all zero before stage 2
  reserved   mirror, working memory, spare: exactly zero until the organ exists

Changing anything here means a new organism.
"""
from __future__ import annotations
import torch

LAYOUT = {"language": (0, 64), "environment": (64, 97), "presence": (97, 100),
          "echo": (100, 108), "mirror": (108, 124), "memory": (124, 140), "spare": (140, 148)}
WIDTH, LANG, RES = 148, 64, 48
LANG_SCALE = 1.0            # the ear has the language block to itself, so it is not turned down (§6.7)
NOBODY, MATRON, ROCKY = ([1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0])


def can_be_silent(organ: str) -> torch.Tensor:
    """Which of an organ's numbers read exactly zero when there is nothing to read ("none"):
    the language block, the four readings through the window, the echo flags and the reserved
    channels.  Not what it holds or its marks (zero there is a real amount)."""
    x = torch.zeros(WIDTH, dtype=torch.bool)
    x[:LANG] = True; x[92:96] = True; x[100:106] = True; x[108:] = True
    if organ == "kin": return torch.cat([x[100:108], torch.zeros(35, dtype=torch.bool)])
    return x[:LANG] if organ == "ear" else x[LANG:]


def changes(X: torch.Tensor, prev: torch.Tensor | None = None) -> torch.Tensor:
    """What the sense of change reads from a run of consecutive moments: echo (8) | change of
    the 33 environment readings | change of held and marks.  [N, 43].  ``prev`` = the moment
    before X[0]."""
    P = torch.cat([X[:1] if prev is None else prev[None], X[:-1]])
    return torch.cat([X[:, 100:108], X[:, 64:97] - P[:, 64:97], X[:, 106:108] - P[:, 106:108]], 1)
