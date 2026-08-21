"""s059 — grounded mini-language world (Stage 2 of the logic/language arc).

NOT a corpus. Sentences are generated from a grammar and GROUNDED in a
scene; every sentence has a truth value in its scene, so comprehension is
supervised by the world, not by text statistics.

Scene  : 2-4 objects, each (color, shape, size, x, y) on a 4x4 grid.
Grammar (depth d = nesting of NOT / AND / OR / relation clauses):
  S  -> NP VP                                  (d=0)
  NP -> the [ADJ] NOUN                          ("the red ball", "the ball")
  VP -> is ADJ | is REL NP | is not VP | VP and VP | VP or VP
  REL -> left of | right of | above | below | near | bigger than
Words ~ 40 (see VOCAB). A sentence is a list of word ids (class_cap slots).
Tasks
  comprehension : (scene, sentence) -> true / false   (balanced)
  production    : scene -> a TRUE sentence of requested depth (teacher set)
Splits
  train_stage_i : vocabulary grows in 3 stages (continual acquisition)
  compositional : held-out (ADJ, NOUN) pairs never seen together in train
  depth         : nesting depth deeper than any train sentence
Encoding for the chain: scene -> [n_obj * OBJ_DIM] floats (the "sense");
sentence -> [L_MAX, CLASS_CAP] one-hot symbol evidence (Link-0 shape), PAD=0.

Self-test:  python3 experiments/progenitor/language_world.py
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch, torch.nn.functional as F

COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["ball", "cube", "star", "ring"]
SIZES = ["small", "big"]
RELS = {"left": lambda a, b: a.x < b.x, "right": lambda a, b: a.x > b.x,
        "above": lambda a, b: a.y < b.y, "below": lambda a, b: a.y > b.y,
        "near": lambda a, b: abs(a.x - b.x) + abs(a.y - b.y) <= 1,
        "bigger": lambda a, b: a.size > b.size}
FUNC = ["the", "is", "not", "and", "or", "of", "than", "thing"]
VOCAB = ["<pad>"] + COLORS + SHAPES + SIZES + list(RELS) + FUNC
WID = {w: i for i, w in enumerate(VOCAB)}
CLASS_CAP = 32          # slots for the symbol layer (VOCAB is 25; room to grow)
L_MAX = 24
OBJ_DIM = len(COLORS) + len(SHAPES) + 1 + 2     # color, shape, size, x, y
MAX_OBJ = 4
DOUBLE_NOT = True     # allow "not not": depth-2 negation = parity over NOTs; a split of its own

# continual acquisition: which words are allowed per stage
STAGES = [
    set(["red", "blue", "ball", "cube", "small", "big", "the", "is", "thing", "not"]),
    set(["green", "star", "left", "right", "above", "below", "of", "and"]),
    set(["yellow", "ring", "near", "bigger", "than", "or"]),
]


@dataclass
class Obj:
    color: str; shape: str; size: int; x: int; y: int


def sample_scene(rng: random.Random, allowed: set) -> List[Obj]:
    cols = [c for c in COLORS if c in allowed] or COLORS
    shp = [s for s in SHAPES if s in allowed] or SHAPES
    n = rng.randint(2, MAX_OBJ)
    cells = rng.sample([(x, y) for x in range(4) for y in range(4)], n)
    return [Obj(rng.choice(cols), rng.choice(shp), rng.randint(0, 1), x, y)
            for (x, y) in cells]


# ------------------------------------------------------------ grammar
# A sentence is a tree; truth is evaluated against the scene.
def np_(rng, scene, allowed, hold=()):
    """Pick an NP that refers uniquely to one object (so truth is well defined)."""
    for _ in range(20):
        o = rng.choice(scene)
        adj = rng.choice([None, o.color, SIZES[o.size]])
        noun = o.shape if "thing" not in allowed or rng.random() < .7 else "thing"
        if adj and adj not in allowed: continue
        if (adj, noun) in hold: continue
        ref = [p for p in scene if (noun == "thing" or p.shape == noun)
               and (adj is None or p.color == adj or SIZES[p.size] == adj)]
        if len(ref) == 1:
            words = ["the"] + ([adj] if adj else []) + [noun]
            return words, ref[0]
    return None


def vp_(rng, scene, allowed, depth, subj, hold, no_not=False):
    """Returns (words, truth). depth = remaining nesting budget."""
    if depth > 0:
        ops = [o for o in ["not", "and", "or"] if o in allowed and not (o == "not" and no_not)]
        if ops:
            op = rng.choice(ops)
            if op == "not":
                w, t = vp_(rng, scene, allowed, depth - 1, subj, hold, no_not=not DOUBLE_NOT)
                return ["is", "not"] + w[1:] if w[0] == "is" else ["not"] + w, (not t)
            a, ta = vp_(rng, scene, allowed, depth - 1, subj, hold)
            b, tb = vp_(rng, scene, allowed, depth - 1, subj, hold)
            return a + [op] + b, (ta and tb) if op == "and" else (ta or tb)
    # leaf predicate
    rels = [r for r in RELS if r in allowed]
    if rels and rng.random() < .5:
        r = rng.choice(rels)
        other = np_(rng, scene, allowed, hold)
        if other and other[1] is not subj:
            tail = ["than"] if r == "bigger" else ["of"] if r in ("left", "right") else []
            return ["is", r] + tail + other[0], bool(RELS[r](subj, other[1]))
    adj = rng.choice([a for a in COLORS + SIZES if a in allowed])
    truth = (subj.color == adj) or (SIZES[subj.size] == adj)
    return ["is", adj], truth


def sample_sentence(rng, scene, allowed, depth, hold=()) -> Optional[Tuple[List[str], bool]]:
    np = np_(rng, scene, allowed, hold)
    if np is None: return None
    words, subj = np
    vp, t = vp_(rng, scene, allowed, depth, subj, hold)
    s = words + vp
    return (s, t) if len(s) <= L_MAX else None


# ------------------------------------------------------------ encoding
def encode_scene(scene) -> torch.Tensor:
    v = torch.zeros(MAX_OBJ, OBJ_DIM)
    for i, o in enumerate(scene):
        v[i, COLORS.index(o.color)] = 1
        v[i, len(COLORS) + SHAPES.index(o.shape)] = 1
        v[i, len(COLORS) + len(SHAPES)] = o.size
        v[i, -2:] = torch.tensor([o.x / 3, o.y / 3])
    return v.flatten()


def encode_sentence(words) -> torch.Tensor:
    ids = torch.tensor([WID[w] for w in words] + [0] * (L_MAX - len(words)))
    e = F.one_hot(ids, CLASS_CAP).float()
    e[ids == 0] = 0                     # PAD = no evidence
    return e


def build(n, stage, depth, seed, hold=(), balanced=True):
    """Comprehension set: X_scene [n, MAX_OBJ*OBJ_DIM], X_sent [n, L_MAX, CLASS_CAP],
    y [n] truth, plus the raw sentences (production teacher set = the TRUE ones)."""
    rng = random.Random(seed)
    allowed = set().union(*STAGES[:stage + 1])
    S, E, Y, raw = [], [], [], []
    want = {True: n // 2, False: n - n // 2}
    while len(Y) < n:
        scene = sample_scene(rng, allowed)
        out = sample_sentence(rng, scene, allowed, depth, hold)
        if out is None: continue
        words, t = out
        if balanced and want[t] == 0: continue
        want[t] -= 1
        S.append(encode_scene(scene)); E.append(encode_sentence(words)); Y.append(int(t))
        raw.append((scene, words, t))
    return torch.stack(S), torch.stack(E), torch.tensor(Y), raw


HOLD_OUT = [("red", "cube"), ("blue", "star"), ("green", "ball")]   # compositional split


if __name__ == "__main__":
    print(f"vocab {len(VOCAB)} words, class_cap {CLASS_CAP}, L_MAX {L_MAX}, scene dim {MAX_OBJ*OBJ_DIM}")
    for stage in range(3):
        for d in range(3):
            S, E, Y, raw = build(2000, stage, d, seed=stage * 10 + d, hold=HOLD_OUT)
            lens = [len(w) for _, w, _ in raw]
            print(f"stage {stage} depth {d}: n={len(Y)} true={Y.float().mean():.2f} "
                  f"len {min(lens)}-{max(lens)} (mean {sum(lens)/len(lens):.1f})")
            for scene, w, t in raw[:2]:
                print("    ", " ".join(w), "->", t, "|",
                      "; ".join(f"{SIZES[o.size]} {o.color} {o.shape}@({o.x},{o.y})" for o in scene))
    # majority / shallow sanity: no word-order-free leak? (bag-of-words truth rate)
    S, E, Y, raw = build(4000, 2, 2, seed=99)
    from collections import defaultdict
    bow = defaultdict(list)
    for _, w, t in raw: bow[tuple(sorted(w))].append(t)
    amb = sum(1 for v in bow.values() if 0 < sum(v) < len(v))
    print(f"bag-of-words keys {len(bow)}, keys with BOTH truth values {amb} "
          f"(truth depends on scene, not on words alone: good)")
