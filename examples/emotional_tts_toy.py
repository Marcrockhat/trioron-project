"""Emotional TTS toy — trioron as device-side context memory.

Replaces a multimodal "feel-the-scene" model with a tiny task-aware
trioron that reads the text being spoken and decides the TTS preset
(speed, intensity, volume). Four contexts are taught from a handful
of example sentences each; novel inputs trip a novelty gate and route
to an `excited` mode ("ooh, that's new!") instead of pretending to
recognize them.

Stubs Qwen / Kokoro:
  - text encoder = stable bag-of-hashed-words (no transformers download)
  - TTS engine   = a print() that shows the call signature

Both are swappable for real implementations via the same shape — the
encoder just needs `.encode(text) -> Tensor[input_dim]` and the TTS
just needs `.__call__(text, *, speed, intensity, volume)`.

Run:
    python3 examples/emotional_tts_toy.py
"""
from __future__ import annotations
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trioron.api import (
    TaskData, TrioronConfig, build_donor, extend, load_organism,
)


# ---------------------------------------------------------------------
# Stub 1: text encoder (replace with sentence-transformers / Qwen)
# ---------------------------------------------------------------------

INPUT_DIM = 64


def _stable_hash(s: str, mod: int) -> int:
    """Deterministic across processes (Python's hash() is salted)."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % mod


class HashEncoder:
    """Bag-of-hashed-words → unit-norm feature vector. Cheap stand-in
    for a real text encoder; preserves enough lexical signal to let the
    trioron separate distinct vocabularies into distinct contexts."""

    def __init__(self, dim: int = INPUT_DIM):
        self.dim = dim

    def encode(self, text: str) -> torch.Tensor:
        v = torch.zeros(self.dim)
        words = text.lower().split()
        for w in words:
            v[_stable_hash(w, self.dim)] += 1.0
        # Bigrams give a little more separation between contexts that
        # share unigrams ("you're safe" vs "you're late").
        for a, b in zip(words, words[1:]):
            v[_stable_hash(a + "_" + b, self.dim)] += 0.5
        n = v.norm()
        return v / n if n > 0 else v


# ---------------------------------------------------------------------
# Stub 2: TTS engine (replace with Kokoro)
# ---------------------------------------------------------------------


class StubTTS:
    """Prints the TTS call signature instead of synthesizing audio."""

    def __call__(
        self, text: str, *, speed: float, intensity: float, volume: float,
    ) -> None:
        print(f"    [TTS] speak({text!r}, speed={speed:.2f}, "
              f"intensity={intensity:.2f}, volume={volume:.2f})")


# ---------------------------------------------------------------------
# The mode palette: class id ↔ (label, preset)
# ---------------------------------------------------------------------

MODE_PALETTE: Dict[int, Tuple[str, Dict[str, float]]] = {
    0: ("calm",    {"speed": 0.90, "intensity": 0.40, "volume": 0.50}),
    1: ("gentle",  {"speed": 0.95, "intensity": 0.50, "volume": 0.40}),
    2: ("firm",    {"speed": 1.00, "intensity": 0.80, "volume": 0.70}),
    3: ("urgent",  {"speed": 1.20, "intensity": 1.00, "volume": 0.90}),
}

# Reserved id for the novelty fallback — trioron is NOT trained on this
# class. The novelty gate routes here when manifold log-lik is too low.
EXCITED_MODE = ("excited", {"speed": 1.15, "intensity": 0.95, "volume": 0.80})


# ---------------------------------------------------------------------
# Training corpus — a handful of example sentences per context
# ---------------------------------------------------------------------

CORPUS: Dict[int, List[str]] = {
    0: [  # calm
        "good morning the sky is blue",
        "the lake is quiet today",
        "let us read a slow story",
        "breathe in and out together",
        "the garden smells like rain",
        "soft music plays in the background",
        "we have all day to finish this",
        "rest your eyes for a moment",
    ],
    1: [  # gentle
        "you are safe with me now",
        "everything is going to be okay",
        "take your time there is no rush",
        "i am right here beside you",
        "your feelings make sense little one",
        "we will figure this out together",
        "you did your best and that matters",
        "it is alright to feel sad",
    ],
    2: [  # firm
        "stop right there and listen carefully",
        "wait for me before you cross",
        "you must follow the rules now",
        "put that down it is not yours",
        "do not touch the hot stove",
        "stay inside the lines please",
        "answer the question i just asked",
        "look at me when i am speaking",
    ],
    3: [  # urgent
        "warning slow down obstacle ahead",
        "danger fire alarm is sounding",
        "evacuate the building immediately",
        "watch out a car is coming",
        "grab the railing the floor is wet",
        "hurry the train is leaving now",
        "alert smoke detected in the kitchen",
        "move back the wire is live",
    ],
}


def build_tasks(corpus: Dict[int, List[str]]) -> List[TaskData]:
    """One TaskData per context. 80/20 train/test split per task."""
    enc = HashEncoder(INPUT_DIM)
    tasks: List[TaskData] = []
    for cls, sents in corpus.items():
        X = torch.stack([enc.encode(s) for s in sents])
        y = torch.full((len(sents),), cls, dtype=torch.int64)
        n_train = max(1, int(0.75 * len(sents)))
        label, _ = MODE_PALETTE.get(cls, EXCITED_MODE)
        tasks.append(TaskData(
            name=f"mode_{label}",
            X_train=X[:n_train], y_train=y[:n_train],
            X_test=X[n_train:],  y_test=y[n_train:],
            classes=[cls],
        ))
    return tasks


# ---------------------------------------------------------------------
# Inference: encode → trioron forward → novelty gate → TTS dispatch
# ---------------------------------------------------------------------


def calibrate_novelty_threshold(organism, tasks: List[TaskData]) -> float:
    """Score the manifold's actual training inputs and pick the 10th
    percentile as the novelty floor. Anything below that is 'this looks
    less like my training data than 90% of what I learned from.'

    Uses task.X_train directly (the rows the manifold actually fit on)
    — calibrating against held-out or off-corpus inputs would bake OOD
    outliers into the threshold.

    Per-class manifold σ collapses on zero-variance dimensions (sparse
    bag-of-hashed-words has many such dims), so a few training rows can
    score with huge magnitude. The percentile is robust to that; mean
    ± std is not.
    """
    branch = organism.branches[0]
    z_canon = []
    for t in tasks:
        z = organism.project_l0(t.X_train.float())
        z_canon.append(z)
    z = torch.cat(z_canon, dim=0)
    log_lik = branch.per_class_log_likelihood(z)        # (N, C)
    max_per_row = log_lik.max(dim=-1).values            # (N,)
    p10 = float(torch.quantile(max_per_row, 0.10))
    median = float(torch.quantile(max_per_row, 0.50))
    thresh = p10 - 0.05 * abs(p10)  # small margin below the 10th pctile
    print(f"  [novelty calibration] training max-log-lik "
          f"median={median:.2f}  p10={p10:.2f}  threshold={thresh:.2f}")
    return thresh


def decide_and_speak(
    text: str, *, organism, encoder, tts, novelty_threshold: float,
) -> None:
    """The end-to-end inference path that would replace a multimodal
    model: encoder → trioron → novelty gate → TTS dispatch."""
    branch = organism.branches[0]
    x = encoder.encode(text).unsqueeze(0)
    z = organism.project_l0(x)
    log_lik = branch.per_class_log_likelihood(z)        # (1, C)
    max_log_lik, pred_class_local = log_lik[0].max(dim=-1)
    pred_class_global = branch.archive_classes[int(pred_class_local)]
    is_novel = float(max_log_lik) < novelty_threshold

    if is_novel:
        label, preset = EXCITED_MODE
        print(f"\n  > {text!r}")
        print(f"    [trioron] max-log-lik {float(max_log_lik):+.2f} "
              f"< threshold {novelty_threshold:+.2f}  → NOVELTY")
        print("    [inner voice] ooh, that's new!  excited mode.")
    else:
        label, preset = MODE_PALETTE[pred_class_global]
        print(f"\n  > {text!r}")
        print(f"    [trioron] max-log-lik {float(max_log_lik):+.2f}  "
              f"→ recognized as {label!r}")
    tts(text, **preset)


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------


def main() -> int:
    print("Emotional TTS toy — trioron as context memory\n")
    encoder = HashEncoder(INPUT_DIM)
    tts = StubTTS()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ---- Phase 1: teach the four core contexts ----
        print("=" * 56)
        print("PHASE 1: teach the 4 core modes (calm, gentle, firm, urgent)")
        print("=" * 56)
        tasks = build_tasks(CORPUS)
        cfg = TrioronConfig(cap_bytes=8_000)
        donor = build_donor(
            tasks=tasks, label="emotional_tts_v1",
            out_path=td / "donor.pt",
            seed=42, epochs_per_task=4, config=cfg,
        )
        org = load_organism(donor)
        thresh = calibrate_novelty_threshold(org, tasks)

        # ---- Phase 2: known sentences route to learned modes ----
        print("\n" + "=" * 56)
        print("PHASE 2: known sentences (one per learned mode)")
        print("=" * 56)
        known_inputs = [
            "the lake is quiet today",                  # calm
            "you are safe with me now",                 # gentle
            "stop right there and listen carefully",    # firm
            "warning slow down obstacle ahead",         # urgent
        ]
        for s in known_inputs:
            decide_and_speak(
                s, organism=org, encoder=encoder, tts=tts,
                novelty_threshold=thresh,
            )

        # ---- Phase 3: novel sentence trips the excited fallback ----
        print("\n" + "=" * 56)
        print("PHASE 3: novel sentences (no mode trained for them)")
        print("=" * 56)
        novel_inputs = [
            "fascinating discovery in quantum chemistry",
            "the catalyst molecule rearranges the lattice",
            "purple jellyfish glow under ultraviolet light",
        ]
        for s in novel_inputs:
            decide_and_speak(
                s, organism=org, encoder=encoder, tts=tts,
                novelty_threshold=thresh,
            )

        # ---- Phase 4: extend with a 5th mode ('whispered') ----
        print("\n" + "=" * 56)
        print("PHASE 4: extend with a 5th mode (whispered)")
        print("=" * 56)
        whispered_corpus = {
            4: [
                "the secret is hidden in the attic",
                "do not let the cat hear us",
                "tiptoe past the sleeping baby",
                "they cannot know we were here",
                "leave no trace behind us",
                "sneak through the back door",
            ],
        }
        whispered_palette = {
            "speed": 0.85, "intensity": 0.30, "volume": 0.20,
        }
        new_tasks = build_tasks(whispered_corpus)
        extended = extend(
            donor_path=donor,
            base_tasks=tasks, new_tasks=new_tasks,
            out_path=td / "donor_extended.pt",
            extension_cap_bytes=12_000, epochs_per_task=4,
            permanent_int8=False,
        )
        org2 = load_organism(extended)
        # Recalibrate against the extended corpus.
        thresh2 = calibrate_novelty_threshold(org2, tasks + new_tasks)

        # Patch the palette in-place so the dispatcher knows class 4.
        MODE_PALETTE[4] = ("whispered", whispered_palette)

        print("\n  re-trying the same novel sentences after extension:")
        for s in novel_inputs:
            decide_and_speak(
                s, organism=org2, encoder=encoder, tts=tts,
                novelty_threshold=thresh2,
            )

        print("\n  trying a whispered sentence:")
        decide_and_speak(
            "the secret is hidden in the attic",
            organism=org2, encoder=encoder, tts=tts,
            novelty_threshold=thresh2,
        )

        # Sanity: original modes still work after extension.
        print("\n  sanity: original mode still works after extension:")
        decide_and_speak(
            "the lake is quiet today",
            organism=org2, encoder=encoder, tts=tts,
            novelty_threshold=thresh2,
        )

        print("\n" + "=" * 56)
        print("done. trioron donor on disk = ", end="")
        sz_kb = donor.stat().st_size / 1024
        print(f"{sz_kb:.1f} KB (replaces a multimodal scene-feeling model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
