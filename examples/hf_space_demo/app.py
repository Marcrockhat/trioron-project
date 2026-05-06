"""Trioron TTS Demo v3 — semantic encoder (MiniLM) + segment-level intonation.

The demo input can be:
  - plain text → auto-split on sentence boundaries, trioron picks
    a mode per segment;
  - or marked up like "<excited>Whoa!</excited> <calm>The sky is
    very clear</calm>." → each tagged span uses that mode directly,
    untagged spans fall back to per-sentence trioron routing.

Eight learned modes (calm / gentle / firm / urgent / excited / sad /
curious / whispered) plus a "neutral" novelty fallback for inputs
that match nothing the trioron has been taught.

Encoder: sentence-transformers `all-MiniLM-L6-v2` (~22M params,
384-dim). Replaces the v1/v2 hash-bag-of-words baseline so the
trioron sees real semantic neighborhoods. The trioron itself is
unchanged — this just gives it intelligible inputs.

Audio renders client-side via the browser's Web Speech API; one
SpeechSynthesisUtterance per segment, queued in order.
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trioron.api import (
    TaskData, TrioronConfig, build_donor, load_organism,
)


# ---------------------------------------------------------------------
# Encoder — sentence-transformers MiniLM-L6-v2
# ---------------------------------------------------------------------

INPUT_DIM = 384  # matches all-MiniLM-L6-v2 output dim
_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_ENCODER = None


def _get_encoder():
    """Lazy singleton. The model loads from HF on first call (~3s on
    a warm Space, longer if the weights aren't cached) and stays in
    memory afterwards."""
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer
        print(f"[trioron-demo] loading encoder {_ENCODER_MODEL}...")
        _ENCODER = SentenceTransformer(_ENCODER_MODEL)
        _ENCODER.eval()
        print(f"[trioron-demo] encoder loaded "
              f"(out_dim={_ENCODER.get_sentence_embedding_dimension()})")
    return _ENCODER


def encode(text: str, dim: int = INPUT_DIM) -> torch.Tensor:
    """Returns a 384-dim sentence embedding, L2-normalized.
    Normalization is what makes the manifold's diagonal-Gaussian fit
    meaningful — without it, magnitude differences between sentences
    would dominate the per-class log-pdf."""
    enc = _get_encoder()
    with torch.no_grad():
        v = enc.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    return v.detach().cpu().float()


def encode_batch(texts: List[str]) -> torch.Tensor:
    """Faster than calling `encode` per sentence — used at training
    time when we encode all corpus sentences at once."""
    enc = _get_encoder()
    with torch.no_grad():
        v = enc.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return v.detach().cpu().float()


# ---------------------------------------------------------------------
# Piper TTS backend (optional server-side renderer)
# ---------------------------------------------------------------------

PIPER_VOICE_ID = "en_US-lessac-medium"
_PIPER_VOICE = None


def _get_piper_voice():
    """Lazy singleton. Downloads the voice (~60 MB) on first use,
    cached under HF's standard cache dir afterwards."""
    global _PIPER_VOICE
    if _PIPER_VOICE is None:
        from huggingface_hub import hf_hub_download
        from piper.voice import PiperVoice
        print(f"[trioron-demo] downloading Piper voice {PIPER_VOICE_ID}...")
        onnx_path = hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=f"en/en_US/lessac/medium/{PIPER_VOICE_ID}.onnx",
        )
        # Force the .json sidecar to land next to the .onnx (PiperVoice
        # expects them in the same dir; hf_hub_download symlinks both
        # to the snapshot dir so this is automatic).
        hf_hub_download(
            repo_id="rhasspy/piper-voices",
            filename=f"en/en_US/lessac/medium/{PIPER_VOICE_ID}.onnx.json",
        )
        print("[trioron-demo] loading Piper voice...")
        _PIPER_VOICE = PiperVoice.load(onnx_path)
        print(f"[trioron-demo] Piper voice ready "
              f"(sample_rate={_PIPER_VOICE.config.sample_rate})")
    return _PIPER_VOICE


def render_with_piper(segments: List[Dict]):
    """Synthesize each segment with its preset and concatenate into
    one audio array. Returns (sample_rate, int16_array) suitable for
    gr.Audio. Each segment's speed → length_scale (inverse), intensity
    → noise_scale, volume passes through. ~150-300 ms per sentence on
    CPU."""
    import numpy as np
    from piper import SynthesisConfig
    voice = _get_piper_voice()
    sample_rate = voice.config.sample_rate
    silence_gap = np.zeros(int(sample_rate * 0.30), dtype=np.int16)
    pieces = []
    for i, r in enumerate(segments):
        # speed > 1 → faster speech → shorter length_scale
        length_scale = 1.0 / max(0.1, float(r["speed"]))
        # intensity 0..1.15 → noise_scale 0.30..1.28 (wider than the
        # v3.1 mapping; pushes deadpan-sad and amped-urgent further
        # apart in voice variability)
        noise_scale = 0.30 + 0.85 * float(r.get("intensity", 0.6))
        cfg = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            volume=float(r["volume"]),
        )
        chunks = list(voice.synthesize(r["segment"], cfg))
        for c in chunks:
            pieces.append(np.frombuffer(c.audio_int16_bytes, dtype=np.int16))
        if i < len(segments) - 1:
            pieces.append(silence_gap)
    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    return sample_rate, audio


# ---------------------------------------------------------------------
# Mode palette + corpus
# ---------------------------------------------------------------------

# Dramatic preset spread. Web Speech rate is safe up to ~1.5 across
# browsers; below 0.5 voice quality degrades. Piper handles a wider
# range natively (length_scale = 1/speed) — these values give Piper
# room to really stretch sad/whispered and accelerate urgent.
MODE_PALETTE: Dict[int, Tuple[str, Dict[str, float]]] = {
    0: ("calm",      {"speed": 0.80, "intensity": 0.30, "volume": 0.55}),
    1: ("gentle",    {"speed": 0.82, "intensity": 0.40, "volume": 0.40}),
    2: ("firm",      {"speed": 1.05, "intensity": 0.85, "volume": 0.85}),
    3: ("urgent",    {"speed": 1.50, "intensity": 1.10, "volume": 1.00}),
    4: ("excited",   {"speed": 1.40, "intensity": 1.15, "volume": 0.95}),
    5: ("sad",       {"speed": 0.70, "intensity": 0.20, "volume": 0.40}),
    6: ("curious",   {"speed": 1.10, "intensity": 0.75, "volume": 0.70}),
    7: ("whispered", {"speed": 0.95, "intensity": 0.25, "volume": 0.18}),
}
NEUTRAL_MODE = ("neutral", {"speed": 1.00, "intensity": 0.60, "volume": 0.70})

# Novelty fallback fires only when the top two classes are basically
# tied (no class is decisively winning). Real OOD detection in a tight
# 8-class semantic space is genuinely hard with manifold-only signal;
# the user is better served by a confident-but-sometimes-wrong
# classification than by frequent "neutral" cop-outs. Use markup tags
# to override when auto-routing picks the wrong mode.
NOVELTY_GAP_FLOOR = 1.0

# Lookup by name (markup → preset).
_PALETTE_BY_NAME = {label: preset for label, preset in MODE_PALETTE.values()}
_PALETTE_BY_NAME["neutral"] = NEUTRAL_MODE[1]
_NAME_TO_CLASS = {label: cid for cid, (label, _) in MODE_PALETTE.items()}


CORPUS: Dict[int, List[str]] = {
    0: [  # calm — present-tense peaceful states, low arousal
        "good morning the sky is blue",
        "the lake is quiet today",
        "let us read a slow story",
        "breathe in and out together",
        "the garden smells like rain",
        "soft music plays in the background",
        "we have all day to finish this",
        "rest your eyes for a moment",
        "the afternoon is peaceful and still",
        "watch the clouds drift by slowly",
        "the candle flickers gently on the table",
        "everything is settled and there is no hurry",
        "the room is warm and the tea is ready",
        "let us sit by the window and just breathe",
    ],
    1: [  # gentle — comforting, reassuring, addressed to someone
        "you are safe with me now",
        "everything is going to be okay",
        "take your time there is no rush",
        "i am right here beside you",
        "your feelings make sense little one",
        "we will figure this out together",
        "you did your best and that matters",
        "it is alright to feel sad",
        "let me hold your hand for a while",
        "you are loved just as you are",
        "you do not have to be strong right now",
        "rest in my arms for as long as you need",
        "no one is upset with you sweetheart",
        "i believe in you completely",
    ],
    2: [  # firm — directive, commanding, present-tense imperative
        "stop right there and listen carefully",
        "wait for me before you cross",
        "you must follow the rules now",
        "put that down it is not yours",
        "do not touch the hot stove",
        "stay inside the lines please",
        "answer the question i just asked",
        "look at me when i am speaking",
        "i need you to do this right now",
        "this is the last time i ask",
        "sit down and finish your homework",
        "no means no and i mean it",
        "you will apologize to your sister",
        "hand me the keys this instant",
    ],
    3: [  # urgent — immediate physical danger, action required NOW
        "warning slow down obstacle ahead",
        "danger fire alarm is sounding",
        "evacuate the building immediately",
        "watch out a car is coming",
        "grab the railing the floor is wet",
        "hurry the train is leaving now",
        "alert smoke detected in the kitchen",
        "move back the wire is live",
        "run the bridge is collapsing",
        "duck the branch is falling",
        "stop the bleeding press hard",
        "get down there is broken glass",
        "the gas valve is open turn it off now",
        "jump the floor is giving way",
    ],
    4: [  # excited — present-tense joy / amazement / good surprise
        "whoa look at that incredible view",
        "yes we finally did it together",
        "this is the best day of the year",
        "i cannot believe what just happened",
        "wow the colors are amazing tonight",
        "we won the championship trophy",
        "the puppy is finally home today",
        "guess what i have wonderful news",
        "look the rocket is about to launch",
        "you got accepted into the school",
        "the rainbow is right above the bridge",
        "tomorrow is going to be amazing",
        "i can taste it from here it smells incredible",
        "this is exactly what i was hoping for",
    ],
    5: [  # sad — past-event regret / loss / longing / mistakes
        "i miss the days when we were together",
        "the old house is empty now",
        "she did not come back this year",
        "the garden has not bloomed since spring",
        "we lost the match by a single point",
        "the letter never arrived in time",
        "i wish i had said goodbye properly",
        "it has been raining for three days",
        "the photograph is faded and torn",
        "no one remembers the song anymore",
        # Past-event regret framings (added to disambiguate from
        # "excited"-style past-event-good-surprise sentences).
        "remember when we used to play here every weekend",
        "i wish we had brought the umbrella that day",
        "we got soaked at the picnic and i still feel bad about it",
        "the trip was canceled because of the storm last summer",
        "we made the same mistake again and i should have known better",
        "the dog has not waited at the door since she left",
    ],
    6: [  # curious — open questions, exploration, "i wonder"
        "i wonder why the sky turns red at dusk",
        "what is that strange sound in the attic",
        "how does the bird know where to fly",
        "why do the leaves change color in autumn",
        "where does the river end its journey",
        "what would happen if we tried this way",
        "tell me more about how it works",
        "could that little box really hold so much",
        "i have never seen this kind of flower before",
        "show me how you made it light up",
        "is that what they meant by the old word",
        "what does it look like from the other side",
        "how did you figure that out so quickly",
        "i wonder if it works underwater too",
    ],
    7: [  # whispered — secrets, hushed warnings, conspiratorial
        "the secret is hidden in the attic",
        "do not let the cat hear us",
        "tiptoe past the sleeping baby",
        "they cannot know we were here",
        "leave no trace behind us",
        "sneak through the back door slowly",
        "stay quiet until the guard passes",
        "i will tell you when it is safe",
        "the password is whispered only once",
        "follow me and do not make a sound",
        "do not say her name out loud",
        "the door at the end of the hall is unlocked",
        "lean closer i do not want them to hear",
        "we have to move before the lights come back on",
    ],
}


def build_tasks() -> List[TaskData]:
    tasks = []
    for cls, sents in CORPUS.items():
        X = encode_batch(sents)
        y = torch.full((len(sents),), cls, dtype=torch.int64)
        n_train = max(1, int(0.8 * len(sents)))
        label, _ = MODE_PALETTE[cls]
        tasks.append(TaskData(
            name=f"mode_{label}",
            X_train=X[:n_train], y_train=y[:n_train],
            X_test=X[n_train:],  y_test=y[n_train:],
            classes=[cls],
        ))
    return tasks


# ---------------------------------------------------------------------
# Inference helpers — pure trioron public API (no private accesses)
# ---------------------------------------------------------------------


def calibrate_gap_floor(organism, tasks):
    """Diagnostic: report the typical top-vs-runner-up log-lik gap on
    training data, so the chosen NOVELTY_GAP_FLOOR can be justified
    against the actual data distribution. Not used at inference."""
    branch = organism.branches[0]
    z = torch.cat(
        [organism.project_l0(t.X_train.float()) for t in tasks], dim=0
    )
    log_lik = branch.per_class_log_likelihood(z)       # (N, C)
    sorted_ll, _ = log_lik.sort(dim=-1, descending=True)
    gaps = (sorted_ll[:, 0] - sorted_ll[:, 1]).cpu()
    return float(gaps.median()), float(gaps.min())


# ---------------------------------------------------------------------
# Cold-start: train (or load cached) donor
# ---------------------------------------------------------------------

CACHE_PATH = Path(os.environ.get("TRIORON_DEMO_CACHE", "/tmp/trioron_tts_donor_v3_1.pt"))


def _build_or_load_donor():
    if CACHE_PATH.exists():
        print(f"[trioron-demo] loading cached donor from {CACHE_PATH}")
        org = load_organism(CACHE_PATH)
        return org, build_tasks()
    print(f"[trioron-demo] cold start — encoding corpus + training "
          f"8-mode donor (one time, ~10s)")
    tasks = build_tasks()
    cfg = TrioronConfig(cap_bytes=20_000)  # roomier; 384-dim L0 + 8 modes
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_donor(
        tasks=tasks, label="emotional_tts_v3", out_path=CACHE_PATH,
        seed=42, epochs_per_task=4, config=cfg,
    )
    return load_organism(CACHE_PATH), tasks


print("[trioron-demo] initializing...")
ORG, TASKS = _build_or_load_donor()
GAP_MEDIAN, GAP_MIN = calibrate_gap_floor(ORG, TASKS)
print(f"[trioron-demo] ready. training gap median={GAP_MEDIAN:+.2f}  "
      f"min={GAP_MIN:+.2f}  novelty floor={NOVELTY_GAP_FLOOR:+.2f}")


# ---------------------------------------------------------------------
# Markup parsing + sentence segmentation
# ---------------------------------------------------------------------

_TAG_RE = re.compile(r"<\s*(\w+)\s*>(.*?)<\s*/\s*\1\s*>", re.DOTALL | re.IGNORECASE)
# Keep the punctuation with its segment.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def parse_markup(text: str) -> List[Tuple[Optional[str], str]]:
    """Return [(mode_or_None, span_text), ...] — None = untagged.
    Untagged spans get further split into sentences upstream."""
    parts: List[Tuple[Optional[str], str]] = []
    last = 0
    for m in _TAG_RE.finditer(text):
        if m.start() > last:
            untagged = text[last:m.start()].strip()
            if untagged:
                parts.append((None, untagged))
        mode_name = m.group(1).strip().lower()
        span = m.group(2).strip()
        if span:
            parts.append((mode_name, span))
        last = m.end()
    if last < len(text):
        tail = text[last:].strip()
        if tail:
            parts.append((None, tail))
    return parts


def classify_segment(text: str) -> Tuple[str, Dict[str, float], float, str]:
    """Run trioron on one untagged segment. Manifold-argmax for
    classification (the trained head's full-softmax is unreliable
    under masked-CE continual training — only the per-class manifold
    archive captures the trained discriminator). Novelty fires when
    the top class isn't decisively above the runner-up — i.e., the
    trioron itself is unsure between modes.

    Returns (mode_label, preset, top_log_lik, source).
    `source` is 'trioron' (decisive) or 'neutral' (no class decisive).
    """
    branch = ORG.branches[0]
    x = encode(text).unsqueeze(0)
    z = ORG.project_l0(x)

    log_lik = branch.per_class_log_likelihood(z)[0]       # (C,)
    sorted_ll, sorted_idx = log_lik.sort(descending=True)
    top_log_lik = float(sorted_ll[0])
    runnerup_log_lik = float(sorted_ll[1]) if len(sorted_ll) > 1 else float("-inf")
    gap = top_log_lik - runnerup_log_lik

    if gap < NOVELTY_GAP_FLOOR:
        return NEUTRAL_MODE[0], NEUTRAL_MODE[1], top_log_lik, "neutral"

    pred_global = int(branch.archive_classes[int(sorted_idx[0])])
    label, preset = MODE_PALETTE[pred_global]
    return label, preset, top_log_lik, "trioron"


def decide_segments(
    text: str,
) -> Tuple[List[Dict], str]:
    """Top-level: parse markup, split untagged spans into sentences,
    classify each sentence. Returns (rows, summary)."""
    text = (text or "").strip()
    if not text:
        return [], "(empty input)"
    parsed = parse_markup(text)
    rows: List[Dict] = []
    for mode_tag, span in parsed:
        if mode_tag is not None:
            preset = _PALETTE_BY_NAME.get(mode_tag)
            if preset is None:
                # Unknown markup tag → fall back to trioron classification.
                rows.append(_row_for_segment(
                    span, source_override="unknown_tag",
                ))
            else:
                rows.append({
                    "segment": span,
                    "mode": mode_tag,
                    "speed": preset["speed"],
                    "volume": preset["volume"],
                    "score": "—",
                    "source": "markup",
                })
        else:
            for sentence in split_sentences(span):
                rows.append(_row_for_segment(sentence))
    summary = (
        f"{len(rows)} segment{'s' if len(rows) != 1 else ''} → "
        f"{', '.join(r['mode'] for r in rows)}"
    )
    return rows, summary


def _row_for_segment(
    sentence: str, source_override: Optional[str] = None,
) -> Dict:
    label, preset, score, source = classify_segment(sentence)
    if source_override:
        source = source_override
    return {
        "segment": sentence,
        "mode": label,
        "speed": preset["speed"],
        "volume": preset["volume"],
        "score": f"{score:+.1f}",
        "source": source,
    }


def decide_raw_for_ui(text: str, backend: str = "Web Speech (browser)"):
    """No-trioron baseline: speak the whole input as ONE utterance with
    the neutral preset. No segmentation, no manifold scoring, no mode
    routing. The "before" the user A/Bs against the trioron-routed
    output to hear what the trioron is actually contributing."""
    text = (text or "").strip()
    if not text:
        return [], "(empty input)", "[]", None
    speed = NEUTRAL_MODE[1]["speed"]
    volume = NEUTRAL_MODE[1]["volume"]
    intensity = NEUTRAL_MODE[1]["intensity"]
    table = [[text, "neutral (raw)", speed, volume, "—", "no-trioron"]]
    summary = "raw passthrough — one utterance, neutral preset, no trioron"
    if backend == "Piper (server)":
        rows = [{
            "segment": text, "mode": "neutral",
            "speed": speed, "intensity": intensity, "volume": volume,
        }]
        sr, audio = render_with_piper(rows)
        return table, summary, "[]", (sr, audio)
    import json
    speech_payload = json.dumps([
        {"text": text, "rate": speed, "volume": volume},
    ])
    return table, summary, speech_payload, None


def decide_for_ui(text: str, backend: str = "Web Speech (browser)"):
    """Top-level Gradio handler. Always returns the per-segment table
    + summary + a Web-Speech JSON payload + an audio tuple for the
    server-side renderer. Whichever output channel matches the
    selected backend gets used; the other is empty/None."""
    rows, summary = decide_segments(text)
    if not rows:
        return [], summary, "[]", None
    table = [
        [r["segment"], r["mode"], r["speed"], r["volume"], r["score"], r["source"]]
        for r in rows
    ]
    if backend == "Piper (server)":
        # Server-side TTS render. Cold-start cost only on first call.
        # Add intensity to each row for Piper's noise_scale mapping.
        for r in rows:
            label = r["mode"]
            for cid, (lab, preset) in MODE_PALETTE.items():
                if lab == label:
                    r["intensity"] = preset["intensity"]
                    break
            else:
                r["intensity"] = NEUTRAL_MODE[1]["intensity"]
        sr, audio = render_with_piper(rows)
        # Empty JS payload so the .then() branch is a no-op.
        return table, summary, "[]", (sr, audio)
    # Web Speech (browser) path: encode segments as JSON for the JS
    # callback; no server-side audio.
    import json
    speech_payload = json.dumps([
        {"text": r["segment"], "rate": r["speed"], "volume": r["volume"]}
        for r in rows
    ])
    return table, summary, speech_payload, None


# ---------------------------------------------------------------------
# Client-side speech (queue every segment in order)
# ---------------------------------------------------------------------

SPEAK_JS = """
(payload_json) => {
  if (!('speechSynthesis' in window)) {
    console.warn('[trioron] Web Speech API not available.');
    return;
  }
  let segs;
  try {
    segs = JSON.parse(payload_json || '[]');
  } catch (e) {
    console.warn('[trioron] bad payload:', e, payload_json);
    return;
  }
  if (!segs.length) return;

  const speakAll = () => {
    if (window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
    }
    const voices = window.speechSynthesis.getVoices();
    const en = voices.find(v => v.lang && v.lang.startsWith('en')) || voices[0];
    segs.forEach((s, i) => {
      const u = new SpeechSynthesisUtterance(String(s.text));
      u.rate = parseFloat(s.rate) || 1.0;
      u.volume = parseFloat(s.volume) || 1.0;
      if (en) { u.voice = en; u.lang = en.lang || 'en-US'; }
      u.onerror = (e) => console.warn('[trioron] seg ' + i + ' error:',
                                      e.error || e);
      window.speechSynthesis.speak(u);
    });
  };

  if (window.speechSynthesis.getVoices().length === 0) {
    window.speechSynthesis.addEventListener('voiceschanged', speakAll,
                                            { once: true });
    setTimeout(speakAll, 300);
  } else {
    speakAll();
  }
}
"""


# ---------------------------------------------------------------------
# Tab 2: Book Memory — trioron as a 2.5 MB compressed-book-memory layer
# in front of a frozen 540 MB SmolLM2-135M-Instruct.
#
# Shows the same input answered two ways:
#   - "LLM alone": baseline SmolLM2 with no book context.
#   - "LLM + Trioron": the same LLM, with a 16-token soft prompt
#     emitted by a trioron head trained on Around the World in 80 Days
#     and Alice's Adventures in Wonderland (in that order, with EWC +
#     dream-replay across the two books).
# ---------------------------------------------------------------------

import json as _json

BOOK_LLM_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
BOOK_HEAD_PATH = Path(__file__).parent / "book_memory" / "head_bio_alice_v2.pt"
BOOK_ARCHIVE_PATH = Path(__file__).parent / "book_memory" / "entity_archive.pt"
BOOK_80DAYS_QA_PATH = Path(__file__).parent / "book_memory" / "questions.json"
BOOK_ALICE_QA_PATH = Path(__file__).parent / "book_memory" / "alice_questions.json"

_BOOK_LLM = None
_BOOK_TOKENIZER = None
_BOOK_HEAD = None
_BOOK_ARCHIVE = None
_BOOK_N_SOFT = 16


def _load_book_stack():
    """Lazy load LLM + tokenizer + trioron head + entity archive once."""
    global _BOOK_LLM, _BOOK_TOKENIZER, _BOOK_HEAD, _BOOK_ARCHIVE
    if _BOOK_LLM is not None:
        return
    print(f"[book-memory] loading {BOOK_LLM_NAME}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _BOOK_TOKENIZER = AutoTokenizer.from_pretrained(BOOK_LLM_NAME)
    if _BOOK_TOKENIZER.pad_token is None:
        _BOOK_TOKENIZER.pad_token = _BOOK_TOKENIZER.eos_token
    _BOOK_LLM = AutoModelForCausalLM.from_pretrained(
        BOOK_LLM_NAME, dtype=torch.float32,
    )
    _BOOK_LLM.eval()
    for p in _BOOK_LLM.parameters():
        p.requires_grad_(False)

    print(f"[book-memory] loading head from {BOOK_HEAD_PATH}...")
    from book_memory import build_head_from_ckpt, EntityArchive
    ckpt = torch.load(BOOK_HEAD_PATH, map_location="cpu", weights_only=False)
    _BOOK_HEAD = build_head_from_ckpt(ckpt, "cpu")
    _BOOK_HEAD.eval()
    n_params = _BOOK_HEAD.n_parameters()
    n_grows = ckpt.get("stats", {}).get("n_grow_events", "?")
    print(f"[book-memory] head: {n_params:,} params "
          f"(~{n_params * 4 / 1024:.1f} KB fp32, {n_grows} grow events)")

    if BOOK_ARCHIVE_PATH.exists():
        _BOOK_ARCHIVE = EntityArchive.load(BOOK_ARCHIVE_PATH)
        size_kb = BOOK_ARCHIVE_PATH.stat().st_size / 1024
        print(f"[book-memory] entity archive: {len(_BOOK_ARCHIVE)} entities, "
              f"τ={_BOOK_ARCHIVE.threshold:.2f}, ~{size_kb:.1f} KB")
    else:
        print(f"[book-memory] (no entity archive at {BOOK_ARCHIVE_PATH} — "
              f"entity routing disabled)")


@torch.no_grad()
def _book_generate_baseline(question: str, max_new_tokens: int = 80) -> str:
    """LLM-only path. Tokenize the prompt, generate greedily, decode."""
    _load_book_stack()
    prompt = f"Q: {question}\nA: "
    enc = _BOOK_TOKENIZER(prompt, return_tensors="pt", add_special_tokens=False)
    out = _BOOK_LLM.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=_BOOK_TOKENIZER.pad_token_id,
        eos_token_id=_BOOK_TOKENIZER.eos_token_id,
    )
    new_ids = out[0, enc["input_ids"].shape[1]:]
    return _BOOK_TOKENIZER.decode(new_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def _entity_lookup(question: str) -> Optional[Tuple[str, float]]:
    """Cosine retrieval over the entity archive. Returns (entity_text, score)
    when a key scores above the archive's threshold; None otherwise.

    The archive lives in pooled-LLM-embedding space, so the lookup vector
    is the same `pooled` we feed to the head. No extra encoder needed."""
    if _BOOK_ARCHIVE is None or len(_BOOK_ARCHIVE) == 0:
        return None
    from book_memory import pool_query_embeddings
    prompt = f"Q: {question}\nA: "
    enc = _BOOK_TOKENIZER(prompt, return_tensors="pt", add_special_tokens=False)
    embed_layer = _BOOK_LLM.get_input_embeddings()
    pooled = pool_query_embeddings(embed_layer(enc["input_ids"]),
                                   enc["attention_mask"])
    hit = _BOOK_ARCHIVE.lookup(pooled.squeeze(0))
    if hit is None:
        return None
    ent_id, score = hit
    return _BOOK_ARCHIVE.text_of(ent_id), score


@torch.no_grad()
def _book_generate_trioron(question: str, max_new_tokens: int = 80) -> str:
    """LLM + Trioron path. Two-stage routing:

    1. Cosine-lookup the entity archive. If a key scores above τ
       (typically 0.95), the trioron has confidently decided this
       question maps to a specific named entity in its archive — return
       that entity text directly. No LLM call. The archive *is* the
       trioron's voice for entity recall; routing through the LLM only
       degrades it (the LLM ignores hints and rambles on greedy decode).
    2. Otherwise: current path — pool query embeddings, run through head,
       prepend 16-token soft prompt, generate. Stylistic conditioning
       is the lift here; entity facts are not expected.

    The archive route is what makes the demo's headline question
    ('What is the name of Phileas Fogg's manservant?' → 'Passepartout')
    actually work.
    """
    _load_book_stack()
    hit = _entity_lookup(question)
    if hit is not None:
        entity_text, score = hit
        return (f"{entity_text}.  _(entity archive · cos {score:.3f} · "
                f"no LLM call)_")

    # Fallback: soft-prompt path (current behavior, kept for descriptive
    # questions where the head's stylistic conditioning is the main lift).
    from book_memory import pool_query_embeddings
    prompt = f"Q: {question}\nA: "
    enc = _BOOK_TOKENIZER(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    mask = enc["attention_mask"]

    embed_layer = _BOOK_LLM.get_input_embeddings()
    q_emb = embed_layer(ids)                            # (1, Lq, D)
    pooled = pool_query_embeddings(q_emb, mask)         # (1, D)
    soft = _BOOK_HEAD(pooled)                           # (1, n_soft, D)
    inputs_embeds = torch.cat([soft, q_emb], dim=1)
    soft_mask = torch.ones(1, _BOOK_N_SOFT, dtype=mask.dtype)
    full_mask = torch.cat([soft_mask, mask], dim=1)

    out = _BOOK_LLM.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        pad_token_id=_BOOK_TOKENIZER.pad_token_id,
        eos_token_id=_BOOK_TOKENIZER.eos_token_id,
    )
    # When inputs_embeds is used, generate returns only the NEW tokens.
    return _BOOK_TOKENIZER.decode(out[0], skip_special_tokens=True).strip()


def _load_book_question_sets():
    """Plain list of preset-question strings, prefixed with the book
    they came from. The dropdown displays them verbatim; the handler
    strips the prefix to recover the raw question."""
    items: List[str] = []
    for label, path in [
        ("80 Days", BOOK_80DAYS_QA_PATH),
        ("Alice", BOOK_ALICE_QA_PATH),
    ]:
        if path.exists():
            try:
                data = _json.loads(path.read_text())
                for q in data.get("questions", []):
                    items.append(f"[{label}] {q['question']}")
            except Exception as e:
                print(f"[book-memory] failed to parse {path}: {e}")
    return items


def _strip_book_prefix(s: str) -> str:
    """[80 Days] foo → foo. Leaves un-prefixed strings alone."""
    if s and s.startswith("[") and "] " in s:
        return s.split("] ", 1)[1]
    return s


def book_qa_for_ui(preset_str: str, custom_question: str):
    """Top-level Gradio handler. Picks the question (custom takes
    priority over preset), generates both baseline and trioron answers,
    returns markdown for both panes plus a one-line stats summary."""
    custom = (custom_question or "").strip()
    preset_q = _strip_book_prefix(preset_str or "").strip()
    question = custom or preset_q
    if not question:
        return ("_(pick a preset or type a question)_",
                "_(pick a preset or type a question)_",
                "")

    base = _book_generate_baseline(question)
    trio = _book_generate_trioron(question)

    base_md = f"**Q:** {question}\n\n**A:** {base or '_(empty)_'}"
    trio_md = f"**Q:** {question}\n\n**A:** {trio or '_(empty)_'}"
    n_params = _BOOK_HEAD.n_parameters() if _BOOK_HEAD else 0
    summary = (
        f"head: {n_params:,} params (~{n_params * 4 / 1024 / 1024:.2f} MB fp32) — "
        f"trained on 80 Days + Alice via frustration-driven growth + "
        f"EWC + cross-book dream replay."
    )
    return base_md, trio_md, summary


BOOK_DESCRIPTION = """
**Trioron as a small router over a frozen LLM.** The frozen LLM is
[`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
(~270 MB fp32). Two paths the trioron picks between:

1. **Entity archive.** A ~290 KB sidecar of cosine keys, one per
   stored question. When the user's query matches a stored question
   closely enough (cos ≥ 0.95), the trioron returns the entity that
   question pointed at — no LLM call. Structurally this is the same
   shape as a 1990s retrieval-based chatbot, just with learned
   embedding keys instead of hand-written regex patterns.

2. **Stylistic conditioning.** When the archive doesn't fire, the
   2.5 MB head emits a 16-token continuous soft prompt that biases
   SmolLM2-135M toward Victorian-novel cadence. Period prose, not factual
   recall.

Why two paths: a soft prompt has limited authority over the LLM's
BPE-piece prior. Names like "Passepartout" fragment into 4 rare BPE
pieces (`P` `asse` `part` `out`); navigating that corridor under
greedy decode from soft-prompt distillation alone is hard. The
archive sidesteps it — when the trioron knows, it just answers;
otherwise SmolLM2-135M rambles in period style.

**Honest scope.** Compared to a typical RAG stack over the same
books (chunked text + FAISS index + a separate ~80 MB sentence-
encoder like MiniLM), the lookup table is comparable in size to the
index — the saving is reusing the LLM's own embedding layer instead
of shipping a second model. Where RAG retrieves raw passages, this
archive only stores short entity labels — descriptive questions
("describe the tea-party") still go through the soft-prompt fallback
and confabulate plausibly, not factually. **The trioron contributes
~3 MB of personalisation + entity-recall next to the 270 MB frozen
LLM. It does not replace passage-level retrieval.**

Paraphrase tolerance is bounded by what's in the archive. A query
close to a stored question hits; a query in unfamiliar phrasing
falls through to the soft-prompt path. We've hand-augmented the
preset entities with a few paraphrases each so common rewordings
work, but novel phrasings will miss.

*Private demo — please do not share until the paper publishes.*
"""


# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------

EXAMPLES = [
    "the lake is quiet today",
    "you are safe with me now",
    "warning slow down obstacle ahead",
    "i wonder why the sky turns red at dusk",
    "the old house is empty now",
    # The mixed-emotion paragraph from Rocky's spec, plain text:
    "Whoa, look at that view! The sky is very clear today. "
    "Let's have a picnic by the lake. But remember the umbrella, "
    "we got soaked last time and i still feel bad about it.",
    # Same paragraph with explicit markup overrides:
    "<excited>Whoa, look at that view!</excited> "
    "<calm>The sky is very clear today.</calm> "
    "<curious>Let's have a picnic by the lake.</curious> "
    "<sad>But remember the umbrella, we got soaked last time.</sad>",
]

DESCRIPTION = """
**Trioron as device-side context memory for TTS.**  Reads each
sentence (or each `<mode>tag</mode>` span) and decides how it should
be spoken — speed, volume, intonation — replacing what would
normally take a multimodal model.

- **Plain text** auto-splits on sentence boundaries; trioron picks a
  mode per sentence.
- **Markup** like `<excited>Whoa!</excited> <calm>The sky is clear.</calm>`
  forces a specific mode for that span.
- Inputs that match no learned mode trip the **neutral** fallback —
  the trioron's way of saying "I haven't been taught this."

Two buttons for direct A/B:
- **Speak (with trioron)** — segment + route + per-segment preset.
- **Speak raw (no trioron)** — same text, one neutral preset, no
  routing. The "before" picture so you can hear what the trioron
  is actually contributing.

*Private demo — please do not share until the paper publishes.*
"""

BOOK_PRESETS = _load_book_question_sets()


with gr.Blocks(title="Trioron Demos") as demo:
    gr.Markdown("# Trioron Demos")
    gr.Markdown(
        "Two views of the same trioron substrate: as a tiny personalization "
        "layer in front of a TTS engine, and as a compressed book-memory "
        "layer in front of a small LLM."
    )

    with gr.Tab("Emotional TTS"):
        gr.Markdown("## Trioron TTS — segment-level intonation")
        gr.Markdown(DESCRIPTION)
        text_in = gr.Textbox(
            label="Text to speak (plain or with <mode>...</mode> markup)",
            placeholder="Type a paragraph and click Speak…",
            lines=4,
        )
        backend_in = gr.Radio(
            choices=["Web Speech (browser)", "Piper (server)"],
            value="Web Speech (browser)",
            label="Renderer",
            info=(
                "Web Speech uses your browser's built-in TTS — instant, "
                "voice quality depends on OS. Piper renders server-side "
                "(adds ~200ms/sentence + ~5s on first use to download "
                "the voice); audio quality is consistent across platforms."
            ),
        )
        with gr.Row():
            speak_btn = gr.Button("Speak (with trioron)", variant="primary")
            speak_raw_btn = gr.Button("Speak raw (no trioron)", variant="secondary")
            clear_btn = gr.Button("Clear")
        gr.Examples(examples=EXAMPLES, inputs=text_in)
        summary_out = gr.Textbox(
            label="Routing summary", interactive=False, lines=1,
        )
        table_out = gr.Dataframe(
            headers=["segment", "mode", "speed", "volume", "log-lik", "source"],
            datatype=["str", "str", "number", "number", "str", "str"],
            interactive=False,
            wrap=True,
            label="Per-segment decisions",
        )
        audio_out = gr.Audio(
            label="Server-rendered audio (Piper)",
            autoplay=True,
            interactive=False,
        )
        payload_out = gr.Textbox(visible=False)

        speak_btn.click(
            fn=decide_for_ui, inputs=[text_in, backend_in],
            outputs=[table_out, summary_out, payload_out, audio_out],
        ).then(
            fn=None,
            inputs=[payload_out],
            outputs=None,
            js=SPEAK_JS,
        )
        speak_raw_btn.click(
            fn=decide_raw_for_ui, inputs=[text_in, backend_in],
            outputs=[table_out, summary_out, payload_out, audio_out],
        ).then(
            fn=None,
            inputs=[payload_out],
            outputs=None,
            js=SPEAK_JS,
        )
        clear_btn.click(
            fn=lambda: ([], "—", "[]", None),
            outputs=[table_out, summary_out, payload_out, audio_out],
        )

    with gr.Tab("Book Memory"):
        gr.Markdown("## Trioron as compressed book memory")
        gr.Markdown(BOOK_DESCRIPTION)
        with gr.Row():
            book_preset = gr.Dropdown(
                label="Preset question (optional — or type your own below)",
                choices=BOOK_PRESETS,
                value=None,
            )
        book_custom = gr.Textbox(
            label="Or type your own question",
            placeholder="e.g., Who is Phileas Fogg's manservant?",
            lines=2,
        )
        with gr.Row():
            book_ask_btn = gr.Button("Ask both", variant="primary")
            book_clear_btn = gr.Button("Clear")
        with gr.Row():
            book_baseline_out = gr.Markdown(label="LLM alone")
            book_trioron_out = gr.Markdown(label="LLM + Trioron")
        book_summary_out = gr.Textbox(
            label="Trioron head", interactive=False, lines=1,
        )

        book_ask_btn.click(
            fn=book_qa_for_ui,
            inputs=[book_preset, book_custom],
            outputs=[book_baseline_out, book_trioron_out, book_summary_out],
        )
        book_clear_btn.click(
            fn=lambda: ("", "", "", None, ""),
            outputs=[book_baseline_out, book_trioron_out, book_summary_out,
                     book_preset, book_custom],
        )


if __name__ == "__main__":
    demo.launch()
