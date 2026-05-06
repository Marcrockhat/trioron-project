---
title: Trioron Demos
emoji: 🗣️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.0
app_file: app.py
pinned: false
license: mit
short_description: Trioron as a small router — TTS prosody + book entity recall
---

# Trioron TTS Demo

A live demo of **trioron** — a tiny task-aware substrate
(~145 KB on disk) that learns from a handful of examples and runs
on-device. Here it reads a sentence and decides how it should be
spoken (speed, intensity, volume), replacing what would normally take
a multimodal "feel-the-scene" model.

> **Private demo — please do not share publicly until the paper
> publishes.** This Space is unlisted but the link is shareable, so
> please keep it within the review group.

---

## Contents

1. [What this demonstrates](#1-what-this-demonstrates)
2. [How to use the UI](#2-how-to-use-the-ui)
3. [Things to try (demo script)](#3-things-to-try-demo-script)
4. [What's happening under the hood](#4-whats-happening-under-the-hood)
5. [The mode palette](#5-the-mode-palette)
6. [Limitations and known issues](#6-limitations-and-known-issues)
7. [Where this fits in the trioron paper](#7-where-this-fits-in-the-trioron-paper)
8. [Book Memory tab](#8-book-memory-tab)

---

## 1. What this demonstrates

The standard way to make a TTS engine sound emotionally appropriate
is to run a multimodal model (vision-language or speech-conditional)
that "understands" the scene and emits style tokens or prosody
parameters. Those models are 100MB–10GB and need a GPU.

Trioron's claim is different: a 145 KB task-aware memory, *taught
from a handful of in-context examples per mode*, can fill the same
role for narrow deployment domains. No multimodal model, no GPU,
microseconds per inference.

The Space lets you poke at this directly. You type text, the trioron
classifies the sentence into one of four learned modes (calm, gentle,
firm, urgent), and your browser renders the audio with the
trioron-chosen rate and volume. When the input doesn't match any
learned mode, a **novelty gate** trips and the *excited* fallback
fires — the trioron's way of saying "I haven't been taught this; I'm
not going to pretend I have."

Audio is rendered one of two ways, selectable via the **Renderer**
radio in the UI:

- **Web Speech (browser)** — default. Your browser's built-in TTS,
  zero server cost. Voice quality depends on your OS. No audio bytes
  cross the network — only the text and the trioron-decided
  parameters do.
- **Piper (server)** — a local ONNX TTS model running on the Space.
  Consistent voice across platforms, applies trioron's `intensity`
  knob (mapped to Piper's `noise_scale`), ~200 ms render per
  sentence. First Piper request triggers a one-time ~60 MB voice
  download.

---

## 2. How to use the UI

1. **Wait for the Space to wake up.** First request after a 48-hour
   sleep takes ~15–25s — the encoder downloads (~80MB if not cached)
   and the donor trains from scratch on the in-context examples.
   Subsequent requests are ~50–100ms (encoder forward + trioron is
   microseconds; encoder dominates).
2. **Type a sentence** in the text box. Or click one of the example
   chips below it.
3. Two buttons let you A/B-test the trioron's contribution:
   - **Speak (with trioron)** — segment, route, per-segment preset.
   - **Speak raw (no trioron)** — same text, one neutral preset, no
     routing. Hit both with the same input and the difference IS
     what the trioron is adding.
4. The right-hand panel shows the trioron's decision:
   - **Predicted mode** — `calm` / `gentle` / `firm` / `urgent` /
     `excited` (the last one only via novelty fallback).
   - **speed / intensity / volume** — the TTS parameters dispatched
     to your browser.
5. Below that:
   - **Decision** — a one-liner with the manifold log-likelihood
     score and whether the novelty gate tripped.
   - **Per-class manifold log-lik** — the full breakdown: log-pdf
     under each learned class's Gaussian. The `max` row is what the
     decision is based on.
6. Your browser plays the audio with the chosen `rate` and `volume`.

If you don't hear audio: your OS may not have a default voice
installed, or the Web Speech API is disabled in your browser. Linux
Chrome usually needs `espeak` installed; macOS / Windows / iOS / most
Android browsers have voices out of the box.

---

## 3. Things to try (demo script)

These are the experiments worth running, in order:

### 3a. Known sentences route to the right mode

Try one sentence per mode (the example chips already cover these):

| Sentence                                  | Expected mode |
|-------------------------------------------|---------------|
| "the lake is quiet today"                 | calm          |
| "you are safe with me now"                | gentle        |
| "stop right there and listen carefully"   | firm          |
| "warning slow down obstacle ahead"        | urgent        |

You'll hear the audio play with noticeably different pace and
volume per mode. The "max-log-lik" should be in the `+200` range for
all four — that's the trioron saying "I have a strong prior that this
text is in this context."

### 3b. Sentence variations that still belong to a learned mode

Try sentences the trioron *was not specifically trained on* but that
share vocabulary with a learned mode:

| Sentence                                  | Likely route |
|-------------------------------------------|--------------|
| "the morning lake is calm and blue"       | calm         |
| "everything will be okay i promise"       | gentle       |
| "stop and put your hands down now"        | firm         |
| "evacuate immediately fire alarm"         | urgent       |

Most should still route correctly with somewhat lower confidence
(maybe `+150` to `+200`). This is what "task-aware generalization"
looks like in practice: the trioron's manifold archive is wide enough
to recognize new in-distribution sentences.

### 3c. Novel sentences trip the excited fallback

Try sentences that have nothing to do with any learned mode:

| Sentence                                            | Expected |
|-----------------------------------------------------|----------|
| "fascinating discovery in quantum chemistry"        | excited (novelty) |
| "the catalyst molecule rearranges the lattice"     | excited (novelty) |
| "purple jellyfish glow under ultraviolet light"    | excited (novelty) |

The "max-log-lik" will be deeply negative (sometimes huge magnitude
like `-1e9` because of the manifold's near-zero variance on
unseen-vocabulary dimensions — see [§6](#6-limitations-and-known-issues)).
Below the calibrated threshold, the novelty gate fires:
*"ooh, that's new!  excited mode."*

This is the deliberately-designed "don't be a know-it-all party
ruiner" behavior. The trioron flags ignorance instead of
confabulating a confident answer.

### 3d. Sanity: the threshold is meaningful

Compare the *known* and *novel* max-log-lik values shown in the
"Decision" line. The gap is enormous (often `+200` vs `-3000`),
which is what makes the novelty gate work even with a hand-tuned
threshold. The threshold itself is set at the 10th percentile of
training-set max-log-lik — robust to outliers, no
hyperparameter-sweep needed.

---

## 4. What's happening under the hood

```
text  →  hash-bag encoder  →  z₀ (64-dim)
                                  │
                                  ▼
                          frozen L0 random
                          projection (128-dim)
                                  │
                                  ▼
                              z (L0 code)
                            ┌─────┴─────┐
                            ▼           ▼
                    grown L1 + head   manifold archive
                       │                 │
                       ▼                 ▼
                  4-class logits   per-class log p(z|c)
                                         │
                                         ▼
                                  max over classes
                                         │
                              ┌──────────┴──────────┐
                       max ≥ thresh           max < thresh
                              │                     │
                              ▼                     ▼
                    dispatch matching         dispatch
                    mode's preset           excited preset
```

**Encoder** (`sentence-transformers/all-MiniLM-L6-v2`): real semantic
embeddings, 384-dim, L2-normalized. ~22M params, ~80MB on disk,
~50ms per encode on CPU. The model loads once on cold start and
stays in memory; subsequent encodes are fast. The trioron sees
inputs that genuinely encode meaning, not just lexical overlap —
"the sky is very clear" and "good morning" cluster together as
*calm* even though they share zero words.

(v1/v2 of this Space used a hash-bag-of-words encoder, which couldn't
tell semantic neighborhoods apart. v3 swapped in MiniLM. The trioron
itself didn't change — it just got intelligible inputs.)

**Trioron substrate**: standard chained-curriculum training. The 4
modes are 4 sequential tasks; each adds 1 head class and lets the
trioron grow new L1 capacity if needed. Total ~3500 parameters
trainable, ~145 KB on disk.

**Manifold archive**: per-class diagonal Gaussian over L0 code-space,
fit once per task as a side-effect of training. Stores `(μ_c, σ_c)`
per class. Doubles as the novelty signal at inference: the per-class
log-pdf is computed in closed form (no sampling), and the max across
classes is the confidence.

**Novelty threshold**: the 10th percentile of training-set max-log-lik,
minus a 5% margin. Calibrated automatically on cold start. No manual
tuning.

**Inference cost**: encode (~50µs) + L0 forward (~20µs) + L1 forward
(~10µs) + manifold log-lik over 4 classes (~15µs) ≈ 100µs total on
CPU. Web Speech adds whatever your OS's TTS latency is (~200ms
typical for the first chunk).

---

## 5. The mode palette

| Mode      | speed | intensity | volume | Example                            |
|-----------|------:|----------:|-------:|------------------------------------|
| calm      | 0.80  | 0.30      | 0.55   | "the lake is quiet today"          |
| gentle    | 0.82  | 0.40      | 0.40   | "you are safe with me now"         |
| firm      | 1.05  | 0.85      | 0.85   | "stop right there..."              |
| urgent    | 1.50  | 1.10      | 1.00   | "warning slow down obstacle ahead" |
| excited   | 1.40  | 1.15      | 0.95   | "whoa look at that view"           |
| sad       | 0.70  | 0.20      | 0.40   | "i miss the days when we were together" |
| curious   | 1.10  | 0.75      | 0.70   | "i wonder why the sky is red"      |
| whispered | 0.95  | 0.25      | 0.18   | "the secret is in the attic"       |
| neutral   | 1.00  | 0.60      | 0.70   | (novelty fallback only)            |

`speed` and `volume` are passed through to `SpeechSynthesisUtterance.rate`
and `.volume` directly. `intensity` is shown for completeness — Web
Speech doesn't expose a per-call intensity knob, so it's not applied
in this demo (it would be in a real Kokoro/Coqui backend, which
exposes more parameters).

---

## 6. Limitations and known issues

**Voice quality depends on your OS.** Web Speech uses your operating
system's TTS engine. macOS, Windows, and iOS sound natural; desktop
Linux Chrome typically uses `espeak` and sounds robotic. This is a
property of Web Speech, not of trioron — the trioron's *parameter
choice* is identical across platforms; only the renderer differs. A
real deployment would use Kokoro, Piper, or Coqui for consistent
voice quality.

**`intensity` isn't applied.** Web Speech only exposes
`rate` / `pitch` / `volume`. The intensity dimension shows what
trioron decided but the browser ignores it. Swap in a TTS engine
that exposes intensity (most local engines do) to use it.

**Manifold log-lik can have huge magnitude on novel inputs.** When a
sentence uses vocabulary the manifold has zero variance on, the
per-class log-pdf can be `-1e9` or worse. This is a real numerical
property, not a bug — and it makes the novelty gate easier (the
"known" vs "novel" gap is enormous). Don't read magnitudes literally;
read whether they're above or below the calibrated threshold.

**Cold start trains the donor from scratch.** First request after the
Space sleeps for 48h takes ~5–10s. After that, requests are
microseconds. The trained donor is cached at `/tmp/trioron_tts_donor.pt`.

**Single donor, no absorption demo here.** The Space shows trioron's
classification + novelty mechanism but not multi-donor absorption,
seed-mismatch fallback, or the dream-archive deployment loop. Those
are demonstrated in the project repo's `examples/` and `experiments/`
directories.

**The encoder dominates inference latency.** The trioron forward is
~30µs; MiniLM encoding is ~50ms (CPU). For a real edge deployment
you'd swap MiniLM for a smaller distilled model (~5ms) or a
domain-specific encoder. The trioron handles either without change.

---

## 7. Where this fits in the trioron paper

This Space is a single concrete instance of the deployment pattern
described in §4.6 of the paper (the "device-conscience" / ship-wake-
extend loop). It exercises:

- **Continual learning** — 4 modes taught as 4 sequential tasks
  without catastrophic forgetting. Original modes still work after
  later modes are added.
- **Manifold-driven routing + novelty detection** — the same
  per-class Gaussian archive used for replay during training is also
  the runtime confidence signal. One mechanism, two roles.
- **Sub-MB deployment** — the entire learned context memory is 145 KB.
  Replaces what a 100MB+ multimodal model would otherwise do for the
  context-to-prosody decision.

The pieces *not* shown in this Space (multi-donor absorption,
random-projection seed-mismatch fallback, the dream-archive int8
quantization simulation, the extend-from-substrate resume path) are
exercised in the project's regression tests and bench scripts.

For the full picture see the paper draft and `MANUAL.md` in the
project repo.

---

## 8. Book Memory tab

A second tab demonstrating the same trioron substrate in a different
role: a small router/memory layer in front of a frozen language
model.

**The frozen LLM is
[`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)**
— ~270 MB at fp32. It is loaded once and never updated. All trioron
weights sit in front of it.

What the trioron contributes (~3 MB total):

- **Entity archive** (~290 KB sidecar). Cosine retrieval over per-
  question keys. When the user's question matches a stored question
  closely enough (cos ≥ 0.95), the trioron returns the entity that
  question pointed at, with no LLM call. Structurally a 1990s
  retrieval-based chatbot table — every (question, entity) pair is a
  searchable record. The keys are learned embeddings (mean-pooled
  SmolLM2-135M input embeddings), which is what makes natural paraphrases
  cluster.
- **Stylistic conditioning head** (~2.5 MB). When the archive
  doesn't fire, the head emits a 16-token continuous soft prompt
  that biases SmolLM2-135M toward Victorian-novel cadence. The output is
  fluent and period-flavored, but it is *not* factual recall.

Why the two paths: a soft prompt has limited authority over
SmolLM2-135M's BPE-piece prior. Names like "Passepartout" fragment into
4 rare BPE pieces; navigating that corridor under greedy decode
from soft-prompt distillation alone is hard. The archive sidesteps
that — when the trioron knows, it just answers.

**Honest comparison with RAG.** A typical RAG stack over the same
books would consist of: chunked text (~1 MB), a FAISS / hnswlib
index (~200 KB at int8), and a separate sentence-encoder model
(~80 MB, e.g. MiniLM). The trioron's lookup table is roughly the
same size as the RAG index — the saving is reusing the LLM's own
embedding layer for the lookup vectors instead of shipping a second
model. Where RAG can paste back a raw passage for "describe the tea-
party," the archive only stores short entity labels, so descriptive
questions go through the soft-prompt fallback and confabulate
plausibly rather than factually. The trioron is not a replacement
for passage-level RAG; it is a compact entity-recall + style layer.

**Books.** The head + archive are trained on
*Around the World in Eighty Days* (Verne) and *Alice's Adventures
in Wonderland* (Carroll), both Project Gutenberg public-domain
texts. ~1052 (Q, A) pairs across the two books, plus a small set of
hand-written paraphrase variants for the entity-presets so the
archive tolerates rewording.

**Limits.** The archive is not robust to question phrasings that
diverge from anything stored. A query like "What does Mr Fogg call
his butler?" misses where "Who is Phileas Fogg's manservant?" hits,
because the stored keys cluster around the latter shape. We treat
this as a feature of an honestly small system rather than a flaw to
hide — the trioron is competent inside the support of its training
data and falls through cleanly outside it.
