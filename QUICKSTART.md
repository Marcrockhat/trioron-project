# Trioron — Quickstart

Reproduce the lossless multi-branch absorption result on commodity CPU
in **under five minutes** of wall time. No GPU required.

The four CLI commands below reproduce the N=2 row of paper §4.6:

| metric | upper bound | organism |
|---|---|---|
| task-aware accuracy | 0.9823 | **0.9823** (Δ = 0) |
| full-union accuracy | — | 0.6099 |

Numbers are deterministic given the L0 seed (default `42`).

---

## 1. Install

### From a local clone (fast, no PyPI account required):

```bash
git clone https://github.com/marcrockhat/trioron-project.git
cd trioron-project
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Directly from GitHub:

```bash
pip install git+https://github.com/marcrockhat/trioron-project.git
```

### From PyPI (when published):

```bash
pip install trioron
```

The install pulls in `torch`, `numpy`, `torchvision`. CPU-only PyTorch
is sufficient — the benchmark runs on a laptop CPU.

---

## 2. Train two donors at the shared L0 seed

Each donor is a self-contained skill pack: a frozen L1 substrate +
head over its trained classes + per-class manifold archive (μ, σ).
Both are born with **the same L0 random projection** (seed 42), which
is what makes paste-and-go absorption work without retraining.

```bash
trioron train --donor digits   --out donor_digits.pt
trioron train --donor fashion  --out donor_fashion.pt
```

Each takes ~1–2 minutes on a CPU. The first run downloads MNIST and
Fashion-MNIST into `outputs/data/`; subsequent runs reuse the cache.

Available splits (paper §4.6):

| split | dataset | global classes |
|---|---|---|
| `digits` | MNIST | 0–9 |
| `fashion` | Fashion-MNIST | 10–19 |
| `emnist` | EMNIST letters A–J | 20–29 |
| `emnist_kt` | EMNIST letters K–T | 30–39 |
| `emnist_uz` | EMNIST letters U–Z (partial) | 40–45 |

---

## 3. Absorb donors into one organism

Zero-shot. No training, no calibration pass, no fusion layer. The
shared-L0 invariant is checked at load time; mismatched seeds fail
fast with a clear error.

```bash
trioron absorb \
  --donors donor_digits.pt,donor_fashion.pt \
  --out organism.pt
```

You should see something like:

```
loaded digits   arch=[128, 52, 10]  classes=[0..9]   l0_seed=42
loaded fashion  arch=[128, 52, 20]  classes=[10..19] l0_seed=42

[trioron absorb] organism with 2 branch(es) → organism.pt
  union_classes = [0..19]
  storage: 471 KB total (L0 392 KB shared, branch substrate 59 KB, archive 20 KB)
```

---

## 4. Evaluate the assembled organism

Runs each donor's test split through the assembled organism and
reports task-aware (the production metric for trioron's deployment as
a context-conditioned classifier) and full-union (argmax over all
covered classes) accuracy.

```bash
trioron eval --organism organism.pt
```

You should see the headline:

```
Headline (mean across union):
  task-aware (production) = 0.9823  full-union = 0.6099  ← soft + per-branch log-softmax
  task-aware (raw)        = 0.9821  full-union = 0.2590  ← soft, no normalization
```

The first line matches the paper's lossless-absorption claim; the
second line shows what happens without per-branch log-softmax
calibration.

---

## 5. Run inference on a single image

```bash
trioron infer --organism organism.pt --image path/to/image.png
```

The image is auto-resized to 28×28 grayscale (matching the chained-15
input shape). Output reports the routing gates per branch and the
top-k union-class predictions. For an MNIST digit the digits branch
should fire at ~1.0 and the fashion branch at ~0.0.

---

## What's the point of all this?

`trioron eval` reports `task-aware = 0.9823`, which is the
**donor-standalone upper bound** to four decimal places. The paper's
core claim is that you get this for free at composition time —
zero-shot, with no calibration training, under the shared-L0
invariant. The mechanism scales lossless to N=5 donors / 46 classes
on chained-15-scale data (paper §4.6).

The full reproduction of §4.6 (saturation curve, BTM baselines, dream-
cycle calibration) lives in `experiments/`:

```bash
# Saturation curve N=1..5
python3 -m experiments.bench_saturation_curve

# Branch-Train-Merge comparison at N=3
python3 -m experiments.bench_btm_baseline

# Dream-cycle calibration trade-off
python3 -m experiments.dream_cycle_calibration
```

These are research scripts with experimental knobs; the `trioron` CLI
above exposes only the curated production path.

---

## Going further

- Methods: `paper/methods.tex` (compile to `methods.pdf`).
- Results: `paper/results.tex` (compile to `results.pdf`).
- Architecture deep dive: `trioron_blueprint.md`.
- Cross-modal bridge (encoders + tool dispatch): `trioron/bridge/`.

For deployment as an orchestrator over a frozen LLM, see
`trioron/bridge/` (encoders for text/image/audio, tool dispatcher with
JSON-schema and decorator interfaces).
