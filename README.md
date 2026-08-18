# Trioron — an epigenetic-inspired self-expanding architecture

A continual-learning architecture built around the **trioron**: a node with three coupled state variables (weight, plasticity coefficient, utility) that grows, prunes, and consolidates under a per-curriculum byte budget. Designed for device-conscious deployment on agentic-AI / IoT / embedded hardware.

The full design is in `trioron_blueprint.md`. The paper draft is in `paper/`.

- **Want a 5-min reproduction of the paper headline?** → [QUICKSTART.md](QUICKSTART.md)
- **Want to build your own trioron network and deploy it as an agent?** → [MANUAL.md](MANUAL.md)
- **Just the cross-modal bridge / encoders?** → [BRIDGE.md](BRIDGE.md)
- **Want a visual walk-through of the mechanisms?** → [tour/](tour/) — a static
  Canvas petri-dish demo, thirteen chapters with one knob each. Live at
  <https://marcrockhat.github.io/trioron-project/tour/>.
- **Want an example of what it can do?** → Check out <https://huggingface.co/spaces/Marcrockhat/trioron-demo>.

## Headline

On a 30-class class-incremental curriculum (chained-15: MNIST → Fashion-MNIST → EMNIST-letters), with growth + dreaming + manifold replay enabled:

- **0.601 ± 0.008 full-softmax / 0.677 ± 0.007 domain-aware / 0.961 ± 0.001 task-aware** at 30 KB of replay storage (n = 10 seeds, paired).
- σ-confident wins over PackNet, HAT, Online EWC, and LwF + EWC on full-softmax and domain-aware (+10σ to +28σ paired). Matches a K = 50 hippocampal exemplar buffer within 0.04 absolute full-softmax at 1/25th the storage.
- BF16 + int8 dream-archive: **157 KB total deployment** (network + manifold buffer), Δ ≤ 0.0008 lossless.
- Ship-wake-extend loop validated end-to-end at 23 tasks / 46 classes, **168 KB total**, original tasks preserved at task-aware ≥ 0.93.
- Multi-branch absorption: zero-shot composition of independently-trained donors via a 4-byte L0 handshake (R · S factorization); SOFT routing tracks the per-donor upper bound (Δ ≤ 0.0002 task-aware) out to N = 5 donors.

Method and result details: `paper/paper.pdf` (built from `paper/paper.tex`).

## Install

```bash
pip install trioron            # 0.3.1+ — earlier wheels lack trioron.pcll and trioron.api
```

Or straight from GitHub for the latest unreleased changes:

```bash
pip install git+https://github.com/marcrockhat/trioron-project.git
```

Everything a user should import lives in **`trioron.api`**. There are three
ways in, and — the thing that trips people up — **each is fed differently.**
Trioron is not a `fit(X, y)` library; only the first path takes a dataset.

### 1. Continual classification / donors (dataset in, donor out)

The paper's flow. You bring `(X, y)` per task; the network grows, locks,
dreams and rehearses on its own under a byte budget.

```python
from trioron.api import TaskData, TrioronConfig, build_donor

tasks = [
    TaskData(name="cats_vs_dogs",
             X_train=Xtr, y_train=ytr,   # (N, 784) float32, (N,) int64
             X_test=Xte,  y_test=yte,
             classes=[0, 1]),
    TaskData(name="birds_vs_fish",
             X_train=..., y_train=..., X_test=..., y_test=...,
             classes=[2, 3]),
]
donor = build_donor(label="my_donor", tasks=tasks, seed=42,
                    config=TrioronConfig(cap_bytes=32_000),
                    out_path="my_donor.pt")
```

Compose donors with `absorb`, keep teaching with `extend`, deploy with
`deploy_agent` — all from `trioron.api`; see [MANUAL.md](MANUAL.md).
(`absorb(rec, donor, ...)` on 2.0 `Substrate` objects is the head-merged
graft — one substrate whose forward is the exact sum of its siblings';
MANUAL §13.7.)

### 2. The substrate itself (a growing net you train like any torch module)

The 2.0 core: cells with a **triparametric node** (weight, epigenetic lock
λ, axonal gain), a hard parameter envelope, growth/pruning/locking as
lifecycle events. Trained by consequence — a loss you choose, TD, anything
that produces a gradient.

```python
import torch
from trioron.api import construct, seeded, Envelope, default_dispatch_table

sub = construct(base=seeded(784, 10, interior_cells=32, nonlinear=True),
                envelope=Envelope(max_parameter_bytes=200_000),
                dispatch_table=default_dispatch_table(),
                capacity=1024, sparsity_k=0)
sub.compile(); sub.prepare_training()          # prepare_training() is required
opt = torch.optim.Adam(sub.trainable_tensors(), lr=3e-3)
loss = torch.nn.functional.cross_entropy(sub(x), y)
opt.zero_grad(); loss.backward(); sub.zero_dormant_grads(); opt.step()
```

Spec: `paper/v3/spec.md` (§2–§6); canonical short reference:
`docs/TRIORON_MANUAL.md`.

**Deploying it — the substrate is a training-time structure, not an
inference-time cost.** The live forward walks the arena (that is what
lets it grow); for serving, fold it to a fixed module:

```python
from trioron.api import export_dense
module = export_dense(sub)                                  # exact, buffers-only
fast = torch.jit.freeze(torch.jit.trace(module, x[:1]))
```

Measured on the world organism (1 CPU thread, batch 1): arena forward
~485 µs → exported ~50 µs — the same per-call latency as a 27 K-param
DQN MLP, at 1/5 the parameters. The export does not learn; keep the arena
checkpoint for learning and re-export after each wake/extend/dream cycle.

### 3. Phasecyte — the gradient-free learner (a stream in, no labels required)

The second learner on the same body: single-pass, phase-coherent lock-in
over a stream, sufficient statistics only (no stored data). Leaves are
enrolled as domains appear; a manifold router arbitrates; a gradient
substrate can then be **dreamed** from the leaves' own sketches with no
wake gradients (chained-15: dreamed 0.540 vs phasecyte-nest 0.474 vs
monolith 0.403, n = 3).

```python
from trioron.api import PhasecyteNest, dream_distill, dreamed_predict

nest = PhasecyteNest(sense)                # sense: X -> descriptor tensor
nest.enroll(group=0, genesis_pool=X0)      # when domain 0 first appears
nest.observe(0, X_batch, labels)           # single pass, label-free tolerant
router = nest.fit_router()
pred_group = nest.route(X_query)
```

Spec §10; `docs/design/pcll_substrate_integration.md`.

### What is *not* in the package yet: the embodied organism

The survival showcase (<https://marcrockhat.github.io/trioron-project/tour/phasecyte.html>)
— drives → primitive leaves → consequence-taught router → structural
dreaming from its own cause-of-death table — still lives in
`archive/experiments/world/` and requires hand-written skill masters. It is
being reduced to a **"declare your drives"** contract (drive-only
vocabulary reaches 112 ± 23 survival vs 148 ± 13 master-built, n = 3, zero
policy code; see `docs/handoff/HANDOFF.md`). Until that ships, use the
recipe in `docs/learning_methods.md` and the scripts under `archive/`.

## Setup (WSL2)

The section below is for *reproducing the paper*, not library use. If you only need the API, the quick install above is enough.

```bash
# Move into WSL filesystem (NOT /mnt/c — that's slow)
cd ~/trioron-project

# Use a venv
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements.txt
```

Torch CPU wheel is ~750 MB. First install is the slow part.

## Run the unit tests

```bash
python3 -m pytest tests -q          # v2 substrate + phenotype + Phasecyte tests
```

Four known pre-existing failures (test_learning TestCredit ×2, test_lifecycle
×2) are tracked in `docs/handoff/HANDOFF.md`; everything else passes.

## Reproduce the headline results

The n = 10 panels driving the paper's headline table run unattended via:

```bash
bash experiments/run_n10_paper.sh
```

This sequentially runs the manifold-replay panel, the five-family competitor sweep (PackNet / HAT / Online EWC / LwF + EWC / hippocampal K = 50), and the dream-archive panel. Individual panels can also be launched directly:

```bash
# chained-15 manifold-grown panel, n = 10
python3 experiments/bench_manifold_replay_n10.py

# Competitor sweep on chained-15 (n = 10)
python3 experiments/bench_packnet_chained_15_n10.py
python3 experiments/bench_hat_chained_15_n10.py
python3 experiments/bench_online_ewc_chained_15_n10.py
python3 experiments/bench_lwf_chained_15_n10.py

# Dream-archive Phase 1 + Phase 2 (storage win, n = 3 pending rerun)
python3 experiments/bench_archive_n3.py

# Ship-wake-extend loop (chained-15 → +8 EMNIST K..Z)
python3 experiments/bench_chained_extend.py
```

CSVs and `*_run*.log` files land in `outputs/`. Run logs from every reported panel are committed; CSVs are gitignored.

## Layout

```
trioron-project/
├── README.md                    # this file
├── MANUAL.md · QUICKSTART.md    # donor API manual; 5-min reproduction
├── docs/TRIORON_MANUAL.md       # canonical short reference (subordinate to the spec)
├── docs/handoff/HANDOFF.md      # cross-session state of record (rewritten every session)
├── paper/v3/spec.md             # Trioron 2.0 architecture spec — source of truth
├── trioron/                     # the package (pip install trioron)
│   ├── api.py                   # PUBLIC SURFACE — import from here
│   ├── core/                    # cell, epigenome, graph, envelope, arena, construct
│   ├── phenotype/               # how genes express into ops (linear, dendrite, …)
│   ├── bases/                   # construction recipes (seeded, minimal, developmental, …)
│   ├── learning/                # credit, frustration, dream, manifold, router
│   ├── lifecycle/               # growth, evolution, ship, graft, compact
│   ├── pcll/                    # Phasecyte (phase-coherent lock-in) + nest + wake/dream
│   ├── evolution/ · viz/        # multi-substrate controller; recorder / viewer
│   ├── compat/                  # v1 ↔ v2 bridge
│   └── legacy/                  # v1 modules (donor API implementation, benches, competitors)
├── archive/experiments/         # research drivers (world/, progenitor/, …) — not packaged
├── experiments/                 # paper bench scripts (CSV + log outputs)
├── outputs/                     # bench CSVs (gitignored) + run logs (committed)
├── paper/                       # paper.tex / paper.pdf / refs.bib
├── tour/                        # static Canvas demo + phasecyte showcase; GitHub Pages
├── hf_space_build/              # Hugging Face Space deployment build
└── tests/                       # unit tests
```

## Status

- [x] Step 1: TrioronLayer + tests
- [x] Step 2: TrioronNetwork + 2-task continual-learning verification
- [x] Step 3: Scripted incubation environment
- [x] Step 4: Three-condition growth trigger (plateau / rank / grad-stability)
- [x] Step 5: Cellular division routine
- [x] Step 6: Pruning loop
- [x] Step 7: Hard ceilings (cap_bytes pre-flight)
- [x] Step 8: Benchmark vs same-param fixed MLP (falsification gate cleared)
- [x] Phase 4.5: Dreaming phase (replay / compress / purge / archive)
- [x] Manifold replay (storage-free pseudo-rehearsal)
- [x] Dream archive (Phase 1 row-lock + Phase 2 int8 quant)
- [x] BF16 mixed-precision deployment substrate
- [x] Ship-wake-extend loop (chained-15 → chained-23)
- [x] Five-family competitor sweep (PackNet / HAT / Online EWC / LwF / hippo) at n = 10
- [x] Multi-branch absorption + L0 handshake translator (R · S factorization)
- [x] Tour: 13-scene Canvas demo at <https://marcrockhat.github.io/trioron-project/tour/>
- [x] Full integrated paper draft (`paper/paper.tex`, 29 pages)
- [ ] ArXiv submission (pending endorsement)
- [x] PyPI release (`pip install trioron`); 0.3.x adds Phasecyte nest + wake/dream (`trioron.pcll`)
- [ ] Embodied organism as a package API ("declare your drives", no hand-written masters)
- [ ] Deployment script + ready-to-use checkpoint for Orange Pi 5B

## Disclosure

This work was carried out in collaboration with two personified AI assistants
in defined supporting roles: **Gemma** (Gemini Pro 3.1, academic-advisory) and
**Chloe** (Claude Opus 4.7 1M-context, engineering). Human-led problem framing
and final decision-making; AI-supported implementation, analysis, and writing.
The human author holds sole responsibility for all claims, methodological
choices, and interpretations. Per recent editorial guidance
([Nature 2023](https://www.nature.com/articles/d41586-023-00191-1),
[WAME 2023](https://wame.org/page3.php?id=106)), AI systems are not listed
as authors of record.

## License

[MIT](LICENSE). Copyright © 2026 Marcelinus R Hatorangan.
