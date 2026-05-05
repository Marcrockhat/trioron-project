# Trioron — an epigenetic-inspired self-organizing network

A continual-learning architecture built around the **trioron**: a node with three coupled state variables (weight, plasticity coefficient, utility) that grows, prunes, and consolidates under a per-curriculum byte budget. Designed for resource-conscience deployment on agentic-AI / IoT / embedded hardware.

The full design is in `trioron_blueprint.md`. The paper draft is in `paper/`.

## Headline

On a 30-class class-incremental curriculum (chained-15: MNIST → Fashion-MNIST → EMNIST-letters), with growth + dreaming + manifold replay enabled:

- **0.604 full-softmax / 0.960 task-aware** accuracy at 30 KB of replay storage (n = 3, paired).
- σ-confident wins over PackNet, HAT, Online EWC, LwF + EWC, and exemplar-rehearsal hippocampal buffers on full-softmax and domain-aware (+6σ to +47σ).
- BF16 + int8 dream-archive: **157 KB total deployment** (network + manifold buffer), Δ ≤ 0.0008 lossless.
- Ship-wake-extend loop validated end-to-end at 23 tasks / 38 classes, **168 KB total**, original tasks preserved at task-aware ≥ 0.93.

Method details: `paper/methods.tex` (or `paper/methods.pdf`).

## Setup (WSL2)

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
python3 test_node.py            # TrioronLayer
python3 test_network.py         # TrioronNetwork
python3 test_classification.py  # multi-class head
python3 test_dreaming.py        # dream block (replay/compress/purge/archive)
python3 test_frustration.py     # plateau-counter multiplier
python3 test_pruner.py
python3 test_triggers.py        # plateau / rank-saturation / grad-stability
python3 test_incubator.py
python3 test_ceilings.py        # cap_bytes pre-flight
python3 test_packnet.py
python3 test_hat.py
```

Expected: all PASS, 0 FAIL on each file.

## Reproduce the headline results

```bash
# chained-15 manifold-grown panel, n = 3
python3 experiments/bench_manifold_replay_n3.py

# 5-family competitor sweep on chained-15
python3 experiments/bench_packnet_chained_15_n3.py
python3 experiments/bench_hat_chained_15_n3.py
python3 experiments/bench_online_ewc_chained_15_n3.py
python3 experiments/bench_lwf_chained_15_n3.py

# Dream-archive Phase 1 + Phase 2 (storage win)
python3 experiments/bench_archive_n3.py

# Ship-wake-extend loop (chained-15 → +8 EMNIST K..R)
python3 experiments/bench_chained_extend.py
```

CSVs and `*_run*.log` files land in `outputs/`.

## Layout

```
trioron-project/
├── README.md                    # this file
├── trioron_blueprint.md         # full design doc
├── trioron/                     # core modules
│   ├── node.py                  # TrioronLayer (per-node λ, u, r)
│   ├── network.py               # TrioronNetwork (multi-layer, EWC)
│   ├── classification.py        # CE head + grow_class
│   ├── triggers.py              # plateau / rank / grad-stability
│   ├── pruner.py                # cellular pruning (cosine-nearest redistribute)
│   ├── incubator.py             # growth probe
│   ├── ceilings.py              # cap_bytes pre-flight
│   ├── dreaming.py              # replay / compress / purge / archive
│   ├── frustration.py           # plateau multiplier
│   ├── curriculum.py            # chained-15 / chained-23 builders
│   ├── packnet.py               # PackNet competitor
│   └── hat.py                   # HAT competitor
├── experiments/                 # bench scripts (CSV + log outputs)
├── outputs/                     # bench CSVs (gitignored) + run logs (committed)
├── paper/
│   ├── methods_draft.md         # §3 Methods, markdown
│   ├── methods.tex              # §3 Methods, LaTeX (compiles with pdflatex)
│   ├── methods.pdf              # built artifact
│   └── intro_review.md          # intro draft for §1
└── test_*.py                    # unit tests
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
- [x] Five-family competitor sweep (PackNet / HAT / Online EWC / LwF / hippo)
- [~] §3 Methods + §4 Results paper draft (in `paper/`); §1 Intro and §5 Conclusion pending
- [ ] Deployment script + ready-to-use checkpoint for Orange Pi 5B

## Citation / co-authorship

This work is co-authored with two AI assistants in defined roles: Gemma (engineering, Gemini Pro) and Chloe (research direction, Claude Opus 4.7).

## License

(Pending — public repo will accompany paper release.)
