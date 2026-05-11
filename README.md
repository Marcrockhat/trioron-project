# Trioron — an epigenetic-inspired self-organizing network

A continual-learning architecture built around the **trioron**: a node with three coupled state variables (weight, plasticity coefficient, utility) that grows, prunes, and consolidates under a per-curriculum byte budget. Designed for resource-conscience deployment on agentic-AI / IoT / embedded hardware.

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
- Ship-wake-extend loop validated end-to-end at 23 tasks / 38 classes, **168 KB total**, original tasks preserved at task-aware ≥ 0.93.
- Multi-branch absorption: zero-shot composition of independently-trained donors via a 4-byte L0 handshake (R · S factorization), lossless on task-aware out to N = 5 donors.

Method and result details: `paper/paper.pdf` (built from `paper/paper.tex`).

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

# Ship-wake-extend loop (chained-15 → +8 EMNIST K..R)
python3 experiments/bench_chained_extend.py
```

CSVs and `*_run*.log` files land in `outputs/`. Run logs from every reported panel are committed; CSVs are gitignored.

## Layout

```
trioron-project/
├── README.md                    # this file
├── trioron_blueprint.md         # full design doc
├── trioron/                     # core modules
│   ├── api.py                   # public build_donor / train / extend API
│   ├── cli.py                   # command-line entry point
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
│   ├── multibranch.py           # multi-branch organism (zero-shot absorption)
│   ├── composition/             # L0 handshake translator (R · S factorization)
│   ├── senses/                  # sensory-organism / CIFAR-side experiments
│   ├── bridge/                  # cross-modal encoders (see BRIDGE.md)
│   ├── packnet.py               # PackNet competitor
│   └── hat.py                   # HAT competitor
├── experiments/                 # bench scripts (CSV + log outputs)
├── outputs/                     # bench CSVs (gitignored) + run logs (committed)
├── paper/
│   ├── paper.tex                # integrated paper source (compiles with pdflatex)
│   ├── paper.pdf                # built artifact
│   └── refs.bib                 # bibliography
├── tour/                        # static Canvas demo (13 scenes); GitHub Pages source
├── hf_space_build/              # Hugging Face Space deployment build
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
- [x] Five-family competitor sweep (PackNet / HAT / Online EWC / LwF / hippo) at n = 10
- [x] Multi-branch absorption + L0 handshake translator (R · S factorization)
- [x] Tour: 13-scene Canvas demo at <https://marcrockhat.github.io/trioron-project/tour/>
- [x] Full integrated paper draft (`paper/paper.tex`, 29 pages)
- [ ] ArXiv submission (pending endorsement)
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
