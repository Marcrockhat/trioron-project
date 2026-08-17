# Trioron — User Manual

This is the long-form user manual for the `trioron` Python package and
its `trioron` CLI. Read it end-to-end if you are building your own
trioron network. For a 5-minute reproduction of the paper's
multi-branch absorption result, jump to [QUICKSTART.md](QUICKSTART.md)
instead. For the cross-modal bridge / encoder reference,
[BRIDGE.md](BRIDGE.md) is the dedicated doc.

---

## Table of contents

1. [What trioron is](#1-what-trioron-is)
2. [Install](#2-install)
3. [The shared-L0 invariant](#3-the-shared-l0-invariant)
4. [Bringing your own data](#4-bringing-your-own-data)
5. [Training a donor](#5-training-a-donor)
6. [Tuning the architecturally-distinctive knobs](#6-tuning-the-architecturally-distinctive-knobs)
7. [Absorbing donors into one organism](#7-absorbing-donors-into-one-organism)
8. [Extending a deployed organism](#8-extending-a-deployed-organism)
9. [Evaluating an organism](#9-evaluating-an-organism)
10. [Deploying as an agent (REPL + HTTP)](#10-deploying-as-an-agent-repl--http)
11. [Tool registration](#11-tool-registration)
12. [The Python API at a glance](#12-the-python-api-at-a-glance)
    - [12.1 Beyond the donor API — substrate, Phasecyte, dense export](#121-beyond-the-donor-api--substrate-phasecyte-dense-export)
13. [Trioron 2.0 — profiles, axes, topography](#13-trioron-20--profiles-axes-topography)
14. [Troubleshooting](#14-troubleshooting)
15. [Reference: CLI commands](#15-reference-cli-commands)

---

## 1. What trioron is

Trioron is a continual-learning architecture. Three things distinguish
it from PackNet / HAT / Online EWC / LwF:

- **Growth under a byte budget.** The hidden width is not fixed at
  init — it grows when the network plateaus, and growth events that
  would exceed `cap_bytes` fail their pre-flight (`trioron/ceilings.py`).
- **Dreaming.** Between tasks the network fires a consolidation cycle:
  manifold-replay → cosine-similarity probe → starve / merge / purge.
  See `trioron/dreaming.py`.
- **Multi-branch absorption.** Donors trained at the same L0 seed can
  be assembled into one organism with no calibration training, no
  fusion layer, no gradient updates. The shared random projection
  makes it lossless.

Together these give the deployment story trioron exists for: a small
on-device network you can train in pieces, ship, then extend in the
field — and use as the inner-voice orchestrator over a frozen LLM with
tool-use plumbing.

---

## 2. Install

CPU-only PyTorch is enough for everything in this manual.

### From a local clone

```bash
git clone https://github.com/marcrockhat/trioron-project.git
cd trioron-project
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Optional extras

| extra | what it pulls in | when you need it |
|---|---|---|
| `[bridge-text]`  | `sentence-transformers` | text encoder for the bridge |
| `[bridge-image]` | `open-clip-torch + Pillow` | image encoder for the bridge |
| `[bridge-audio]` | `openai-whisper` | audio encoder for the bridge |
| `[bridge-all]`   | all three | one-stop install |
| `[serve]`        | `fastapi + uvicorn + pydantic` | `trioron serve --http` |
| `[dev]`          | `pytest + build + twine` | contributing |

```bash
pip install -e '.[bridge-text,serve]'
```

### WSL2 note

Use the Linux filesystem (`~/trioron-project`), not `/mnt/c/...` —
torch CPU wheels are ~750 MB and `/mnt/c` I/O is several times
slower.

---

## 3. The shared-L0 invariant

**Strongly recommended:** every donor in a population should be built
with the same `seed`.

Trioron's L0 is a frozen random projection. Two donors trained with
the same `seed` see the same L0 weights bit-for-bit, which is what
makes paste-and-go absorption lossless: the organism keeps one shared
L0 and stacks the per-donor L1 + head + manifold archive on top of
it. If two donors used different seeds, their L1 spaces are
incomparable — there is no rotation that aligns them — so the
organism's L1 routing has to bridge them somehow.

Default seed = 42. Override with `--seed` on `trioron train` or
`seed=` on `trioron.api.build_donor`.

### Mismatched seeds — random-projection fallback (untested)

Since 2026-05-06 the `absorb` step **does not** hard-error on
mismatched seeds. Instead it picks a canonical L0 (the most common
seed across donors; ties → first appearance) and builds a
deterministic Gaussian random projection adapter
`A_i ∈ R^{canon_dim × donor_i_dim}` for each non-canonical branch.
The seed pair `(canonical_seed, donor_seed)` deterministically seeds
each adapter, so the same donor mix always reconstructs the same
projection.

A loud `[trioron absorb] WARNING` fires once per non-canonical branch
and a summary fires at the absorb call. The path is **untested** —
the donor's L1 was trained on its own L0 outputs, and a random
projection of canonical-z does not reproduce that distribution. The
behavior probe at `experiments/probe_random_projection_fusion.py`
shows that on a 2-donor synthetic case (one canonical seed=42, one
mismatched seed=7), the canonical donor stays at 1.00 task-aware
while the mismatched donor drops to ~0.4 task-aware. The handoff
estimate of "10-30% accuracy hit" is a floor, not a ceiling — real
losses can be larger.

**Use this path only if you genuinely cannot rebuild your donors at
the same seed.** Different-input-dim cases (problem (c) in the
design doc) are punted to the Bridge layer (see `BRIDGE.md`); only
seed and L0-width mismatches are handled by the projection fallback.

---

## 4. Bringing your own data

Trioron consumes data as a list of `TaskData` objects. Each
`TaskData` represents one task in the curriculum — typically a binary
or n-ary classification task — and the donor sees them in order.

```python
import torch
from trioron.api import TaskData

# Tensor shape contract:
#   X_train, X_test:   (N, input_dim) float32 in [0, 1]
#   y_train, y_test:   (N,) int64 in GLOBAL class space
#
# Default architecture expects input_dim = 784 (28x28 grayscale
# flattened). Use a different input_dim by passing input_dim= via
# AdvancedConfig (the L0 width is independent of input_dim — they
# are linked only by the L0 random projection).

tasks = [
    TaskData(
        name="cats_vs_dogs",
        X_train=X_train_cats_dogs,    # (N, 784) float32
        y_train=y_train_cats_dogs,    # values are 0 or 1
        X_test=X_test_cats_dogs,
        y_test=y_test_cats_dogs,
        classes=[0, 1],               # GLOBAL class IDs this task uses
    ),
    TaskData(
        name="birds_vs_fish",
        X_train=X_train_birds_fish,
        y_train=y_train_birds_fish,   # values are 2 or 3
        X_test=X_test_birds_fish,
        y_test=y_test_birds_fish,
        classes=[2, 3],
    ),
]
```

### Global vs local class IDs

Labels are in **global** space — the index in the donor's head, not
a per-task local index. Two donors that cover disjoint global class
ranges (e.g. donor A covers `[0..9]`, donor B covers `[10..19]`) can
be absorbed into one organism with no head collision.

If your data is in local space (every dataset starts at 0), remap
before constructing TaskData: `y_global = y_local + offset`.

### From a CLI loader file

For the CLI (`trioron train --from-py`), wrap the same construction
in a function returning the list:

```python
# my_loader.py
import torch
from trioron.api import TaskData

def make_tasks():
    # ... your data loading ...
    return [TaskData(...), TaskData(...)]
```

Then:

```bash
trioron train --from-py my_loader.py:make_tasks --label my_donor \
              --seed 42 --epochs 8 --out my_donor.pt
```

The `path:fn` syntax also accepts dotted module paths
(`my.package.module:make_tasks`) if your loader lives inside an
installed package.

---

## 5. Training a donor

A donor is one self-contained skill pack: frozen L0 + L1 substrate +
head over its trained classes + per-class manifold archive (μ, σ).

### From the CLI (custom data)

```bash
trioron train --from-py my_loader.py:make_tasks \
              --label digits_donor \
              --seed 42 \
              --epochs 8 \
              --cap-bytes 32000 \
              --out digits_donor.pt
```

### From the CLI (built-in chained-15 splits, for reproducing the paper)

```bash
trioron train --donor digits  --out donor_digits.pt
trioron train --donor fashion --out donor_fashion.pt
```

Available built-in splits: `digits` (MNIST 0..9 → global 0..9),
`fashion` (FashionMNIST 0..9 → global 10..19), `emnist` (letters A..J
→ 20..29), `emnist_kt` (K..T → 30..39), `emnist_uz` (U..Z partial →
40..45).

### From Python

```python
from trioron.api import build_donor, TrioronConfig

donor_path = build_donor(
    label="digits_donor",
    tasks=tasks,                              # list of TaskData
    seed=42,
    epochs_per_task=8,
    config=TrioronConfig(cap_bytes=32_000),
    out_path="digits_donor.pt",
)
```

The donor checkpoint contains: `state_dict`, manifold archive
(`{class_id: (μ, σ)}`), the input dim, the L0 seed, the arm choice,
and the `TrioronConfig` it was trained with (so `extend` can inherit
it).

---

## 6. Tuning the architecturally-distinctive knobs

These are the knobs that actually differ from PackNet / HAT / EWC /
LwF. The ones that materially affect the headline result are exposed
at the top level; growth-trigger sub-knobs are gated behind
`--advanced` because wrong values silently kill growth.

### Cheatsheet

```bash
trioron tune --show
trioron tune --inspect my_donor.pt
```

### Primary knobs

| flag | API field | default | what it does |
|---|---|---|---|
| `--cap-bytes` | `cap_bytes` | `32_000` | hard byte budget for trainable params (4 B/param). Growth events that would exceed this fail their pre-flight. Set to 0 to disable. |
| `--dream-replay-steps` | `dream_replay_steps` | `50` | replay batches per dream cycle. Paper chained-15 default; 50-task scaled to 200. |
| `--dream-buffer-threshold` | `dream_buffer_threshold` | `0` | min past tasks before the first dream fires. 0 = dream after every task. |
| `--manifold-noise-scale` | `manifold_noise_scale` | `1.0` | σ multiplier when sampling from the per-class manifold archive. 0.0 = μ-only (loses ~7% full-softmax). |
| `--routing-temperature` | `routing_temperature` | `1.0` | soft-routing T at the organism. T → 0 = hard routing; T = 1.0 = paper default; T → ∞ = uniform. Consumed by absorb/eval/serve. |
| `--per-class-bias` | `per_class_bias` | `False` | per-class bias offsets at eval (dream-cycle calibration; closes ~80% of the BTM-MoE gap with no real data). |

### Advanced knobs (gated behind `--advanced`)

```bash
trioron train --from-py my_loader.py:make_tasks \
              --advanced \
              --h-init 32 \
              --n-grow-per-task 4 \
              --ewc-intertask 30 \
              --dream-compression starve \
              --out my_donor.pt
```

| flag | API field | default | notes |
|---|---|---|---|
| `--h-init` | `AdvancedConfig.h_init` | `32` | initial L1 hidden width |
| `--n-grow-per-task` | `n_grow_per_task` | `4` | nodes added per growth event |
| `--ewc-intertask` | `ewc_intertask_strength` | `30.0` | EWC strength between tasks |
| `--ewc-dream` | `ewc_dream_strength` | `30.0` | EWC strength inside dream replay |
| `--dream-replay-fraction` | `dream_replay_fraction` | `0.25` | fraction of past tasks per dream |
| `--dream-compression` | `dream_compression_action` | `starve` | `starve` / `merge` / `none` |
| `--dream-max-downscales` | `dream_max_downscales_per_layer` | `1` | per-layer downscale cap (sRNA-style) |
| `--dream-apoptosis` | `dream_apoptosis_on` | `True` | apoptosis spike-decay |
| `--no-freeze-l0` | `freeze_l0` | `True` | train L0 instead of freezing |
| `--l0-width` | `l0_width` | `128` | L0 random-projection width |

If you change `--l0-width` or `--no-freeze-l0`, every donor in the
population must use the same value — the shared-substrate invariant
extends to those choices, not just to the seed.

### Inspecting what a donor was trained with

```bash
trioron tune --inspect my_donor.pt
```

Prints the label, L0 seed, arm, classes covered, architecture, input
dim, and the full `TrioronConfig` (including AdvancedConfig if it
was set). Use this to sanity-check that two donors you're about to
absorb agree on the substrate.

---

## 7. Absorbing donors into one organism

Zero-shot. No training, no calibration pass, no fusion layer.

### From the CLI

```bash
trioron absorb --donors donor_a.pt,donor_b.pt,donor_c.pt \
               --out organism.pt
```

The shared-L0 invariant is checked at load time; mismatched seeds
fail fast with a clear error. The output is a `multibranch_organism`
checkpoint that `eval`, `infer`, `serve`, and `extend` all consume.

### From Python

```python
from trioron.api import absorb

organism_path = absorb(
    donor_paths=["donor_a.pt", "donor_b.pt"],
    out_path="organism.pt",
)
```

### Storage breakdown

The absorb step prints something like:

```
storage: 471 KB total (L0 392 KB shared, branch substrate 59 KB,
                       archive 20 KB)
```

L0 is shared across all branches (it's the same projection, stored
once). Branch substrate (L1 + head) and archive (μ, σ per class) are
per-donor and scale linearly in the number of donors absorbed. The
deployment Pareto curve is determined by how many donors you
absorb, not by their internal width.

---

## 8. Extending a deployed organism

The ship-wake-extend loop (paper §4.6 / extension experiment): take
an existing donor, fire the shipping-consolidation dream, lift the
cap, then learn new tasks on top.

```bash
trioron extend \
    --donor my_donor.pt \
    --base-py my_loader.py:make_base_tasks \
    --new-py  my_loader.py:make_new_tasks \
    --extension-cap-bytes 64000 \
    --epochs 8 \
    --out my_donor_extended.pt
```

### Why `--base-py` is required

Donor checkpoints carry the substrate state (weights + EWC anchors +
Fisher + archive locks + manifold), but they don't carry the base
training data — that would balloon the file. The extension boundary
fires a shipping-consolidation dream that replays real data over the
base tasks, and the final eval covers both base and extension classes,
so the same `TaskData` you used to build the donor must be re-supplied
here.

The base curriculum itself is **not** re-trained — `extend` resumes
from the donor's substrate (skipping the base per-task loop entirely)
when the checkpoint includes resume metadata (`version >= 2`,
introduced 2026-05-06). Older donors fall back to the legacy
integrated path with a printed warning; rebuild them with the current
trioron version to get the speedup.

Resumed accuracy matches integrated within seed-noise but is not
bit-exact: the boundary dream's RNG state isn't serialized so it gets
reseeded on resume.

### What happens during extension

1. **Resume path (default)**: hydrate the donor's net + manifold from
   the checkpoint, skip the base training loop.
   **Legacy path (v1 donors)**: replay the base curriculum on the
   donor's L0 seed and arm.
2. Fire shipping-consolidation dream (full-coverage replay over
   `base_tasks`, archive-aware grad masking, archive-lock).
3. Permanently snap archived rows to int8 (skip with
   `--no-permanent-int8`).
4. Lift the cap from `cap_bytes` → `--extension-cap-bytes`.
5. Train on `--new-py` tasks. The network grows new plastic capacity
   while archived rows stay locked at int8.

Paper baseline (`extension_experiment_result`): chained-15 → +8
EMNIST K..R lands at 23 tasks / 38 classes / **168 KB total
deployment**, original tasks survive at task-aware ≥ 0.93.

### From Python

```python
from trioron.api import extend

extended = extend(
    donor_path="my_donor.pt",
    base_tasks=my_base_tasks,        # list of TaskData
    new_tasks=my_new_tasks,
    out_path="my_donor_extended.pt",
    extension_cap_bytes=64_000,
    permanent_int8=True,
)
```

---

## 9. Evaluating an organism

```bash
trioron eval --organism organism.pt
```

Reports per-task accuracy plus two headline numbers:
- **task-aware** — accuracy when the task ID is known at inference
  (the production metric for trioron's deployment as a
  context-conditioned classifier).
- **full-union** — argmax across all covered classes (the harder
  metric).

Both are reported with and without per-branch log-softmax
normalization. The normalized variant is what the paper's lossless-
absorption claim measures; the raw variant exposes what happens
without calibration.

For built-in chained-15 donors the eval splits are loaded
automatically by branch label. For organisms whose donors were built
with `train --from-py`, point eval at a loader that returns the
held-out test set in the same `TaskData` shape (only `X_test` /
`y_test` are read):

```bash
trioron eval --organism organism.pt \
             --from-py my_loader.py:make_eval_tasks
```

Or from Python (same loader works programmatically):

```python
from trioron.api import evaluate

result = evaluate(
    organism_path="organism.pt",
    eval_tasks=my_test_tasks,
    routing_temperature=1.0,
)
print(result["task_aware_mean"], result["full_union_mean"])
```

---

## 10. Deploying as an agent (REPL + HTTP)

This is the deployment story trioron exists for: an inner-voice
orchestrator that decides which tool to fire based on the input.

### REPL mode

```bash
trioron serve \
    --organism organism.pt \
    --encoder my_agent.py:make_encoder \
    --tools my_agent.py:tools \
    --class-map my_agent.py:class_to_tool
```

Drops you at a `> ` prompt. Type a query, see the chosen tool, the
args dispatched, and the tool result.

### HTTP mode

```bash
pip install -e '.[serve]'   # installs FastAPI + uvicorn + pydantic
trioron serve --http 8000 \
    --organism organism.pt \
    --encoder my_agent.py:make_encoder \
    --tools my_agent.py:tools \
    --class-map my_agent.py:class_to_tool
```

REPL and HTTP run together by default; pass `--no-repl` to disable
the prompt. HTTP exposes:

| endpoint | purpose |
|---|---|
| `GET /health` | branches, union classes, status |
| `GET /tools` | OpenAI/Anthropic-format tool schemas |
| `GET /classes` | union classes + class→tool map |
| `POST /decide` | route only; returns class, tool, gates, top-k |
| `POST /act` | route + dispatch; returns tool result |

`POST /decide` and `POST /act` accept JSON: `{"input": <whatever>,
"topk": 5}`. `act` additionally accepts `"args": {...}` to override
the args resolver.

### What the encoder, tools, class-map, and resolver are

```python
# my_agent.py
import torch
from trioron.legacy.bridge import ToolDispatcher

# 1. Encoder: any object with .encode_dim and __call__(raw) → tensor
class MyEncoder:
    encode_dim = 384
    def __call__(self, raw):
        # raw is whatever you POST to /act ("input" field) or type at REPL
        return self._embed(raw)         # returns (encode_dim,) or (1, encode_dim)

def make_encoder():
    return MyEncoder()

# 2. Tool registry
tools = ToolDispatcher()

@tools.tool
def lookup_user(user_id: str) -> dict:
    """Return profile for the given user."""
    return {"user_id": user_id, "name": "..."}

# 3. Class-id → tool-name map (one entry per union class)
class_to_tool = {0: "lookup_user", 1: "lookup_user", 2: "query_db"}

# 4. (Optional) args resolver
def resolve_args(raw, tool_name, decision):
    if tool_name == "lookup_user":
        return {"user_id": str(decision.union_class)}
    return {}
```

The encoder dim and the organism's L0 width don't have to match —
the bridge auto-builds an `L0Adapter` (random projection seeded by
the organism's L0 seed) when they differ.

### Reference encoders (cross-modal)

The `[bridge-text]` / `[bridge-image]` / `[bridge-audio]` extras give
you ready-made encoders:

```python
from trioron.legacy.bridge.encoders.text import TextEncoder      # 384-dim
from trioron.legacy.bridge.encoders.image import ImageEncoder    # 512-dim
from trioron.legacy.bridge.encoders.audio import AudioEncoder    # 384-dim
```

See [BRIDGE.md](BRIDGE.md) for the full catalogue.

---

## 11. Tool registration

Two equivalent paths.

### Path A — Python decorator (type-hint inference)

```python
from trioron.legacy.bridge import ToolDispatcher

tools = ToolDispatcher()

@tools.tool
def describe_scene(image_path: str, max_words: int = 20) -> str:
    """Generate a brief description of the scene in the image."""
    ...

@tools.tool(name="add", description="Add two numbers.")
def add_numbers(a: int, b: int) -> int:
    return a + b
```

The schema is inferred from type hints; the description from the
first line of the docstring.

### Path B — JSON schema (OpenAI / Anthropic-compatible)

```python
tools.register_json_schema(
    name="describe_scene",
    description="Generate a brief description of the scene in the image.",
    parameters={
        "type": "object",
        "properties": {
            "image_path": {"type": "string"},
            "max_words":  {"type": "integer"},
        },
        "required": ["image_path"],
    },
    fn=lambda image_path, max_words=20: f"...{image_path}...",
)
```

Use Path A for new code; Path B when migrating existing OpenAI /
Anthropic tool definitions or when you want the schema to live in a
config file.

### Dropping the registry into an LLM call

```python
print(tools.to_openai_dicts())   # list of OpenAI tool dicts
print(tools.to_json())           # JSON string for an Anthropic API call
```

---

## 12. The Python API at a glance

Everything users should import lives in `trioron.api` (0.3.1+; earlier
wheels shipped no `trioron.api` module — upgrade). The research scripts
under `archive/experiments/*` remain available for paper reproduction
but are not part of the supported surface.

`trioron.api` is one import path over three learners/flows — the donor
API below (v1 flows, implemented in `trioron.legacy.api`), the 2.0
substrate (`construct`, `seeded`, `Envelope`, `default_dispatch_table`,
`export_dense`), and Phasecyte (`PhasecyteNest`, `dream_distill`,
`dreamed_predict`, `PCLLController`). §12.1 below covers the two that
this manual's donor flow does not.

```python
from trioron.api import (
    TaskData,         # dataclass — one task in a curriculum
    TrioronConfig,    # primary tunable knobs
    AdvancedConfig,   # growth/EWC/L0-width sub-knobs
    build_donor,      # train one donor on TaskData curriculum
    absorb,           # compose donors → organism
    load_organism,    # rebuild organism from .pt
    extend,           # ship-wake-extend loop
    evaluate,         # accuracy summary on held-out test tasks
    deploy_agent,     # wrap organism in a BridgedOrganism
)
from trioron.legacy.bridge import (
    Encoder, L0Adapter,
    Tool, ToolDispatcher,
    BridgedOrganism, Decision,
)

# 2.0 surface — see §13 for usage. All opt-in; not needed for the
# 1.0 build_donor → absorb → deploy flow.
from trioron import (
    TrioronProfile, OPEN, REASONING, CLASSIFICATION, EDGE, PRESETS,
    grid_positions_2d, pixel_positions_2d, pool_id, pool_centroid,
    build_lcn_mask, locality_metric,
)
from trioron.api import (
    set_axonal_gain, archive_input, insert_layer,
    axis6_spawn, inherit_dendrite,
    pool_matched_absorb, merge_manifold,
    ManifoldStore, settle_head_via_manifold, settle_head_with_retry,
)
```

A full mini-app in 30 lines:

```python
from trioron.api import (
    TaskData, TrioronConfig, build_donor, absorb, deploy_agent,
)
from trioron.legacy.bridge import ToolDispatcher
from trioron.legacy.bridge.encoders.text import TextEncoder

tasks_a = [TaskData(name="...", X_train=..., y_train=..., X_test=..., y_test=..., classes=[0, 1])]
tasks_b = [TaskData(name="...", X_train=..., y_train=..., X_test=..., y_test=..., classes=[2, 3])]

build_donor(label="A", tasks=tasks_a, seed=42, out_path="A.pt", config=TrioronConfig(cap_bytes=32_000))
build_donor(label="B", tasks=tasks_b, seed=42, out_path="B.pt", config=TrioronConfig(cap_bytes=32_000))
absorb(donor_paths=["A.pt", "B.pt"], out_path="organism.pt")

tools = ToolDispatcher()

@tools.tool
def hello(name: str) -> str:
    """Say hello."""
    return f"hi {name}"

agent = deploy_agent(
    organism_path="organism.pt",
    encoder=TextEncoder(),
    tools=tools,
    class_to_tool={0: "hello", 1: "hello", 2: "hello", 3: "hello"},
    args_resolver=lambda raw, name, dec: {"name": str(raw)},
)
print(agent.act("world"))
```

---

### 12.1 Beyond the donor API — substrate, Phasecyte, dense export

**Feed contracts differ.** The donor API takes a dataset. The substrate
learns from any gradient (a loss you choose, TD, imitation). Phasecyte
learns from a stream, single pass, tolerant of missing labels. Neither of
the latter is `fit(X, y)`; if you are looking for a dataset-in /
model-out flow, stay with §4–§9.

**The 2.0 substrate directly** — a growing net you train like a torch
module (spec §2–§6; `docs/TRIORON_MANUAL.md` is the short reference):

```python
import torch
from trioron.api import construct, seeded, Envelope, default_dispatch_table

sub = construct(base=seeded(784, 10, interior_cells=32, nonlinear=True),
                envelope=Envelope(max_parameter_bytes=200_000),
                dispatch_table=default_dispatch_table(),
                capacity=1024, sparsity_k=0)
sub.compile()
sub.prepare_training()            # REQUIRED — turns on grads and compiles
opt = torch.optim.Adam(sub.trainable_tensors(), lr=3e-3)
loss = torch.nn.functional.cross_entropy(sub(x), y)
opt.zero_grad(); loss.backward()
sub.zero_dormant_grads()          # commitment 5: dormant cells stay frozen
opt.step()
```

Gotchas: `compile()` alone leaves `requires_grad` off (loss has no
`grad_fn`); checkpoints must save every tensor in
`sub.trainable_tensors()` — `branch_alpha` is trainable and silently
dropping it lobotomises quad cells (s049 bug).

**Phasecyte** — the gradient-free learner (spec §10):

```python
from trioron.api import PhasecyteNest, dream_distill, dreamed_predict

nest = PhasecyteNest(sense)                # sense: X -> descriptor tensor
nest.enroll(group=0, genesis_pool=X0)      # when domain 0 first appears
nest.observe(0, X_batch, labels)           # single pass; no stored data
router = nest.fit_router()                 # manifold recognition router
groups = nest.route(X_query)
```

`dream_distill(leaf, names, ...)` trains a gradient substrate on
pseudo-samples drawn from the leaf's own class sketches — no wake
gradients, no stored data (chained-15: dreamed 0.540 vs phasecyte-nest
0.474 vs monolith 0.403, n = 3).

**Dense export — the serving artifact** (spec §5.4 / §6.4). The live
substrate forward walks the arena (dormant capacity included) — that is
the price of growth, ~10× the arithmetic. For deployment, fold the
compiled plan to a fixed, buffers-only module:

```python
from trioron.api import export_dense, verify_export
import torch

module = export_dense(sub)                       # exact: fp32 rel. err ~1e-7
assert verify_export(sub, module, x) < 1e-5 * float(sub(x).abs().max())
fast = torch.jit.freeze(torch.jit.trace(module, x[:1]))
torch.jit.save(fast, "leaf.pt")                  # dependency-free serving
```

Measured (world router/leaf, 113 live cells, 1 CPU thread, batch 1):
arena forward ~485 µs → export ~78 µs eager / **~50 µs jit** — the same
latency as a 27 K-param DQN MLP at 1/5 the parameters; a nest decision
(router + leaf) ~105 µs. **The export does not learn**: no grads, no
arena, no λ, no growth/dream. Keep the arena checkpoint (`ship`) for
learning; after each wake/extend/dream cycle re-export (milliseconds)
and swap. There is no export → arena path. Supported buckets:
LINEAR / DENDRITE / TANH; receptor (PCLL) cells and
ATTENTION / CONV / RECURRENT raise `NotImplementedError` — serve those
with the live forward.

## 13. Trioron 2.0 — profiles, axes, topography

The 2.0 substrate adds extra architectural axes on top of the 1.0
donor. They are all opt-in: `TrioronConfig` stays 1.0-shape and the
default donor build does not enable any 2.0 axis.

| axis | what it adds | API entry point |
|---|---|---|
| 2 — plastic fanout sparsity | Lock an input column (anchor + Fisher zero + grad mask) | `archive_input` |
| 3 — depth growth | Insert a new layer between two existing layers, identity-init | `insert_layer` |
| 4 — slow axonal gain | Per-cell multiplicative output gain, written from reward / attention | `set_axonal_gain` |
| 5 — dendritic compartmentalization | K parallel branches per cell with optional supralinear σ_branch | `inherit_dendrite` + `TrioronProfile` |
| 6 — field-conditional spawn | New cells spawn at perturbed positions near a stressed parent | `axis6_spawn` |
| topography (LCN) | Locally-connected L0 mask: each L0 cell sees a Gaussian image patch | `build_lcn_mask` + `grid_positions_2d` + `pixel_positions_2d` |

**Empirical caveat (2026-05-23):** at n=1 on the standard chain-15
curriculum (default bench config), turning on depth + LCN + positional
plasticity together **regresses** full accuracy by ~8 points vs the
1.0 baseline, at +17% wall-clock. The 2026-05-21 n=3 result for axes
3+4+5 showed the same direction at the same magnitude. The
closed-form Phase 6 falsification gate (concentric rings,
fine-discrimination) passed at +0.49 absolute lift for K=2 dendrites,
so the substrate primitives work — they just don't lift the chain-15
default config. The current best practice is: leave 2.0 axes off for
chain-15-style continual-learning benches; the axes are exploratory
research surface, not the default deployment configuration.

### 13.1 Profiles

`trioron.profile` ships four named profiles. Each is a frozen bundle
of substrate-construction defaults. Activate one before
`build_donor` and every `TrioronLayer` constructed inside that scope
inherits its defaults.

| profile | branch_activation | B_max | grow_node | grow_branch | insert_layer | hardware caps |
|---|---|---|---|---|---|---|
| `OPEN` (default) | quad | 8 | on | on | on | none |
| `REASONING` | quad | 8 | on | on | on | none |
| `CLASSIFICATION` | identity | 1 | on | off | on | none |
| `EDGE` | identity | 1 | on | off | off | 256 MB / 30 s |

`OPEN` matches pre-2.0 construction byte-for-byte. `REASONING` is the
depth-heavy regime; `CLASSIFICATION` strips dendrites (Axis 5
functionally off) so capacity scales through cell count, matching the
"freeze trioron as a feature bank" pattern. `EDGE` is sized for
Orange-Pi-class deployment.

```python
import trioron.profile as tp
from trioron.api import build_donor, TrioronConfig

# Scoped activation — restores the previous profile on exit.
with tp.TrioronProfile.use(tp.REASONING):
    build_donor(
        label="reasoner", tasks=tasks, seed=42, out_path="reasoner.pt",
        config=TrioronConfig(cap_bytes=64_000),
    )

# Or process-wide:
tp.TrioronProfile.set_active(tp.EDGE)
```

Explicit kwargs at layer construction always override the profile.

### 13.2 Depth growth (Axis 3)

```python
from trioron.api import insert_layer

new_idx = insert_layer(
    net,
    between=(1, 2),         # between layer 1 and layer 2 (the head)
    n_nodes=64,
    activation="linear",
    init_mode="identity",   # Net2Net-style: forward unchanged at insertion
    K_insert=3,             # cap on inserts landing in the same slot
)
# Caller MUST rebuild any optimizer — new Parameter objects appear,
# and the downstream layer's W is replaced.
```

Identity-init keeps the forward pass mathematically unchanged at the
moment of insertion, so the network's existing accuracy is preserved.
Gradients flow through the inserted layer immediately, so the new
depth is plastic from the first post-insert step.

### 13.3 Topography — LCN masks

LCN replaces L0's dense random projection with a locally-connected
one: each L0 cell sees only a Gaussian-weighted local image patch.
For data with spatial structure (CIFAR-100, natural images), this
adds inductive bias without losing the random-projection freeze. On
chain-15 it is null (28×28 glyphs leave nothing for locality to
exploit).

```python
from trioron import build_lcn_mask, grid_positions_2d, pixel_positions_2d

# Lay 128 L0 cells on a 16×8 retinotopic grid on the unit square.
cell_pos  = grid_positions_2d(grid_x=16, grid_y=8)
pixel_pos = pixel_positions_2d(h=28, w=28)
# Gaussian mask, σ=0.10 — each L0 cell sees a soft local patch.
mask = build_lcn_mask(cell_pos, pixel_pos, sigma=0.10)
net.layers[0].W.data *= mask
```

Per-layer LCN can also be declared on `TrioronProfile.lcn` (dict
keyed by layer index); `TrioronLayer.enable_lcn` installs the mask
post-construction.

### 13.4 Positional plasticity (Axis 6)

```python
from trioron.api import axis6_spawn, inherit_dendrite

# Caller picks the stressed parent.
parent_idx = net.layers[1].field_conditional_growth_candidate()
child_idx = axis6_spawn(
    net, layer_idx=1, candidate_idx=parent_idx, position_jitter=0.3,
)
# Optionally: child inherits parent's dendritic structure with an
# ε perturbation, so it starts specialized but distinguishable.
inherit_dendrite(net, layer_idx=1,
                 parent_idx=parent_idx, child_idx=child_idx,
                 perturb_frac=0.05)
```

Use this when growth should respect the cell field's geometry — new
cells appear near stressed regions rather than at random positions in
parameter space.

### 13.5 Slow axonal gain (Axis 4)

```python
from trioron.api import set_axonal_gain

# Per-cell multiplicative output gain on layer 1.
set_axonal_gain(net, layer_idx=1, signal=reward_vec,
                mode="multiplicative")
```

Modes: `"absolute"` (replace), `"additive"`, `"multiplicative"`.
Gains are clamped to ≥ 0. Written from arbitrary signals — reward,
attention, manual priors. The gain is part of the substrate state, so
it persists across `state_dict` save/load.

### 13.6 Plastic fanout sparsity (Axis 2)

```python
from trioron.api import archive_input

archive_input(net, layer_idx=1, col_idx=17)
# Caller must invoke layer.mask_archived_input_grads() after .backward()
# and before optimizer.step() to enforce the grad mask.
```

Idempotent. Useful for protecting inputs that consolidation has
identified as load-bearing.

### 13.7 Pool-matched cell-level absorption

The 1.0 `absorb` step composes donors at *branch* granularity — the
organism gets one branch per donor. `pool_matched_absorb` operates at
the *cell* level inside one organism's substrate: the recipient
gains one new cell per donor cell, with both networks respecting the
same spatial pool partition.

```python
from trioron.api import (
    pool_matched_absorb,
    merge_manifold,
    settle_head_via_manifold,
)

# Both networks must share the L0 width and the head width.
new_pairs = pool_matched_absorb(
    recipient,
    donor,
    layer_idx=1,
    grid_size=4,                       # 4×4 = 16 pools
    pool_capacity=8,
    donor_trained_classes=[0, 1, 2, 3],
    recipient_trained_classes=[4, 5, 6, 7],
)
merge_manifold(recipient.manifold, donor.manifold)
settle_head_via_manifold(
    recipient, recipient.manifold, seen_classes=range(8),
)
# Caller MUST rebuild the optimizer — the W Parameter objects are
# replaced by the underlying grow_layer call.
```

Use this when you want one substrate (not a routed multi-branch
ensemble) carrying the union of two donors' cells.

### 13.8 Continual-learning benchmark — split-CIFAR-100

`experiments/bench_2_0_cifar_continual.py` is the canonical
v2.0-vs-v1.0 head-to-head on split-CIFAR-100 (10×10 continual
classification, raw 32×32 RGB with augmentation). Four arms isolate
contributions:

| Arm | Substrate | Differentiates |
|---|---|---|
| a | v1.0, dense random-Gaussian L0 | floor |
| b | v1.0 + LCN L0 (frozen retinotopic mask) | LCN's contribution |
| c | v2.0 static oracle (LCN + L1 baked at +24 cells) | architecture's contribution |
| d | v2.0 self-arrange (LCN + Axis 6 spawn) | the growth process |

All arms share the same CL machinery (credit-based row freezing +
cosine head + manifold replay + **post-phase
`settle_head_with_retry`**). The post-phase head re-settle is
load-bearing: without it, full-softmax collapses to the
"remembers-only-last-task" floor (~5%) regardless of arm; with it,
v2.0 lifts to ~12% at n=5 and the v2.0-vs-v1.0 gap becomes
statistically significant.

```
CIFAR_N_SEEDS=5 CIFAR_EPOCHS=8 CIFAR_ARMS=a,b,c,d \
  python3 -u experiments/bench_2_0_cifar_continual.py
```

n=5 headline result (2026-05-23): v2.0 self-arrange beats v1.0 floor
by Δ=+0.0187 full at 7.6σ paired-significance, Δ=+0.0283 task-aware
at 8.6σ. Per-step contributions (LCN, baked axes, self-arrangement)
are each 1-2σ; the headline emerges by stacking. See
`outputs/bench_2_0_cifar_continual_n5_v3_8ep.{csv,log}` for the raw
numbers.

Experimental arms `e/f/g/h` extend the bench with LCN-on-L1,
learnable L0/L1 positions, and pool/channel weight-tying for
"conv-by-emergence" probes — all null at the bench's current scale
(H_INIT=64-128). Kept env-gated for future capacity-pushed retries.

---

## 14. Troubleshooting

### "donors have mismatched L0 seeds {42, 7}"

Two donors you're trying to `absorb` were built with different
`--seed` values. The shared-L0 invariant is non-negotiable — rebuild
one of them with the same seed.

### "cap_exceeded(projected=15204B > cap=8000B)"

Growth was attempted but the new architecture would exceed
`--cap-bytes`. Either raise the cap or accept that the donor will
stop growing at the current width. This is expected behavior; the
dream cycle's purge step will try to free space.

### "trioron serve --http requires the [serve] extra"

Install the optional dependency: `pip install -e '.[serve]'`.

### REPL prints "tool=None"

Your input was routed to a class index that has no entry in
`class_to_tool`. Either add the missing class or the routing is
working as designed (the input doesn't match any donor's domain).

### Donor trains but task-aware accuracy is low

For tiny smoke runs (`--epochs 1`, ~200 samples/task) accuracy will
be near chance — this is expected. Real training takes paper-default
`--epochs 8` per task. Check `trioron tune --inspect donor.pt` to
confirm `cap_bytes` isn't artificially small for your data.

### Loader errors with "labels {…} appear in y but not in classes"

Your `TaskData.classes` doesn't cover every label in
`y_train`/`y_test`. Either add the missing class IDs or filter your
data.

### Can I extend an absorbed organism?

No. Extension operates on a single growing substrate; absorbed
organisms have multiple frozen branches. Extend the individual
donors before absorbing.

### Can I mix donors with different `--l0-width`?

No. The shared-substrate invariant covers L0 width, L0 freeze
status, and the L0 random projection seed. Mixing them requires a
fusion layer that trioron deliberately does not have.

### EWC has no effect / model keeps forgetting despite high `ewc_strength`

Only relevant if you bypass `build_donor` / `extend` and drive a raw
`TrioronNetwork` from your own training loop (joint mode, plain
SGD/Adam, no task boundaries). The trioron node has three coupled
state variables `(w, λ, u)`; `λ` is populated only by the
consolidation cycle:

```python
layer.update_fisher()        # after .backward(), before optimizer.step()
layer.update_lambda()        # at task end
layer.anchor_weights()       # at task end
```

If your loop skips this cycle, `λ` stays at zero and `ewc_penalty()`
is mathematically zero regardless of `ewc_strength`. The network
keeps training and looks healthy but the substrate has silently
degraded to a regular MLP. As of 0.2.2, `ewc_penalty()` emits a
one-shot `RuntimeWarning` the first time it sees an all-zero `λ`.
Sanity-check at the end of training: `layer.lam.max() > 0`.

If your training format doesn't have task boundaries (joint mode,
classification fine-tuning), call once after training converges,
before any downstream EWC-protected fine-tuning:

```python
net.populate_lambda(
    batches=loader,                       # iterable of (x, y)
    loss_fn=torch.nn.functional.cross_entropy,
    n_batches=200,
    rescale_mean=True,                    # default — see below
)
```

This wraps `estimate_fisher → update_lambda_all → anchor_all` and
clears stale gradients. `rescale_mean=True` (default) normalizes
each layer's λ to mean 1.0 so `ewc_strength` becomes an
optimizer-independent stiffness knob — raw Fisher under Adam at
convergence is tiny (gradients vanish at the optimum), so without
rescaling callers need `ewc_strength` in the 1e5–1e7 range. Pass
`rescale_mean=False` to preserve raw Fisher magnitudes.

### Does λ have to come from Fisher?

No. In the trioron `(w, λ, u)` formulation, `λ` is the per-cell
plasticity gate — a stand-in for biological gating mechanisms like
BDNF methylation and perineuronal-net maturation, both of which are
environmentally regulated. Fisher information is one signal you can
write into λ (the canonical EWC channel), but any upstream source
is legitimate. The substrate doesn't care where the values came
from — only that they gate how rigid each cell is.

Use `set_lambda_all` to write λ from arbitrary signals:

```python
# Per-layer signals, one tensor per layer, shape (n_nodes,).
signals = [
    sensor_to_layer_signal(temp, light, motion, n=net.layers[0].n_nodes),
    reward_to_layer_signal(recent_reward,        n=net.layers[1].n_nodes),
    ...
]

net.set_lambda_all(signals, mode="absolute")
# modes:
#   "absolute"       λ ← signal               (replace)
#   "additive"       λ ← λ + signal           (layer on top of Fisher)
#   "multiplicative" λ ← λ * signal           (scale, e.g. a global
#                                              sleep cycle in [0, 1])
```

Concrete use cases:

* **Environment sensors** (the namesake "environment sense") — on an
  edge deployment with temperature/light/IMU/microphone, derive a
  per-cell salience signal from sensor state and write it as λ.
  Cells whose past activations correlated with the current
  environment stiffen; the rest stay plastic.
* **Reward signals** — high-reward episodes raise λ on the cells
  that produced them, protecting useful policies from being
  overwritten by later low-reward exploration.
* **Attention / saliency masks** — task-salient cells stiffen for the
  duration of a task, then relax.
* **Hand-injected priors** — large λ on cells you want frozen for
  reasons outside the loss (regulatory, interpretability, safety),
  or λ = 0 on cells you want to wake up for forced relearning.
* **Sleep / arousal modulation** — `mode="multiplicative"` with a
  scalar in [0, 1] applied uniformly globally damps the entire
  network's λ during a "high-plasticity" window (e.g., a dream
  rehearsal phase) and restores it after.

Fisher and external signals compose: run `populate_lambda` for the
cognitive-importance baseline, then `set_lambda_all(sensors,
mode="additive")` to layer the environmental signal on top.

---

## 15. Reference: CLI commands

```
trioron train     train one donor (built-in split or --from-py)
trioron tune      show distinctive knobs / inspect a saved donor
trioron absorb    compose donors → organism
trioron extend    ship-wake-extend loop on an existing donor
trioron eval      evaluate organism on union test set (built-in
                  splits or --from-py)
trioron infer     single-image inference through an organism
trioron serve     deploy as an agent (REPL + optional HTTP)
```

Run `trioron <command> --help` for the full flag list per command.
