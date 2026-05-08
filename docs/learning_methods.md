# Trioron learning methods — manual

This document describes the regimes under which a trioron substrate can be
trained, what each regime requires from the data, and the anti-patterns
that have repeatedly burned past sessions. Read this before starting any
new experiment that doesn't fit the chained-15-task / split-MNIST mold.

> **TL;DR.** Trioron is a class-conditional substrate. Its growth, archive,
> and routing all assume that *samples of the same class cluster together
> in L0 code space*. When a task respects that assumption, drive it with
> Mode A (supervised classification). When the target task does not — but
> can be **decomposed into primitive concepts** that each respect it —
> use Mode E (primitive curriculum) to train a vocabulary of small
> donors, absorb them into one organism, then `extend` onto the target.
> Mode B (unsupervised watch) is reserved as a fallback when primitive
> decomposition is genuinely impossible.

---

## 1. Substrate

A trioron organism is a feedforward stack of `TrioronLayer`s plus a
side-channel **manifold archive** that records per-class statistics over
the first layer's outputs.

```
input ──► L0 ──► L1 ──► … ──► head ──► logits
            │                            ▲
            │                            │
            └──► archive {c: (μ_c, σ_c)} ┘
```

| Component | Role | Plastic? |
|-----------|------|----------|
| **L0** (layer 0) | Frozen-at-init random projection. Reduces input → 128-d code (`l0_width`). Same seed across donors → shared L0 substrate (paper §3.10 invariant). | No, when `freeze_l0=True` (default) |
| **L1** (layer 1) | Anchored hidden layer. Grows via `grow_layer(1)` on the 3-condition trigger. New nodes start fully plastic (λ=0, u=0). | Yes; grows |
| **Head** (final) | Linear layer over the global class space. Extended in lockstep with L1 growth (input column added when L1 gains a node). New class slots opened by `extend_output_head` when new classes appear. | Yes |
| **Archive** | Per-class diagonal Gaussian `(μ_c, σ_c)` accumulated over L0 outputs `z = L0(x)` during training. Used for routing across branches and per-class novelty scoring. | Yes; accumulates |
| **Anchors** (`W_anchor`, `b_anchor`, `routing_scale_anchor`) | Snapshot of layer state at task boundaries. EWC penalty pulls live weights back toward the anchor weighted by Fisher. | No (snapshot) |

### What "growth" actually requires

`triggers.py` enforces three conditions, ALL of which must hold inside a
window before `grow_layer` may fire:

1. **Loss plateau** — task loss has not improved by more than `ε_loss`
   over `W` steps.
2. **Effective-rank saturation** — the entropy-based effective rank of L1
   activations is within `ε_rank` of the full hidden width.
3. **Gradient-norm stability** — `‖∇‖₂` is inside `[g_min, g_max]` over
   the same window (rules out optimizer pathology being mistaken for
   capacity saturation).

Growth therefore needs **a real loss to plateau on**. There is no purely
novelty-driven growth path in the current substrate. Any "unsupervised"
mode must still produce a loss — see Mode B for how.

### What the archive actually captures

`(μ_c, σ_c)` is a *generative* model of "what L0 codes look like for
class c." Updates happen during training; the archive is read at
inference for branch routing and at absorb time for the shared-L0
invariant check. **The archive only does useful work if class-c samples
actually cluster in L0 space.** This is the load-bearing assumption.

---

## 2. Mode A — supervised classification

The default. Used for chained-15-task, split-MNIST, the entity-archive
demo, the drawing live-learn demo. Drive via `trioron.api.build_donor` /
`extend` / `absorb`.

### What it expects

- Each sample is `(x, y)` with `x ∈ ℝ^d_input`, `y ∈ ℕ` a global class ID.
- Samples of the same `y` **cluster in L0 space** — i.e., `L0(x)` for
  `x ∈ class_c` lies in a roughly Gaussian region distinct from other
  classes' regions.
- Tasks may introduce disjoint new class IDs (continual learning) or
  revisit old ones; the API supports both.

### What gets trained

- **L1**: SGD on CE loss + EWC penalty against anchored past weights.
  Grows when the 3-condition trigger fires.
- **Head**: SGD on the same loss. Extended whenever a new class slot is
  needed (`extend_output_head`).
- **Archive**: each class's `(μ_c, σ_c)` is updated from L0 outputs
  during the training pass.

### When to use

- Per-class clustering in L0 actually holds (visual digits, pooled
  language embeddings, drawing-strokes).
- The label space is finite and externally meaningful (the y you supply
  has interpretive content the user cares about).

### When to NOT use

- The y is conditional on context rather than a property of x. **Atari
  state → action label is the canonical anti-pattern**: "press UP" is
  the right action in many unrelated game states, so UP-samples don't
  cluster in L0; the archive's Gaussian for `y=UP` becomes a wide bag
  smeared across the whole manifold and frustration/growth fires on the
  wrong signal. See §6.
- The label is a noisy label of a continuous quantity (regression
  recasted as classification with too many bins).
- Multiple distinct sub-populations share a y but no shared L0
  geometry — the archive will average them into a meaningless centroid.

### Knobs that matter

| Knob | Default | What it controls |
|------|---------|------------------|
| `cap_bytes` | 32_000 | Hard upper bound on trainable parameter bytes. Growth events that would exceed this fail pre-flight. |
| `dream_replay_steps` | 50 | Replay batches per post-task dream cycle. |
| `manifold_noise_scale` | 1.0 | Multiplier on per-class σ when sampling from the archive during dream replay. 0 = μ-only. |
| `advanced.h_init` | 32 | Initial L1 hidden width. |
| `advanced.n_grow_per_task` | 4 | Nodes added per growth event. |
| `advanced.freeze_l0` | True | Whether L0 is frozen at init. Frozen L0 is required for the shared-L0 invariant. |
| `advanced.l0_width` | 128 | L0 dim. The compression target. |

---

## 3. Mode E — primitive curriculum

The recommended path when the target task is too complex for Mode A
directly but **decomposes into primitive concepts** that each respect
the per-class clustering assumption. The trioron substrate learns the
primitives one at a time as small Mode-A donors, the donors are
composed via `absorb` into a *vocabulary organism*, and the target
task is learned as an `extend` on top of that vocabulary.

The pattern is the structural analogue of how the book-memory demo
works: the heavy substrate (sentence-transformer encoder) was
pre-trained elsewhere on language primitives, and trioron only had to
learn the small task of "given a question, route to an entity." Mode E
is the same shape, but we own and train the primitives ourselves.

### Why primitives instead of the target task directly

For a complex target like Pong, the natural-seeming labels (actions,
or skills like "press UP when ball above paddle") **do not respect the
per-class clustering assumption**. UP-states are scattered across the
L0 manifold because UP is the right action in many unrelated game
states. Pushing those labels through Mode A causes the failure in §6.1.

Primitives sidestep this. A primitive like "UP" is *defined* on a
synthetic environment where its class-positive frames really do
cluster — frames showing upward motion of any object share a
directional-flow signature in pixels, regardless of what else is on
screen. The donor for UP can be small, fast to train, and clustering-
clean by construction.

### Design constraints on primitives

A primitive donor is well-formed iff:

1. **Class-positive frames cluster in L0.** Verify with a probe before
   training the full donor: extract L0 codes for ~100 positive samples
   and ~100 negative samples; check that mean intra-class distance is
   meaningfully smaller than mean inter-class distance.
2. **The primitive is reusable.** Class IDs should be assigned in a
   global namespace shared across all primitives in the vocabulary, so
   `absorb` works without collision. E.g.:
   ```
   100  UP-positive          110  DOWN-positive
   101  UP-negative           ...
   ```
3. **The primitive is composable, not contextual.** "UP" is a property
   of the frame's motion content; it does not require a Pong-specific
   context to evaluate. If a primitive only makes sense given another
   primitive's value, factor it differently or it will mode-collapse.
4. **Synthetic data is honest.** The synthetic environment that
   generates training data must produce frames in the same input shape
   as the target task's preprocessor (e.g., 84×84×4 flattened to
   28224-d for ALE). Otherwise the substrate cannot transfer.

### Pipeline (two equivalent shapes)

**Shape A — single donor, multi-task curriculum (recommended for one
target task):**
```
[Primitive 1, ..., Primitive N, target_task]
    ──► build_donor ──► full_donor.pt
```
The donor's built-in continual-learning machinery (EWC + dream replay)
preserves earlier primitives as later tasks are added. One artifact,
no `absorb` needed.

**Shape B — primitive donor then extend (recommended when target
arrives later than primitives, or for ship-wake-extend deployment):**
```
[Primitive 1, ..., Primitive N] ──► build_donor ──► primitive_donor.pt
primitive_donor.pt + target_task ──► extend ──► full_donor.pt
```
The primitive donor can be shipped, quantized, deployed, then extended
in place when target data arrives. Same end result as Shape A.

**Shape C — independent skill packs (use only when composing
SEPARATE training runs):**
```
build_donor for skill-pack-1 ──► donor_1.pt
build_donor for skill-pack-2 ──► donor_2.pt
[donor_1.pt, donor_2.pt] ──► absorb ──► organism.pt
```
Use `absorb` for composing donors that were trained INDEPENDENTLY.
**Absorbed organisms are NOT extendable** (`api.extend` rejects them).
If you need both composition AND further extension, extend each donor
first, then absorb.

### Required invariants

- **Shared L0 seed** across all donors that will be absorbed (paper
  §3.10).
- **Disjoint class IDs** across donors that will be absorbed (avoid
  head collisions).
- **Single-donor extensibility** — only single-donor checkpoints can
  go through `extend`.

### Vocabulary design (worked example: Pong)

| Class IDs | Primitive | Class-positive | Class-negative | Synthetic data |
|-----------|-----------|----------------|----------------|----------------|
| 100, 101 | **UP** | Frames showing an object moving upward | Static or downward-moving | Animate any sprite vertically |
| 110, 111 | **DOWN** | Mirror of UP | Mirror | Same generator, mirrored |
| 120, 121 | **MOVE** | Frame-pair where pixels changed | Static-pair | Two frames; vary across all motions |
| 130, 131 | **BALL** | Small bright object somewhere | Empty or paddle-only | Render ball at varied positions |
| 140, 141 | **PADDLE** | Vertical bar somewhere | No paddle | Render paddle at varied positions |
| 150, 151 | **APPROACHING** | Object trajectory shrinking distance to a reference | Receding | Two consecutive frames |

Six primitives × ~5K synthetic samples each = small, fast donors. After
`absorb`, the vocabulary organism's archive has 12 Gaussian clusters
covering the Pong primitive vocabulary.

### Extending onto the target task

Once the vocabulary organism exists, the target task (e.g. Pong) is
trained via `api.extend`:

- `base_tasks` = a small replay set covering the primitive vocabulary
  (so the consolidation dream rehearses primitive recognition).
- `new_tasks` = the Pong-specific task — but the labels here can now
  be primitive-composition tokens like "UP-when-ball-above" (still
  respects clustering because the *combination* of primitive activations
  uniquely picks out the state) instead of raw action classes.

The substrate's L0 + primitive archive stay frozen; growth fires for
new compound classes only.

### When to use

- Target task is complex but decomposable (Pong, Breakout, multi-step
  tool use, language generation conditioned on emotional state).
- Primitives can be defined on synthetic or hand-curated microenvs
  where clustering is verifiable.
- Primitives are reusable across multiple downstream tasks (one
  vocabulary, many extensions).

### When to NOT use

- The target task does not decompose into primitives that cluster —
  e.g. the only meaningful structure is at the trajectory level
  (rare; usually a sign you should rethink the labeling).
- The cost of building synthetic primitive envs exceeds the cost of
  letting the substrate discover structure unsupervised (then use
  Mode B).

### Knobs

| Knob | What it controls |
|------|------------------|
| `seed` | The shared L0 seed across primitive donors. Must be identical for `absorb` to work. |
| `cap_bytes` per primitive | Each primitive donor gets its own cap. Small primitives can use small caps (e.g. 4_000 B). |
| Class ID layout | Disjoint integer ranges per primitive. Document the layout explicitly. |

---

## 4. Mode B — unsupervised watch (optional fallback)

> **Status:** optional. Use only when the target task **cannot** be
> decomposed into primitives (Mode E). For most cases Mode E is
> strictly cleaner — the substrate gets verifiable clustering, the
> archive stays interpretable, and growth fires on real loss signals
> rather than on auto-generated cluster CE.

Used when the data does not respect Mode A's per-class-clustering
assumption, but you still want trioron to discover structure in it. The
canonical case is **streaming sensory data with no externally meaningful
labels**: gameplay frames, raw audio, raw video.

The wrapper lives in `trioron/bridge/watch.py` (planned). It does not
introduce a new core mechanism — it generates the y stream on the fly
and feeds the standard supervised pipeline.

### Mechanism

For each input `x` arriving in the stream:

1. Compute `z = L0(x)`.
2. Score `z` against every existing archive Gaussian `(μ_c, σ_c)`:
   `s_c = log p(z | μ_c, σ_c)`.
3. **If** `max_c s_c > novelty_threshold`: assign this sample's `y` to
   the matching cluster c. The archive update on this batch will refine
   `(μ_c, σ_c)`.
4. **Else** (genuine novelty): open a new class slot `c_new`, seed
   `μ_{c_new}` from this sample's `z`, initialize `σ_{c_new}` to a
   prior, and assign `y = c_new`. Head is extended by one slot; growth
   may follow on the next plateau.
5. Hand the assembled mini-batch of `(x, y_dynamic)` to the standard
   training loop. CE + EWC + manifold-archive update all proceed
   exactly as in Mode A.

### What you end up with at end of watch

- An archive of `N_clusters` Gaussians, where `N_clusters` was
  determined by the data and `novelty_threshold`, not externally fixed.
- Each cluster represents "a kind of input trioron has seen often
  enough to recognize" — server-position, ball-mid-rally, post-score
  reset, etc., for a Pong stream.
- L1 codes that separate those clusters (because L1 was trained to make
  CE on the cluster IDs separable, and grew when separation plateaued).
- A head that emits logits over discovered cluster slots.

### What "unsupervised" means here, precisely

The y is **not** externally supplied. It is generated online from the
archive's nearest-cluster assignment, with novelty triggering new
clusters. The CE loss the trioron sees is real — it's just over
auto-generated labels. This is structurally identical to **online
hard-EM clustering** with the trioron's L0 fixed and L1+head playing
the role of a discriminator that sharpens the cluster assignments
across passes.

### Phase split

Mode B is typically followed by a separate phase-2 head that maps the
trioron's L1 code (or archive-likelihood vector) to whatever downstream
target the deployment needs (action, tool name, language token, etc.).
See §5.

### When to use

- Streaming sensory inputs with no clean class labels (raw frames,
  audio, mixed multimodal streams).
- A teacher exists that can *play* a task but cannot give you per-step
  labels with the right cluster-in-L0 property (Atari RAM-skill teacher
  in `experiments/atari_trioron`).
- You want trioron's archive to stratify "kinds of state" independent
  of any downstream policy or tool, to be reused across multiple
  downstream heads (one substrate, many phase-2s).

### When to NOT use

- You already have meaningful y labels with the per-class-clustering
  property — Mode A is strictly cleaner.
- You need the archive to encode something other than L0-space
  clustering (e.g. trajectory-level structure — see "future work" at
  the bottom).

### Knobs

| Knob | What it controls |
|------|------------------|
| `novelty_threshold` | Log-likelihood threshold for "novel enough to open a new cluster." Lower = more clusters; higher = fewer, broader. Choose by calibrating on a held-out stream. |
| `min_samples_per_cluster_before_freeze` | Avoid one-shot clusters that never get refined; merge or drop them at end of watch. |
| `max_clusters` | Hard cap on cluster count, to avoid runaway growth on pathological streams. |
| All Mode A knobs | Apply unchanged once labels are generated. |

---

## 5. Mode C — bridged operation (encoder + L0Adapter)

Used when raw input is not in the substrate's native code space — text
sentences, images, audio waveforms, raw frames. The `trioron/bridge/`
subpackage handles this.

### Pipeline

```
raw input ──► Encoder ──► L0Adapter ──► trioron substrate
              (frozen)    (deterministic
                          random projection,
                          seeded)
```

- `Encoder` is any frozen feature extractor whose output dim is
  declared via `encode_dim`. Reference encoders ship for text
  (`sentence-transformers`), image (`open-clip-torch`), and audio
  (`openai-whisper`).
- `L0Adapter` projects `encoder_dim → l0_dim` via a deterministic
  Gaussian random projection seeded from the L0 seed. When
  `encoder_dim == l0_dim` it collapses to identity.
- The shared-L0 invariant generalizes: any donor and any recipient that
  share **both an L0 seed and an encoder choice** also share the
  adapter, and absorb works without retraining.

### When to use

- Cross-modal organisms (text branch + image branch absorbed under one
  L0 seed).
- Mode B over rich raw inputs — the Atari watch session goes through
  here, with an `AtariFrameEncoder` that's just a flatten + normalize
  (or a learnable feature extractor in a later iteration).

### Anti-patterns

- Mismatched L0 seeds across donors that share an encoder. The
  fallback path (`_build_random_projection`) is **untested** and known
  to take a ~30pp accuracy hit; treat it as a debugging convenience,
  not a deployment path.
- Swapping encoders mid-deployment. The L0Adapter is part of the
  shared substrate; changing the encoder invalidates it.

---

## 6. Mode D — policy head over trioron codes (self-imitation)

The deployment story: trioron summarizes input state, a small downstream
head turns that summary into an action / tool / token. Trioron stays
frozen during phase-2 training (or grows only on novel state), and the
phase-2 head is trained on filtered behavioral data.

### Pipeline

```
phase 1 (Mode B): trioron watches a stream → frozen substrate + archive
phase 2 (Mode D): action_head: L1_code → action
                  trained by self-imitation on top-k rallies by reward
```

### Mechanism

1. Roll out in the environment using the *current* organism + action
   head as the policy (or the teacher policy on the first pass).
2. Segment trajectories by reward boundary (rally / episode).
3. Filter to top-k segments by total reward.
4. Bag the (state, action) pairs from those segments.
5. Train the action head on those pairs by CE / MSE.
6. Optionally let trioron grow on genuinely novel states encountered
   during phase 2 (re-enter a brief Mode B sub-loop).

### Why this isn't Mode A in disguise

The action head is *not* the trioron head. The trioron's manifold
archive does not learn an action-conditional Gaussian — it stays
clustered on the *kind-of-state* labels from phase 1. The action head
is a tiny external module that sees the L1 code and emits an action
distribution; it does not require its targets to cluster in L0.

### When to use

- RL-style tasks where the substrate's job is recognition and the
  policy is a thin shell on top.
- Multi-task deployment where one trioron substrate serves several
  downstream heads (one Mode B, many Mode D heads).

---

## 7. Anti-patterns gallery

Concrete failures from past sessions, with the symptom and the fix.

### 7.1 Pushing `(state, action)` tuples through Mode A on Atari

**Symptom:** loss plateaus at `≈ ln(K)` (uniform CE for K classes).
Growth fires the prescribed number of times but loss doesn't drop.
Confusion matrix (when computed) shows one class predicted everywhere.

**Cause:** "press UP" is the right action in many unrelated game states.
UP-samples don't cluster in L0. The archive's `(μ_UP, σ_UP)` becomes a
wide bag spanning the whole L0 manifold. Frustration fires on
"everything looks like UP" and growth has nothing useful to grow into.

**Fix:** decompose into primitives via Mode E. Build a vocabulary of
clustering-clean concepts (UP, DOWN, MOVE, BALL, …), absorb them, then
extend onto Pong. Mode B is a fallback if primitive decomposition
genuinely fails.

**Where this happened:** `experiments/atari_trioron/skill_curriculum.py`
in the previous session. See `outputs/atari_trioron_skill/run.log`.

### 7.2 `terminal_on_life_loss=True` at data-collection time

**Symptom:** Breakout dataset is ~26× smaller than Pong's despite the
same `n_episodes` setting.

**Cause:** `gymnasium.wrappers.AtariPreprocessing(terminal_on_life_loss=True)`
turns each life loss into an episode termination. Pong has no lives
(scoring doesn't terminate), so 16 episodes ≈ 16 full games. Breakout
has 5 lives, so 16 "episodes" = 16 single-life rollouts.

**Fix:** at *data-collection* time, use `terminal_on_life_loss=False`
and let the rollout run a full Breakout game per episode. Use
`terminal_on_life_loss=True` only at *training* time, where each life
is supposed to be precious.

**Where this happened:** `experiments/atari_trioron/skills.py` in the
previous session.

### 7.3 Asserting mode collapse without measuring it

**Symptom:** memory note claims "predicts class X for every input,
accuracy = prevalence of X" but the cited percentages don't appear in
any committed log.

**Cause:** the diagnosis was inferred from `loss ≈ ln(K)`, not from a
direct confusion-matrix measurement. `loss ≈ ln(K)` is *consistent with*
mode collapse but also with under-training, eval-time index
misalignment, or a head/archive ordering bug.

**Fix:** before claiming mode collapse, compute and commit a confusion
matrix. The trioron API does not currently emit one; experiments need to
do it themselves with a small post-fit eval pass that compares
`organism(X_train).argmax(-1)` to `y_train`.

### 7.4 Conflating teacher roles

**Symptom:** unclear what trioron is being asked to imitate; teacher
biases bleed into pattern discovery without it being noticed.

**Cause:** "teacher" in an RL-flavored setting splits into two roles —
**stream generator** (whose trajectories trioron sees) and **utility
signal** (what tells trioron which patterns matter). Conflating them
turns "behavior cloning of a fixed teacher" into "self-imitation with a
fixed-teacher start," which is a different experiment.

**Fix:** state explicitly which entity plays each role in any
experiment writeup. See the discussion in `atari_trioron` planning.

### 7.5 Per-class manifold for trajectory-level concepts

**Symptom:** archive Gaussians have huge σ; routing accuracy is near
chance even on data the head classifies correctly.

**Cause:** the concept the user meant by "class c" is a property of the
*trajectory* (e.g. "winning rally," "successful tool call") rather than
of the *state*. A trajectory's frames don't share an L0 geometry; the
archive averages them into a meaningless centroid.

**Fix:** either use Mode B with state-level cluster IDs and a
trajectory-level phase-2 head, or extend the archive mechanism to
operate over pooled trajectory codes (future work; not yet implemented).

### 7.6 Skipping primitive bootstrap on a task that needs it

**Symptom:** a task that "should" decompose cleanly is being trained
end-to-end via Mode A or via Mode B and underperforming.

**Cause:** Mode E was skipped because primitives "felt obvious" or
"too small to be worth a separate donor." But the substrate's archive
ends up trying to learn the primitive vocabulary AND the task
composition AND the routing all at once, and growth fires on the wrong
plateau.

**Fix:** spend the cheap iteration to write a primitive vocabulary
and train each primitive as its own ~5K-sample synthetic donor. The
substrate cost is negligible (small caps per primitive); the
extension on top almost always learns faster than the end-to-end
attempt.

---

## 8. Future work / what this manual does not cover

- **Trajectory-level archives.** The current archive is per-state-class.
  A pooled-trajectory archive would unlock proper "winning-rally
  recognizer" mode without leaving Mode A. Open question.
- **Hierarchical clusters.** Mode B currently produces a flat list of
  clusters. A hierarchical version (cluster-of-clusters) would let the
  archive express "Pong scenes vs. Breakout scenes" *and* the within-
  game sub-clusters in one structure.
- **On-the-fly cluster merging.** Mode B opens new clusters readily;
  merging redundant ones lives in the dream phase, but it's not yet
  exercised against discovered (rather than supervised) clusters.

When any of these become load-bearing, update this manual rather than
silently re-interpreting an existing mode.
