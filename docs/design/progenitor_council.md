# Progenitor–Council architecture (design)

**Status:** design / exploratory. Captured session 022 (2026-06-07).
Not yet built, not yet in `paper/v3/spec.md`. This is the candidate
direction; promote to spec once a slice is validated.

> Origin: Rocky's correction that the v2 growth mechanism is
> biologically wrong and that the assistant had been making the
> conventional ML decision in his place. This document records the
> model we converged on so it survives across sessions/PCs.

---

## 1. Why (what's broken in v2)

- **`divide()` lets mature neurons divide.** `grow.py:43` has no
  plasticity gate — it splits any cell handed to it, and copies the
  parent's epigenome (`grow.py:65`), so a differentiated LINEAR cell
  spawns another LINEAR cell. That's **clonal expansion of mature
  neurons** — biologically impossible, and exactly the dead,
  undifferentiated growth behind the s021 "grew to cap for zero gain".
  The growth audit (s020/s021) was measuring a broken mechanism.
- **Growth can't add a cell *type* the problem needs.** The DENDRITE
  (quad) phenotype exists as a gene, but `divide()` only clones the
  parent's type, so a LINEAR-seeded substrate grows only LINEAR cells —
  it can never cross a nonlinear wall by growing (confirmed on the
  disruptor-dog problem: linear cells cap ~0.79, Bayes 0.85).
- **Perception transforms are hand-coded.** We hand-applied `log` to
  the weight feature. A monotone warp like log is just a hidden layer
  of units at different weight magnitudes — the substrate should *learn*
  it, not have us pre-transform the input.
- **Frustration is orphaned.** `FrustrationDetector` is a global
  task-loss scalar; `check_growth_trigger` has no live caller in core
  (only one experiment). Differentiation (`MorphogenField`) is decided
  by **position**, blind to the data. Nothing makes the *data* decide
  what a cell becomes.

## 2. The capital reframe

Parameter growth is **not** to be minimised during learning. The
minimum-parameter solution (e.g. ~19–33 params for the disruptor) is the
**consolidation target** — the deployment form — not a constraint on the
learning phase. Biology *overproduces* (neurons/synapses) then prunes;
the progenitor pool is **working capital**. The s021 error was never
"grew too much" — it was "grew *undifferentiated*". Capital is fine;
idle, undifferentiated capital is the waste. Numa gates *whether the
capital is being consumed by learning*; "dispose" = **consolidate**, not
**refuse to grow**.

## 3. The model

### 3.1 Germline vs soma
- **Germline** — the (one, wide) **progenitor** + its **council**.
  Always plastic, **never frozen, never pruned**. Consolidation /
  credit-locking must never touch this class. Grafting transplants
  germline (plastic potential), so it survives across populations.
- **Soma** — the differentiated branch cells the progenitor spawns per
  the council's verdict. These do the work and **are** what
  consolidation prunes to the minimum form.

### 3.2 Progenitor
- **One wide progenitor** (Rocky's choice). "Wide" = fan-in scales with
  input dimensionality. Its job on sensory input is **retinal
  compression**: ingest the raw sensor field, project down to the
  working resolution, structured (its positional sensor cells are the
  center-surround equivalent). The dimensionality ceiling is absorbed at
  the sensor, never pushed to the council.
- **Working resolution = 1,572,864 (1536×1024, "1.5 Mi").** A touch
  above the human optic nerve (~1.2M ganglion cells ≈ HD) so we're not
  under-human, and below Full HD (2.07M). Everything past the progenitor
  sizes to this. On full vision the progenitor compresses ~100 MP → 1.5
  M (~80:1, the retina's order).

### 3.3 Council
- **One 4-cell group per expression gene**, even multiplicity so a split
  vote is a legitimate "undecided" the data must topple (not a deadlock
  to break). **Never pruned** — it's the standing decision organ, reused
  across decisions. Originally 5 phenotypes × 4 = 20; **s030 (Rocky):
  TANH added as a sixth expression gene → 6 × 4 = 24.**
- Phenotypes are the existing `EXPRESSION_GENES`
  (`epigenome.py:21`): **LINEAR, DENDRITE, CONV, ATTENTION, RECURRENT,
  TANH (s030)** — mapping to tabular / relational-curvature /
  image-localized / routing / temporal / bounded-saturation. The palette
  is in the gene layout (tanh = bit 12, spec §3.10).

### 3.4 Differentiation = trial-vote (data decides, not position)
- Progenitor spawns a cohort of trial daughters, one per phenotype.
- They **compete on the branch's own learning signal** (gradient /
  credit / local frustration = the vote). The phenotype the local
  problem rewards wins; the data breaks the council's balance.
- Progenitor commits a soma cell of the winning phenotype onto the
  branch and **stays plastic** (asymmetric, self-renewing division).
- **The vote at the perception layer *is* input classification** —
  modality is whatever phenotype wins. No separate classifier.

### 3.5 Division law (mature neurons don't divide)
Division is **plasticity-gated**: only high-plasticity (germline /
shallow) cells divide; a cell that takes a phenotype matures and loses
divisibility. Plasticity ≈ the λ epigenetic lock (low λ = plastic).

Probability is a **niche gradient over depth × a resource penalty**:

```
P_divide(z, u) = [ ln(1/z) / ln(1/z_inner) ] · (1 − u)
```

- `z = layer / depth`, z=0 at perception (input side, near progenitor),
  z=1 at output. `z_inner` = first interior layer.
- `ln` chosen over `tan`: it gives the niche shape — steepest near the
  input with a graded transit-amplifying tail — naturally; the z=0 pole
  is moot because perception cells don't divide.
- `(1 − u)`, u = fraction of parameter budget used: the resource
  penalty, applied at **all** layers. Division is costly; each division
  raises u → lowers the next one's probability (homeostatic
  self-throttle). At **u=0 there is no base tax** (Rocky's call) — a
  fresh substrate divides across the whole proliferative column.
- Depth is **biologically shallow** — tens of layers, not thousands
  (cortex ≈ 10 hierarchical levels; the 100-step rule caps serial depth
  ~100). Capital goes into **width**, and RECURRENT supplies
  depth-in-time. The division gradient is steep over a short depth.

Table at u=0, depth 10 (layer 1 = next to progenitor, layer 10 = output):

| layer | z | P_divide(u=0) |
|---:|---:|---:|
| 1 | 0.1 | 1.000 |
| 2 | 0.2 | 0.699 |
| 3 | 0.3 | 0.523 |
| 4 | 0.4 | 0.398 |
| 5 | 0.5 | 0.301 |
| 6 | 0.6 | 0.222 |
| 7 | 0.7 | 0.155 |
| 8 | 0.8 | 0.097 |
| 9 | 0.9 | 0.046 |
| 10 | 1.0 | 0.000 |

### 3.6 Perception that learns the transform
- Progenitor spawns **sensor + dendritic cells** that **chunk the
  input**, each positionally aware (owns a region/scale). Cells at
  different weight magnitudes reconstruct a monotone warp (log) —
  the transform we hand-coded, grown instead.
- They **feed branches directly** and progressively **take over the
  input layer**; the progenitor steps back but stays plastic.
- **Stop signal = "make sense of the input"**, read as the council's
  balance toppling to a stable verdict (consensus = comprehension).
  Keep spawning until the data decisively represents/classifies.

## 4. What already exists vs the gaps

**Exists (in `lifecycle/developmental.py`, `bases/developmental.py` —
uncommitted, carried since s005):**
- `EXPRESSION_GENES` = the 5-phenotype palette.
- Stem cells: `STEM_SENTINEL`, `spawn_stem`, `is_stem` — the germline.
- `MorphogenField` (position→phenotype) + `differentiate` /
  `differentiate_all_stems` with primitive lateral inhibition.

**Gaps to build:**
1. **Plasticity-gate `divide()`** — refuse division of mature (λ-locked)
   cells; division becomes stem-only.
2. **Data-driven differentiation** — replace "position decides
   phenotype" (morphogen argmax) with the council trial-vote on local
   frustration/credit over the 5 genes. Morphogen stays as a positional
   *prior*; data topples it.
3. **Local frustration field** — frustration is currently global +
   orphaned; make it per-cell/per-region (the comprehension signal).
4. **Council scaffold** — standing 5×4 balanced council; `never-freeze`
   germline class; consensus = stop signal.
5. **Wide progenitor + retinal compression** to the 1.5 Mi working
   resolution; `PHENOTYPE_CHANNELS` currently wires only 4 channels
   (no DENDRITE/RECURRENT) — extend to the full five.

## 5. Testbed

The 6-class weight problem (`experiments/growth_exercise/chicken_goat.py`)
is the differentiation probe:
- **dog** = `Uniform(2, 85)` — a *disruptor* spanning chihuahua→great
  dane; overlaps cat below and goat above. Creates a **non-linearly-
  separable** Bayes partition (dog & goat each win two disjoint
  intervals) → linear cells cap ~0.79, needs a DENDRITE.
- **elephant** = `Normal(5000, 800)` — larger than cow but cleanly
  separated (recall 1.0).
- 6-class weight-only Bayes ceiling = **0.852**. A correct council
  should grow a DENDRITE for dog and stay linear elsewhere.
- Toy seed (D=1, C=6, council 5×4, one wide progenitor): ≈28 nodes,
  ~215 params (+28 λ) — ~0.4% of the 50K Phase-1 budget.

## 6. Open knobs

- Niche steepness γ on `ln(1/z)^γ` (sharper cliff vs fatter tail).
- Whether the resource tail is multiplicative (small nonzero) or
  subtractive (hard-cut mature layers to 0).
- Council vote integration window (how much data topples the balance).
- Exact retinal-compression ratio / structure in the progenitor.

## 7. Branch-id locality architecture (s027 — supersedes council/divide growth)

The council/`divide()` growth tangled on the capacity-hard taxonomy: every
soma divided from one shared parent and `project_to_consumers` wired each
child to all of the parent's consumers, so later somata wired into earlier
ones — a rank-12 sibling pile-up from 29 cells (audit: 112 later-sibling
back-wires; longest path = 11 somata in series, not composition), plus an
all-to-all hidden→output readout that dilutes every weight. `same_rank_edges`
broke the bipartite lock but only produced the tangle; it gave no controlled
*width*. The `divide()` symmetric/axial flag is cosmetic (position only — the
edge policy never reads it).

**The replacement is a 2-D sheet of shallow thin columns**, addressed by a
**hard `branch_id`** (`[x,y]`) and a **`layer`** (depth):

- **Thin chain:** exactly one cell per `(branch_id, layer)`. Width = number of
  branches; depth = column height, **bounded shallow** (cortex-like — scale
  lives in width, not depth).
- **Two independent axes per growth event** (separate, not coupled):
  - *Orientation*: **lateral** = mint a fresh `branch_id`, a new cell at layer 1
    reading all perception (parallel feature / new column); **axial** = inherit
    the `branch_id`, `layer+1`, drawing only from the column's own current top.
  - *Phenotype*: linear / dendrite(GCU) / attention / … (how it combines). The
    deep-linear-collapse physics (substrate has no ambient nonlinearity) makes a
    linear *axial* cell vacuous, so meaningful depth requires a nonlinear
    phenotype — the physics biases the grid, it does not gate it.
- **Readout reads TOPS only:** each branch's top cell feeds the outputs; when a
  column deepens, its output edges are **re-pointed** (`edge_src` rewritten in
  place) from the old top to the new — sparse, no all-to-all dilution, no edge
  removal needed.

**Frustration splits `[F_lateral, F_depth]`, measured PER BRANCH (locality):**

- `F_lateral[branch]` = an *unread input direction* (local effective rank below
  local input dim). It **saturates cleanly** when the basis is complete — width
  stops minting redundant branches on its own (this also kills the inherit/new
  edge-dup bug, which was a symptom of the tiny rank-0 pool).
- `F_depth[branch]` = the residual that *survives* width — by definition
  non-linear in the local features. Because the depth trigger is **per-branch**,
  it is **not starved by global width competition** — the documented fix for
  *"depth never fires"* (the recurring failure across prior experiments).

**Loop:** Phase 1 widen until `F_lateral` saturates → Phase 2 deepen the
most-frustrated column with a nonlinear phenotype until `F_depth` saturates.
Saturate-then-deepen, locally.

**First result** (`step5_branch.py` / `run_branch.py`, GCU dendrite, hard
taxonomy, seed 0): width saturates on its own at **13 branches** (0.077→0.409 ≈
the linear ceiling); **DEPTH FIRES** and helps (**+0.068**, 0.409→0.478). No
tangle (max depth 5, one column, all composition branch-local). Validates the
mechanisms; absolute accuracy (0.478 vs council ~0.78) is low because **Phase 2
concentrated all depth in ONE column** — post-convergence `|w·g|` is a weak
depth-target (every converged top reads ~0, the selector tie-breaks to branch 0,
then its fresh GCU cell dominates). **Next:** spread depth across columns with a
residual / which-class-still-wrong depth-target signal instead of raw `|w·g|`.

### Still open / known caveats (s027)

- `trioron/phenotype/dendrite.py` is now **GCU `σ(z)=z·cos z` GLOBALLY** (Rocky's
  call) — tests asserting `z+z²` will fail; a future cleanup is per-cell/config
  selection rather than a global swap.
- Branch architecture lives at experiment level (`step5_branch.py`); it does NOT
  use the package `divide()` (which still tangles). Promotion into
  `trioron/progenitor/` waits until depth-spread + readout are validated.
- The epigenetic lock λ and per-cell utility `u` are still **unwired** in the
  growth path (`arena.utility` reads all-zero; no `accumulate_saliency`). The
  prune arm and the normalized-λ lock (log-domain, per-weight self-ratio) are
  designed but unbuilt.

## 8. Conjoined-twin CONV spawn rule (s040 — makes the CONV gene non-trivial)

**Origin: Rocky, s040.** The whole-session thread (table-vs-image): the
substrate is a flat *table* learner — readout sums over columns
(permutation-invariant), division judges one column at a time. Neither
reads column *adjacency*, so an image's geometry is invisible. Encoding
position into a column *value* (amplitude-ramp Gabor, NJ-tree-as-features,
chord/Fourier positional code, stereo/PSD ratio) is **inert**: the
readout never relates two columns, so a position *value* never becomes a
position *relation* (the s039 NJ-tree probe recovered the 2D topology
exactly — 1.58 px — yet was NULL as features, for exactly this reason).
Spatial structure is a **wiring + a cross-column operation**, not a value.

The three fixes that put that wiring *in the organism* (so it stays
data-type-adaptive — image → grows spatial structure, table → stays flat):

1. **Spatial coordinates** — a topographic field on the inputs:
   `f → (row, col, orient, scale)`. For our senses this is a *known exact
   lookup*, not something to discover (raw: `f→(f//28, f%28)`; gabor:
   `c=f//49, (i,j)=(f%49)//7, f%49%7`). §3.6 already calls for
   "positionally aware sensor cells (owns a region/scale)" — this is that
   field, made an explicit arena buffer.
2. **Proposer** — growth that reads the field: spawn/split on a
   coordinate-local *patch*, not the worst anonymous column. The §3.4
   council vote (CONV maps to "image-localized") or §7's lateral-growth
   event is the proposer; it gains a coordinate input.
3. **Weight-tying** — the conjoined-twin rule below.

**The reduction that makes the rule necessary.** `trioron/phenotype/conv.py`
(its own "Reduction guarantee," lines 15–20): when a CONV cell is its own
`lineage_root`, every edge reads its *own* weight, so `y = b + Σ w·a` —
**byte-identical to LINEAR.** Convolution does not exist until **≥2 cells
share a root.** So a council CONV win, as specced, spawns a *linear cell in
disguise* — the CONV gene is dead on arrival.

**The rule.** When CONV wins (or a lateral event is tagged image-localized),
the progenitor spawns a **conjoined cohort, not one cell**:
- ≥2 CONV soma cells sharing **one** `lineage_root` (the kernel lives once);
- each twin's fan-in = the **raw input/perception columns in its own patch**
  (NOT the phase-injected RECEPTOR cells — those carry angles,
  `scheduler.py:292`; the conv cell *is* the sensor, spec §3.4), wired in
  **matched relative-offset (tap-ordinal) order** so `conv.forward_batch`'s
  `(root, tap)` share lines up tap-i of twin A with tap-i of twin B;
- twin outputs feed the quantizer as new pocket dims (composer-cell path);
- the pair is the **seed**; the full form **tiles the cohort across the
  region** (free coverage — new sensor cells, one kernel, §3.4).

**Conditions (non-negotiable):**
- Twins must read **different, translation-related** patches. Same fan-in →
  identical redundant cells; arbitrary different fan-in → one kernel forced
  to explain unrelated inputs (a handicap). Only neighboring/translated
  patches = real convolution.
- Both twins must be **CONV phenotype** — tying lives only in
  `conv.forward_batch`; a LINEAR twin silently reads its own weights.

**First test (deferred proposer):** hand-wire a tied pair/cohort to a small
input grid and validate the *mechanism* (fixes #1+#3) before automating the
spawn (#2): (a) correctness — a single CONV cell == LINEAR (reduction), a
tied pair applies one kernel at two positions (translation equivariance) and
gradient ties them; (b) benefit — on a translation task (same local pattern
at varying positions) tied twins generalize/sample-efficient over untied
independent cells. `experiments/progenitor/test_conjoined_conv.py`.
The conv-on-conv stack is a later stage (needs the layer-1 feature map to
carry positions too); first layer over the sensor field only.

**Validated (s040).** `test_conjoined_conv.py`: reduction PASS (lone CONV ==
linear, max|Δ|=5e-7), tying PASS (translation equivariance exact + gradient
tying), benefit PASS (tied train 1.000/test 0.977 vs untied 0.571/0.273 on a
held-out-position translation task — *and* a key side-finding: the benefit
only appears with a translation-invariant POOLING readout, never a
per-position head — "the consumer must pool too"; our PCLL matched filter
already sums over columns, so it is compatible). `conv_proposer.py`: the
§3.4 trial-vote, instantiated as CONV-cohort-vs-LINEAR on held-out
competence — DATA-TYPE ADAPTIVITY PASS (spawns on real 2-D grid adv +0.57,
rejects when columns shuffled / locality destroyed, conv→chance).

**Real-data validation (s040, `conv_proposer_chained15.py`).** On the real
30-way sensor field, vote uses conv-maxpool vs raw-logreg:

| arm | C=12 REAL | C=48 REAL | C=48 SHUF |
|---|---|---|---|
| raw-logreg | 0.798 | 0.796 | 0.794 |
| conv-maxpool | 0.432 | 0.683 | 0.639 |
| conv-flatten | 0.763 | 0.754 | 0.762 |

Vote **REJECTS** CONV at both channel counts (conv < raw). Corrected
reading (the C=12 run alone was UNDER-POWERED — a methodological caveat,
s040): 4× channels did NOT let conv-flatten cross logreg (0.754 < 0.796),
so the reject is *not* a channel-count artifact. Nuances: (1) at C=48
conv-maxpool shows real>shuf (+0.044) → locality IS exploited, just not
enough to beat raw on CENTERED data; (2) conv-maxpool reaches 86% of
logreg accuracy at ~11% of the params — a vote weighing accuracy-per-param
(the 50K envelope) could flip, UNSETTLED. **Untested lever: DEPTH** — this
is all SINGLE-LAYER conv; a conv→pool→conv hierarchy is deferred (§8 "later
stage"). Honest claim: *single-layer conv does not beat logreg on centered
chained-15* — NOT "conv can't help". Consistent with the s039 keystone
(perception not the chained-15 bottleneck) but does not close depth.
**Before promotion: test depth, and/or demonstrate positive CONV value on a
translation-structured task (CIFAR / shifted-MNIST / embodied arc).**

**Depth result + positive (s041, `conv_depth_shifted_mnist.py`).** The §8
"later stage" — the conv→pool→conv stack — is now built and validated, and it
is the FIRST positive CONV result on real-ish data. Task: shifted-MNIST (a
28×28 digit dropped at a random offset in [0,8]² on a 36×36 canvas, so the
object MOVES — a tabular learner that sums fixed columns cannot follow it).
Both conv arms use the real `conv.forward_batch` weight-tie; the new piece is a
**second conjoined-twin cohort that tiles over the 2×2-pooled L1 feature map**,
each L2 channel reading ACROSS all L1 channels in its K2×K2 patch in matched
tap order (real cross-channel 2nd-layer convolution, one shared kernel per
channel). Pooling is the consumer's job (ReLU→maxpool between layers), as §8
predicted ("the consumer must pool too").

| arm | REAL | SHUFFLED |
|---|---|---|
| raw-logreg | 0.493 | 0.488 |
| conv1 (1-layer) | 0.498 | 0.261 |
| **conv2 (DEPTH)** | **0.842** | 0.417 |

(n=2 seeds, C1=8 K1=5 s2 / pool 2×2 / C2=12 K2=3 s1, 8 epochs, Adam.) Three
findings: (1) **DEPTH is the lever.** Single-layer conv merely ties the tabular
logreg (0.498 vs 0.493 — neither follows the moving object); the conv→pool→conv
hierarchy lifts it to 0.842 (**+0.344** over 1-layer, **+0.350** over logreg).
This is the positive that single-layer could never show on centered chained-15
— DEPTH was indeed the untested lever, not channel count. (2) **The advantage
is LOCALITY, not capacity.** With columns shuffled by a fixed permutation
(geometry destroyed), conv2 collapses 0.842→0.417 (a **−0.426** locality gap)
while permutation-invariant logreg is unchanged (0.493→0.488). The depth win
comes from the spatial wiring, not from being a bigger nonlinear net. (3) The
gradient-free regime is still untested (this is Adam-on-kernels, the s040
caveat carries). Honest scope: this proves *the conjoined-twin depth mechanism
helps where geometry matters* — it does NOT revisit centered chained-15 (still
the wrong demo) and does NOT yet show gradient-free conv. Next: either run the
depth stack on centered chained-15 to close that gap too, or move conv toward
the gradient-free PCLL pipeline / a package home (§7 lateral-growth event).

## 9. Phasor optics — gradient-free spectral front-end (s041 exploration)

**Origin: Rocky, s041.** A whole-session design exploration (NOT promoted; five
toy scripts under `experiments/progenitor/`). The frame: treat a data sample as
a **filter film**; an emitter sweeps phasor references through it; identity,
position, and motion are read as **optical responses**. Computationally this is
a gradient-free, fixed-filter front-end whose back-end is the existing
**ManifoldArchive** (per-class Gaussian centroid). It targets the standing
PCLL open problem ("can a conv/feature front-end be learned/built without
backprop") and the s039 keystone ("position is a relation between references,
not a value"), and continues the s037–039 frequency/Gabor direction.

**The receptor facts it builds on** (`core/receptor.py`, `pcll/lockin.py`):
per-sample `quantize` maps each feature to one pocket `q∈[0,1000]`, `θ=2πq/1000`
(history-independent contrast code; q=0/1000 are reference, masked). The
canonical frame (`mixed.py`) re-references samples into a stream-wide
`[FLO,FHI]` that widens until frozen — so absolute brightness drifts unless
frozen. Lock-in deposits a UNIT phasor per feature; magnitude is discarded and
re-emerges only as **coherence** (resultant length). Margin = amplitude/√n =
R·√N: coherent features grow ∝N over the stream and clear k; incoherent ones
hug the √N floor (and a discriminative-across-classes feature stays incoherent —
the "ripple" that flags a new class). The full circle has a **cos-fold** about π
(q and 1000−q share a real part), so mid-gray features partly cancel; keeping
the complex (cos,sin) pair avoids the collapse.

**The three channels (each validated on toys):**
1. **WHAT — dimensionality-adaptive scattering lens** (`fingerprint_lens.py`).
   A hypercube patch matched to the data's dimensionality (1D→1×k, 2D→k×k,
   nD→kⁿ); each patch → a complex mean phasor `(re,im)` (cos/sin = the two
   deflection axes); concatenate → descriptor → nearest class **centroid**
   (= ManifoldArchive). On the 3-class 3×5 toy: 2×2 + **contrast-normalized**
   q → **0.93** nearest-centroid (vs 0.68 with the brightness-keeping canonical
   frame — contrast-norm is the lever; it is the per-sample `quantize`, NOT
   Gabor, which only detects oriented edges and still scales with contrast).
   The global quanta histogram alone overlaps — the spatial grouping is what
   separates.
2. **WHERE — stereo dual-frequency emitter** (`stereo_emitter_2d.py`). Two
   emitters impose phase ramps across an axis at **incommensurate rates**
   (periods 16/13; the 0.3π offset only de-aligns the sawtooths). A single
   emitter wraps (resolves one period); the two-frequency pair is unique over
   `lcm(periods)` → **absolute position by Vernier unwrap, MAE 0.05 col over a
   48-col field.** One emitter pair per axis (col, then row for 2D, depth for
   3D) — emitters as dimensionality-adaptive as the lens. This is the s039
   "position is a relation" fix: the cross-emitter phase **difference** is the
   relation.
3. **WHEN — lock-in as moving screen** (design only this session). The lock-in's
   coherent integration over the stream IS a temporal detector; a moving object
   decomposes as spatial scattering × temporal lock-in. Ties to Axis 7
   (temporal gene). Untested.

**Combined what+where** (`combined_what_where.py`, `digit_bench_2d.py`). Stereo
localizes → crop registers → lens identifies. Moving off-center 3-class:
pos MAE 0.00, id 0.967. Scaled to **10 digits, 2D off-center, 16×40 field**:
2-pair where → **id 0.645** (vs col-only ablation 0.265; chance 0.10).
- **Falsified hypothesis (tested, not assumed):** giving the field row headroom
  made localization WORSE (row MAE 1.17→4.71→8.47 as H 16→28→40) — background
  accumulates in the marginal and an extended shape-varying object biases the
  phase centroid. Stereo localizes a *point* well, an extended object poorly.
- **Architectural fix:** **couple where↔what** — move the aperture to maximize
  the identity margin (the screen *seeks* the lock-in). id 0.555→0.645,
  row MAE 1.17→0.62. Where and what are **coupled, not feed-forward.**

**Ambiguity / refusal margin.** Centroid gap (nearest vs 2nd-nearest) is a
confidence signal. A 2→Z morph collapses it 0.66→0.08 at the midpoint and the
label destabilizes (2→7); clean-digit vs ambiguous margins separate (a refusal
threshold ~0.25). This is the novelty/refusal mechanism for the mixed stream
(§10.7 division-by-frustration) in the spectral front-end's terms.

**Open threads (NEXT):** (1) the full **wavelength sweep** (multi-carrier
filterbank spectrum) — only the complex scattering lens is built, not the swept
spectrum; (2) the **WHEN** channel on an object moving over *time*; (3) the
**word/symbol capacity** claim (continuous centroid descriptor dodges the
1000-pocket aliasing ceiling) — untested; (4) **3D/4D** with a matched moving
screen; (5) **promotion** to a package (`ScatteringLens` + stereo emitter +
manifold path) — deferred until the exploration matures (Rocky's call). All
gradient-free; fits PCLL. None promoted.
