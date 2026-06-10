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
- **5 phenotypes × 4 cells = 20**, even multiplicity so a split vote is
  a legitimate "undecided" the data must topple (not a deadlock to
  break). **Never pruned** — it's the standing decision organ, reused
  across decisions.
- Phenotypes are the existing `EXPRESSION_GENES`
  (`epigenome.py:21`): **LINEAR, DENDRITE, CONV, ATTENTION, RECURRENT** —
  mapping to tabular / relational-curvature / image-localized /
  routing / temporal. The palette is already in the gene layout.

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
