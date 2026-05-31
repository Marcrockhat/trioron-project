# Axis 7 — Temporal / Working Memory (design proposal)

**Status:** proposal for review (Chloe, 2026-05-31). Not yet folded into
`paper/v3/spec.md`. Written at Rocky's request during the sense→logic
design session. Review, then promote the accepted parts into spec §3.7
and §9 (partition map) before any `trioron/` code lands.

> **Numbering correction.** The `temporal-cognition-gap` memory guessed
> temporal "probably maps to Axis 6." It does **not** — Axis 6 is already
> taken by cell-spawning (`axis6_spawn`, spec line 3127). The v1.1 axis
> write-functions are: `set_input_source` (1), `archive_input` (2),
> `insert_layer` (3), `set_axonal_gain` (4), `grow_branch` (5),
> `axis6_spawn` (6). The temporal axis is therefore **Axis 7**.

---

## 1. Why temporal, and why now

The project's curriculum is **sense → logic → symbol → language**, built
bottom-up ("logic before language" — discover fire before telling stories
around it; `logic-before-language-principle`). Today's session built the
first rung above perception — a grounded sense→valence world
(`experiments/bench_grounded_world.py`) where **emotion = the learned
anticipation of an innate drive-consequence** (fire *looks* dangerous
because the organism learned fire→pain).

That definition of emotion is the forcing function for this axis:

- **Emotion is predictive.** "Percept now → consequence later" is a
  *temporal* credit assignment. A purely feed-forward, single-pass
  substrate has no place to hold "I saw fire a moment ago" while the pain
  signal arrives. Emotion-as-learned-valence is impossible without a
  temporal substrate.
- **Multi-step inference needs it.** The (4) reasoning rung —
  `A→B, B→C ⊢ A→C`, or the branch `fire→red→hot→{meal, danger}` — requires
  holding intermediate conclusions across compute steps. That is
  composition *in time*, the complement of composition *in space* (depth,
  Axis 3 / the self-arrange work). The two are the two halves of "can it
  reason."
- **The substrate has no working memory today.** Activations live for one
  forward pass and vanish. Confirmed empirically this session: the
  grounded-world substrate memorizes surface cues and does not compose
  (nourish-recall 0.99 i.i.d. → 0.35 on a held-out cue), and self-arranged
  depth does not fix it. Static depth is not the missing piece; *state over
  time* is the untested one.

---

## 2. Two timescales (the axis has two parts)

### 2.1 Within-pass recurrence — **already specced, never exercised**

Spec §3.5: the `recurrent` gene (bit 3) lets a cell carry self-edges; the
scheduler unrolls `K_unroll` (default 1, max 8) steps per forward pass.
This is the mechanism for multi-step inference *inside one input* — a
3-hop chain unrolls in 3 recurrent steps.

**The gap is not the primitive — it is that nothing grows or exercises
it.** No base seeds recurrent cells; no growth path promotes a cell to
recurrent; no bench needs it. Like the bipartite-MLP finding
(`substrate-is-bipartite-mlp`), the capability exists in the arena but the
*policies* never reach for it.

**Proposed addition (self-arranged recurrence, not hand-crafted — per
`substrate-self-organizes-architecture`):** under *temporal frustration*
(§4), promote a high-utility cell to `recurrent` (set bit 3, add a
self-edge with small init weight, `K_unroll` starts at 2) — the temporal
analogue of frustration→`divide`. Depth-in-time emerges the same way depth
-in-space does.

### 2.2 Across-pass working memory — **new**

A persistent state trace that survives *between* forward passes within a
**wake phase**, so the organism can relate input at step *t* to input at
step *t−k*. This is what `temporal-cognition-gap` described and what
emotion-as-prediction requires.

**New gene — bit 10, `mnemonic`** (epigenome bits 10–15 are reserved for
future genes, spec §2.2). A mnemonic cell carries a per-cell memory trace:

```
m_v^{(t)} = λ_v · m_v^{(t-1)} + (1 - λ_v) · a_v^{(t)}     # leaky trace, within wake phase
y_v^{(t)} = combine(a_v^{(t)}, m_v^{(t)})                 # current activation reads its own past
```

- `λ_v` ∈ [0,1) is a per-cell, learnable decay (the memory horizon). λ→0
  is memoryless (byte-identical to linear — preserves backward compat,
  matching the dendrite K=1 convention §3.6). λ→1 is long persistence.
- The trace is *state*, not a parameter: ~2 floats/cell (`m_v`, `λ_v`).
  Bounded, cheap (cf. lifetime budget, §6).

---

## 3. The phase signal IS the clock

Trioron already has a coarse temporal rhythm: the **wake → frustration →
dream → consolidation** cycle. Reuse it as the time coordinate instead of
per-event timestamps (which bloat linearly with activity — wrong
granularity, `temporal-cognition-gap`).

- The forward API gains a `phase ∈ {wake, dream}` argument and a
  within-wake `step` index.
- **Wake:** mnemonic traces accumulate and persist across passes; recurrent
  cells unroll; plasticity is high.
- **Dream onset:** traces **consolidate** — the information a mnemonic cell
  held across the wake phase is written into weights via the existing dream
  / KIBRA-tag / replay machinery — then traces **reset** to zero. Plasticity
  is gated to the consolidation pathway only.
- "Learning consumes time" emerges for free: a freshly grown recurrent /
  mnemonic cell is immature; the maturation gradient (grow → tag →
  consolidate → lock) is the explicit time-cost of acquiring a temporal
  skill.

This keeps a single, biologically-motivated time axis rather than two
competing notions of "step."

---

## 4. Temporal frustration (the growth trigger)

Existing frustration (`lifecycle/grow.py`, spec §4.2) fires on sustained
high loss and triggers `divide` (more width). **Temporal frustration** is a
new internal-state signal that fires when loss is reducible *only* by
carrying information across time — detectable cheaply as: error remains
high while the *current* input is individually ambiguous but the *recent
trace* would disambiguate it (i.e. the gradient w.r.t. a held trace would
be large if a trace existed).

First-cut proxy (no new gradient plumbing): if frustration is sustained
**and** the task is flagged sequential/delayed, route the growth event to
`promote_recurrent` / `promote_mnemonic` instead of `divide`. Later, make
the routing itself learned (a drive, like the others).

This is the key design commitment: **temporal capacity grows under
temporal stress, the same way width grows under ordinary stress.** No
hand-crafted recurrent layers.

---

## 5. Write functions (spec convention)

Following the `set_input_source / … / axis6_spawn` family:

- `set_recurrence(arena, cell_id, k_unroll=2)` — set bit 3, add self-edge,
  set `K_unroll`. (Formalizes what §3.5 already allows.)
- `promote_recurrent(arena, cell_id)` — growth-time helper; the temporal
  analogue of `divide`. Picks a high-utility active cell, makes it
  recurrent.
- `set_working_memory(arena, cell_id, decay=0.5)` — set bit 10 `mnemonic`,
  init `m_v=0`, `λ_v=decay`.
- `promote_mnemonic(arena, cell_id)` — growth-time helper.

All preserve the envelope check and the DAG/cycle invariants (recurrent
back-edges are the *only* legal cycles, spec §2.3 / §3.5).

---

## 6. Lifetime budget (sizing)

Per the `lifetime-dendritic-budget-sizing` pattern (Axis 5 added only
1.105× to the 80-yr envelope). Working memory adds **state, not
parameters**: ~2 floats per mnemonic cell (`m_v`, `λ_v`) + 1 self-edge
weight per recurrent cell. If a fraction *f* of cells become mnemonic, the
multiplier on per-cell footprint is `1 + f · (8 bytes / ~500 bytes/cell)`
≈ `1 + 0.016·f` — negligible (<0.2% at f=10%). Recurrence adds at most one
self-edge per recurrent cell. **Temporal is the cheapest axis by storage;**
its cost is *latency* (K_unroll re-dispatch), bounded at 8× on recurrent
buckets only.

---

## 7. Falsification gate — **PASSED** (2026-05-31, n=3)

> **Result.** `experiments/bench_temporal_gate.py` (delayed grounded recall).
> temporal-off is pinned at chance (0.200, 5 balanced classes) at every
> delay — memoryless cannot do delayed recall. temporal-on (substrate carries
> its own leaky interior trace, BPTT through the sequence) recovers the cue:
> 0.995±0.003 (1 step) → 0.957±0.011 (2 steps) → 0.726±0.010 (4 steps),
> λ=0.6. Δ = +0.80 / +0.76 / +0.53 — **PASS at every delay** (bar 0.30), with
> a clean memory-horizon decay. The faithful version: internal recurrence via
> fed-back `live_activations`, not a handed-in echo of past input. Contrast:
> self-arrange *depth* never lifted any task (Δ≈0); temporal memory does. Axis
> 7 ships live (gate-level); next is the per-cell `mnemonic` gene in core.

Original gate spec (for reference):

Mirror the Phase 6 dendrite-delta gate (`phase_6_dendrite_delta_pass`): a
minimal task where temporal **must** help, with a clean A/B.

**Task: delayed grounded association.** Reuse the grounded world. Deliver a
percept's features across *two* steps (cue at t, outcome-relevant feature
at t+1), so the consequence is only predictable by relating the two. Or:
the multi-hop chain `fire→red→hot→danger` delivered one hop per step.

- **temporal-off** (no mnemonic, K_unroll=1): cannot relate across steps →
  pinned at the marginal.
- **temporal-on** (mnemonic / promote_recurrent under temporal
  frustration): should recover the delayed association.

**Gate:** temporal-on beats temporal-off by Δ ≥ (TBD, e.g. +0.3 abs on the
delayed task) at n=3. If it does not, the axis stays **dormant** (gene
defined, never auto-promoted) rather than shipping live — the same
discipline that kept Axis 3 `insert_layer` off after it regressed on
chained-15 (`all_axes_chained15_regression`).

---

## 8. Relation to this session's work

- The grounded-world bench (`bench_grounded_world.py`) is the *static*
  precursor: percept→consequence in one pass. The temporal version is
  percept(t)→consequence(t+k) — the delayed-association gate above. Emotion
  becomes genuinely predictive once Axis 7 is in.
- The self-arrange depth work (`--self-arrange`, `grow.py:same_rank_edges`)
  is composition-in-space; this is composition-in-time. They are
  orthogonal and combinable, but should be gated **separately** — combining
  un-validated axes regressed before (`all_axes_chained15_regression`).

---

## 9. Open questions

1. **Trace vs. recurrence overlap.** A mnemonic leaky trace and a
   `recurrent` self-edge both carry information forward. Are both needed,
   or is the leaky trace just a recurrent cell with a fixed self-weight
   `λ`? Likely the trace is the *across-pass* (between inputs) form and the
   self-edge is the *within-pass* (unroll) form — different scopes, both
   needed. Confirm before implementing both.
2. **Where does `step` come from on a non-sequential bench?** Most current
   benches present i.i.d. batches with no time order. Temporal needs an
   *ordered* data presentation. The grounded delayed-association task
   supplies it; chained-15 etc. do not. Temporal benches are a new data
   family.
3. **Consolidation math.** "Write the wake trace into weights at dream" is
   stated but not specified. Likely reuses manifold replay / KIBRA; needs
   its own derivation.
4. **Temporal-frustration detector.** The cheap proxy (route growth to
   recurrent when sequential+frustrated) is a placeholder; the principled
   version (gradient-w.r.t.-held-trace) needs plumbing.
5. **Interaction with credit-based locking.** A mnemonic cell's utility is
   spread over time; the engagement/utility EMAs (spec §2.1) may need a
   temporal correction so memory cells aren't pruned for low instantaneous
   activation.
