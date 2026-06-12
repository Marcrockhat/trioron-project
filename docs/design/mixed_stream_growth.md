# Mixed-Stream Growth — division, composers, the structural contract (design, s032)

> Status: **APPROVED (Rocky, s032) — D11–D16 all approved; building in M-phase order.** Companion to
> `pcll_substrate_integration.md` (D1–D10, the s030 integration this extends)
> and `receptor_period_frustration.md` (the PCLL method). Implements Rocky's
> s031 regime decision (mixed stream; division IS class discovery) and the
> s032 corrections (genome-constrained composers; growth must be
> arena-countable). Decisions flagged **[D11]–[D16]**, continuing the register.

---

## 0. Why this doc exists — the trust criterion (Rocky, s032)

The rebuild's aim: **a network that changes its own topology and depth to
adapt to the problem, and learns continually with less forgetting.** The five
experimental features (progenitor/genesis, council/votes, frustration gate,
PCLL, manifold adapter) all serve that aim — and the test of each is
structural: **cell, edge, and rank counts must move when the organism learns.**

The s032 census of the integrated I5 organism, after learning all 32
capacity-hard classes:

```
cells: 69/192   computing (fwd=True): 12   bookkeeping: 57
edges: 0        ranks: {0}                 classes: 32
```

Twelve computing cells fixed at tick 1, zero edges, zero depth. Every growth
event to date allocated a bookkeeping row. The validated growth results
(division 1→75 classes at 0.711 one-shot; composer spawn loop) live in
standalone probes only. This design wires them into the substrate under a
contract that makes the census numbers move — or fails its gates.

Current feature status this design must repair:

| feature | status |
|---|---|
| progenitor + genesis | wired; only ever spawns perception tissue |
| council + votes | wired; decide() signals have NO consumer; settlement pays false credit |
| frustration gate | wired; period-level only — blind to sample-level mode blur |
| PCLL | wired, working (zero forgetting); grows templates, not tissue |
| manifold adapter for PCLL | **does not exist** |

---

## 1. The structural contract  **[D11]**

Every growth decision maps to a declared arena-structure delta, asserted by
gates:

| event | arena delta |
|---|---|
| division (class discovery) | parent astrocyte row → 2 sibling astrocyte rows, parent lineage recorded (`arena.parent`); bookkeeping by design — classes are memory, not tissue |
| composer spawn | **+1 computing cell** (`forward_inclusion=True`, epigenome = the winning expression gene), **+2 edges** from its source perception cells, **rank = max(src rank)+1** — depth is earned, never asserted |
| receptor attach (sensation) | existing perception cell flips ACTIVE + RECEPTOR |
| composer prune (PATIENCE) | cell retired, its edges masked, bet repaid to the book |

**Census instrument:** a `census(arena)` readout — cells by kind
(computing/progenitor/council/astrocyte), edges, ranks present, classes —
emitted in every `MeetingReport` and printed at every growth event. Gate
runners assert structural deltas (before/after counts), not accuracy alone.
This is standing instrumentation, not a probe.

## 2. Genesis births ONE world-class; division is the consumer  **[D12]**

Mixed-stream regime (Rocky, s031): no class periods, no labels, no
boundaries. The stream arrives shuffled; a period is W=1000 stream samples
(distinct from N_QUANTA=1000 pocket resolution).

- **Genesis:** the first period births one blurred world-class (one
  signature, one astrocyte). All early FRUSTRATED stress is mode blur — and
  that is now the *intended* discovery signal, not noise.
- **The consumer:** `StressRouter.decide(FRUSTRATED)` — today emitted and
  dropped (s031 diag: 14 signals, no consumer) — routes to the progenitor,
  who **divides** the attributed class: partition its rolling buffer
  (circular 2-means on the best per-dim split), spawn sibling signatures +
  astrocytes, retire the parent row.
- Probe-validated mechanics carried in: split acceptance is **per-dim**
  (a global mean gain can never fire, ~gain/12); the best child must beat
  `max(parent+GAIN_D, NULL_SPLIT=0.72)` (the 2/π noise-slicing null);
  **per-class rolling buffers**, never window members (windows starve once
  classes multiply); MIN_CHILD=25 recurrence floor; growth self-arrests on
  the gain criterion (probe: 0 divisions p12–16 with no cap).
- Expected census trajectory (the M2 gate): astrocyte rows 1 → ~60–76 on the
  32-truth stream, one-shot ≥ 0.65 (probe: 0.711 vs 0.386 class-sequential).

## 3. Per-class settlement attribution  **[D13]**

The false-credit confound (s031 diag): settlement judges "is the stress gone
at the next boundary?" globally, so the book paid SENSATION to its floor and
left every phenotype group flat at 4.500 while the world changed underneath.

Fix: a pending growth decision carries the **class it answered**
(`pending = (driver, class_name)`). Settlement succeeds iff *that class's*
stress cleared — its buffer coherence recovered (division) or its resolution
margin cleared the floor (sensation) — not iff the period status improved.
Vote transfers keep step4 semantics (equal share, floors, Σ=28 conserved);
only the success predicate changes.

## 4. The composer arm — overproduction over the GENOME  **[D14]**

The s032 principle: a council seated by expression genes cannot vote for a
function with no gene bit; a spawn nothing can express is not a growth event.
The small fixed genome is a regularizer — an unbounded function family always
finds structure in selection-biased buffers (the probe's seed-0 junk spawns).

- **Candidate set = expressible genes with a scalar composition form over a
  receptor dim pair (c_i, c_j):**
  - LINEAR: w₀c_i + w₁c_j, wirings sum/diff
  - TANH: tanh(2.5·(w₀c_i + w₁c_j)) — bounded, anti-detonation (spec §3.10)
  - DENDRITE: two branches, one input each, quad σ(z)=z+z² → c_i²+c_j²
    (the Phase-6 ring form)
  - ATTENTION / CONV / RECURRENT: **deferred** — no scalar form on receptor
    dims; their composer semantics is an open design question, recorded here
    so it is decided, not improvised.
- **Importance-gated wiring** (failure-mode-4 fix, part 1): candidate dim
  pairs draw only from `Germline.perception_importance` — noise columns have
  no lock-in margin in any class and never enter a trial.
- **Future-deposit settlement** (part 2): the trial selects on the buffer
  (permutation-null-corrected gain, split-half confirm — failure modes 1–3
  fixes carry over), but the spawn is **settled only by deposits that arrive
  after it exists** (D9 next-boundary semantics). Membership-induced
  confabulation cannot survive: the selecting buffer never pays the bet.
- **Causal quantization** (replaces the probe's non-causal batch min-max):
  every candidate form above is **bounded by construction** on pocket inputs
  (linear/tanh ⊂ [−1,1]; dendritic quad ⊂ [0,0.5]), so each composed dim
  carries a **fixed static frame** known at spawn time — one sample
  quantizes alone, the frame registry (D10d) stamps it like any other dim.
- **Spawn = real tissue** per §1: the new cell computes its composition in
  the forward path each tick and deposits into its own lock-in row like any
  receptor. Settlement success transfers one vote to the winning gene's
  council group (`winner_phenotype()` finally gains its consumer); PATIENCE
  failure retires the cell and repays the bet.
- Growth ladder (decided s031): frustrated → divide (cheap) → division
  can't clear → composer trial → spawn winner gene; void → receptor attach.

## 4b. Probe amendments to the composer arm (s033 — **APPROVED, Rocky, s033**)

The genome redo of the relational testbed (`run_composer_genome.py`,
handoff s033 NEXT 1) validated the §4 mechanism and surfaced four
corrections, each probe-measured. None is built into the substrate yet;
they need Rocky's sign-off as D-register entries before M4 integration.

1. **Trial statistic = residual-incoherence ratio** (amends D14's
   "permutation-null-corrected gain"). R saturates near 1 and ring
   marginals are clumpy (cos²φ piles at extremes): null carve 0.93 vs
   relation 0.995 = "gain" 0.06, invisible in R units. Score
   `(1−null)/(1−carve) > NULL_RATIO=2`: noise pairs 1.0-1.2, tanh
   relations 2.5-2.9, dendrite on ring pools 13-22.
2. **[D17 proposed] Family trials** — per-buffer trials can NEVER
   discover manifold relations: division legitimately fragments
   continuous manifolds into raw-separable arcs (the s031 premise "no
   raw dim separates E|F" is false for fragments), so no confounded
   buffer of trial size ever exists. Repair: trial over unions of
   LINEAGE SIBLINGS (current buffers under a retired division ancestor,
   deepest pool first, one spawn per sitting). Reunified-E ratio 21.4
   vs control unions 1.2-1.5. The composer thereby doubles as the
   anti-fragmentation mechanism — the probe-level answer to the
   deferred under-division/sibling-merge carry (s031 NEXT 4). A merge
   CONSUMER (collapse a family the composed dim explains) remains
   future work.
   **Relation scope (Rocky, s033): the pool walk is BOUNDED to
   FAMILY_DEGREE generations** — family trials follow the active
   fragmentation frontier; unbounded ancestor pools re-test near-root
   mixtures every sitting (cost + multiple-comparisons surface). Swept
   at 8 seeds: degree-1 0.676±.067, degree-2 0.698±.067, **degree-3
   0.736±.031, degree-4 0.735±.031, degree-5 0.735±.031** (d4/d5
   byte-identical — no pool 4+ generations above a leaf changes a
   decision), degree-6 = unbounded 0.726±.048 (division-only
   0.606±.030). The measured tree: max lineage depth 7, median leaf
   depth 6, every seed/arm. The bound must cover the depth a manifold
   fragments to before its family pool reaches trial size — rings go
   ~3 generations here, so 1-2 degrees leaves ring signal on the
   table; the curve PLATEAUS at 3-5 and degrades only when near-root
   mixtures re-enter (d6/unbounded). FAMILY_DEGREE = 3 — first point
   on the plateau, frontier-local, lowest variance.
3. **Importance = any-class buffer coherence** (sharpens D14's
   "importance-gated wiring"). The genesis margin RANKS informative
   dims above noise (2.8-3.8 vs 0.7-2.2) but an absolute K=3 cut
   starves trials (one dim clears → no pairs). The design's own words
   pick the gate: "noise columns have no lock-in margin IN ANY CLASS" —
   wire dims on which any judgable-size (≥ MIN_MEMBERS) class buffer
   coheres at R > 0.9. Smaller buffers confabulate noise coherence
   (failure mode 4 reaches importance too).
4. **Division and composition COMPETE at the sitting** (the ladder's
   "division can't clear", read locally): a strict division-first
   ladder starves the composer (relational buffers fragment forever on
   legitimate micro-modes). The spawn wins iff its carve beats the best
   division's (the s031 probe's validated loop). Also: PATIENCE
   settlement judges on the virgin data available (≥ 2·MIN_CHILD)
   rather than auto-failing a near-full store.

Probe result with all four (3 seeds, 8 windows): genome-composer 0.701
vs division-only 0.607; A .50→.68, B .34→.52, E .28→.47; spawns 3-4 per
seed, future-deposit prunes 0-2, **zero noise-pair spawns**; books
differentiate (linear/tanh/dendrite earn on their own spawns). The
absolute A/B/E/F numbers are capped by fragmentation + the raw
equal-weight readout (M5's σ-readout territory) — the M4 substrate gate
should be set against THESE measured baselines, not the s031 non-genome
probe's 0.595/0.561.

## 5. The manifold adapter for PCLL  **[D15]**

Feature 5 of the rebuild, currently absent. Reuse
`trioron/learning/manifold.py` (`ManifoldAstrocyte`: per-class μ/σ sketch,
`sample()`, `log_likelihood()`) over **pocket space** (code_dim = number of
read dims):

- **Update:** at each boundary, the meeting feeds the period's per-class
  member pockets to that class's sketch (streaming μ/σ — the signature
  already holds first moments; the sketch adds spread).
- **Consumers, in priority order:**
  1. **Post-quiescence annealing** (handoff NEXT 5): after divisions
     self-arrest, `sample()` synthesizes deposits that re-anneal templates —
     fixes the slow late decay seen in the freeze run (~0.75 plateau).
  2. **σ-weighted readout** (handoff NEXT 3): `log_likelihood()` replaces the
     raw equal-weight matched filter — the measured ~0.62→0.99 half of the
     Bayes-ceiling gap (target 0.993, Rocky s031).
  3. **Replay guard for grown tissue:** when composer cells (and any future
     gradient training) reshape the read space, spec-§4.5 replay from the
     sketches protects PCLL-discovered classes — the "less forgetful" half
     of the rebuild aim, wired to the PCLL path for the first time.

## 6. Build phases + gates (each falsifiable, census-asserted)

| phase | build | gate (PASS required before next) |
|---|---|---|
| **M1** | census instrument + structural fields in `MeetingReport` | **PASS (s032, 5fbf5b3)** — accuracy identical (0.387/0.386/0.384); baseline 12 computing / 0 edges / depth 0 asserted, all seeds |
| **M2** | world-class genesis + division consumer (mixed stream) | **PASS (s032, a0031f6)** — one-shot 0.760 (probe 0.711), 31–32/32 truths, every decide() consumed, astrocytes 1→75–84. Two corrections: pockets translated to a canonical EXTREMES frame (D10 affine, frozen at genesis) before division — mean-frame clamping or no frame at all divides forever; self-arrest = RATE COLLAPSE (≤5% in tail, 2-epoch verified) bounded by the world's true mode count (data_hard = 32×3 = 96 modes), not absolute-zero windows |
| **M3** | per-class settlement attribution | **PASS (s033)** — i5_diag scenario: the flat-4.500 false-credit signature is dead (0 settlements, all 14 frustrated decisions DEFER — their classes never testify again; book pristine 24/4, Σ=28). Mixed scenario: 74–83 divisions/seed each settle against their OWN children's split-dim coherence, success 0.89–0.95 (mean 0.924; the ~8% failures repay honestly); discrimination holds 26–27 votes (the old global predicate bled it every frustrated discovery window). Settlement granularity: one pending per division (a division answers ONE class); subjects with no testimony defer — silence pays nobody. The PCLL-path subject rule is strict (testifies only as resolution WINNER): an ambiguous pair attributed to the wrong tentative winner defers forever rather than collecting unearned credit (I3-C dog case, accepted) |
| **M4** | composer arm (genome-constrained) | **PASS (s033)** — testbed through the substrate: composer 0.694 > division-only 0.624 (relational gain A+B+E +0.51); census EVERY seed: computing +2/+3, edges +2·spawns, **depth 0→1 (the first earned rank)**; zero noise-pair spawns; forward-path composed pockets ≡ buffer-side composition ≤ 6 quanta; Σ=28; future-deposit prunes fired (1-2/seed). Found+fixed: construct() defaulted to an EMPTY dispatch table (every grown computing cell silently skipped — the s020 inert-growth class); `divide_tries` (worlds with pure-noise dims never divide on worst-dim-only). data_hard check (per-class, 3 seeds): **zero spawns** — the genome regularizer + relation null correctly quiet on axis-separable frame-moving data; mean 0.760→0.811 from multi-try division alone (sp026 +0.46, sp020 +0.30; 94 vs 84 classes). Dendrite spawns occurred but none survived settlement on the testbed (frame-jitter on virgin stores — open observation, not a gate item) |
| **M5** | manifold adapter | annealing recovers the freeze-decay; σ-readout closes measured gap toward 0.99 on clean classes with raw arm reported alongside (s029 comparability); ship/wake round-trips sketches |
| **M6** | spec §10 amendment + promotion | spec-first rule (D6); manual updated |

## 7. Decision register (continues D1–D10)

| # | decision | status |
|---|---|---|
| D11 | structural contract: every growth event has a declared, gate-asserted arena delta; standing census in MeetingReport | approved (Rocky, s032) |
| D12 | mixed-stream genesis births ONE world-class; decide(FRUSTRATED) consumed by progenitor division (per-dim acceptance, NULL_SPLIT floor, rolling buffers) | regime decided s031; wiring approved (Rocky, s032) |
| D13 | settlement carries the answered class; success = that class's stress cleared, not global period status | approved (Rocky, s032) |
| D14 | composer candidates = LINEAR/TANH/DENDRITE-quad only (scalar-composable genes); ATTENTION/CONV/RECURRENT deferred explicitly; importance-gated wiring + future-deposit settlement; fixed static frames (causal quantization); spawn = real cell + 2 edges + rank>0 | approved (Rocky, s032) |
| D15 | manifold adapter = `ManifoldAstrocyte` sketches over pocket space; consumers = annealing, σ-readout, replay guard | approved (Rocky, s032) |
| D16 | prune semantics: PATIENCE failure hard-retires the composer cell (edges masked, bet repaid); soft-apoptosis recoverable latch deferred until a probe shows re-spawn churn | approved (Rocky, s032) |
