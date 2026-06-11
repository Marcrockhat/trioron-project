# PCLL → Substrate Integration (design, s030)

> Status: **DESIGN FOR DISCUSSION — nothing here is built.** Companion to
> `receptor_period_frustration.md` (the PCLL spec). That doc defines the method;
> this one defines how it becomes trioron substrate machinery, per Rocky's s029
> directive ("integrate this into triorons") and the s030 developmental flow
> (Rocky): **period 1 = perception generation by the progenitor; periods 2+ =
> council meetings at each period boundary; genesis.py's batch pre-pass is
> REPLACED.** Grounded against the real code on both sides (file:line refs).
> Decisions flagged **[D1]–[D7]**; D1–D6 carried from the first draft with
> Rocky's s030 answers folded in, D7 is new.

---

## 0. The developmental timeline (the s030 reframe)

The organism starts as **the germline only: the progenitor WITH her council
attached** — one unit, born together, never frozen, never pruned
(`progenitor_council.md` §3.1: germline = "the (one, wide) progenitor + its
council"). No pre-pass, no probe training, no hand-built input layer.
Division of labour throughout: **the council judges, the progenitor spawns**
(mature cells don't divide; the council never spawns — it is the standing
decision organ). Structure is spawned live from data:

```
period 1 — PERCEPTION GENERATION (council-judged, progenitor-executed)
  tick 1:  first sample arrives. The progenitor reads it through her own
           perception node (a built-in aperture: sees the raw vector's
           dimensionality and value structure; transforms nothing).
           → spawns one PERCEPTION cell per input column
             (the status-quo trioron input layer, genesis.py:81-90 shape,
              but spawned by the progenitor, not by a pre-pass).
  tick 2:  second sample arrives. Receptor adaptation is now calculable
           (two samples = first usable range/structure estimate).
           → progenitor spawns/equips RECEPTOR per column accordingly
             (continuous vs discrete vs flat — see §1).
  ticks 3..S: receptors keep calibrating their STRUCTURAL parameters
           (distinct-value census for discreteness, k levels), and the council
           watches the per-column evidence accumulate (deposit activity,
           distinctness, first coherence) — this is HOW it judges the
           importance of features/dimensions: not by gradient saliency but by
           which columns produce evidence. The per-sample quantisation stays
           history-free (PCLL spec §2) — what the history informs is the
           spawn-time structural decision, exactly the epigenome principle:
           phenotype chosen from the problem at the growth event (manual §2.5).
  boundary: the council's FIRST sitting — commits each column's structural
           verdict (discrete k / continuous / starve) and the initial
           importance ranking that seeds the vote ledger.

periods 2..N — COUNCIL LOOP (one meeting per period boundary)
  during the period: ticks stream; each receptor deposits (cos θ, sin θ)
           into its lock-in accumulator (mask rule: q∈{0,1000} deposit
           nothing). No meetings mid-period; no gradients ever.
  at the boundary (the council MEETING):
    1. resolve        → status ∈ {RESOLVED, FRUSTRATED, EMPTY} + margins
    2. schedule-learn → coherence gate → tangential χ² fit → update matched
                        signature OR birth a new one
    3. vote           → stress driver routes the conserved vote economy
                        (empty → perception group, ambiguity → composers)
    4. spawn/retire   → commit what the votes selected; habituation updates
    5. reset          → lock-in accumulators zeroed; next period starts clean
```

**Cost (Rocky's question: does this slow learning?) — NO.** A tick is one
quantisation + two trig adds per feature (~100 flops at F=16); no backward
pass exists anywhere in the loop. The s029 standalone discovery already runs
~16k ticks (32 classes × S=500) in seconds of pure Python; deposits are
additive and order-free within a period, so a period can be vectorised as a
batch when the stream allows. 10⁵–10⁶ ticks is minutes worst-case unvectorised,
sub-second vectorised. The council meeting is O(C·F) once per period —
negligible. Perception generation adds exactly one period of latency, total.

---

## 1. Period 1: progenitor replaces genesis  **[D1] [D7]**

`genesis.py`'s two batch stages (variance apoptosis, then a 21-step saliency
probe with gradients) are **retired** for PCLL organisms. Their jobs are
subsumed natively:

| genesis stage | subsumed by |
|---|---|
| variance apoptosis (kill dead columns) | the **mask rule**: a flat column quantises to a constant/extreme → deposits nothing → its receptor never accumulates evidence → retired by habituation. Dead columns die of starvation, not of a pre-pass. |
| saliency probe (gradient |w·g| ranking) | **lock-in evidence itself**: a column that never contributes to a coherent signature has nothing protecting it. No gradients needed. |

**[D1] (carried, Rocky's flow confirms it): receptor is a gene.** New epigenome
gene `RECEPTOR` (bit 11), co-expressed with `PERCEPTION` (bit 5). The scheduler
already special-cases perception injection (`scheduler.py:181-183`,
`act[:, perc_ids] = x[:, :n_perc]`); a RECEPTOR cell receives the phase
instead:

```
q_j = round(1000 · (x_j − lo) / (hi − lo))   lo = min(0, x.min()), hi = x.max()
θ_j = 2π · q_j / 1000                        (per-sample, stateless — PCLL §2)
```

**[D7] — discreteness detection (new, needed by the tick-2 adaptation).** The
progenitor decides each column's receptor structure from the early-period
census:

- maintain the set of distinct quantised values seen per column during period 1;
- if |distinct| ≤ K_DISCRETE (propose 8) at period end → **discrete receptor**:
  labeled lines q = 1000·(2j+1)/(2k) (PCLL §11.3), `receptor_levels[cell] = k`;
- else → continuous receptor (`receptor_levels = 0`);
- never-varying column → no receptor structure survives (mask-rule starvation).

Tick 2 gives the *first* estimate (Rocky: "bias has been calculated" =
receptor adaptation); the census just keeps refining it until the period-1
boundary commits the structural choice. Per-cell metadata: one new int tensor
`receptor_levels[capacity]`.

**Partitioning rule stands** (s029 dead-3-sensors finding): receptors stay 1:1
with columns, no mixing before lock-in. If a perception NN stack ever sits
upstream, it must be partitioned per sensor group — out of scope, standing rule.

**Working resolution / aperture adaptation (s022 §3.2) — what is built vs not.**
The progenitor's aperture ceiling is the 1.5 Mi working resolution
(1,572,864 = 1536×1024); everything past her sizes to it. **Adaptive
perception SIZING is built and validated** (Rocky, s030 correction): the
oversized-aperture → converge-to-the-real-features loop succeeded in
`step2_genesis` (single feature recovered from the aperture stand-in),
`step2b_apoptosis` (apoptosis-FIRST ordering — a signal diluted 1-of-1.5M IS
noise, so the empty bulk dies by variance in one pass before saliency runs),
and `genesis.py` (64 → exactly 7). I2 preserves that capability gradient-free
(zero-variance → census-constant starve; |w·g| saliency → coherence +
habituation), and `positions.py` carries the foveated position tagging.
What remains UNBUILT is **dense-field compression**: input where every slot
carries variance (real vision, ~100 MP), where selection can't reduce anything
and the ~80:1 reduction must come from LOCAL pooling — per the partitioning
rule, never a dense global projection (the NN-front-end failure mode at
scale); the s022 center-surround positional cells are the intended per-region
units. Implementation debts at that scale: the census is a Python set per
column (fine at 64, not at 1.5 M); lock-in tensors are fine (~18 MB). Slot
after I5 — it needs a dense wide-input testbed to gate against.

---

## 2. Lock-in accumulator → per-cell state  **[D2]**

Three new per-cell **non-trainable** arena tensors, lifecycle-identical to
`engagement`/`utility` (driver-updated, SGD-invisible, outside the parameter
envelope):

| tensor | shape | meaning |
|---|---|---|
| `lockin_re` | [capacity] f32 | Σ cos θ this period (zero-mean carrier, PCLL §11.1) |
| `lockin_im` | [capacity] f32 | Σ sin θ this period |
| `lockin_n`  | [capacity] f32 | deposits this period (masked q∈{0,1000} excluded) |

No complex dtype needed — two f32 columns are the SoA-native encoding. Zero on
non-RECEPTOR cells (same pattern as `branch_alpha` on non-dendrite cells).

**Period boundary = `end_task()`** (`construct.py:72`) — the circadian point
made literal. The meeting (§0 steps 1–5) runs there, gated on "any RECEPTOR
cells exist", as the one new always-on boundary action. It is microseconds.
**[D3] (carried):** the full `dream_cycle` (replay→consolidate→rejuvenate,
`dream.py:123`) stays driver-called — we keep the manual's "no built-in hook
calls the learning machinery" contract; the PCLL meeting is the cheap
exception, and a RESOLVED meeting is the natural place for a driver to *choose*
to dream.

**Resolution margins → frustration interface (carried).** `PCLLResolution`
adapter exposes `.is_frustrated` / `.multiplier` (driven by the parameter-free
margins) so `FrustrationDetector` consumers (`step4_grow.py:149-248`) are
untouched. The loss-plateau detector survives for gradient-trained organs;
replacement is rejected (chained-15 and the council loops depend on it).

---

## 3. The council meeting → vote economy  **[D4]**

Rocky (s030): **votes fire when one period is over** — one meeting per
boundary, which is exactly the cadence `stress.py` already validated. One
economy, not two:

- The per-phenotype vote ledger (`step4_grow.py:_vote_transfer`, conserved
  Σ=20) gains **perception as a votable group**. Empty-stress transfers votes
  toward it (spawn = recruit a dropped/new receptor); ambiguity-stress toward
  composer phenotypes (LINEAR/DENDRITE), as `grow_council_frustration` already
  biases its winner.
- **Status exclusivity holds** (s029): one status per period → no cross-driver
  arbitration; votes are within-group prioritisation (driver credibility).
- **Habituation = the retire pattern**: ×0.5 per fruitless growth, floor 0.2
  (PCLL §8). Lives on the driver. In the developmental flow it is also the
  apoptosis path for period-1 over-spawn (noise-column receptors).
- **Recruit-on-ambiguity** uses shadow signatures (§4): re-attach a
  sensed-but-not-read receptor, template extended from shadow, zero relearning
  (validated 0.224→0.377 monotone, s029).

**[D4] (carried, recommendation unchanged):** the perception vote group is the
**live receptor population** (a group key in the ledger), not a germline
quartet — receptors are 1:1 with sensors; fake council cells would be
meaningless. To be precise about who holds what: **the council holds the vote
ledger and the judgment** (it is the standing decision organ, germline,
attached to the progenitor from tick 1); the receptor population is what is
voted ON; **the progenitor is the sole spawner** that executes the verdict
(s022: mature cells don't divide, the council never spawns).

---

## 4. Signatures ↔ ManifoldArchive  **[D5]**

A learned PCLL class IS a running-mean complex signature (arg = range center,
|T| = width) — and `ManifoldAstrocyte.update()` is already an online running
mean (Welford, `manifold.py:51-189`) stored in an arena cell with
`forward_inclusion=False`:

- **Encoding:** T ∈ ℂ^F as code_dim = 2F real (re/im interleaved).
  `spawn(class_id, code_dim=2F)` per birth; `update_class()` per matched
  period; `finalize_all()` at dream. All existing API.
- **Welford `_m2` is load-bearing:** per-dim variance over (re, im) is the raw
  material for the tangential χ² fit null (df ≈ active features, PCLL §11.4).
  Promotion replaces `schedule_learn.py`'s private store with the astrocyte and
  keeps the fit math.
- **Shadows** = astrocytes for sensed-but-not-read features; a `shadow` flag in
  the signature registry, not a new gene (already scheduler-invisible).
- **Replay semantics differ and that's fine:** signatures feed the Resolver
  (matched filter), not gradient rehearsal. Zero forgetting is structural (new
  class = new accumulator). The archive is storage + retrieval here; in a mixed
  organism the same astrocyte can serve both readouts.

**[D5] open detail (carried):** astrocyte `finalize` writes count into
`engagement` (`manifold.py:89-99`) — confirm compaction/saliency never reads
astrocyte engagement as a lock signal before reusing it as the period count.

---

## 5. Code placement + spec obligation  **[D6]**

Spec §9 partition map updates *before* code lands (project rule):

- **New package `trioron/pcll/`**, promoted module-by-module as gates pass:
  `receptor.py` (quantise + labeled lines + discreteness census),
  `lockin.py` (deposit + mask rule), `resolve.py` (matched filter + floors),
  `signature.py` (astrocyte-backed registry + fit + shadows),
  `stress.py` (drivers + habituation + vote routing),
  `progenitor.py` (the period-1 perception-generation loop).
- **Spec:** new §10 "Phase-Coherent Lock-in Learning" + §9 partition entry +
  §9.14 index row, signed off before the first promotion commit. RECEPTOR gene
  (bit 11) enters the §2.2 gene table. The genesis replacement is documented in
  §5.7's developmental-program section (the progenitor flow IS a developmental
  program).
- Experiment modules stay put until promotion; runners keep passing against
  both.
- **Envelope:** lock-in tensors + signatures are non-trainable state, outside
  the 50K Phase-1 parameter count (same standing as `engagement` / manifold
  sketches).

---

## 6. Build phases + gates (each falsifiable, runner-verified)

| phase | build | gate (PASS required before next) |
|---|---|---|
| **I1** | RECEPTOR gene + `receptor_levels` + lockin tensors + boundary meeting (resolve + schedule-learn + reset) + `PCLLResolution` adapter | `run_schedule_learning.py` reproduced *through the Substrate forward path*: 3 classes born pure, alignment ≥0.999, zero structural forgetting |
| **I2** | germline instantiation (progenitor + attached council) + period-1 loop (council-judged, progenitor-executed perception/receptor spawn; discreteness census; first-sitting verdict; no genesis) | on the genesis testbed: noise-column receptors retired by habituation, real columns retained — kept-set ≥ genesis.py's kept-set quality, with ZERO gradient probes |
| **I3** | stress → vote economy (perception group key, habituation as retire) | `stress.py` scenarios reproduced on the substrate council: hidden-signal ≤2 attaches; true-void → accepted-empty after exactly 3; ambiguity-recruit → RESOLVED |
| **I4** | signatures as astrocytes + shadows | recruit-without-relearn 0.224→0.377 monotone with arena-stored shadows; ship/wake round-trips signatures |
| **I5** | NN-front-end stress test rerun on the integrated stack, full developmental flow (progenitor → period 1 → council) | ≥ standalone numbers (32 classes, purity ≥0.97); dead-3-sensors grace ≥ raw arm (0.30) |

Mixture-aware birth and period segmentation (handoff NEXT 5–6) stay OUT of this
integration; they slot in after I1 (birth logic) and the boundary machinery
(segmentation) respectively.

---

## 7. Decision register

| # | decision | status |
|---|---|---|
| D1 | receptor = gene (bit 11) on perception cells | recommended; s030 flow consistent with it |
| D2 | new `lockin_re/im/n` tensors (not reuse engagement/utility) | recommended — reuse would corrupt credit-locking |
| D3 | boundary meeting always-on + cheap; `dream_cycle` stays driver-called | recommended |
| D4 | perception vote group = live receptor population, progenitor spawns for it | recommended |
| D5 | signature store = astrocyte (code_dim=2F); verify engagement reuse is safe | recommended, one check open |
| D6 | `trioron/pcll/` + spec §10 + gene-table entry before promotion | recommended (spec-first rule) |
| D7 | discreteness via distinct-value census in period 1, K_DISCRETE=8 | approved s030; built (I2) |
| D8 | sensation side = 4 perception seats held by the progenitor (Σ=28; 4 seats × floor ¼ ⇒ exactly 3 payable votes = the habituation walk) | built (I3); flagged to Rocky |
| D9 | growth decisions auto-SETTLE at the next boundary (success = the stress answered is gone); a decision implies the progenitor acts before the next boundary | built (I3); flagged to Rocky |
| — | genesis batch pre-pass replaced by progenitor period-1 | **DECIDED (Rocky, s030)** |
| — | council votes at period boundaries (not per tick) | **DECIDED (Rocky, s030)** |
| — | tick-2 "bias" = receptor adaptation (range/structure estimate) | **DECIDED (Rocky, s030)** |
