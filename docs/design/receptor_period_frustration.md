# PCLL — Phase-Coherent Lock-in Learning (Trioron)

> **The method is named PCLL: Phase-Coherent Lock-in Learning by Trioron**
> (Rocky, s029). Magnitude is **phase-coded** into 1000 quanta, swept over a
> period, and **coherently integrated** (lock-in detection) so a real class grows
> ∝N while empty noise random-walks ∝√N; the per-class readout is a **matched
> filter**. This is the streaming, single-pass, label-free alternative to
> gradient-trained hidden depth. (File name kept as-is for git history; this
> document IS the PCLL spec.)
>
> Design captured from the s028 session (2026-06-11). This is the **forward
> direction** — it supersedes the branch/GCU growth experiments (which hit a hard
> wall: capacity-limited *and* numerically unstable; see §1). Almost all of this
> lives only in the s028 conversation; this doc is the durable record. Read it in
> full before building. Co-designed by Rocky; Chloe wrote it up.

---

## 0. One paragraph

Replace gradient-trained hidden depth with a **biologically-paced perceptual loop**.
A continuous input is reduced by an **adaptive receptor** to a small set of discrete
channels, each quantised per-sample into **1000 phase pockets**. Data is fed as a
**sweep over one period** (1000 quanta = one circadian-like cycle). Trig units
(**gcos / sinc / tan**) placed *right after perception* (never stacked in depth)
do **phase-coherent lock-in** over the period: a real class accumulates amplitude
∝N, noise only ∝√N. The **period boundary is the frustration trigger** — one cycle is
the whole learning budget. Output is **graded/relative** ("more likely chicken than
dog"); resolution is **margin over the √N noise floor** (parameter-free). A first-class
**empty class** = the noise floor itself: a valid terminal answer *and* a stress signal
that drives growth (sensory deprivation is aversive). Growth has two drivers —
**empty-stress → grow sensation (receptors)**, **ambiguity-stress → grow discrimination
(composers)** — routed by the conserved council vote economy, and **bounded by
habituation** (empty-stress decays on fruitless growth; the parameter envelope = the
"skull" is the hard backstop, left uncapped for the trial because this data converges).

---

## 1. Why this redesign — the wall the branch/GCU experiments hit

The s027→s028 branch architecture (`step5_branch.py`) grew thin columns indexed by
`branch_id`. This session diagnosed it to a dead end on two independent axes:

- **Capacity-limited.** Grow → freeze → long-train (6000 steps) asymptotes at **0.569**
  (Bayes 0.937). More epochs make it *worse*, not better (`run_capacity_check.py`). The
  thin single-projection columns cannot represent the target regardless of training time.
- **Numerically unstable (the GCU detonation).** `σ(z)=z·cos z` is **expansive for
  |z|>1** (the z² envelope). A deep stack compounds magnitude with depth
  (`run_gcu_diag.py`: deepest column depth 10; L3 activation 216 → L4 622 → **119,400 by
  step 750** → acc collapses to chance 0.085 → **permanently dead**, frozen identically
  at steps 1000/3000/6000). Crucially **the weights stay ~5** the whole time — this is
  *forward activation amplification*, NOT weight blow-up, so **gradient clipping does not
  fix it**. The fix is bounding z (RMS-norm the pre-activation) and/or not stacking the
  oscillator in depth at all.
- **No rollback.** `grow_class_driven`/`grow_council_frustration` commit a spawned cell,
  train, measure Δ — and on a *negative* Δ keep the cell anyway (only a per-class no-gain
  counter affects future *selection*). So proven-regressional deep GCU cells accumulate.
  This is philosophy-A ("extra cells are harmless") falsified yet again.

Supporting facts established this session:
- Per-class residual used by the loops is **mean cross-entropy ∈ [0,∞)**, NOT 0–1.
  Anchors: 0 = perfect, **log(C)=3.466** (C=32) = chance = where an untrained zero-logit
  class sits, >log(C) = worse than chance. This is why `mean+k·std` is baseless and a
  **chance-anchored** signal `f_c = CE_c/log(C)` is the right normalisation.
- The capacity-hard taxonomy is **96 modes, not 32 classes** (32 classes × 3 Gaussian
  modes + 5 wide disruptors). k-means k=32 → ARI 0.204; **hierarchical pure clusters cap
  at 27/32 classes** — the 5 disruptors form **zero** pure clusters at any resolution.
  Decomposition: k-means k=96 purity 0.798, GMM-96 (diag cov) 0.870 ≈ Bayes 0.938. The
  32-way label is **not** recoverable from feature geometry — supervised labels carry
  information clustering cannot. (`kmeans_probe.py`.)
- **Bias redundancy:** a spawned *linear* cell's bias is mathematically aliased by the
  output bias (`b_o + Σ w_ov·b_v` is a reparametrised output bias). A linear cell earns
  value only through its *weights*. A *dendrite* (z+z²) bias is NOT fully absorbable (it
  sits outside the nonlinearity).
- **Zero-init rule:** safe for **output** edges (driven directly by the class-label
  gradient), **fatal for hidden** cells if *both* sides are zero (∂L/∂a=0 → ∂L/∂w_in=0 →
  stillborn). Keep hidden spawns asymmetric-nonzero (perception side seeded); outputs zero.
- We **do** train before measuring impact (spawn → `train(TRAIN_PER)` → measure), so Δ
  reflects the trained cell — but only because edges init at N(0,0.01²), not 0.

---

## 2. The adaptive receptor  (BUILT — `experiments/progenitor/receptor.py`)

Each sample self-normalises to its **own** range and drops into one of **1000 discrete
pockets** — gain-adaptive, history-independent, like a sensory receptor adapting to the
current stimulus (the eye encodes contrast, not absolute luminance; few discrete
channels for taste/skin).

```
lo = min(0, sample.min())      # floor at true-zero, OR a negative value becomes the new min
hi = sample.max()              # the peak feature saturates to 1000
q  = round(1000 · (v - lo) / (hi - lo))     # partition in [0, 1000]
θ  = 2π · q / 1000                          # phase in [0, 2π]   (2π/1000 per quantum)
```

- **Per-sample, history-free.** `max` is the current sample's max across its features;
  the past max is irrelevant. So it is **scale-invariant**: `[5,10,15,20,25]`,
  `[10,20,30,40,50]`, `[1,2,3,4,5]` all encode IDENTICALLY to `q=[200,400,600,800,1000]`
  — absolute magnitude discarded, only the within-sample pattern survives.
- **Negatives → the new MIN.** Zero is the floor for non-negative data; a negative value
  extends the floor down and becomes partition 0 (`[-5,0,10,25]→[0,167,500,1000]`).
- **The partition IS the phase** ("signal × radian"): a feature's magnitude becomes its
  angle in the period. q=1000 → 2π = one full period = the "covered one period, called
  off" point.
- Verified against Rocky's worked examples (all pass).

---

## 3. The feed / sweep model  (verified, not yet a module)

A **class = a per-feature range *schedule* across the period**, NOT a vector. Example
(quantum units, "ext" = the 0/1000 extremes feature, shared/non-discriminative):

```
chicken = {f0:1–5,  f1:0/1000, f2:80–100, f3:30–40}
dog     = {f0:6–10, f1:0/1000, f2:30–50,  f3:35–80}
```

Data is fed as a **sweep over quantum value v = 0 → 1000** (one period). At each `v`,
every feature **slices** for whichever class's range contains `v`; if none → **empty**.
The class is read from *which features slice at each quantum*. A class's features fire at
*different* v's, so the signature is a **temporal pattern over the period**. Verified to
reproduce Rocky's spec exactly:

```
v=0:  3 empty, 1 chicken, 1 dog   (f1 'ext' contains 0 → votes both; f0/f2/f3 empty)
v=1:  3 empty, 1 chicken          (f0∈[1,5] → chicken; rest empty)
v=8:  3 empty, 1 dog              (f0∈[6,10] → dog)
v=35: 2 empty, 2 dog, 1 chicken   (dog f2∈[30,50] & f3∈[35,80]; chicken f3∈[30,40])
v=90: 3 empty, 1 chicken          (f2∈[80,100] → chicken)
```

The ranges are **learned/normalised over runs** ("after Nth run the chicken class has its
features normalised to …") — the organism discovers each class's feature-range schedule
unsupervised; labels are unknown.

---

## 4. Trig units + lock-in  (TO BUILD)

- **gcos / sinc / tan**, one triplet **per feature** (Rocky: per-feature 1:1; 4 features →
  12 trig nodes), placed **right after perception, NOT stacked in hidden depth** — this is
  the GCU-detonation fix (no deep oscillator chain to compound).
- They read the receptor **phase** θ (the "signal × radian"). gcos = z·cos z, sinc =
  sin z/z, tan = z; sin & cos give the quadrature/phase, tan flags the phase crossing.
- **Lock-in over one period:** integrate each receptor against the carrier across the 1000
  quanta. A **coherent** (real, recurring) signal accumulates in phase → amplitude ∝ **N**;
  an **empty/incoherent** one is a random walk → ∝ **√N**.
- **Phase is integrated over the stream** (Rocky's pick): a recurrent/phase accumulator,
  internal time — NOT an external clock.

**Dendrite restored to `σ(z)=z+z²`** (`trioron/phenotype/dendrite.py`, reverted from the
s027 global GCU swap). The trig oscillators replace the GCU's role, at the input only.

---

## 5. One-cycle frustration + the √N noise floor

- **One period = the entire learning budget.** Same council frustration trigger, but
  repetition limited to **1 cycle**. Single-pass learning → light. If a cycle doesn't
  resolve → frustration → grow. The period boundary is also the natural **dream /
  consolidation** point (circadian = sleep).
- **Output is graded/relative** — "more likely chicken than dog", a ranking over
  accumulated per-class evidence, not a one-hot. Overlap (e.g. v=35 voting both) becomes
  *measured shared evidence that lowers the margin*, which is honest, not an error.
- **Resolution = margin over the √N noise floor** (parameter-free; Rocky agreed). After
  N=1000 cycles a real class sits ~1000, noise ~√1000≈32. Two tests:
  - **vs empty (absolute):** does any class clear the √N floor?
  - **vs each other (relative margin):** is the top class clearly ahead of the second?
  - Top clearly leads → **resolved**. Ambiguous (top-two in the noise band) →
    **frustrated** → grow. This is the *basis* `mean+k·std` never had.

---

## 6. The empty class — terminal answer AND stress driver

The **empty class** is new (absent from all prior designs, which force-fit every input
into one of N classes — that is literally what the 5 disruptors were: incoherent inputs
misclassified). It is the **noise floor itself** — the null hypothesis, **not a learned
prototype** (so it stays parameter-free).

**Dual nature (the keystone):** empty is a *valid terminal output* AND a *stress signal*.
"Humans feel stress if they lose their senses" — sensory deprivation is aversive, so
empty *drives growth*. This dissolves the "true-void vs missed-detection" problem we
couldn't solve in one cycle: empty always registers as stress → the organism grows its
senses → growth either **finds a hidden signal** (it was a missed detection; stress
relieved) or **doesn't** (settles toward accepted-empty). The distinction is made *over*
cycles, by the growth, not up front.

---

## 7. Two growth drivers, routed by the vote economy

Growth now has two complementary drivers, wanting different responses:

| stress | meaning | grow what |
|---|---|---|
| **empty-stress** | nothing clears the floor | **sensation** — new receptor / different carrier frequency (reach toward the world) |
| **ambiguity-stress** | signal clears the floor but classes won't separate | **discrimination** — composing triorons that sharpen the margin |

Both route through the **conserved council vote economy** (built this session,
`step4_grow.py`): empty-stress biases the vote toward **perception/receptors**,
ambiguity-stress toward **composing capacity**. Same machinery, driven by *which* stress
dominates.

---

## 8. Bounding — habituation + the skull

- **Habituation tames the void** (Rocky's call): empty-stress **decays with each fruitless
  sensory growth and resets when growth finds signal**. Acute void → grow receptor → still
  empty → drive decays → after a few fruitless expansions it settles into accepted-empty;
  the moment a growth lifts something above the floor, the stress was justified →
  re-sensitise. This is the **same `NOGAIN_PATIENCE` retire-on-repeated-failure pattern**
  already built in `grow_class_driven`, pointed at empty-driven sensory growth.
- **The skull = the parameter envelope** — the hard backstop, always there conceptually,
  but **left uncapped for the trial** because this data type converges on its own.

---

## 9. What is built vs to-build

**Built this session (committed):**
- `receptor.py` — adaptive per-sample contrast quantiser (§2). Verified.
- `frustration_gate.py` — chance-anchored gate `frustrated = plateaued AND CE/log(C)>τ`
  (the pre-receptor frustration model; superseded in spirit by §5's √N margin but the
  chance-anchor reasoning carries over). τ=0.5→0.582, τ=0.25→0.707 on the hard taxonomy;
  auto-retired the 5 disruptors as irreducible.
- `step5_branch.py` — asymmetric fan fix (`_frustrated_perc` sparse input + wide axial
  fan) + `grow_class_driven` (class-driven growth, 0.762 with distributed depth).
- `step4_grow.py` — conserved council **vote economy** (`_vote_transfer`, per-cell votes
  Σ=20, ±1/event mirror, `GROUP_VOTE_FLOOR=1.0`, plateau-patience stop, no Bayes).
- `dendrite.py` — restored to `z+z²`.
- Diagnostics: `audit_branch.py`, `run_capacity_check.py`, `run_gcu_diag.py`,
  `kmeans_probe.py`, `run_class.py`.

**To build (next session), in order:**
1. **Feed module** — class-as-range-schedule generator + the sweep (§3), reproducing the
   chicken/dog example; start with a 2-class / 4-feature toy.
2. **Trig units + lock-in accumulator** (§4) — per-feature gcos/sinc/tan on the receptor
   phase, phase integrated over the stream, amplitude per class per period.
3. **One-cycle resolution** (§5) — margin-over-√N-floor; graded output; empty = below floor.
4. **Empty-stress + two-driver growth + habituation** (§6–8) — wire into the vote economy.
5. **Then** revisit the supervised hard-taxonomy only as a stress-test of the new loop.

**Smallest first test (de-risk before the full organism):** a stream where some windows
carry a coherent signal at the carrier and others are empty noise; receptor → phase →
lock-in over one period; show coherent windows clear the √N floor and empty ones don't —
**without labels**. If that separation holds, the core mechanic is real.

---

## 10. Open questions for the build

- **Lock-in carrier:** RESOLVED (Rocky, s029) — ONE carrier, the sweep phase itself;
  class identity emerges from which quanta features slice, not per-class carriers.
- **Frustration-on-empty residual:** habituation handles bounding, but do we also test the
  *residual* for leftover structure to prioritise *where* to grow a receptor? (Deferred —
  habituation may make it unnecessary.)
- **GCU/RMS-norm:** if any oscillator ever does sit in depth, RMS-norm its pre-activation
  (grad clipping won't help — the blow-up is forward, weights stay small).
- **Mixture semantics (s029):** a RECURRING mixture stream fails fit vs both constituents
  and is born as its own class. Legitimate (a recurring pattern is discoverable) or to be
  decomposed? Related: unimodal templates blur multimodal classes (1-shot 0.38 vs 0.62
  with per-mode periods) — mixture-aware birth (split a class when its own deposits stop
  fitting) is the proposed mechanism.
- **One-sided feature activity:** a feature coherent in the stream but unknown to a class
  (or vice versa) should auto-reject in the fit; not yet handled.
- **Componential-semantics output (s029, Rocky named it):** the graded output should be
  not just a ranking (unsigned typicality, "duck-ish, almost chicken") but a *signed,
  relational* descriptor against neighbours — "goat-ish chicken, not pig". The term is
  **componential semantics**: a class is described by ± distinctive features relative to
  its neighbours (Saussurean *valeur* / componential analysis), computed as **vector
  arithmetic over the signature space** (T_chicken − T_duck = the discriminating
  direction; "−goat", "not pig" = signed projections onto neighbour differences). The
  raw material already exists — learned signatures are vectors and `resolve.py` has the
  ranking; what's unbuilt is the signed neighbour-difference readout. UNBUILT.

---

## 11. Build-time corrections (s029 — all verified, see module docstrings)

The architecture was built end-to-end in s029 (`feed.py`, `lockin.py`, `resolve.py`,
`stress.py`, `schedule_learn.py` + runners, all in `experiments/progenitor/`). Four
corrections to this design surfaced from the math/experiments — Rocky signed off:

1. **Quadrature = (cos θ, sin θ), NOT (gcos, sinc)** — the √N floor needs zero-mean
   carriers; E[sinc] = Si(2π)/2π ≈ 0.226 would grow noise at 0.23·N. gcos/sinc/tan
   stay as the post-perception units; the accumulator integrates the zero-mean pair.
2. **Saturated (q=1000) and silent (q=0) receptors are reference, not evidence** — the
   pinned max is a 1/F DC that makes uniform noise read coherent (measured 0.275 ≈ 1/4).
   Bonus: flat input deposits nothing → reads EMPTY (the §6 deprivation semantics free).
3. **Discrete/binary features bypass the receptor** (no gain to adapt) and sit at
   labeled lines q = 1000·(2j+1)/(2k) — zero-mean over levels so the floor holds;
   binary's antipodal deposits give a 1-D null → matched threshold K=4 (else 3).
4. **The fit-test null is tangential (χ², df≈active features), not isotropic** — narrow-
   arc deposits vary along the arc; the isotropic null split true classes. Criterion via
   Wilson–Hilferty quantile at K_FIT=4.

Headline validations: label-free coherent-vs-empty separation PASS (margins ~31.6 vs
<2.6); resolution trichotomy 200/200; stress scenarios (hidden-signal / true-void /
ambiguity-recruit) all PASS; unsupervised discovery 3/3 classes pure + zero structural
forgetting; **the loop CONVERGES through a frozen deterministic NN** (32/32 classes,
purity ≥0.97, all seeds) — but a dense front-end forfeits incomplete-input grace
(dead-3-sensors: raw 0.40→0.30 vs NN 0.38→0.10) → partition the perception stack.
