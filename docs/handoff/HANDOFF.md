# Trioron Handoff

**Session date:** 2026-06-11
**Session number:** 029
**Session title:** **The receptor/period/frustration architecture (s028 design) built
END-TO-END and every gate PASSES — feed sweep, trig lock-in, one-cycle resolution,
two-driver stress growth with habituation, and UNSUPERVISED schedule discovery. Stress
test: the discovery loop CONVERGES through a frozen deterministic NN (32/32 classes,
purity ≥0.97) — but a dense front-end forfeits incomplete-input grace. NEXT: integrate
into the trioron substrate proper (Rocky's directive).**

---

## READ THIS FIRST

1. `docs/design/receptor_period_frustration.md` — the s028 design **+ new §11**
   (build-time corrections, all Rocky-approved) and updated §10 (carrier RESOLVED:
   one carrier = the sweep phase).
2. The five module docstrings in `experiments/progenitor/` — `lockin.py`,
   `resolve.py`, `stress.py`, `schedule_learn.py`, `feed.py` — each carries its full
   rationale. They are the spec of record for this layer until promoted.

## The arc (commits, in order)

All on `progenitor-council`, all committed + pushed:

1. **`78b969f`** — feed sweep (class-as-range-schedule, chicken/dog reproduced exactly,
   `test_feed.py`) + trig/lock-in + label-free de-risk **PASS**: coherent margin ~31.6
   vs empty <2.6 at K=3, 5 seeds, zero overlap. Two corrections found in the math and
   verified: quadrature must be zero-mean (cos,sin) — E[sinc]≈0.226 would grow noise
   ∝0.23N; saturated/silent receptors are reference not evidence (pinned-max DC = 1/F,
   measured 0.275). Flat input → deposits nothing → EMPTY (deprivation semantics free).
2. **`7a0d572`** — discrete/binary labeled lines: `theta_discrete` q=1000·(2j+1)/(2k)
   (zero-mean roots-of-unity placement; rejected (j+1)/(k+1) has 1/k DC, measured
   false-coherent at margin 16) + `matched_k` (binary antipodal → 1-D null → K=4).
   De-risk extended to k∈{2,3,5}: all PASS.
3. **`39d8f21`** — items 4+5. `resolve.py`: matched-filter per-class evidence
   E=ΣRe(A·conj(T)), exact Gaussian nulls → parameter-free absolute + top-vs-second
   gap floors; graded ranking output. 200/200 on chicken/dog/mix/noise (pure RESOLVED
   gap≈8, mix FRUSTRATED gap≈0, noise EMPTY). `stress.py`: §7 table routes status →
   driver; conserved vote economy (step4_grow semantics) as driver credibility;
   habituation ×0.5/fruitless, floor 0.2. Scenarios: hidden-signal found ≤2 attaches;
   true-void → accepted-empty after exactly 3; ambiguity → recruit sensed-but-unread
   feature → RESOLVED gap 42.
4. **`e117521`** — **unsupervised schedule learning** (`schedule_learn.py`). A learned
   class IS a running-mean signature (arg=range center, |T|=width) feeding Resolver
   directly. Coherence gate (noise can never birth) → χ² goodness-of-fit → match
   updates running mean / no-fit births a new accumulator (structural zero
   interference). **Fit-null geometry correction:** narrow-arc deposits vary
   TANGENTIALLY → null is χ²(df≈F) not isotropic; first run split classes (5 births).
   Wilson–Hilferty quantile at K_FIT=4. Tests (5 seeds): 3 classes born pure among
   noise, alignment 0.99999, learned-template Resolver 6/6; sequential blocks → births
   at 1/9/17, final mixed block 100% (**zero forgetting, structural**).
5. **`7f2aad9`** — **NN-front-end stress test** (`run_stress_nn.py`) on the
   capacity-hard taxonomy (data_hard: 32c×3m, ~5 disruptors, Bayes 0.937),
   class-periods S=500, genesis → discovery → eval. **Rocky's question answered: YES,
   converges through a frozen random 12→24→16 tanh MLP** — 32 classes, 27/27 clean
   coverage, purity 0.97–1.00, all seeds, ≈ raw-features control.

## Stress-test findings (the numbers)

| arm | converged | classes | purity | 1-shot | dead-3-sensors |
|---|---|---|---|---|---|
| raw ×3 seeds | yes | 32–34 | 1.000 | 0.38–0.41 | **0.30–0.31** |
| nn ×3 seeds | yes | 32 | 0.97–1.00 | 0.38–0.40 | **0.10** |
| nn-mode ×2 | yes | **96** | ~1.0 | **0.61–0.62** | 0.12–0.15 |

- **Incomplete input:** raw/partitioned degrades gracefully (dead channel = masked =
  no evidence); dense NN collapses (mixing turns absent evidence into corrupted
  evidence everywhere). Rocky's genesis-partitioning argument, quantified → if a
  perception stack sits upstream, PARTITION it (per-sensor-group subnets).
- **Recruit-on-ambiguity:** discovery reading top-6 units (30 classes, purity 0.918,
  1-shot 0.224) → re-add dropped units one at a time, templates extended from SHADOW
  signatures (sensed-but-not-read) → monotone 0.224→0.377 = exactly the full-16
  reading, **zero relearning**.
- **Disruptors are NOT empty unsupervised** — each born as its own wide (fit-tolerant)
  class. Differs from the supervised frustration-gate retire; arguably more correct.
- **Unimodal templates blur multimodal classes** (1-shot 0.38 class-level vs 0.62
  mode-level) → mixture-aware birth is the next mechanism lever.

## Honest caveats (Rocky: "too amazing")

The conditions are friendly and the purity-1.0 numbers are real but PERIOD-LEVEL:
- A period = 500 observations of ONE class with the boundary GIVEN. Period
  segmentation (unknown boundaries, drifting/mixed streams) is unbuilt and is where
  this gets hard.
- 1-shot numbers (0.38/0.62) are NOT comparable to the supervised council 0.775 —
  unsupervised templates, single observation, matched filter only.
- The fit/floor statistics assume iid draws within a period; real streams correlate.

## Decisions made (sign-off Rocky)

- Carrier = the sweep phase, single (design §10 resolved).
- Quadrature = zero-mean (cos,sin); gcos/sinc/tan remain post-perception units.
- q=0/1000 are reference, not evidence (mask rule).
- Discrete features bypass the receptor; midpoint labeled lines; matched K (4 binary).
- Votes = driver credibility/within-group prioritisation (status exclusivity means
  cross-driver arbitration never arises per period); §7 table routes directly.
- Habituation constants 0.5/0.2 (≈NOGAIN_PATIENCE shape; resolution itself stays
  parameter-free).

## Open questions / NEXT (priority order)

**Rocky's directive: integrate this into triorons.** The standalone layer works; it
must become substrate machinery. Sketch (discuss before building):

1. **Receptor + trig units → perception-side phenotypes** (epigenome genes; receptor
   is per-sample/stateless, trig bank reads its phase). Genesis partitioning maps onto
   the existing `genesis.py` apoptosis→saliency stages.
2. **Lock-in accumulator → per-cell state, period boundary → end_task/dream boundary**
   (the design's circadian point: period end = consolidation point). Resolution
   margins replace/feed `FrustrationDetector` for the progenitor loops.
3. **Stress drivers → council vote economy** (`step4_grow.py`) — empty-stress biases
   perception spawns, ambiguity-stress composer spawns; habituation as retire pattern.
4. **Learned signatures ↔ manifold archive** — a signature is a tiny per-class sketch
   (complex F-vector ≈ astrocyte material); shadow accumulators = recruit-without-
   relearn. Natural fit with ManifoldArchive/astrocytes (spec §4.5, §2.11).
5. **Mixture-aware birth** (split a class when its own deposits stop fitting) — the
   1-shot 0.38→0.62 lever.
6. **Period segmentation** — unknown boundaries; likely the real frontier.
7. **Componential-semantics output** (Rocky named the term) — the readable graded output
   is a SIGNED relational descriptor ("goat-ish chicken, not pig"), not just the unsigned
   typicality ranking `resolve.py` has. Componential semantics = ± distinctive features
   vs neighbours (Saussurean valeur), computed as **vector arithmetic over signature
   space** (T_chicken − T_duck = discriminating direction). Signatures are already
   vectors → small build on `resolve.py`. See design §10. Context: per-species one-shot
   on the 10-animal set (`run_taxonomy10_oneshot.py`, commit after `595cda8`) is 0.82
   overall; the two failures (dog disruptor 0.12, chicken≈duck 0.32) show top-two margins
   +0.03/+0.07 = the FRUSTRATED band — argmax scoring forces a one-hot where the system
   honestly says "ambiguous, grow a discriminator", so per-species argmax UNDERSTATES it.

## State of the build

- Branch `progenitor-council`, all work committed AND pushed through `7f2aad9` + this
  handoff. **DO-NOT-COMMIT carries (still excluded, verified):**
  `trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
  `trioron/viz/export.py`; `.claude/`, `runs/` untracked.
- New live modules (`experiments/progenitor/`): `feed.py` (+`test_feed.py`),
  `lockin.py`, `resolve.py`, `stress.py`, `schedule_learn.py`; runners
  `run_lockin_derisk.py`, `run_resolution.py`, `run_stress_growth.py`,
  `run_schedule_learning.py`, `run_stress_nn.py`. All runners print PASS and run in
  seconds on CPU (`python3`, from `experiments/progenitor/`).
- The s028 branch/GCU growth path remains superseded; do not reinvest.

## Pointers

- Design + corrections: `docs/design/receptor_period_frustration.md` (§11 new).
- The s028 diagnostics (GCU detonation, capacity wall, 96-modes): unchanged, see
  s028 handoff in git history (`fe8a3fa`).
- Spec §4.2 FrustrationDetector — the integration point for resolution margins.
- Vote economy semantics: `step4_grow.py` (`_vote_transfer`, mirrored in `stress.py`).

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`. Python 3.10.12
  (NB: no nested same-quote f-strings), torch 2.11.0, WSL2, 12 cores, **`python3`**,
  `OMP_NUM_THREADS=8`. All new runners are seconds-fast; no backgrounding needed.
- Session was build-heavy but everything is committed, pushed, and green; safe break.
