# Trioron Handoff

**Session date:** 2026-08-23
**Session number:** 062
**Session title (s062, Rocky: "not not not", then "wrap up all and produce a stable API"):**
**TRIPLE NEGATION GENERALISES ZERO-SHOT IN SEQ MODE. Logophosis `seq` (one token per step, scene
as per-step context, NOT = same weights at every depth) trained on 0–2 NOTs: not³ .873±.052,
not⁴ .688±.118 (n=3, 6.6K params) vs flattened `tied` .550±.026 / .521±.052 (in-dist .838, comp .773, 20.7K params). STABLE ORGANISM API
SHIPPED: `trioron/organisms/` (Logophosis + Kinopsis, `API_VERSION 1.0`, spec §9.16, 5 tests).**

## READ THIS FIRST
- The organisms are PACKAGE classes now: `from trioron.organisms import Logophosis, Kinopsis`.
  `experiments/progenitor/{logophosis,kinopsis}.py` are world-side wrappers (data, splits,
  scoring). Edit the package class, never the runner, for organism behaviour.
- Seq mode has TWO init gotchas (below). Don't re-add RMS norm; it hurts.
- OOM rule tightened: a Link-0 build (2.8–2.9 GB) OOMs if ANY other 40K language process is
  training at the same time (3 kills this session, all the `tied` arm). Run one language process
  at a time, or gate launches on the other's "woke in" line. `ChordLink0.evidence` is now chunked
  (4000 sentences) — the whole-set chord tensor was part of the peak.

## Not-not-not experiment (this session)
World: `language_world.py` `NOT_ONLY=1` — depth d = EXACTLY d NOTs (no and/or), so nesting
depth = NOT count. Runner `logophosis.py` with `DEPTH_MIX=1` (train uniform over depths 0..2,
40K), `TEST_DEPTHS=1,2` (zero-shot 3 and 4 NOTs, 4K each), MAX_OBJ=2, L_MAX=10, comp split as
rung 1. Budget matched across arms: 10 ep/stage, patience 6, MAX_LINKS 2 (seq: 1 thinking step).
Logs `outputs/logophosis_s062_notnotnot_seq_s{0,1,2}.log`, `..._tied.log`.

| arm | in-dist | comp | **not³ (zero-shot)** | **not⁴** | params |
|---|---|---|---|---|---|
| **seq** n=3 | .941±.014 | .927±.015 | **.873±.052** (.825/.865/.929) | .688±.118 (.591/.654/.820) | 6,602 |
| tied (flattened) n=3 | .838±.009 | .773±.012 | .550±.026 (.549/.525/.577) | .521±.052 (.580/.492/.489) | 20,714 |

Readings:
1. **The involution is learned as a primitive only when re-application depth = nesting depth**
   (Rocky's s061 diagnosis confirmed on the real language world, not just `not_k`). Flattened
   reading stays at chance on unseen depth as in s061; seq carries parity one depth beyond
   training on 3/3 seeds and two depths on 2/3 (seed 2: .82).
2. Seq is also BETTER in-distribution (.94 vs .84) at 1/3 the params — the flattened chain's
   .84 shows it can't even fit 0–2 NOTs at this budget.
3. Decay with depth (not⁴ < not³) = the state drifts per application; Rocky's crisp-state
   hypothesis again. NOT RMS norm (tested, harmful). Untested: tanh/clip squash between steps,
   the gated snap, Phasecyte-between-steps.

### Seq-mode gotchas (s062 probe, `scratchpad/probe_seq.py`, oracle L0, 6K, 6 ep)
| link init | state norm | acc | loss |
|---|---|---|---|
| std .01 (default) | none | .503 | .693 (ln 2 stall) |
| std .01 | rms | .500 | .694 |
| **fan-in** | **none** | **.609** | **.631** |
| fan-in | rms | .497 | .694 |
(i) 10 std-.01 applications in series vanish the signal → `LinkChain(link_fan_in=True)` (the head
already had it since s061); `Logophosis` defaults it on for seq. (ii) RMS norm rescales the
init noise to unit and kills it. (iii) Quad σ(z)=z+z² over 11 applications BLOWS UP the train
loss at stage 2 (1.4e3 / 1.5e5 / 1.4e2 across seeds) while test acc still rises (.92→.95) —
the optimiser survives via grad-clip; this is the s061 "loss exploded" fragility, untreated.
The first seq launch (rms, 8 ep) = `..._seq_s{0,1}_rmsnorm.log`, chance everywhere.

## Stable organism API (commit 6fd4d9e + 1e7c965)
`trioron/organisms/base.py`: `Organism` (`save(path)->bytes`, `load` with type check,
`n_params`, `substrates`, `torch_modules`), helpers `leaf`, `train_leaf`, `chunked`, `Std`.
`trioron/organisms/logophosis.py`: `ChordLink0` (PhasecyteLeaf on chord tokens; `ev_mode`
plain|centered|softmax), `OracleLink0`, `Logophosis(scene_dim, class_cap, seed, mode=seq|tied|grown,
link0=phasecyte|oracle, max_links, H, interior, seq_norm, link_fan_in, link0_kw)` →
`.wake(ids) .fit(S,ids,y,S_te,ids_te,y_te, epochs, patience, target) .comprehend(S,ids)
.accuracy .depth`. Token length L is fixed at first encode.
`trioron/organisms/kinopsis.py`: `Kinopsis(seed, leaves, std, empty_mshape)` /
`.from_organ(organ, empty_mshape, seed)` → `.settle_heads .acquire_msil .fit_router
.acquire_vel` (chainable) → `.shape(S, E, mode) .velocity(Zm)`; `Router`, `QDAVoter`, `conf`,
`vote` live here (`motion_router.py`/`motion_leaf.py` re-export). `settle_heads` rank clipped
to stream dim (was NaN for d ≤ 32). `trioron/senses/chord.py`: `chord_table(n, bins, k, seed)`,
`chords_of(ids, table, gen, jitter)` — bit-identical to the s060 tables (verified).
`LinkChain` new kwargs: `ctx` (per-step context dims, seq), `norm` (none|rms), `link_fan_in`.
Spec §9.16 + top-level tree updated. Tests `tests/test_v2/test_organisms.py` (5, ~70 s).
Pre-existing failures NOT touched: `test_lifecycle.py::TestDreamCycle` ×2 (fail on the parent
commit too).
Caveats: Kinopsis pickles from before s062 won't load (class path moved); Logophosis save is
~42 MB (Phasecyte Link-0 capacity-4096 substrate) — Link-0 compaction/dream_distill is open.
`logic_chain.py` `pure_seq` arm untouched (no ctx, no fan-in) — its s061 numbers stand.

## NEXT (s063)
1. **not⁴ and beyond:** state squash between seq steps (tanh or clip — NOT rms), re-run n=3;
   also the loss blow-up should go away with it. Then depth+3 (not⁵).
2. Rung 1 FULL world (and/or + not) in seq mode — does seq keep the .93/.90 of the 40K tied run
   and add `not not`? Expect yes; this replaces the s061 headline table.
3. Mode E transfer: lock the NOT-only seq link, compose into the full world.
4. Kinopsis end-to-end regression through the package class (`kinopsis.py` runner, n=1) — the
   refactor is covered by synthetic tests only; the real-world numbers (s057) not re-verified.
5. Link-0 compaction for `Logophosis.save`; MODE=continual for Logophosis (port `run_continual`).

## Gotchas (cumulative, new first)
- `pgrep -f` is fine to LIST; `pkill -f` kills this shell. Kill by PID via `/proc/PID/environ`.
- One 40K language process at a time (see READ THIS FIRST). Seq costs ~10× tied per epoch
  (L=10 token steps): ~25 min/stage at 40K with 4 threads.
- `git stash` while background runs are live is safe (they've imported already) but don't.

## State of the build / Pointers
- Commits on `conscience-core`: 6fd4d9e (API), 1e7c965 (seq fix + seq logs), + final handoff commit.
  `main` NOT advanced. Pushed at session end.
- Memory: `logophosis_seq_triple_negation.md`, `logophosis_organism_api.md`.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs / uncommitted
logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM (Rocky is
  considering a new laptop — RAM is the binding constraint, not CPU).

---
Below = the s061 handoff body, still current for rung-1 (impure/pure flattened) tables, Link-0
findings and the logic-chain depth gate.

---
## What was built (commit a35b3d1 + this handoff)
- `trioron/lifecycle/chain.py` (package, spec §9.6 row added): `new_link` (Seeded quad, optional
  `fan_in_init`), `LinkChain(mode=tied|grown)` with a SUBSTRATE head link `Seeded(H→C, interior
  16)`, `StallTrigger` (Numa patience rule, epoch-grained), `fit_stage`, `fit_grow` (stall → grow →
  joint train, lock-after-settle; no credit lock at spawn).
- `experiments/progenitor/language_world.py`: `chord_table`, `chords_of`, `ids_of` (env CHORD_BINS 32,
  CHORD_K 3, CHORD_JITTER .05).
- `experiments/progenitor/logophosis.py`: `ChordLink0` = PhasecyteLeaf over the token chord
  stream (WAKE=20000 tokens, WINDOW=1000, class_cap 32, capacity 4096; labels `c{wid}` are naming
  taps only); evidence = `leaf.evidence()` CENTERED branch → softmax τ=1 (`EV_MODE`, `EV_TAU`) →
  zero-padded to CLASS_CAP, PAD rows zero. Arms `tied,grown`; env as language_chain.py + LINK0/WAKE/
  WINDOW/EV_MODE.
- Logs: `outputs/language_s060_rung1_s{0,1,2}.log` (impure refs), `..._pure_s{0,1,2}.log`,
  `..._pure_oracle_s0.log`.

## Rung 1 (DTRAIN=1, MAX_OBJ=2, L_MAX=10, N=20K, 20 ep/stage, MAX_LINKS 4, patience 8)
| arm | in-dist | comp | depth+1 (`not not`) | params |
|---|---|---|---|---|
| **pure tied** (Phasecyte L0) n=3 | **.868±.010** | .792±.012 | .458±.030 | 20.7K |
| pure tied, oracle L0 (n=1, s0) | .858 | .780 | .588 | 20.7K |
| pure grown (fresh links, joint) n=3 | .848±.018 | .776±.017 | .474±.029 | 81K |
| impure tied (torch head, oracle L0) n=3 | .846±.013 | .780±.011 | .552±.007 | ~16K |
| impure grown_joint (n=2; seed 2 OOM-killed at final eval, in-dist .822) | .837±.002 | .773±.006 | .538±.006 | ~64K |
| mlp3 | .855±.010 | .792±.006 | .585±.013 | 19K |
| gru | .866±.011 | .804±.008 | .479±.038 | 36K |
| bow | .704±.008 | .720±.003 | .341±.012 | — |
Tied traces (pure, seeds≈): R=1 .84 → R=2 .86 → R=3 .87 → R=4 .87; train loss .06 at R=4 with test
.87 ⇒ GENERALISATION-limited at 20K (the hop-3 data-wall shape; N=40K is the obvious next lever).

## Findings
1. **Substrate head > torch head.** Pure chain + oracle L0 at R=1 = .849 vs .718 with the torch
   Linear head; the quad head link is load-bearing. BUT two std-.01 substrates in series emit
   ~1e-5 logits and sit at ln 2 (smoke) — fixed by fan-in-scaled edge init on the HEAD only
   (`fan_in_init=True`, links unchanged for rung-0 comparability). Substrate output std .002→1.4.
2. **Phasecyte symbols ≈ oracle symbols** once read out correctly: raw matched-filter evidence is
   dense (13–30 on EVERY class, argmax word-pure) and the chain can't use it (R=2 .726); the
   CENTERED evidence (s039 common-mode fix) + softmax gives .993 mean top-1 (crisp symbol, genuinely
   ambiguous tokens stay soft = doubt carried, Rocky's framing) → pure tied .868 ≥ oracle-pure .858.
3. Link-0 discovery needs exposure: WAKE=3K tokens → 4 classes; 20K → 24–26 classes, 21–22/24 named,
   purity .996 (3 words merge into neighbours; chord max cosine .78 between some word pairs).
4. **Double negation is NOT discovered zero-shot at rung 1.** Pure tied .458 / GRU .479 are
   below chance = systematic "not not X ⇒ not X"; BoW .34 shows the label-flip structure. The
   depth+1 split is exactly the test Rocky asked for (DOUBLE_NOT allowed) — it is an open rung,
   not a bug. Candidate levers (untested): (a) more data (40K) since train loss ≪ test; (b)
   train on ≤1 NOT but with explicit `not` as a sequential token read (item B) so re-application
   depth maps onto nesting depth; (c) rung 1b: include `not not` in training at MAX_OBJ=2 and
   test depth+2 (triple) for the generalisation claim.

## Gotchas (new)
- OOM: 7 GB RAM. Link-0 build peaks ~2.1 GB (evidence over 200K tokens); 4 language procs +
  inspection scripts tripped the OOM killer (took the seed-2 impure ref at its final eval). Run
  ≤3 language processes at once.
- Load: the impure refs (3×4 threads) + probes pushed load to 20/12 CPUs; every stage ~25 min
  under that. Sequence runs instead of stacking them.
- `pgrep -f`/`pkill -f <pattern>` kills the assistant's own shell (exit 144) — hit AGAIN. Kill by
  PID: `ps -eo pid,args | grep "^ *[0-9]* python3 <script>"` then check `/proc/PID/environ`.
- `logophosis.py` std warnings for n=1 are harmless (`std()` on a single seed).

## NEXT (s062)
A. (done) pure grown n=3 in table: re-application (tied, 20.7K) beats fresh links (81K) on every
   axis again, as at rung 0. Rerun impure grown_joint seed 2 only if its comp/depth are needed.
B. Rung 1 double-negation: run the levers in Finding 4 (start with N=40K pure tied, one seed, to
   see whether depth+1 moves at all); then rung 2 (MAX_OBJ 3–4 + relations) with refs at a
   learnable scale first.
C. MODE=continual (3 vocab stages, forgetting matrix) on the pure organism — `logophosis.py`
   has no continual main yet; port `run_continual` from `language_chain.py`.
D. Sequential token reading (Axis-7 style, 32-d/step) to replace the flattened L_MAX×32 input;
   pairs naturally with nesting depth = re-application depth.
E. Paper: the rung-0 table in the s060 handoff is IMPURE (torch head, oracle L0); any language
   number for the paper must come from `logophosis.py`. Rung 0 pure rerun not done.

Below = the s060/s059 handoffs, still current for the depth gate / logic chain / world details.

---
**Previous session number:** 059
**Session title:** **LOGIC/LANGUAGE ARC, STAGE 0 DEPTH GATE — PARTIAL PASS. A grown chain of
quad trioron links forms real depth (parity-4: 1-link .50 → 3 links 1.00; hop-2: tied link .92
vs shallow .49) — but only with a PATIENT Numa trigger and JOINT training (lock after settlement).
Frozen-greedy growth (lock before spawn) NEVER helps. Wall at parity-8 / hop-3 for every chain arm;
ReLU reference also collapses at hop-3; curriculum (lifecycle) lifts hop-3 only .59→.65, zero-shot
depth generalisation ≈ chance. Language world drafted (`language_world.py`), untrained.**

---

## READ THIS FIRST

Rocky present most of the session. Asks answered this session: (1) "how will it form language?" —
answer given and accepted: not an LLM; symbols (Phasecyte) → grounding (link eats symbol+world
state, credit lock) → composition = chain depth (what the gate tests) → production = recurrence
reversed → Stage 3 conditions a frozen LLM. Falsifiable arc claim: ~50-word grounded language with
negation + nesting, acquired continually with zero forgetting, compositional generalisation to
unseen combos and deeper nesting than trained. (2) Rocky: **double negation must NOT be forbidden**
— it is the discoverable depth case; `DOUBLE_NOT = True` in language_world.py, planned split =
train ≤1 NOT, test `not not`.

## Stage 0 depth gate — `experiments/progenitor/logic_chain.py`
Worlds: `parity_k` (12 bits, parity of first k; quad σ(z)=z+z² ⇒ exact lower bound ⌈log₂k⌉
layers), `hop_k` (8 slots with pointer symbols, label f^k(0), 8-way). Link 0 = ORACLE one-hot
symbol evidence in PhasecyteLeaf.evidence() shape `[L, class_cap=8]` (Phasecyte symbol discovery
deliberately deferred so the gate isolates depth). Link i≥1 = `Seeded(32+E→32, interior 48,
nonlinear quad)`, input concat[h_{i-1}, e], torch Linear head on the last link. 8K train / 4K
test, Adam 3e-3, bs 128, grad-clip 1, n=3. Arms: shallow (1 link), mlp3 (ReLU 3×48 ref), grown
(stall → freeze links → spawn → head moves → train new link only), grown_joint (spawn on stall,
ALL links stay trainable), tied (one link re-applied R times, BPTT, R+=1 on stall), curric (tied,
continual hop-1→2→3→4 same weights, zero-shot probe at each new depth). MAX_LINKS 5.
Logs: `outputs/logic_chain_s059{,_joint,_joint_p8,_fixed2,_fixed_p8,_curric}.log`.

| task | k | shallow | mlp3 | grown(frozen) | grown_joint | tied |
|---|---|---|---|---|---|---|
| parity | 2 | 1.00 | 1.00 | 1.00 | — | 1.00 |
| parity | 4 | .50 | 1.00 | .50 [5] | **1.00±.00 [3,3,3]** patience 8 / .50 [5] patience 3 | .81±.27 [5] |
| parity | 8 | .50 | .72±.24 | .50 | .50 (patience 8; FIXED 2/3/4-link joint 60 ep: 4→1.00, 8→.49/.50) | .50 |
| parity | 12 | .51 | .98 | .50 | .50 | .50 |
| hop | 1 | 1.00 | 1.00 | 1.00 | — | 1.00 |
| hop | 2 | .49 | .975 | .51 | — | **.92±.00** |
| hop | 3 | .54 | .60 | .555 | — | .59 |
| hop | 4 | .52 | .60 | .55 | — | .575 |
| curric hop 1/2/3/4 | | trained 1.00 / .911 / .649 / .594; zero-shot next depth .12/.23/.16/.29 (chance .125) |

**Readings**
1. Depth forms: parity-4 needs ≥2 quad layers and the grown chain reaches 1.00 at 3 links (one
   overshoot — stall fires once more before the plateau breaks); hop-2 solved by the tied link.
2. **Growth protocol is the whole story at k≤4:** frozen-greedy (credit lock BEFORE the new link
   settles) = chance on every task — parity has zero low-order signal so link 1 learns nothing and
   link 2 builds on garbage. Joint training + PATIENCE=3 also fails (spawns before the ln 2
   plateau breaks, ends with 5 half-trained links). Joint + PATIENCE=8, 30 ep/stage → PASS.
   **Rule: lock after settlement, never before spawn; stall horizon must outlast the plateau.**
3. Wall: parity-8 unreachable for 3/4-link quad chains even fixed + joint 60 ep (loss ≈ .68) while
   ReLU MLP gets .72/.98 — quad stacks don't optimise to high-degree parity at width 48 (parity is
   SQ-hard; negation nesting in the language world is parity-2/3, inside what works). Hop-3 is a
   depth wall for EVERY arm incl. the fixed reference (.60); recurrence/curriculum lifts to .65
   only, no zero-shot depth generalisation. Hypothesis: tied state h is an unnormalised linear
   readout; position info decays per hop — needs a winner-take-all / normalisation between links
   (lateral inhibition as a substrate primitive, not a hand op). UNTESTED.
4. Gate verdict: **PARTIAL PASS** — depth is formable by growth, under the corrected protocol; the
   ≥3-hop regime is open. Per the s058 plan ("if FAIL fix depth before language"): fix the hop-3
   wall before Stage 1/2 training, but the language world can be built/validated in parallel.

## Language world — `experiments/progenitor/language_world.py` (drafted, self-test passes)
Scenes 2–4 objects (color/shape/size/x/y on 4×4), 25-word vocab (CLASS_CAP 32, L_MAX 24),
grammar NP VP with ADJ/REL predicates, NOT/AND/OR nesting to depth d, truth evaluated in the scene
(comprehension = scene+sentence → T/F, balanced; production teacher = the TRUE sentences). 3
continual acquisition STAGES of vocabulary; HOLD_OUT adj–noun pairs for the compositional split;
depth split = deeper nesting than trained; DOUBLE_NOT allowed (Rocky). Encodings: scene 44-d
float, sentence [24, 32] one-hot evidence (Link-0 shape). No model trained on it yet.

## s060 arms run IN THIS SESSION (Rocky: "let's continue now") — ALL NULL so far
Logs `outputs/logic_chain_s059_{channels,channels_d2,ch_curric,wta,crisp}.log`; arms added to
`logic_chain.py`: `channels` (NCH windows + combiner, CH_DEPTH layers per link), `tied_wta`
(softmax τ=0.3 between applications), `tied_hard` (argmax one-hot, straight-through), `tied_gated`
(Rocky's gated ambiguity filter: snap to one-hot when top-2 margin > THETA=0.3, else pass soft),
`CH_CURRICULUM=1` (Mode E: each channel learns parity of its OWN window under its own head, locks;
then combiner learns parity-k).
| arm | parity-4 | parity-8 | parity-12 | hop-3 |
|---|---|---|---|---|
| channels 1-layer (degree 4 — MY BOUND ERROR for k=8) | 1.00 (60 ep; .50 if early-stopped at patience 10) | .50 | — | — |
| channels 2-layer, end-to-end | — | .50 | (killed) | — |
| channel curriculum 2-layer, 60 ep/stage | — | stage A per-channel 1.00/.67/.83, stage B .66±.29 (bimodal) | stage A ≈.66 each, stage B .51 | — |
| tied_wta | — | — | — | .53 (hop-2 .82 < tied .92) |
| tied_hard | — | — | — | .52 |
| tied_gated | — | — | — | .54 |
| (plain tied, ref) | | | | .59; mlp3 .60 |
Readings: (i) partition alone does not create intermediate signal — end-to-end channels fail
parity-8 for the same no-partial-progress reason; (ii) with own-frustration supervision the
COMPOSITION works on seeds whose channels settled (stage B bimodal), so the blocker is
seed-chaotic acquisition of the parity-4 primitive within 60 ep (fixed 2-layer channel ≈1/3 of
seeds) — remedy = the validated stall→grow rule applied per channel, not more hand layers;
(iii) every state-discretisation between tied applications is null on hop-3 — the wall is not
crispness of the raw 32 units. Remaining hypotheses for hop-3: the state must be a LEARNED symbol
(Phasecyte between links, Rocky's gated filter routing the ambiguous branch to Phasecyte
frustration) or BPTT through ≥3 quad applications is the optimisation limit — run a small GRU
reference on hop-3 first to know whether the task is even learnable at this budget.
Rocky's framing to keep: **the core is logic on crisp symbols; a gated ambiguity filter resolves
vague states into crisp ones; doubt is carried, not computed with.** Also: Stage 2 input should be
CHORD-encoded words (synthetic frequency sets per word, numbers not audio) so Link 0 = a real
Phasecyte discovering the inventory (IPA-free; clicks/nasals are just signatures); one-hot stays as
the oracle control.

## NEXT (s061)
0. **Purity audit (Rocky asked):** learner = pure substrate (`construct(Seeded(..., nonlinear=True))`).
   NOT package: linear torch head (66 p), the tied re-application loop (script, not Axis-7
   satellites), the stall→grow controller (script, not growth-trigger/credit-lock), Link 0 oracle
   one-hot (not Phasecyte). Promote the chain/tied controller into the package on the existing
   growth-trigger discipline and swap head+Link 0 BEFORE any language number goes in the paper.
A. **Rung 1 — negation:** DTRAIN=1, MAX_OBJ=2, L_MAX=10; test depth+1 = `not not` (DOUBLE_NOT) —
   the discovered-double-negation case. Then rung 2: MAX_OBJ=3–4 + relations (needs more data —
   run refs at learnable scale first). Then MODE=continual (3 vocab stages, forgetting matrix).
B. Sequential token reading for the chain (32-d per step, Axis-7 style) to drop the 300-d input.
C. Chord-encoded words + Phasecyte Link 0 (Rocky); 40K hop zero-shot probe; hop-3 Phasecyte-
   between-links only if a rung actually needs it.
-1. ~~GRU reference~~ DONE (above). Then **Phasecyte-between-links** tied arm
   (no BPTT; per-stage training like grown) on hop-3. Per-channel stall→grow for the parity
   curriculum (expect stage A → 3/3 seeds, stage B → 1.00).
0a. **Channels (Rocky, end of s059): width by division + depth by frustration.** Parity composes by
   PARTITION: parity(12) = parity(parity(1-4), parity(5-8), parity(9-12)); arm = 3 x k=4 channel
   links on disjoint bit windows + combiner link (any partition works for parity, so division on
   random subsets suffices). Predict PASS on parity-8/12. Rocky's prime sizes (2,3,5,7,11) =
   modular-residue/periodicity detectors — keep for the rhythm/temporal tasks, not parity.
0b. **Discrete state between links:** Phasecyte (a discretiser) or softmax/WTA between tied
   applications — the substrate's version of the LLM scratchpad token. Decides hop-3.
0. **Hop-3 wall probe:** tied link + normalised state between applications (softmax/WTA over h
   or a competitive-inhibition primitive); also wider link (interior 96) and a 2-link tied
   block. PASS = hop-3 ≥ .9 and zero-shot hop-4 > chance. Keep patience 8 / joint / lock-after-
   settle as the protocol.
1. Promote the protocol finding: growth controller = stall(patience≥8) → spawn → joint train →
   credit-lock settled links only after TARGET reached (not at spawn). Check against the
   s021 LR-noise guard.
2. Language Stage 1/2 on `language_world.py`: comprehension first (chain eats sentence evidence
   + scene), continual over the 3 vocab stages with forgetting metric, compositional + depth +
   double-negation splits; references: mlp, BoW-linear (leak check), small GRU. n=3.
3. Replace oracle Link 0 with PhasecyteLeaf on the token stream (symbol discovery), then
   dream_distill a settled segment.
4. Motion arc items (paper table, msil voter, save format) remain as listed in s058, secondary.

## GOTCHAS
- `logic_chain.py`: run from repo root; env SEEDS TASKS KS ARMS EPOCHS PATIENCE MAX_LINKS NTRAIN
  THREADS; `CURRICULUM=1` switches to the curriculum main. ~16 ms per link per batch of 128 →
  a 5-link grown run ≈ 10–20 min per (k, arm). Scratch capability script (fixed N links) lived in
  the session scratchpad, not committed; reproduce via `Chain(joint=True)` + `grow()` + PATIENCE huge.
- Substrate input-gradients flow (verified) so BPTT through links works.
- s057/s058 gotchas unchanged (Organ._train grad-bearing last_activations; motion.py from root).

## State of the build / Pointers
- Commits on `conscience-core` this session: logic_chain.py + language_world.py + logs + handoff.
  `main` NOT advanced. Pushed.
- Package (`trioron/`) untouched. Kinopsis untouched.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs / uncommitted
logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python
  3.10.12, torch 2.11.0, WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM.
