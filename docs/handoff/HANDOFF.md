# Trioron Handoff

**Session date:** 2026-08-23
**Session number:** 061
**Session title (s061 = "run 1+", Rocky):** **PURE-TRIORON LANGUAGE ORGANISM BUILT AND RUN AT RUNG 1
(negation). Everything that learns is substrate: Phasecyte Link 0 discovers the word inventory
from CHORD-encoded words (24 internal classes, 21–22/24 named, cluster purity .996), substrate
chain (`trioron/lifecycle/chain.py`), substrate head. Pure tied n=3: in-dist .868±.010 (ties GRU
.866, beats impure tied .846 by +.022), comp .792±.012 (GRU .804), depth+1 = `not not` .458±.030
— BELOW chance for pure tied and GRU alike: double negation is read as single negation. Rung-1
zero-shot double-negation target NOT met by any arm (best mlp3 .585).**

## READ THIS FIRST
Rocky's "+" = pure trioron. Scope chosen by Rocky: FULL purity incl. Phasecyte Link 0 (not just
head + controller). Rocky: "use a math function as initial feeder (like Kinopsis)" → words are
synthetic chords (K=3 Gaussian partials on 32 bins, `language_world.chord_table`, per-occurrence
jitter .05), numbers not audio; one-hot `encode_sentence` stays the oracle control (`LINK0=oracle`).

## What was built (commit a35b3d1 + this handoff)
- `trioron/lifecycle/chain.py` (package, spec §9.6 row added): `new_link` (Seeded quad, optional
  `fan_in_init`), `LinkChain(mode=tied|grown)` with a SUBSTRATE head link `Seeded(H→C, interior
  16)`, `StallTrigger` (Numa patience rule, epoch-grained), `fit_stage`, `fit_grow` (stall → grow →
  joint train, lock-after-settle; no credit lock at spawn).
- `experiments/progenitor/language_world.py`: `chord_table`, `chords_of`, `ids_of` (env CHORD_BINS 32,
  CHORD_K 3, CHORD_JITTER .05).
- `experiments/progenitor/language_pure.py`: `ChordLink0` = PhasecyteLeaf over the token chord
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
- `language_pure.py` std warnings for n=1 are harmless (`std()` on a single seed).

## NEXT (s062)
A. (done) pure grown n=3 in table: re-application (tied, 20.7K) beats fresh links (81K) on every
   axis again, as at rung 0. Rerun impure grown_joint seed 2 only if its comp/depth are needed.
B. Rung 1 double-negation: run the levers in Finding 4 (start with N=40K pure tied, one seed, to
   see whether depth+1 moves at all); then rung 2 (MAX_OBJ 3–4 + relations) with refs at a
   learnable scale first.
C. MODE=continual (3 vocab stages, forgetting matrix) on the pure organism — `language_pure.py`
   has no continual main yet; port `run_continual` from `language_chain.py`.
D. Sequential token reading (Axis-7 style, 32-d/step) to replace the flattened L_MAX×32 input;
   pairs naturally with nesting depth = re-application depth.
E. Paper: the rung-0 table in the s060 handoff is IMPURE (torch head, oracle L0); any language
   number for the paper must come from `language_pure.py`. Rung 0 pure rerun not done.

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
