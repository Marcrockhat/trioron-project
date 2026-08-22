# Trioron Handoff

**Session date:** 2026-08-22
**Session number:** 060
**Session title (s060, in progress):** **HOP-3 WALL WAS A DATA WALL. At 40K samples (5×) hop-3:
GRU .923±.019, tied trioron link .888±.007, MLP3 .816±.029 (all ≈.58–.60 at 8K). Depth gate hop
side CLEARED: the re-applied link tracks 3-hop composition within .035 of a matched GRU. Lesson:
run the references at the data scale where the task is learnable BEFORE judging substrate arms.**
Logs `outputs/logic_chain_s060_{gru,gru40k,tied40k_s0..2}.log`; `gru` arm + `SEED_LIST` env added.
**LANGUAGE RUNG 0 (s060 afternoon) — SUBSTRATE COMPREHENDS.** `experiments/progenitor/language_chain.py`
(MODE=joint, MAX_OBJ=2, DTRAIN=0, L_MAX=8, N=20K, 20 ep/stage, MAX_LINKS 4, n=3; logs
`outputs/language_s060_rung0_s{0,1,2}.log`). Input = concat[scene 44-d, sentence 8×32 one-hot];
label = truth of sentence in scene (balanced). Tests: in-dist / COMPOSITIONAL (held-out adj-noun
pairs only) / DEPTH+1 (negation, untrained).
| arm | in-dist | comp | depth+1 | depth | params |
|---|---|---|---|---|---|
| tied (one link re-applied) | **.905±.003** | **.878±.006** | .609±.008 | R=4 | ~16K |
| grown_joint | .894±.010 | .876±.008 | .610±.018 | 4 links | ~64K |
| mlp3 ref | .901±.005 | .858±.002 | .596±.025 | — | 19K |
| gru token-by-token ref | .887±.003 | .837±.010 | .527±.012 | — | 36K |
| bow leak bar | .583±.004 | .609±.007 | .555±.013 | — | — |
Tied stage trace (all seeds ≈): R=1 .637 → R=2 .83 → R=3 .89 → R=4 .905 (cap). Readings: ties MLP
in-dist, beats MLP +.020 / GRU +.041 on compositional (>2σ); re-application > fresh links at 1/4
params; depth+1 ≈ .60 for all (negation unseen) → rung 1 target. BoW .58 = binding needed.
**Gotchas found today:** substrate forward on an 844-d input = 407 ms/batch (25× the hop link) —
keep L_MAX tight (8) or read tokens sequentially; a 4-object/depth-1/L_MAX=24 world is NOT learned by
ANY reference at 20K (bow .73, mlp .76, gru .72) — climb rungs (Mode E). Never `pkill -f <script>`
from the assistant shell (kills own shell, exit 144) — kill by PID from /proc environ. 40K hop
curriculum zero-shot probe was killed for CPU (oversubscription 25 load / 12 CPU); still open.
Below = the s059 handoff, still current for everything else.

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
