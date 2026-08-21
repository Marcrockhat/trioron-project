# Trioron Handoff

**Session date:** 2026-08-22 (started 2026-08-21)
**Session number:** 059
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

## NEXT (s060)
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
