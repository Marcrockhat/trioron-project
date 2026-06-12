# Trioron Handoff

**Session date:** 2026-06-13
**Session number:** 034
**Session title:** **COMPONENTIAL ARC, END TO END: (1) the modifier classes
Rocky predicted (chicken-duck, dog-goat) appear UNSUPERVISED and allowing
them is worth +0.13/+0.16 — but their names are LABEL INFORMATION (three
label-free readouts falsified). (2) Rocky's answer — labels as REFERENCE
CARRIERS — built as D22 (label tap bank): non-disturbance EXACT, taxonomy
headroom fully recovered at 5% label coverage, "chicken-duck" named by the
organism. Mode-smearing limit recorded on data_hard; (class × label)
counts proposed as the next increment.**

---

## READ THIS FIRST

1. **Governing docs:** `docs/design/mixed_stream_growth.md` (D11–D22;
   D22 is this session's row, full measured record) and spec §10.6–10.10
   (+ §9.15 partition row for `pcll/labels.py`). The componential
   falsification record: `receptor_period_frustration.md` §10 (s029
   bullet, amended) + `run_componential.py` docstring. The tap-bank
   record: `run_label_taps.py` docstring + the D22 row.
2. **Everything from s033 stands** (rebuild M3–M6, two laws,
   membership-quality defaults, M2 0.829). This session added the
   componential probe, then `trioron/pcll/labels.py` + a minimal
   `mixed.py` integration (observe(labels=), boundary deposit,
   state_dict) — learning paths untouched (gate A: bit-identical).
3. **A session was killed between s033 and this one** — it was asked to
   verify M1–M6 implemented + run bench-15 (chained-15) trials. Neither
   happened; still open if Rocky still wants them.

## Arc part 1 — componential semantics (s029 backlog): built + falsified

`run_componential.py` (commit `8d363b2`), 3 seeds data_hard + 3-epoch
taxonomy:

1. The semantic-modifier classes EXIST unsupervised: taxonomy
   chicken:0.49/duck:0.48 (one stable class), goat:0.61/dog:0.39,
   cat:0.81/dog:0.19; data_hard 14–16 true blends/seed among ~77.
2. Allowing them is worth +0.128/+0.158 (0.829→0.957, 0.785→0.943)
   under composition-true scoring.
3. NO label-free readout recovers the names — three falsifications:
   signed projection 0.00 precision (prototype self-contamination);
   LOO mixture decomposition 0.00 (incoherent pollution averages to
   zero phasor; mode-smeared prototypes); buffer crossmatch 0.02
   (consolidation already moved everything the matched filter could
   see). **Law (semantic form of s033 law 1): a buffer's blend
   identity is label information.**
4. Vocabulary gap is structural: "chicken-duck" needs duck as a
   concept; the percept genuinely doesn't separate them.

## Arc part 2 — D22 label tap bank (Rocky's design): built + gated

Rocky's directive: feed labels "at a different frequency" so they
cannot disturb the learning frequency. Lock-in translation: deposits
are an unordered phasor sum, so the second frequency = multiply by the
reference at deposit time, then integrate — **the label is a reference
carrier, not a data dim.** `trioron/pcll/labels.py` (LabelTapBank);
`MixedStreamController.observe(x, labels=)` routes labeled rows'
value-phasors into per-label taps at the boundary (after
canonicalization). Write-only from learning's side; rides state_dict
for ship/wake. Labels only ever come from TRAINING-stream rows; all
accuracies read on the held-out, never-labeled test draw (Rocky's
separation instruction).

Gate battery (`run_label_taps.py`, all PASS):

- **A. Non-disturbance: EXACT** — same seed, labels OFF vs 100%:
  templates bit-identical, 16k labeled deposits absorbed.
- **C. Taxonomy (unimodal labels): headroom RECOVERED** — member-mix
  relational accuracy 0.789 → **0.948 = the oracle (0.945) at 5%
  coverage** (271 labels). At 100%, blends are NAMED: "chicken-duck"
  (+duck 0.30; member mix duck .51/chicken .49 vs truth .48/.49),
  "goat-dog". The s029 ask is now produced by the organism.
  `member_mix` must rank by margin E/σ, not raw E (D20 ruling recurs:
  raw E let the stronger tap of the 0.996-collinear chicken/duck pair
  absorb every member).
- **B. data_hard (multimodal labels): MODE-SMEARING LIMIT** — a
  species = 3 scattered modes; its tap prototype is the phasor mean
  over them (washed out — disruptor-invisibility physics at species
  scale). Tap-primary 0.53 vs majority-map 0.829; relational sets
  LOSE to strict; P/R ≤ 0.09/0.30 at any coverage. Species-grain
  carriers cannot name mode-grain classes.

Tests: 107 passed + the SAME 4 carries (no new failures).

## Current gate battery (s033 numbers unchanged, all green)

| gate | number | note |
|---|---|---|
| M2 mixed-stream | 0.829 | defaults stack |
| M3 settlement | 0.924 | pinned config |
| M4 composer | 0.729 > 0.620 | |
| M5 manifold | 0.836 end, −0.013 decay | |
| D20 membership | triplet | reject 65.6%@100% |
| D22 label taps | A exact; C 0.948=oracle@5%; B limit recorded | NEW |
| tests | 107 + same 4 carries | |

## State of the build

- Branch `progenitor-council`, committed through `6a0f200`
  (componential probe `8d363b2`, handoff `7f88609`, D22 `6a0f200`),
  pushed after this rewrite.
- **DO-NOT-COMMIT carries (unchanged):** `trioron/bases/developmental.py`,
  `trioron/lifecycle/developmental.py`, `trioron/viz/export.py`;
  `.claude/`, `runs/` untracked.
- `trioron/pcll/mixed.py` split still OVERDUE (~1050 lines now).

## Open questions / NEXT (priority order)

1. **(class × label) counts — the D22 next increment (needs Rocky):**
   at the boundary, membership and labels align; counting labels per
   class (still write-only annotation) captures the data_hard 0.957
   oracle directly and beats mode-smearing — naming at the grain the
   organism actually discovered. Small build (the boundary already
   has both vectors in hand).
2. **The killed session's tasks, if still wanted:** M1–M6 check +
   bench-15 (chained-15) trials.
3. **D21 nursery (proposed, needs Rocky).** Doubly motivated now
   (refused/unknown matter is where labels would help most).
4. **mixed.py split + constructor profiles.**
5. Deferred escalations: label-supervised consolidation (the law-1
   boundary-placement lever); label-pressured division (recommend
   against — perceptual-taxonomy ruling). Carried: dendrite
   settlement threshold; composer semantics doc §4; PCLL-path
   compaction (tap bank adds vocab × dims complex — joins the
   lifetime-memory concern); buffer drop-at-freeze; external
   validation arc; 4 pre-existing test_v2 failures.

## Pointers

- D22: `trioron/pcll/labels.py`, `run_label_taps.py` (gates),
  `mixed.py` observe/period_boundary/state_dict (annotation carrier
  blocks), spec §9.15 row, design-doc D22 row.
- Componential: `run_componential.py`,
  `receptor_period_frustration.md` §10.
- Gates: `run_m2_mixed`, `run_m3_settlement`, `run_m4_composer`,
  `run_m5_manifold`, `run_d18_merge`, `run_d20_membership`,
  `run_label_taps`.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, WSL2, `python3`, `OMP_NUM_THREADS=8`.
  Label-taps battery ~10 min; componential probe ~6 min; full gate
  battery ~15 min; tests ~12 s.
