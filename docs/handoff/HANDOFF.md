# Trioron Handoff

**Session date:** 2026-11-24
**Session number:** 066
**Session title (s066: recovery after a shutdown; training-regime correction; PHASECYTE PILOT; night test; Vee's memory in the console):**
**s065 ended in an unplanned shutdown (~14:50) with the consolidation test in flight; s066 reran
it. Rocky then corrected the training regime — the organism is never batch-trained, it is a baby
that keeps absorbing — and named the Phasecyte as the fact-memory learner. A 40-fact streaming
pilot shows the phasecyte keeps old facts while learning new ones (.95–.98 with the class cap
off), that overfeeding per boundary destroys it, and that it does not beat a running class mean
(.993) or the s065 one-shot key table (.983). A night test then showed that only helper-written
rewordings beat no rewording at all, and the fact memory (day table + phasecyte fed at night) was
wired into the console: Vee answers the 40 pilot facts from memory and can be taught new ones.**

## s066 — READ THIS FIRST
- **Rocky's ruling (binding): never batch-train the organism.** A fixed-size leaf trained on
  shuffled batches is a CONTROL arm, not "the trioron arm". The s064/s065 lines "trioron has not
  earned its place" tested the quad cell as a static layer only. For fact memory the learner is
  the Phasecyte; feed rate vs the boundary (manifold / dream trigger) must be set with care —
  overfeeding loses information. Rocky's concern for the pilot: "that it can keep learning".
- Rocky confirmed: phasecyte is canon (v0.3.3); the eye (§10.11) "is just a limb". He approved
  option 2 (one class per fact by repeated exposure) and the 40 × 50 pilot size.
- Consolidation test + readout check: see the two bullets in the s065 body below. Net: facts in
  weights do not beat the table; the leaf lost to softmax-classifier TRAINING; quad = linear
  under a cosine objective.

- Later in s066 Rocky: helper-token cost of rewordings is acceptable; "forgetting when you're old
  is normal but when you're a child, I think some memories hold almost for a lifetime" → report
  the EARLIEST wave separately in every memory test, its loss is the serious failure; he asked
  for the memory to be wired into the console after the night test (done, below).

## Night test (s066) — `pilot_reword.py`, `pilot_night.py`, log `pilot_night_s066.log`
Same stream, held-out helper wordings and scoring as the pilot (W=50, no cap, 3 splits); only the
SOURCE of a fact's night material changes (original wording always included). Top-1 after wave 3.

| night material | phasecyte w0 | phasecyte mean | class mean of same feed |
|---|---|---|---|
| helper rewordings (the pilot) | .92 | .978 ± .009 | .993 ± .003 |
| Qwen's own rewordings (41.5 per fact, one fact got 0; free; meaning often drifts) | .90 | .937 ± .011 | .967 ± .007 |
| original + borrowed wording noise (no text written) | .95 | .958 ± .014 | .969 ± .005 |
| original only, repeated | .93 | .948 ± .010 | .970 ± .008 |
| reference: one-shot key table, no night | .95 | .983 ± .005 | |

Reading (Chloe's): only helper rewordings beat no rewording (+.02–.03). Qwen's rewordings are
the worst source. ONE wording through the plain 64-number sense already gives .97, so this
40-fact test is near its ceiling and cannot separate much. The phasecyte is 1–3 points under the
class mean of the same feed in every source.

## Vee's memory in the console (s066) — `vee_memory.py`, `console.py`
- `FactMemory`: day table (one read state per fact, cosine in the 64-number sense) + ONE
  phasecyte leaf fed by `sleep()` (original + 40 noise copies per fact, ~50 samples per boundary,
  no class cap, earlier nights never replayed). A fact the leaf recalls from its original wording
  leaves the day table; otherwise it stays for the next night. Both answer only above the
  threshold that lets 5 % of 2,000 untaught stage-2 questions through. Classes are named from
  current members (`mixed.tags`), not `PhasecyteLeaf.names()`.
- Childhood = the 40 pilot facts (four nights, helper rewordings). The leaf is NOT saved:
  `wake()` rebuilds it by replaying the childhood and the user's nights (`vee_facts.jsonl/.npz`)
  in order (a few seconds now; grows with life — a state file via `mixed.state_dict()` is open).
- Console: recall before the reflex; `/teach <answer>`, `/wrong <answer>` (teaches), `/sleep`,
  `/reflex` (ask Qwen anyway), `/facts`. `MEMORY=0` turns it off. Scripted check: taught "name of
  my cat → Miso", recalled from "What is my cat called?" same day, after `/sleep`, and after a
  restart; "What is my dog called?" ALSO returned Miso (a false recall).
- Self-test (`python3 experiments/incubator/vee_memory.py`, log `vee_memory_selftest_s066.log`,
  3 splits): after the childhood, N more nights of facts heard once (stage-2 questions, tested on
  their 2 other wordings; false recall on 300 untaught questions from the same articles):

| later nights | childhood w0 before → after | later facts recalled (false) | all facts in a plain sense table (false) | classes |
|---|---|---|---|---|
| 20 × 1 fact | .92 → .92 | .93 (.12) | .91 (.05) | 81–120 |
| 20 × 3 facts | .92 → .93 | .92 (.08) | .86 (.03) | 138–276 |
| 100 × 1 fact | .92 → .91 | .89 (.07) | .82 (.04) | 226–406 |

  Reading (Chloe's): the earliest facts hold through 100 further nights; one or three facts per
  night works. The leaf recalls more than the plain table but with about twice the false recall,
  so the two are NOT compared at a matched false rate, and the table here is the plain sense,
  not the stronger s065 learned key. Classes: ~2 per later fact, never merged.

## Phasecyte pilot (s066) — `experiments/incubator/pilot_stream.py`
40 facts (20 articles × 2 questions), 50 matron rewordings each, 10 held out per fact, 20 untaught
questions as negatives; four waves of 10 facts, nothing frozen, old waves never shown again; one
`PhasecyteLeaf` on a 64-number PCA sense of the read state; 3 splits. Top-1 on held-out wordings
after wave 3 (w0..w3, mean). W = questions per boundary. Phasecyte rows = told, named by buffer tags.

| setting | w0 | w1 | w2 | w3 | mean | classes |
|---|---|---|---|---|---|---|
| cap 43, W=25 | .92 | .77 | .53 | .31 | .63 | 43 |
| cap 43, W=50 | .92 | .87 | .80 | .50 | .77 | 43 |
| cap 43, W=100 | .93 | .93 | .98 | .77 | .90 | 43 |
| cap 43, W=200 | .89 | .86 | .99 | .67 | .85 | 42 |
| cap 43, W=400 (whole wave per boundary) | .00 | .20 | .43 | .17 | .20 | 8 |
| cap 43, W=100, heard 3× | .92 | .60 | .23 | .43 | .55 | 43 |
| no cap, W=25 | .91 | 1.00 | .99 | 1.00 | .976±.004 | 142 |
| no cap, W=50 | .92 | .99 | 1.00 | 1.00 | .978±.009 | 67 |
| no cap, W=100 | .91 | .97 | .98 | .97 | .954±.008 | 46 |
| no cap, W=100, heard 3× | .94 | .96 | .99 | .97 | .966±.027 | 139 |
| control: running class mean (64 numbers) | .97 | 1.00 | 1.00 | 1.00 | .993±.003 | — |
| control: s065 one-shot key table (32 numbers, 1 wording) | .95 | .99 | .99 | 1.00 | .983±.005 | — |

Reading (Chloe's, not reviewed by Rocky):
- Keeps learning: wave 0 holds .91–.94 after three later waves; all 40 facts get a class (no cap).
  Untold arm (no labels, named afterwards) .92–.96.
- Overfeeding confirmed: 400 per boundary → 8 classes for 40 facts. One division sitting per
  boundary; too much per sitting blurs facts together.
- The cap-43 failures at small W are LATE facts blocked by the class cap (classes multiply with
  the number of boundaries: 3.5 per fact at W=25), not forgetting.
- Does not beat either control, and stores more (template + rolling member buffer per class).
- **`PhasecyteLeaf.names()` is wrong for many small classes** (lifetime label ledger inherits
  counts through divisions): .54 vs .90 on the same organism at W=100. The driver names from
  `mixed.tags` instead. `names()` itself NOT changed (shared with the nest benches) — decide
  with Rocky.
- Caveats: 40 facts; one sense size; 12 % of wordings are near-duplicates (> .9 similar);
  thresholds set on test negatives; class count still rising at the end (no self-arrest seen).
- First smoke on synthetic wordings was misleadingly easy (all 1.00); do not use SYNTH=1 for
  anything but mechanics.

**s065 summary (INCUBATOR STAGE 2 — fact memory, first cut):**
**Vee got a key leaf and a memory (`learn_keys` / `remember` / `recall`). The matron wrote two
rewordings of each of the 2,808 stage-1 questions. A learned 32-number key recalls a taught fact
from a rewording better than a nearest-neighbour cache holding 3,072 numbers per fact (.907 vs
.833 at 1 % false override). But a learned LINEAR key does as well as the quad leaf — the second
stage in a row where trioron itself has not earned its place.**

## READ THIS FIRST
- **Design of record: `docs/design/incubator_matron.md`** (stage-2 result is in §3). Binding
  decisions from s064 stand: incubator organism has no parameter cap (85 % of WSL RAM, swap
  excluded; benches/CI keep 50 K); paper is repo-only; sandbox in WSL with offline Simple English
  Wikipedia; creativity = SCAMPER routes; matron budget = up to 50 % of the weekly limit (Max 5×).
- Rocky's framings to keep: humans hallucinate too (told > doesn't-fit > inner doubt); the
  self-check monologue is stage 4; self-hosting milestones ("Linux on Minix").
- **Always carry a linear control in incubator gates.** "Beats the cache" is not "trioron pays".
- Suspect the branch partition first for 1-layer relational nulls (s063 rule).
- Rocky was NOT asked anything this session; nothing here is newly signed off by him. The stage-2
  protocol and the "PASS" reading below are Chloe's and await his review.

## Stage 2 — fact memory, first cut (n=5 article-level splits)
Pipeline: 2,808 questions × (original + light rewording p1 + heavy rewording p2), written by four
Sonnet helpers (+1 small fix job; ~590 K subagent tokens in total) → the reflex READS each text
without answering (`Reflex.read`, 0.53–0.69 s/text, 8,424 texts) → two states per text: last
prompt token and mean over the question's tokens (3,072 numbers together) → gate
`experiments/incubator/stage2_memory.py`.

Protocol: 30 % of articles held out. One question from each of half of them is taught (original
wording stored). Positives = its two rewordings. Negatives = the other question of the same
article (hard) + every question of untaught held-out articles (easy), all three wordings.
"Recalled" = right fact is nearest AND above the threshold that lets 1 % of negatives through.
"Crowded" = memory also holds the ~1,970 training questions.

| key (numbers per fact) | recalled, 197 facts | recalled, 2,164 facts |
|---|---|---|
| cache, raw states (3,072) | .782±.026 | .600±.029 |
| cache, centred states (3,072) — best cache | .833±.030 | .620±.024 |
| cache, random projection (32) — best small cache | .766±.049 | .557±.060 |
| cache, PCA-32 (32) | .552±.015 | .380±.017 |
| learned LINEAR key (32), 8,224 params | .892±.023 | **.718±.019** |
| **organism quad key leaf (32), 26,112 live weights (41,400 allocated slots)** | **.907±.019** | .681±.035 |

Honest reading:
- Gate as written (vs a nearest-neighbour cache) passes: +.074 / +.062 with 96× fewer numbers
  per fact; +.14 / +.12 over a cache of the same size.
- Quad leaf = linear key (ahead by .015 uncrowded, behind by .037 crowded). Learning the key
  pays; the trioron leaf does not, yet.
- Ranking is nearly solved for every full-size arm (right fact nearest .986–.997). The
  difficulty is the THRESHOLD; half to three quarters of false overrides are the same-article
  sibling question.
- Load: 197 → 2,164 facts costs every arm ~.2 at 1 % false override (.98 → .90 at 5 %).
- State ablation (3 splits): last token alone ≈ both (organism .865/.649, best cache .852/.630 —
  there the learned key buys only compression); mean alone weaker (.718/.615 vs cache .464/.352).
- Key-size sweep (Rocky asked whether the quad leaf stores less per fact; 3 splits for 8 and 16):
  linear / quad at 8 = .626 / .598, at 16 = .834 / .817, at 32 = .892 / .907 (crowded .407/.365,
  .616/.643, .718/.681). No packing advantage. A learned 16-number key equals the full cache.
- **Drift test** (`stage2_drift.py`, 3 splits, 5 lessons, no replay; old facts stored after
  lesson 0, new after lesson 5; linear / quad): free learning destroys old facts (.888→.022 /
  .873→.029, key cosine .32); epigenetic lock holds them at every strength (old .86–.91, new
  .90–.92), uniform λ = saliency λ; but FROZEN after lesson 0 does as well (old .888/.873, new
  .916/.901 vs free new .926/.906) — one fifth of the data already gives a full-quality key, so
  the lock buys no plasticity here. Critical period (learn early, freeze) suffices.
- **Dream test** (Rocky: "we have the dreaming phase"; `stage2_dream.py`, 5 splits, lessons = 5
  topic DOMAINS by k-means on mean read-state, no replay; old facts = first domain, new = later
  domains; one memory; old / new, linear | quad): frozen .881/.846 | .866/.835; free stale
  .000/.862 | .000/.832; free + dream re-key .896/.849 | .871/.814; lock s=30 + dream .896/.844 |
  .883/.835; joint all-domains ceiling .908/.892 | .919/.870. Reading: re-key works (.000 → .896)
  and re-reading a question reproduces its state exactly (cos 1.000000, 40 texts), so the dream
  needs only question text. Domain shift opens headroom (joint − frozen on new = +.045±.020
  linear, 5/5 splits) but sequential no-replay learning does not reach it (free + dream +.003±.040
  over frozen). Re-key-only dream = safety, no accuracy; the gain needs REPLAY in the dream.
- **Replay dream** (log `stage2_dream_replay_s065.log`): dream = retrain the shadow on ALL kept
  lessons together, then re-key. Linear: 15 passes once .924/.890, 60 passes .921/.898, nightly
  (after every lesson, 15 passes) .936/.897, vs frozen .881/.846 and joint .908/.892 → reaches the
  ceiling; per-split gain on new facts +.044±.046 / +.052±.036 / +.051±.044. Quad: .901/.851,
  .898/.846, .902/.846 vs frozen .866/.835, joint .919/.870 → gain inside the noise.
- **Consolidation test** (run finished s066; `stage2_consolidate.py`, log
  `stage2_consolidate_s065.log`, 3 splits, ~2,170 facts, same h numbers per fact; recall at 1 %
  false override, h = 8 / 16 / 32): table .407 / .616 / .727; rows trained, key leaf frozen
  .320 / .607 / .701; rows + key leaf trained .338 / .652 / .707 (paired vs table −.069±.036,
  +.036±.010, −.019±.030); trioron quad leaf with one output cell per fact, top probability
  .030 / .082 / .469 (right fact nearest .395 / .661 / .901 vs table .766 / .946 / .986).
  Reading: weights do not beat the one-shot table; the leaf fails mostly at the threshold.
  The leaf arm changed the readout too, so s066 separated them (`stage2_consolidate_cos.py`,
  log `stage2_consolidate_cos_s066.log`): same softmax-trained leaf read by cosine codes
  .108 / .310 / .473; quad cells trained contrastively as a table .389 / .600 / .726 (ties
  linear, paired −.018±.069 / −.016±.049 / −.001±.044); quad rows + trunk, cosine classifier
  .355 / .606 / .706. Cause = softmax-classifier TRAINING over ~2,170 output cells, not the
  quad cell and not the readout alone.
- **PARAM COUNT CORRECTION:** `Organism.n_params()` counts ALLOCATED arena slots (cells × 69), not
  live weights. Quad key leaf = 26,112 live (not 41,400; ~3× linear, not 5×); stage 1 leaf = 6,240
  (not 12,834). Conclusions unchanged. `n_params` itself NOT changed (shared with Logophosis /
  Kinopsis) — decide with Rocky whether it should count live weights.
- Thresholds were set on the test negatives, identically for all arms (an arm comparison, not a
  deployable number).
- s064's early worry is retired: a question's nearest neighbour is rarely its sibling, but that
  does not hurt retrieval of the same question reworded.

## Side test — phasor cell vs quad (Rocky's resonance question)
`experiments/progenitor/phasor_vs_quad.py`, standalone torch (NOT the arena), 5 seeds, 8 cells,
log `outputs/phasor_vs_quad_s065.log`. Phasor cell: complex weights, h = |Σ w x|²; sees only
relative phase by construction. Test accuracy:

| task | quad one-hot | quad cos/sin | phasor |
|---|---|---|---|
| same/different, held-out symbols | .426±.108 | .987±.030 | 1.000±.000 |
| offset (b−a mod 16), held-out pairs | .025±.016 | .978±.018 | 1.000±.000 |
| binding by phase tag, 100 samples | — | .785±.048 | .907±.027 |
| binding by phase tag, 4,000 samples | — | .988±.005 | .993±.003 |

Reading: the big effect is the phase ENCODING (one-hot cannot generalise a relation to unseen
symbols; any phase code can). The phasor CELL adds a smaller, real gain: exact generalisation and
better sample efficiency (+.12 at 100 samples), no gain with plenty of data. Caveats: toy tasks,
phases handed to the network (nothing here discovers the encoding), not on the substrate. Upgrade
path noted, unverified: activation Re(z)+|z|² with real weights equals the quad z+z².

## What was built (this session's commit)
- `trioron/organisms/vee.py`: stage 2 — `learn_keys(H, group, block, k, e)`, `remember(H, texts)`,
  `key(H)`, `recall(H)` → (index, cosine), `memory_bytes()`. Keys are stored EMBEDDED, so the key
  leaf must not drift after facts are stored (untested).
- `trioron/organisms/base.py`: `group_contrast`, `train_contrastive`.
- `experiments/incubator/reflex.py`: `Reflex.read(question)` (keeps the prompt head decoded;
  `answer()` invalidates it). `stage2_collect.py` (resumable), `stage2_memory.py` (gate).
- Spec §9.16 rows for vee.py and base.py; design §3 stage-2 result; test
  `tests/test_v2/test_organisms.py::test_vee_fact_memory_roundtrip` (organism tests 7/7).
- Logs committed: `outputs/incubator/stage2_memory_s065.log`, `..._state_last.log`, `..._state_mean.log`,
  `..._e8.log`, `..._e16.log`, `stage2_drift_s065.log`, `outputs/phasor_vs_quad_s065.log`.
- `experiments/incubator/stage2_drift.py`; `base.py`: `leaf(nonlinear=)`, `train_contrastive(penalty=, after_backward=)`.
- Day/night on Vee: `study(H, group, block)` trains a SHADOW copy of the key leaf (live leaf and
  stored keys untouched), `dream(H_cues)` swaps it in and re-keys every fact from re-read states
  of `vee.cues` (`remember(..., cues=)`). Test `test_vee_study_and_dream` (organism tests 8/8).
  `experiments/incubator/stage2_dream.py`, log `outputs/incubator/stage2_dream_s065.log`.
- Replay: `dream(H_cues, H_lessons, epochs=15)`; `learn_keys(..., texts=)` / `study(..., texts=)`
  keep lesson questions (`lesson_cues`, `lesson_group`, `lesson_block`; ids unique across lessons).
  `leaf(max_bytes=)`. Console: `/exit`, `/q` leave; unknown `/command` no longer goes to the reflex.
- Rocky ran a console session (10 verdicts, `console_verdicts.*`, local): doubt sensible on fact
  questions, but .91–.98 on all four ordering puzzles right or wrong (out of distribution);
  reflex failed 2 of 4 ordering puzzles; `/wrong` given without the right answer 3 times.

## Environment facts (s064, still true)
- Qwen GGUF already on disk: `~/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf`
  (0.5B sibling at `~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`). ~1.8 GB RSS, ~21 tok/s read
  on this Ryzen 5 PRO 4650U (6 cores; use 6 threads, not 12).
- Wikipedia: `~/data/wikipedia/wikipedia_en-simple_all_nopic_2026-09.zim` (1.0 GB, 400 K
  articles, full-text index); `pip install --user libzim` done.
- bubblewrap 0.6.1 installed; verified: scratch dir writable, home invisible, network and
  system writes blocked. Time/memory limits per run NOT set up yet.
- Host disk has ~65 GB free (the 907 GB WSL reports is the virtual disk ceiling).
- `sudo` cannot run through the `!` prefix; Rocky runs it in a separate terminal.
- Rocky's plan: Max 5× ($100). **Matron budget DECIDED: up to 50 % of the weekly limit.**
  Not yet calibrated in tokens: ask Rocky for the weekly % from `/usage`. Measured s064 use:
  Fable 131 K out / 365 K new in / 18.0 M cached; Sonnet question job (2,985 questions)
  166 K out / 635 K new in / 2.7 M cached.
  Calibration (Rocky's `/usage`, 2026-11-24): the Fable weekly bar read 8 % after s064, so
  one s064-sized day ≈ 8 % and the 50 % cap ≈ 6 such days/week ≈ 0.8 M Fable output tokens
  (rough: assumes s064 was the week's only Fable use; how Sonnet helpers count is unknown —
  re-read `/usage` before and after the first loop run to measure it).

## Gotchas (cumulative, new first)
- **llama_cpp:** never build `Llama(embedding=True)` — it inflates the context to 256
  sequences (3.7 GB) and breaks decoding. Use `llama_set_embeddings(ctx, True)` on a plain
  context; logits stay bit-identical and `llama_get_embeddings_ith(ctx, -1)` gives the last
  token's hidden state. `logits_all=True` is slow and allocates n_ctx × 152 K floats.
- Hidden states differ slightly with batch chunking (cos .9994): always extract the same way.
- The lead sampler draws with replacement → duplicate articles; split and de-dup by title/id.
- Different question writers differ in difficulty (25–32 % Qwen accuracy): always report
  within-writer and a no-brain control.
- Branch partition (s063). Rewake destroys downstream symbol mappings. `pkill -f` kills this
  shell — kill by PID. One heavy process at a time on 7.6 GB RAM.

## Gotchas (cumulative, new first)
- **`Reflex.read` keys match `answer()`'s h_pre only to cos ~.999** (different batch chunking).
  Compare read-keys only with read-keys; the console must call `read` for memory and `answer`
  for doubt.
- Sonnet writers copy the original for ~12–19 % of "light" rewordings on the first pass; three of
  four fixed themselves, one needed a fix job. Always audit for verbatim copies and answer leaks.
  Parallel helpers share the scratchpad: tell them to use unique scratch file names.
- llama_cpp: never `Llama(embedding=True)`; use `llama_set_embeddings(ctx, True)` on a plain
  context. `logits_all=True` is slow. Extract hidden states the same way every time.
- The lead sampler draws with replacement → de-dup by title/id. Question writers differ in
  difficulty: report within-writer and a no-brain control.
- Branch partition (s063). Rewake destroys downstream symbol mappings. `pkill -f` / `pgrep -f`
  hit this shell — kill by PID. One heavy process at a time on 7.6 GB RAM.

## Open questions for Rocky
1. Review the stage-2 reading: is "beats the cache, ties linear" a pass for you, or does stage 2
   stay open until trioron-specific machinery (growth, λ firmness, per-fact σ) shows a gain?
2. ANSWERED: Fable weekly bar 12 % after s065 (8 % after s064) → this session ≈ 4 points. Still
   unknown whether the Sonnet helpers (~590 K tokens) count on that bar or a separate one.
3. Terms check before any matron-written data is committed (carried from s064).

## NEXT (s067)
00. Phasecyte fact memory (Rocky's direction, s066):
   a. merge consumer for same-fact classes / self-arrest (class count must stop rising);
   b. two-leaf nest arm (crossing the 43-class leaf boundary) — planned in s066, not run;
   c. scale past 40 facts; feed-rate rule from the sweep (W ≈ 50–100 for 10 new facts);
   d. where could it BEAT the class mean? candidates: untold discovery (no labels), per-fact σ
      from the manifold sketch as the threshold, far fewer wordings per fact;
   e. fix or replace `PhasecyteLeaf.names()` (buffer-tag naming) — needs Rocky's decision;
   f. console memory: compare leaf vs table at a MATCHED false rate and against the s065 learned
      key; false recall on near questions ("my dog" → the cat's name) — sibling-aware threshold;
      save the leaf (`mixed.state_dict()`) instead of replaying life at wake; Rocky to try it:
      `python3 experiments/incubator/console.py`;
   g. a harder night test (hundreds of facts) — the 40-fact one is at its ceiling.
0. Stage 2, where trioron could earn its place (pick with Rocky):
   a. per-fact threshold from each fact's own spread (manifold σ; needs ≥2 wordings per fact);
   b. DONE s065 (drift test): lock works, freeze works as well. Open: a regime where later
      lessons improve the key (new domain / tiny first lesson) so lock vs freeze can separate;
      DONE s065 (dream test): domain shift gives +.045 headroom, reached only by joint training.
      DONE s065 (replay dream): linear reaches the joint row with 15 passes; quad does not gain.
      Not done: the dream end to end through the reflex (re-read ~0.5 s per kept question);
      DONE s066 (consolidation test): facts in weights do not beat the table; trioron leaf far
      behind because of softmax training (readout check); quad = linear under a cosine objective;
   c. growth-pays curve: recall vs key-leaf size and vs facts stored; bytes per parameter;
   d. deployable threshold set on training articles; end-to-end override accuracy with the reflex.
1. DONE s066: memory wired into the console (phasecyte + day table, `vee_memory.py`). Open: the
   standing loop: checkpoint/resume, token logging, budget stop.
2. Serial graft as a substrate op + external-state rerun through it.
3. Snap on the Logophosis notᵏ curve (design §2.1 row 2).
4. Sandbox per-run time and memory limits; the code-running action (stage 3).
5. Phasor cell: if Rocky wants it, next is a phasor phenotype on the arena (verify the
   Re(z)+|z|² = quad upgrade path) and a task where phase must be DISCOVERED, not given.
6. Carried: public manual Ch3–5, Axis-5 field-driven branch membership, receptor-freeze-on-rewake,
   Logophosis continual mode, Link-0 compaction, Kinopsis regression, four pre-existing v2 test
   failures (TestCredit ×2, TestGrowth ×1, TestDreamCycle ×1).

## State of the build / Pointers
- Branch `conscience-core`; s065 and s066 work committed and pushed (see `git log -3`). `main` still at
  e8e7b40 (v0.3.3).
- **Deliberately uncommitted (local to this PC):** everything matron-written or derived in
  `outputs/incubator/`: s064's `stage1_*` data and `vee_stage1.pt`; s065's
  `stage2_para_in_part*.tsv`, `stage2_para_part*_c*.jsonl` (paraphrases; part9 = the fix job),
  `stage2_keys.jsonl`, `stage2_keys.npz` (52 MB), collect logs; s066's `pilot_in_*.tsv`, `pilot_para_*.jsonl`
  (2,100 rewordings, ~330 K helper tokens), `pilot_keys.jsonl`, `pilot_keys.npz` (`pilot_collect.py`, 25 min), `pilot_qwen_para.jsonl`, `pilot_qwen_keys.*` (`pilot_reword.py`, 70 min, Qwen-written), `vee_facts.*` (what the user taught Vee). Rebuild: paraphrases need the
  matron again; keys via `stage2_collect.py` (~75 min).
- Memory (this PC only): `stage2_fact_memory_result.md`, `incubator_matron_arc.md`,
  `external_state_depth_result.md`.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs / uncommitted
logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python 3.10.12, torch 2.11.0,
  WSL2, `python3` (NOT `python`), 12 logical CPUs (6 cores), 7.6 GB RAM. New this session:
  `libzim` (pip --user), `bubblewrap` (apt). Already present: `llama_cpp` 0.3.23,
  `transformers` 5.8.0.

---
History pointers (no dated archives): s064 handoff at e67b6ce (incubator design, limb tests, stage 1); s063 handoff at e8e7b40 (Logophosis Mode E transfer,
public manual, branch-partition finding); s062 at c4ac3db (seq gotchas, organism API).
