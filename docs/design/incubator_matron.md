# The incubator: a reflex brain, a growing organism, and a matron (design, s064)

**Status:** signed off in the main (s064); stage 1 passed (s064); stage 2 first cut run (s065).
**Owner:** Rocky (direction) / Chloe (engineering; also plays the matron).
**Name (Rocky, s064): the organism is Vee** — from Eve and Veda. Code: `trioron/organisms/vee.py`.

**Matron budget (Rocky, s064): up to 50 % of the weekly subscription limit (Max 5×).** The loop must log tokens and stop at the cap.

**Question:** can a trioron organism, grown on top of a small frozen language
model and raised by a stronger model, show that *growth pays* — that as its
parameter count rises, measurable awareness, logic depth, vocabulary and
creativity rise with it?

**Budget (Rocky, s064, APPROVED): no parameter cap.** The organism may grow
until it would threaten the machine, i.e. up to a **memory ceiling** of 85 % of
the WSL RAM allocation (§5.1), not a parameter count. The incubator organism is
**exempt from the Phase 1 contract** (spec §6: 50 K params per substrate). The
Phase 1 cap stays binding for the existing benches and CI, so past results stay
comparable. The envelope already supports this: it is uncapped by default
(spec §1.2) and enforces whatever cap it is given.

**Why the cap had to go: memory.** 50 K is enough for stage 1 (a readout of a
1536-d hidden state) and for the stage-3 policy. It is not enough for stage 2:
each remembered fact needs a key in hidden-state space — ~3,000 numbers at full
width (μ and σ), ~130 compressed to 64-d — so 50 K holds roughly 16 to 400
facts. Memory grows in proportion to what the organism knows; any fixed cap is
a ceiling on knowledge. (Arithmetic, not measurement.)

**Expected return on size, per axis (to be tested, §5):**
- Memory / vocabulary: scales with parameters almost directly. The axis where
  "growth pays" should be easiest to show.
- Awareness: saturates early.
- Logic depth: not bought by parameters; depends on §2.1.
- Creativity: unknown.

## 0. Where this comes from

- Rocky, s064: stop trying to make trioron itself a smart edge model. Pair it
  with a tiny model that is fast and often wrong, and let trioron be the part
  that grows.
- `device_conscience_pattern`, `learn_to_use_not_from_principle`: trioron as a
  memory / meta-policy layer over a frozen LLM, not a distilled clone.
- Mode E (s016): primitives + curriculum from a teacher, composed by the
  organism, outlasted its masters (92.1 vs 87.4). Holistic imitation capped
  below the master (s015).
- s021: growth to cap for zero gain. s059: depth forms only under a protocol.
  Growth has to be judged, not assumed.

## 1. Roles

| Part | What it is | What it does | Learns? |
|---|---|---|---|
| **Reflex brain** | Qwen2.5-1.5B-Instruct, q4_k_m GGUF, frozen | Answers every query fast. Need not be correct. Supplies language, binding and working memory (its context window). | No |
| **Organism** | Trioron nest (gradient leaves + Phasecyte) | Reads the reflex's internal state. Learns when the reflex is wrong, remembers corrections, steers the reflex, later chains it. | Yes, continually |
| **Matron** | Fable via headless Claude Code (`claude -p`) from Python | Designs lessons, grades what the environment cannot grade, supplies corrections, judges whether a growth event paid. | No (logs only) |
| **Environment** | A sandbox on Rocky's PC | The world the organism senses and acts in. Grades for free (exit codes, file state). | — |
| **Higher intelligence** | Web lookup (Wikipedia), stronger model | What the organism may escalate to, at a cost. | — |

Measured on this PC (s064): Qwen loads in 3 s, ~1.8 GB RSS, ~21 tok/s on CPU,
exposes final-layer per-token hidden states (1536-d) through `llama_cpp`.

## 2. What trioron is missing (the limbs), and who supplies them

| Missing limb | Evidence | In this design |
|---|---|---|
| Binding / working memory | Satellites give delayed recall only | **Borrowed** from the reflex brain's context. Not built. |
| Open-ended output | Trioron emits a class, not text | **Borrowed**: the reflex writes; the organism chooses among discrete actions. |
| Fact memory (key → content) | Manifold stores per-class (μ,σ), not content | **To build** (stage 2): manifold key in hidden-state space → stored text. Entity archive (s2026-05-06) is the precedent. |
| A sense for the reflex's state | None exists | **To build** (stage 1): `trioron/senses/` module turning 1536-d hidden states into leaf input. |
| Growth that pays | s021 inert growth; arena `divide()` depth never lifted a task | **To build**: growth settled by evidence (§5), judged per event. |
| Crisp state across steps | not³ .86 → not⁴ .77 → not⁵ .64 (s063) | **Open.** Only stage 4 needs it. |
| Depth past ~3 links | hop-3 wall .59–.65, cause unconfirmed | **Open.** Only stage 4 needs it; hop-k with dup is its prerequisite test. |

Reading: two limbs are borrowed, three are engineering on proven mechanisms,
two are genuinely open. Deferring them is not a plan; §2.1 is.

### 2.1 Plan for the two open limbs (and the borrowed one)

The principle: **do not ask the network to be deep. Make each step shallow and
the state between steps discrete.** Analog state decays under iteration;
digital state is restored at every step. Trioron already does one or two hops
well (hop-2 .92, s059); what fails is carrying an analog state through many.

| Limb | Fix | Test | Pass | If it fails |
|---|---|---|---|---|
| Depth past ~3 links | First find out whether the wall is wiring: tied arm with state and evidence duplicated into both dendritic halves (`tied_dup`, s064) | hop-2/3/4, tied vs tied_dup, n=3 | hop-3 moves well above .59–.65 | Wall is not the branch partition; rely on the next two rows |
| Crisp state (internal) | Snap the state to the nearest Phasecyte symbol between steps | notᵏ curve on the dup arm, with and without snap | curve flat: not⁵ ≈ not³ | Internal state cannot be made crisp; use external state only. Caveat: s059 WTA/gated variants were null, but that was before dup, so confounded |
| Crisp state (external) | The intermediate result is written out as a symbol (a token in the reflex's context or a line in the sandbox) and read back in; each step is one hop | hop-k as k single hops with written intermediates | accuracy ≈ p^k for 1-hop accuracy p; flat when each step is checked against the environment | Trioron cannot conduct multi-step logic; stage 4 goes to the reflex brain alone, trioron stays judge and memory |
| Binding (own, later) | A rewritable slot is the stage-2 fact memory used within a task: write key → value, read it back | Falls out of stage 2 | — | Stays borrowed from the reflex |

**Row 1 result (s064, n=3, `outputs/logic_chain_s064_hop_dup.log`):**

| task | tied | tied_dup |
|---|---|---|
| hop-2 | .935±.018 (5 applications, 571 s) | **.982±.007 (1 application, 45 s)** |
| hop-3 | .583±.007 | .706±.015 |
| hop-4 | .573±.005 | .608±.005 |

Reading: the partition was hiding a real capability and is not the wall. With
every slot pair in-branch, hop-2 (a degree-2 polynomial in the evidence) is
solved in ONE quad application, 13× faster. hop-3 moves +.12 but stays far from
solved; hop-4 barely moves. Composing by re-applying a learned step is still
not learned. So the plan leans on the external-state row, and that row now has
a strong, checkable prediction: a 2-hop step at p≈.98 iterated over written
symbols should give hop-4 ≈ .96 and hop-6 ≈ .94, against .61 today.

**Row 3 result — external state PASSES (s064, n=3,
`experiments/progenitor/hop_external_state.py`,
`outputs/hop_external_state_s064.log`):**

A 1-hop organ (start one-hot + map, duplicated input, ONE quad link, 16,272
params) trained on random starts reaches p = .9984±.0003 in 10–12 epochs. Frozen
and iterated with no further training:

| depth k | snap (written symbol) | soft (probabilities) | end-to-end tied_dup |
|---|---|---|---|
| 2 | .998±.002 | .997±.002 | .982 |
| 3 | .998±.002 | .995±.003 | .706 |
| 4 | .997±.002 | .991±.004 | .608 |
| 6 | .997±.002 | .989±.004 | — |
| 8 | .998±.002 | .987±.006 | — |

A 2-hop organ from an arbitrary start (two quad layers, 19,584 params) does NOT
settle: p = .641±.008 after 40 epochs.

Reading:
- Depth is a step count once the state between steps is a symbol. Flat to
  depth 8.
- Snap vs soft: the analog hand-over decays (.999 → .987), the written symbol
  does not. Small here because the step is nearly perfect; the gap is the
  per-depth decay of s063 in miniature.
- **Design rule: one product per step.** A step that needs a two-way product
  (state × evidence) settles in one quad layer; a three-way product does not
  settle even with the depth to express it.
- Caveats: oracle symbols, synthetic task; random 8-slot maps fall into cycles,
  so deep targets are partly forgiving (errors can re-merge), which is why snap
  sits at or above p^k. And the decomposition into 1-hop steps was GIVEN by the
  trainer (intermediate targets) — the organism did not discover it. That is
  the matron's job in this design (Mode E: teach the primitive), and S5 is where
  the organism would have to find decompositions itself.

Consequence: serial grafting (a settled organ's output symbol wired to the next
organ's start input) is now worth making a real substrate operation; this test
is its function-level proof.

Order: `tied_dup` done, external state done (s064); snap on the Logophosis notᵏ
curve after stage 1. The external-state row is the one the incubator design depends
on, because it replaces network depth with iteration plus verification.

## 3. Developmental stages and their gates

Each stage is one of Rocky's axes. A stage passes only against its baseline;
thresholds below are proposals to fix at sign-off.

### Stage 1 — Awareness: predict when the reflex is wrong
- Reflex answers a question. The organism reads the hidden state and predicts
  right / wrong.
- Labels: gold answers from the environment or a QA set. The matron is not
  needed for grading here.
- **Baseline that must be beaten:** the reflex's own confidence (mean token
  log-probability of its answer). If that predicts errors as well as the
  organism does, the organism adds nothing.
- **Gate:** error-prediction AUROC beats the log-prob baseline on held-out
  questions, n ≥ 3 seeds, with the per-seed spread reported.

**Stage 1 RESULT — PASS, modest (s064).** Data: 2,808 questions written by the
matron (four Sonnet helpers, ~635 K tokens) from 1,600 Simple English Wikipedia
leads; the reflex answered closed-book at 29.2 % correct (~1.6 s/item); graded
by gold-string match (audit of 30: ~1 doubtful grade). Scored by AUROC for "the
reflex is wrong" on held-out articles
(`experiments/incubator/stage1_awareness.py`, `stage1_controls.py`; logs
`outputs/incubator/stage1_{awareness,controls}_s064.log`).

| arm (10 article-level splits unless noted) | AUROC |
|---|---|
| reflex's own confidence, min token log-prob (best baseline) | .780±.015 |
| **organism, [h_pre, h_post], k=64, 6,240 live weights (12,834 allocated slots)** | **.817±.010** |
| paired difference organism − baseline | **+.037±.009, 10/10 splits** |
| linear readout on the same 64 components | .810±.011 |
| organism + the reflex's log-probs combined | .841±.011 |
| no-brain control (writer, question length, answer length) | .590±.020 |
| organism on h_pre alone (before the reflex answers; 3 splits) | .774±.011 |
| organism on h_post alone (3 splits) | .763±.008 |

Reading:
- The gate passes on every split, by a small margin. Inside each writer pair it
  holds too (.824 vs .788, .810 vs .772), so it is not a question-style
  shortcut.
- The arm that passes uses both states. h_pre or h_post alone does NOT beat the
  baseline; the arm was chosen after seeing 3-split results, then confirmed on
  10 fresh splits of the same data.
- **The quad leaf adds nothing over a linear readout here** (.817 vs .810). The
  signal is linear in the hidden state. Trioron earns its place in later stages
  (memory, growth), not in this readout.
- The state BEFORE answering already predicts the error about as well as the
  reflex's confidence after answering (.774 vs .780): doubt is available early
  enough to decide whether to look up first.
- The two signals are complementary: combined .841.
- Final layer only (GGUF). Middle layers are reported in the literature to carry
  more; untested here.

Console: `python3 experiments/incubator/console.py` — ask, see answer + reflex
confidence + organism doubt, `/right` `/wrong` to teach by hand, `/look` for
offline Wikipedia, `/learn` to retrain.

**Expectation (Rocky, s064): humans hallucinate too.** A person can state a
wrong fact confidently and only finds out when something does not fit or
someone says so. So an inner doubt signal is expected to be partial, in the
organism as in people, and the design must not depend on it being strong. The
three ways an error gets caught, in order of reliability:
1. **Told** — the matron or a lookup contradicts it (stage 2 stores it).
2. **Something does not fit** — a consequence fails in the environment, or the
   answer conflicts with a stored fact (stages 2–4).
3. **Inner doubt** — this stage. Useful mainly for deciding when a check is
   worth its cost (stage 3), not as a verdict.
A stage-1 pass means "better than the reflex's own confidence", not "knows
when it is wrong".

### Stage 2 — Memory: keep corrections (vocabulary axis)
- When wrong, the matron or the environment supplies the correction. The
  organism stores it and overrides the reflex on a repeat or paraphrase.
- **Gate:** corrected items stay corrected after N later lessons (retention),
  paraphrases are caught, and unrelated questions are not hijacked (false
  override rate). Compared with a plain nearest-neighbour cache on the same
  hidden states.

**Stage 2 RESULT, first cut — PASS against the cache, NOT against a linear
key (s065).** Data: the 2,808 stage-1 questions, each rewritten twice by the
matron (four Sonnet helpers: a light and a heavy rewording; audit: no answer
leaks, 85 verbatim copies redone). The reflex READS each wording without
answering (0.53 s) and gives two states: the last prompt token and the mean
over the question's tokens (3,072 numbers together). Protocol
(`experiments/incubator/stage2_memory.py`, log
`outputs/incubator/stage2_memory_s065.log`): 30 % of articles held out; one
question from each of half of them is taught in its original wording;
positives are its two rewordings; negatives are the OTHER question of the same
article (hard) and all questions of untaught articles (easy), in all three
wordings. "Recalled" = the right fact is nearest AND above the threshold that
lets 1 % of negatives through. "Crowded" = memory also holds the ~1,970
training questions. 5 splits.

| key (numbers per fact) | recalled, 197 facts | recalled, 2,164 facts |
|---|---|---|
| cache, raw states (3,072) | .782±.026 | .600±.029 |
| cache, centred states (3,072) — best cache | .833±.030 | .620±.024 |
| cache, random projection (32) — best small cache | .766±.049 | .557±.060 |
| cache, PCA-32 (32) | .552±.015 | .380±.017 |
| learned LINEAR key (32), 8,224 params | .892±.023 | **.718±.019** |
| **organism quad key leaf (32), 26,112 live weights (41,400 allocated slots)** | **.907±.019** | .681±.035 |

Reading:
- The gate as written passes: the organism's key beats the best cache (+.074,
  +.062 crowded) while storing 96× fewer numbers per fact (64 bytes at
  float16), and beats a cache of its own size by +.14 / +.12.
- **The quad leaf again adds nothing over a linear map** (.907 vs .892; crowded
  .681 vs .718, linear ahead). Same finding as stage 1. What pays is LEARNING
  the key, not the trioron leaf. Trioron has still not earned its place.
- The right fact is the nearest one almost always (.986–.997 for every
  full-size arm). The whole difficulty is the threshold: telling "a taught
  fact, reworded" from "a different question". At 1 % false override half to three
  quarters of the false overrides are the sibling question of the same article
  (siblings are a third of the negatives).
- Load costs every arm about the same: 197 → 2,164 facts drops recall by
  ~.2 at 1 % false override (less at 5 %: .98 → .90). The threshold, not the
  ranking, is what crowds.
- Which state carries the key (3 splits, `stage2_memory_s065_state_{last,mean}.log`):
  the last prompt token alone does nearly as well as both (organism .865 /
  .649 crowded; best cache .852 / .630 — there the learned key only buys the
  96× compression, not accuracy). The mean over question tokens alone is weaker
  (organism .718 / .615; cache .464 / .352). Both together is best.
- Does the quad leaf pack a fact into a SMALLER key (Rocky's question, s065)?
  No. Recall at 1 % false override, 197 / 2,164 facts (key sizes 8 and 16 on
  3 splits, `stage2_memory_s065_e{8,16}.log`; 32 on 5):

  | key size | linear key | quad key leaf |
  |---|---|---|
  | 8 | .626±.042 / .407±.035 | .598±.029 / .365±.020 |
  | 16 | .834±.036 / .616±.017 | .817±.009 / .643±.025 |
  | 32 | .892±.023 / .718±.019 | .907±.019 / .681±.035 |

  The two track each other at every size, within the spread. A 16-number
  learned key already equals the full 3,072-number cache (.841 / .608); 8 is
  too small for either.
- **Drift test (s065, 3 splits, `experiments/incubator/stage2_drift.py`, log
  `outputs/incubator/stage2_drift_s065.log`).** Training articles cut into 5
  lessons taught in order, no replay. ~100 "old" facts stored after lesson 0
  with the keys of that moment; ~100 "new" facts stored after lesson 5. Recall
  at 1 % false override; "key cos" = cosine between an old fact's key as stored
  and as the final leaf would compute it. Both arms are trioron leaves here
  (linear phenotype vs quad).

  | condition | linear: old / new / key cos | quad: old / new / key cos |
  |---|---|---|
  | frozen after lesson 0 | .888 / .916 / 1.00 | .873 / .901 / 1.00 |
  | free (keeps learning) | **.022** / .926 / .32 | **.029** / .906 / .35 |
  | lock, λ = 1 everywhere, strength 1 | .907 / .912 / .97 | .856 / .917 / .96 |
  | lock, strength 30 | .895 / .916 / 1.00 | .868 / .897 / .99 |
  | lock, λ from saliency, strength 1 | .914 / .912 / .96 | .860 / .919 / .95 |
  | free + old facts re-keyed (needs raw states) | .888 / .926 / 1.00 | .877 / .906 / 1.00 |

  Reading: (1) drift is real and total — four lessons of free learning take old
  facts from .89 to .02. (2) The epigenetic lock prevents it at every strength
  tried, in both arms. (3) But simply FREEZING the key leaf after lesson 0 does
  as well: later lessons add almost nothing (new facts .916 frozen vs .926
  free), because one fifth of the data already gives a full-quality key. So the
  lock is not buying plasticity here; a critical period (learn the key early,
  then freeze) is enough. (4) Saliency-driven λ = uniform λ. (5) Linear = quad,
  again. Spread across splits is ±.01–.05, larger than the differences between
  lock settings.
- **Dream test (s065, 5 splits, `experiments/incubator/stage2_dream.py`, log
  `outputs/incubator/stage2_dream_s065.log`).** Rocky's point: the dreaming
  phase can re-key stale facts. Lessons are now five topic DOMAINS (cosine
  k-means on each article's mean read-state), one domain per lesson, no replay,
  sense fitted on the first domain only. Old facts (26–65) belong to the first
  domain, new facts (137–174) to the later ones; one memory, one threshold.
  Recall at 1 % false override, old / new:

  | condition | linear | quad |
  |---|---|---|
  | frozen after the first domain (= live leaf by day) | .881 / .846 | .866 / .835 |
  | free, keys left stale | .000 / .862 | .000 / .832 |
  | free + dream (re-key) | .896 / .849 | .871 / .814 |
  | lock s=1, stale | .793 / .861 | .735 / .841 |
  | lock s=1 + dream | .890 / .834 | .872 / .833 |
  | lock s=30, stale | .869 / .852 | .845 / .838 |
  | lock s=30 + dream | .896 / .844 | .883 / .835 |
  | joint: all domains at once (ceiling) | .908 / .892 | .919 / .870 |

  Reading: (1) the dream re-key works: stale keys go from .000 to the level of
  a never-drifted leaf, and it also repairs the small staleness a weak lock
  leaves (.793 → .890). (2) The domain shift did open headroom: joint training
  beats the frozen leaf on new facts by +.045 ± .020 (linear, 5 of 5 splits).
  (3) Learning the later domains ONE AT A TIME without replay does not reach
  it: free + dream is +.003 ± .040 over frozen on new facts, lock + dream −.002.
  So a re-key-only dream buys safety but no accuracy; the gain needs the dream
  to REPLAY earlier lessons (the joint row is what a full-replay dream would
  give). (4) Linear = quad again. Re-reading a stored question reproduces its
  state exactly (cosine 1.000000 on 40 texts), so the dream needs only the
  question text, not the 3,072-number state.
  Built: `Vee.study` (shadow copy learns by day, live leaf frozen), `Vee.dream`
  (swap + re-key), `remember(..., cues=)`.
- **Replay dream (s065, same test, log `stage2_dream_replay_s065.log`).**
  The dream now replays every kept lesson together before the re-key. Old /
  new, linear | quad: replay 15 passes once at the end .924 / .890 | .901 /
  .851; 60 passes .921 / .898 | .898 / .846; a dream after EVERY lesson (15
  passes over all lessons so far) .936 / .897 | .902 / .846; joint ceiling
  .908 / .892 | .919 / .870; frozen .881 / .846 | .866 / .835. Per-split gain
  over the frozen leaf, linear: new +.044 ± .046 (15 passes), +.052 ± .036 (60),
  +.051 ± .044 (nightly); old +.04 to +.055. So for the LINEAR leaf the replay
  dream reaches the joint ceiling, 15 passes are enough, and nightly dreams
  lose nothing against one big dream. For the QUAD leaf the gain on new facts
  is +.01 to +.02, inside the noise, and below its own joint row (+.035 ±
  .062): the quad leaf benefits less from replay and is no better than linear
  anywhere in this test. Built: `Vee.dream(H_cues, H_lessons)`; `learn_keys`
  and `study` keep lesson questions when given `texts=`.
- **Parameter counts corrected (s065).** `Organism.n_params()` counts the
  arena's ALLOCATED slots (cells × 69), not live weights. The quad key leaf has
  26,112 live weights (not 41,400), about 3× the linear key, not 5×; the stage 1
  leaf has 6,240 (not 12,834). No conclusion changes.
- **Consolidation test (s065 script, run finished s066; `stage2_consolidate.py`,
  log `stage2_consolidate_s065.log`, 3 splits).** Question: is a fact cheaper
  in trained WEIGHTS than as a row in the key table? ~2,170 facts in memory,
  same h numbers per fact in every arm, recall at 1 % false override (right
  fact nearest in brackets). Table (learned linear key, one stored row per
  fact): h=8 .407 ± .035 (.766), h=16 .616 ± .017 (.946), h=32 .727 ± .013
  (.986). Rows trained as cosine prototypes, key leaf frozen: .320 / .607 /
  .701. Rows + key leaf trained: .338 / .652 / .707; paired against the table
  −.069 ± .036, +.036 ± .010, −.019 ± .030. Trioron quad leaf with one output
  cell per fact (h interior cells), scored by top probability: .030 / .082 /
  .469 (right fact nearest .395 / .661 / .901); by top logit .017 / .027 /
  .139. Reading: consolidating facts into weights does not beat the one-shot
  table (a small gain at 16 numbers only, 3/3 splits; a loss at 8). The trioron
  leaf is far behind, mostly at the THRESHOLD: at h=32 it ranks the right fact
  first .901 of the time but its confidence does not separate stored from
  unstored questions. Caveat: the leaf arm differs from the row arms in two
  things at once — the quad leaf and the readout (unnormalised logits /
  softmax, against a cosine score for the rows) — so this run does not say
  which one costs the recall.
- **Readout check (s066; `stage2_consolidate_cos.py`, log
  `stage2_consolidate_cos_s066.log`, same 3 splits).** Readout held at cosine,
  trunk varied; h = 8 / 16 / 32. Linear table .407 / .616 / .727. The SAME
  softmax-trained leaf read by cosine over its interior codes: .108 / .310 /
  .473 (softmax top probability .030 / .082 / .469). Quad cells trained like
  the linear key (contrastive), row = code: .389 / .600 / .726 (paired vs
  linear −.018 ± .069, −.016 ± .049, −.001 ± .044). Quad rows + trunk trained
  with the cosine classifier: .355 / .606 / .706. Reading: the loss was neither
  the quad cell nor the readout alone — it was TRAINING the leaf as a softmax
  classifier over ~2,170 output cells. Trained with a cosine objective the
  quad cells tie the linear key at every size; swapping only the readout on
  the softmax-trained weights recovers little. All of this is batch training
  of a fixed-size leaf (Rocky s066: not how the organism is meant to learn).
- **Training-regime correction (Rocky, s066).** Every leaf above was trained
  as a fixed-size network on shuffled batches of the whole set. That tests the
  quad cell as a static layer, not the organism: "it is never a batch at the
  beginning — it is an embryo and a baby that keeps absorbing". A batch-trained
  leaf is from now on a CONTROL arm. For fact memory Rocky names the Phasecyte
  (spec §10) as the learner; feed rate against the boundary (manifold / dream
  trigger) is a first-class setting, because overfeeding loses information.
- **Phasecyte pilot (s066; `pilot_collect.py`, `pilot_stream.py`, logs
  `pilot_stream_s066.log`, `pilot_stream_nocap_s066.log`; 3 splits).** 40 facts
  (20 articles × 2 questions, so each has a same-subject sibling), 50 matron
  rewordings each (12 % are > .9 similar to another wording of the same fact),
  10 held out per fact; 20 untaught questions as negatives. Facts arrive in
  four waves of 10; nothing is frozen and earlier waves are never shown again.
  One `PhasecyteLeaf` on a 64-number PCA sense of the reflex's read state
  (sense fitted on other articles). Top-1 on held-out wordings after wave 3,
  per wave w0..w3, mean; W = questions per boundary:
  - class cap 43 (default): W=25 .92 .77 .53 .31 (.63); W=50 .92 .87 .80 .50
    (.77); W=100 .93 .93 .98 .77 (.90); W=200 .89 .86 .99 .67 (.85); W=400
    (a whole wave in one boundary) .00 .20 .43 .17 (.20), 8 classes for 40
    facts; W=100 heard 3× .92 .60 .23 .43 (.55).
  - no cap: W=25 .91 1.00 .99 1.00 (.976 ± .004), 142 classes; W=50 .92 .99
    1.00 1.00 (.978 ± .009), 67 classes; W=100 .91 .97 .98 .97 (.954 ± .008),
    46 classes; W=100 heard 3× .94 .96 .99 .97 (.966 ± .027), 139 classes.
    Untold (no labels, classes named afterwards by majority) .92–.96.
  - controls: running class mean (64 numbers per fact) .993 ± .003; s065
    one-shot key table (32 numbers, ONE stored wording) .983 ± .005.
  Reading: (1) the phasecyte KEEPS LEARNING — wave 0 stays at .91–.94 after
  three later waves at every working feed rate, and all 40 facts get a class;
  (2) overfeeding is real — one boundary per 400 questions leaves 8 classes;
  (3) underfeeding per boundary is not harmful by itself, but every boundary
  divides, so classes multiply (3.5 per fact at W=25) and the default 43-class
  cap then blocks LATE facts, not early ones; (4) false recall is not the
  problem (threshold at 5 % of untaught questions costs < .01); (5) it does
  not beat either control, and its store is larger than both (per class a
  template plus a rolling buffer of member samples); (6) the shipped
  `PhasecyteLeaf.names()` is wrong for this use — it reads the lifetime label
  ledger, which inherits counts through divisions (W=100: .54 against .90 for
  the same organism named by its current buffer tags; the weakness is recorded
  in `mixed._majorities`, s034b). Caveats: 40 facts, one sense size, matron
  wordings with near-duplicates, thresholds set on the test negatives, class
  count still rising when the stream ends (no self-arrest seen). Not done: the
  two-leaf nest arm; a merge of same-fact classes; larger fact counts.
- Not yet tested: (a) the replay dream end to end through the reflex (re-read
  cost ~0.5 s per kept question: ~50 min for the 5,900 lesson rows here);
  (b) a per-fact threshold (each fact's own spread — the manifold σ the design
  names); (c) end-to-end override accuracy with the reflex (stage 3's number);
  (d) bytes per parameter for the memory ceiling. Thresholds here were set on
  the test negatives, the same for every arm; a deployed threshold must be set
  on training articles.

### Stage 3 — Steering and escalation
- The organism picks an action: trust reflex / recall memory / look up /
  ask the stronger model. Steering is at text level (inject recalled memory
  into the reflex's prompt); soft-prompt injection is a later option.
- **Gate:** system accuracy beats Qwen alone, and beats always-escalate on
  cost at matched accuracy.

### Stage 4 — Logic: chain reflexes and check consequences
- The organism sequences several reflex calls and verifies each step against
  the environment.
- **Prerequisite:** hop-k with dup (s063 NEXT-0) — done s064, see §2.1.
- **Gate:** accuracy vs chain depth, compared with the reflex doing the same
  task in one shot.

**Concrete form: the self-check (Rocky, s064).** "I said X. If X then Y. Y
holds, and so does Z. But Z cannot hold with X. I am very sure of Z, so X must
be wrong — I need to check." As steps over written symbols (§2.1):
1. *Derive:* from claim X, write down a consequence Y (the reflex proposes it,
   or a stored rule supplies it).
2. *Compare:* does Y, or X itself, clash with a stored fact Z? One claim
   against one fact is a two-way comparison — one product per step.
3. *Weigh:* which is held more firmly? Firmness is a per-fact strength, the
   natural use of λ (how locked a memory is) and of how often it was confirmed.
4. *Act:* weaken the less firm one and raise the "check it out" action
   (stage 3).
Start with facts where a clash is checkable by rule (one value per subject:
birth year, capital, author), then free text. Gate: planted false claims that
contradict a firm stored fact one or two steps away are caught, at a low false
alarm rate, more often than the reflex catches them when asked "is this
consistent?". Risk: step 1 depends on a 1.5B reflex producing sound
consequences.

### 3.1 Self-hosting milestones (Rocky, s064: "how Linux was compiled on Minix")

The reflex brain and the matron are the host the organism is built on. The aim
is that it comes to stand on its own, one borrowed part at a time, the way
early Linux replaced its Minix parts.

| # | Borrowed today | Replaced by | Milestone reached when |
|---|---|---|---|
| S1 | Reflex's knowledge of facts | Organism's fact memory (stage 2) | Taught items are answered from memory with the reflex's answer ignored |
| S2 | Matron's grading | Organism's own error prediction (stage 1) + environment checks | Lessons run with no matron grading call and the same learning curve |
| S3 | Reflex's context as working memory | Rewritable slot (§2.1 binding row) | A value written in step 1 is used correctly in step k without being in the reflex's prompt |
| S4 | Reflex's multi-step reasoning | Organism's stepwise chain over crisp state (§2.1) | Stage-4 tasks solved with the reflex called only to read input and write output |
| S5 | Matron's curriculum | Organism choosing what to study next (curiosity / learning progress, cf. s019 arm B) | Unbuilt; listed so it is not forgotten |

**Standing measure: the Qwen-off test.** At each checkpoint, rerun the held-out
set with the reflex's answers withheld (the reflex still tokenises input and
renders output). Report the fraction of items still handled correctly. Plotted
over lifetime beside parameter count, this is the "outgrow" curve.

**Expected limit:** language itself. The organism has no language faculty and
no path to one at RAM-bounded sizes, so reading and writing text stay with the
reflex. Self-hosting is expected for memory, judgment and possibly logic.

### Stage 5 — Creativity (Rocky's definition, s064)

**Definition:** the ability to create a thing that did not exist before, by
finding an action along one of the SCAMPER routes: **S**ubstitute, **C**ombine,
**A**dapt, **M**odify, **P**ut to another use, **E**liminate, **R**everse.

- The seven routes are the organism's creative **action set** (a discrete
  choice, which is what trioron emits). The reflex carries the route out in
  text or code; the organism chooses the route and the material from memory.
- **Task form:** a known thing (a working snippet, a remembered fact, a tool)
  and a goal it does not meet. The organism picks a route (and, for Combine, a
  second item from memory); the reflex produces the result.
- **Valid** = the environment accepts it (code runs and meets the goal's
  check), or the matron judges it where no check exists.
- **New** = not in the organism's memory or lesson history. Its own fact
  memory supplies the test: distance from every stored key. New means new to
  the organism, not to the world.
- **Gate:** rate of valid-and-new products per attempt beats (a) a random
  route choice and (b) the reflex asked to "be creative" with no route.
- **Expected to scale with memory:** Combine and Substitute need stored
  material, so this axis should rise with what the organism knows. That is a
  testable form of "growth pays".
- Honest limit: the created content is the reflex's writing. The organism's
  contribution is choosing the route and the material, and recognising what is
  new. Depends on stages 2 and 3.

## 4. The environment (sandbox)

The PC is structured state natively: paths, text, markup, exit codes. That
avoids the pixel ceiling trioron hit on Atari and CIFAR.

- **Where:** a sandbox inside WSL (Rocky, s064).
- **Senses:** read-only, on a whitelist of folders.
- **Actions:** confined to a scratch directory; a fixed small action set.
- **Running code:** the reflex writes it; the organism decides whether to run
  it and reads the outcome (exit code, output, files changed). Each run is
  isolated with bubblewrap: no network, time and memory limits, write access
  to the scratch directory only, home folder not visible. (bubblewrap is not
  installed yet: `sudo apt install bubblewrap`.)
- **Wikipedia: offline.** Simple English, text only
  (`~/data/wikipedia/wikipedia_en-simple_all_nopic_2026-09.zim`, 1.0 GB, read
  with `libzim`). Offline keeps the sandbox fully network-free, makes runs
  repeatable, and has no rate limits. Plain language suits a 1.5B reflex. Full
  English (~50 GB) does not fit comfortably: the host disk has 65 GB free
  (the 907 GB WSL reports is the virtual disk's ceiling).
- **Lookup is an action with a cost** (stage 3); what it returns can be stored
  as a corrected fact (stage 2).
- **Out of reach:** mail, credentials, browser profiles, anything private.
  Whatever the organism reads can reach logs, matron calls and the public repo.
- Access widens only by Rocky's decision.

## 5. Growth must pay

- A growth event (new leaf, new link, new Phasecyte class) is provisional.
- It settles only if a held-out measure improves and the improvement persists
  (Numa, not Mima). Otherwise it is reverted or left unlocked.
- Standing record per event: params before/after, which axis moved, by how
  much. The headline figure is each axis against total parameter count over
  the organism's lifetime.

### 5.1 The memory ceiling (the only size limit)

- The limit is **resident memory of the whole incubator**, measured, not a
  parameter count. WSL has 7.6 GB RAM + 2 GB swap (s064).
- Standing costs measured s064: system ~1.6 GB, reflex brain ~1.8 GB. The
  matron's Claude Code process and a safety margin are still to be measured.
- Rule (approved s064): growth is refused when projected total use would exceed
  85 % of RAM; swap is never counted as budget (swapping stalls the PC, which
  is the system-level threat). A watchdog checkpoints and pauses the organism
  if actual use crosses the line.
- The ceiling converts to parameters only by measurement: bytes per parameter
  in training (weights, edge indices, λ buffers, anchors, gradients, optimiser
  state) is unmeasured for a large arena. Measure it in stage 1 before quoting
  a parameter figure.
- **Expected real limit: time, not memory.** The arena forward is ~485 µs at
  27 K params on this CPU (s050); a sparse edge list at millions of parameters
  will be slow to train long before RAM runs out. Dense export serves fast but
  does not learn.
- Raising the WSL allocation (`.wslconfig`) raises the ceiling with no design
  change.
- No cap does not mean free growth: §5 still applies. Size is allowed, not
  rewarded.

## 6. The matron loop

- One matron call per **lesson**, never per sample (seconds of latency).
- Lesson = (a) matron writes a batch of items for the current stage, (b) the
  reflex answers, (c) environment grades what it can, matron grades the rest in
  one call, (d) organism trains, (e) matron reads the summary and decides:
  repeat, advance, or approve / reject a growth event.
- Every matron output is logged and replayable; a rerun can use the log
  instead of new calls.
- The loop checkpoints and resumes, so a subscription rate limit pauses it
  without loss.
- The matron teaches primitives and curriculum (Mode E). The organism is not
  trained to reproduce matron text.

## 7. Decisions for Rocky

1. Stage-1 data: an existing QA set with gold answers, sandbox-generated
   questions, or both? (Recommended: both; QA set first because it is fastest.)
2. Where the code lives: a new organism under `trioron/organisms/` plus a sense
   under `trioron/senses/`, with the incubator loop world-side in
   `experiments/`. This needs a spec §9 partition update before code.
3. DECIDED s064: creativity = SCAMPER routes (stage 5).
3b. DECIDED s064: incubator organism exempt from the Phase 1 cap; 85 % RAM
   ceiling. Spec §6 and the project CLAUDE.md carry the exemption line.
4. Check Anthropic's terms on using model output as training signal before any
   matron-authored data is committed to the public repo.

## 8. First build step (after sign-off)

Stage 1 only: the hidden-state sense, one leaf, the log-prob baseline, the
gate. No matron loop, no growth, no sandbox actions until stage 1 passes.
