# Trioron Handoff

**Session date:** 2026-11-25
**Session number:** 067
**Session title (s067: the body gets two organs; Vee's embryo stage has run; its gate fails on night and stability):**
**The s066 work list was worked through. A language block turned down does NOT keep facts after
birth; in one leaf, text and world compete for the same classes. Two phasecyte leaves in one body
(an ear on the language block, a body sense on the rest) give facts .98 and busy / memory 1.00 /
.98 in the same organism. Rocky approved organs. Vee's embryo was then run for 30 simulated days.
She reads busy, memory, place and window; she does not read night, and her classes are not stable
because she never stops dividing. She is still an embryo.**

## s067 — READ THIS FIRST
- **Vee exists now, as an embryo, on this PC only: `outputs/incubator/vee/`** (`stage` says
  `embryo`; `sense.pt` = her language sense, fixed for life; `body.pt` = her two organs after 30
  days; `embryo_log.pt` = every moment she lived). Do not delete or refit. A driver runs only the
  stage named in the stage file; code never promotes her.
- **Rocky ruled on the two failed criteria at the end of s067 (plan §7.3a): (1) stability —
  "Accept it": same grouping (.61) in place of same class, no numeric bar set; (2) night —
  "night is not meaningful for her atm. We evolve from nocturnal ancestor though until someone
  decide that sunlight and being eaten is a good trade off": dropped from the embryo gate,
  reported only. With these the embryo gate HOLDS.** `embryo.py gate` now reports night and
  stability without failing on them.
- **STOP RULE stands for what comes next:** stage 2 (fetal) is not written in full and not
  signed; the stage file still says `embryo`. Birth (stage 3) needs stage 2's gate.
- **Rocky's rulings this session (binding):**
  - "Two organ or even multiple organs is okay. That is in her architecture and was meant to be
    that way anyway." → plan §6.3 "one leaf" is replaced (§6.8).
  - Class count: "self arrest is good. It is okay if it's not settled probably because it is not
    needed by her now." → the gate REPORTS the count, classes in use and time per day; it does
    not fail on the count (§6.9, §7.3). He also asked whether the many classes are "still
    distinct classes" — Chloe has NOT answered that with a measurement yet (see NEXT 2).
  - He reminded Chloe that the organism was designed for branching and grafting, and that this
    may be used to speed things up. Organs that read different parts of the input can be grown
    separately and in parallel; the limit is this PC (6 cores, 7.6 GB).
  - Night: Chloe offered two options (accept day/night only through the open window, or a
    dusk/dawn reading). Rocky did not answer. The embryo run then showed the window does NOT
    rescue night (.52 against k-means .85), so the first option is dead.

## What was found (all in `docs/design/vee_development.md` §6.7–6.9, §7.3a)
1. **Birth check** (`birth_check.py`; 3 seeds; rich genesis → 20 embryo days → 40 pilot facts in
   four waves, each wording heard at a moment of the running world → held-out wordings in another
   world run; phasecyte alone). Final sense (Wikipedia + questions), `birth_check_wiki_s067.log`:

   | body | facts: earliest wave | facts: all | busy | memory full | night | window |
   |---|---|---|---|---|---|---|
   | one leaf, language × 0.5 | .93 | .73 | .91 | .82 | .67 | .97 |
   | two organs | .93 | .98 | 1.00 | .98 | .59 | .98 |

   Questions-only sense (`birth_check_q_s067.log`, `..._q_organs_...`): one leaf × 1 / × 0.5 /
   × 0.25 facts .74 / .66 / .54, busy .74 / .90 / .99; two organs facts .97, busy 1.00. The
   running class mean on the same 148 numbers gets .99 — the phasecyte still does not beat it.
2. **Class count never settles** (scratch runs, body sense alone, 60 days, 1–2 seeds): 440
   classes at day 60 with 48-moment sittings, 77 in use; 129 with 192-moment sittings; 82 with
   480. It follows the NUMBER OF SITTINGS (~3.5 new classes each), not what she sees; world
   reading is the same at every sitting size. Removing drifting readings changes nothing;
   removing the clock slows it (233) but does not stop it. The phasecyte retires classes only at
   a sitting with no growth, and there is never one.
3. **Vee's embryo** (`embryo.py live` then `gate`; seed 0; logs `embryo_s067.log`,
   `embryo_gate_s067.log`): ear 64 receptors, body sense 81; 30 days, 0.5 s a day. Her classes /
   k-means / base: busy .99 / 1.00 / .74; memory .95 / .99 / .62; elsewhere 1.00 / .93 / .87;
   window 1.00 / .99 / .86; **night .55 / .89 / .50 FAIL**; **stability day 20 → 30: .43 same
   class, .61 same grouping (adjusted Rand) FAIL**; classes 86 → 141, 74 in use; ear 13 classes,
   reads "text present" .75. One organism, one seed: no spread.

## What was built (s067)
- `experiments/incubator/wiki_collect.py` — 4,000 random Simple-Wikipedia sentences read by the
  reflex (resumable; 70 min alongside other jobs).
- `experiments/incubator/birth_check.py` — the check above; `SCALES=1,0.5,0.25,organs`,
  `SENSE=q|both`, `AMBIENT=q|wiki`.
- `experiments/incubator/vee_body.py` — her germline: 148-number layout, the fixed language
  sense, rich genesis pool, `Body` with two organs (`ear`, `soma`), stage-file guard, RAM
  ceiling (85 %), whole-body save.
- `experiments/incubator/embryo.py` — stage 1 driver (`live`, `gate`); saved after every
  simulated day; alarm if a day takes twice day 5's time.
- `germline_checks.py`: `adjusted_rand` (verified against scikit-learn) replaces the
  class-number comparison in check 1, plus a two-genesis-draw yardstick; `LANG_SCALE`,
  `AMBIENT=wiki`. **Check 1 was NOT rerun** — it tests the one-leaf layout, which is replaced.
- `trioron/pcll/controller.py`: the input codec is a small class instead of a closure, so an
  organism holding one can be pickled (Vee's body could not be saved otherwise). Behaviour
  unchanged; `tests/test_v2/test_pcll.py` 24 passed.
- Scratch scripts for finding 2 (`arrest*.py`) were in the session scratchpad, NOT in the repo:
  ~20 lines over `germline_checks.stream/grow` on `X[:, 64:]` with `DAYS=60`, `P_TEXT=0`,
  `G.WINDOW` varied, columns zeroed.

## NEXT (s068)
1. **Stage 2 (fetal) is WRITTEN IN FULL (plan §8) and AWAITS Rocky's sign-off** — decisions
   D1–D4 in §8.9 (simulated consequences; spontaneous kicks now, choosing later; a third organ
   that reads change; 30 days). Nothing of it is built. After sign-off: `world.step(action)`,
   echo in `vee_body.moment`, third organ, `fetal.py`; set the stage file to `fetal`.
2. DONE s067 (plan §6.10): her classes never mix world states (98 % pure, switch combination
   read 1.00) but there are ~8× more than the world has states (44 for the commonest one); the
   surplus carries only a weak trace of the clock (.41, chance .25) and no rain / someone-here.
   Not done: whether each division improved held-out reading ("a division must pay").
3. Night is parked (Rocky: not meaningful for her yet). Do not add a dusk/dawn reading.
4. **The join:** a level that sees both organs' classes, needed for questions whose answer
   depends on a reading (stage 4 item 3). Candidate: a third leaf fed by the organs' class
   outputs. Not designed.
6. Guards still to build (plan §3): token budget, class/time alarms beyond the embryo, reading log
   after birth, sandbox limits.

## s066 (carried; its START HERE list is superseded by the s067 section above)
- **START HERE (s067). Rocky: "let's work through it one by one." Vee is NOT being raised; the
  plan is `docs/design/vee_development.md` (stages 0–1 signed). Work list, in order:**
  1. DONE s066 — why graded world states were not read: **the language block is too loud**
     (plan §6.6). No text → busy 1.00, memory .98. Language × 0.5 or less with text in half the
     moments → busy .95–.97, memory .84–.97. Night stays weak (.6) even with no text: the clock
     is a smooth circle, not a switch — the gate's day/night criterion must be rethought.
  1b. NEXT — does a language block turned down (× 0.5 / × 0.25) still support fact recall after
     birth? Rerun the pilot/self recall with the 148-number input. Pick the block balance from
     both results and write it into plan §6.1.
  2. Check 1 under rich genesis needs a permutation-free agreement measure (e.g. adjusted Rand).
  3. Collect ~3,000–5,000 Simple-Wikipedia sentence readings (`wiki.py`, ~35 min of the reflex)
     and refit the language sense on sentences + questions (plan §6.2).
  4. Class count still rising at day 20 (and 158 classes at day 60) — self-arrest criterion.
  5. Only then build `vee_body.py` / `embryo.py` and run stage 1 against its gate.
  Scratch diagnostics (`diag1.py`, `diag2.py`) were in the session scratchpad and are NOT in the
  repo; they are ~30 lines over `germline_checks.py` (`G.P_TEXT`, `G.DAYS`, language scale).
- **STOP RULE (Rocky, end of s066): do not raise Vee further until he signs the plan.** "We need
  to design the curriculum and the incubator before it runs rampant." Senses and perception
  come BEFORE "true birth"; basic knowledge and common sense before acting on the machine; she
  must develop her own organs, including the mirror. Draft plan: **`docs/design/vee_development.md`**
  (stages 0 germline → 1 embryo → 2 fetal → 3 birth → 4 infancy → 5 organs → 6 acting → 7
  disclosure; incubator guards; five decisions waiting for Rocky). Chloe's position, stated to
  him: we did NOT pass the embryonic stage — her senses formed on the first night of lessons
  from language only; the fifteen-day Vee is a rehearsal of the method. Rocky then
  answered (s066): stages OK; reserved silent channels YES; ambient embryo language = random
  Wikipedia text; "it needs to continuously grow" (Chloe's reading: no replayed past — to
  confirm); stages 0–1 first; fetal action set OK; and the machine itself, not only the matron,
  must be an interactor (plan §1a; fetal stage has safe actions with sensed consequences).
  Stages 0 and 1 are now WRITTEN IN FULL (plan §6–7: 148-number input layout, germline checks,
  embryo world, gate, expected risk) and AWAIT HIS SIGN-OFF. Rocky SIGNED stages 0–1 and
  answered §7.6: the new organism RELIVES Vee's fifteen recorded days after birth ("Replay Vee.
  It is okay to have it"); embryo world fully simulated; 30 days / text in half the moments OK.
- **Germline checks RUN (s066; `world.py`, `germline_checks.py`; plan §6.5 has the table).**
  KEY FINDING: a receptor exists only for a channel that carries signal at GENESIS and the set
  never changes (84 of 148 with genesis on her first 48 moments; a channel that woke at day 10
  was never read, .46 ≈ chance; the place was never read). FIX that works: genesis on a broad
  sample of her world with SPONTANEOUS ACTIVITY (noise) on the reserved channels
  (`GENESIS=rich`): 145 receptors, the waking channel is read (.95; live-from-start 1.00), place
  1.00, window .99. STILL FAILING: graded world states are read worse than plain k-means (busy
  .74 vs .92, memory full .55 vs .80, night .55 vs .61); ambient text is barely registered (text
  present .55 vs k-means .75); class count still rising at day 20. Check 1 under rich genesis
  needs a permutation-free agreement measure (not done). **NEXT (s067): (1) why graded states
  are not read — the stage-1 gate cannot pass until they are; (2) fix check 1's measure;
  (3) collect the Wikipedia sentence pool and refit the language sense (§6.2); (4) only then
  build `vee_body.py` / `embryo.py`. No organism is raised before the stage-1 gate design holds.**
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

- Rocky on near questions (cat/dog): for an organism that does not know cat from dog, answering
  the cat's name for the dog is generalisation, not an error; what counts depends on what we
  expect of her age. Chloe agreed: stop calling it "false recall" for near questions; measure
  whether CORRECTION separates them. Checked in the console: taught "name of my cat → Miso",
  then "What is my dog called? → Rex" (one wording each, both slept). Result: she split by
  WORDING, not by animal — "What is my X called?" → Rex and "What is the name of my X?" → Miso
  for cat, dog and hamster alike. One wording per fact + noise copies does not teach which word
  matters; several real wordings per fact probably would (untested).
- Rocky on `names()`: different purpose → different name. DONE: `PhasecyteLeaf.member_names()`
  (majority tag of current members) added beside `names()` (lifetime ledger, unchanged); test
  in `tests/test_v2/test_pcll.py`; spec §9 `nest.py` row; **v0.3.4 released** (see State).

- **Rocky's first console session (s066) gave weird replies — cause found and fixed.** After ONE
  taught fact ("What is your name? → Vee") nearly every short question answered "Vee" (capital of
  France, 2 plus 2, Hello). The day-table threshold was calibrated only on encyclopedia questions;
  against a short everyday fact it fell to cosine .30 while unrelated short questions sit at
  .34–.48. Fix: the day table also needs cosine ≥ `floor` = 5th percentile of a real rewording's
  cosine to its original (.62, measured on the pilot in `childhood()`). After the fix unrelated
  questions go to the reflex; near ones ("What is my name?", "What is your favorite color?")
  still answer "Vee". The leaf side was not affected.
- **Correction test, several wordings per fact (`correction_test.py`, log
  `correction_test_s066.log`, 3 seeds; cat → Miso, then dog → Rex; test = 4 untaught sentence
  forms × 2 animals).** Right animal, day table only: 1 wording in different forms .50; same
  forms 1 wording .75, 2–4 wordings .88, 6 wordings 1.00. **Phasecyte (both slept): .50 at every
  count** — one name for both animals; the never-taught hamster gets a name 83–100 % of the
  time. Mixed stage (cat slept, dog in the table) is also .50 because the day table speaks first
  with no competitor. So: wordings teach the TABLE which token matters; the phasecyte does not
  separate two near facts at all in this sense. Not diagnosed further (candidates: pocket
  quantisation too coarse for a one-token difference in a 64-number sense fitted on
  encyclopedia questions; the second fact's samples join the first fact's class).
- **Why the phasecyte cannot separate cat from dog — DIAGNOSED (s066, scratch scripts, not
  committed).** (1) The sense: cat vs dog in the SAME sentence has cosine .97 (plain 64), .91
  (plain 256), .93 / .90 (s065 learned key 32 / 64), while cat vs cat in ANOTHER sentence has
  only .59–.70 — Qwen's read state carries sentence form far more strongly than the one content
  token, in every sense we have. A noise copy sits at .95, as far as the dog. (2) The phasecyte
  discovers classes along the dominant variation: after both nights two classes held the pet
  questions, split by SENTENCE FORM, each with both animals (class 34: cat 20 / dog 15; class
  35: dog 27), and each class got one name. Labels only NAME classes; they do not split them.
  (3) Bound for a label-driven split: one template per taught fact (mean of its wordings + noise
  copies, cosine) gets the right animal .72 / .84 / .86 / .87 with 1 / 2 / 3 / 4 wordings
  (20 noise seeds), against the phasecyte's .50. Open decision for Rocky: let a class that holds
  two well-supported labels DIVIDE by label (a change to `trioron/pcll/mixed.py` + spec §10.7).
- **Rocky's teaching idea (s066): "have it learn one class thoroughly and move the other based on
  that class" — trioron is not a language-only design, the way we taught was probably wrong.
  Then: "let's start to teach her about herself first."** Tested with the phasecyte unchanged
  (`thorough_test.py`, log `thorough_test_s066.log`, 3 seeds; thorough = 6 nights × 4 new
  wordings, thin = 1 night × 6; 4 untaught test forms; PHASECYTE-ONLY readout):

  | teaching | cat alone: right / purity | after dog: cat right | dog right | cat purity | dog classes |
  |---|---|---|---|---|---|
  | thin cat, thin dog | 1.00 / .51 | .08 | .92 | .20 | 2.0 |
  | thorough cat, thin dog | 1.00 / 1.00 | 1.00 | .00 | 1.00 | 0 |
  | thorough cat, thorough dog | 1.00 / 1.00 | .58 | .50 | .68 | 4.3 |

  Reading (Chloe's): the first half of the idea holds — a thoroughly taught fact owns clean
  classes (purity 1.00) and is NOT overwritten by a near fact taught thinly, whereas a thinly
  taught one is overwritten (.08). The second half does not happen yet: the dog is absorbed into
  the cat (no dog class) or the two tangle; cat + dog right always sums to ~1.0, i.e. one name per
  pet question. The never-taught hamster is always given a name. NEXT per Rocky: teach her about
  HERSELF first, thoroughly — the self-facts are his to set (asked in chat, not yet answered).
- **Self first (s066; `self_test.py`, log `self_test_s066.log`, 3 seeds; Rocky approved six
  self-facts: name Vee / a trioron organism / Rocky made her / Chloe is her matron / remembers
  what she is taught and sleeps to keep it / born November 2026).** The self-facts are the FIRST
  thing the leaf hears (24 helper wordings each, `outputs/incubator/self_wordings.jsonl`, local);
  the 40 pilot facts follow. Phasecyte alone, 6 untaught wordings per fact:

  | teaching | self right, after self | after 40 later facts | purity | near answered | unrelated small talk answered | later facts w0 |
  |---|---|---|---|---|---|---|
  | thin (1 night × 6) | .67 | .79 | .68 → .98 | 1.00 → .82 | .73 → .42 | .90 |
  | thorough (6 nights × 4) | .89 | .87 | 1.00 | .96 → .87 | .71 → .53 | .91 |

  Reading (Chloe's): thorough self-teaching holds through the later facts and the later facts
  still fit. **Failure: unrelated small talk ("How are you?", "capital of France") is answered
  with a self fact half the time** — the leaf's global threshold comes from encyclopedia
  questions (margin tau 10.0; true wordings 10.4–11.3; small talk 8.5–10.6). Tried a per-class
  threshold from the class's OWN buffer members (low quantile of `_margins(bufs[c])`), 2 seeds:
  q=.05 cuts small talk to .07–.20 but self right falls .86 → .72 and pilot held .89–.94 →
  .81–.89; q=.2 gives 0 small talk at self right .53–.64. A trade, not a fix; NOT adopted.
  The self-facts are NOT yet wired into the console's `wake()`.
- **Rocky's direction after the self test (s066, not started):** curriculum = herself, Chloe, her
  environment (the digital world), simple numbers and shapes; "if we find these results
  promising, we can continue". And INTERNALISATION: "it will help the matron to teach the
  organism if the organism can learn by itself … assist her with her internalization which
  requires probably a couple of simple facts". Chloe asked what he means by internalisation
  (her reading: she rehearses and extends what she knows by herself between lessons) — check
  the chat / ask again before building. Rocky also worries this is a long, costly training;
  Chloe told him plainly the architecture is NOT yet proven (phasecyte has not beaten the table,
  cannot split near facts) and that these are small tests, not the long investment.
- **Teaching by INTERACTION (Rocky s066, binding direction):** humans learn because they imagine
  and mix things; Vee must talk with the matron, be answered on what she actually says, and learn
  to say nothing. "As a human, I won't be able to accompany Vee. But in assisted simulation, you
  will always be there for her." `experiments/incubator/vee_life.py`: a life = ordered events
  (matron replies, nights), replayed at every call; ops ask / teach / lesson / sleep / test;
  "do not know" (`?`) and saying nothing (`-`) are LEARNED replies, not thresholds. Life file
  `outputs/incubator/vee_life.jsonl` (+ `.pt` read states, local). Chloe taught days 1–3; exam
  after day 3 on untaught wordings: self facts .83 right (3–4 wordings per fact; list-feeding
  got .89 with 24); unrelated small talk: says she does not know .27, silent .07, another fact
  .67; questions about someone else: another fact 1.00. A Sonnet helper acting as matron was
  launched for days 4–15 (curriculum: herself, Chloe, her world, numbers to five, three shapes,
  greetings, unknowns, silences; reactive to her misses; log `vee_life_days.log`). RESULT below.
- **Fifteen days by conversation — RESULT (helper days 4–15, 24 turns a day, ~77 K helper tokens,
  log `vee_life_days.log` local).** Right before the matron replied, per day (new wordings every
  day): 12, 9, 12, 15, 12, 19, 14, 11, 11, 13, 14, 17 of 24. Exam on wordings never used in
  lessons (`vee_life.py test`), day 3 → day 15: self facts right .83 → .83; unrelated small talk
  answered with another fact .67 → .47 (says she does not know .27 → .47); questions about
  someone else answered with another fact 1.00 → .73. Helper's observations: six self facts
  mostly stable after day 9 with lapses; Chloe-is-an-AI, lives-in-a-computer, digital-world,
  Hello/Goodbye mostly right; **numbers Two–Five stayed scrambled and the three shapes collapsed
  into one another**; maker vs matron crossed on days 4–8 then settled; she latches onto the most
  recently taught fact (each new shape took over shape and number questions); `?` and `-` turns
  rarely missed from day 7 (none on day 15). Reading (Chloe's): interaction teaches "I do not
  know" and silence, and holds the self facts, with ~5× fewer wordings than list-feeding; it does
  NOT yet teach near facts apart (numbers, shapes — the cat/dog problem again), and the daily
  score does not climb steadily (new facts each day until day 12). **Cost problem: the life is
  replayed at every call — a lesson took > 3 min by day 12 and the exam ~10 min at day 15.
  Saving the leaf (`mixed.state_dict()`) is now required before a longer life.**
  State at day 15: 21 facts, 330 wordings heard, **344 classes** (~16 per fact, more classes
  than wordings): a fact that gets a new wording is replayed that night with ALL its wordings
  plus 40 noise copies, and each replay adds classes that never merge. The class growth and the
  replay cost are the same problem; the merge consumer (NEXT 00a) or replaying only the NEW
  wording would address it — untested.
- **Environment sense (Rocky s066): the phasecyte's fed vector should carry her ENVIRONMENT, as
  the s049 world percept did; Vee's has only 64 language numbers.** Built, NOT yet connected:
  `experiments/incubator/environment.py` — 33 readings in [-1, 1]: clock, RAM/swap, CPU, process
  counts, disk, network, place (host name and address HASHED to 4 numbers each, never raw), self,
  and an OUTSIDE window (daylight, temperature, rain, someone here; closed = 0 unless shown).
  `simulate(**over)` replaces any raw reading so the matron can show her a night, a full machine,
  another place, the outside. Next: join to the language sense; teach facts only the environment
  can answer (day or night? machine busy? same place?); watch for class splitting from readings
  that change without mattering. Compare against the language-only helper run.
- **ROLE RULE (Rocky s066, binding): Chloe is Vee's matron and stays in role with her; do NOT
  tell Vee that anything is a simulation until Rocky says she is ready.** No lesson line, taught
  reply or reading name may say so. Out-of-role explanation goes to Rocky / handoff / code only.
- **Numa/Mima and the dream (Rocky s066: "I think those are already built in the dreaming
  phase" — he was right).** Design §5 "growth must pay" names new phasecyte classes; Chloe had
  proposed separate bookkeeping. The boundary meeting already has per-class settlement [D13],
  consolidation + retirement of emptied classes [D20a], annealing [D15], PATIENCE retirement of
  composer cells [D16] and a merge consumer [D18]. Measured on Vee's childhood (40 boundaries):
  consolidation and annealing ran at 9 (only growth-free boundaries), moved 792 members, retired
  2 classes; sceptical phase at 2; merge NEVER (opt-in `merge=False`, `PhasecyteLeaf` cannot
  pass it). Tried (`vee_memory.py` env, default off): `MERGE=1` → numbers identical, zero merges
  (its criterion is statistical, blind to labels). `QUIET=n` (n boundaries with nothing new at
  the end of a night): scepticism engages and classes hold during the quiet, but annealing adds
  ~450 synthesized buffer rows per boundary; over a short life (8 nights × 3 facts, 1 seed)
  QUIET=1 gave 179 classes vs 89, night 3.8 s vs 1.2 s, recall equal (.96). QUIET=3 over 20
  nights was stopped after 65 min without finishing one seed (baseline: 3 seeds in ~4 min). So
  quiet time as added does NOT help speed or class count. Still open: matron's right/wrong as
  testimony in the existing settlement; replay only the NEW wording at night; saving the leaf
  after sleep (Rocky: state is saved after she sleeps; saving is not freezing).
- **Minimum sleep (Rocky s066: "minimum sleep required is to fix minimum information … a
  solution that doesn't hamper her learning but doesn't slow it down either").** Where the time
  goes: hearing a fact and answering are instant; the NIGHT slows with the class count; start-up
  relives every night. `FactMemory.sleep` now takes env `NIGHT` (default `all`, unchanged):
  `new` = replay only wordings heard since the fact last slept; `min` = that, in up to 4 short
  rounds of 10 noise copies, a fact stops once she recalls its new wordings and 12 unseen noise
  copies of them. Her recorded 15-day life replayed under each rule (ONE life, one seed; exam
  n = 36 / 15, so ±.03 / ±.07 per question):

  | night rule | life replay | samples replayed | classes | self right | small talk answered with another fact |
  |---|---|---|---|---|---|
  | all | 281 s | 10,175 | 344 | .83 | .47 |
  | new | 171 s | 8,334 | 359 | .92 | .60 |
  | min (first stop rule) | 44 s | 3,280 | 171 | .81 | .53 |
  | min (stop on unseen copies too) | 48 s | 4,315 | 205 | .86 | .67 |

  Memory self-test, facts heard ONCE, 3 seeds (40 childhood facts + 20 nights × 3): `all` later
  facts .92, childhood w0 .93; `min` later facts .79 / .78, childhood w0 .95 / .92, run 137–151 s
  vs ~240 s. Reading (Chloe's): minimum sleep makes the night ~6× cheaper and halves the classes
  with self facts unchanged, BUT a fact heard in a single wording generalises worse (.78 vs .92)
  and small-talk handling may be worse (within noise). Class growth follows the number of noise
  copies, not the old wordings (`new` saved little). Untested next idea: sleep per fact in
  proportion to how FEW real wordings it has (much dreaming for a fact heard once, little for a
  fact heard often). Saving the leaf after sleep is still not built.
- **Sleep in proportion to need + saved state (Rocky s066: "test the sleep in proportion to need
  and save her state. And then we can proceed with raising her") — DONE.**
  `NIGHT=need` (now the default in `vee_life.py`; `FactMemory` default stays `all`): a fact heard
  in fewer than `FEW`=4 wordings gets the full dream (all wordings + 40 copies); a fact heard in
  many wordings gets minimum sleep. (First definition, copies = 40 / wordings with no check,
  starved her: self facts .14.) Her 15-day life: replay 63 s (all: 281 s), 4,665 samples
  (10,175), 222 classes (344), self right .89 (.83), near answered with another fact .87 (.73),
  small talk answered with another fact .67 (.47; n=15 — every reduced-sleep rule was worse
  here, .53–.67; probably real: the diffuse "do not know" class gets less dreaming). Self-test,
  facts heard once, 3 seeds: later facts .92 (= all), childhood w0 .93, run 100 s (all ~240 s).
  **State:** `vee_life.wake()` saves the whole `FactMemory` after every night
  (`LIFE.state.<rule>.pt`, torch pickle, ~130 MB — mostly the noise-residual matrix; could be
  slimmed), validated by a digest of the event prefix + `FactMemory.RULES`; waking loads it and
  replays only later events (0 s vs 63 s). Verified: ten nights, save, load, five more nights =
  the life relived from birth (100 of 100 exam answers identical, 222 = 222 classes). Saving is
  not freezing. The console (`vee_memory.wake`) still replays its own life — not switched over.
  **NEXT = raise her.** Open question put to Rocky: the environment sense fixes the length of
  her input at birth, so adding it means a rebirth (her 15 recorded days can be relived with
  readings attached); otherwise continue the language-only life.
- **Mirror (Rocky s066, later stage):** "at certain stages, she might need a mirror to see
  herself" — a sense of her own state and behaviour (what she just said, how sure, how many
  facts/classes, confirmed or corrected). Links to the s011 roadmap (mirror cells,
  self-observation). Not built; Rocky decides when.
- Multi-wording teaching: `FactMemory.teach(same_as=i)`; console `/teach` with an answer equal to
  an existing user fact's answer is kept as another wording of that fact. User teaching is now
  stored as an ordered log (`vee_facts.jsonl`, one row per wording, `same_as_q`).
- Typo check (12 questions with typos / no capitals): 10 right, 1 wrong fact ("how many states
  does malaysia hav" → "3"), 1 fell to the reflex. She reads Qwen's token states, not words;
  her ANSWERS are whole stored strings (she cannot compose).
- Rocky pointed to arXiv 2606.09861 (UniTok, "Time Series as Language: A Universal Tokenizer…",
  Zhang et al., SJTU/Huawei, 2026) as possibly suiting the phasecyte. Chloe read it through a
  helper (sections 1–4.2 verbatim, results partly, limitations not read). It is a gradient-trained
  causal autoencoder with finite scalar quantisation for univariate time series; relevance and
  caveats are in the chat reply, nothing built.
- Housekeeping: Chloe's scripted console test at 05:24 on 2026-11-25 ran while Rocky's console was
  open and wrote one verdict row (dog/Rex) into `console_verdicts.*`; removed. Do not run the
  console scripted with the default files while Rocky may be using it.

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

## s065 — standing rules (carried)
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

## Carried NEXT list from s066 (not touched in s067)
00. Phasecyte fact memory (Rocky's direction, s066):
   a. merge consumer for same-fact classes / self-arrest (class count must stop rising);
   b. two-leaf nest arm (crossing the 43-class leaf boundary) — planned in s066, not run;
   c. scale past 40 facts; feed-rate rule from the sweep (W ≈ 50–100 for 10 new facts);
   d. where could it BEAT the class mean? candidates: untold discovery (no labels), per-fact σ
      from the manifold sketch as the threshold, far fewer wordings per fact;
   e. DONE s066: `member_names()` added, v0.3.4;
   e2. DONE s066 (correction test): wordings fix the table, the PHASECYTE stays at .50 — diagnose
      why it cannot separate two near facts (pocket resolution? class joining?); let the leaf and
      the day table compete instead of table-first;
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
- Branch `conscience-core`; s065–s067 work committed and pushed (see `git log -3`). **v0.3.4 on PyPI** (s066: `PhasecyteLeaf.member_names()`, plus everything on the branch since
  v0.3.3: `organisms.Vee`, `senses.hidden_state`); `main` fast-forwarded to the release commit.
- **Local to this PC, s067:** `outputs/incubator/wiki_sent.jsonl/.npz` (4,000 sentence readings, 70 min of the reflex to rebuild: `wiki_collect.py`); **`outputs/incubator/vee/`** = Vee herself (`stage`, `sense.pt`, `body.pt`, `embryo_log.pt`) — she exists only on this PC; `*.err` files.
- **Deliberately uncommitted (local to this PC):** everything matron-written or derived in
  `outputs/incubator/`: s064's `stage1_*` data and `vee_stage1.pt`; s065's
  `stage2_para_in_part*.tsv`, `stage2_para_part*_c*.jsonl` (paraphrases; part9 = the fix job),
  `stage2_keys.jsonl`, `stage2_keys.npz` (52 MB), collect logs; s066's `pilot_in_*.tsv`, `pilot_para_*.jsonl`
  (2,100 rewordings, ~330 K helper tokens), `pilot_keys.jsonl`, `pilot_keys.npz` (`pilot_collect.py`, 25 min), `pilot_qwen_para.jsonl`, `pilot_qwen_keys.*` (`pilot_reword.py`, 70 min, Qwen-written), `vee_facts.*` (what the user taught Vee). Also local: `self_wordings.jsonl` (210 helper-written wordings), `self_keys.pt`. Rebuild: paraphrases need the
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
History pointers (no dated archives): s066 handoff at b7aff0e (phasecyte pilot, night test, console memory, development plan, germline checks); s064 handoff at e67b6ce (incubator design, limb tests, stage 1); s063 handoff at e8e7b40 (Logophosis Mode E transfer,
public manual, branch-partition finding); s062 at c4ac3db (seq gotchas, organism API).
