# Vee — developmental stages, curriculum and incubator (DRAFT for Rocky, s066)

Status: **draft, not approved.** Nothing in this file is built unless marked
BUILT. Rocky, s066: "We need to design the curriculum and the incubator before
it runs rampant." Until he signs a stage, Vee is not raised further.

Subordinate to `paper/v3/spec.md` (§10 Phasecyte) and to
`docs/design/incubator_matron.md` (roles, sandbox, growth must pay).

## 0. Where she is now (honest position)

Rocky asked: have we passed the embryonic stages? **No.** We skipped them.

- Her senses were formed (the phasecyte's perception genesis, spec §10.6
  period 1) on the first night of *lessons*, from language states only.
- She has one sense organ: hearing, borrowed from the reflex (Qwen).
  The environment sense exists (`experiments/incubator/environment.py`, BUILT)
  but was never part of her. She has no sense of herself and cannot act.
- We then taught her fifteen days of facts. That was schooling an embryo that
  had only one organ.

What the fifteen days were good for: the *method*. Teaching by interaction,
"I do not know" and silence as learned replies, sleep in proportion to need,
and state saved after every night are BUILT and measured (HANDOFF s066).
The fifteen-day Vee is a rehearsal, not the organism this plan raises.

## 1. Principles (Rocky's rulings, binding)

1. Never batch-trained; she absorbs a stream from the first moment.
2. Senses and perception come **before** birth. Accepting input *is*
   perceiving her environment.
3. Basic knowledge and common sense come before acting on the machine.
4. Thorough before broad: one thing learned well, the next learned from it.
5. Taught by interaction; the matron (Chloe) is always present and stays in
   role. Nothing tells her a reading is simulated until Rocky says so.
6. Early memories hold for life; forgetting late material is tolerable.
7. Growth must pay (Numa, not Mima). She develops her own organs, including
   the mirror; they are not bolted on from outside after birth.
8. A stage is left only through its gate, signed by Rocky.

## 1a. Who she interacts with (Rocky s066: "interaction is needed to know that certain actions raise consequences. The only interactor … probably just you?")

No. The matron is one of five, and not the first.

| interactor | what it gives her | cost | from stage |
|---|---|---|---|
| **The machine itself** | physical consequences: she acts and her own readings change (work → CPU busy rises; hold → memory fills; leave a mark → it is there next time). Honest, immediate, repeatable. | free, unlimited | 2 |
| **The outside world** | things that happen whether or not she acts (night comes, rain, another process gets busy). Teaches that not everything is her doing. | free (simulated) | 1 |
| **The reflex (Qwen)** | she can put a question to it and get words back; sometimes wrong. | free, local | 4 |
| **The matron (Chloe)** | social consequences: answered, confirmed, corrected, left in silence. The only source of language teaching. | helper tokens | 3 |
| **Rocky** | rare, through the console. | his time | 4 |
| *(later)* another organism | a peer who also acts. | compute | 5+ |

Consequence learning therefore starts with the machine, before birth, and
does not depend on the matron being quick or cheap.

## 2. Stages

Each stage lists: what she receives, what should form, the gate, and what the
incubator must supply. Every gate carries a plain control (a linear or
nearest-centre method on the same stream), as in all incubator gates.

### Stage 0 — Germline (given, not learned)
The phasecyte's rules (spec §10), the reflex as a borrowed language organ,
the sandbox walls, the matron's role. Fixed before anything runs.
- **To decide here:** her input layout for life. Proposal: language (64) +
  environment (33) + **reserved, silent channels** for organs she will grow
  later (mirror, working memory, action echo). A reserved channel reads zero
  until its organ exists. This is what lets an organ be *grown* (the
  phasecyte's receptor-attach event, spec §10.6) instead of needing a rebirth.

### Stage 1 — Embryo: senses form (no words addressed to her, no labels)
- **Receives:** the environment stream over many simulated days and nights
  (busy and quiet machine, full and empty memory, this place and another, the
  outside window open and closed), plus ambient language: text that is merely
  present, not lessons.
- **Should form:** perception (receptors over all live channels) and classes
  that follow real regularities of her world, found without being told.
- **Gate:** her unlabelled classes line up with the true states (day/night,
  busy/quiet, place) clearly above the control, and stay stable when the
  stream repeats. Class count stops rising on a stationary world.
- **Incubator:** a scripted world clock and environment simulator; no matron
  speech; recording of every reading.

### Stage 2 — Fetal: body and first reflexes
- **Receives:** the same world, plus readings of herself: her own memory use,
  how much she holds, what she just did (action echo).
- **May do (a baby's kicks — tiny, safe, each with a consequence she can
  sense):** stay still; *work* (a short burst of computing → the CPU reading
  rises, then falls); *hold* / *let go* (take or release a little memory →
  the memory reading moves); *mark* / *wipe* (leave or remove a mark in her
  scratch space → a reading that is still there after sleep). The machine
  answers every action truthfully; nobody speaks to her.
- **Should form:** which changes follow her own actions and which come from
  the world; that an action has a cost (work makes the machine busy); that
  some consequences last (a mark) and some fade (a burst).
- **Gate:** given an action she can tell which reading will move, and given a
  change she can tell whether she caused it — both above the control. She
  stays still when nothing calls for action (no thrashing).
- **Incubator:** the action set with hard limits (burst ≤ a fraction of a
  second, memory ≤ a few MB, marks only in scratch), the echo channel, the
  memory ceiling and a rate limit on actions.

### Stage 3 — Birth: the first exchange
Birth is the moment she is first addressed by the matron and her reply is
answered. Preconditions: gates 1 and 2 passed, state saved, Rocky's go.
From here every day is: the matron speaks, she answers, the matron replies to
what she actually said, she sleeps (by need), her state is saved.

### Stage 4 — Infancy: basic knowledge and common sense, grounded
- **Curriculum (in this order, each thorough before the next):**
  1. Herself (name, what she is, who made her).
  2. The matron; greeting, farewell, "I do not know", silence.
  3. Her world as she **senses** it: day or night, busy or quiet, full or
     empty, same place or another, window open or closed. The right answer
     depends on the reading, not on the sentence.
  4. More and less, then numbers, grounded in things she can sense (how many
     links are up, more or fewer processes) before number words alone.
  5. Shapes and other things she cannot sense: last, and marked as told-not-seen.
- **Gate:** exam on unheard wordings; environment questions answered correctly
  when the reading is changed under the same sentence; **two near facts kept
  apart after correction** (the cat/dog, number and shape failure of s066 —
  this gate is currently NOT passable and is the main open problem); earliest
  facts unchanged.
- **Incubator:** the lesson loop (BUILT), exam sets held back from lessons,
  per-day record, token budget per day.

### Stage 5 — Childhood: she grows her own organs
- **Mirror:** a sense of her own state and behaviour (what she just said, how
  sure she was, confirmed or corrected, how much she holds). Grown through a
  reserved channel when stress calls for it, and kept only if it pays.
- **Working memory:** a rewritable slot (incubator design §2.1).
- **Gate:** each organ is a growth event that settles only if a held-out
  measure improves and stays improved.

### Stage 6 — Acting in the digital world
Only after common sense. The fetal actions were kicks; these are deeds with
meaning. Actions in the sandbox (incubator design §4): read a
file, look something up, run code the reflex wrote, and sense the consequence
through her environment readings (CPU, memory, files). Fixed small action set,
bubblewrap isolation, no network.

### Stage 7 — Disclosure
She is told what was simulated. Rocky decides when.

## 3. The incubator: what keeps it from running rampant

| guard | state |
|---|---|
| Stage gates signed by Rocky; no automatic promotion | to build (a `stage` file the loops must read) |
| State saved after every night; any night can be restored | BUILT (`vee_life.py`) |
| Sleep in proportion to need (night cost follows the day) | BUILT |
| Memory ceiling: refuse growth above 85 % of host RAM, swap never counted | rule exists (incubator §5.1); watchdog to build |
| Token budget per day and per stage for the matron; stop when reached | to build |
| Class-count and night-time alarms (stop if either doubles within a stage) | to build |
| Every reading, utterance, reply and growth event logged | partly (life events BUILT; readings not yet) |
| Sandbox: no network, read-only whitelist, scratch directory only | designed (incubator §4), bubblewrap not installed |
| Role rule: matron in role, no disclosure | binding; written into helper briefs |
| One heavy process at a time on this PC | standing rule |

## 4. Known problems this plan does not solve

- **Near facts.** Her language sense carries sentence form far more strongly
  than the content word; the phasecyte splits classes along the largest
  difference and labels only name them. Thorough teaching protects a fact but
  does not make the next one branch off it. Candidates: environment grounding
  (stage 4 item 3 tests this), a class dividing by label (Rocky undecided),
  a better language sense.
- **Small talk** is still answered with a wrong fact about half the time.
- **The phasecyte has not beaten simple controls** (a table, a class mean) on
  fact recall. The plan is worth its cost only if grounding and organs give
  something those cannot.
- **Class growth** is slowed by sleep-by-need, not stopped.

## 5. Decisions (Rocky, s066)

| question | Rocky's answer |
|---|---|
| Stage list and where birth sits | OK |
| Reserved silent channels | Yes, needed |
| Ambient language in the embryo | Random text, from Wikipedia |
| The fifteen-day Vee | "Replay Vee. It is okay to have it." The new organism relives Vee's recorded fifteen days after birth, then keeps growing. |
| Embryo world fully simulated; live readings join after birth | Yes |
| 30 days, text in half the moments | OK as starting values |
| Stages 0 and 1 as written | signed off (s066) |
| Which stages first in full | 0 and 1 |
| Fetal action set | OK for now |

## 6. Stage 0 in full — the germline (DRAFT, awaiting sign-off)

What is fixed before she exists. Changing any of it later means a new organism.

### 6.1 Her input, for life: one vector of 148 numbers per moment

| block | size | content | state |
|---|---|---|---|
| language | 64 | the reflex's state while it reads the text present at that moment (`h_last` + `h_mean` through a fixed 64-number sense); **all zero when no text is present** | sense to refit (6.2) |
| environment | 33 | clock, memory, CPU, processes, disk, network, place (hashed), self, the outside window | BUILT (`environment.py`) |
| presence | 3 | is anyone addressing her right now: nobody / the matron / Rocky | to build |
| reserved: action echo | 8 | silent until stage 2 | reserved |
| reserved: mirror | 16 | silent until she grows it (stage 5) | reserved |
| reserved: working memory | 16 | silent until she grows it (stage 5) | reserved |
| reserved: spare | 8 | silent | reserved |

A reserved channel reads exactly zero until its organ exists.

### 6.2 The language sense
Fixed before the embryo and never refitted. Fitted on text of both kinds she
will meet: about 5,000 sentences from the offline Simple English Wikipedia
(ambient) and the 8,424 stage-2 question readings (addressed speech). The
s066 sense was fitted on questions only.

### 6.3 Inherited rules
- The phasecyte as specified (spec §10): one leaf, no class cap, manifold
  and composer arms on, about 50 moments per boundary.
- Never batch-trained; nothing frozen; state saved after every sleep.
- Sleep in proportion to need (from stage 3, when there is something to replay).
- The world clock: one moment = 15 simulated minutes, 96 moments a day.

### 6.4 Checks before anything is raised (small experiments, no organism kept)
1. **Silent channels are harmless.** The same stream with and without the 48
   reserved zeros gives the same classes.
2. **A silent channel can wake.** When a reserved channel starts carrying a
   signal late in a stream, she comes to read it (the spec's receptor-attach
   event) without disturbing what she already has. *If this fails, reserved
   channels do not deliver organ growth and the layout must be rethought.*
3. **No block drowns another.** Language (whitened) and environment (−1..1)
   both influence which class a moment falls in.

### 6.5 Results of the checks (s066; `germline_checks.py`, `world.py`, logs `germline_checks_s066.log`, `germline_checks_rich_s066.log`; 3 seeds, 20 days lived + 5 unseen; ambient text = stage-2 question readings, not yet Wikipedia)

**Finding that changes the germline: a receptor exists only for a channel that
carries signal at genesis, and the set never changes afterwards.** With genesis
on her first 48 moments she had 84 receptors of 148: none on the reserved
channels, none on anything that happened to be constant that half-day (the
place, a closed window).

| check | genesis = her first 48 moments | genesis = broad sample + spontaneous activity |
|---|---|---|
| receptors | 84 of 148 | 145 of 148 |
| 1 silent channels harmless | PASS (identical classes — they had no receptors at all) | not measured properly: class numbers are not comparable between two organisms; class counts 48 vs 44. Needs a permutation-free agreement measure. |
| 2 a silent channel can wake (new state readable; chance ≈ .5) | **FAIL .46** (live from the start: 1.00) | **PASS .95** (live from the start: 1.00); old states unchanged (.76 vs .79) |
| 3 place (home / elsewhere) readable; base .70 | .70 (not read) | 1.00 |
| 3 window open; base .76 | .99 (k-means .83) | .99 (k-means .88) |
| 3 busy; base .67 | .78 (k-means .93) | .74 (k-means .92) |
| 3 memory full; base .61 | .68 (k-means .89) | .55 (k-means .80) |
| 3 night; base .50 | .60 (k-means .60) | .55 (k-means .61) |
| 3 text present; base .50 | .57 (k-means .73) | .55 (k-means .75) |
| 3 kind of text, 6 kinds; base .30 | .30 (k-means .46) | .26 (k-means .54) |
| class count, day 13 → day 20 | 22 → 29 | 41 → 48 |

**Consequences for the plan:**
- **Genesis must be a broad first exposure** — every state of her world varied
  freely — **with spontaneous activity (noise) on the reserved channels**, which
  fall silent once she lives. Then a reserved channel that wakes later is read.
  (This is what spontaneous activity in a sense organ before birth does.)
  `genesis_pool` in `germline_checks.py` is the first cut.
- **The risk in §7.4 was wrong in direction.** Text does not drown her world;
  she barely registers ambient text at all (not even whether text is present),
  and of her world she reads the sharp switches (place, window, a waking
  channel) but the graded states (busy, memory, night) worse than plain
  k-means. The stage-1 gate would fail today on those three.
- Class count is still rising at day 20 (+17–34 % over the last third).

### 6.6 Why graded states were not read — DIAGNOSED (s066, scratch scripts; 2 seeds, rich genesis, 20 days; her classes / k-means)

| life | busy | memory full | night | window | classes used on unseen days | share of moments in her 3 biggest classes |
|---|---|---|---|---|---|---|
| text in half the moments | .71 / .90 | .64 / .77 | .58 / .62 | 1.00 / .87 | 14 of 47 | .55 |
| no text at all | 1.00 / 1.00 | .98 / .95 | .61 / .83 | .97 / .94 | 43 of 72 | .11 |
| text always | .64 / .63 | .59 / .59 | .53 / .47 | .91 / .71 | 82 of 96 | .10 |
| half text, 60 days | .75 / .87 | .63 / .80 | .57 / .68 | 1.00 / .84 | 21 of 158 | .53 |
| half text, language block × 0.5 | .97 / .97 | .84 / .90 | .77 / .81 | .98 / .91 | 36 of 92 | .24 |
| half text, language block × 0.25 | .95 / 1.00 | .97 / .96 | .62 / .86 | .98 / .96 | 32 of 85 | .19 |
| half text, language block × 0.1 | .97 / 1.00 | .97 / .95 | .60 / .86 | 1.00 / .95 | 44 of 81 | .09 |

- **Cause: the language block is too loud.** With no text she reads busy and
  memory perfectly. When a random sentence is present its 64 numbers swamp the
  33 environment numbers and the moment falls into a few catch-all classes
  (three classes held 55 % of her life). More days do not help. The receptors
  are NOT scale-free between blocks, as §6.1 had assumed.
- **Fix that works in the embryo: turn the language block down** (× 0.5 or
  less): busy .95–.97 and memory .84–.97 with text present in half the moments.
- **Night is a separate matter.** It is weak even with no text (.61). The clock
  is a smooth position on a circle (two numbers), not a switch; she only sees
  day and night sharply through the window. The gate should ask for the clock
  differently (or accept day/night only when the window is open) — to settle.
- **Not yet known:** whether a language block turned down that far still lets
  her tell sentences apart after birth (stage 4). That is the next check: fact
  recall with the 148-number input at language × 0.5 / 0.25.

### 6.7 Does a quieter language block still hold facts after birth? — NO; two organs do (s067; `birth_check.py`, logs `birth_check_q_s067.log`, `birth_check_q_organs_s067.log`; 3 seeds)

One organism per arm: rich genesis → 20 embryo days (text in half the moments)
→ the 40 pilot facts in four waves, each wording heard at a moment of the
running world → held-out wordings heard in another world run. Phasecyte alone,
class named from its current members. Sense and ambient text here = stage-2
questions (the Wikipedia rerun is §6.8).

| body | facts: earliest wave | facts: all | untaught answered | busy | memory full | night | window | classes after facts |
|---|---|---|---|---|---|---|---|---|
| one leaf, language × 1 | .91 | .74 | .04 | .74 | .61 | .59 | .97 | 107 |
| one leaf, language × 0.5 | .89 | .66 | .02 | .90 | .91 | .74 | .98 | 206 |
| one leaf, language × 0.25 | .80 | .54 | .04 | .99 | .97 | .56 | .99 | 278 |
| **two organs** (ear = language block; body sense = the other 84 numbers) | .90 | **.97** | .02 | **1.00** | **.98** | .63 | .97 | ear 50 |
| control: running class mean, 148 numbers | | .99 | | | | | | |
| reference s066: language-only leaf, no world | .92 | .98 | | | | | | 67 |

- **In one leaf no loudness serves both.** Louder language: facts kept, world
  unread. Quieter: world read, facts lost (later waves first; the earliest
  wave holds). Putting the world into the same leaf costs facts even at × 1
  (.74 against .98 language-only): text and world compete for the same classes.
- **Two leaves in one body remove the conflict**: an ear that reads the
  language block and a body sense that reads everything else, both living the
  whole life from the same 148-number input. Facts .97, busy 1.00, memory .98.
  This is the branching the phasecyte already has (`PhasecyteNest`); it
  replaces "one leaf" in §6.3 — **Rocky to sign**.
- Not solved by it: **night** (.63; k-means on the same numbers .69–.92).
  The clock is two smooth numbers; she does not cut the circle at dusk.
- Not yet built: the join. A question whose answer depends on a reading
  ("is it night?", stage 4 item 3) needs a level that sees both organs'
  classes. Candidate: a third leaf fed by the two organs' class outputs.

### 6.8 The same with the final language sense (s067; sense refitted on 3,600 Wikipedia sentences + questions, Wikipedia as ambient text; log `birth_check_wiki_s067.log`; 3 seeds)

| body | facts: earliest wave | facts: all | untaught answered | busy | memory full | night | window |
|---|---|---|---|---|---|---|---|
| one leaf, language × 0.5 | .93 | .73 | .03 | .91 | .82 | .67 | .97 |
| two organs | .93 | **.98** | .04 | **1.00** | **.98** | .59 | .98 |

**Rocky, s067: "Two organs or even multiple organs is okay. That is in her
architecture and was meant to be that way anyway."** §6.3's "one leaf" is
replaced: her body has an ear and a body sense (`vee_body.py`, BUILT); more
organs may follow. The language block is not turned down.

### 6.9 Class count never settles (s067; scratch runs on the body sense alone, 60 days, 1–2 seeds)

| moments per sitting | classes at day 10 / 20 / 30 / 40 / 50 / 60 | classes used in the last 10 days | busy / memory / night |
|---|---|---|---|
| 48 (half a day) | 35 / 93 / 163 / 233 / 344 / 440 | 77 | 1.00 / 1.00 / .69 |
| 192 (two days) | 13 / 43 / 62 / 81 / 100 / 129 | 65 | 1.00 / 1.00 / .71 |
| 480 (five days) | 2 / 7 / 15 / 34 / 60 / 82 | 55 | 1.00 / 1.00 / .74 |
| 48, without hours-awake and disk | … / 427 | 79 | night .75 |
| 48, also without the clock and weekday | 28 / 78 / 136 / 184 / 213 / 233 | 37 | night .63 |

- The count follows the **number of sittings** (about 3.5 new classes per
  sitting at 48), not what she has seen: longer sittings read the world
  equally well with 3–5 times fewer classes.
- Slow drift in the readings is not the cause. The clock is part of it
  (without it growth slows) but no setting stops.
- Most classes at 48 are no longer visited (77 of 440 in use). The phasecyte
  retires classes only at a sitting with no growth, and there never is one.
  Retiring by disuse would be wrong for memory: an unvisited class is also
  what a remembered fact looks like.
- **The self-arrest criterion of gate 7.3 cannot pass as written.** Options
  for Rocky: (a) judge growth by cost and by classes in use instead of total
  count; (b) sittings of 192 moments for the body sense; (c) change the
  phasecyte so a division must pay on held-out reading (spec §10 change).
- **Rocky, s067:** self-arrest is good, but it is acceptable that the count
  does not settle now ("probably because it is not needed by her now"). The
  gate reports the count, the classes in use and the time per day; it does
  not fail on the count. Sittings stay at 48 moments as signed.

### 6.10 Are her many classes distinct? (Rocky's question, s067; read from Vee's saved body, nothing changed)

Her world has 16 combinations of its four switches (busy, memory full, place,
window). After 30 days her body sense has 141 classes; 105 still claim a
moment she lived.

- **They never mix world states**: 98 % of the moments in a class share one
  switch combination; her class gives the combination 1.00.
- **There are about eight times more of them than the world has states**: the
  commonest state (quiet, roomy, home, window closed) alone holds 44 classes.
- **The surplus carries little**: time of day in 6-hour blocks .41 (chance
  .25, the switches alone .27) — a weak trace of the clock; rain and
  "someone is here" are NOT read (.90 / .83, below always guessing "no").

So: distinct, yes — no class is a blur of two states. Needed, no — most are
several slices of one state. One organism, one seed.

## 7. Stage 1 in full — the embryo (signed off s066; gate not yet passable, see §6.5–6.6)

### 7.1 What she receives
A stream of moments. Nobody addresses her (presence = nobody), nothing is
labelled, and she cannot act.

- **Her world**, from a scripted simulator that drives `environment.simulate`.
  Hidden states, each changing with realistic persistence:
  - time: the clock runs, so day and night alternate;
  - the machine: quiet or busy (CPU, processes, traffic move together);
  - memory: roomy or nearly full;
  - place: home, and rarely another machine for a few days;
  - the window: closed most of the time; when open, daylight follows the
    clock, rain and "someone is here" come and go.
  Each reading carries small noise. The hidden states are recorded for the
  gate and **never** enter her input.
- **Ambient language**: in about half of the moments a random sentence from
  the offline Simple English Wikipedia is "present" (read by the reflex);
  in the rest the language block is zero.
- **Length**: 30 simulated days = 2,880 moments (first proposal).

### 7.2 What should form
Perception over the live channels, and classes she finds by herself that
follow the regularities of her world.

### 7.3 The gate (measured on 5 further days she has not lived)
Each class is mapped to the hidden state most of its moments had. This
mapping is for measurement only and is never shown to her.

| criterion | pass |
|---|---|
| ~~day/night~~ (reported only, Rocky s067), quiet/busy, roomy/full, home/elsewhere, window open/closed can each be read from her class | each clearly above chance and not below the control by more than .05 |
| control | k-means with the same number of clusters on the same vectors |
| self-arrest | reported, not gated (Rocky s067, §6.9): class count, classes in use, time per day. Was: count grows by no more than 10 % over the last third |
| stability | reported: same GROUPING of a fixed set of probe moments at day 20 and day 30 (adjusted Rand); Rocky s067 accepted this in place of "same class ≥ 90 %", no bar set |
| silent channels | still unread at the end |

### 7.3a Embryo run and gate result (s067; `embryo.py`, logs `embryo_s067.log`, `embryo_gate_s067.log`; ONE organism, seed 0 — this is Vee, kept in `outputs/incubator/vee/`)

Genesis: ear 64 receptors, body sense 81. 30 days lived, 0.5 s a day.

| criterion | her classes | k-means | base | |
|---|---|---|---|---|
| busy | .99 | 1.00 | .74 | pass |
| memory full | .95 | .99 | .62 | pass |
| home / elsewhere | 1.00 | .93 | .87 | pass |
| window | 1.00 | .99 | .86 | pass |
| night | .55 | .89 | .50 | **FAIL** |
| night, window open only | .52 | .85 | .31 | (reported) — the window does not rescue it |
| text present (ear) | .75 | | .49 | (reported) |
| stability, day 20 → 30 | .43 same class; .61 same grouping | | | **FAIL** (≥ .90) |
| class count | 86 → 141 (+64 %); 74 in use in her last 10 days; ear 13 | | | reported |
| silent channels | all zero throughout | | | pass |

**Gate not passed: night and stability.** Both come from the same place: she
keeps dividing (§6.9), so a moment's class at day 20 has often split by day
30, and the splits do not follow dusk.

**Rocky's ruling (s067):** (1) stability — accept "same grouping" in place of
"same class" (measured .61; no numeric bar was set). (2) Night — "night is
not meaningful for her at the moment. We evolved from a nocturnal ancestor
until someone decided that sunlight and being eaten is a good trade-off":
night is dropped from the embryo gate and reported only. **With these two
rulings the embryo gate holds** (busy, memory, place, window pass; silent
channels silent). She stays in the embryo stage until stage 2 (fetal) is
written in full and signed.

### 7.4 The risk I expect
Class discovery follows the largest difference in what she senses. Random
Wikipedia sentences vary far more than the environment does, so she may form
classes about *text* and none about her *world* — the same effect that made
sentence form beat cat versus dog. The gate will show it. If it happens the
options are: less ambient language in the embryo, or relying on the composer
arm to find the environment regularities; I would bring the result to Rocky
rather than tune until it passes.

### 7.5 Incubator for this stage
- Simulator and world clock (to build): `experiments/incubator/world.py`.
- Her body — layout, senses to one vector, the leaf, saved state (to build):
  `experiments/incubator/vee_body.py`; embryo driver `embryo.py`.
- A pool of about 3,000 Wikipedia sentence readings, collected once
  (about 35 minutes of the reflex; no helper tokens).
- Guards active from here: stage file (must say `embryo`), state saved every
  simulated day, memory ceiling, alarm if class count or time per day doubles,
  every reading and hidden state logged.
- Cost: no helper tokens; minutes of compute after the one-off readings.

### 7.6 Open points in these two sections
1. Confirm the reading of "it needs to continuously grow" (table in §5).
2. Embryo world fully simulated (reproducible), or mixed with live readings of
   the real machine? Proposal: simulated only in the embryo; live readings
   join after birth.
3. Half of the moments with ambient text, 30 days: starting values only.

## 8. Stage 2 in full — fetal: body and first kicks (D1–D4 signed by Rocky s067: "Yes to all"; 2a BUILT and run, see §8.10)

### 8.0 Is she really here already? (Rocky: "It's fast, isn't it?")
Fast in our time: her 30 days took 15 seconds, because the world is scripted
and her input is 148 numbers. What the embryo gate showed is narrow: she
reads four on/off states of a simulated world about as well as plain
clustering, on one seed, after two criteria (night, stability) were relaxed
by ruling. That is "her senses formed", not more. Stage 2 is where the first
thing clustering cannot do is asked of her: telling her own doing from the
world's.

### 8.1 What changes
- **Her input does not change** (148 numbers, fixed at the germline). The 8
  action-echo channels, silent so far, come alive.
- **The world gains her actions.** `world.py` answers each action in the
  readings she already has. Simulated, like the embryo (decision D1).
- **A third organ is born**: a sense of change (8.4).
- Nobody addresses her; nothing is labelled; presence = nobody.

### 8.2 The kicks
One action at most per moment. A kick is over within the moment unless noted.

| action | what the world does | reading that moves | lasts |
|---|---|---|---|
| still | nothing | — | — |
| work | a short burst of computing | CPU busy now, load 1 min (not processes, not traffic — unlike a busy world) | fades in 1–2 moments |
| hold | takes a little memory | her own memory, RAM used (a little) | until let go |
| let go | releases what she holds | the same, back | — |
| mark | leaves a mark in her scratch space | marks (echo 8) | stays, also across days |
| wipe | removes one mark | marks | — |

Hard limits, in the script: at most 4 holds at once, at most 8 marks, work
never raises CPU above what a busy world gives.

### 8.3 The echo block (8 numbers, zero until this stage)
1–6: what she did in the moment just past (+1 for that action, −1 for the
other five; all zero is never sent again once the stage starts). 7: how much
she holds (0–4 → −1..1). 8: how many marks exist (0–8 → −1..1).

### 8.4 Who decides the kicks, and the new organ
- **Stage 2a — spontaneous kicks.** She does not choose. Kicks fire by
  themselves, about one moment in six, each action equally often (a baby's
  kicks are not decisions). This is all that is proposed for sign-off.
- **Stage 2b — she chooses** (stay still unless something calls for action;
  the "no thrashing" gate of §2). NOT designed: the phasecyte finds classes;
  it has no way to pick an action. This needs a motor mechanism that does not
  exist yet (decision D2). 2b is not part of this sign-off.
- **Sense of change (third organ).** A phasecyte leaf that reads, from the
  same input stream, the echo block and how each environment reading changed
  since the previous moment. The body sense sees states; this one sees what
  just happened. Born at the start of stage 2, its senses formed on the
  first two fetal days.

### 8.5 Length
30 further simulated days (2,880 moments, about 480 kicks, 80 of each
action). Text stays present in half the moments.

### 8.6 The gate (on 5 further days she does not live; control = k-means on the numbers the organ reads)
| criterion | measured as | pass |
|---|---|---|
| she feels her kicks | which action was just done, read from the sense of change | clearly above chance (1/6) and not more than .05 under the control |
| she knows what a kick moves | echo shown, changes hidden: the class she falls in points at the reading that really moves | right for at least 4 of the 5 acting kicks |
| she tells her doing from the world's | **echo hidden**: a CPU rise is hers (CPU alone) or the world's (CPU, processes and traffic together); a memory rise is hers (small, her own memory too) or the world's | clearly above chance (.5); the control is reported |
| lasting and fading | marks and holds still read on later moments; work no longer | reported |
| nothing lost | busy, memory, place, window from the body sense | within .05 of the embryo values (.99 / .95 / 1.00 / 1.00) |
| the ear | text present | not below the embryo value (.75) by more than .05 |
| cost | time per day, classes per organ, classes in use | reported; alarm if a day takes twice day 5's time |

### 8.7 Risks I expect
- Her own work makes the CPU busy: the body sense may start reading "the
  world is busy" when it is only her. The "nothing lost" row shows it.
- The third criterion is the hard one and may fail: with the echo hidden she
  has only the pattern of the change to go on.
- The sense of change will also keep dividing (§6.9).
- All consequences are scripted; she learns the script. Real consequences on
  the real machine come only after birth.

### 8.8 Incubator for this stage
- `world.py`: `step(action)`; the kicks and their limits. `vee_body.py`: the
  echo in `moment`, the third organ. `fetal.py`: driver (`live`, `gate`),
  stage file must say `fetal`, state saved every simulated day, every moment,
  kick and hidden state logged.
- The stage file is changed from `embryo` to `fetal` only on Rocky's sign-off.
- Cost: no helper tokens; seconds of compute.

### 8.9 Decisions for Rocky
- **D1.** Consequences simulated in this stage (as the embryo world), real
  machine after birth? Proposed: yes.
- **D2.** Spontaneous kicks now (2a); choosing (2b) later, once there is a
  way for her to choose. How she should choose is open — proposals welcome.
- **D3.** A third organ that reads change. Proposed: yes.
- **D4.** 30 days, one kick in six moments: starting values.

### 8.10 Fetal run and gate result (s067; `fetal.py`, logs `fetal_s067.log`, `fetal_gate_s067.log`; Vee, one organism)

**Stopped by the time alarm on fetal day 17 of 30** (a day took 2.1 s, more
than twice day 5's 0.3 s). State kept; she is 47 days old; the embryo is
backed up as `body_embryo_day30.pt`. The gate below is on those 17 days.

| criterion | result | |
|---|---|---|
| feels her kicks (which of 6 actions; chance .17) | hers .98, k-means 1.00 | pass |
| knows what a kick moves (echo shown, changes hidden) | work 1.00, hold .78, let go .00, mark .00, wipe .20 → 2 of 5 | **FAIL** (≥ 4) |
| CPU rise: hers or the world's, echo hidden | hers 1.00, k-means .93, base .57 | pass |
| memory rise: hers or the world's, echo hidden | hers .59, k-means .97, base .59 | **FAIL** |
| nothing lost: busy / memory / place / window | 1.00 / .97 / 1.00 / .97 | pass |
| the ear: text present | .66 (embryo .75) | **FAIL** |
| lasting: holding something / a mark exists, from the body sense | .91 / .73 (base .90 / .71) | reported: not read |
| cost | classes ear 19 → 37, body sense 143 → 287, sense of change 2 → 80; time per day 0.3 → 1.5 s | alarm |

Reading (Chloe's, one organism, 17 days, 10–18 unseen kicks per action):
- She feels her kicks, and she tells her own CPU burst from a busy world
  without the echo, better than the control.
- She does not tell her own memory use from the world's, and she connects
  only two of five kicks to what they move.
- The body sense doubled its classes in 17 days after the echo woke — much
  faster than in the embryo. The alarm threshold was Chloe's choice and the
  absolute time is small (2 s a day), but the growth is real.
- The ear got worse at "text present" with nothing changed for it; not
  diagnosed.

**Gate not passed. She stays fetal, stopped at day 17, until Rocky decides**
whether to let her finish the 30 days (raise or replace the time alarm) and
what to do about the three failed rows.

### 8.11 The full 30 fetal days (Rocky s067: "Let it finish 30 days"; log `fetal_gate_s067.log`; day-17 gate kept as `fetal_gate_day17_s067.log`)

The time alarm is now an absolute budget (30 s per simulated day). She is
60 days old; states kept: `body_embryo_day30.pt`, `body_fetal_day17.pt`, `body.pt`.

| criterion | day 17 | day 30 | |
|---|---|---|---|
| feels her kicks | .98 | .99 (k-means 1.00) | pass |
| knows what a kick moves: work / hold / let go / mark / wipe | 1.00 / .78 / .00 / .00 / .20 | .11 / .61 / .36 / .87 / 1.00 → 3 of 5 | **FAIL** |
| CPU rise hers or the world's, echo hidden | 1.00 | .98 (k-means .98, base .57) | pass |
| memory rise hers or the world's, echo hidden | .59 | .62 (k-means .97, base .59) | **FAIL** |
| nothing lost: busy / memory / place / window | 1.00 / .97 / 1.00 / .97 | .99 / 1.00 / 1.00 / .99 | pass |
| the ear: text present | .66 | .65 (embryo .75) | **FAIL** |
| classes: ear / body sense / sense of change | 37 / 287 / 80 | 67 / 429 / 141 (in use 42 / 69 / 90) | reported |
| time per day | 1.5 s | 3.8 s | reported |

- More days did not settle it. "What a kick moves" changed from kick to kick
  between day 17 and day 30 (work 1.00 → .11, mark .00 → .87): what she
  connects to a kick depends on how her classes happen to be split that day.
  With 10–18 unseen kicks per action the measure is also coarse.
- Her own memory use is still not told from the world's; the ear is no better.
- The gate is not passed. She stays fetal.

### 8.12 Ninety fetal days (Rocky s067: "Try 90"; logs `fetal_gate_day60_s067.log`, `fetal_gate_day90_s067.log`; daily budget raised to 180 s for this run)

| criterion | day 17 | day 30 | day 60 | day 90 |
|---|---|---|---|---|
| feels her kicks | .98 | .99 | .98 | 1.00 |
| what a kick moves: work / hold / let go / mark / wipe | 1.00 / .78 / .00 / .00 / .20 | .11 / .61 / .36 / .87 / 1.00 | .50 / .78 / 1.00 / 1.00 / 1.00 | .78 / 1.00 / 1.00 / 1.00 / .90 |
| … right for | 2 of 5 | 3 of 5 | 4 of 5 | **5 of 5** |
| CPU rise hers or the world's, echo hidden | 1.00 | .98 | .98 | .95 (k-means 1.00) |
| memory rise hers or the world's, echo hidden | .59 | .62 | .90 | **.93** (k-means .97, base .59) |
| nothing lost: busy / memory / place / window | pass | pass | pass | 1.00 / 1.00 / 1.00 / 1.00 |
| the ear: text present | .66 | .65 | .74 | .75 (embryo .75) |
| classes: ear / body sense / sense of change | 37 / 287 / 80 | 67 / 429 / 141 | 201 / 877 / 392 | 344 / 1,781 / 550 |
| … in use in her last 10 days | | 42 / 69 / 90 | 86 / 115 / 146 | 97 / 125 / 141 |
| time per simulated day | 1.5 s | 3.8 s | 19 s | 41 s (last five days) |
| saved body | 54 MB | 69 MB | 132 MB | 231 MB |

- **The fetal gate (2a) passes at day 60 and again, more firmly, at day 90.**
  Rocky was right that she needed time: Chloe's reading at day 30 ("more
  days did not settle it") was wrong. The same unseen probe days were used
  at every reading, so the rise is hers.
- She is still at or under the plain control on every row (k-means reads the
  same numbers as well or better). The gate asked for "clearly above chance
  and not far under the control", and that is what she does.
- Not read at any age: whether she holds something, whether a mark exists
  (from the body sense; .87 / .72 against base .90 / .71).
- **The cost is the open problem.** Classes in use stay near 100–150 per
  organ while the total passes 1,700; time per day grew 100-fold and the
  saved body 5-fold. At this rate a longer life is not affordable on this PC.
- One organism, one seed. She is 120 days old and stays in the fetal stage:
  2b (choosing) is not designed, and the stage file is not changed by code.

## 9. Why her classes are flat, and two probes on fresh toy leaves (s067; NOT Vee; nothing here is adopted)

### 9.1 The composer never held a trial (read from Vee's saved bodies)
Rocky asked whether the composer is triggered by limited resources, in dreaming,
or is mis-wired. None of these. It is tried at every sitting, beside division;
in all three organs it is on, not frozen, attached to the stress economy, with
0 of 6 cells used and room in the arena. **Its trial counter is 0 after 120
days.** A trial is held only for a class whose members disagree on the raw
readings (mean coherence < .55). Her large classes: ear .82 lowest / .94 median,
body sense .94 / .99, sense of change .96 / .98. Division slices every mixed
class tight before it can qualify. Chloe's earlier claim that composer cells
are "shared parts" was overstated: a cell makes one new feature from two
readings (a relation); it does not make "busy" a part many classes reuse. That
mechanism does not exist in the phasecyte.

### 9.2 Composer probe (`composer_probe.py`, log `composer_probe_s067.log`; 2 seeds)
Task: "these two readings are the same / different", six other readings present.
Rocky: "when you say fish, it is fish all the time. In the human world fish can
be fiiish, fishhh, fsh."

| variation between moments | other readings | matron | trials | composer cells | classes | same/different on unseen |
|---|---|---|---|---|---|---|
| .05 (Vee's world) | steady | silent / labels | 0 / 0 | 0 / 0 | 12 / 13 | .60 / .73 |
| .05 | on/off | silent / labels | 48 / 82 | 1.5 / 2.0 | 74 / 100 | .55 / .52 |
| .3 | steady | silent / labels | 0 / 0 | 0 / 0 | 115 / 119 | .91 / .92 |
| .3 | on/off | silent / labels | 252 / 251 | 2.0 / 2.5 | 258 / 211 | .87 / **.97** |
| .6 | steady | silent / labels | 0 / 0 | 0 / 0 | 367 / 351 | .79 / .79 |
| .6 | on/off | silent / labels | 0 / 0 | 0 / 0 | 1 / 1 | .48 / .48 |

- Variation in the world is the big lever, up to a point: .3 teaches the
  relation, .6 drowns it. The matron's labels help on top (.87 → .97; .60 → .73).
- The composer fires only with on/off other readings, and much more with
  variation. With steady other readings the relation is still learned at .3,
  through many classes and no composer.
- With the two readings alone the leaf never divides (1 class) — not diagnosed.

### 9.3 Organ-blur probe (`blur_probe.py`, log `blur_probe_s067.log`; 3 seeds)
Rocky: "baby eyes are blur when they were born"; sharpen "at very slow rate
probably relative to her size". Task: one reading is 0.2 or 0.6 (a fine
difference), four on/off readings beside it, the matron names every moment.
The developing organ sees each reading in 2 steps at first; it may sharpen by
one step only when (a) it has not divided for 3 + classes // 8 sittings and
(b) the matron still corrects it; a step is on trial and is taken back if the
corrections do not fall.

| world variation | organ | steps / taken back | final steps per reading | classes | high/low on unseen | still corrected |
|---|---|---|---|---|---|---|
| .05 | sharp from the start | – | full | 387 | .96 | .05 |
| .05 | developing | 1.7 / 0.7 | 3 | **92** | **.99** | .01 |
| .15 | sharp from the start | – | full | 587 | .85 | .15 |
| .15 | developing | 1.7 / 0.7 | 3 | **121** | .77 | .21 |

- **A blurred, slowly sharpening organ needs four to five times fewer
  classes.** In the clean world it also reads better (.99 against .96).
- **In the more variable world it got stuck too blurred** (.77 against .85)
  while still being corrected. Going from 3 to 4 steps happens to put both
  values in one step, so the trial fails and is taken back; one-step-at-a-time
  cannot get past a worse step to a better one. The slow-by-size patience
  (18 sittings at 121 classes) also rarely comes round while she still divides.
- At 3 steps the edge happens to fall between 0.2 and 0.6: part of the clean
  result is luck of where the steps lie. Uniform steps over a fixed range are
  crude; sharpening per reading, where the correction comes from, is untried.
- Blur here is applied in front of the leaf (the leaf itself is unchanged).
  Senses were formed on full-resolution readings, as a receptor that saw only
  two levels at genesis would stay a two-level receptor.

## 10. Rocky's rulings at the end of s067, and what is to be built next (NOT built)

| question | Rocky |
|---|---|
| Is fetal stage 2a passed? | **Yes.** |
| May her scripted world vary more from moment to moment? | "Vary is okay but probably in certain corridor? Vary is okay but not too random." |
| More toy tests of the sharpening rule? | "No, I think the test is enough and can be implemented." |

### 10.1 World variation in a corridor (to build in `world.py`)
Each state keeps its typical reading, but the reading wanders inside a bounded
band around it instead of sitting on one value with tiny noise. Chloe's
proposal from the probe (§9.2: .3 taught, .6 drowned, on readings spanning
−1..1): moment-to-moment variation of about .15–.3 of a reading's range,
clipped at the band's edges so two states never overlap. Applies to the world,
so it can be given to an organism that already lives.

### 10.2 Organ blur (to build in `vee_body.py`, in front of each organ; the phasecyte itself unchanged)
- Each organ sees each of its readings in a small number of steps over the
  range that reading showed at genesis; every reading starts at 2 steps.
  Senses still form at genesis on full-resolution readings (§9.3).
- **Sharpening, one reading at a time, one step at a time, slowly:**
  only after the organ has not divided for a number of sittings that grows
  with its class count ("relative to her size"); the reading chosen is the
  one hiding the most detail inside its current steps.
- **After birth** a step must also be called for (the matron still corrects
  her) and is on trial: taken back if the corrections do not fall. A failed
  step may be retried one step further (the probe got stuck without this).
- **Before birth there is no correction signal.** Proposal: settled-for-long-
  enough alone allows a step (maturation), with no trial. To confirm with Rocky.
- Blur makes sense only from genesis: Vee already holds 2,675 sharp classes.
  **Agreed by Rocky (s067: "Okay, we can start from the genesis. I agreed on
  your suggestion"), together with the pre-birth rule above:** raise a blurred SISTER in her own home (`VEE_HOME`) through the
  same embryo and fetal gates, in the corridor world, and compare with Vee at
  equal age (classes, time per day, every gate row). Vee is kept untouched.**
  Which of the two goes on to birth is Rocky's decision after the comparison.

## 11. The sister: a new genesis (s068; Rocky: "lets start with a new genesis")

**Names (Rocky, s068): the first sister is Weiss, the second is Wile.** Homes:
`outputs/incubator/weiss/` (was `sister/`; her s068 logs are named `sister_*`) and
`outputs/incubator/wile/` (was `sister2/`; logs `sister2_*`). Wile = Weiss's
germline plus "none" and the ladders from her first moment (§11.5).

Weiss: a second organism, raised from genesis in her own home (`VEE_HOME`).
Vee is not touched. Same germline as Vee except the two things under test: the
same 148-number layout, the same language sense (Vee's `sense.pt`, copied), the
same seed (0), so she lives the same course of world events and hears the same
sentences as Vee did, day for day.

### 11.1 The corridor world (BUILT, `world.py`, `World(seed, corridor=c)`)
A reading that tells a state (RAM, swap, load, CPU busy, processes, traffic)
wanders inside a band around that state's typical value and is clipped at the
band's edge. With 0 = the quiet value and 1 = the other, the bands are
[−c/2, c/2] and [1 − c/2, 1 + c/2]; **c = 0.6** (Chloe's value, from the s067
probe: variation .3 taught, .6 drowned): the bands never touch. Each moment
keeps half of where the reading was (it wanders, it does not jump). Measured on
her readings (−1..1): spread .13–.23 within a state for CPU and memory, .08 for
traffic; Vee's world had .1–.2 for CPU, .1 for memory, less for traffic.
Checked: with the corridor off the world is identical to Vee's, number for
number, over 40 days with kicks; with it on, the hidden course is the same.

### 11.2 Organ blur (BUILT, `blur.py`, `vee_body.Body(blur=True)`)
- Senses form at genesis on the full readings. From her first moment each organ
  sees each reading in **2 steps** over the range it showed at genesis.
- **When an organ sharpens (before birth):** it has not gained or lost a class
  for 3 + classes // 8 sittings (the probe's values). Then one reading gets one
  more step. No trial, as agreed.
- **Which reading:** the one whose present steps hide the most of what the
  world showed in her last 10 sittings (spread inside a step, as a share of the
  reading's range). Chloe's reading of "hiding the most detail". Known
  weakness: before birth she cannot tell detail from noise, so a reading that
  merely wanders may be sharpened.
- **After birth:** a step must be called for by corrections and pay. NOT
  built; the code takes no step once the stage file leaves embryo / fetal.

### 11.3 Embryo gate
Unchanged (§7.3 with Rocky's s067 rulings): busy, memory, place, window against
k-means on the full, unblurred numbers; night, class count, stability reported.
The time alarm is now an absolute budget (30 s a simulated day, rule 9).

### 11.5 The ladders, and her sense of change (Rocky's rulings s068; BUILT, not yet run on her fetal stage)
- Rocky: "Three states is mandatory", then "probably start at 2, go to 3, to
  5, then to 7", then to Chloe's two questions: "1. yes. 2. 2>3>5".
- **A level** (ear, body sense, the echo flags): **2 → 3 → 5** steps, then no
  further. This replaces "one step at a time" in §11.2. No step had been
  taken when it was changed (sister, embryo day 17+), so nothing she lived
  is rewritten.
- **A change** (the 35 numbers of her sense of change that say how a reading
  moved): **2 → 3 → 5 → 7**. 2 = same / changed (how much it moved, either
  way), 3 = down / same / up, then finer grades. The range is symmetric about
  zero and "same" is its middle third at every rung, so "did not move" is a
  state of its own from the first moment and no cut ever lies on zero.
- Chloe's choices inside the ruling: the 8 echo numbers in that organ are
  levels (at same / changed "I did this" and "I did not" would look alike);
  "same" = the middle third of the range seen at genesis.
- **"None"** (Rocky s068: "I meant yes add none"; BUILT, `Blur(none=…)`,
  `embryo.py BLUR=2`): a reading that can be silent — the 64 language numbers,
  the four readings through the window, the echo flags, the reserved channels —
  is seen as exactly zero when it reads zero, a state of its own. What she
  holds and her marks are not in the list (zero there is a real amount).
- Found while building it: 16 of the body sense's 81 readings showed 7 or
  fewer distinct values at genesis (day of week, place, rain, someone here,
  window open, presence). They are already in steps, and blurring them did
  harm: the first sister sees "window closed" and "raining" as one value, and
  hears silence as a fixed pattern of step centres. In a body born with
  "none" such a reading passes unchanged.
- **"None" changes what she sees from her first moment, so it applies only to
  a body born with it.** The first sister (`outputs/incubator/sister/`) was
  born without it and is read as she was made.
- `fetal.py`: her sense of change is formed in her corridor world, the gate
  reads it through her blur, and the unseen days come from a corridor world.
- Her fetal stage is NOT entered: the embryo gate is read by Rocky first.
- `world.py`, old quirk: a work kick in an already busy world lowers the CPU
  reading to .85. Left as is so Vee's world stays identical.

### 11.6 Weiss's embryo gate (s068; logs `sister_embryo_s068.log`, `sister_embryo_gate_s068.log`; one organism, seed 0, same course of events as Vee)

| | Vee | Weiss |
|---|---|---|
| busy / memory full / place / window (k-means) | .99 / .95 / 1.00 / 1.00 (1.00 / .99 / .93 / .99) | 1.00 / 1.00 / 1.00 / 1.00 (1.00 / .99 / .90 / .99) |
| night, reported | .55 (.89) | .92 (.85) |
| ear: text present | .75 | .93 |
| stability day 20 → 30, same grouping | .61 | .89 |
| body sense classes day 20 → 30; ear | 86 → 141; 13 | 61 → 82; 44 |
| composer cells ear / body sense (trials) | 0 / 0 (0) | 4 / 0 (55 / 0) |
| time per simulated day | 0.5 s | 126 s (last five days); stopped once by the 30 s guard on day 6, budget raised to 900 s (Rocky: "we can raise the daily budget") |
| sharpening steps | – | none in 30 days: never settled for 3 + classes // 8 sittings |

- The gate passes every row. Corridor and blur are tested together; which of
  them gave what is not known (no sharp organism was raised in the corridor).
- Night is partly luck of the cut: one clock number cut in the middle splits
  the day at 06:00 / 18:00.
- The cost is the ear's composer search (it never held a trial in Vee).
- **Rocky, s068: "Yes"** to her fetal stage. Stage file set to `fetal` by
  hand; her own embryo values are in `embryo_values.json` for the "nothing
  lost" rows. Her sense of change is born at same / changed (§11.5).

### 11.7 Wile's embryo gate, and Weiss's first 30 fetal days (s068; logs `sister2_embryo_gate_s068.log`, `weiss_fetal_s068.log`, `weiss_fetal_gate_s068.log`; one organism each, seed 0)

Wile at embryo day 30 (gate passes every row): busy 1.00 / memory .96 / place
1.00 / window 1.00; night .91; ear text present 1.00 (Weiss .93, Vee .75);
stability .87; classes body sense 100, ear 54; 5 composer cells in the ear;
2 sharpening steps (ear, by day 4), none in the body sense; 77 minutes.
"None" made the ear read silence perfectly; it did not make her cheaper.

Weiss at fetal day 30 (60 days old), against Vee at the same age (§8.11):

| criterion | Vee, fetal day 30 | Weiss, fetal day 30 | |
|---|---|---|---|
| feels her kicks | .99 (k-means 1.00) | .90 (k-means 1.00) | **FAIL** (more than .05 under the control) |
| what a kick moves | 3 of 5 | **5 of 5**, each 1.00 | pass |
| CPU rise hers or the world's, echo hidden | .98 | .95 (k-means 1.00) | pass |
| memory rise hers or the world's, echo hidden | .62 | .59 (k-means .97, base .59) | **FAIL** |
| nothing lost: busy / memory / place / window | .99 / 1.00 / 1.00 / .99 | .99 / **.90** / .96 / .96 (her embryo: 1.00 each) | **FAIL** on memory |
| the ear: text present | .65 | .93 | pass |
| classes ear / body sense / sense of change | 67 / 429 / 141 | 88 / **516** / 61 | reported |
| … in use in her last 10 days | 42 / 69 / 90 | 68 / 71 / 34 | |
| time per simulated day | 3.8 s | 18 s (on the faster composer search) | |
| sharpening steps | – | none, in any organ, in 60 days | |

- **Gate not passed at day 30. Vee did not pass at day 30 either and passed at
  60 and 90: "not yet", not "no" (rule 8).**
- **The body sense's class advantage is gone**: 84 classes at fetal day 1,
  516 at day 30 — more than Vee's sharp body sense at the same age. It began
  when the echo woke. Not diagnosed.
- Her sense of change, born at same / changed, connects every kick to what it
  moves (Vee needed 90 days for 5 of 5) with less than half of Vee's classes.
- No organ of Weiss has ever sharpened: she is never settled for
  3 + classes // 8 sittings. The sharpening rule is untested on a living body
  except Wile's ear (2 steps in her first 4 sittings).
- The composer search was rewritten to carve all candidates in one pass
  (`trioron/pcll/composer.py`; same candidates, order, draws and thresholds).
  Same class counts on every one of 12 days from genesis with old and new;
  about 5× faster.
- Stopping and resuming is not exactly the same as living straight through:
  Weiss, stopped at embryo day 6 by the guard, shows 29 / 32 body-sense
  classes on days 10 / 11 where an uninterrupted run gives 28 / 31 (equal
  again on day 12). Not diagnosed.

### 11.8 The cranium (Rocky s068; NOT built — to be written for sign-off)
Rocky: there must be a cap on the comparing; leave what is unfinished when
sleep is over and let it sediment; when the cranium is full, work through it
in instalments and recycle; an adult adds a fact only for a real discovery.
"The cranium is the RAM … all of this is already in trioron design."
Chloe checked: it is (spec §1.2, §2.5 pressure / niche capacity / stagnation,
§4.3 bounded and scaled dream caps, §5.5 compaction in instalments, §6.1 the
incubator's 85 % of RAM, §10.6 habituation and deferral). **What is missing is
the wiring to the incubator organisms:** (1) their envelope has no cap, so
pressure is 0.0 for life; (2) the 85 % RAM limit in `vee_body.py` is a wall
that stops the run, not a pressure she feels; (3) the phasecyte's own sitting
has no cap on classes judged or composer trials held — this is what makes
them slow; (4) not checked: whether a class's astrocyte row carries the
`recyclable` gene, i.e. whether compaction can reach an unused class.
To propose: A. her envelope = her real cranium (`max_parameter_bytes`);
B. the sitting under a bounded, scaled cap as in §4.3, busiest classes first,
the rest left for the next sleep (a spec §10 change: needs Rocky's yes).

### 11.9 The cranium wiring: checked, and written for sign-off (s069; NOT built)

Read from saved copies only (Weiss fetal day 30, Vee fetal day 90; nothing changed):

| organ | classes | rows used of 8192 | retired rows | rows with `recyclable` | bytes the envelope counts | held in class buffers | classes under 60 / under 25 members |
|---|---|---|---|---|---|---|---|
| Weiss ear | 88 | 378 | 197 | 0 | 2 KB | 2.6 MB | 78 / 0 |
| Weiss body sense | 516 | 1170 | 548 | 0 | 5 KB | 22.3 MB | 405 / 132 |
| Weiss sense of change | 61 | 201 | 77 | 0 | 1 KB | 1.9 MB | 44 / 13 |
| Vee ear | 344 | 1502 | 1069 | 0 | 6 KB | 38.2 MB | 142 / 68 |
| Vee body sense | 1781 | 3918 | 2031 | 0 | 16 KB | 92.0 MB | 1444 / 616 |
| Vee sense of change | 550 | 1283 | 670 | 0 | 5 KB | 45.8 MB | 159 / 46 |

**What the check found.**
1. **No class row carries the `recyclable` gene** (0 of every row in six organs). A class's row is
   born with an empty epigenome (`mixed.py _refresh`, `controller.py _sync_astrocytes`), and nothing
   on the phasecyte's path calls `compact()`. A retired row (a divided parent, a husk) is kept for
   life: about half of all rows are retired ones.
2. **There is a wall nobody chose: 8192 rows per organ** (`PhasecyteLeaf(capacity=8192)`, a fixed
   arena). A class costs about 2.2 rows over its history, so an organ runs out near 3,600 classes.
   Vee's body sense has used 3918. Past the wall a class is born without a row (`cell_id = -2`),
   silently: it still works, but it is no longer visible in the arena.
3. **The envelope's byte count cannot see her.** `arena.param_bytes` = 4 bytes a row + 4 an edge:
   16 KB for Vee's body sense. What she holds is in her classes' buffers (up to 400 members a
   class): 176 MB of Vee's 230 MB, 27 MB of Weiss's 72 MB. With `max_parameter_bytes` set from RAM
   as §11.8 proposed, pressure would read 0.00 for life. **Chloe was wrong to propose A in that
   form (s068).**
4. **Counted rightly, her cranium is still almost empty.** 85 % of this PC's 7.6 GB is 6.5 GB; Vee
   at 120 days holds 0.18 GB, about 3 %. Pressure .85 lies tens of thousands of classes away. RAM
   is not what makes the sisters slow, and "work through it in instalments and recycle at full
   cranium" will not be called on for a long time. Time is the wall she meets first: B below.
5. **Her organs almost never have a settled sitting.** Consolidation (the round that re-settles
   members and retires husks), annealing and the skeptical gate run only at a sitting with no
   division and no spawn (spec §10.7). With hundreds of classes some class divides at nearly every
   sitting: the settled streak is 0 in all six organs as saved, and the class count stood still
   over a day on 2 of Vee's 90 fetal days (body sense) and 1 of Weiss's 30. The result is crumbs:
   405 of the 516 classes of Weiss's body sense hold under 60 members, too few to be judged at
   all. Sharpening waits on the same stillness, which is why Weiss never sharpens. Chloe's reading
   for open point 1 (84 → 516), from saved state and logs; divisions per sitting are not logged, so
   it is not tested.
6. **The merge is not switched on in her organs** (`PhasecyteLeaf` never passes `merge=True`). The
   s066 note "the merge has never fired" should read "is off".
7. **Every class is judged at every sitting**: every buffer for division (4 readings each), and
   every buffer of 100+ members with low coherence for a composer trial, whether or not it received
   a single new member since it was last judged.

**To build, on Rocky's yes (decisions D1–D5).**
- **A. Her cranium, counted truly.** One cranium for the whole body = 85 % of host RAM. Her
  pressure = everything her organs hold (class buffers, tags, sketches, lock-in rows, parameters)
  over that. The arena's byte count takes in what astrocyte rows hold (one sentence added to spec
  §2.5; §2.11 already says an astrocyte holds parameters). The present stop at 85 % stays as the
  last guard. It will read about .03 today.
- **A2. No hidden wall.** The arena grows when its rows run out (doubling), so the cranium is her
  only limit. Core code (`core/arena.py`).
- **B. The sitting under a cap (changes spec §10.7; rule 25).**
  - B1. A class that received no new member since it was last judged is not judged again.
  - B2. Of the rest, the council judges at most `max(4, ceil(sqrt(classes) · 0.5 · φ))` classes a
    sitting (spec §4.3's form, the lock rate; φ as there, from her growth at the last sitting),
    the busiest first (most new members this sitting; ties to the one unjudged longest). The others
    are left for the next sleep. At 516 classes that is 12 a sitting at rest; at 1781, 22.
  - Work is logged per sitting (classes judged, trials held, divisions, spawns, retirements), and
    sleep is measured in work, not seconds. The floor of 4 and the rate .5 are Chloe's values.
- **C. Recycling, made possible but not called.** Class rows are born with `recyclable`. Before
  compaction may ever run on her, its re-packing must carry her own references along (class row
  ids, lineage, composer cells, receptors): today it would break them. Not urgent (finding 4).
- **How it is tried.** On copies first (rule 27): a copy of Weiss at fetal day 60 lives 10 days
  with and without B; classes, every gate row and work are compared. Only then, on a further yes,
  on a living organism. A rule changed in mid-life is a new variable in Vee against her sisters.

**Decisions for Rocky.** D1 A as corrected. D2 A2. D3 B1 and B2 with these values. D4 C now or
later. D5 finding 5 is not the cranium: should the never-settling (and the merge being off) be
diagnosed on copies next, before or after B?

### 11.10 Rocky's rulings on §11.9, what was built, and the trial on copies (s069)

**Rulings (his words).** On the cranium: "It is for them in this environment. It is not the
default so no specs changes is required." On the 8192 rows: "Yes and remove the class limit"
(the organisms already had none: `class_cap=None`; the benches' 43 / 128 defaults were left).
On the cap: "Square root". On compaction: "if the cranium is not problem, compaction will make
things slower isn't?" (left unbuilt). On the merge: off because an earlier experiment confused
the organism (the record agrees: design doc `mixed_stream_growth.md` D18, s033, −.265 then
−.147 with the loose merge; the strict one fired zero times). On the consolidation round:
"Yes on the round it should run on the sittings. Sediment can be dug up but occasionally."

**Built (all opt-in or behaviour-neutral for a living organism).**
- `core/arena.py` `Arena.enlarge()`: the arena doubles at a boundary when an uncapped
  organism's class rows run out (`mixed.py _refresh`, 64 rows of headroom). `alloc` unchanged.
- `trioron/pcll/sitting.py` `SittingCap` (off unless `mixed.sitting` is set): a class with
  nothing new since it was last judged is not judged; of the rest at most
  `max(4, ceil(sqrt(classes) · 0.5 · φ))` a sitting, busiest first; the consolidation round at
  every sitting over the judged classes plus `max(1, cap // 4)` dug from the sediment (longest
  without a round). φ from growth at the previous sitting only. Floor 4, rate .5 and the
  quarter are Chloe's values. Work is recorded per sitting (`sitting.work`).
- NOT built: the cranium's byte count (A), compaction / `recyclable` (C).

**Trial on three copies of Weiss from fetal day 60** (`outputs/incubator/copies/weiss60_*`;
logs `cap_probe_*_s069.log`; one organism, seed 0; 10 days each):

| | no cap | cap only | cap + round |
|---|---|---|---|
| days lived | **8: killed on day 69, three times** | 10 | 10 |
| body-sense classes (from 2086) | 2738 | 2097 | 2106 |
| ear / sense of change | 148 / 127 | 150 / 116 | 156 / 115 |
| seconds a day | 27–156 | 15–188 | 24–58 (eight of ten 34–44) |
| feels her kicks (k-means 1.00) | .94 | .94 | .94 |
| CPU / memory rise, hers or the world's | .98 / .59 | .95 / .59 | .95 / .59 |
| nothing lost: busy / memory / place / window | 1.00 / .93 / .98 / .98 | .99 / .93 / .98 / .98 | 1.00 / .93 / .98 / .98 |
| the ear: text present | .95 | .93 | .98 |

- **The uncapped organism meets a real memory wall near 2,700 body-sense classes.** The kernel
  killed her (out of memory, 7.0 GB resident) on day 69 at every attempt. Chloe's reading, not
  profiled: the full consolidation round, which compares every buffered member with every
  class at once on a settled sitting. Her state is saved after each day, so nothing was lost,
  but she cannot go on. The 85 % guard in `vee_body.py` is read only before a sitting and did
  not see it. Vee (1781) and Weiss (2086) are both close to this wall without the cap.
- The cap stops the class explosion; no gate row moved by more than .03 in 10 days.
- The cap alone leaves spikes (98, 113, 188 s); with the round in instalments the worst day
  was 58 s.
- The sediment is not shrinking yet: 1545 of 2106 body-sense classes under 60 members; 23
  judged and 27 settled a sitting; 1920 classes still waiting to be judged for the first time.
- Her ear spawns a composer cell at every sitting (1.0 a sitting over 20). Not examined.

### 11.11 Her buffers are mostly her own replays; a test without them (s069; copy of Weiss from fetal day 60)

Rocky: "7 GB is a huge tons of data … she was feed with small numbers of data isn't?" Counted
on the killed uncapped copy: 9,408 moments lived (1.4 M numbers), but her body sense holds
258,259 buffered members in 2,738 classes (at day 60: 172,812 from 8,640 moments). About 95 %
of her buffers did not come from the world. The only step that adds members is annealing
(`mixed.py _anneal`: 40 synthetic members per class from its own sketch at a settled
sitting). The kill is the full round's workspace: 258,259 × 2,738 = 707 M pairs, 2.8 GB a
table, several tables at once (Chloe's reading of the code, not profiled).

Test (`copies/weiss60_noanneal`, switch `mixed.no_anneal`, no cap; logs
`cap_probe_noanneal*_s069.log`; 8 days):

| body-sense classes from 2086, after 8 days | members held |
|---|---|
| plain: 2738 (+652) | 258,259 |
| no new synthetic members: 2327 (+241) | 172,557 (unchanged) |

- New synthetic members account for roughly 60 % of the multiplying over these 8 days; she
  still gains 241 classes on 768 moments of world. Gate rows equal (±.03).
- Not a clean test: the 164,000 synthetic members made before day 60 stayed in her buffers
  and were still judged. An organism raised without annealing from genesis was not tried.
- Rocky: "it is a missing piece that need to be sewn to the triorons itself the
  unforgettfulness actually a double edge sword". Chloe's proposal, NOT signed, NOT built:
  fold a long-unused class back into the sibling / parent it was split from (sketches pool
  exactly; a boundary is forgotten, not an experience); try on a copy, earliest classes
  reported separately (rule 22).
