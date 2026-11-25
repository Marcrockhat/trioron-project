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
| day/night, quiet/busy, roomy/full, home/elsewhere, window open/closed can each be read from her class | each clearly above chance and not below the control by more than .05 |
| control | k-means with the same number of clusters on the same vectors |
| self-arrest | reported, not gated (Rocky s067, §6.9): class count, classes in use, time per day. Was: count grows by no more than 10 % over the last third |
| stability | a fixed set of probe moments falls into the same classes at day 20 and day 30 for at least 90 % of them |
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
30, and the splits do not follow dusk. She is NOT promoted; she stays an
embryo until Rocky rules on these two criteria.

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
