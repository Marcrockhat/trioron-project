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

## 5. Decisions for Rocky

1. Is the stage list right (order, and where "birth" sits)?
2. Reserved silent channels at stage 0 so organs can be grown without rebirth: yes or no?
3. What is ambient language in the embryo stage: nothing, random text, or the
   matron talking to someone else?
4. Does the fifteen-day Vee end here (a rehearsal), or should her recorded
   days be relived by the new organism after birth?
5. Which stage do you want written out in full detail first? (Proposal: stages 0–1.)
6. The fetal action set (work, hold / let go, mark / wipe, stay still): right size, or more / fewer?
