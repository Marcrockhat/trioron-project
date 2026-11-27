# Trioron Handoff

**Session date:** 2026-11-27
**Session number:** 072
**Session title:** The Numa trial lengthened; one random channel; the fourth organ built as a living organ; the organs separated from the core (`trioron/organs/`); Vee, Weiss and Wile retired, a new organism to be raised

**Summary.** (1) Rocky: "the numa gate seems not enough and make the organism learn less".
The trial length was changed first (decision H), tried on a second copy: 4 Numa in 349
divisions, gate not passed at day 60 or 90. (2) He asked for one input channel of random
numbers, as a stand-in for the real world's randomness (the rest of the world stays varied,
not random). Built on copies only. Chloe predicted the Numa rule could not reject divisions
on that channel and was wrong: none of them settled. (3) Rocky: "Lets put the fourth organ
and wrap all the three siblings. Rebuild the code, and make sure we clearly separate the
organs and the triorons core." Done in code: `trioron/organs/` holds everything a body is
made of; the core names no organ rule; the fourth organ is a living organ that tries
divisions outside the working organ; the gate reads both ways; a relative gate (`PROBE`)
stops the fetal stage at the first pass. One copy trial with the fourth organ passes the
fetal gate at day 90 with 354 body-sense classes (Wile needed 4,540). **But the control
(the same copy with no division ever carried out, 346 classes) passes at day 90 too: the
pass is not the fourth organ's doing.** The body is now in `trioron/organs/` as well.
(4) Vee, Weiss and Wile are retired (homes kept). **The new organism is NOT born: it needs
Rocky's name and his yes.**

## READ THIS FIRST (s073)
- Read `docs/design/matron_rules.md` and `docs/design/vee_development.md` **§13.9, §13.10
  and §14** before touching an organism. The plan still has NO s070 section (the s070
  handoff at `0247e27` is that record).
- **No living organism. Three retired, on this PC only, uncommitted. Never delete, overwrite
  or refit any home:**
  - `outputs/incubator/wile/` — Wile, fetal day 90, gate passed (s071). RETIRED s072.
  - `outputs/incubator/weiss/` — Weiss, fetal day 180, gate not passed. RETIRED s072.
  - `outputs/incubator/vee/` — Vee, fetal day 90. RETIRED s070. Her `sense.pt` is the
    language sense a new organism copies into its home before genesis.
  Rocky, s072, asked what "wrap all the three siblings" means: **"Retire all three, raise a
  new one"**. None of the three was touched this session (only read, for copies).
- Copies (not organisms, may be deleted) under `outputs/incubator/copies/`, new in s072:
  `wile30_numa2`, `wile30_noise_numa`, `wile30_noise_plain`, `wile30_fourth`, `wile30_still`.
- **The new organism waits on Rocky** (see "Decisions waiting"). Nothing was born, no home
  made, no stage file written.

## Rocky's rulings this session (his words)
- "Yeah change the trial lenght first. I forgot about that. On 1. it is b. on 2. Test the
  lenght first." (1 b = the fourth organ as a *living* organ, not a record-keeper.)
- "Can you add one factor that only filled by random numbers?" Then: "I mean I agreed that
  it is varied and not random, but judging on how the architecture work, a channel of random
  value is needed probably to simulate the real world nad only this channel is random, the
  rest is not."
- "I think we can agreed that the enchancement fix our problem but too long fetal days is
  bad too (as you meant by giving meanings). Lets put the fourth organ and wrap all the
  three siblings. Rebuild the code, and make sure we clearly separate the organs and the
  triorons core."
- Asked four questions, he chose: siblings **"Retire all three, raise a new one"**; the
  fourth organ **"As drafted, gate reads both"** (plan §14.5 J1–J5); how the fetal stage
  ends, in his own words: **"You need to put a probe when the gate passed and then the fetal
  state is passed (a relative gate)."**; the code **`trioron/organs/`**.

## 1. The code, rebuilt in three layers (spec §9.17; all 32 phasecyte tests pass)
1. **The learners** — `trioron/core`, `learning`, `lifecycle`, `pcll`. They import nothing
   from `organs/` and name no organ rule (`test_core_names_no_organ_rule` enforces it).
   `pcll/mixed.py` keeps three empty places an organ may fill: `sitting`, `numa`, `court`.
2. **The organs** — `trioron/organs/`: `layout.py` (the 148 numbers, `can_be_silent`,
   `changes`), `home.py` (`HOME`, the stage file, the RAM ceiling), `body.py` (`Body`),
   `blur.py`, `sitting.py` (`SittingCap`), `numa.py` (`NumaLedger`), `fourth.py`
   (`FourthOrgan`). Imports the learners, never `experiments/`.
3. **The world** — `experiments/incubator/`: world, drivers, gates, probes.
   `vee_body.py` keeps only `moment`, the genesis pools and the one-time fitting of the
   language sense, and re-exports the body under its old names.
- **Old bodies still load:** aliases for `trioron.pcll.numa`, `trioron.pcll.sitting` live in
  `organs/__init__.py`; `experiments/incubator/blur.py` and `vee_body.py` are shims. Checked:
  Wile's body loads; the day-90 gate of `copies/wile30_fourth` prints identically before and
  after the body was moved.
- `MixedStreamController(sitting_cap=True)` now raises and points to
  `leaf.mixed.sitting = SittingCap()`.

## 2. The Numa rule, trial lengthened (plan §13.9; `copies/wile30_numa2`; one copy, seed 0)
A pair now stays on trial until it has its batches; it is taken back when a batch fails or a
child goes unfed by the world for 60 boundaries ("starved"). Result from Wile's day-30 body:
4 Numa in 349 divisions; gate not passed at day 60 (memory full .90) nor at day 90 (memory
full under the line, "work" kick 0 of 18); 196 pairs still on trial. 3 of 17,280 world rows
were refused by every class, so a record-keeper fourth organ would never be consulted.
Chloe's reading: 48 world moments a sitting cannot confirm divisions that need 100 members;
the batch (25 × 2) is out of scale with his world. Logs `numa_probe2_s072*.log`.

## 3. One random channel (plan §13.10; copies only; one seed)
Input channel 140 (a spare) carries N(0,1) every moment, from its own generator
(`NOISE_CH=140`; `fetal.py` refuses it outside `copies/`). It reaches the body sense only
(its column 73).
- Rule on (`wile30_noise_numa`): gate **passes at day 60** (memory full .93; 426 body-sense
  classes), **not at day 90** (memory full .90, "work" 0 of 18).
- Rule off (`wile30_noise_plain`, day 60): 885 body-sense classes, 33 of 569 divisions on
  the random column, gate not passed. (Wile at day 60 without the channel: 1,352 classes;
  why fewer with it is not diagnosed.)
- **Chloe was wrong:** she predicted the Numa rule could not reject divisions on the random
  column. None of 24 settled; 8 were taken back, all "starved": one child receives every
  world row. Why is not diagnosed.
- The losses between day 60 and 90 appear in both rule-on copies, with and without the
  channel: they come with the rule in the working organ, not with the channel.
- **Rule 16 ("never at random") is NOT amended.** It needs Rocky's wording before the
  channel is part of a living organism's world.

## 4. The fourth organ, living (plan §14; `trioron/organs/fourth.py`)
A candidate division is not carried out in the working organ. It is held in the fourth organ
as a living pair, fed by the world's rows only, judged by the Numa test (25 × 2). What
settles is carried out in the working organ; what fails or goes unfed falls asleep (60
sittings) and may be woken; held children may divide inside (depth 3); one living pair per
class and column. Switch on: `leaf.mixed.court = FourthOrgan()` on each organ, then save.
- **Different from the draft, said to Rocky:** a moment falls to a held child by the
  children's templates over ALL the organ's columns, not by the split column alone (by the
  split column alone the test would pass by construction).
- **The gate reads both ways** (`fetal.py gate`): the working organs alone, and with the held
  children added (`FourthOrgan.refine`). Chloe built "reads both" as: passed when ONE reading
  passes every row; both printed. Rocky has not confirmed this reading of his answer.
- **The relative gate:** `PROBE=n` reads the gate every n days while living and stops at the
  first pass (copy `body_fetal_gate_day{N}`, organism HELD). It promotes no stage.
- **Copy trial** (`copies/wile30_fourth`, random channel on, Wile's day-30 body → day 90;
  logs `fourth_probe_s072*.log`; one copy, seed 0):

| | day 60 | day 90 |
|---|---|---|
| classes ear / body sense / change | 119 / 348 / 182 | 119 / 354 / 182 |
| feels his kicks | .89 FAIL | .97 |
| what a kick moves | 5 of 5 | 5 of 5 |
| CPU / memory rise | .98 / .97 | .98 / .97 |
| memory full, working organs / with held children | .90 / .90 FAIL | .91 / .95 |
| gate | not passed, either reading | **passed, both readings** |
| body sense: heard / settled and carried out | 718 / 2 | 1,162 / 8 |
| pairs on the random column: held / settled | 49 / 0 | 84 / 0 |
| seconds a day | 45 | 68 |

  Almost nothing was carried out (8 of 1,162; none in the ear or the sense of change), so
  the working organs at day 90 are nearly the day-30 body. The pass on the working organs
  alone is thin (rows at the line). The held children add .04 on memory full.
- **The control** (`copies/wile30_still`: a fourth organ that can never settle a pair, so the
  working organs get no division at all; logs `still_probe_s072*.log`; plan §14.8): classes
  stay 119 / 346 / 182; day 60 not passed (kicks .89, memory full .89); **day 90 passed, both
  readings** (kicks .97, memory full .91, "work" 18 of 18).
- **So, said plainly:** a body that has not gained one class since day 30 passes the fetal
  gate at day 90. The pass is not the fourth organ's doing. What helps is that divisions are
  kept OUT of the working organs; the failing rows rise with the days. Wile's 4,540 classes
  were not needed for this gate, and Chloe's earlier reading that memory full needs real
  growth is not supported. The fourth organ's only credit is .95 against .91 on memory full
  in the second reading: one copy, one seed, a row at its line (needs .91).

## Decisions waiting on Rocky
- **K. The new organism.** Its name; his yes for genesis. And with it:
  - K1. Is the random channel part of its world from genesis? Then rule 16 needs his wording,
    and the `copies/`-only assert in `fetal.py` is lifted for it.
  - K2. Born like Wile (blur with "none", ladders from the first moment)? Chloe's proposal.
  - K3. The relative gate's interval (Chloe's value: every 10 days), and whether the embryo
    stage ends by a relative gate too.
  - K4. The fourth organ from genesis on all organs. **Said plainly:** a newborn starts from
    one class an organ and can grow only as fast as the fourth organ settles (8 divisions in
    60 days on the copy). With batch 25 × 2 it may stay too coarse to pass the embryo gate.
    The batch scale (asked in s072, not answered) decides this.
- **M. What the control means for the fourth organ.** The gate passes with growth simply
  stopped after day 30. Chloe's question to Rocky: is the fourth organ still wanted from
  genesis (where growth IS needed, from one class), and at what batch scale?
- **L. "Gate reads both"**: one reading passing is enough (as built), or both must pass?
- **F1–F5. Stage 2b** (plan §12.7), unanswered.
- Carried s070 decisions E, A–D concern Vee and Weiss; with all three retired Chloe takes
  them as moot unless Rocky says otherwise.

## What was built / committed
- `trioron/organs/` (seven files), `pcll/mixed.py` (rules out, three places, court hooks),
  `experiments/incubator/fetal.py` (`NOISE_CH`, `PROBE`, gate both ways), `numa_probe.py`
  (modes `witness`, `fourth`, `still`; `dims`), `vee_body.py` and `blur.py` shims, tests,
  spec §9.17, plan §13.9, §13.10, §14, the `*_s072.log` files.
- Tests: 172 pass; the same four v2 tests fail as before (not run this time, not looked at):
  `test_learning.py::TestCredit::test_lock_after_sustained_engagement`,
  `::test_cooldown_prevents_immediate_relock`, `test_lifecycle.py::TestGrowth::
  test_growth_trigger_logic`, `::TestDreamCycle::test_dream_locks_eligible_cells`.
- NOT done: the new organism; rule 16; plan section for s070; the manual's phasecyte line;
  `mixed.py` is ~1,560 lines (over the 500-line discipline, not split); the gate script
  still prints "her"; the last part of `test_fourth_organ_tries_divisions_for_a_leaf` (the
  unfed case) asserts little.

## Open / not diagnosed
1. Why, in a pair split on the random column, one child takes every world row.
2. Why the "work" kick is lost by day 90 when pairs are tried inside the working organ, and
   kept with the fourth organ.
3. What a fetal body needs class growth for after day 30, if the gate passes without any;
   whether the gate measures what growth would give.
4. The fourth organ's cost grows with the days (22 → 68 s a day in 60 days) while the working
   organs do not; not followed past day 90. Inner pairs: 109 born, none settled.
5. Why the rule-off copy with the random channel has fewer classes at day 60 than Wile.
6. Carried: the ear's failing batches; which of "none" and the ladders gives Wile his
   memory-rise row; stop-and-resume not exactly equal to living straight through; the
   `world.py` work-kick quirk; the older s070 items (handoff at `10e0003`).

## NEXT (s073)
0. Rocky's answers to K and L first. Nothing is born without his name and yes.
1. On his yes: a new home `outputs/incubator/<name>/`, copy `outputs/incubator/vee/sense.pt`
   into it, genesis and embryo with `embryo.py` (read it first: it must set
   `leaf.mixed.court = FourthOrgan()` on each organ at genesis, which is NOT wired yet),
   stage file by hand on his word, fetal with `PROBE=<n> DAYS=<limit> python3
   experiments/incubator/fetal.py live`.
2. If the batch scale is to change: a copy first, from Wile's day-30 body:
   `VEE_HOME=outputs/incubator/copies/<name> NOISE_CH=140 python3
   experiments/incubator/numa_probe.py setup outputs/incubator/wile 30 fourth`, then
   `DAYS=60 DAY_BUDGET=900 python3 experiments/incubator/fetal.py live`, `fetal.py gate`,
   `numa_probe.py dims 140`.
3. Stage 2b's sign-off (F1–F5); stage 3 (birth) is not written in full.
4. Carried: plan section for s070; the manual's phasecyte line; the join of the organs;
   guards (token budget, sandbox limits).

## Standing rules (carried)
- `docs/design/matron_rules.md` binds the matron. In role with an organism, no word of
  simulation until Rocky says so. Never batch-train an organism. A stage is entered only on
  Rocky's yes, stage file changed by hand. Gates are not relaxed. Every gate carries a plain
  control. One organism on one seed is said to be so.
- Incubator organism: no parameter cap, 85 % of host RAM, swap excluded; benches / CI keep
  50 K. Paper is repo-only. Helper budget ≤ 50 % of the weekly limit (none spent in
  s069–s072).
- One heavy process at a time on this PC (6 cores, 7.6 GB). `python3`, not `python`.
  `pkill -f` / `pgrep -f` hit the shell — kill by PID.
- While a chain of organism runs is going, do not edit `experiments/incubator/`,
  `trioron/pcll/` or `trioron/organs/` unless the change is neutral for them.
- A long background job in this harness can be killed without a message; check the log's
  last day before trusting "finished".

## State of the build / pointers
- Branch `conscience-core`, pushed. `main` is at the v0.3.5 release commit `b4b6789`.
- Local to this PC, uncommitted by rule: the three homes, `copies/`, `outputs/incubator/
  wiki_sent.*`, the s064–s066 matron-written data, `*.err`. Logs `*_s072.log` committed.
- Environment: Qwen GGUF at `~/models/qwen2.5-1.5b-instruct-gguf/`; Simple-Wikipedia ZIM at
  `~/data/wikipedia/`; bubblewrap installed; Python 3.10.12, torch 2.11.0, WSL2.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs / uncommitted logs untracked; `outputs/data/**` caches.

---
History pointers (no dated archives): s071 handoff at 10e0003; s070 at 0247e27 (fuller
tables at 4f7039a); s069 at 05a3ef1; s068 at fc11bd9; s067 at d22ebd2; s066 at b7aff0e; s064
at e67b6ce.
