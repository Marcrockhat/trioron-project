# Trioron Handoff

**Session date:** 2026-11-26 to 2026-11-27
**Session number:** 071
**Session title:** Wile reopened, passes the fetal gate at day 90; the Numa rule for class growth built and tried on a copy; stage 2b drafted

**Summary.** (1) Rocky asked again for the s070 graft numbers, the memory-rise row, the
difference between the three organisms and the fetal gate (all explained from the record;
nothing new measured). (2) He reopened **Wile** (retired in s070) and let him live uncapped
to fetal day 60, then 90: **the gate passes every row at day 90**, at the cost of 4,540
body-sense classes and days of up to 532 s. Rocky: "So we finally find the solution. Wile is
the key for the development." He now says "him" of Wile. (3) Stage 2b (he chooses) drafted
for sign-off, plan §12: NOT signed. (4) Rocky: "The numas concept is not wired isn't?"
Correct; the Numa rule for class growth was written (plan §13), signed ("Yes to all, build it
and run the copy trial"), built opt-in (`trioron/pcll/numa.py`) and tried on a copy of Wile
from his day-30 body: classes bounded (424 against 4,540), four times cheaper, gate passes at
day 60 and misses one row at day 90 (memory full .89, needs .91), and only one division in
406 settled as Numa. (5) His fourth-organ idea (keep what is taken back) is sketched in plan
§13.6, NOT built. Rocky ended the session: "Okay lets wrap things up and meet in a new
session."

## READ THIS FIRST (s072)
- Read `docs/design/matron_rules.md` and `docs/design/vee_development.md` **§12 and §13**
  before touching an organism. The plan still has NO s070 section (the s070 handoff at
  `0247e27` is that record).
- **Two living organisms, one retired, on this PC only, uncommitted. Never delete, overwrite
  or refit any home:**
  - `outputs/incubator/wile/` — **Wile, 120 days old, fetal day 90**, stage `fetal`,
    uncapped, no opt-in rule on. **Reopened by Rocky in s071** ("Yes, reopen Wile and
    increase her fetal day"). Copies `body_embryo_day30`, `body_fetal_day{30,60,90}.pt`.
    **HELD at day 90: Rocky's word reached day 90 only.** Gate passed at day 90 by the
    script's verdict; no stage is promoted, and there is no signed stage to enter.
  - `outputs/incubator/weiss/` — Weiss, 210 days old, fetal day 180, cap + round + fold,
    gate not passed. Untouched in s071. **HELD at day 180.**
  - `outputs/incubator/vee/` — Vee, fetal day 90. RETIRED (s070). Untouched.
- Copies (not organisms, may be deleted): `outputs/incubator/copies/` now also holds
  `wile30_numa` (the Numa trial, at day 90, with `body_fetal_day{60,90}.pt`) and
  `wile30_plaincheck` (rule off, day 32; only a check). The s069–s070 copies are still there.
- Wile uncapped cannot live much longer as he is: his last five days averaged 259 s, the
  slowest 532 s of the 900 s budget, body file 455 MB. Chloe's estimate, not measured: a
  further uncapped stretch hits the budget within days.

## Rocky's rulings this session (his words)
- On Wile's day-30 gate: "If you see Wile, it only failed at tiny margin isn't?" Then: "Yes,
  reopen Wile and increase her fetal day." Asked how far and under which rules: to day 60,
  uncapped as born. After day 60: "Yes, let her live to day 90".
- After day 90: "So we finally find the solution. Wile is the key for the development. How
  is the interaction with him?" (Chloe's answer: none yet; birth is the first exchange.)
- "On the size we found the solution by using folding isn't? We need to design 2b though."
  (Chloe: only partly; the cap bounds the time, the fold lowers the count, neither made her
  smaller, and none of it was tried on Wile.)
- "The numas concept is not wired isn't?" Then: "Yes, write the Numa rule for class growth
  first. One thing that I keep wondering more classes means more creativity though but it
  will raise a confused organism. Can you raise another organism to chew on this classes
  instead? The fourth organs?"
- On G1–G4 of plan §13.7: "Yes to all, build it and run the copy trial."

## 1. Wile, fetal day 30 → 90, uncapped (logs `wile_fetal_d{60,90}_s071.log`, `wile_fetal_gate_day{60,90}_s071.log`; one organism, seed 0)
Started from his day-30 body (identical to its copy). No alarm.

| fetal gate | d30 (s069) | d60 | d90 |
|---|---|---|---|
| feels his kicks (k-means 1.00) | .89 FAIL | .97 | .98 |
| what a kick moves | 5 of 5 ("mark" .67) | 5 of 5 ("hold" .89) | 5 of 5, each 1.00 |
| CPU rise, his / world's (base .57) | .98 | .98 | .95 (k-means 1.00) |
| memory rise (base .59) | .93 | .97 (k-means .97) | .97 (k-means .93) |
| nothing lost: memory full (embryo .96, needs .91) | .85 FAIL | .90 FAIL | .95 |
| nothing lost: busy / place / window | 1.00 / 1.00 / .99 | 1.00 / 1.00 / .99 | 1.00 / 1.00 / 1.00 |
| ear: text present (embryo 1.00) | 1.00 | 1.00 | 1.00 |
| gate | not passed | not passed | **all criteria pass** |
| classes ear / body sense / change | 119 / 347 / 187 | 175 / 1,352 / 402 | 231 / 4,540 / 710 |
| in use, last 10 days | 64 / 64 / 84 | 92 / 99 / 130 | 116 / 122 / 143 |

- Reported rows "holding something" (.90) and "a mark exists" (.71) from the body sense sit
  at their base: he does not read them there.
- Days 11–49 s to day 60, then 32–532 s; body file 70 → 137 → 455 MB; the gate reading at
  day 90 peaked 1.85 GB (peak while living not measured).
- Weiss fails memory rise at .59 after 180 days; Wile reads .93–.97. Wile differs from Weiss
  in "none" AND the ladders from his first moment: not separated. Vee (sharp world, sharp
  organs) also passed every row at days 60 and 90.

## 2. Stage 2b — he chooses (plan §12; DRAFT, NOT signed, NOT built)
Kicks keep arising by themselves as urges. Inherited strain on two readings (RAM used, CPU
busy, upper step as he sees them). For a candidate action his sense of change gives a
forecast (the class he falls in with the changes blank, as gate row 2 reads it). He holds
back an urge that raises a strained reading, and does the one action that lowers it (let go
while memory is full and he holds something). The rule is given; the forecast is his. Gate:
holds back, does not freeze, relieves, no thrashing, nothing lost; control = the same body
with choosing off. **Decisions F1–F5 wait on Rocky** (plan §12.7); he put the Numa rule
first, so 2b waits behind it.

## 3. The Numa rule for class growth (plan §13; signed, built opt-in, one copy trial)
- **Found in the code:** a division is final at once; the buffer it is judged on is mostly
  the class's own replays (40 a class at a settled sitting), so a class can divide on what
  the world never showed; the organisms run without the stress council's settlement.
- **The rule:** the two children of a division are a provisional pair; only members the
  world gave count; a provisional class is not divided and gets no replays; two coherent
  batches of 25 world members each settle it (Numa); a failed batch, or 60 sittings without
  the batches, joins the children again by the fold's routine (Mima); a pair that lost a
  child by another road is "gone". Not retroactive. 25 / 2 / 60 are Chloe's values.
- **Built:** `trioron/pcll/numa.py` (`NumaLedger`; switch on with
  `leaf.mixed.numa = NumaLedger()` on each organ, then save); hooks in `mixed.py`
  (`_fold_stale`'s joining is now `_fold_pairs`, shared; behaviour-neutral, old fold tests
  pass); spec §9 rows for `numa.py` and `sitting.py`; tests
  `test_numa_ledger_settles_or_takes_back`, `test_numa_rule_on_a_leaf`;
  `experiments/incubator/numa_probe.py` (`setup <source home> <day> [plain]`, `ledger`).
- **Neutrality check:** a plain copy of Wile's day-30 body with the new code, rule off,
  relived his days 31 and 32 with identical class counts. Wile himself is the control.
- **Trial** (`copies/wile30_numa`, rule on in all three organs, uncapped, day 30 → 90; logs
  `numa_probe_s071.log`, `numa_probe_s071_gate_day{60,90}.log`; one organism, seed 0):

| | Wile d60 | copy d60 | Wile d90 | copy d90 |
|---|---|---|---|---|
| feels his kicks | .97 | .95 | .98 | .98 |
| what a kick moves | 5 of 5 | 5 of 5 | 5 of 5 | 5 of 5, each 1.00 |
| CPU / memory rise | .98 / .97 | .98 / .97 | .95 / .97 | .98 / .97 |
| memory full (needs .91) | .90 FAIL | .93 | .95 | **.89 FAIL** |
| busy / place / window; ear | 1.00 / 1.00 / .99; 1.00 | 1.00 each; 1.00 | 1.00 each; 1.00 | 1.00 each; 1.00 |
| gate | not passed | **all pass** | **all pass** | not passed |
| classes ear / body sense / change | 175 / 1,352 / 402 | 149 / 428 / 232 | 231 / 4,540 / 710 | 163 / 424 / 222 |
| s a day (last five); body file | 37; 137 MB | 24; 99 MB | 259; 455 MB | 53; 107 MB |

  Ledger at day 90 (born / Numa / Mima / gone / on trial): ear 116 / 0 / 72 / 0 / 44; body
  sense 181 / 1 / 88 / 2 / 90; sense of change 109 / 0 / 57 / 1 / 51. Peak memory 0.98 GB.
- **One Numa in 406 divisions:** in effect his classes were frozen near their day-30 count.
  Body sense: 85 of 88 Mima ran out of time, 60 of them with a child the world never fed in
  30 days (unpaid growth), 8 after passing a first batch (the rule too slow for them). Ear:
  44 of 72 gathered a batch and failed it. Sense of change: all 57 ran out of time.
- Kicks and memory rise rose (.89 → .98, .93 → .97) with no net new classes: those rows come
  from the days. Memory full read .93 then .89 around its line while Wile's own rose to .95
  with growth: Chloe's reading is that this row needs some real growth the values starve;
  one copy does not settle it.
- **Chloe was wrong** in plan §13.3 ("100–200 body-sense classes"): the rule is not
  retroactive, the copy kept its 347.
- No value was changed after the result. The gate is not relaxed.
- **The fourth organ** (plan §13.6; Rocky's idea; sketch only): a boundary taken back goes
  to a fourth organ as a small record, consulted only for moments the working organ refuses,
  and put on trial again if it fits them. Rocky's "yes to all" took Chloe's recommendation:
  in his body, after the trial. NOT built. `NumaLedger.taken_back` already keeps each Mima's
  record (no sketches).

## Decisions waiting on Rocky
- **H (newest, asked, not answered). The trial length of the Numa rule.** Chloe's proposal:
  a pair stays on trial until it has gathered its batches; it is taken back only when a
  batch fails or a child receives nothing from the world for 60 sittings. A second copy
  trial from Wile's day-30 body, about 40 minutes. Or leave the rule, or something else.
- **I. Wile himself.** He passed; he is held at day 90 uncapped and cannot go far as he is.
  Which rules he lives under next (uncapped, cap + round + fold, the Numa rule once a copy
  trial satisfies Rocky) is a rule change on his organs (rule 25): his yes, copy first.
- **F1–F5. Stage 2b** (plan §12.7), including F3: is 2b required before birth?
- **G3 follow-up.** The fourth organ, now that the trial shows how much is taken back (217
  boundaries in 60 days on the copy).
- Carried from s070, unanswered: **E** (the organ-wise body of Vee's and Weiss's organs),
  **A** (kin-only placement in the round), **B** (the resting footprint), **C** (Weiss after
  day 180), **D** (the sharpening trigger). Rocky's turn to Wile may make E, C and D moot;
  that is his to say.

## What was built / committed (all pushed)
- `818e38e`: `trioron/pcll/numa.py`, `mixed.py` hooks, tests, spec §9 rows,
  `experiments/incubator/numa_probe.py`, plan §12 (2b draft) and §13 (the rule, the trial),
  the seven `*_s071.log` files. Not released: `main` and PyPI stay at v0.3.5.
- Tests: 170 pass; the same **four v2 tests fail as before** (not looked at):
  `test_learning.py::TestCredit::test_lock_after_sustained_engagement`,
  `::test_cooldown_prevents_immediate_relock`, `test_lifecycle.py::TestGrowth::
  test_growth_trigger_logic`, `::TestDreamCycle::test_dream_locks_eligible_cells`.
- NOT done: plan section for s070; the manual's phasecyte line ("constant in samples seen");
  `mixed.py` is ~1,540 lines (over the 500-line discipline, not split); the gate script
  still prints "her". Chloe did not read `docs/TRIORON_MANUAL.md` or the spec sections at
  the start of this session (the handoff, the matron's rules and plan §8, §11 were read).

## Open / not diagnosed
1. Whether memory full needs real class growth (the copy's .93 → .89 against Wile's .95).
2. Why the ear's provisional pairs fail their batch (44 of 72) where the other organs' run
   out of time; the ear also spawns a composer cell at every sitting (carried).
3. Which of "none" and the ladders-from-first-moment gives Wile his memory-rise row.
4. Under the rule a class taken back may divide again at the next sitting: the churn was
   not counted.
5. Carried from s070: why Weiss's sense of change never divided on the memory readings;
   which sharpening step cost the "work" kick; why the nest router cannot tell two brains
   apart; whether replays protect anything over a long time; the second cause of
   multiplying; why only .29 of the ear's members sit in their best class; stop-and-resume
   not exactly equal to living straight through; the `world.py` work-kick quirk.

## NEXT (s072)
0. Rocky's answer to H first, then I. Act only on his answers. Wile stays HELD at fetal day
   90, Weiss at day 180.
1. On his yes to H: change the trial length in `numa.py` (and plan §13.2, the test), run the
   second copy trial from Wile's day-30 body exactly as the first:
   `VEE_HOME=outputs/incubator/copies/<name> python3 experiments/incubator/numa_probe.py
   setup outputs/incubator/wile 30`, then `DAYS=60 DAY_BUDGET=900 python3
   experiments/incubator/fetal.py live`, copy `body.pt` to `body_fetal_day60.pt`,
   `fetal.py gate`, `numa_probe.py ledger`; the same to day 90.
2. Stage 2b's sign-off (F1–F5); stage 3 (birth) is not written in full.
3. Write the plan section for s070 from the handoff at `0247e27`; correct the manual's
   phasecyte line.
4. Carried, untouched: the join of the organs; guards (token budget, sandbox limits); the
   s066 fact-memory list (s067 handoff at `d22ebd2`).

## Standing rules (carried)
- `docs/design/matron_rules.md` binds the matron. In role with an organism, no word of
  simulation until Rocky says so. Never batch-train an organism. A stage is entered only on
  Rocky's yes, stage file changed by hand. Gates are not relaxed. Every gate carries a plain
  control. One organism on one seed is said to be so.
- Incubator organism: no parameter cap, 85 % of host RAM, swap excluded; benches / CI keep
  50 K. Paper is repo-only. Helper budget ≤ 50 % of the weekly limit (none spent in
  s069–s071).
- One heavy process at a time on this PC (6 cores, 7.6 GB). `python3`, not `python`.
  `pkill -f` / `pgrep -f` hit the shell — kill by PID.
- While a chain of organism runs is going, do not edit `experiments/incubator/` or
  `trioron/pcll/` unless the change is neutral for them: later stages import the new code.
- A long background job in this harness can be killed without a message; check the log's
  last day before trusting "finished".
- How to switch the s069–s070 opt-in rules on in a living organism (ONLY on his yes): set
  `l.mixed.sitting = SittingCap()` or `SittingCap(fold=True)` on each organ, save; for the
  fold rebuild traces first (`trace_rebuild.py`); relative stillness `body.still_share`.

## State of the build / pointers
- Branch `conscience-core`, pushed. `main` is at the v0.3.5 release commit `b4b6789`.
- Local to this PC, uncommitted by rule: the three homes, `copies/`, `outputs/incubator/
  wiki_sent.*`, the s064–s066 matron-written data, `*.err`. Logs `*_s071.log` committed.
- A new organism's language sense: copy `outputs/incubator/vee/sense.pt` into her home
  before genesis.
- Environment: Qwen GGUF at `~/models/qwen2.5-1.5b-instruct-gguf/`; Simple-Wikipedia ZIM at
  `~/data/wikipedia/`; bubblewrap installed; Python 3.10.12, torch 2.11.0, WSL2.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs / uncommitted logs untracked; `outputs/data/**` caches.

---
History pointers (no dated archives): s070 handoff at 0247e27 (fuller s070 tables at
4f7039a); s069 at 05a3ef1; s068 at fc11bd9; s067 at d22ebd2; s066 at b7aff0e; s064 at
e67b6ce.
