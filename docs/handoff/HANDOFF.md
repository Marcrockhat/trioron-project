# Trioron Handoff

**Session date:** 2026-11-25 to 2026-11-26
**Session number:** 070
**Session title:** the memory wall fixed (v0.3.5); Weiss to fetal day 180, gate not passed; why he never sharpens; Vee's organs pass his gate as a second brain

**Summary.** (1) What killed the uncapped copy was the working memory of the membership step,
not what she holds: fixed bit-identically, released as **v0.3.5** (main, tag, PyPI), tested
alive. (2) Lineage routing measured read-only. (3) Rocky's rulings: Weiss lives under cap +
round + fold, first to fetal day 90, then to day 180; **Wile and Vee retired**. Weiss's gate
is NOT passed at days 90, 120, 150, 180 (kicks, memory rise, memory full). (4) Why the Ws
never sharpen was found (the trigger cannot fire in a large organ); relative stillness was
tried on a copy with a control: it makes him sharpen, gives no gate row, loses the "work"
kick; the kicks row passes from days alone by day 240. (5) On Rocky's "graft Vee to Weiss":
merging her classes into his leaves him with more classes; as a second brain her sense of
change and body sense pass every row of HIS gate in HIS world, her ear does not, and the
nest router cannot tell the two brains apart. Rocky calls Weiss "he" (the gate script still
prints "her"). Rocky ended the session at Chloe's request (very heavy): "Okay lets move to
the next session."

## READ THIS FIRST (s071)
- Read `docs/design/matron_rules.md` and `docs/design/vee_development.md` **§11.9–11.12**
  before touching an organism. (The plan has NO s070 section: this handoff is the record.)
- **One living organism, two retired, on this PC only, uncommitted. Never delete, overwrite
  or refit any home:**
  - `outputs/incubator/weiss/` — **Weiss, 210 days old, fetal day 180**, stage `fetal`, under
    `SittingCap(fold=True)` on all three organs since day 60, `still_share` off. Copies:
    `body_embryo_day30`, `body_fetal_day{30,60,90,120,150,180}.pt` (day 60 is before the rule
    change). **HELD at day 180: Rocky's word reached day 180 only.**
  - `outputs/incubator/vee/` — Vee, 120 days old, fetal day 90. **RETIRED (Rocky, s070).**
  - `outputs/incubator/wile/` — Wile, 60 days old, fetal day 30. **RETIRED (Rocky, s070).**
  - "Retired" = never run again, home kept: Chloe's reading of his "retired both.", told to
    him, not contradicted. Reading their saved bodies is allowed and was done (rule 27).
    **Using Vee's organs in a new body (below) is a question only Rocky answers.**
- Copies for experiments (not organisms, may be deleted): `outputs/incubator/copies/`
  `weiss60_{plain,capped,round,noanneal,fold}` (`weiss60_plain` is at fetal day 75),
  `weiss180_still` (day 300, `still_share = 0.01`), `weiss180_control` (day 300, rule off).
- Memory is no longer the wall for an uncapped organism (v0.3.5; tested to 3,419 classes at
  1.49 GB); time is (an uncapped day with a full round: 200–288 s and rising).

## Rocky's rulings this session (his words)
- "about the class hoarding, probably the problems is more systematic?" … "I meant not in the
  classes per se but the RAM growth".
- "Okay, we need to fix this. Afterwards we need to ush and update the main and the pypi
  version. Afterwards we need to test it. And anyway why it need to compare each class to
  each table and not takingthe important table afterall."
- "Yes, try the lineage routing on a copy."
- On the five decisions carried from s069: "1. 90. 2-5. Put the sister to rest." Asked: cap +
  round + fold for Weiss; "retired both." (Wile and Vee).
- "Okay lets wait until he's fully developed." After the day-90 gate: "Well then we need to
  increase the fetal day to 90 I guess." Asked how far: "90 more, to day 180".
- "Yes, run it. Because they should have their sense sharpened gradually and there must be
  reason why it never triggered." Then, of two options for the trigger: "Option 1, try it on
  a copy of Weiss." Then: "Okay run the control."
- "Can you try to graft Vee to Weiss and check if the class reduced?" Asked which: "Vee's
  classes into Weiss". Then: "Let it finish, then try the native one. Because as I can
  remember, trioron let training being done in different machine and can be combine without
  significant miss of ability." Then: "if you see tis, Vee pass all the check mark even in
  very few incubation days. When it pass, I think we need to just graft it. I imagine that
  these are two different brain in an organism." And: "I thought nesting already handling
  the voting or am I wrong?" (He was right: the nest's router picks the leaf.)

## Decisions waiting on Rocky
- **E (newest, asked, not answered). The organ-wise body.** Build on a copy a body with Vee's
  sense of change and body sense and Weiss's ear, read the full gate on it, then let it live
  some days? Cautions told to him: classes 178 + 1,781 + 550 (more than Weiss now); her
  organs are sharp and uncapped, so cap + round + fold on them is a rule change (rule 25)
  needing his yes; Vee is retired, so her organs living again in another body is his call.
- **A. Kin-only placement in the round**, opt-in, tried on a copy (k = 4 to start).
- **B. The resting footprint**: the sketch as the class, the buffer a short window; to be
  written for sign-off, toy leaf first. Not built.
- **C. Weiss himself after day 180**: more days with the rule off (the control suggests the
  kicks row passes by day 240), or held until the memory row is understood.
- **D. The sharpening trigger**: as signed it never reaches his body sense; relative
  stillness as tried gave no gate row and cost the "work" kick. Drop it, change it, or first
  read which step cost the kick (read-only, on `weiss180_still`). The pace (one body-sense
  step in ~100 days) would be a second ruling.

How to switch the opt-in rules on in a living organism (ONLY on his yes): load her body, set
`l.mixed.sitting = SittingCap()` or `SittingCap(fold=True)` on each organ, save; for the fold
rebuild traces first (`VEE_HOME=<home> [APPLY=1] python3 experiments/incubator/
trace_rebuild.py`); for relative stillness set `body.still_share = 0.01`. Run:
`VEE_HOME=outputs/incubator/<name> DAYS=<total fetal days> DAY_BUDGET=900 python3
experiments/incubator/fetal.py live`, copy `body.pt` to `body_fetal_day<N>.pt`, then
`fetal.py gate`.

## 1. The memory wall (one organism, seed 0)
- `_margins` built a members × classes table and held three copies of it. On Weiss's day-60
  body sense (172,812 × 2,086; he holds 177 MB): **+4.23 GB**. Fix (`trioron/pcll/mixed.py`):
  best class and margin taken inside each row chunk and written in place, `_evidence` fills
  one table, budget `EVIDENCE_ELEMS`. Bit-identical; test
  `test_margins_in_chunks_equal_the_whole_table`. Now +0.32 GB; a whole `_consolidate()` on
  the killed copy (258,259 × 2,738) peaks 1.15 GB, 168 s. Allocator note: appending small
  results between megabyte temporaries fragments the heap; writing in place bounds it.
- Tested alive (`cap_probe_plain_fix_s070.log`): `weiss60_plain`, uncapped, fetal day 68 →
  75, 2,738 → 3,419 body-sense classes, peak 1.49 GB.
- **The resting footprint is the second, unfixed cause** (decision B): a class is up to 400
  members × 648 B = 259 KB against ~1 KB for its sketch, and replays fill every class toward
  400. Contradicts the manual's "constant in samples seen".
- Released **v0.3.5**: commit `b4b6789`, `main` fast-forwarded, tag, PyPI. The wheel also
  carries s068–s069 (`Arena.enlarge`, `pcll/sitting.py`, `mixed.no_anneal`).

## 2. Lineage routing against the exhaustive membership (read-only, Weiss day 60; `lineage_route_probe.py`, logs `lineage_route_probe*_s070.log`)
His lineage is a comb (23 divisions deep on average). Descent from the root fails (.17 same
class as exhaustive). From the class a member is already in, compared with kin under the
ancestor k divisions up, body sense: k = 0 .863; 2 .912 (18 comparisons); 4 .937 (49, 2.3 %);
8 .964 (166, 7.9 %); misses keep .995 of the evidence. Sense of change .977 → .995 at k = 3.
The ear is different: only .289 of its members sit in their best class (not diagnosed). Not
tested alive (decision A). Toy check: `try_divide` splits a single Gaussian mode wider than
~0.55 rad; a blob wide on 4 dimensions tiles into 31 classes (a candidate for classes that
claim no lived moment; Chloe's reading, untested on an organism).

## 3. Weiss, fetal day 60 → 180 under cap + round + fold (logs `weiss_trace_rebuild_s070.log`, `weiss_fetal_d90_s070.log`, `weiss_fetal_d180_s070.log`, `weiss_fetal_gate_day{90,120,150,180}_s070.log`)
Traces rebuilt from his own log before day 61. ~33 s a day, peak 1.23 GB, no alarm.

| fetal gate | d60 | d90 | d120 | d150 | d180 |
|---|---|---|---|---|---|
| feels his kicks (k-means 1.00) | .94 FAIL | .92 FAIL | .92 FAIL | .94 FAIL | .93 FAIL |
| what a kick moves | 5 of 5 | 5 of 5 ("work" .78) | 5 of 5 | 5 of 5 ("work" .67) | 5 of 5 |
| CPU rise, his / world's | .95 | .98 | .98 | .98 | .98 |
| memory rise (base .59, k-means .93–.97) | .59 FAIL | .59 FAIL | .59 FAIL | .59 FAIL | .59 FAIL |
| nothing lost: memory full | .93 FAIL | .94 FAIL | .94 FAIL | .95 FAIL | .95 FAIL |
| nothing lost: busy / place / window | 1.00 / .98 / .98 | 1.00 / .98 / .98 | .99 / 1.00 / 1.00 | .99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| ear: text present | .91 | .95 | .95 | .97 | .97 |
| classes ear / body sense / change | 137 / 2086 / 114 | 120 / 1946 / 52 | 147 / 1850 / 58 | 160 / 1813 / 86 | 178 / 1769 / 93 |

**GATE NOT passed at any reading, the same three rows.** Body-sense classes fall slowly.
Chloe said after day 180 that all three rows were flat; **for kicks that was wrong** (the
control passes it by day 240, §5). Day 90 did not equal the s069 copy trial (1,945 classes,
memory full .96): not diagnosed.

## 4. Why the Ws do not sharpen; the memory row (read-only; `blur_row_probe.py`, logs `blur_row_probe_{weiss,wile}_s070.log`)
- **The trigger cannot fire in a large organ.** It needs the class count unchanged for
  3 + classes // 8 sittings in a row (`vee_body._mature`). Weiss at day 180: ear 178 classes,
  needs 25, longest still run ~6, 0 steps; body sense 1,769, needs 224, ~2, 0 steps; sense of
  change 93, needs 14, ~14, 2 steps (reading 7: 2 → 3 → 5). Wile the same. (Runs read from
  daily counts; per-sitting counts are not logged.) The earlier record "nothing has ever
  sharpened in Weiss" is superseded by those 2 steps.
- **Blur is NOT why memory rise fails (Chloe's guess was wrong).** Through his 2-step view
  only 2 of 736 lived events look alike; a reader of that view scores .93 on the 29 unseen.
  His classes do not use it: 489 of 736 lived events sit in one class holding both kinds.
  Wile's classes hold one kind each (.93). With the true echo he scores 1.00, hidden or
  shown as "no kick" .59: his sense of change tells his doing from the world's by the echo
  alone. Weiss was born without "none", so a hidden echo looks to him like all six flags on
  (never lived); Wile sees silence; "none" is fixed at birth. Why his classes never divided
  on the memory readings: not diagnosed.

## 5. Relative stillness on a copy, with a control (logs `still_probe_s070*.log`, `still_control_s070*.log`)
`vee_body.Body.still_share` (opt-in, default None = as before): a sitting is still when the
class count moved by no more than that share of the organ's classes since the sitting before
(Chloe's choice: against the sitting before, not the start of the run). Both copies from
Weiss's day-180 body, 120 days, cap + round + fold, ~41–44 s a day, peak 1.45 GB.

| day 300 | rule on (`weiss180_still`) | rule off (`weiss180_control`) |
|---|---|---|
| steps taken ear / body sense / change | 0 / 1 / 6 | 0 / 0 / 2 (none new) |
| feels his kicks | .97 pass (d240 .97) | .99 pass (d240 .98) |
| what a kick moves | 4 of 5, "work" .00 (d240 .78) | 5 of 5, all 1.00 |
| CPU rise / memory rise | .98 / .59 FAIL | .98 / .59 FAIL |
| nothing lost: memory full | .94 FAIL | .93 FAIL (d240 .90) |
| ear: text present | .96 | .98 |
| classes ear / body sense / change | 246 / 1276 / 120 | 240 / 1351 / 125 |

- The kicks row passes from the days alone. The lost "work" kick came with the sharpening
  (which of the 4 steps is not read). Memory rise unmoved in both; memory full no difference.
- Body-sense pace with the rule: one step in ~100 days. One copy each, one seed.

## 6. Vee grafted to Weiss (in memory only; no home changed)
trioron's arena graft (`lifecycle/graft.py`, spec §5.3) was NOT run: it transplants arena
cells, nothing in `trioron/pcll/` uses it, and a phasecyte organ's knowledge is its classes'
members. Rocky was told so.

**a. Her classes merged into his** (`graft_probe.py`, log `graft_probe_s070.log`; Chloe's
routine, only the settling `_consolidate` is trioron's). Her members are read back to the
numbers she sensed (checked on her lived log: mean error .0003–.02; discrete columns go
through `codec_levels`), seen through his blur and receptors, added as classes beside his,
4 consolidation rounds.

| organ | his | hers | joined | after 4 rounds (his / hers left) | Weiss alone, same rounds |
|---|---|---|---|---|---|
| ear | 178 | 344 | 522 | 487 (160 / 327) | 166 |
| body sense | 1,769 | 1,781 | 3,550 | 2,067 (1,646 / 421) | 1,673 |
| sense of change | 93 | 550 | 643 | 498 (93 / 405) | 93 |

Below the sum, but more classes than he had, in every organ. No gate read; no fold.

**b. Two brains under the nest router** (`nest_probe.py`, log `nest_probe_s070.log`). Her
organs beside his, nothing merged; `ManifoldRouter` (full covariance, fitted from each leaf's
lived log, unblurred numbers) picks one leaf a moment. Weiss's gate, his world; her brain
only classifies. Peak 4.8 GB (two bodies and both logs in memory: watch the 85 % line).

| row | his brain | her brain | the nest (routed) |
|---|---|---|---|
| feels the kicks (k-means 1.00) | .93 FAIL | 1.00 | 1.00 |
| what a kick moves | 5 of 5 | 5 of 5 ("work" .78) | 5 of 5 ("work" .78) |
| CPU rise (base .57) | .98 | .98 | .95 |
| memory rise (base .59, pass above .64) | .59 FAIL | **.86** | .62 FAIL |
| nothing lost: busy / full / elsewhere / window | 1.00 / .95 / 1.00 / 1.00 | 1.00 / **.99** / 1.00 / 1.00 | .99 / .95 / .99 / 1.00 |
| ear: text present (pass from .88) | .97 | .75 FAIL | .89 |

- Her sense of change and body sense pass every row of his gate in his world without having
  lived it; her ear does not.
- The router is the weak part: it gives her brain .24 / .39 / .74 of his world's moments
  (change / body / ear) and only .62 / .80 / .76 of her own lived ones. Both brains sense the
  same world, so recognition has little to go on.
- Read from the columns (each row depends on one organ): her sense of change + her body
  sense + his ear would pass every row. NOT built or run (decision E).

## What was built / released (all committed and pushed)
- `trioron/pcll/mixed.py` fix + test; spec §9 partition row; **v0.3.5** on PyPI.
- `experiments/incubator/`: `lineage_route_probe.py`, `blur_row_probe.py`, `graft_probe.py`,
  `nest_probe.py` (all read-only / in memory); `vee_body.py` gains opt-in `still_share`.
- Tests: 168 pass; the same **four v2 tests fail as before** (not looked at):
  `test_learning.py::TestCredit::test_lock_after_sustained_engagement`,
  `::test_cooldown_prevents_immediate_relock`, `test_lifecycle.py::TestGrowth::
  test_growth_trigger_logic`, `::TestDreamCycle::test_dream_locks_eligible_cells`.
- NOT done: plan §11.13 for s070; the manual's phasecyte line ("constant in samples seen");
  `mixed.py` is ~1,500 lines (over the 500-line discipline, not split).

## Carried from s069 (unchanged; detail in the s069 handoff and plan §11.9–11.12)
- Fetal day 30: Vee kicks .99, memory rise .62, memory full 1.00, ear .65, classes 67 / 429 /
  141; Wile kicks .89 FAIL, **memory rise .93**, memory full .85 FAIL, ear 1.00, classes 119 /
  347 / 187. Vee passed every row at fetal days 60 and 90 (s067).
- The cranium check: no class row is `recyclable`, nothing calls `compact()`;
  `arena.param_bytes` cannot see what she holds; her organs almost never have a settled
  sitting; the merge is off in her organs.
- She holds far more than she was fed (annealing: 40 replays a class at a settled sitting);
  93 % of Weiss's day-60 body-sense classes claim no moment he lived; the fold makes the
  class count fall but she is not faster and not smaller.

## Open / not diagnosed
1. Why Weiss's sense of change never divided on the memory readings (Wile's did; Vee's
   organ reads it at .86 in his world).
2. Which sharpening step cost the "work" kick.
3. Why the nest router cannot tell the two brains apart; whether organ-wise choice is the
   right form of "two brains" (decision E).
4. Whether the replays protect anything over a long time.
5. The second cause of multiplying (+241 classes on 768 moments with no new replays).
6. Why only .29 of the ear's members sit in their best class.
7. Where a day's time goes now that memory is bounded.
8. Carried: stop-and-resume not exactly equal to living straight through; the `world.py`
   work-kick quirk; the ear spawning a composer cell at every sitting.

## NEXT (s071)
0. Rocky's answer to E first (it was the open question when the session ended), then A–D.
   Act only on his answers. Weiss stays HELD at fetal day 180.
1. On his yes to E: build the organ-wise body on a copy (own home under `copies/`), read the
   full gate, report; living on only with his word on the rules for her organs.
2. Cheap read-only work that needs no ruling: open points 1 and 2.
3. Write plan §11.13 from this handoff; correct the manual's phasecyte line.
4. Carried, untouched: stage 2b not designed; the join of the organs; guards (token budget,
   sandbox limits); the s066 fact-memory list (s067 handoff at `d22ebd2`).

## Standing rules (carried)
- `docs/design/matron_rules.md` binds the matron. In role with an organism, no word of
  simulation until Rocky says so. Never batch-train an organism. A stage is entered only on
  Rocky's yes, stage file changed by hand. Gates are not relaxed. Every gate carries a plain
  control. One organism on one seed is said to be so.
- Incubator organism: no parameter cap, 85 % of host RAM, swap excluded; benches / CI keep
  50 K. Paper is repo-only. Helper budget ≤ 50 % of the weekly limit (none spent in s069–s070).
- One heavy process at a time on this PC (6 cores, 7.6 GB). `python3`, not `python`.
  `pkill -f` / `pgrep -f` hit the shell — kill by PID.
- While a chain of organism runs is going, do not edit `experiments/incubator/` or
  `trioron/pcll/` unless the change is neutral for them: later stages import the new code.
- A long background job in this harness can be killed without a message; check the log's
  last day before trusting "finished".

## State of the build / pointers
- Branch `conscience-core`, pushed. `main` is at the v0.3.5 release commit `b4b6789`
  (behind `conscience-core` by the later s070 commits: probes, logs, `still_share`).
- Local to this PC, uncommitted by rule: the three homes, `copies/`, `outputs/incubator/
  wiki_sent.*`, the s064–s066 matron-written data, `*.err`. Logs `*_s070.log` committed.
- A new organism's language sense: copy `outputs/incubator/vee/sense.pt` into her home
  before genesis.
- Environment: Qwen GGUF at `~/models/qwen2.5-1.5b-instruct-gguf/`; Simple-Wikipedia ZIM at
  `~/data/wikipedia/`; bubblewrap installed; Python 3.10.12, torch 2.11.0, WSL2.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs / uncommitted logs untracked; `outputs/data/**` caches.

---
History pointers (no dated archives): the fuller s070 tables (lineage routing, the s069
copy trials) are in this file's previous version at 4f7039a; s069 handoff at 05a3ef1; s068
at fc11bd9; s067 at d22ebd2; s066 at b7aff0e; s064 at e67b6ce.
