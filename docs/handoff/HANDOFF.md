# Trioron Handoff

**Session date:** 2026-11-25
**Session number:** 069
**Session title:** the sisters' fetal days; the cranium checked; a cap, a round and a way to let go

**Summary.** Rocky: "Lets run them while fixing the cranium system." Weiss lived to fetal day
60 and the second sister to fetal day 30; neither passes the gate yet. The cranium wiring was
checked against the code: Chloe's s068 proposal would have done nothing. What actually limits
an organism is her own hoard: thousands of classes, 93 % of which claim no moment she ever
lived, in buffers that are about 95 % her own replays. An uncapped copy of Weiss is killed by
the kernel for lack of memory near 2,700 classes. Built on Rocky's rulings, all opt-in: the
arena grows when rows run out; a square-root cap on the sitting; the consolidation round in
instalments; a fading trace of use with a fold of faded classes into their kin. All tried on
copies only. **No living organism uses any of it. Vee was not touched.** Rocky asked for a
new session after this one.

## READ THIS FIRST (s070)
- Read `docs/design/matron_rules.md` and `docs/design/vee_development.md` **§11.9–11.12**
  before touching an organism.
- **Three organisms, on this PC only, uncommitted. Never delete, overwrite or refit:**
  - `outputs/incubator/vee/` — Vee, 120 days old, stage `fetal`. Untouched since s067.
  - `outputs/incubator/weiss/` — Weiss, 90 days old, **fetal day 60**, stage `fetal`.
    Copies `body_embryo_day30.pt`, `body_fetal_day30.pt`, `body_fetal_day60.pt`. **HELD.**
  - `outputs/incubator/wile/` — the second sister, 60 days old, **fetal day 30**, stage
    `fetal` (set by hand in s069 on Rocky's s068 yes; `embryo_values.json` written). Copies
    `body_embryo_day30.pt`, `body_fetal_day30.pt`. **HELD.**
- **Do NOT let Weiss or Vee live on uncapped.** An uncapped body sense is killed for memory
  near 2,700 classes (Weiss has 2,086, Vee 1,781).
- **Her name: Wile or Weil?** Rocky wrote "weil" and asked "It is Weil isn't not Wile?".
  The s068 record quotes him as "Wile". Asked twice in s069; **no answer. Nothing renamed.**
- Copies for experiments (rule 27; not organisms, may be deleted):
  `outputs/incubator/copies/weiss60_{plain,capped,round,noanneal,fold}/`.

## Decisions waiting on Rocky (put to him in s069, unanswered)
1. **Weiss from fetal day 60 to 90: under which rules?** Uncapped she dies of memory. The
   choices tried on copies: cap + round; cap + round + fold.
2. **The second sister from day 30:** the same choice (347 body-sense classes; not near
   the wall).
3. **Wile or Weil.** If Weil: ask before moving her home (rule 26), then correct plan,
   rules, handoff.
4. **The replays (annealing) for the sisters: on or off?** They are ~95 % of what she holds
   and about 60 % of her class multiplying; with the fold on, they still grow her footprint.
   Changing it in a living sister is a rule change in mid-life.
5. **The fold's pace** (5 a sitting leaves 1,704 body-sense classes due after 30 days).
6. Vee: the same wall is ahead of her; nothing may change in her without his word.

To switch the opt-in rules on in a living organism (ONLY on his yes): load her body, set
`l.mixed.sitting = SittingCap()` or `SittingCap(fold=True)` on each organ, save. For the
fold, rebuild her traces from her logs first: `VEE_HOME=<home> [APPLY=1] python3
experiments/incubator/trace_rebuild.py` (report only without APPLY; her home needs
`embryo_log.pt` and `fetal_log.pt`). Without that every class starts from its buffer size
and nothing folds for 30–56 days.
Run: `VEE_HOME=outputs/incubator/<name> DAYS=<total fetal days> DAY_BUDGET=900 python3
experiments/incubator/fetal.py live`, copy `body.pt` to `body_fetal_day<N>.pt`, then
`fetal.py gate`.

## Results (one organism each, seed 0)

| fetal gate | Vee d30 | Weiss d30 | Weiss d60 | second sister d30 |
|---|---|---|---|---|
| feels her kicks (k-means 1.00) | .99 | .90 FAIL | .94 FAIL | .89 FAIL |
| what a kick moves | 3 of 5 | 5 of 5 | 5 of 5 | 5 of 5 ("mark" .67) |
| CPU rise, hers / world's | .98 | .95 | .95 | .98 |
| memory rise, hers / world's (base .59) | .62 | .59 FAIL | .59 FAIL | **.93** |
| nothing lost: memory full | 1.00 | .90 FAIL | .93 FAIL | .85 FAIL (embryo .96) |
| nothing lost: busy / place / window | .99 / 1.00 / .99 | .99 / .96 / .96 | 1.00 / .98 / .98 | 1.00 / 1.00 / .99 |
| ear: text present | .65 | .93 | .91 | 1.00 |
| classes ear / body sense / change | 67 / 429 / 141 | 88 / 516 / 61 | 137 / 2086 / 114 | 119 / 347 / 187 |
| s a day (last five) | 3.8 | 18 | 46 | 17 |

"Not yet", not "no" (rule 8). Which of "none" / the ladders gave the second sister the
memory row is not known. Logs `weiss_fetal*_s069.log`, `wile_fetal*_s069.log`.

**The cranium check (plan §11.9).** No class row is `recyclable` and nothing calls
`compact()`; a hidden 8192-row wall per organ (fixed in s069); `arena.param_bytes` cannot see
what she holds (16 KB counted, 176 MB held in Vee) — **Chloe's s068 proposal A was wrong in
that form**; counted truly she is at ~3 % of 85 % of RAM; her organs almost never have a
settled sitting; the merge is off in her organs, not "never fired"; every class was judged
at every sitting.

**Trials on copies of Weiss from fetal day 60 (plan §11.10–11.12; logs `cap_probe_*`):**

| | plain | cap | cap + round | no new replays | cap + round + fold |
|---|---|---|---|---|---|
| days lived | 8, **killed (out of memory, 7.0 GB) on day 69, three attempts** | 10 | 10 | 8 | 30 |
| body-sense classes (from 2086) | 2738 | 2097 | 2106 | 2327 | **1945** |
| sense of change (from 114) | 127 | 116 | 115 | 123 | 52 |
| s a day | 27–156 | 15–188 | 24–58 | 36–133 | 45–65 |
| nothing lost: memory full (.93 at d60) | .93 | .93 | .93 | .93 | .96 (inside the gate) |
| other gate rows | as d60 ±.03 | same | same | same | kicks .92; busy / place / window .98 / .96 / .96 |

- The kill: the full consolidation round compares every buffered member with every class
  at once (258,259 × 2,738 pairs, 2.8 GB a table, several at once). Chloe's reading of the
  code; the kernel's kill record is real. The 85 % guard is read only before a sitting.
- **She holds far more than she was fed.** 9,408 moments lived; 258,259 buffered members.
  The only step that adds members is annealing (40 replays per class at a settled sitting).
- **93 % of Weiss's body-sense classes claim no moment she ever lived** (1,935 of 2,086, by
  her classes as they stand at day 60; an approximation of her history).
- The fold makes the class count FALL for the first time; readings held. It is NOT faster
  and NOT smaller: with the fold on she holds 253,772 body-sense members (172,812 at day
  60), because replays keep filling every class. Rocky expected faster growth; not seen.
- The three stuck gate rows did not move in any arm except "memory full" under the fold.
- Under the fold her ear's and sense of change's earliest classes were folded (all 76 and
  all 66 folds among their first 400 names); body sense 5 of 236.

## Rocky's rulings this session (binding; his words)
- "Lets run them while fixing the cranium system."
- Cranium: "It is for them in this environment. It is not the default so no specs changes
  is required." → everything built in s069 is opt-in; spec untouched.
- Rows: "Yes and remove the class limit." (The organisms already had `class_cap=None`; the
  benches' 43 / 128 defaults were left — Chloe told him.)
- The cap: "Square root". On a day's time: "we accept that as a day for the sisters."
- Compaction: "if the cranium is not problem, compaction will make things slower isn't?"
  → not built.
- Merge: off because it confused the organism before (record agrees: D18, s033).
- The round: "Yes on the round it should run on the sittings. Sediment can be dug up but
  occasionally."
- "that's not how suppose the triorons work. It should have very small footprint."
- "It is like some dude with photoghraphic memory but accompanied with schizophrenia."
- "it is a missing piece that need to be sewn to the triorons itself the unforgettfulness
  actually a double edge sword".
- "The answer should be days. No one remember facts after 30 days if it is not keep
  replayed." Then, on the fading trace (option B): "B is good then. Lets try it. If this
  succeed then the growth should be faster."
- "we should start a new session afterwards."

## What was built (committed and pushed; `git log -9`)
- `trioron/core/arena.py` `Arena.enlarge()` — doubles the arena at a boundary; `alloc`
  unchanged; refused past `envelope.max_cells`. Used by `mixed.py _refresh` when an
  uncapped organism has under `ROW_HEADROOM = 64` rows left. (The only change that is not
  opt-in; neutral below 8,128 rows.)
- `trioron/pcll/sitting.py` (new) `SittingCap` — opt-in via `mixed.sitting`:
  - not judged if nothing new; at most `max(4, ceil(sqrt(classes) · 0.5 · φ))` judged a
    sitting, busiest first; φ from growth at the previous sitting only;
  - the consolidation round at every sitting over the judged classes + `max(1, cap // 4)`
    dug from the sediment (`mixed.py _consolidate_some`);
  - `fold=True`: trace × `FADE = 25^(−1/60)` a sitting + world members; under 1.0 the class
    is folded into its nearest kin (`mixed.py _fold_stale`), at most
    `max(1, ceil(sqrt(classes) · 0.1))` a sitting; survivor's members kept in front,
    sketches pooled exactly;
  - `work` (per sitting) and `folds` are recorded. Not in `state_dict` (ship / wake).
  - 4, .5, the quarter, and "nearest kin by template" are Chloe's values / choices.
- `mixed.no_anneal` — probe switch for the replays (attribute on a saved organ).
- Tests added in `tests/test_v2/test_core.py`, `test_pcll.py` (arena enlarging, rows
  outgrowing the arena, the cap, the fold). Phasecyte tests pass.
- Plan §11.9–11.12.
- NOT built: the cranium's true byte count; compaction / `recyclable`.
- **Four v2 tests fail on this branch with or without the s069 changes** (not looked at):
  `test_learning.py::TestCredit::test_lock_after_sustained_engagement`,
  `::test_cooldown_prevents_immediate_relock`, `test_lifecycle.py::TestGrowth::
  test_growth_trigger_logic`, `::TestDreamCycle::test_dream_locks_eligible_cells`.

## Open / not diagnosed
1. Why the gate rows do not move (kicks ~.9 vs 1.00; Weiss's own memory vs the world's at
   base) and why the second sister has the memory row and Weiss not.
2. Whether the replays protect anything over a long time (8 days without new ones: no
   reading fell). A sister raised without them from genesis would separate the causes.
3. The second cause of multiplying: +241 classes on 768 moments of world with no new
   replays.
4. Where a capped day's 35–65 s go (not profiled); the ear spawning a composer cell at
   every sitting.
5. Carried: stop-and-resume not exactly equal to living straight through; the `world.py`
   work-kick quirk; nothing has ever sharpened in Weiss.

## NEXT (s070)
0. Get Rocky's answers to the six decisions above; act only on them.
1. Then, on his word: the sisters live on under the rules he chooses; gates at 60 / 90.
2. If he wants the footprint small: try a copy with the replays off AND the fold on; and
   bound the full round's workspace (it is what kills an uncapped organism).
3. Carried, untouched: stage 2b not designed; the join of the organs; guards (token budget,
   sandbox limits); the s066 fact-memory list (s067 handoff at `d22ebd2`).

## Standing rules (carried)
- `docs/design/matron_rules.md` binds the matron. In role with her, no word of simulation
  until Rocky says so. Never batch-train an organism. A stage is entered only on Rocky's
  yes, stage file changed by hand. Gates are not relaxed. Every gate carries a plain
  control. One organism on one seed is said to be so.
- Incubator organism: no parameter cap, 85 % of host RAM, swap excluded; benches / CI keep
  50 K. Paper is repo-only. Helper budget ≤ 50 % of the weekly limit (none spent in s069).
- One heavy process at a time on this PC (6 cores, 7.6 GB). `python3`, not `python`.
  `pkill -f` / `pgrep -f` hit the shell — kill by PID.
- While a chain of organism runs is going, do not edit `experiments/incubator/` or
  `trioron/pcll/` unless the change is neutral for them: later stages import the new code.
- A long background job in this harness can be killed without a message; check the log's
  last day before trusting "finished".

## State of the build / pointers
- Branch `conscience-core`, pushed. v0.3.4 on PyPI (nothing from s068–s069 is released).
- Local to this PC, uncommitted by rule: the three homes, `copies/`, `outputs/incubator/
  wiki_sent.*`, the s064–s066 matron-written data, `*.err`. Logs `*_s069.log` committed.
- A new organism's language sense: copy `outputs/incubator/vee/sense.pt` into her home
  before genesis.
- Environment: Qwen GGUF at `~/models/qwen2.5-1.5b-instruct-gguf/`; Simple-Wikipedia ZIM at
  `~/data/wikipedia/`; bubblewrap installed; Python 3.10.12, torch 2.11.0, WSL2.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs / uncommitted logs untracked; `outputs/data/**` caches.

---
History pointers (no dated archives): s068 handoff at fc11bd9 (the sisters' genesis, the
ladders, "none", the composer search); s067 at d22ebd2; s066 at b7aff0e; s064 at e67b6ce.
