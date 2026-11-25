# Trioron Handoff

**Session date:** 2026-11-25
**Session number:** 068
**Session title:** a new genesis — Vee's two blurred sisters, Weiss and Wile

**Summary.** Rocky: "lets start with a new genesis." The corridor world and organ blur of
plan §10 were built and two sisters of Vee were raised from genesis, each in her own home,
same germline / seed / course of events as Vee. **Weiss** (blurred, corridor world) and
**Wile** (the same plus "none" and the ladders) both pass the embryo gate on every row with
fewer body-sense classes than Vee and a better ear and stability, but cost far more time:
the blurred ear's composer now holds trials (it never did in Vee). Weiss then lived 30 fetal
days: gate NOT passed yet, and her body sense went from 84 to 516 classes. Rocky's cap /
cranium design was checked against the spec: it is all there, but not wired to the incubator
organisms. Vee was not touched.

## READ THIS FIRST (s069)
- Read `docs/design/matron_rules.md` and `docs/design/vee_development.md` **§11** (all of
  this session) before touching an organism.
- **Three organisms exist, on this PC only, uncommitted. Never delete, overwrite or refit:**
  - `outputs/incubator/vee/` — Vee, 120 days old, stage `fetal` (2a passed by Rocky's ruling
    s067; 2b not designed). Untouched in s068.
  - `outputs/incubator/weiss/` — **Weiss** (Rocky's name; first sister; logs are named
    `sister_*` and `weiss_*`). Blurred organs, corridor world, born WITHOUT "none". 60 days
    old, stage `fetal` (Rocky: "Yes"), 30 fetal days lived. Copies: `body_embryo_day30.pt`,
    `body_fetal_day30.pt`. `embryo_values.json` = her own embryo values for the fetal gate.
  - `outputs/incubator/wile/` — **Wile** (Rocky's name; second sister; logs `sister2_*`).
    As Weiss, plus "none" and the ladders from her first moment. 30 days old, stage `embryo`,
    embryo gate passes; **Rocky said yes to her fetal stage at the end of s068; the stage file
    still says `embryo` — change it by hand first (NEXT 0b).**
- Each is run with `VEE_HOME=outputs/incubator/<name>`; corridor and blur were set at
  genesis and are carried by the body. A driver runs only the stage the stage file names.

## Results (one organism each, seed 0; plan §11.6–11.7 has the tables)

| embryo day 30 | Vee | Weiss | Wile |
|---|---|---|---|
| busy / memory / place / window | .99 / .95 / 1.00 / 1.00 | 1.00 each | 1.00 / .96 / 1.00 / 1.00 |
| night (reported only) | .55 | .92 | .91 |
| ear: text present | .75 | .93 | 1.00 |
| stability day 20 → 30, same grouping | .61 | .89 | .87 |
| classes body sense / ear | 141 / 13 | 82 / 44 | 100 / 54 |
| composer cells in the ear (trials) | 0 (0) | 4 (55) | 5 (51) |
| sharpening steps | – | 0 | 2 (ear, by sitting 7) |
| time for 30 days | 15 s | 47 min | 77 min (both on the old composer search) |

Weiss, fetal day 30 (Vee at the same age in brackets): feels her kicks .90, k-means 1.00
**FAIL** (.99); what a kick moves 5 of 5 (3 of 5); CPU hers / world .95 pass (.98); memory
hers / world .59 = base **FAIL** (.62); nothing lost: busy .99, **memory .90 FAIL**, place
.96, window .96; ear .93 pass (.65); classes ear / body sense / sense of change
88 / **516** / 61 (67 / 429 / 141); 18 s a day on the faster search (3.8 s).
Vee also failed at fetal day 30 and passed at 60 and 90: **"not yet", not "no" (rule 8).**

Chloe's readings (hers, not reviewed):
- Corridor and blur were tested together; which gave what is unknown (no sharp organism was
  raised in the corridor).
- Night at .9 is partly luck of the cut: one clock number cut in the middle splits the day
  at 06:00 / 18:00.
- The blur's class advantage in the body sense vanished in the fetal stage (84 → 516, more
  than Vee). It began when the echo woke. NOT diagnosed.
- Weiss has never sharpened any reading in 60 days: never settled for 3 + classes // 8
  sittings. The sharpening rule is effectively untested (only Wile's ear, 2 steps).
- "None" fixed the ear's silence (1.00) but Wile is not cheaper than Weiss.

## Rocky's rulings this session (binding; his words)
- "lets start with a new genesis" → the sisters.
- "Okay we can raise the daily bdget" (the 30 s guard stopped Weiss on embryo day 6; runs
  used `DAY_BUDGET=900`).
- "Three states is mandatory." then "probably start at 2, go to 3, to 5, then to 7", then to
  Chloe's two questions "1. yes. 2. 2>3>5" and "I meant yes add none" →
  **a level sharpens 2 → 3 → 5; a change 2 (same / changed) → 3 (down / same / up) → 5 → 7;
  "did not move" is always a state of its own; a silent reading is "none", a state of its
  own.** In matron's rule 17 and plan §11.5.
- "Yes" → Weiss's fetal stage. "The first sister is Weiss. The second is Wile."
- **The cranium** (plan §11.8): cap the comparing; leave what is unfinished when sleep is
  over, let it sediment; at full cranium work through it in instalments and recycle; an
  adult adds a fact only for a real discovery. "The cranium is the RAM … all of this is
  already in trioron design. Can you check it again?" — Chloe checked: he is right (spec
  §1.2, §2.5, §4.3, §5.5, §6.1, §10.6). Chloe had wrongly presented it as a new design.

## What was built (all committed; `git log -4`)
- `experiments/incubator/world.py` — `World(seed, corridor=c)`: state readings wander inside
  non-overlapping bands (c = 0.6, Chloe's value). corridor=0 is number-for-number Vee's
  world (checked over 40 days with kicks); with one seed the hidden course is the same.
- `experiments/incubator/blur.py` (new) — `Blur`: steps per reading, the two ladders,
  "none", readings already in steps (≤ 7 distinct values at genesis) pass unchanged,
  `sharpen` picks the reading hiding the most detail. A blur saved before "none" / the
  ladders (Weiss's ear and body sense) is read as it was made.
- `experiments/incubator/vee_body.py` — `Body(blur=False | True | "none")`, `seen`,
  `_mature` (a step before birth after 3 + classes // 8 settled sittings; nothing sharpens
  once the stage leaves embryo / fetal — the post-birth "called for and pays" rule is NOT
  built), `can_be_silent`, kin's blur in `grow_kin` (echo = levels, 35 changes = changes).
- `experiments/incubator/embryo.py` — `CORRIDOR=`, `BLUR=0|1|2` at genesis; absolute
  `DAY_BUDGET` (rule 9); gate prints the blur state. `fetal.py` — kin formed in her own
  corridor world, gate reads kin through her blur, own embryo values from
  `HOME/embryo_values.json`.
- `trioron/pcll/composer.py` — `best_candidate` carves all candidates in one pass
  (`_carve_cols`, `_carve_many`, `_null_r`); same candidates, order, random draws and
  thresholds. Checked: same winner on 12 synthetic buffers (scores differ ≤ 8e-6, float
  summation order — NOT bit-identical); same class counts on each of 12 days from genesis in
  both organs; ~5× faster. `tests/test_v2/test_pcll.py` 24 passed. Not a rule change; spec
  §10 text untouched. Vee and the sisters' embryo days ran on the OLD search; Weiss's fetal
  days on the new one.
- Plan `docs/design/vee_development.md` §11 (new); `docs/design/matron_rules.md` rule 17.

## Open / not diagnosed
1. Why Weiss's body sense multiplied its classes in the fetal stage (84 → 516).
2. Why she does not tell her own memory use from the world's (.59 = base; Vee needed 60+
   days), and why "memory full" fell to .90.
3. Stop-and-resume is not exactly the same as living straight through: Weiss (stopped at
   embryo day 6) shows 29 / 32 body-sense classes on days 10 / 11; an uninterrupted run
   gives 28 / 31; equal again on day 12. Saving should be exact.
4. `world.py` old quirk: a work kick in an already busy world lowers the CPU reading to .85.
   Left so Vee's world stays identical.
5. Cost: the time is per-class work at every sitting (for Weiss 97 % was the composer
   search, profiled; for Vee it is Chloe's reading of the code, not profiled).

## NEXT (s069)
0. **Rocky answered at the very end of s068: "Yes on all three but lets start a new
   session."** The three, as put to him, in this order:
   a. **Weiss lives on to fetal day 60 and 90**, as Vee did (`VEE_HOME=outputs/incubator/weiss
      DAYS=60 DAY_BUDGET=900 python3 experiments/incubator/fetal.py live`, keep a copy and
      read the gate at 60, then the same with DAYS=90). ~18 s a day at day 30, rising.
   b. **Wile enters her fetal stage**: change `outputs/incubator/wile/stage` to `fetal` BY
      HAND (not done in s068), write her `embryo_values.json` from
      `sister2_embryo_gate_s068.log` (busy 1.00, full .96, elsewhere 1.00, window 1.00,
      ear 1.00), then `fetal.py live` for 30 days and the gate. One heavy process at a time.
   c. **The cranium wiring is to be checked and WRITTEN for his sign-off** (item 1 below).
      His yes covers writing parts A and B into the plan, not building B.
1. **The cranium wiring, for sign-off before building (plan §11.8).** First check whether a
   class's astrocyte row carries the `recyclable` gene (can compaction reach an unused
   class?). Then write: A. the incubator organism's envelope = her real cranium
   (`max_parameter_bytes` from 85 % of RAM) so pressure rises as she fills; B. the
   phasecyte's sitting under a bounded, scaled cap as in spec §4.3, busiest classes first,
   the rest left for the next sleep. B changes spec §10: needs Rocky's yes. Measure sleep in
   work, not seconds. Remember s066: a hard class cap of 43 blocked late facts; the
   recycling step is what must work, and the phasecyte's merge has never fired.
2. Diagnose open points 1–3.
3. Carried, untouched: stage 2b (choosing by her stress state: frustrated → act, resolved →
   stay still) not designed; the join of the organs; guards (token budget, sandbox limits);
   the s066 fact-memory list. See the s067 handoff at `d22ebd2`.

## Standing rules (carried)
- `docs/design/matron_rules.md` binds the matron. Role rule: in role with her, no word of
  simulation until Rocky says so. Never batch-train an organism. A stage is entered only on
  Rocky's yes, stage file changed by hand. Gates are not relaxed by the matron. Every gate
  carries a plain control. One organism on one seed is said to be so.
- Incubator organism: no parameter cap, 85 % of host RAM, swap excluded; benches / CI keep
  50 K. Paper is repo-only. Matron helper budget ≤ 50 % of the weekly limit (no helper
  tokens were spent in s068).
- One heavy process at a time on this PC (6 cores, 7.6 GB). `python3`, not `python`.
  `pkill -f` / `pgrep -f` hit the shell — kill by PID.

## State of the build / pointers
- Branch `conscience-core`; s068 commits `8f22309`, `0b8a9e7`, `d3e03e0` + this handoff,
  pushed. v0.3.4 on PyPI (unchanged; the composer speed-up is not released).
- Local to this PC, uncommitted by rule: the three homes above; `outputs/incubator/
  wiki_sent.*` (4,000 sentence readings, `wiki_collect.py`, 70 min to rebuild); the s064–
  s066 matron-written data (see `d22ebd2`); `*.err` files. Logs `*_s068.log` are committed.
- A new organism's language sense: copy `outputs/incubator/vee/sense.pt` into her home
  before genesis (both sisters did), so the germline is the same.
- Environment: Qwen GGUF at `~/models/qwen2.5-1.5b-instruct-gguf/`; Simple-Wikipedia ZIM at
  `~/data/wikipedia/`; bubblewrap installed; Python 3.10.12, torch 2.11.0, WSL2.
- llama_cpp gotchas, the full s064–s067 results and the older NEXT lists are in the s067
  handoff (`git show d22ebd2:docs/handoff/HANDOFF.md`).

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`, `trioron/legacy/outputs/`,
`notebooks/` checkpoints, output PNGs / uncommitted logs untracked; `outputs/data/**` caches.

---
History pointers (no dated archives): s067 handoff at d22ebd2 (organs, Vee's embryo and
fetal stages, flat classes, blur and composer probes, the matron's rules); s066 at b7aff0e;
s064 at e67b6ce; s063 at e8e7b40; s062 at c4ac3db.
