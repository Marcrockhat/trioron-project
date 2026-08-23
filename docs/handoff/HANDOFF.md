# Trioron Handoff

**Session date:** 2026-08-24
**Session number:** 063
**Session title (s063, Rocky: "run next 1-3, finally the closing arc for our Logophosis"):**
**MODE E TRANSFER WORKS — WITH A FIXED LINK-0. `warm_nowake` (NOT-only seq organism loaded into
the full not/and/or world, Link-0 untouched, all links free) matches scratch on the new world
(in-dist .956±.009, comp .932±.010) AND flips `not not` from systematically inverted (.368) to
learned (.685±.111), retaining the home NOT-world (.887±.029). The killer was Link-0 rewake
drift: `observe_more` merges/moves existing Phasecyte classes → retained falls to ~chance on 2/3
seeds BEFORE any gradient. Also: tanh state squash kills the seq loss blow-up (not⁵ .643±.142
three depths beyond training; bimodal — seed 2 found the involution: not³ .965 / not⁴ .980 /
not⁵ .804).**

## READ THIS FIRST
- **Transfer recipe:** load organism + do NOT rewake + train free. Unseen words (`and`/`or`)
  are readable as their consistent soft projection on the existing matched filters — Link-0
  never needs to see them. Rewake (continued Phasecyte observation) is DESTRUCTIVE to the
  downstream chain (class merge/index drift); a receptor-freeze-on-rewake (grow new classes
  only) is the open substrate fix.
- The λ-lock now exists at the organism surface: `Logophosis.lock_links(strength)` (anchor +
  `set_lambda` on the chain links, `ewc_penalty` added to the loss via a new `penalty` hook in
  `fit_stage`/`fit_grow`). It protects the fragile seed but costs acquisition (below).
- `LinkChain(norm="tanh")` is the seq default choice now (runner env `SEQ_NORM=tanh`); `clip`
  also exists, both probed vs none (tanh best). Do not use rms (s062).
- One 40K language process at a time (7 GB RAM); seq stage ≈ 25 min at 4 threads.

## Item 1 — NOT-only world, tanh squash, n=3 (`outputs/logophosis_s063_notnotnot_seq_tanh.log`)
Same protocol as s062 (NOT_ONLY, DEPTH_MIX 0–2, 40K, 10 ep/stage, MAX_LINKS 2), SEQ_NORM=tanh,
TEST_DEPTHS=1,2,3:

| arm | in-dist | comp | not³ | not⁴ | not⁵ | params |
|---|---|---|---|---|---|---|
| seq tanh n=3 | .917±.009 | .910±.010 | .893±.063 | .747±.204 | .643±.142 | 6,602 |
| (s062 seq none, ref) | .941±.014 | .927±.015 | .873±.052 | .688±.118 | — | 6,602 |

Readings: (i) the s062 quad loss blow-up (train loss 1433 at stage 2) is GONE — tanh bounds the
state; (ii) depth generalisation up on mean but **bimodal**: seed 2 = .965/.980/.804 (found the
involution nearly exactly — not⁴ above its own in-dist), seeds 0/1 = .60–.66 at not⁴; (iii)
per-application decay persists for non-involution seeds — tanh bounds but does not crisp; the
gated snap / Phasecyte-between-steps remain untested. Per-seed organisms saved:
`outputs/organisms/logophosis_notonly_seq_tanh_s{0,1,2}.pt` (~44 MB each, untracked).

## Items 2+3 — full world (not/and/or) seq + Mode E transfer
Runner `experiments/progenitor/logophosis_transfer.py` (new): arms scratch / warm / lock /
warm_nowake / lock_nowake; INIT=the item-1 saves; LOCK=1.0; DTRAIN=1, N=40K, 10 ep/stage,
MAX_LINKS 2, n=3. RETAINED = the NOT-only home world (0–2 NOTs) + not³ zero-shot after
full-world training. Logs `outputs/logophosis_s063_transfer_{full,nowake}.log`.

| arm | in-dist | comp | `not not` | ret. home | ret. not³ |
|---|---|---|---|---|---|
| scratch | .950±.003 | .936±.003 | .368±.038 | .618±.011 | .322±.117 |
| warm (+rewake) | .907±.059 | .873±.082 | .444±.129 | .689±.077 | .498±.091 |
| lock (+rewake) | .708±.037 | .694±.013 | .479±.044 | .576±.110 | .540±.065 |
| **warm_nowake** | **.956±.009** | **.932±.010** | **.685±.111** | **.887±.029** | .672±.212 |
| lock_nowake | .846±.026 | .797±.023 | .654±.079 | .884±.017 | .672±.119 |

Readings:
1. **Item 2 answered:** scratch seq on the full world beats the s061 tied-40K headline
   (.950/.936 vs .930/.896) at 1/3 the params — but `not not` is BELOW chance (.368): trained
   with ≤1 NOT, the seq link reads every NOT as odd. Depth data (or transfer) is required; seq
   alone does not discover the involution from shallow data.
2. **Item 3 (Mode E) answered:** warm_nowake = full acquisition + involution transfer +
   retention, all without the lock. The involution partially survives free full-world training
   (notnot .685, seed 2 .813) — locality of the seq link does most of the protecting.
3. **The λ-lock is real but situational:** on the fragile seed 1 it lifts notnot .610→.743 and
   ret-not³ .440→.709; on means it only costs (in-dist −.11). Use when the primitive matters
   more than the new world.
4. **Link-0 drift is the dominant forgetting channel** — retained → chance at rewake, before
   any gradient (seed 1 also MERGED 28→25 classes). Locking the link while Link-0 drifts is
   worst of all (lock+rewake .708): the link defends weights tuned to moved symbol indices.
   This is the s063 headline gotcha: **in a stacked organism, protect the SYMBOL LAYER first.**

## What was built (commits f00a096, 5a10a6d, + this handoff)
- `trioron/lifecycle/chain.py`: `norm="tanh"|"clip"` squash options; `penalty` hook in
  `fit_stage`/`fit_grow` (optional callable added to the loss).
- `trioron/organisms/logophosis.py`: `ChordLink0.observe_more`, `Logophosis.rewake`,
  `Logophosis.lock_links(strength, lam)` + `_penalty` (epigenetic lock on chain links, head
  free); window stored on ChordLink0.
- `experiments/progenitor/logophosis_transfer.py` (new runner, 5 arms as above).
- `experiments/progenitor/logophosis.py`: SAVE takes `{seed}` per-seed paths.
- Tests: `tests/test_v2/test_organisms.py` 5/5 pass. Pre-existing `test_lifecycle.py::
  TestDreamCycle` ×2 failures untouched (fail on parent too).

## NEXT (s064)
1. **Receptor-freeze-on-rewake**: Phasecyte grows NEW classes without touching existing
   receptors/filters (controller change — habituation/merge currently mutates them). Then
   re-run warm_nowake vs warm_freezewake; expect freezewake ≥ nowake with truly crisp and/or
   symbols.
2. Crisp-state for the non-involution seeds: gated snap (top-2 margin) or Phasecyte between
   seq steps; target = not⁴ ≥ .9 on 3/3 seeds (seed 2 proves the solution exists in this
   parametrisation).
3. Continual MODE for Logophosis (3 vocab stages + forgetting matrix, port `run_continual`) —
   now directly the warm_nowake recipe chained.
4. Link-0 compaction for `Logophosis.save` (44 MB → target < 1 MB; dream_distill a settled
   segment).
5. Kinopsis end-to-end regression through the package class (unchanged from s062 NEXT-4).

## Gotchas (cumulative, new first)
- **Rewake destroys downstream symbol mappings** (this session, above). Never call
  `observe_more`/`rewake` on a deployed organism until item 1 ships.
- `pgrep -f` LIST ok; `pkill -f` kills this shell. Kill by PID via `/proc/PID/environ`.
- One 40K language process at a time; Link-0 build ~2.9 GB peak.
- `git stash` while background runs live: don't.

## State of the build / Pointers
- Commits on `conscience-core`: f00a096 (tanh + lock_links + transfer runner + item-1 log),
  5a10a6d (round-1 transfer log + nowake arms), + final handoff commit. `main` NOT advanced.
  Pushed at session end.
- Untracked keepers: `outputs/organisms/*.pt` (per-seed NOT-only organisms, 44 MB each — the
  transfer INIT files; regenerate via item-1 command below if lost).
- Repro item 1: `NOT_ONLY=1 DEPTH_MIX=1 DTRAIN=2 MAX_OBJ=2 L_MAX=10 N=40000 EPOCHS=10
  MAX_LINKS=2 PATIENCE=6 SEEDS=3 ARMS=seq SEQ_NORM=tanh TEST_DEPTHS=1,2,3
  SAVE='outputs/organisms/logophosis_notonly_seq_tanh_s{seed}.pt' python3
  experiments/progenitor/logophosis.py`
- Repro items 2+3: `MAX_OBJ=2 L_MAX=10 N=40000 EPOCHS=10 MAX_LINKS=2 PATIENCE=6 SEEDS=3
  SEQ_NORM=tanh LOCK=1.0 ARMS=scratch,warm,lock,warm_nowake,lock_nowake python3
  experiments/progenitor/logophosis_transfer.py`
- Memory: `logophosis_mode_e_transfer.md`.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs / uncommitted
logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python 3.10.12, torch 2.11.0,
  WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM (Rocky considering a new laptop — RAM is
  the binding constraint).

---
Below = pointers into git history (no dated archives kept): s062 handoff (seq gotchas, organism
API, OOM rules) at c4ac3db; s061 (rung-1 flattened tables, Link-0 findings) and s059/s060
(depth gate, logic chain) inside it.
