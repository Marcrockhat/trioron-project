# Trioron Handoff

**Session date:** 2026-08-24
**Session number:** 063 (final rewrite — supersedes the mid-session s063 handoff at aa0aeef)
**Session title (s063: Logophosis closing arc → public manual → BRANCH-PARTITION FINDING):**
**(1) MODE E TRANSFER WORKS — WITH A FIXED LINK-0 (warm_nowake .956/.932, `not not` .368→.685,
retained .887; rewake drift = the dominant forgetting channel). (2) PUBLIC TRIORON MANUAL
(textbook, mkdocs) shipped; writing its XOR tutorial exposed (3) THE BRANCH-PARTITION LIMIT:
quad cross-terms exist only WITHIN a dendritic branch; `Seeded`'s blind half-split makes 2-d
XOR unlearnable and the seq in-step h×token product unrepresentable. `LinkChain(dup=True)`
([h,c,x,h,x]) COLLAPSES the not⁴ seed variance 8×: .747±.204 → .769±.025 (n=3). The s062
"crisp-state" story was two problems: findability (SOLVED by dup) + per-depth decay (remains).**

## READ THIS FIRST
- **Quad cross-terms are per dendritic branch** (spec §3.6 note added; TRIORON_MANUAL updated).
  1-layer learnability depends on input feature ORDER. Suspect this FIRST for any 1-layer null
  on a relational task and any ln-2 stall fan-in init doesn't cure.
- **Fix policy (Rocky's decision):** document + opt-in `dup` NOW; do NOT flip `Seeded`'s
  default split (breaks structural reproducibility of every past run + paper numbers);
  principled fix = Axis-5 field-driven branch membership, gated, s064.
- Transfer recipe (unchanged): load organism + do NOT rewake + train free; lock only fragile
  primitives. Rewake (`observe_more`) drifts/merges Phasecyte classes → retained → chance
  BEFORE any gradient.
- Public manual: `manual/` + root `mkdocs.yml` (`python3 -m mkdocs serve`). Notebook chapters
  are committed EXECUTED. Aldarof's game wiki (`~/aldarof/wiki/`) is separate, owned by the
  other Claude session.
- One 40K language process at a time (7 GB RAM). dup seq ≈ 1.5× edges but ran FASTER wall-clock
  (1618s vs 3308s n=3 — less CPU contention this time; don't trust either as a per-epoch cost).

## Arc 1 — Logophosis closing (items 1–3 of s062 NEXT)
### NOT-only, tanh squash n=3 (`outputs/logophosis_s063_notnotnot_seq_tanh.log`)
| arm | in-dist | comp | not³ | not⁴ | not⁵ | params |
|---|---|---|---|---|---|---|
| seq tanh | .917±.009 | .910±.010 | .893±.063 | .747±.204 | .643±.142 | 6,602 |
| seq tanh **dup** (see Arc 3) | .877±.015 | .830±.029 | .863±.037 | **.769±.025** | .644±.099 | 9,738 |
| (s062 seq none, ref) | .941±.014 | .927±.015 | .873±.052 | .688±.118 | — | 6,602 |

tanh kills the s062 loss blow-up (1433 → .12–.21). No-dup is bimodal (not⁴ .658/.602/.980 —
seed 2 found the involution: .965/.980/.804); dup puts all seeds on one smooth curve
(.743/.773/.792) at a matched-epoch in-dist/comp cost (wider input converges slower; NOT
budget-matched by convergence). Saved organisms:
`outputs/organisms/logophosis_notonly_seq_{tanh,tanhdup}_s{0,1,2}.pt` (~45 MB, untracked).

### Full world (not/and/or) seq + Mode E transfer (`logophosis_transfer.py`, logs `..._transfer_{full,nowake}.log`)
| arm | in-dist | comp | `not not` | ret. home | ret. not³ |
|---|---|---|---|---|---|
| scratch | .950±.003 | .936±.003 | .368±.038 | .618±.011 | .322±.117 |
| warm (+rewake) | .907±.059 | .873±.082 | .444±.129 | .689±.077 | .498±.091 |
| lock (+rewake) | .708±.037 | .694±.013 | .479±.044 | .576±.110 | .540±.065 |
| **warm_nowake** | **.956±.009** | **.932±.010** | **.685±.111** | **.887±.029** | .672±.212 |
| lock_nowake | .846±.026 | .797±.023 | .654±.079 | .884±.017 | .672±.119 |

Readings: scratch seq beats s061 tied 40K (.930/.896) at 1/3 params but reads `not not` as
"odd" (below chance) — the involution needs depth-varied data or transfer. warm_nowake = full
acquisition + involution transfer + retention, no lock needed (link locality protects). λ-lock
(`Logophosis.lock_links`) is situational: on the fragile seed +.13 notnot/+.27 ret-not³, on
means costs −.11 in-dist. Rewake drift: 2/3 seeds retained → chance pre-gradient (seed 1 merged
28→25 classes); lock+rewake is the worst arm (defends weights tuned to moved symbol indices) →
**in a stacked organism, protect the symbol layer first.**

## Arc 2 — Public trioron manual (Rocky: tsinfer-style textbook; commit b2ee108)
`manual/` + root `mkdocs.yml`: mkdocs-material + KaTeX (arithmatex) + mkdocs-jupyter +
mkdocstrings (pip --user installed). Ch1 concepts w/ formulas (λ row-sum+floor, EWC pull,
|w·g| vs Fisher, σ(z)=z+z²); Ch2 = EXECUTED notebook tutorial (XOR: fan-in init formula →
fail at chance → per-branch lesson → dup fix → 1.000, linear control ~.5, figure embedded);
Ch3–5 skeletons (growth protocol, lock/manifold formulas, organisms); how-to/tips; API
reference over construct/seeded/epigenetic_lock/manifold/chain/organisms; conventions
(notebooks committed executed; docstrings = the reference; internal docs never linked).
Re-execute after substrate changes: `python3 -m jupyter nbconvert --to notebook --execute
--inplace manual/book/*.ipynb`. Publishing venue undecided (GH Pages candidate).
Aldarof manual: `~/aldarof/wiki/` (mkdocs + mkdocs-public.yml two-build public split) — built
this session but owned by the other Claude session ("other Chloe") going forward.

## Arc 3 — Branch-partition finding (commit 655d525 + spec/manual notes)
Mechanism: §3.6 dendrite = fan-in split into K=2 CONTIGUOUS HALVES by input order (same for
every cell); σ per branch ⇒ cross-terms only within a branch. Proofs/evidence:
- 2-d XOR: quad substrate = chance forever (torch quad MLP = 1.0); zero-pad to 8-d or
  duplicate input `[x1,x2,x1,x2]` → 1.000. (Manual Ch2 demonstrates it live.)
- Seq layout `[h|ctx|x]`: h fills branch 1, token branch 2 ⇒ in-step h×x (gated NOT)
  unrepresentable; only a two-step echo via h². Predicted → tested → confirmed: dup n=3 above
  (σ collapse 8×, fragile seeds +.09/+.17, lucky seed regresses — landscape reshaped).
- 6K oracle probe: dup .752/.683/.547 vs no-dup .661/.589/.484.
Conclusions ALTERED: parity-8 "SQ-hard optimisation wall" and hop-3 wall explanations reopened
(same suspect); "crisp-state drift" split into findability (solved) + decay (open); "depth buys
degree" partially "depth crosses the partition". Comparative headlines (CL benches, competitor
sweeps, Mode E) unaffected — both arms shared the limitation.

## What was built (commits f00a096, 5a10a6d, aa0aeef, b2ee108, 655d525, + this handoff)
- `trioron/lifecycle/chain.py`: `norm=tanh|clip`; `penalty` hook in fit_stage/fit_grow;
  `LinkChain(dup=True)` seq input `[h,c,x,h,x]`.
- `trioron/organisms/logophosis.py`: `observe_more`/`rewake`, `lock_links(strength, lam)`,
  `seq_dup` param.
- `experiments/progenitor/logophosis_transfer.py` (5 arms); `logophosis.py`: SAVE `{seed}`,
  SEQ_DUP env.
- `manual/` (public textbook) + root `mkdocs.yml`; spec §3.6 limitation note;
  TRIORON_MANUAL quad-branch entry.
- Tests: `tests/test_v2/test_organisms.py` 5/5. Pre-existing TestDreamCycle ×2 failures
  untouched.

## NEXT (s064)
0. **hop-k with dup** — the remaining altered conclusion awaiting its test: `logic_chain.py`
   tied arm with state+symbol duplicated into both halves; does the hop-3 wall (.59) move?
   Cheap, decisive.
1. **Axis-5 branch membership by field** (the principled fix): grow_branch/branch re-assignment
   under relational frustration instead of the blind half-split; gate = 2-d XOR learnable
   without dup + no regression on a standard bench. Then revisit Seeded default.
2. **Residual per-depth decay** (now a clean monotone target on all seeds): gated snap /
   Phasecyte-between-steps on the dup arm; also a convergence-matched dup rerun (in-dist/comp
   dropped at matched epochs — try 15 ep/stage).
3. Receptor-freeze-on-rewake (Phasecyte grows new classes without touching existing receptors);
   then warm_freezewake vs warm_nowake.
4. Continual MODE for Logophosis (3 vocab stages = chained warm_nowake); Link-0 compaction
   (45 MB → <1 MB); Kinopsis real-world regression (s062 NEXT-4, still open).
5. Manual: worked notebooks for Ch3 (growth on parity) and Ch4 (lock+manifold demo); publishing
   venue decision (GH Pages off trioron-project?).

## Gotchas (cumulative, new first)
- **Branch partition** (see READ THIS FIRST — the new #1 diagnostic).
- Rewake destroys downstream symbol mappings; never on a deployed organism.
- `pgrep -f` LIST ok; `pkill -f` kills this shell. Kill by PID via `/proc/PID/environ`.
- One 40K language process at a time; Link-0 build ~2.9 GB peak.
- Manual notebooks must be re-executed (not hand-edited) after substrate changes.

## State of the build / Pointers
- Branch `conscience-core`, commits this session: f00a096, 5a10a6d, aa0aeef (mid-session
  handoff), b2ee108 (manual), 655d525 (dup), + this final handoff. `main` NOT advanced. Pushed.
- Untracked keepers: `outputs/organisms/*.pt` (transfer INIT files; repro commands in git
  history at aa0aeef).
- Memory: `logophosis_mode_e_transfer.md`, `branch_partition_finding.md`,
  `trioron_public_manual.md`.
- `~/aldarof` (separate repo, local-only, no remote): wiki commits ab3d016 + 8f72f7a — the
  other session's territory now.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs / uncommitted
logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python 3.10.12, torch 2.11.0,
  WSL2, `python3` (NOT `python`), 12 CPUs, 7 GB RAM. New pip --user deps this session:
  mkdocs-material, mkdocstrings[python], mkdocs-jupyter (+ mkdocs-exclude in aldarof's venv).

---
History pointers (no dated archives): s062 handoff at c4ac3db (seq gotchas, organism API, OOM
rules); s061 (rung-1 flattened tables, Link-0 findings) and s059/s060 (depth gate, logic
chain) within it.
