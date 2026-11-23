# Trioron Handoff

**Session date:** 2026-11-24
**Session number:** 065
**Session title (s065: INCUBATOR STAGE 2 — fact memory, first cut):**
**Vee got a key leaf and a memory (`learn_keys` / `remember` / `recall`). The matron wrote two
rewordings of each of the 2,808 stage-1 questions. A learned 32-number key recalls a taught fact
from a rewording better than a nearest-neighbour cache holding 3,072 numbers per fact (.907 vs
.833 at 1 % false override). But a learned LINEAR key does as well as the quad leaf — the second
stage in a row where trioron itself has not earned its place.**

## READ THIS FIRST
- **Design of record: `docs/design/incubator_matron.md`** (stage-2 result is in §3). Binding
  decisions from s064 stand: incubator organism has no parameter cap (85 % of WSL RAM, swap
  excluded; benches/CI keep 50 K); paper is repo-only; sandbox in WSL with offline Simple English
  Wikipedia; creativity = SCAMPER routes; matron budget = up to 50 % of the weekly limit (Max 5×).
- Rocky's framings to keep: humans hallucinate too (told > doesn't-fit > inner doubt); the
  self-check monologue is stage 4; self-hosting milestones ("Linux on Minix").
- **Always carry a linear control in incubator gates.** "Beats the cache" is not "trioron pays".
- Suspect the branch partition first for 1-layer relational nulls (s063 rule).
- Rocky was NOT asked anything this session; nothing here is newly signed off by him. The stage-2
  protocol and the "PASS" reading below are Chloe's and await his review.

## Stage 2 — fact memory, first cut (n=5 article-level splits)
Pipeline: 2,808 questions × (original + light rewording p1 + heavy rewording p2), written by four
Sonnet helpers (+1 small fix job; ~590 K subagent tokens in total) → the reflex READS each text
without answering (`Reflex.read`, 0.53–0.69 s/text, 8,424 texts) → two states per text: last
prompt token and mean over the question's tokens (3,072 numbers together) → gate
`experiments/incubator/stage2_memory.py`.

Protocol: 30 % of articles held out. One question from each of half of them is taught (original
wording stored). Positives = its two rewordings. Negatives = the other question of the same
article (hard) + every question of untaught held-out articles (easy), all three wordings.
"Recalled" = right fact is nearest AND above the threshold that lets 1 % of negatives through.
"Crowded" = memory also holds the ~1,970 training questions.

| key (numbers per fact) | recalled, 197 facts | recalled, 2,164 facts |
|---|---|---|
| cache, raw states (3,072) | .782±.026 | .600±.029 |
| cache, centred states (3,072) — best cache | .833±.030 | .620±.024 |
| cache, random projection (32) — best small cache | .766±.049 | .557±.060 |
| cache, PCA-32 (32) | .552±.015 | .380±.017 |
| learned LINEAR key (32), 8,224 params | .892±.023 | **.718±.019** |
| **organism quad key leaf (32), 41,400 params** | **.907±.019** | .681±.035 |

Honest reading:
- Gate as written (vs a nearest-neighbour cache) passes: +.074 / +.062 with 96× fewer numbers
  per fact; +.14 / +.12 over a cache of the same size.
- Quad leaf = linear key (ahead by .015 uncrowded, behind by .037 crowded). Learning the key
  pays; the trioron leaf does not, yet.
- Ranking is nearly solved for every full-size arm (right fact nearest .986–.997). The
  difficulty is the THRESHOLD; half to three quarters of false overrides are the same-article
  sibling question.
- Load: 197 → 2,164 facts costs every arm ~.2 at 1 % false override (.98 → .90 at 5 %).
- State ablation (3 splits): last token alone ≈ both (organism .865/.649, best cache .852/.630 —
  there the learned key buys only compression); mean alone weaker (.718/.615 vs cache .464/.352).
- Key-size sweep (Rocky asked whether the quad leaf stores less per fact; 3 splits for 8 and 16):
  linear / quad at 8 = .626 / .598, at 16 = .834 / .817, at 32 = .892 / .907 (crowded .407/.365,
  .616/.643, .718/.681). No packing advantage. A learned 16-number key equals the full cache.
- Thresholds were set on the test negatives, identically for all arms (an arm comparison, not a
  deployable number).
- s064's early worry is retired: a question's nearest neighbour is rarely its sibling, but that
  does not hurt retrieval of the same question reworded.

## What was built (this session's commit)
- `trioron/organisms/vee.py`: stage 2 — `learn_keys(H, group, block, k, e)`, `remember(H, texts)`,
  `key(H)`, `recall(H)` → (index, cosine), `memory_bytes()`. Keys are stored EMBEDDED, so the key
  leaf must not drift after facts are stored (untested).
- `trioron/organisms/base.py`: `group_contrast`, `train_contrastive`.
- `experiments/incubator/reflex.py`: `Reflex.read(question)` (keeps the prompt head decoded;
  `answer()` invalidates it). `stage2_collect.py` (resumable), `stage2_memory.py` (gate).
- Spec §9.16 rows for vee.py and base.py; design §3 stage-2 result; test
  `tests/test_v2/test_organisms.py::test_vee_fact_memory_roundtrip` (organism tests 7/7).
- Logs committed: `outputs/incubator/stage2_memory_s065.log`, `..._state_last.log`, `..._state_mean.log`,
  `..._e8.log`, `..._e16.log`.

## Environment facts (s064, still true)
- Qwen GGUF already on disk: `~/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf`
  (0.5B sibling at `~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`). ~1.8 GB RSS, ~21 tok/s read
  on this Ryzen 5 PRO 4650U (6 cores; use 6 threads, not 12).
- Wikipedia: `~/data/wikipedia/wikipedia_en-simple_all_nopic_2026-09.zim` (1.0 GB, 400 K
  articles, full-text index); `pip install --user libzim` done.
- bubblewrap 0.6.1 installed; verified: scratch dir writable, home invisible, network and
  system writes blocked. Time/memory limits per run NOT set up yet.
- Host disk has ~65 GB free (the 907 GB WSL reports is the virtual disk ceiling).
- `sudo` cannot run through the `!` prefix; Rocky runs it in a separate terminal.
- Rocky's plan: Max 5× ($100). **Matron budget DECIDED: up to 50 % of the weekly limit.**
  Not yet calibrated in tokens: ask Rocky for the weekly % from `/usage`. Measured s064 use:
  Fable 131 K out / 365 K new in / 18.0 M cached; Sonnet question job (2,985 questions)
  166 K out / 635 K new in / 2.7 M cached.
  Calibration (Rocky's `/usage`, 2026-11-24): the Fable weekly bar read 8 % after s064, so
  one s064-sized day ≈ 8 % and the 50 % cap ≈ 6 such days/week ≈ 0.8 M Fable output tokens
  (rough: assumes s064 was the week's only Fable use; how Sonnet helpers count is unknown —
  re-read `/usage` before and after the first loop run to measure it).

## Gotchas (cumulative, new first)
- **llama_cpp:** never build `Llama(embedding=True)` — it inflates the context to 256
  sequences (3.7 GB) and breaks decoding. Use `llama_set_embeddings(ctx, True)` on a plain
  context; logits stay bit-identical and `llama_get_embeddings_ith(ctx, -1)` gives the last
  token's hidden state. `logits_all=True` is slow and allocates n_ctx × 152 K floats.
- Hidden states differ slightly with batch chunking (cos .9994): always extract the same way.
- The lead sampler draws with replacement → duplicate articles; split and de-dup by title/id.
- Different question writers differ in difficulty (25–32 % Qwen accuracy): always report
  within-writer and a no-brain control.
- Branch partition (s063). Rewake destroys downstream symbol mappings. `pkill -f` kills this
  shell — kill by PID. One heavy process at a time on 7.6 GB RAM.

## Gotchas (cumulative, new first)
- **`Reflex.read` keys match `answer()`'s h_pre only to cos ~.999** (different batch chunking).
  Compare read-keys only with read-keys; the console must call `read` for memory and `answer`
  for doubt.
- Sonnet writers copy the original for ~12–19 % of "light" rewordings on the first pass; three of
  four fixed themselves, one needed a fix job. Always audit for verbatim copies and answer leaks.
  Parallel helpers share the scratchpad: tell them to use unique scratch file names.
- llama_cpp: never `Llama(embedding=True)`; use `llama_set_embeddings(ctx, True)` on a plain
  context. `logits_all=True` is slow. Extract hidden states the same way every time.
- The lead sampler draws with replacement → de-dup by title/id. Question writers differ in
  difficulty: report within-writer and a no-brain control.
- Branch partition (s063). Rewake destroys downstream symbol mappings. `pkill -f` / `pgrep -f`
  hit this shell — kill by PID. One heavy process at a time on 7.6 GB RAM.

## Open questions for Rocky
1. Review the stage-2 reading: is "beats the cache, ties linear" a pass for you, or does stage 2
   stay open until trioron-specific machinery (growth, λ firmness, per-fact σ) shows a gain?
2. ANSWERED: Fable weekly bar 12 % after s065 (8 % after s064) → this session ≈ 4 points. Still
   unknown whether the Sonnet helpers (~590 K tokens) count on that bar or a separate one.
3. Terms check before any matron-written data is committed (carried from s064).

## NEXT (s066)
0. Stage 2, where trioron could earn its place (pick with Rocky):
   a. per-fact threshold from each fact's own spread (manifold σ; needs ≥2 wordings per fact);
   b. retention when the key leaf keeps learning after facts are stored (lock with λ / credit vs
      free drift) — the real "after N later lessons" test;
   c. growth-pays curve: recall vs key-leaf size and vs facts stored; bytes per parameter;
   d. deployable threshold set on training articles; end-to-end override accuracy with the reflex.
1. Wire memory into the console (`/wrong <answer>` → `remember`; recall before the reflex
   answers) and into a standing loop: checkpoint/resume, token logging, budget stop.
2. Serial graft as a substrate op + external-state rerun through it.
3. Snap on the Logophosis notᵏ curve (design §2.1 row 2).
4. Sandbox per-run time and memory limits; the code-running action (stage 3).
5. Carried: public manual Ch3–5, Axis-5 field-driven branch membership, receptor-freeze-on-rewake,
   Logophosis continual mode, Link-0 compaction, Kinopsis regression, four pre-existing v2 test
   failures (TestCredit ×2, TestGrowth ×1, TestDreamCycle ×1).

## State of the build / Pointers
- Branch `conscience-core`; s065 work committed and pushed (see `git log -3`). `main` still at
  e8e7b40 (v0.3.3).
- **Deliberately uncommitted (local to this PC):** everything matron-written or derived in
  `outputs/incubator/`: s064's `stage1_*` data and `vee_stage1.pt`; s065's
  `stage2_para_in_part*.tsv`, `stage2_para_part*_c*.jsonl` (paraphrases; part9 = the fix job),
  `stage2_keys.jsonl`, `stage2_keys.npz` (52 MB), collect logs. Rebuild: paraphrases need the
  matron again; keys via `stage2_collect.py` (~75 min).
- Memory (this PC only): `stage2_fact_memory_result.md`, `incubator_matron_arc.md`,
  `external_state_depth_result.md`.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)
`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `archive/runs/`,
`trioron/legacy/outputs/`, `notebooks/` checkpoints, output PNGs / uncommitted
logs untracked; `outputs/data/**` caches.

## Environment notes
- `/home/marcrockhat/trioron-project/`, branch `conscience-core`, Python 3.10.12, torch 2.11.0,
  WSL2, `python3` (NOT `python`), 12 logical CPUs (6 cores), 7.6 GB RAM. New this session:
  `libzim` (pip --user), `bubblewrap` (apt). Already present: `llama_cpp` 0.3.23,
  `transformers` 5.8.0.

---
History pointers (no dated archives): s064 handoff at e67b6ce (incubator design, limb tests, stage 1); s063 handoff at e8e7b40 (Logophosis Mode E transfer,
public manual, branch-partition finding); s062 at c4ac3db (seq gotchas, organism API).
