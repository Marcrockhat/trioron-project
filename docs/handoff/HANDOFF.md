# Trioron Handoff

**Session date:** 2026-11-23
**Session number:** 064
**Session title (s064: THE INCUBATOR — reflex brain + growing organism + matron):**
**Rocky's new direction: stop making trioron itself a smart edge model. A frozen Qwen2.5-1.5B is
the fast, often-wrong REFLEX brain; a trioron ORGANISM grows on top of it; a Fable MATRON (via
Claude Code on Rocky's subscription) raises it in a sandbox. Design drafted and largely signed
off; the two "dead limb" questions were tested (depth = step count over written symbols: .998
flat to hop-8); stage 1 (awareness) built and PASSED its gate modestly (.817 vs .780).**

## READ THIS FIRST
- **Design of record: `docs/design/incubator_matron.md`.** Read it before anything else this
  arc touches. Roles, the missing-limb plan (§2.1), five stages with gates (§3), self-hosting
  milestones (§3.1), sandbox (§4), growth-must-pay + memory ceiling (§5), matron loop (§6).
- **Decisions Rocky made this session (binding):**
  - The incubator organism has **no parameter cap**; limit = 85 % of WSL RAM, swap excluded.
    Spec §6.1 and CLAUDE.md carry the exemption. Benches and CI keep the 50 K cap.
  - **Paper is published in the repo only.** No ArXiv, venue, endorsement outreach or Reddit.
    Do not size experiments to venue requirements.
  - Environment = sandbox in WSL; it runs code and has offline Simple English Wikipedia.
  - Creativity = finding an action along a SCAMPER route (stage 5).
  - Stage-1 data = Wikipedia-derived questions; code in `trioron/organisms` + `trioron/senses`
    (package) and `experiments/incubator/` (world side).
- **Rocky's framings to keep:** humans hallucinate too (inner doubt is the weakest error
  channel: told > doesn't-fit > doubt); the self-check monologue is stage 4's concrete form;
  "how Linux was compiled on Minix" = self-hosting milestones; most humans are ~4–5 steps deep.
- Suspect the branch partition first for 1-layer relational nulls (s063 rule, still #1).

## Arc 1 — The limb question (is the monkey paralysed?)
Both tests in `experiments/progenitor/`, n=3.

### hop-k with duplicated inputs (`logic_chain.py`, new arm `tied_dup`; log `outputs/logic_chain_s064_hop_dup.log`)
| task | tied | tied_dup |
|---|---|---|
| hop-2 | .935±.018 (5 applications, 571 s) | **.982±.007 (1 application, 45 s)** |
| hop-3 | .583±.007 | .706±.015 |
| hop-4 | .573±.005 | .608±.005 |

The partition hid a real ability (hop-2 in ONE quad step) but is NOT the wall: re-applying a
learned step end-to-end is still not learned.

### External state (`hop_external_state.py`; log `outputs/hop_external_state_s064.log`)
1-hop organ (start one-hot + map, dup input, one quad link, 16,272 params), p = .9984; frozen,
then iterated with no further training:
| depth | snap (argmax→one-hot) | soft (probabilities) |
|---|---|---|
| 2 | .998±.002 | .997±.002 |
| 4 | .997±.002 | .991±.004 |
| 8 | .998±.002 | .987±.006 |

A 2-hop-from-any-start organ (three-way product, two quad layers) does NOT settle (.641±.008).
**Design rule: one product per step; depth is a step count once the state between steps is a
written symbol.** Caveats: decomposition was GIVEN (intermediate targets), oracle symbols,
cyclic maps forgive errors. Serial graft (organ output symbol → next organ's start input) is
justified but unbuilt; today's `graft()` is side-by-side only.

## Arc 2 — Stage 1, awareness (PASS, modest)
Pipeline: 1,600 Simple English Wikipedia leads → 2,985 questions written by four Sonnet helper
agents (~635 K tokens; the first matron job) → de-duplicated by article to 2,808 → Qwen answers
closed-book (29.2 % correct, ~1.6 s/item, 86 min) → gold-string grading (audit of 30: ~1
doubtful) → organism trained on Qwen's hidden states.

| arm (10 article-level splits unless noted) | AUROC, "reflex is wrong" |
|---|---|
| Qwen's own confidence (min token log-prob, best baseline) | .780±.015 |
| **organism [h_pre,h_post], k=64, 12,834 params** | **.817±.010** |
| paired organism − baseline | **+.037±.009, 10/10** |
| linear readout on the same 64 components | .810±.011 |
| organism + log-probs combined | .841±.011 |
| no-brain control (writer, question/answer length) | .590±.020 |
| organism h_pre only / h_post only (3 splits) | .774 / .763 (do not pass alone) |

Honest reading: passes on every split and within each question-writer pair, but (a) the quad
leaf adds nothing over a linear readout — trioron is not yet earning its place; (b) the passing
arm was chosen after 3-split results, then confirmed on 10 new splits of the same data;
(c) final layer only. Useful: the state BEFORE answering predicts the error about as well as
Qwen's confidence after answering.

## What was built (all in this session's commit)
- `docs/design/incubator_matron.md` — the design.
- `trioron/senses/hidden_state.py` — `HiddenStateSense` (z-score → PCA-k → whiten → dup).
- `trioron/organisms/syneidesis.py` — `Syneidesis` (working name): `wake` / `fit` / `doubt`.
- `experiments/incubator/`: `stage1_leads.py` (seeded article sampler), `reflex.py` (Qwen via llama_cpp: answer + log-probs + hidden state
  in ONE pass), `wiki.py` (offline Wikipedia: search, paragraphs), `stage1_collect.py`
  (resumable), `stage1_awareness.py` (gate), `stage1_controls.py`, `console.py`.
- `experiments/progenitor/logic_chain.py` (+`tied_dup`), `hop_external_state.py`.
- Spec: §6.1 exemption paragraph; §9.16 rows + tree lines for the new files. CLAUDE.md:
  exemption bullet. Test: `tests/test_v2/test_organisms.py::test_syneidesis_doubt_roundtrip`
  (organism tests 6/6).
- Console for Rocky (own terminal): `python3 experiments/incubator/console.py` — ask, see
  answer + Qwen confidence + organism doubt; `/right`, `/wrong [answer]`, `/look [query]`,
  `/learn`, `/quit`. Verdicts are stored and used on `/learn`.
- **Console, after Rocky's first session (commit b11992e):** it froze once on him; NOT
  reproduced (replay ran clean, no OOM kill). Hardened anyway: reflex errors are caught,
  `[thinking...]` / retraining notices, long-question guard, `/look` searches content words
  (whole questions searched badly). If it freezes again, ask for the last input and whether
  `[thinking...]` was showing. Rocky stored no verdicts (Qwen was mostly right on his
  questions); told him `/right` is training data too. **He prefers typed commands over
  one-key shortcuts (fewer errors) — do not add shortcuts.** His questions are easier than
  the training set (29 % correct), so his verdicts are a different distribution.

## Environment facts found this session
- Qwen GGUF already on disk: `~/models/qwen2.5-1.5b-instruct-gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf`
  (0.5B sibling at `~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf`). ~1.8 GB RSS, ~21 tok/s read
  on this Ryzen 5 PRO 4650U (6 cores; use 6 threads, not 12).
- Wikipedia: `~/data/wikipedia/wikipedia_en-simple_all_nopic_2026-09.zim` (1.0 GB, 400 K
  articles, full-text index); `pip install --user libzim` done.
- bubblewrap 0.6.1 installed; verified: scratch dir writable, home invisible, network and
  system writes blocked. Time/memory limits per run NOT set up yet.
- Host disk has ~65 GB free (the 907 GB WSL reports is the virtual disk ceiling).
- `sudo` cannot run through the `!` prefix; Rocky runs it in a separate terminal.
- Rocky's plan: Max 5× ($100). Matron budget cap proposed (20 % of weekly), not yet confirmed.

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

## Open questions for Rocky
1. Name: keep `Syneidesis`?
2. Matron daily/weekly token budget (proposed 20 % of weekly limit).
3. Terms check before any matron-written data is committed. Until then the question files and
   answers stay UNCOMMITTED in `outputs/incubator/` (see below).

## NEXT (s065)
0. **Stage 2 — fact memory** (design §3): key in hidden-state space → stored text; override
   the reflex on repeats and paraphrases; gates = retention after N later lessons, paraphrase
   hits, false-override rate, vs a nearest-neighbour cache. This is where trioron must start to
   earn its place (stage 1 was linear) and where "growth pays" is first measurable; measure
   bytes per parameter here to turn the 85 % RAM ceiling into a parameter figure.
1. Wire console verdicts into a standing loop (the first incubator loop): checkpoint/resume,
   token logging, daily budget stop.
2. Serial graft as a substrate op (`lifecycle/graft.py` wiring="serial") + the external-state
   test rerun through it.
3. Snap on the Logophosis notᵏ curve (design §2.1 row 2).
4. Sandbox: per-run time and memory limits; the code-running action (stage 3).
5. Carried from s063, lower priority now: public manual chapters (Ch3–5), Axis-5 field-driven
   branch membership, receptor-freeze-on-rewake, Logophosis continual mode, Link-0 compaction,
   Kinopsis regression, four pre-existing v2 test failures (TestCredit ×2, TestGrowth ×1,
   TestDreamCycle ×1).

## State of the build / Pointers
- Branch `conscience-core`; s064 work committed and pushed (see `git log -3`). `main` is still
  at e8e7b40 (v0.3.3); nothing merged or released this session.
- **Deliberately uncommitted (local to this PC):** `outputs/incubator/` data —
  `stage1_leads*.{jsonl,txt}`, `stage1_qa_part*.jsonl` (matron-written questions),
  `stage1_answers.jsonl`, `stage1_hidden.npz` (17 MB), `syneidesis_stage1.pt`. Rebuildable:
  leads via `experiments/incubator/stage1_leads.py` (seeded), questions need
  the matron again, answers via `stage1_collect.py`. The result logs ARE committed.
- Memory (this PC only): `incubator_matron_arc.md`, `external_state_depth_result.md`,
  `paper_repo_only_publication.md`; ArXiv pathway + endorsement-outreach memories deleted.

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
History pointers (no dated archives): s063 handoff at e8e7b40 (Logophosis Mode E transfer,
public manual, branch-partition finding); s062 at c4ac3db (seq gotchas, organism API).
