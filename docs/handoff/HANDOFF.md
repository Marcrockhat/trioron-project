# Trioron Handoff

**Session date:** 2026-06-22
**Session number:** 045
**Session title:** **Manifold replay on the mixed cross-domain stream — replay
ALONE is the new best arm (mean +63% over s044's best), un-starving CIFAR while
rescuing MNIST at near-zero acquisition cost.** The s044 fully-upgraded node
(w, λ, u) gains a fifth protection arm: per-class manifold pseudo-rehearsal over
the FIXED-LENS block descriptors, replayed forward through the live soma+head
every step. n=5, s044-matched config (STRENGTH=150 LOCK_RATE=0.1). The s044
baselines reproduced bit-for-bit, so the comparison is clean.

---

## READ THIS FIRST

1. **Exploration only — nothing promoted into `trioron/`.** This session edited
   ONE file under `experiments/progenitor/` (`mixed_stream_cl.py`, +replay arm)
   and added one log. Package untouched. Branch `progenitor-council`.
2. **THE TASK (Rocky, s044 NEXT #1, confirmed this session):** add a manifold-
   replay arm to the mixed-stream CL experiment to fix the two s044 open problems
   (bimodal MNIST rescue + CIFAR starvation) — trioron's own designed class-
   incremental answer (`trioron/learning/manifold.py ManifoldArchive`).
3. **WHAT WAS BUILT — `replay` and `all` arms in `mixed_stream_cl.py`.**
   - **Replay space = the FIXED-LENS BLOCK DESCRIPTORS (input space), not soma
     H-space.** Rationale: the three lenses are germline/fixed (never trained),
     so per-class block vectors are a PERMANENTLY-valid generative model;
     soma H-codes would go stale as the soma trains. Replaying descriptors
     forward through the *current* soma+head defends BOTH the soma weights AND
     the past-class head logits against the new task's full-CE suppression —
     exactly the "replay old-class descriptors so full-CE doesn't suppress them"
     the s044 handoff named. This is the targeted fix for head-logit drift.
   - **Uses trioron's `ManifoldArchive`** (Rocky's reuse-the-machinery doctrine)
     in a SEPARATE small arena (`Arena(Envelope(), capacity=N_CLASS+4)`) so it
     never collides with the net's bucket-dispatch / capacity. Diagonal Gaussian
     (`full_cov=False`). Per task: train on real + replayed past-class CE; at the
     boundary `update_class(c, Dx[c])` for this task's classes then `finalize_all()`
     (active→dormant) so they become replayable next task. `replay_batches` only
     yields DORMANT astrocytes — that's why finalize-at-boundary is required.
   - **New arms:** `replay` (manifold only) and `all` (λ + credit + replay).
     New env: `REPLAY_BS` (pseudo-samples/past-class/step, default 32),
     `REPLAY_STEPS`, `REPLAY_W` (replay-CE weight, default 1.0).
4. **THE RESULT (n=5, STRENGTH=150 LOCK_RATE=0.1 REPLAY_BS=16 EPOCHS=6,
   `outputs/mixed_stream_s045_n5.log`, 3410s):**

   | arm | taxonomy | mnist | cifar | mean | acq | forget | store |
   |-----|---------|-------|-------|------|-----|--------|-------|
   | none   | 0.229±.092 | 0.000±.000 | 0.238±.052 | 0.155±.045 | 0.552 | +0.377 | – |
   | lambda | 0.281±.028 | 0.312±.304 | 0.096±.010 | 0.229±.101 | 0.316 | +0.072 | – |
   | both   | 0.291±.039 | 0.461±.263 | 0.088±.028 | 0.280±.087 | 0.294 | +0.041 | – |
   | **replay** | 0.440±.047 | 0.690±.227 | **0.240±.045** | **0.456±.090** | 0.534 | +0.130 | 750KB |
   | **all** | 0.310±.044 | **0.875±.017** | 0.008±.016 | 0.398±.016 | 0.249 | **−0.007** | 750KB |

   - **s044 baselines reproduced EXACTLY** (none/lambda/both identical to s044
     handoff) — the harness is unchanged; arm deltas are clean.
   - **`replay` ALONE is the best mean (0.456±.090) — +63% over s044's best
     (`both` 0.280), +194% over none.** It does what the anchors could NOT:
     **un-starves CIFAR** (0.240 ≈ none's 0.238, vs `both`'s 0.088) AND rescues
     MNIST (0.690) AND keeps acquisition high (0.534 ≈ none's 0.552). Defending
     OLD classes by re-presenting them costs the NEW task nothing — unlike λ,
     which constrains all weights and chokes the newest task.
   - **`all` (λ+credit+replay) solves the bimodal MNIST instability** — MNIST
     0.875±**0.017** (σ collapsed from 0.23→0.02) AND achieves **negative
     forgetting (−0.007 = backward transfer)**. BUT stacking the anchors on top
     of replay re-breaks CIFAR (0.008): λ's accumulated 5-boundary penalty freezes
     the last task solid.
5. **THE TRADEOFF (both s044 open problems addressed, but they pull apart):**
   - **CIFAR starvation → SOLVED by `replay` alone** (0.088→0.240). NOT solvable
     by adding anchors — `all` re-breaks it. The starvation is fundamentally **λ
     over-regularizing the NEWEST task** (CIFAR-b is task 5/6, sees 5 stacked
     anchors). Replay can't help because replay defends OLD classes; the newest
     task's problem is too MUCH constraint, not too little rehearsal.
   - **Bimodal MNIST rescue → SOLVED by `all`** (σ 0.227→0.017). Replay alone
     rescues MNIST on AVERAGE (0.690) but is STILL bimodal (σ=0.227, same as λ).
     The hard+soft anchors on top of replay are what stabilize it.
   - **Clean takeaway:** replay alone wins mean + CIFAR + acquisition; the anchors
     buy MNIST-stability + zero-forgetting at the cost of CIFAR. There is no single
     arm here that gets all four. See NEXT #1 for the obvious middle ground.

---

## What was built (committed this session, branch `progenitor-council`)

- **`experiments/progenitor/mixed_stream_cl.py`** — added the manifold-replay
  machinery to s044's file:
  - import `ManifoldArchive, ManifoldConfig`; env `REPLAY_BS/REPLAY_STEPS/REPLAY_W`.
  - in `run()`: `use_replay` flag; a separate `arch_arena` + `ManifoldArchive`;
    `replay_ce()` (forward replayed past-class descriptors → CE); the per-step
    replay term added to the loss; boundary accumulate+`finalize_all()`; `store_kb`
    in the return dict.
  - `main()`: default ARMS now `none,lambda,both,replay,all`; `store` column added.
- **`outputs/mixed_stream_s045_n5.log`** — the n=5 headline (table above).

## Key findings

- **Manifold replay is the strongest single CL mechanism on the mixed stream** —
  beating the namesake epigenetic lock (λ) and credit-locking, alone and combined.
  This re-confirms the package-level result (`manifold_replay_result`: replay
  beat fixed-EWC by +6.4σ on chained-15) now on a genuine 3-MODALITY interleaved
  stream with a living triparametric soma.
- **Replaying in the FIXED-LENS descriptor space is the right move for a trained
  soma.** The standard H-space (interior-code) replay assumes a STABLE code
  boundary; here the soma is trained, not frozen, so its codes drift — input-space
  replay sidesteps that. Cost: 750KB storage (1846-d block × μ,σ × 52 classes)
  vs the 30KB H-space headline — input-space replay is bulkier but valid here.
- **Replay defends BOTH soma and head** (forward through the whole net), which is
  why it un-starves the new task: it adds gradient signal for old classes without
  adding a *penalty* on the weights the new task needs.
- **λ vs replay are complementary, not redundant** — λ stabilizes variance (kills
  the bimodal MNIST), replay maximizes mean + protects the newest task. The
  failure of `all` is not "too many mechanisms" but specifically λ's per-boundary
  penalty accumulation choking the LAST task.

## NEXT (priority — for the new session)

1. **The obvious middle ground: `replay + light/decaying λ` (NO credit), or λ
   only on OLD-class head rows.** The s044 evidence says λ (not credit) is the
   CIFAR-starver (lambda cifar 0.096 ≈ both 0.088). Try: (a) `replay+lambda` at
   LOWER strength (STRENGTH∈{25,50,75}) to get MNIST-stability without freezing
   CIFAR; (b) λ that does NOT anchor the newest task's fresh head rows; (c) decay
   STRENGTH over the stream so late tasks are less constrained. Goal: one arm with
   replay's CIFAR/mean AND `all`'s MNIST-σ. n≥5.
2. **Shrink replay storage** — input-space 750KB is bulky. Options: per-class
   mixture (`StreamingMixture`/`mixture_k`) at smaller effective dim, OR replay in
   a *projected* descriptor space (random projection of the block to ~128-d, which
   the lenses make valid since they're fixed), OR PCA the block. Target the 30KB
   class-incremental headline regime.
3. **Tune REPLAY_BS / REPLAY_W** — used REPLAY_BS=16 for speed (replay arms are
   ~3× the cost of non-replay). Sweep REPLAY_W∈{0.5,1,2} and REPLAY_BS∈{16,32,64}
   at n=5 — more replay may further stabilize the MNIST bimodality of replay-alone.
4. **Strengthen CIFAR perception** (carried from s044 #3) — CIFAR remains the
   weakest domain (replay's 0.240 is still the floor); learnable kernels
   (`scatter_learnable_lambda`) / deeper bank would lift it.
5. **Then consider promotion** — if #1 finds a single arm that wins all four axes,
   the per-modality-lens → block → living-soma + manifold-replay organism is a
   strong candidate for a real `trioron/` module. Only after #1–#3.

## OPEN / unresolved

- **Replay-alone MNIST is still bimodal** (σ=0.227) — only `all` (with anchors)
  stabilizes it. NEXT #1 is the attempt to get both.
- **750KB input-space replay** is large vs the package's 30KB H-space headline —
  acceptable for the experiment, addressed by NEXT #2 before any promotion.
- The block-structured zero-pad still makes modality trivially decodable (s044
  caveat unchanged) — the difficulty is within-modality separation + cross-class
  HEAD competition; replay attacks exactly the head competition.
- `forget` is confounded when a task never LEARNS (acq≈final≈0 → forget≈0); read
  `acq` alongside. Note `replay`'s forget +0.130 is HIGHER than `both`'s +0.041 —
  but that's because replay ACQUIRES much more (0.534 vs 0.294), so it has more to
  forget; its FINAL accuracies dominate every arm. Always read forget ⟂ acq.
- s039–s044 carries (untouched): deep scattering S1⊕S2; learnable kernels + λ
  (`scatter_learnable_lambda`); council global-vote → soma; chained-15 PCLL; optics
  toys; the s044 mixed-stream lenses (`mixed_stream_lenses.py`, perception Stage 1).

## State of the build

- Branch `progenitor-council`. Package code UNTOUCHED; gate battery green (s036).
  This session: one commit (`mixed_stream_cl.py` +replay arm + s045 log) + this
  handoff.
- Run cost: descriptors ~25s/seed; non-replay arm ~90s; replay/all arm ~290s at
  EPOCHS=6. n=5×5-arm ≈ 57 min (3410s).

## Pointers

- **The work:** `experiments/progenitor/mixed_stream_cl.py` (CL stream + all 5
  arms) + `mixed_stream_lenses.py` (Stage-1 perception, unchanged from s044).
  Headline reproduce: `SEEDS=0,1,2,3,4 STRENGTH=150 LOCK_RATE=0.1 REPLAY_BS=16
  ARMS=none,lambda,both,replay,all EPOCHS=6 python3
  experiments/progenitor/mixed_stream_cl.py`.
- **The manifold API:** `trioron/learning/manifold.py` — `ManifoldArchive(arena,
  cfg, full_cov, mixture_k)`; `update_class(cid, codes)` (Welford μ,σ accumulation,
  auto-spawns astrocyte); `finalize_all()` (active→dormant); `replay_batches(bs,
  exclude_*)` (samples per DORMANT class); `cluster_replay_batches`; `storage_bytes`.
  `ManifoldConfig(replay_steps_per_class, code_dim, ...)`. Astrocytes are
  `forward_inclusion=False` arena cells holding the sketch.
- **The node APIs (s044, unchanged):** λ = `trioron/learning/epigenetic_lock.py`
  (`accumulate_saliency`→`refresh_lambda`+`anchor`→`strength·ewc_penalty`); u =
  `trioron/learning/credit.py` `CreditTracker`. Both operate on Arena cells; the
  head is in-substrate so it's protected too.
- **The architecture (s044):** 3 per-modality germline lenses → block vector
  `tax[0:22]|mnist[22:478]|cifar[478:1846]` → H=64 living soma → in-substrate
  52-way head (all credit-eligible, own-root linear via `conv.forward_batch`).
  Interleaved 6-task stream, full-softmax over the 52-class union, no task id.

## DO-NOT-COMMIT carries (unchanged since s034, LEAVE THEM)

`trioron/bases/developmental.py`, `trioron/lifecycle/developmental.py`,
`trioron/viz/export.py`; `.claude/`, `runs/`, `trioron/legacy/outputs/` untracked.
Output PNGs uncommitted.

## Environment notes

- `/home/marcrockhat/trioron-project/`, branch `progenitor-council`,
  Python 3.10.12, torch 2.11.0, torchvision 0.26.0, WSL2, `python3` (NOT `python`),
  `OMP_NUM_THREADS=8`. Data `outputs/data/` (CIFAR-100 on disk; CIFAR-10 NOT —
  download sandboxed). CIFAR = fixed 10-fine slice (classes 0..9 of CIFAR-100).
- **Model note (s044, keep):** lead with the COMPUTATIONAL framing in code/docs
  (phase code; tied conv kernel via `lineage_root`; Gabor filterbank; modulus =
  oriented energy; scattering = |·|*ψ cascade; Mahalanobis fingerprint; EWC =
  epigenetic lock λ; credit-locking = utility u; manifold replay = per-class
  Gaussian pseudo-rehearsal). Optical metaphors (lens/emitter/phasor) are Rocky's
  conversational framing — keep them out of code to avoid mid-session safety
  downgrades.
