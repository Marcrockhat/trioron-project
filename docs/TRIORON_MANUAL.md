# TRIORON — Canonical Reference Manual

> **Purpose.** A session-stable ground-truth reference for *what trioron is*, written
> because the assistant's recollection of the fundamentals drifts between sessions
> (e.g. forgetting the triparametric node and the optimization axes). This is checked
> in (so it syncs across PCs, unlike `~/.claude` memory) and is **subordinate to
> `paper/v3/spec.md`** (the 3464-line authoritative spec). When this and the spec
> disagree, the spec wins — and fix this file. Every claim below carries a spec §
> or `file:line` so it can be re-verified, not trusted.
>
> Maintained by: rewrite/extend whenever a session re-derives a fundamental that
> isn't here. Last verified: session 014 (2026-06-04).

---

## 0. One paragraph

Trioron is a **growing, self-organizing neural substrate** built from one object — the
**cell** — that learns *continually* (many tasks in sequence) while **minimizing
catastrophic forgetting**, under a hard parameter **envelope**. It is not a fixed MLP:
cells divide (grow), differentiate (express phenotypes via genes), lock (consolidate),
and rehearse (replay). Its deployment role is a **personalization / memory layer** that
context-conditions a frozen LLM ("inner voice"), and a **meta-policy over a frozen
model** (learn-to-*use*, not learn-from). Classification benches are mechanism
stress-tests, not the product. (spec §1; memory: `device_conscience_pattern`,
`learn_to_use_not_from_principle`.)

---

## 1. The node is NOT just weights+bias — it is (v1.1) triparametric

**This is the #1 thing to not forget.** A trioron node carries **three** per-cell
parameter roles, not two:

| param | symbol | role |
|---|---|---|
| incoming **weights** | `W` (`weights[fan_in, output_dim]`) | edge weights into the cell |
| **bias** | `b` | per-output bias |
| **axonal gain** | `u` (`routing_scale`) | per-cell output multiplier — the cell's "volume knob" |

The legacy/v1.1 forward is literally `F.linear(x, W * routing_scale.unsqueeze(1), b)`
(`trioron/legacy/dreaming.py:602`). "**Routing starvation**" (a dream operation) drives
a victim cell's `routing_scale → 0` to silence it without touching its weights
(`legacy/dreaming.py:597-617`). So the node is **triparametric: (W, b, u)**.

**Version caveat (the subtle part that causes drift):**
- **v1.1 / legacy** (`trioron/legacy/`): full triparametric node `(W, b, routing_scale)`.
- **v2.0 default substrate** (`trioron/core/`): the arena carries **`bias` + `edge_weight`**
  as the trainable tensors (`construct.py` `trainable_tensors()` returns
  `[arena.bias, arena.edge_weight]`); **axonal-gain is not in the default v2 forward** —
  it is **Axis 4**, reachable through the compat shim (§4 below). The world experiments
  use the v2 core, so there `W, b` are the live params — but the *architecture* is
  triparametric, and axonal-gain is a real lever we have used (routing starvation,
  dream consolidation).

Per-cell **non-trainable** state also matters and is easy to forget (spec §2.1):
- `engagement` = running activation rate → drives **credit-based locking** (active→dormant).
- `utility` = running gradient magnitude → drives **pruning / saliency**.
- `state` ∈ {`active`, `dormant`} (reversible); `position` ∈ [0,1]³; `epigenome` (genes);
  `rank` (topological); `forward_inclusion` (false for astrocytes).

---

## 2. The epigenome — phenotype is a *gene*, not soft routing (spec §2.2)

A 16-bit mask per cell. Phenotype is a **structural** decision made by the lifecycle
(inheritance / division / explicit API), **never optimized by SGD**.

| bit | gene | meaning |
|---|---|---|
| 0 | `linear` | default op; on for every active cell |
| 1 | `attention` | Q/K/V head (lineage-tagged) |
| 2 | `conv` | convolutional receptor (spatial fan-in) |
| 3 | `recurrent` | self-edges, unroll depth in `phenotype_data` |
| 4 | `dendrite` | fan-in partitioned into branches w/ per-branch nonlinearity (**Axis 5**) |
| 5 | `perception` | consumes raw input; can't lock before task 1 ends |
| 6 | `output` | contributes a head logit (one gene per class slot) |
| 7 | `credit_eligible` | engagement-credit may lock this cell |
| 8 | `recyclable` | slot reclaimable under envelope pressure |
| 9 | `weight_tied_lineage` | shares weight tensor with lineage siblings |
| 10–15 | reserved | future genes |

Multiple expression genes co-fire (e.g. `attention|conv`): the scheduler runs the cell
once per expressed gene and **adds** contributions. The five shipped phenotypes are
**linear / attention / conv / recurrent / dendrite** (spec §3.2–3.6).

> **Experiment note:** the embodied-organism arc added `MIRROR = 10` as a marker gene
> (`trioron/core/epigenome.py`) — mirror cells dispatch as LINEAR; "mirror" is
> connectivity + localized credit, not a new forward op. This is an experiment
> extension into the reserved range, not part of the base spec.

---

## 3. The substrate (spec §2)

- **Cell** (§2.1): the only object; user-facing `Cell` is a view into the arena.
- **Arena** (§2.4): Structure-of-Arrays — all per-cell fields are parallel tensors
  (`arena.bias`, `arena.edge_weight`, `arena.edge_src/edge_dst`, `arena.epigenome`,
  `arena.state`, `arena.engagement`, `arena.utility`, `arena.edge_protected`, …).
  **The substrate is a flat sparse edge list, NOT layered** — there is no native
  "forward from layer N" (memory: `substrate_no_spatial_memory`).
- **States** (§2.8): `active` ⇄ `dormant` (reversible). Dormant cells hold parameters
  but their grads are zeroed (`scheduler.zero_dormant_grads()`, honors `edge_protected`).
- **Scheduler** (§2.7): rank-batched dispatch by phenotype. `Substrate` exposes
  `forward`, `compile`, `end_task`, `trainable_tensors`, `zero_dormant_grads`,
  `last_activations` / `live_activations` (`construct.py`). **No built-in hook calls the
  learning machinery — the training driver must call credit/manifold/dream itself.**
- **Envelope** (§2.5): bounded parameter budget; growth is blocked past the cap.
- **Astrocytes** (§2.11): cells with `forward_inclusion=false` — hold parameters
  (e.g. manifold sketches) but are not in the forward pass.

---

## 4. The optimization AXES (v1.1) and their v2 expression

The other thing the assistant forgets: trioron has a set of **axes we optimized for
adapting to a problem and minimizing catastrophic forgetting**. In **v1.1** each was an
explicit write function; in **v2.0** they **collapsed into epigenome genes** (spec §1.3
line 103, §8.1 line 3125). Reach the v1.1 API through the compat shim.

| axis | v1.1 write fn | what it tunes | v2.0 expression |
|---|---|---|---|
| **Axis 1** | `set_input_source` | input sources / long-range inputs | absence of a sequential-source convention |
| **Axis 2** | `archive_input` | archived/dormant cells as inputs | dormant cells + `edge_protected` |
| **Axis 3** | `insert_layer` | depth insertion | self-organized depth via division ranks |
| **Axis 4** | `set_axonal_gain` | **per-cell output gain (`routing_scale`)** | the triparametric `u`; via shim |
| **Axis 5** | `grow_branch` | **dendritic compartmentalization** | `dendrite` gene (bit 4) — ships LIVE |
| **Axis 6** | `axis6_spawn` | credit-driven spawn | credit/`divide` lifecycle |

(Memory: `trioron_2_0_axis5_handoff`, `phase_6_dendrite_delta_pass` — Axis 5 passed its
falsification gate, K=2 dendrite lifts K=1 by +0.49–0.76 abs on concentric rings.)

---

## 5. Continual-learning machinery — the anti-forgetting toolkit (spec §4)

This is the heart of trioron and the toolkit the world experiments kept **bypassing**.
None of it auto-runs; the driver wires it.

1. **Credit-based locking** (§4.1, *the primary anchor*) — `CreditTracker`
   (`learning/credit.py`). Drive `update_engagement(activations)` + `update_utility()`
   each batch; `consolidate()` at a task boundary locks high-engagement / low-utility
   cells → `dormant` → protected by `zero_dormant_grads`. **Locks ~1 cell per boundary
   by design** (`_lock_cap ≈ ceil(√n·0.078·φ)`); `consecutive_tasks` defaults to 4
   (needs many boundaries) — set 1 for a single boundary. Deliberately chosen as a
   **hard** alternative to EWC's soft penalty (`legacy/api.py:1396`).
2. **Frustration trigger** (§4.2) — `FrustrationDetector` (`learning/frustration.py`):
   loss-plateau detector → a multiplier that gates **growth** (`check_growth_trigger`).
3. **Dream cycle** (§4.3) — `dream_cycle(substrate, credit, archive, …)`
   (`learning/dream.py`): replay → consolidate → rejuvenate. NOTE: it does **not** call
   `archive.finalize_all()`; `replay_batches` only returns DORMANT astrocytes, so
   finalize at the boundary first. `_run_replay` treats codes as **input-space**
   (`x = batch[:, :n_perc]`) and rehearses via `cross_entropy` — so **action-as-class
   works** for RL policies.
4. **Rejuvenation** (§4.4) — dormant→active for re-engaged cells (`learning/rejuvenate.py`).
5. **Manifold replay** (§4.5, *the headline CL result*) — `ManifoldArchive`
   (`learning/manifold.py`): per-class (μ,σ) **astrocyte** sketches → pseudo-rehearsal.
   **Beat fixed-EWC by +6.4σ on chained-15** at 30 KB (memory: `manifold_replay_result`).
   - **Full covariance (`full_cov=True`) → Mahalanobis `log_likelihood_full`** is the
     **full-softmax accuracy pump**: chained-15 full 0.55→0.68 (dual-manifold H-routing)
     →0.76 (full-cov). full-cov **is in core**; the **routing orchestration is bench-only**
     (`bench_chained_15_v2.py`, not promoted). (commits `62aa57e`, `7e561e4`.)
   - **z2 = the second, H-space (interior-code) ROUTING manifold** — infers which task/
     context a query belongs to from the stable interior code, sidestepping head drift.
     Validated to transfer to the world (session 014: 0.71 full-cov routing).
6. **EWC** (§4.6) — baseline only; per-weight Fisher is **not** in the v2 default
   (instantiated on demand for competitor runs).

**Proven CL numbers (chained-15, n=10):** full-softmax **0.601** / domain 0.677 /
task-aware **0.961** (manifold-grown, 30 KB). PackNet sat at 0.046 full. (paper §4.)

---

## 6. Lifecycle (spec §5)

- **Division / growth** (§5.1) — `divide(arena, parent_id)` (`lifecycle/grow.py`):
  mitosis, child inherits a fraction of edges + new ones; gated by frustration.
- **Grafting** (§5.3) — transplant donor cells into a recipient substrate
  (`lifecycle/graft.py`); the basis of multi-branch absorption / skill-packs.
- **Ship-Wake-Extend** (§5.4) — serialize (`ship`), reload (`wake`), resume training and
  grow within a new cap. The lifetime-deployment loop.
- **Compaction** (§5.5) — recycle low-saliency dormant cells (`compute_saliency` =
  0.6·utility + 0.3·engagement + 0.1·downstream, `lifecycle/saliency.py`).
- **Developmental program** (§5.7) — stem cells, morphogens, redifferentiation.

---

## 7. The forgetting story (the unifying frame, session 014)

The forgetting seen in the embodied world is the **same** forgetting beaten on
chained-15: **unanchored shared parameters (the output head / base policy) drift toward
the last-trained task** (Rocky's *epigenetic-lock* read — `lam=0` → drift → bias to
recent task; memory `epigenetic_lock_hypothesis`). Two regimes:

- **Task-aware** (a task mask at inference) — chained-15 reached **0.96**.
- **Full-softmax** (no task selector) — chained-15 peaked **~0.60**; the **world has no
  task selector, so it lives here.**

The fix that recovered full-softmax was **not** EWC and **not** task-masking — it was
**manifold machinery**: replay (defends weights) **+ H-space full-cov routing** (a
*learned task selector* that picks the skill from the stable interior code). That is the
validated direction for the world (open item: build replay+router; promote routing to
core).

---

## 8. Drift-corrections — things the assistant keeps getting wrong

- ❌ "A node is weights + bias." → ✅ **Triparametric: weights, bias, axonal-gain
  (`routing_scale`).** v2 default forward uses W,b; axonal-gain is Axis 4 / shim.
- ❌ "Trioron is just an MLP that grows." → ✅ Cells with **genes/phenotypes**, **credit
  locking**, **manifold replay**, **dream**, **grafting** — a continual-learning organism.
- ❌ "There are no axes." → ✅ **Six axes** (input-source, archive-input, insert-layer,
  **axonal-gain**, **dendrite**, credit-spawn) — v1.1 functions, v2 genes.
- ❌ "Just retrain / fine-tune to fix forgetting." → ✅ Use the **native machinery**
  (credit-lock + manifold replay + dream); don't hand-roll EWC. Default to *reusing*
  trioron's mechanisms before writing new code (Rocky, session 014).
- ❌ "EWC/Fisher is the trioron anchor." → ✅ The v2 anchor is **credit-based DORMANT
  locking** (hard), chosen *over* EWC's soft penalty. Manifold replay is the headline win.
- ❌ "The substrate is layered." → ✅ Flat sparse edge list; no native forward-from-layer.
- ❌ "Pruner uses strict §3.2 |a·g|." → ✅ Default pruner is `max(act_grad, act_var)`
  (memory `pruner_combined_mode_deviation`).
- ⚠️ The world/RL experiments run **raw Adam over `trainable_tensors()` and bypass** the
  credit/manifold/dream machinery — that bypass is *the* recurring source of "world
  forgetting." Wire the native machinery in.

---

## 9. Source-of-truth pointers

- **`paper/v3/spec.md`** — authoritative (9 sections, ~3464 lines). §9.14 cross-index;
  §9 directory partition; §6 binding perf contract (Phase 1 = 50K params, CPU,
  full-fidelity recording).
- **`paper/paper.tex`** — the integrated paper (chained-15 / manifold / archive numbers).
- **Core:** `trioron/core/` (arena, scheduler, construct, epigenome, state).
  **Learning:** `trioron/learning/` (credit, manifold, dream, frustration, rejuvenate).
  **Lifecycle:** `trioron/lifecycle/` (grow, graft, ship, wake, compact, saliency).
  **Legacy v1.1:** `trioron/legacy/` (triparametric node, EWC, axes API).
- **Chained-15 routing to port:** `experiments/bench_chained_15_v2.py` (H-routing,
  `--full-cov`, `--perc-mixture-k`).
- **`docs/handoff/HANDOFF.md`** — current session state (rewritten every session).
- **Related project:** `~/project-aidos/` vendors the trioron substrate (separate memory dir).
