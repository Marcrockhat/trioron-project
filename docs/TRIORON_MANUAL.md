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
> isn't here. Last verified: session 033 (2026-06-12) — PCLL §10.6–10.10
> amendment mirrored (§2: mixed stream, per-class settlement, composer arm,
> manifold adapter, census); dispatch-table + σ-readout drift traps added (§8).

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

## 1. The node is triparametric — (w, λ, u) — and λ is the namesake (DO NOT FORGET)

**This is THE thing the assistant keeps forgetting.** The name *tri·oron* = **"three
coupled state variables per node"** (founding blueprint `trioron_blueprint.md:23-25`).
The three are NOT (weights, bias, gain). They are:

| param | symbol | role |
|---|---|---|
| core signal **weight** | `w` | learnable edge weight; gradient descent (per-edge) |
| **epigenetic lock** | **`λ`** | **PER-NODE plasticity GATE** (`λ ∈ ℝⁿ`, one per cell). High λ stiffens the cell against drift; low λ = plastic. |
| **utility** | `u` | per-node EMA contribution `sign(reward)·|act·grad|`; pruning + growth-trigger |

**λ is a general gate driven by an importance signal, NOT a fixed Fisher buffer (paper
§A.1, `legacy/node.py:919 set_lambda`).** Whatever the signal, **`λ_i` is the ROW-SUM of
the cell's incoming-edge importance, NOT the row-mean** (the mean divides below any usable
floor at high fan-in → silent-zero EWC penalty; fixed in `81d3785`/`bba9420`). A
**`LAMBDA_FLOOR`** (ε-floor) keeps λ from silently zeroing.

**The driver: `|w·g|` saliency, NOT Fisher `g²` (corrected session 015 — the recurring
mistake).** Trioron's *native* importance signal is `|weight · gradient|` — the same
family used by **KIBRA edge-tagging** (`learning/dream.py kibra_tag`), the **utility `u`**,
and the **pruner**. Fisher (`g²`) is the imported academic-EWC signal and it is the wrong
default here: at convergence `g → 0`, so empirical Fisher collapses to the floor (λ goes
uniform = dead — verified session 015). `|w·g|` keeps the **weight-magnitude** factor, so
large settled weights stay important even as `g` shrinks (a first-order Taylor/OBD
saliency) — it differentiates λ without the washout, using plain labels. (If Fisher is
ever wanted, use the *model-distribution* target `fisher_loss` so it doesn't wash out.)

**v2 API (`trioron/learning/epigenetic_lock.py`):** `accumulate_saliency` (|w·g| EMA, the
default driver) / `accumulate_fisher` (g² EMA) → `refresh_lambda` (row-sum + floor →
`arena.node_lambda`); `set_lambda(signal, mode)` to drive λ from reward/environment/
attention directly; `anchor` (snapshot ŵ at a boundary); `ewc_penalty` (the soft pull, add
`strength·penalty` to the loss); `modulated_scale` (optional `1/(1+λ)` grad scaling);
`fisher_loss` (model-dist target). Storage rides ship/wake in the arena (`edge_fisher` —
the importance EMA buffer, holds g² OR |w·g| by which accumulator you call —
`edge_anchor`, `node_lambda`, `bias_fisher`, `bias_anchor`).

**But Fisher is only ONE driver.** `set_lambda(signal, mode)` (`absolute`/`additive`/
`multiplicative`) lets *any* signal gate plasticity — the **epigenetic** generalization
(BDNF/methylation analogy): **reward** magnitude (protect high-reward cells), an
**environment sense** (temperature/stress on an edge device — "λ becomes a literal
environment sense"), **attention** masks, or hand-injected priors (freeze = large λ,
wake = 0). **This is what gives λ intrinsic value** — driven by reward/environment it's
the cell's response to experience, not a dead extrinsic statistic. *For the embodied
organism, drive λ from survival-reward.*

EWC penalty: `0.5·Σ_i λ_i·Σ_j (w_ij − w_anchor_ij)²` (per-node λ over that cell's edges).
Optional modulated update (blueprint §3.2): scale grads by `1/(1+λ)`. The recurring
mistake is to think "the node is weights+bias" and treat λ as absent — it is the
*defining* variable. bias `b` and axonal-gain `routing_scale` are real params but NOT
the namesake three.

**Version status (the regression):**
- **v1.1 / legacy** (`trioron/legacy/network.py`): full `(w, λ, u)` with `fisher_W`,
  `lam`, `W_anchor`, `ewc_penalty()`, `update_fisher_all()`.
- **v2.0 as shipped** had **DROPPED λ** (spec §8.6: "v2 does not maintain per-weight
  importance estimates"), relying on cell-level credit-locking alone. **That was the root
  of the catastrophic forgetting we kept fighting** — Rocky's `epigenetic_lock_hypothesis`
  (lam=0 → unanchored drift → bias to last task).
- **v2.0 NOW (RESTORED, commit `4e93099`, session 014):** the triparametric node is whole
  again — `arena.node_lambda` + the `epigenetic_lock` module (API above). λ is live and the
  EWC pull bites. **Session 015 correction:** its driver is `|w·g|` saliency, not Fisher
  (Fisher washes out at convergence). Spec §8.6 was fixed to match (session 015).
  **World verdict (n=5, `consolidate_base.py`):** hard credit-locking (FULL-LOCK) still
  wins raw retention + new-skill acquisition; soft λ's distinct value is that it lets the
  shared base keep *improving* the consolidated skill while learning the next (a freeze
  holds it static). Driver ranking confirmed `|w·g|` > Fisher > reward. λ is a
  **complement** to credit-locking, not a replacement.

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
once per expressed gene and **adds** contributions. The six shipped phenotypes are
**linear / attention / conv / recurrent / dendrite / tanh** (spec §3.2–3.6, §3.10;
tanh added s030 — bounded |y|≤1, the anti-detonation counterpart to the quad).
Marker genes (not expression): PERCEPTION, OUTPUT, …, MIRROR (10, experimental),
**RECEPTOR (11, s030)** — PCLL phase injection, and **PCLL itself is spec §10**:
the gradient-free streaming layer (lock-in + matched filter + signatures) living
in `trioron/pcll/`, wired via `PCLLController` / `substrate.attach_pcll`.

**PCLL grew up in s032–s033 (spec §10.6–10.10, all gated M1–M5):** the
deployment regime is the MIXED STREAM (`MixedStreamController`, §10.7 —
genesis births ONE world-class; division-by-frustration IS class
discovery; canonical EXTREMES frame; self-arrest = rate collapse);
settlement is PER-CLASS (§10.6 [D13] — a growth settles only on its own
class's testimony; no-testimony DEFERS); the **composer arm grows REAL
tissue** (§10.8 [D14/D16/D17] — genome-only candidates, the dendrite
computes c_i²+c_j² EXACTLY via edge w = −1/(2π) and σ(z)=z+z²;
residual-incoherence ratio trials; FAMILY trials over lineage-sibling
pools, degree 3; future-deposit settlement; spawn = +1 computing cell,
+2 edges, EARNED rank); the **manifold adapter** (§10.9 [D15]) rides
class astrocyte rows (annealing / σ-readout / replay / full ship-wake).
The census (§10.10 [D11]) is standing instrumentation: growth must MOVE
cell/edge/rank counts or it's bookkeeping.

> **Experiment note:** the embodied-organism arc added `MIRROR = 10` as a marker gene
> (`trioron/core/epigenome.py`) — mirror cells dispatch as LINEAR; "mirror" is
> connectivity + localized credit, not a new forward op. This is an experiment
> extension into the reserved range, not part of the base spec.

### 2.5 Phenotype IS chosen from the problem at the growth event (DO NOT FORGET)

**The recurring mistake (made again session 015): reading "phenotype is never optimized
by SGD" as "phenotype is never chosen from the problem."** Both are true and they are not
the same thing. Phenotype is not a differentiable parameter — but the **lifecycle selects
the child's gene at the division event, conditioned on the problem.** The substrate
*does* adapt its architecture to the task; this is the `substrate_self_organizes_architecture`
design intent, and it is **validated**, not aspirational:

- **Selective quad growth** (`experiments/selective_quad_growth.py`, memory
  `selective-quad-growth`, n=3): under *relational frustration* (loss stuck high AND
  linear-width growth stopped helping), a dividing cell's child takes the **quad/dendrite**
  phenotype. On the relational SAME-DIFFERENT task it grows **~8 quad cells → 1.000**; on a
  **linearly-separable** task the *same* substrate grows **ZERO quad cells** (1.000) because
  frustration resolves with linear width. The nonlinear primitive is spent **only where the
  problem demands it**. (Linear-growth control: 0.507 — width alone can't do relational.)
- The selection signal is **emergent frustration**, not a hand-set gene. This is the
  general mechanism: the same divide-time logic could pick `conv` / `attention` / `recurrent`
  given the right frustration discriminator — the unbuilt generalization, not an unbuilt
  capability.

**Known hard limit (don't over-claim):** the substrate is a **flat sparse edge list with no
spatial locality** (`substrate_no_spatial_memory`), so `conv`-by-emergence is *closed* as a
null result (`conv_by_emergence_null_result` — use a cortex upstream for vision). Selective
**quad/dendrite** growth works; selective **conv** does not, on the bare substrate.

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
| **Axis 4** | `set_axonal_gain` | per-cell output gain (`routing_scale`) | via shim (a real param, but NOT one of the namesake (w, λ, u) three — see §1) |
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
6. **Epigenetic lock λ** (§4.6, the namesake, RESTORED session 014) —
   `learning/epigenetic_lock.py`. The **soft** per-node anchor: drive λ from `|w·g|`
   saliency (`accumulate_saliency`, native) or reward/env (`set_lambda`), `anchor` at a
   boundary, add `strength·ewc_penalty(arena)` to the loss. Complements (does not replace)
   the **hard** credit-based dormant locking. Fisher-`g²` EWC remains available but is the
   washout-prone outlier — see §1.

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

- ❌ "A node is weights + bias." → ✅ **Triparametric: (w, λ, u)** — weight, **epigenetic
  lock λ**, utility. λ is the namesake; v2 dropped it, **session 014 RESTORED it** (§1).
  (Axonal-gain `routing_scale` and bias are real but NOT the three.)
- ❌ "λ is filled with Fisher / a valueless buffer." → ✅ **λ holds an importance signal,
  and the native one is `|w·g|` saliency, not Fisher `g²`** (Fisher washes out at
  convergence — §1). That importance IS its intrinsic value.
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
- ❌ "Growth ran but accuracy didn't move — the mechanism must be wrong." →
  ✅ **Check the dispatch table first.** `construct()` once defaulted to an
  EMPTY dispatch — every grown computing cell silently skipped in forward
  (s020's inert-growth bug, refound s033 by the first composer spawn). It
  now defaults to the real phenotype table (spec §10.10); a custom
  `dispatch_table={}` reintroduces the trap.
- ❌ "σ/likelihood weighting will close the readout gap to Bayes." →
  ✅ FALSIFIED on the post-division organism (s033, M5 gate): fragments
  are tight, diagonal σ gives +0.012, full-cov is noise. The residual gap
  is fragmentation/mapping ambiguity — ROUTING (merge consumer / routing
  layer), not readout math.
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
  **Learning:** `trioron/learning/` (credit, manifold, dream, frustration, rejuvenate,
  **epigenetic_lock** — the restored λ).
  **Lifecycle:** `trioron/lifecycle/` (grow, graft, ship, wake, compact, saliency).
  **Legacy v1.1:** `trioron/legacy/` (triparametric node, EWC, axes API).
- **Chained-15 routing to port:** `experiments/bench_chained_15_v2.py` (H-routing,
  `--full-cov`, `--perc-mixture-k`).
- **`docs/handoff/HANDOFF.md`** — current session state (rewritten every session).
- **Related project:** `~/project-aidos/` vendors the trioron substrate (separate memory dir).
