# Trioron 2.0 — Edge-Level Plasticity

**Author:** Marcelinus R Hatorangan
**Advisory contributions:** Gemma (academic), Chloe (engineering)
**Status:** Draft v0.1 — 2026-05-19
**Supersedes:** the pneuma-side draft `long_range_synapses.md` (2026-05-19) for the trioron-substrate scope.

---

## 1. Framing

Trioron 1.0 was about **nodes**: `grow_node`, `prune_node`, `archive_row`, per-node plasticity gates (λ), per-node utility (u), apoptosis dynamics. The neuron-equivalent was the unit of plasticity.

Trioron 2.0 is about **edges** — the synapse equivalent. The substrate gains:

1. **Long-range reach.** A destination can read from any earlier node, not only its immediate predecessor.
2. **Plastic fanout sparsity.** A source's outgoing connections become a plastic set that can grow and prune per branch.
3. **Depth growth.** New micro-layers can be inserted between existing layers (not Net2Net; localized-growth-direction init).
4. **Per-source modulatory trunk.** A slow-time-scale axonal gain on each source, the substrate-level analog of attention / emotional / neuromodulatory state.

Each of these lands as a small additive change against the existing `TrioronLayer` / `TrioronNetwork` API. The unified rule is: **edge becomes the unit of plasticity; node remains the unit of state.**

## 2. Design Principles

Inherited from the 1.0 blueprint, extended:

- **Justify every edge.** A new edge (long-range column or inserted-layer slot) is initialized from a localized signal — the direction of the substrate's current unfit residual — never random when the signal is computable.
- **Sever cleanly, freeze first.** Edge apoptosis follows the row-archive pattern: lock at anchor, mask gradient, zero Fisher. Hard deletion is the exception, not the default.
- **Bounded fanout per node.** Each destination caps incoming-edge count (K_in); each source caps outgoing-edge count (K_out). No unbounded fanout in the default operating regime.
- **Additive rollout.** Every shipped donor and existing arc continues to work byte-for-byte at the sequential default. 2.0 features are opt-in.

## 3. The Four Axes and Their API Tweaks

### 3.1 Axis 1 — Long-range reach

**Existing:** `grow_input` already adds an input column with full per-column state (W, W_anchor, fisher_W). The mechanism for "another edge into this destination" is in place. What's missing is *which source the column reads from*.

**Tweak:**
- New buffer per layer: `input_sources: list[(layer_idx, node_idx)]`. Default at layer i = `[(i-1, j) for j in range(fan_in)]` matching current sequential behavior.
- `TrioronNetwork.forward` walks a sources registry: for each layer, gather per-column input by looking up `input_sources[col]` in the registry of layer outputs computed so far.
- `grow_input` gains an optional `source: (int, int)` argument that records the new column's provenance. Default = the immediate predecessor's most-recently-added node, preserving 1.0 behavior.

Constraint: `src_layer < dst_layer` (no cycles in the DAG).

### 3.2 Axis 2 — Plastic fanout / branch sparsity

**Existing:** `archived: bool[n_nodes]` + `mask_archived_grads` already implement destination-side (row-level) archival cleanly.

**Tweak (column-side mirror):**
- New buffer per layer: `input_archived: bool[fan_in]`.
- New method: `archive_input(col_idx)` — sets the flag, zeros `fisher_W[:, col_idx]`, snaps `W[:, col_idx]` to `W_anchor[:, col_idx]`.
- New method: `mask_archived_input_grads()` — zeros `W.grad[:, archived_cols]` after backward, before step.
- Network-level helper: when source node `(i, j)` is archived via `archive_row`, traverse `input_sources` across all layers and call `archive_input` on every column tagged with that source.

`prune_input` remains available for hard deletion when an edge is permanently gone (e.g., source layer is structurally removed). Archive is the soft default.

### 3.3 Axis 3 — Depth growth via `insert_layer`

**Existing:** `TrioronLayer` constructor + `grow_node` already handle "small layer with K nodes, each with its own activation." `grow_input` already wires inputs.

**Tweak — one new network-level method:**

```
TrioronNetwork.insert_layer(
    between: tuple[int, int],         # (i, j) where j == i+1 in current topology
    n_nodes: int,
    activation: str = "relu",
    init_mode: str = "growth_direction",
) -> int                              # returns the new layer's index
```

- Constructs a new `TrioronLayer` between layers i and j.
- Re-wires `input_sources` of layer j: columns that pointed at layer i now point at the new layer.
- `init_mode="growth_direction"` initializes the new layer from the top-K right singular vectors of the residual at the insertion point (a K-dim generalization of `compute_growth_direction`).
- `init_mode="identity"` is available for Net2Net-style identity-preserving insertion. Not the default.

The depth claim from §4 of the original proposal is satisfied via this primitive: an inserted micro-layer is a genuine new sequential transformation stage.

**Insertion trigger.** `grow_node` fires on (loss plateau + rank saturation + grad stable). `insert_layer` fires as the second-tier escalation: recent `grow_node` events at layer N followed by sustained loss plateau over the next window. Try widening first; deepen only when widening fails.

**Cap policy.** Default `K_insert = 2–3` insertions per original-pair slot. Unbounded mode behind a flag for research.

### 3.4 Axis 4 — Axonal trunk gain

**Existing:** `routing_scale: torch.ones(n_nodes)` is the *destination-side* per-node multiplicative gain in the current `forward`: `W_eff = W * routing_scale.unsqueeze(1)`.

**Tweak (source-side mirror):**
- New buffer per layer: `axonal_gain: torch.ones(n_nodes)`.
- New buffer per layer: `axonal_gain_anchor: torch.ones(n_nodes)` (for the triparametric anchored-state contract).
- In `TrioronNetwork.forward`'s gather step, when fetching `y[src_layer][src_node]` as input to a destination column, multiply by `axonal_gain[src_layer][src_node]`.
- New method `set_axonal_gain(signal, mode)` mirrors `set_lambda`: write the gain from any external signal — reward magnitude, attention mask, emotional-tag node output, manual prior.

Slow time-scale plasticity (no per-step decay; updated at consolidation cadence or set externally). Default 1.0 across the board, preserving 1.0 behavior.

Biological framing: the substrate-level analog of emotional state, attentional focus, and neuromodulator broadcasts — a small set of source nodes can scale their entire downstream influence without changing any edge weight.

## 4. Growth Signal Reuse

Trioron 1.0 already computes the localized growth direction via `compute_growth_direction` (residual SVD over a contrastive pair). Several callsites bypass this and use random init via `init_vec=None` — `classification.py:95`, `experiments/probe_injection_mechanism.py:82`, `experiments/probe_engram_diversity.py:99`, `grow_input`'s default zero-column path.

2.0 closes the gap with a **non-contrastive generalization** of the same SVD machinery:

1. **Per-class scatter top-eigenvector** at the growth point (when labels are available — supervised heads).
2. **Gradient top-SVD** of the layer's pre-activation gradient matrix across the recent batch (when no labels — unsupervised heads).

Both fall back to `compute_growth_direction` exactly when classes = `{a, b}` and the input is a contrastive pair. They slot into the existing `init_vec` parameter — no new structural API.

This is independent of trioron 2.0's edge work and is a free win on its own.

## 5. Compatibility Surface

### 5.1 State-dict schema bump

New buffers (`input_sources`, `input_archived`, `axonal_gain`, `axonal_gain_anchor`) are appended to `state_dict`. v1 donor load path:

- `input_sources` defaults to sequential
- `input_archived` defaults to all-`False`
- `axonal_gain` and `axonal_gain_anchor` default to all-`1.0`

Every shipped donor must round-trip identically: HF Space tabs, `vocabulary.pt` (8 Pong primitives), EMNIST K-T population, BTM baseline, manifold-grown chained-15. Phase 4 in §6 verifies this.

### 5.2 R·S handshake (donor absorption)

The current absorption handshake (`composition/translator.py`, `composition/subspace.py`) factorizes a dense `W_L0` across donors. With long-range edges added to L0, donors will diverge in column count and the factorization breaks.

**Fix:** the handshake operates on the **standardized subset** of columns — those whose `input_sources` matches the sequential default `[(-1, j)]`. Long-range columns are excluded from cross-donor handshake and treated as branch-private extension. Existing 1.0 donors trivially satisfy "all columns standardized" and absorption is unaffected.

### 5.3 Existing arcs that touch dense W directly

Audit list (no immediate rework required if they stay at sequential default):

- `multibranch.py` — Branch model and absorption
- `dreaming.py` — synaptic downscale operates on `W` rows
- `pruner.py` — `max(act_grad, act_var)` over `W` rows
- `senses/organism.py` — SensoryOrganism A/B/C builds dense MLPs
- `api.extend`, `api.absorb`, `load_organism` — state-dict round-trip

## 6. Phase Sequencing

### Phase 1 — Foundational tweaks (~1 week)
Land all four axes' buffers + `archive_input` + multi-source `forward`, all defaulting to current behavior. **Acceptance: every existing test passes unchanged.**

### Phase 2 — `insert_layer` (~1 week)
Implement insertion with localized-growth-direction init and the K_insert cap. Acceptance test: insert with near-zero noise on a toy task; pre/post-insertion forward agree at the limit.

### Phase 3 — Non-contrastive growth direction (~3 days, parallel to Phase 2)
Generalize `compute_growth_direction` (per-class scatter / gradient top-SVD). Backfill `init_vec=None` callsites. Independent free win.

### Phase 4 — State-dict bump + back-compat (~3 days)
Schema bump with v1 fallback path. Load-test every shipped donor.

### Phase 5 — R·S handshake migration (~3–5 days)
Restrict handshake to the standardized column subset. Absorption regression at parity with current 1.0 numbers.

### Phase 6 — Empirical validation
- **Trioron-side parity:** chained-15, manifold replay, dream archive, extension bench — all at sequential default, must not regress.
- **Trioron-side delta:** new bench with long-range edges enabled.
- **Pneuma-side:** hand off; pneuma adopts the substrate for transformer-FFN architecture-emergence experiments.

## 7. Out of Scope (for v1)

Explicitly NOT in trioron 2.0 v1:

- **Per-edge nonlinearity** as a primitive (vector-valued `W[i,j,:]` with gated AMPA/NMDA-style modes). Depth is delivered via `insert_layer` instead.
- **Recurrence / cycles.** `input_sources` is constrained to `src_layer < dst_layer`.
- **Multi-output sinks.** The brain analogy ("talk while playing while angry") motivates typed output sinks; v1 stays single-output and defers this to a future tweak (`output_sinks` registry on `TrioronNetwork`). Backward-compatible when added.
- **Cross-attention / per-token routing.** Belongs to the senses/organism layer, not the substrate.
- **Spiking / time-varying dynamics.** `axonal_gain` is the slow modulatory analog; faster dynamics defer.
- **Sparse matmul acceleration.** Archived columns continue to participate in the dense matmul (× ~0 weight). Optional follow-up if archive density becomes high in practice.

## 8. Risks and Open Questions

1. **Shipped-donor zoo migration.** Every existing donor must load and forward identically at sequential default. Mechanical but real — Phase 4 must enumerate.

2. **Pruner interaction.** Pruner already deviates from blueprint §3.2 (uses `max(act_grad, act_var)`, not strict `|a·g|`). Long-range columns may have different gradient/activation statistics; thresholds may need recalibration. Smoke test in Phase 1.

3. **Lifetime growth budget.** The 70–80 yr deployment horizon was budgeted for width growth only. Adding `insert_layer` with `K_insert ≥ 2` multiplies the substrate's growth envelope. Caps and apoptosis cadence need lifetime-scale sizing, not paper-task scales. Back-of-envelope budget exercise required before Phase 6.

4. **Credit assignment through sparse long-range edges.** Autograd handles the mechanics; whether the gradient signal through a sparse edge meaningfully updates the source remains empirical. `axonal_gain` is a partial mitigation — high gain on important sources amplifies their gradient contribution.

5. **Multi-sink arbitration (deferred).** If/when output sinks are added, "anger inhibits speech" needs a mechanism — either trunk-gain cross-modulation between sinks, or explicit inhibitory edges with negative diameter. Pick at the time, not now.

## 9. Recommendation

**Additive rollout.** Phases 1–4 ship 2.0 as strictly opt-in. Existing arcs continue at sequential default forever — long-range edges, `insert_layer`, and `axonal_gain ≠ 1.0` are only used by callers that ask. Net-zero regression risk for shipped paper benches and HF Space. Pneuma is the first opt-in customer; trioron's own benches opt in only after the substrate's parity case is established.

---

*This document records the trioron-substrate scope of long-range-synapse work. The pneuma-side proposal (`~/pneuma/docs/proposals/long_range_synapses.md`) covers the same proposal as it concerns pneuma's transformer-FFN integration; once 2.0 lands here, the pneuma-side draft should be reduced to a pointer at this file plus pneuma-specific integration notes.*
