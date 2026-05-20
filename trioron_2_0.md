# Trioron 2.0 — Edge-Level Plasticity and Dendritic Compartmentalization

**Author:** Marcelinus R Hatorangan
**Advisory contributions:** Gemma (academic), Chloe (engineering)
**Status:** Draft v0.2 — 2026-05-20
**Supersedes:** the pneuma-side draft `long_range_synapses.md` (2026-05-19) for the trioron-substrate scope.

---

## 1. Framing

Trioron 1.0 was about **nodes**: `grow_node`, `prune_node`, `archive_row`, per-node plasticity gates (λ), per-node utility (u), apoptosis dynamics. The neuron-equivalent was a point neuron — one weighted sum, one activation, one output.

Trioron 2.0 refines the point neuron in two directions: **between cells** (edges, the synapse equivalent) and **within cells** (dendrites). The substrate gains:

1. **Long-range reach.** A destination can read from any earlier node, not only its immediate predecessor.
2. **Plastic fanout sparsity.** A source's outgoing connections become a plastic set that can grow and prune per branch.
3. **Depth growth.** New micro-layers can be inserted between existing layers (not Net2Net; localized-growth-direction init).
4. **Per-source modulatory trunk.** A slow-time-scale axonal gain on each source, the substrate-level analog of attention / emotional / neuromodulatory state.
5. **Dendritic compartmentalization.** Each cell's incoming columns partition into branches with branch-local nonlinearity; per-cell reasoning depth grows and prunes via a within-niche frustration signal distinct from the population-level signal that drives `grow_node`.

Each of these lands as a small additive change against the existing `TrioronLayer` / `TrioronNetwork` API. The unified rule for 1.0→2.0: **the point neuron refines into a structured cell** — gaining edges between cells (Axes 1–4) and a dendritic tree within the cell (Axis 5), while the cell remains the unit of state.

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

### 3.5 Axis 5 — Dendritic compartmentalization

**Existing:** Each cell in `TrioronLayer` is a point neuron. `forward` computes `z = F.linear(x, W, b)` — one weighted sum across all fan_in columns — then one activation. The cell has no internal structure; its expressive capacity is whatever a single linear function followed by an elementwise nonlinearity can give.

**Motivation:** Poirazi & Mel (2003) — a pyramidal neuron with a branched dendritic tree is functionally a two-layer neural network: each branch performs a local nonlinear sum, the soma pools the branches. Goriounova et al. (2018) and Mohan et al. (2015) — dendritic length and branching in human cortical layer 2/3 correlate with IQ. Computational depth *per cell* — not just population count — is load-bearing for reasoning. 1.0's point neuron leaves this axis entirely on the table.

**Tweak:** Each cell gains an internal *dendritic tree*. Two layers of dendritic structure:

- **Island** — the set of upstream cells feeding the dendrite. Already captured by Axis 1's `input_sources`. Fixed at birth (inherited from parent on `grow_node`, with optional ε perturbation). This is the cell's **niche**.
- **Internal tree** — how those island inputs partition into branches, each with its own local nonlinearity, pooled at the soma. **This** is what grows and prunes during the cell's life. This is the cell's **reasoning capacity**.

**New buffers per layer:**

- `branch_id: long[n_nodes, fan_in]` — for cell `i`, column `j` is assigned to branch `branch_id[i, j] ∈ [0, B_per_node[i])`. Initialized all-zero (every column on branch 0 → single-branch point neuron, byte-identical to 1.0).
- `branch_weight: float[n_nodes, B_max]` — soma-side pooling weight per branch. Initialized `[1.0, 0.0, …, 0.0]` per cell. Plastic via gradient like `W`. Participates in `ewc_penalty()` alongside `W` and `b`: the loss includes a `(branch_weight - branch_weight_anchor)² · fisher_branch_weight` term, scaled by the same per-cell λ. Has an `_anchor` mirror and a `fisher_branch_weight` mirror for the triparametric anchored-state contract. `update_fisher()` accumulates squared gradient on `branch_weight` after every backward, mirroring its treatment of `W`.
- `B_per_node: long[n_nodes]` — current branch count per cell. Initialized 1 throughout. Increments on `grow_branch`, decrements on `prune_branch`.
- `internal_stress: float[n_nodes]` — EMA of `|∂L/∂y_i| · 1(y_i > 0)`. Per-cell within-niche frustration signal (separate from population-level frustration in `frustration.py`).
- `branch_utility: float[n_nodes, B_max]` — EMA of `|branch_weight[i,b] · y_{i,b}|`. Mozer/Smolensky saliency at branch granularity.

**Layer-level config:**

- `branch_activation: str = "quad"` — local nonlinearity at each branch (post-branch-sum, pre-pool). Default `quad` (x²) matches NMDA-compartment-style supralinear summation in the Poirazi/Mel result; this is the **live default** for newly-constructed layers. `sigmoid`, `tanh`, and `identity` are also available. v1-loaded layers default to `identity` (see §5.1) so existing donors functionally behave as point neurons even if they later grow branches.
- `B_max: int = 8` — per-cell branch budget cap. Lifetime envelope; recalibrated alongside `insert_layer`'s lifetime budget (see §8 Risks).

**Forward (two-stage):**

For each cell `i` with `x_eff[j] = axonal_gain[src(j)] · x[j]` (Axis 4):

```
if B_per_node[i] == 1:                                       # K=1 fast path
    y_i = σ_soma( Σ_j W[i,j] · x_eff[j]  +  b_i ) · routing_scale[i]
else:                                                        # K>1 dendritic path
    for b in 0..B_per_node[i]:
        z_{i,b} = Σ_{j: branch_id[i,j] == b}  W[i,j] · x_eff[j]
        y_{i,b} = σ_branch(z_{i,b})
    y_i = σ_soma( Σ_b branch_weight[i, b] · y_{i,b}  +  b_i ) · routing_scale[i]
```

`σ_soma` is the existing `activation`. `σ_branch` is `branch_activation`. **The K=1 fast path bypasses σ_branch entirely** — a cell with a single branch has nothing to compartmentalize, and the branch-local nonlinearity collapses to a redundant elementwise transform inside σ_soma's argument. This rule preserves byte-identical 1.0 forward at K=1 *regardless of the layer's `branch_activation` setting*. σ_branch first enters a cell's life the moment `grow_branch` raises it to K=2.

Consequence: "live by default" means `branch_activation="quad"` is active and ready to fire — but the actual forward only changes once a cell's internal stress crosses the `grow_branch` threshold and the cell jumps from K=1 to K=2. Fresh substrates ship with the dendritic machinery armed; v1-loaded donors ship with it functionally disarmed (σ_branch=identity makes even K≥2 fall back to a linear pool, equivalent to extra DOF on a flat W).

**Two stress signals — population vs. cell:**

The substrate now distinguishes:

- **Overall stress** — drives population-level plasticity (`grow_node`, full cell apoptosis). Unchanged from 1.0: loss plateau + no specialist active on the failing input.
- **Internal stress (per cell)** — drives intra-cellular plasticity (`grow_branch`, `prune_branch`). A cell with high `internal_stress[i]` is *engaged AND failing* — in the right niche but lacking depth to discriminate within it. This is a strictly within-niche signal; cell-level apoptosis does not consult it.

  Formally:
  ```
  internal_stress[i] = EMA( |∂L/∂y_i| · engaged(y_i) )
  engaged(y) =
      1(y > 0)             if σ_soma is ReLU
      1(|y| > ε_engage)    otherwise (default ε_engage = 0.05)
  ```
  The engagement gate is activation-specific because "this cell is actively firing" means different things across activations: positive output for ReLU, output away from zero for tanh/identity, output away from the saturating midpoint for sigmoid (the `|y| > ε_engage` rule handles all three cases for non-ReLU activations).

These never share thresholds and never interfere: a cell with high overall stress (no specialist) and a cell with high internal stress (specialist trying but failing) are answered by different plasticity events.

**Per-branch utility, pruning, and orphaning:**

`branch_utility[i, b]` mirrors `saliency_utility()` at branch granularity. A branch whose utility stays below floor across the window is a `prune_branch` candidate. **Pruning orphans the branch's columns** — they are not redistributed to surviving branches. The cell stops reading them (their entry in this row's `input_archived` is set); other cells may still read the corresponding sources. This matches the "dendrite as identity" framing: a retracted branch removes that input pathway from this neuron's life, but the source itself remains in the substrate.

A cell whose branches all prune away has effectively empty internal structure; full cell apoptosis fires as the natural terminus — `prune_branch` does not handle the last-branch case (it refuses if `B_per_node[i] == 1`).

**New methods:**

- `grow_branch(node_idx: int, source_cols: list[int]) -> int`
  Adds a new branch on cell `node_idx`, reassigning `source_cols` to it (peeled from their existing branches). New branch weight initialized to `0.1 · mean(branch_weight[node_idx, :B_per_node[node_idx]])`. Returns the new branch index. Refuses if `B_per_node[node_idx] == B_max`. **Trigger:** `internal_stress[i]` above threshold across a window, AND the cell's overall-stress contribution is low (otherwise `grow_node` handles it).

- `prune_branch(node_idx: int, branch_idx: int) -> None`
  Soft-archives a branch. Ramps `branch_weight[node_idx, branch_idx]` toward zero; on confirmation (utility stays at floor across the window) the columns assigned to it are orphaned via `input_archived` on this row, branch count decrements. **Trigger:** `branch_utility[i, b]` below floor across the window. Refuses if `B_per_node[i] == 1`.

- `inherit_dendrite(parent_idx: int, child_idx: int, perturb_frac: float = 0.05) -> None`
  Called automatically inside `grow_node` when a `parent_idx` is supplied. Copies `branch_id[parent_idx, :]` to `branch_id[child_idx, :]` with `perturb_frac` of columns randomly reassigned to other existing branches (the one-shot structural ε). Copies `branch_weight` and `B_per_node` accordingly. The child is born a *sister specialist*, not a blank slate.

**Parent selection at `grow_node`:** the existing cell with the highest activation on the frustrated input. Falls back to highest output similarity if no clear winner. `grow_node` gains an optional `parent_idx: int | None` argument; default `None` preserves 1.0 behavior (no inheritance, blank-slate child).

**Out of scope for v1 of Axis 5:** per-branch nonlinearity choice (one `branch_activation` per layer, not per branch); branch merging (only grow + prune); dendritic-tree depth > 2 (no sub-branches of branches — flat partition only).

## 4. Growth Signal Reuse

Trioron 1.0 already computes the localized growth direction via `compute_growth_direction` (residual SVD over a contrastive pair). Several callsites bypass this and use random init via `init_vec=None` — `classification.py:95`, `experiments/probe_injection_mechanism.py:82`, `experiments/probe_engram_diversity.py:99`, `grow_input`'s default zero-column path.

2.0 closes the gap with a **non-contrastive generalization** of the same SVD machinery:

1. **Per-class scatter top-eigenvector** at the growth point (when labels are available — supervised heads).
2. **Gradient top-SVD** of the layer's pre-activation gradient matrix across the recent batch (when no labels — unsupervised heads).

Both fall back to `compute_growth_direction` exactly when classes = `{a, b}` and the input is a contrastive pair. They slot into the existing `init_vec` parameter — no new structural API.

This is independent of trioron 2.0's edge work and is a free win on its own.

## 5. Compatibility Surface

### 5.1 State-dict schema bump

New buffers (`input_sources`, `input_archived`, `axonal_gain`, `axonal_gain_anchor`, and the Axis 5 dendrite buffers `branch_id`, `branch_weight`, `branch_weight_anchor`, `fisher_branch_weight`, `B_per_node`, `internal_stress`, `branch_utility`) are appended to `state_dict`. v1 donor load path:

- `input_sources` defaults to sequential
- `input_archived` defaults to all-`False`
- `axonal_gain` and `axonal_gain_anchor` default to all-`1.0`
- `branch_id` defaults to all-zero (every column on branch 0)
- `branch_weight` defaults to `[1.0, 0.0, …, 0.0]` per cell; `branch_weight_anchor` mirrors it; `fisher_branch_weight` all-zero
- `B_per_node` defaults to all-`1`
- `internal_stress` and `branch_utility` default to all-zero
- `branch_activation` (layer config, not a buffer) defaults to `identity` **only when loading a v1 state-dict** that has no recorded `branch_activation` — preserves exact 1.0 forward for any v1 donor even if it later grows branches. Newly-constructed layers default to `quad` (live by default, per §3.5).

Every shipped donor must round-trip identically: HF Space tabs, `vocabulary.pt` (8 Pong primitives), EMNIST K-T population, BTM baseline, manifold-grown chained-15. Phase 4 in §6 verifies this.

### 5.2 R·S handshake (donor absorption)

The current absorption handshake (`composition/translator.py`, `composition/subspace.py`) factorizes a dense `W_L0` across donors. With long-range edges added to L0, donors will diverge in column count and the factorization breaks.

**Fix:** the handshake operates on the **standardized subset** of columns — those whose `input_sources` matches the sequential default `[(-1, j)]`. Long-range columns are excluded from cross-donor handshake and treated as branch-private extension. Existing 1.0 donors trivially satisfy "all columns standardized" and absorption is unaffected.

**Dendritic state on absorption (Axis 5):** R·S factorizes `W`'s column space and does not depend on `branch_id`. A donor's dendritic state (`branch_id`, `branch_weight`, `B_per_node`) is **reset to single-branch point-neuron form on absorption** — the absorbed substrate joins the host's dendritic regime at K=1 and re-grows branches post-absorption under the host's internal-stress signals. This avoids cross-donor branch-id conflicts and keeps the handshake purely about W. Pre-absorption donor dendritic structure is therefore not portable; it is a *post-training* artifact of each donor, not a transferable identity.

### 5.3 Existing arcs that touch dense W directly

Audit list (no immediate rework required if they stay at sequential default):

- `multibranch.py` — Branch model and absorption
- `dreaming.py` — synaptic downscale operates on `W` rows
- `pruner.py` — `max(act_grad, act_var)` over `W` rows
- `senses/organism.py` — SensoryOrganism A/B/C builds dense MLPs
- `api.extend`, `api.absorb`, `load_organism` — state-dict round-trip

## 6. Phase Sequencing

### Phase 1 — Foundational tweaks (~1 week)
Land all four edge axes' buffers + `archive_input` + multi-source `forward`, all defaulting to current behavior. **Acceptance: every existing test passes unchanged.**

### Phase 1.5 — Dendritic buffers + two-stage forward (~3–4 days)
Add Axis 5 buffers (`branch_id`, `branch_weight`, `branch_weight_anchor`, `fisher_branch_weight`, `B_per_node`, `internal_stress`, `branch_utility`) and the branch-aware forward. Newly-constructed layers default to `branch_activation="quad"` (live by default); v1-loaded layers default to `branch_activation="identity"`. `B_per_node` initializes to 1 everywhere, so the K=1 fast path (`F.linear`-based, σ_branch bypassed) is the only one that executes until Phase 2.5 lands `grow_branch`. **Acceptance: every existing test passes unchanged byte-for-byte at K=1, regardless of layer `branch_activation` setting (the K=1 fast path bypasses σ_branch).** A separate K≥2 forward-equivalence test (manually constructed K=2 cell with σ_branch=identity matching a K=1 cell with the corresponding partitioned columns merged) validates the K>1 path's arithmetic before Phase 2.5 starts using it.

### Phase 2 — `insert_layer` (~1 week)
Implement insertion with localized-growth-direction init and the K_insert cap. Acceptance test: insert with near-zero noise on a toy task; pre/post-insertion forward agree at the limit.

### Phase 2.5 — Dendritic growth + pruning events (~4–5 days)
Implement `grow_branch`, `prune_branch`, `inherit_dendrite`, the `internal_stress` EMA update, the `branch_utility` EMA update, and the within-niche frustration trigger. `grow_node` gains the optional `parent_idx` parameter and the parent-selection helper. Acceptance: a within-niche fine-discrimination toy task (one specialist active, two close subclasses) where K=1 underfits and K=2 separates. Branch pruning + orphaning round-trip cleanly when forced.

### Phase 3 — Non-contrastive growth direction (~3 days, parallel to Phase 2)
Generalize `compute_growth_direction` (per-class scatter / gradient top-SVD). Backfill `init_vec=None` callsites. Independent free win.

### Phase 4 — State-dict bump + back-compat (~3 days)
Schema bump with v1 fallback path covering both edge and dendrite buffers. Load-test every shipped donor. `branch_activation` config defaults to `identity` for v1-loaded layers.

### Phase 5 — R·S handshake migration (~3–5 days)
Restrict handshake to the standardized column subset. Dendritic state reset to K=1 on absorption (§5.2). Absorption regression at parity with current 1.0 numbers.

### Phase 6 — Empirical validation
- **Trioron-side parity:** chained-15, manifold replay, dream archive, extension bench — all at sequential default with `branch_activation="identity"`, must not regress.
- **Trioron-side edge delta:** new bench with long-range edges enabled.
- **Trioron-side dendrite delta:** new bench on a within-niche fine-discrimination curriculum where the K=1 substrate plateaus and dendritic growth lifts it. The discriminating prediction: same cell count, more capacity per cell, better fine-grained accuracy. If this delta fails to show, Axis 5 is biologically motivated but empirically inert and should ship dormant.
- **Pneuma-side:** hand off; pneuma adopts the substrate for transformer-FFN architecture-emergence experiments.

## 7. Out of Scope (for v1)

Explicitly NOT in trioron 2.0 v1:

- **Per-edge nonlinearity** as a primitive (vector-valued `W[i,j,:]` with gated AMPA/NMDA-style modes). Within-cell depth is delivered via Axis 5 (dendritic compartmentalization) and between-cell depth via `insert_layer`. A single edge stays a scalar.
- **Per-branch nonlinearity choice.** Axis 5 ships one `branch_activation` per layer, not per branch or per cell. Per-branch nonlinearity is a follow-up if dendritic depth proves load-bearing.
- **Dendritic sub-branches.** Axis 5 ships a flat column-to-branch partition; branches do not themselves split into sub-branches. Hierarchical dendritic trees defer to v2.
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

6. **Dendritic budget at lifetime scale (Axis 5).** `B_max=8` per cell combines multiplicatively with `n_nodes` and `insert_layer`'s K_insert. A 70–80 yr deployment that grows cells, layers, *and* branches has a parameter envelope substantially larger than the current width-only growth budget. Phase 6 must include a back-of-envelope sizing exercise before dendrites are turned on outside paper benches.

7. **Parent-selection stability at `grow_node` (Axis 5).** "Highest activation on the frustrated input" is well-defined for a single failing input but ambiguous when the trigger fires from a sustained window over many inputs. The implementation must commit to a deterministic aggregation (highest mean activation across the window, or highest activation on the single worst-loss input). Wrong choice produces unstable lineages and noisy branch inheritance.

8. **Two-stress decoupling under shared optimizer state (Axis 5).** Overall stress and internal stress are conceptually orthogonal, but they share Adam state through `W` and `branch_weight` co-updates. A cell whose internal stress is high may also drag the population-level frustration metric up, double-counting the same failure. The window-based triggers and the population-vs-cell granularity of the signals are the first-line defense; if they leak, an explicit gating step (suppress one signal when the other is firing) is the fallback.

9. **K=1 → K=2 forward discontinuity (Axis 5).** With `branch_activation="quad"` (the live default), `grow_branch` flips a cell from `σ_soma(W·x + b)` to `σ_soma(Σ_b w_b · quad(z_b) + b)` in a single step. The cell's input-output function changes shape, not just magnitude. Initializing the new branch weight small (`0.1 × mean(existing)`) keeps the new branch's contribution small at the instant of growth, but `quad(z_0)` on the *surviving* branch is already a different function than `z_0` was. Two empirical mitigations to evaluate in Phase 6: (a) accept the discontinuity and rely on the optimizer adapting `branch_weight` and `b` post-grow, or (b) calibrate `branch_weight[i, 0]` and `b[i]` analytically at the moment of grow so the K=2 forward matches the pre-grow K=1 output on a recent batch. If the discontinuity destabilizes training, fall back to `σ_branch="identity"` until calibration lands.

## 9. Recommendation

**Additive rollout.** Phases 1–4 ship 2.0 as strictly opt-in. Existing arcs continue at sequential default forever — long-range edges, `insert_layer`, and `axonal_gain ≠ 1.0` are only used by callers that ask. Net-zero regression risk for shipped paper benches and HF Space. Pneuma is the first opt-in customer; trioron's own benches opt in only after the substrate's parity case is established.

---

*This document records the trioron-substrate scope of long-range-synapse work. The pneuma-side proposal (`~/pneuma/docs/proposals/long_range_synapses.md`) covers the same proposal as it concerns pneuma's transformer-FFN integration; once 2.0 lands here, the pneuma-side draft should be reduced to a pointer at this file plus pneuma-specific integration notes.*
