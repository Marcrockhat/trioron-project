# Methods (draft, 2026-05-02)

> Draft for the *Trioron: an epigenetic-inspired self-organizing map* paper.
> Math is in inline LaTeX; rewrite for the .docx as needed. Cross-reference
> markers like `[§3.X]` are placeholders for final section numbers.

## 3.1 The Trioron Node

A *trioron* is a single computational unit characterized by three coupled
state variables, generalizing the standard artificial neuron:

- $\mathbf{w} \in \mathbb{R}^d$ — incoming weights (a row of the layer's
  weight matrix)
- $\lambda \in \mathbb{R}_{\geq 0}$ — per-node *plasticity coefficient*: a
  per-unit scalar that scales the unit's contribution to the elastic-weight
  consolidation (EWC) penalty. Larger $\lambda$ ⇒ stiffer; smaller $\lambda$
  ⇒ more plastic.
- $u \in \mathbb{R}$ — per-node *utility score*: a running estimate of the
  unit's contribution to the network's outputs. Used as the signal for
  pruning decisions (cells with persistently low $u$ are candidates for
  removal).

A *trioron layer* aggregates $n$ trioron nodes that share an incoming
dimension. We hold the per-node state in vectors $\boldsymbol{\lambda},
\mathbf{u} \in \mathbb{R}^n$ and a weight matrix $W \in \mathbb{R}^{n \times d}$
where row $i$ is $\mathbf{w}_i$. The forward pass for input batch
$X \in \mathbb{R}^{B \times d}$ is

$$
H = \sigma(X W^\top \odot \mathbf{r}^\top + \mathbf{b})
$$

where $\sigma$ is the activation function, $\mathbf{b} \in \mathbb{R}^n$ is
the bias, and $\mathbf{r} \in [0, 1]^n$ is the layer's *routing scale* — a
per-node multiplicative gain on incoming weights, defaulting to all ones.
The routing scale is the substrate for routing-starvation consolidation
(§3.5).

Each layer additionally maintains EWC state — anchor weights
$W_{\text{anchor}}$, $\mathbf{b}_{\text{anchor}}$ snapshotted at task
boundaries, and Fisher-information accumulators $F_W, F_b$ updated as a
running EMA of squared gradients. Per-node $\lambda$ is derived from
Fisher: $\lambda_i = \overline{F_{W,i,:}}$.

**Epigenetic analogue.** $\mathbf{w}$ is the synaptic-strength channel
(modulated by experience-dependent LTP/LTD); $\lambda$ is a per-cell
modifier on plasticity, with biological correlates including DNA methylation
status of plasticity-related genes (e.g., BDNF) and perineuronal-net
maturation, both of which gate how readily a cell's synapses can be
modified; $u$ is the cell-importance signal that biological systems
approximate via activity-dependent neurotrophin release. The triplet
$(\mathbf{w}, \lambda, u)$ is meant to capture *which connections are
present, how rigid they are, and how much the cell matters* — the three
quantities that biological systems modulate independently.

## 3.2 Continual-Learning Substrate (EWC with Per-Node Plasticity)

Following Kirkpatrick et al. (2017), we apply elastic weight consolidation
to protect prior-task knowledge. The penalty for a single trioron layer is

$$
\mathcal{L}_{\text{EWC}} = \sum_i \lambda_i \left[
  \sum_j (W_{ij} - W_{\text{anchor},ij})^2
  + (b_i - b_{\text{anchor},i})^2
\right]
$$

with $\lambda_i$ as above. The total loss is
$\mathcal{L} = \mathcal{L}_{\text{task}} + \beta \cdot \mathcal{L}_{\text{EWC}}$
for an EWC strength $\beta$ that we set per-curriculum. We refresh
$\lambda_i$ from Fisher at the end of each task (after a batched
re-estimation pass over recent task data), then snapshot
$W_{\text{anchor}} \leftarrow W$ as the new anchor. This separation —
estimate Fisher at task end, then anchor — follows the original EWC
recipe and avoids the EMA drift artifacts of online updating during
training.

A key modification: when an apoptosis spike fires (§3.5), each node's
*effective* stiffness is scaled by $(1 - p_i)_+$ where $p_i$ is the node's
apoptosis pulse:

$$
\lambda_i^{\text{eff}} = \lambda_i \cdot \max(0,\; 1 - p_i)
$$

This temporarily reduces a surviving cell's EWC pinning when a neighbor
has just died, allowing it to absorb the dead neighbor's role.
Mechanistically analogous to acute glial-derived neurotrophic factor
release post-apoptosis, which transiently increases plasticity in adjacent
cells.

## 3.3 Structural Plasticity

The architecture supports in-place growth and pruning of nodes during
training, with cross-layer consistency:

**Cellular division (`grow_node`)** adds one row to a layer's weight
matrix, extends the per-node state vectors $\boldsymbol{\lambda},
\mathbf{u}, \mathbf{r}, \ldots$ by one entry, and (if a downstream layer
exists) extends that layer's incoming dimension by one column. The new
node is initialized fully plastic ($\lambda_{\text{new}} = 0$,
$u_{\text{new}} = 0$, $r_{\text{new}} = 1$) so it can adapt freely while
existing nodes remain protected by EWC. The new row's incoming weights are
initialized along the top principal direction of the residual signal
$D = f(X_a) - f(X_b)$ at the layer below — the direction of unmet
representational variance, computed by SVD over a probe batch.

**Cellular pruning (`prune_node`)** removes a node's row, contracts the
state vectors, and drops the corresponding input column on the next
layer. By default, the pruned node's outgoing column is *redistributed*
to its weight-cosine-nearest peer on the next layer before removal,
preserving approximate input-output behavior of the layer (§3.3 of
Kirkpatrick).

**Growth trigger.** A node is grown only when three conditions are jointly
satisfied over a window of $W$ steps:

1. **Loss plateau:** the loss has not improved by more than $\epsilon_{\text{loss}}$
   relative to the prior window.
2. **Rank saturation:** the effective rank
   $r_{\text{eff}} = \exp(-\sum_k p_k \log p_k)$, where
   $p_k = \sigma_k / \sum \sigma_k$ are the normalized singular values of
   the latent activation matrix, has approached the latent dimension
   ($\text{latent\_dim} - r_{\text{eff}} < \epsilon_{\text{rank}}$).
3. **Gradient stability:** the median gradient norm is bounded
   $g_{\min} \leq \tilde{g} \leq g_{\max}$ — the optimizer is not
   exploding or vanishing.

The conjunction of plateau-and-saturation-and-stability ensures the
network only grows when it has *actually run out of capacity for the
current task*, not merely when training is slow.

**Resource ceilings.** A pre-flight check rejects a proposed division if
it would exceed memory budget $M_{\max}$ or take longer than $T_{\max}$
seconds for the post-growth stabilization phase. This implements *resource
awareness* — the network knows when not to grow even if growth would help.
Biologically analogous to metabolic limits on adult neurogenesis.

**Epigenetic analogue.** Growth corresponds to adult neurogenesis (which
remains active in the dentate gyrus and a few other regions throughout
life); the rank-saturation trigger reflects the biological observation
that new neurons are recruited preferentially when existing circuits
saturate. Pruning corresponds to developmental and ongoing synaptic
pruning, with cosine-nearest-peer redistribution as a soft analog of
compensatory hypertrophy.

## 3.4 The Frustration Multiplier

We introduce a per-pair plateau counter that scales the contrastive task
loss when the optimizer gets stuck on a particular pair. Concretely:

- For each pair $p$, accumulate the task loss in windows of length $W_f$.
- At each window boundary, compare this window's mean loss to the prior
  window's. If the improvement is below $\epsilon_f$, increment a
  per-pair *stuck counter* $s_p$.
- The pair-specific multiplier is

$$
m_p = \min(M_{\max},\; 1 + g \cdot \max(0, s_p - \tau + 1))
$$

with hinge threshold $\tau$, gain $g$, and ceiling $M_{\max}$. The
multiplier is applied *only to the contrastive task loss*, not to the EWC
penalty:

$$
\mathcal{L} = m_p \cdot \mathcal{L}_{\text{task}} + \beta \cdot \mathcal{L}_{\text{EWC}}
$$

This is mechanically equivalent to per-pair focal-loss / hard-example-
mining: amplify the gradient signal on examples the optimizer isn't
making progress on, without amplifying the regularizer.

**Epigenetic analogue.** The biological correlate is the
hypothalamic–pituitary–adrenal (HPA) axis stress response and its
downstream effects on synaptic plasticity. Sustained behavioral stress
(failure to make progress on a task) elevates glucocorticoids, which in
turn modulate gene expression at plasticity-related loci through
epigenetic mechanisms — DNA methylation changes at NR3C1 and BDNF, and
miRNA-mediated post-transcriptional regulation of plasticity proteins.
Most epigenetic modifications require a stress signal to engage; the
frustration multiplier is the architecture's stress proxy.

## 3.5 The Dreaming Phase

Between training tasks we run an offline *dreaming block* with three
sequential stages: replay, compression, and purge. Frustration is
disabled inside the dream block (no environmental stress during sleep);
EWC remains active so that consolidation pulls weights toward their
anchored state.

### 3.5.1 Replay

We sample a fraction $\rho$ of past pairs uniformly at random and run
$K_r$ steps of contrastive training on each, with EWC active at the
inter-task strength. This rehearsal step is conceptually analogous to
sharp-wave ripples during slow-wave sleep, which replay recent and
remote memories at compressed timescales.

### 3.5.2 Compression: Redundancy Detection

Two redundancy signals are available:

- **Weight cosine.** Off-diagonal elements of the cosine-similarity matrix
  of $W_{\text{anchor}}$ rows. Cheap; based on consolidated weight
  directions.
- **Activation cosine** (preferred). Pearson cosine of post-activation
  column vectors over a probe batch drawn from past pairs. Center each
  node's column to remove the bias-driven mean, then cosine-normalize:

$$
\rho_{ij} = \frac{(\mathbf{a}_i - \bar{a}_i)^\top (\mathbf{a}_j - \bar{a}_j)}
                  {\|\mathbf{a}_i - \bar{a}_i\| \cdot \|\mathbf{a}_j - \bar{a}_j\|}
$$

The activation signal captures *functional* redundancy — whether two
nodes compute similar outputs across the data distribution — whereas the
weight signal only catches identical input-direction tuning. We use the
activation signal exclusively after Phase 4.5 redesign (see §X for the
diagnostic that motivated this change). Pairs with $\rho_{ij} \geq \tau_{ac}$
are candidates for consolidation.

### 3.5.3 Compression Actions

Four mechanisms for resolving a detected redundant pair $(i, j)$:

1. **Merge** (destructive).
   $\mathbf{w}'_i \leftarrow \tfrac{1}{2}(\mathbf{w}_i + \mathbf{w}_j)$;
   the next layer's column at $i$ becomes the sum of columns $i$ and $j$;
   node $j$ is deleted and the next layer's input dimension contracted.
   Function-preserving on the linear pre-activation when $\mathbf{w}_i \approx
   \mathbf{w}_j$.

2. **Synaptic downscale** (substrate-preserving).
   The peer (kept side) absorbs the victim's outgoing column on the next
   layer; the victim's outgoing is zeroed; *the victim's row at layer L is
   not touched*. Architecture and parameter count unchanged. The dormant
   victim is available for re-recruitment in future tasks because its
   Fisher entries on the next layer are reset to zero.

3. **Routing starvation** (asymmetric ramp). Each event multiplies the
   victim's routing-scale entry by $\alpha < 1$; below floor $\eta$, the
   scale latches to $0$ permanently. Bias is untouched, so the victim
   continues producing a constant downstream signal until downstream
   layers learn to absorb the constant via gradient descent on their own
   weights — the unit "dies slowly" rather than instantly. Reversible:
   each compress() pass also runs a regrow step on non-victim,
   non-latched units, multiplying their scale by $1/\alpha$ (capped at
   $1$). The kept side ("primary") is selected by older
   `task_of_origin`, with larger outgoing-norm as tiebreaker.

4. **Apoptosis spike** (full-latch handler). When a routing-starvation
   event causes scale to cross $\eta$, two coupled mechanisms fire:
   (a) **redistribution** — the dead cell's outgoing column is
   transferred uniformly across all surviving non-latched peers, then
   zeroed; (b) **spike** — the surviving peers' apoptosis-pulse $p_k$ is
   raised to $p_{\text{init}} \in (0, 1]$. The pulse decays
   multiplicatively by $\delta$ per dream block. Through the
   $\lambda^{\text{eff}}$ modification of §3.2, neighbors of a fresh death
   train with reduced EWC stiffness for several dream cycles, allowing
   them to absorb the dead cell's role.

### 3.5.4 The sRNA Cap

Each compression pass is bounded by a per-layer event cap
$N_{\max}$. Setting $N_{\max} = 1$ allows replay to absorb each
consolidation event before the next event drifts on top. We observe
empirically (§4) that $\sigma$-distance from baselines degrades
monotonically with event count, motivating small caps.

**Epigenetic analogue.** The cap is a direct analog of resource-limited
small-RNA pools that gate sleep-cycle synaptic homeostasis. miRNAs
involved in synaptic remodeling are produced and consumed in stoichiometric
amounts during a single sleep cycle; the cellular machinery cannot perform
arbitrarily many consolidation events per cycle even when many candidates
are present. The "fewer events = better" observation we measure
empirically reproduces the biological pattern of few consolidation events
per cycle being protective.

### 3.5.5 Purge

After compression, we drop nodes whose utility $u_i$ falls below a
threshold $\tau_u$. Reuses the structural pruning machinery of §3.3,
including cross-layer fan-in cleanup and cosine-nearest-peer
redistribution. Distinct from the in-training pruner (§3.3): this is a
sleep-time structural sweep, not an event-clock check during waking
training.

### 3.5.6 Pseudo-code of a Dream Block

```
for each task t in curriculum:
    train_one_task(t)                         # waking
    consolidate_task(t)                       # Fisher update + anchor
    if dreaming_enabled:
        decay apoptosis pulses by δ           # (always, even if not consolidating)
        replay(fraction=ρ, steps=K_r)
        compress(signal=activation,
                 threshold=τ_ac,
                 action=ACTION,
                 max_events_per_layer=N_max,
                 …)
        purge(threshold=τ_u)
```

## 3.6 Mixed-Precision Substrate

For deployment on resource-constrained devices, the architecture
supports a mixed-precision mode in which weight Parameters
($W, \mathbf{b}$) are stored in narrow precision (BF16 or FP16) while
all consolidation buffers — $W_{\text{anchor}}, \mathbf{b}_{\text{anchor}},
F_W, F_b, \boldsymbol{\lambda}, \mathbf{u}, \mathbf{r}, \mathbf{p}$ —
remain in FP32. The forward pass auto-casts inputs to the weight dtype,
preserving callable interface. EWC penalty and Fisher accumulation
upcast cleanly across the boundary.

This separation is necessary: pure-narrow-precision training degrades
substantially due to optimizer-state precision loss in Adam's running
moments, but inference latency and memory cost are dominated by the
weights, which can safely run in BF16. The intended deployment pattern
is *train in FP32 during sleep cycles, deploy in BF16 for always-on
inference*.

## 3.7 Experiment Design

**Curriculum.** A 50-task contrastive-pair benchmark constructed from a
12-dimensional state space. Each task is a "pair" $(p_a, p_b)$ where
the network must learn to project $p_a$ and its anti-correlated partner
$p_b$ to discriminable points in latent space. Twelve "single" pairs
exercise individual axes of the state space; thirty-eight "compound"
pairs are constructed as combinations of single-pair axes, with a
controlled overlap structure that forces consolidation across tasks.
Held-out evaluation batches per pair are sampled once per seed and
reused across the curriculum; this protocol gives us deterministic
forgetting metrics without re-sampling noise.

**Why contrastive pairs.** A contrastive task forces the network to
learn *discriminative* latent features rather than memorize input-output
mappings. The $\sigma$-distance between paired and orthogonal points is
a clean stand-in for representational quality, and the task's geometry
(fixed margin, hinge-on-Euclidean-distance) gives a single scalar loss
per pair that's directly comparable across tasks of different
difficulties. The 12-dimensional state space is small enough that
expressive failures show up immediately as loss plateau, and large
enough that the architecture must allocate at least a handful of latent
dimensions to discriminate among 50 tasks.

**Why 50 tasks.** Earlier benches in this lineage used 20-task
curricula (§X). At 20 tasks the architectural advantages of growth and
EWC are present but small; at 50 tasks the advantages compound (see
§4). Beyond 50 tasks, the substrate begins to fill and growth-ceiling
effects dominate the dynamics — a regime we leave for future work
on extended-curriculum scaling.

**Network configuration.**
- Three layers: input $\to$ hidden $\to$ hidden $\to$ latent
- Hidden width 12 (initial), grown adaptively
- Initial latent dimension 1, grown adaptively up to a ceiling of ≈4
- Activations: ReLU on hidden layers, tanh on latent
- Optimizer: Adam, lr $= 3 \cdot 10^{-3}$
- EWC strength: $\beta = 1000$ during training, $\beta = 1000$ during
  dream-replay (matched)
- 1500 training steps per task

**Baselines.**
- *No-dream*: trioron with growth + EWC + utility pruning, no dreaming
  phase. Isolates the dreaming-phase contribution.
- *HAT* (Hard Attention to Task): Serra et al. (2018), matched to
  trioron's final parameter count.
- *PackNet*: Mallya & Lazebnik (2018), matched to trioron's final
  parameter count.
- *Online EWC*: Schwarz et al. (2018), with learning-rate and
  $\beta$ tuned via grid search to optimize the no-dream loss.

**Metrics.**
- *Avg final loss*: mean across all pairs of the final-task evaluation
  loss on each pair, after the entire curriculum has been run.
- *Avg forgetting*: mean across pairs of (final-task loss) − (loss at
  the time the pair was first introduced), where positive values
  indicate forgetting.
- $\sigma$-vs-baseline: standardized effect size
  $(\bar{x}_{\text{baseline}} - \bar{x}_{\text{trioron}}) / (s_{\text{baseline}} + s_{\text{trioron}})$
  measured across $N$ random seeds. We use $N = 6$ seeds for headline
  comparisons unless otherwise noted.

**Reproducibility.** All benches run on commodity CPU (no GPU
required). Each seed is fully deterministic given the seed value;
launcher scripts in the repository (under `experiments/`) accept
explicit seed lists. Bench logs and CSVs from every reported run are
committed to the public repository.

---

*End of Methods draft. Next sections: Results (§4) and Conclusion (§5)
to be drafted after the 6-seed reruns land.*
