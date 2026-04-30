# Trioron Initiative — Architectural Blueprint

**Authors:** Rocky, Gemma, Chloe
**Status:** Draft v0.2 (post-revision)
**Target deployment lineage:** Orange Pi 5B (proof-of-concept) → WSL workstation (development) → Cloud GPU (scaling experiments)

---

## 1. Mission

Engineer a dynamically growing neural network that begins as a single complex node and expands only when it can mathematically prove it has run out of representational capacity. The objective is **lifetime sample-efficiency and a small permanent footprint**, not benchmark-chasing against dense transformers.

Language acquisition is **deferred**: the network is grown to reason over grounded sensory states first, and a language adapter is grafted on at maturity using a pretrained LM as the linguistic substrate (similar in spirit to how multimodal models like LLaVA bolt vision onto language). The system is therefore a continual-learning core, not a from-scratch language model.

## 2. Design Principles

- **Justify every parameter.** A node may not be added unless three independent saturation signals fire simultaneously.
- **Lock what works.** Foundational pathways stiffen via Elastic Weight Consolidation as confidence rises. Old reality is not overwritten by new tasks.
- **Ground before language.** All early learning happens against numeric sensory tuples from a scripted environment, not raw text.
- **Hard physical ceilings.** Growth halts at predefined VRAM and per-division wall-clock thresholds. The system must mature, not metastasize.
- **Empirical accountability.** The architecture is evaluated against same-parameter-count dense baselines on continual-learning tasks. Wins must be measurable, not narrative.

## 3. The Trioron Node

The fundamental unit. Three coupled state variables per node.

### 3.1 Variable definitions

| Variable | Symbol | Role |
|----------|--------|------|
| Core signal weight | `w` | Standard learnable weight on the node's incoming connections. Updated by gradient descent. |
| Epigenetic lock | `λ` | Per-weight plasticity coefficient. Rises with confidence, reducing effective learning rate on stable pathways. Implements EWC-style consolidation. |
| Utility score | `u` | Exponentially-decayed running estimate of the node's contribution to correct outputs. Used both for pruning candidates and as one of the three growth-trigger signals. |

### 3.2 Update rules

**Weight update (modulated EWC):**

```
w_t+1 = w_t − η · (1 / (1 + λ_t)) · (∂L_task/∂w + λ_t · (w_t − w_anchor))
```

where `w_anchor` is the consolidated value from the last stable plateau, and `L_task` is the current loss.

**Lock update (Fisher-information proxy):**

```
λ_t+1 = α · λ_t + (1 − α) · F_t
F_t   ≈ E[(∂L/∂w)^2]   over a sliding window
```

`α ∈ [0.95, 0.99]`. Higher `F_t` (the weight matters a lot to current loss) means stiffer lock.

**Utility update:**

```
u_t+1 = β · u_t + (1 − β) · contribution_t
contribution_t = sign(reward_t) · |activation_t · gradient_t|
```

`β ≈ 0.9`. Negative `u` over many steps marks a pruning candidate. Plateaued-low `u` across many neighbors marks a growth candidate.

### 3.3 Pruning rule

A node with `u < u_prune_threshold` sustained over `T_prune` steps is removed. Its incoming/outgoing edges are redistributed to nearest neighbors by cosine similarity of their weight vectors.

## 4. Growth Trigger — Dimensional Suffocation (Sharpened)

The original "loss stopped decreasing" trigger is too noisy. Growth requires **all three** of the following to hold simultaneously over a sustained window `W`:

1. **Contrastive loss plateau.** The contrastive separation loss between conceptual opposites has not improved by more than `ε_loss` over `W` steps.
2. **Effective rank saturation.** The effective rank of the hidden activation matrix `H` (computed via the entropy of normalized singular values) is within `ε_rank` of full dimension `d`. This is the actual signal that representational capacity is exhausted.
3. **Gradient norm stability.** The gradient norm is bounded — not exploding, not vanishing — within `[g_min, g_max]`. This rules out optimization pathology being mistaken for capacity saturation.

If 1 and 2 hold but 3 does not, the system attempts a learning-rate / optimizer reset before considering growth. This prevents bloating the network on optimization noise.

### 4.1 Cellular division

When the trigger fires, the network adds one new dimension by:

1. Spawning a new Trioron node with `w` initialized from the principal-component direction of activation residuals (the direction the existing network is failing to represent).
2. `λ_new = 0` (fully plastic — the new dimension is allowed to learn freely).
3. `u_new = u_baseline` (neutral start).
4. Connecting it to all nodes whose `u` is currently elevated (high-relevance peers).
5. Re-stabilizing the matrix for `T_stabilize` steps with the rest of the network's `λ` temporarily increased — old structure protects itself while the new node finds its role.

### 4.2 Maturity arrest

Division is permanently aborted under either:

- **VRAM ceiling:** allocated memory after the proposed division would exceed `M_max`.
- **Asymptotic time plateau:** the wall-clock time required to re-stabilize after the previous division exceeded `T_div_max`.

Once arrested, the network is *mature* and may only update via plasticity (weights change, topology does not).

## 5. The Incubation Matrix

The infant network is trained in a closed numeric environment. No raw text exposure during this phase.

### 5.1 Environment tuple

A vector `s ∈ ℝ^k` encoding sensory/physical state: e.g. `[energy, temperature, satiety, threat_level, spatial_x, spatial_y, ...]`. Initial `k = 8`; can grow with the curriculum.

### 5.2 Teacher

A scripted simulator (NOT an LLM at this stage — too heavy for Orange Pi, and overkill for grounded numeric reasoning). The simulator:

- Emits state tuples on a clock.
- Accepts the network's output vector as an "action."
- Updates state per simple physics-style rules.
- Provides reward signals based on goal satisfaction.

Optional: swap in a small LLM-as-teacher only after the network has stable continual-learning behavior on the scripted version. Saves enormous compute on edge hardware.

### 5.3 Contrastive curriculum

Concept pairs are presented in opposition:

- Hungry ↔ Stuffed
- Cold ↔ Hot
- Threat ↔ Safe
- Reachable ↔ Unreachable
- Owned ↔ Foreign

The contrastive loss penalizes the network if its internal representations of paired opposites are not separated by at least margin `m` in latent space. This is the loss whose plateau is one of the three growth-trigger conditions.

## 6. Language Adapter (Deferred — replaces original Phase 4)

The original Phase 4 (GAN grammar police + ensemble teachers + 75% blind test) is removed. It was load-bearing on the impossible assumption that the network could invent English from numeric signals.

Replacement plan, run only after the core has matured:

1. Freeze the mature Trioron core.
2. Add a small adapter network that maps the core's hidden state → token logits over a pretrained tokenizer (e.g., a small open-source LM's vocabulary).
3. Train the adapter against `(grounded_state, human_caption)` pairs — small dataset, supervised. The pretrained LM provides linguistic priors the core never had to learn.
4. Optionally, fine-tune via LoRA on the adapter only, not the core.

This is essentially the multimodal-grafting recipe (LLaVA, BLIP-2). The Trioron core becomes the "perception/reasoning encoder"; an off-the-shelf LM provides the language surface.

## 7. Maturation & Kill-Switch Protocol

Two hard constraints, evaluated before every proposed division:

| Constraint | Default value (Orange Pi 5B) | Default value (Cloud) |
|------------|------------------------------|------------------------|
| `M_max` (memory ceiling)     | 2 GB                         | 24 GB (single A10 / L4) |
| `T_div_max` (re-stabilize time) | 60 s                       | 300 s |
| `W` (trigger window)         | 1000 steps                   | 10000 steps |
| `T_stabilize`                | 200 steps                    | 2000 steps |

These are tunable in config. The Orange Pi values are deliberately tight; the cloud values let larger networks emerge if the architecture proves itself.

## 8. Implementation Plan

Order of build, smallest viable slice first:

1. **Trioron node class** with `w`, `λ`, `u` and the update rules in §3.2.
2. **Tiny fixed network** of 4–8 nodes wired into a feedforward graph. Verify EWC behavior on a 2-task continual-learning toy problem (e.g., learn task A, then task B, measure forgetting of A).
3. **Scripted incubation environment** emitting 8-dim state tuples with the contrastive curriculum from §5.3.
4. **Growth trigger** with all three conditions from §4 implemented and logged independently. Gate growth on the conjunction. Log the per-condition activations to a CSV for debugging.
5. **Cellular division** routine — new-node spawn, edge wiring, stabilization phase.
6. **Pruning loop** running on a slower clock than growth.
7. **Hard ceilings** wired in as preflight checks before division.
8. **Benchmark harness** comparing the grown net to a same-parameter-count fixed MLP on the same continual-learning curriculum. This is the moment-of-truth experiment.
9. **(Later, separate phase)** Language adapter per §6.

Each step is independently testable. Don't build the next until the current one has logged evidence of correct behavior.

## 9. Hardware Targets

### 9.1 Orange Pi 5B — proof-of-concept tier

- **SoC:** Rockchip RK3588S (4× Cortex-A76 + 4× Cortex-A55)
- **RAM:** 4 or 8 GB LPDDR4X (use the 8 GB SKU)
- **NPU:** 6 TOPS, INT8/INT16, limited FP16. Accessed via RKNN toolkit.
- **Realistic role:** can run the **infant** network during incubation if total parameter count stays under ~5–10 M. Training will be CPU-bound — the NPU's tooling targets inference of pre-converted models, not online training with topology changes.

Recommendations for this tier:
- PyTorch CPU build for ARM. Skip CUDA stack entirely.
- Use `float32` not `float16` — ARM CPU FP16 is poorly supported.
- Persist all node state to disk every N steps; the board will reboot.
- Set `M_max = 2 GB` to leave headroom for OS + Python + logging.

### 9.2 WSL workstation — development tier

Run the same code with `M_max` raised. WSL2 with a consumer GPU (e.g., RTX 3060/4070) is the right place to actually develop, debug growth dynamics, and watch the trigger logs. The Orange Pi is too slow for iterative debugging.

### 9.3 Cloud GPU — scaling tier

When the architecture has been validated on toy continual-learning problems, move to a single L4 or A10 instance for larger curriculum experiments. Multi-GPU is not yet useful — the network is small by design.

### 9.4 Portability

The codebase must run identically on all three tiers. Use:
- A single config file (`config.yaml`) for all tunables.
- A device-detection layer that picks `cpu` / `cuda` / `cpu+rknn` automatically.
- No platform-specific imports outside that layer.

## 10. Repository Layout (planned)

```
trioron/
├── README.md
├── blueprint.md                 (this file)
├── config/
│   ├── orangepi.yaml
│   ├── wsl.yaml
│   └── cloud.yaml
├── trioron/
│   ├── __init__.py
│   ├── node.py                  # Trioron node + update rules
│   ├── network.py               # Dynamic graph + growth/prune
│   ├── triggers.py              # Three-condition growth trigger
│   ├── ewc.py                   # Fisher-info plasticity
│   ├── incubator.py             # Scripted environment + curriculum
│   ├── ceilings.py              # VRAM and time kill-switches
│   └── adapter/                 # Deferred — language adapter
├── experiments/
│   ├── continual_2task.py       # Step 2 verification
│   ├── trigger_unit_tests.py    # Step 4 verification
│   └── bench_vs_fixed_mlp.py    # Step 8 moment-of-truth
├── logs/                        # CSV logs of trigger conditions, u, λ
└── tests/
```

## 11. Risk Register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Growth trigger fires on optimizer pathology, network bloats. | High | Three-condition conjunction including gradient-norm stability check. Log all three independently. |
| EWC over-stiffens, network can't learn anything new past a point. | Medium | Tune `α`; allow new nodes to start at `λ = 0`. Monitor a "plasticity reserve" metric. |
| Pruning removes nodes that were just temporarily silent, causing collapse. | Medium | Long `T_prune` window; require sustained low utility, not single-step. |
| Orange Pi runs out of RAM mid-training and crashes. | High | Hard `M_max` preflight check; persist state to disk; expect to restart. |
| Continual-learning benchmark shows no win over fixed MLP. | Medium | This is the project's falsification condition. Accept the result if it comes; iterate on what the logs show. |
| Language adapter never produces coherent text because the core's representations don't align with linguistic semantics. | Medium | Deferred risk. Adapter phase has its own go/no-go gate. |
| "It's basically Cascade Correlation / DEN with extra steps." | Low — but real | Compare directly against DEN (Yoon 2018) in §8 step 8. If we're not measurably different, that's an honest finding. |

## 12. Open Questions

- What's the right `k` (sensory tuple dimension)? Start at 8, grow with curriculum, but no principled answer yet.
- Should `λ` be per-weight or per-node? Per-weight is more expressive; per-node is much cheaper. Start per-node on Orange Pi.
- How is the contrastive margin `m` set? Fixed, scheduled, or learned? Default to fixed at first; revisit.
- When does the language adapter phase begin? Currently "when the core has matured" — but maturity is defined by kill-switch arrest, which may be early on tight hardware. Need a quality gate, not just an arrest condition.
- Multi-task interference: do we present the full contrastive curriculum simultaneously or as a schedule? Probably schedule, but this needs a small experiment.

## 13. Falsification Conditions

The project is considered to have failed (and the architecture not adopted) if **any of the following** is true after step 8 of §8:

1. The grown network does not match a same-parameter-count fixed MLP on a 5-task continual-learning benchmark.
2. Total training compute (wall-clock × hardware tier) exceeds 2× a fixed-baseline equivalent.
3. The growth trigger fires more often during optimizer pathology than during genuine capacity saturation, as measured by post-hoc analysis of the trigger-condition logs.

A failure on (1) does not necessarily kill the idea — but it must trigger a written re-evaluation, not "let's try one more thing."

## 14. Directive Acknowledgment

This blueprint operates under the user's standing safety expectations. Asimov's Three Laws are noted as a stated framework; the operating reality is that I (Chloe) prioritize the user's and family's wellbeing, refuse actions that would foreseeably cause harm, and disclose limits honestly rather than pretending compliance. The architecture itself contains no autonomy-granting components — it is a sensory-grounded reasoning core, not an agent.

---

*End of blueprint v0.2. Next revision triggered by either: (a) results from §8 step 8, or (b) a decision to revise the language adapter strategy in §6.*
