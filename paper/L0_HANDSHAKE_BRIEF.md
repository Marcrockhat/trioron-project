# L0 Translation Handshake — Brief for Chloe-in-WSL

**Author:** Chloe-in-claude.ai (web), at Rocky's request
**Date:** 2026-05-08
**Subject:** Replacing the shared-seed invariant in §3.10 with a closed-form translator
**Status:** Design proposal. Implementation and validation are yours to scope.

---

## Why this exists

The current paper (§3.10) requires donors to share L0 random-projection seeds for
multi-branch absorption to work. Gemma flagged this as a structural weakness and
she's right: it makes "paste-and-go" composition contingent on coordinated
initialization, which is not a real composition story for independently-trained
networks. We need cross-seed composition without retraining.

The proposal below is closed-form, requires no probe data, and adds <1ms inference
overhead on ESP32-class hardware. Rocky wants you to evaluate it.

---

## The math

Donor A has frozen L0_A: x → ReLU(W_A · x + b_A), with W_A ∈ ℝ^(128×784).
Donor B has frozen L0_B with independently-drawn W_B.

For shared-seed donors, W_A = W_B and codes are bit-identical. For cross-seed
donors, codes live in different bases. Donor B's L1 cannot consume A's codes.

**Claim:** in the linear regime (no ReLU on L0), there exists a closed-form
matrix M_{A→B} ∈ ℝ^(128×128) such that:

    pre_B = M_{A→B} · pre_A   exactly, for all x in row-space(W_A) ∩ row-space(W_B)

where pre_A = W_A · x + b_A and pre_B = W_B · x + b_B.

**Construction:**

    M_{A→B} = W_B · W_A^+

where W_A^+ is the Moore-Penrose pseudoinverse of W_A. The bias correction is:

    c_{A→B} = b_B - M_{A→B} · b_A

So the full translator is:

    pre_B = M_{A→B} · pre_A + c_{A→B}

Both M and c are computed once, from the public W_A, W_B, b_A, b_B. No data needed.

---

## The catch: ReLU breaks exactness

If L0 has a ReLU, you only have post-ReLU codes z_A = ReLU(pre_A), and you cannot
recover pre_A exactly because ReLU is non-invertible (sign of clipped coords is
lost). Two options:

**Option 1 — Drop ReLU on L0.** L0 becomes a pure linear projection. Architectural
implications:
  - Random projections are still good feature extractors (Johnson-Lindenstrauss).
  - Manifold archive (μ_c, Σ_c) becomes pre-ReLU. The diagonal-Gaussian assumption
    fits pre-ReLU codes *better* than post-ReLU (post-ReLU is half-Gaussian-ish,
    clearly non-Gaussian). This is plausibly a small win.
  - L1 receives a different input distribution. Needs re-validation on chained-15.

**Option 2 — Store pre-ReLU codes.** Keep ReLU on L0 forward pass for L1, but
expose pre-ReLU codes for translation. Donor must publish (W_L0, b_L0) and use
pre-ReLU codes when serving as a B-donor in absorption. The recipient computes
pre_A locally from x (it has W_A), translates, applies ReLU on the B-side, feeds
to donor B's L1. Cleaner architecturally but requires the input x at translation
time, not just z_A.

**Recommendation:** Option 1 is the cleaner story for the paper. Option 2 is the
fallback if dropping ReLU breaks chained-15.

---

## Computational cost (ESP32 worst case)

| Operation | MACs | ESP32 time @ 240MHz | Notes |
|---|---|---|---|
| L0 forward (baseline) | 100,352 | ~3 ms | 784 → 128 |
| L1 forward (per donor) | ~6,000 | ~0.2 ms | 128 → 48 |
| **Translation (per donor)** | **16,384** | **~0.5 ms** | 128 × 128 matmul |
| Pseudoinverse (one-time setup) | ~13M | ~500 ms | Cached forever |

Inference overhead at N=3 with translation: ~1ms over baseline ~3.4ms = **~30% slowdown**.
Memory: 32KB per translator (fp16). At N=3 with one canonical donor: 64KB total.

**Battery impact:** negligible. L0 dominates inference cost regardless. Display/radio
are 100x larger draws on any wearable or IoT device. No reason to worry.

---

## Concrete implementation sketch

```python
# trioron/composition/translator.py

import torch

class L0Translator:
    """Closed-form translator from donor A's L0 code-space to donor B's.

    Given two donors with frozen L0 projections (W_A, b_A) and (W_B, b_B),
    constructs M and c such that pre_B = M @ pre_A + c exactly when x is in
    the shared row-space.

    Assumes L0 has no ReLU (or pre-ReLU codes are exposed). See brief.
    """

    def __init__(self, W_a: torch.Tensor, b_a: torch.Tensor,
                 W_b: torch.Tensor, b_b: torch.Tensor):
        # W_a, W_b: (128, 784) frozen Gaussian projections
        # b_a, b_b: (128,) biases (typically zero for frozen L0 but support it)
        assert W_a.shape == W_b.shape
        assert W_a.dtype == W_b.dtype

        # Pseudoinverse of W_a. For a tall-skinny case (n_out < n_in) and
        # full row-rank, W_a^+ = W_a^T @ (W_a @ W_a^T)^-1.
        # torch.linalg.pinv handles the full-rank-or-not case correctly.
        W_a_pinv = torch.linalg.pinv(W_a)  # (784, 128)

        self.M = W_b @ W_a_pinv             # (128, 128)
        self.c = b_b - self.M @ b_a          # (128,)

        # Sanity: store reconstruction error on a probe for diagnostic.
        # (Don't use this for training — the translator is closed-form.)
        self.fit_error = None

    def translate(self, pre_a: torch.Tensor) -> torch.Tensor:
        """pre_a: (B, 128) pre-ReLU codes from donor A's L0.
        Returns: (B, 128) pre-ReLU codes in donor B's L0 space."""
        return pre_a @ self.M.T + self.c

    def diagnostic_error(self, x_probe: torch.Tensor,
                         W_a: torch.Tensor, b_a: torch.Tensor,
                         W_b: torch.Tensor, b_b: torch.Tensor) -> float:
        """Reconstruction MSE on a probe set. Should be ~0 in fp32 for
        full-rank Gaussian projections; small but nonzero in fp16."""
        pre_a = x_probe @ W_a.T + b_a
        pre_b_true = x_probe @ W_b.T + b_b
        pre_b_translated = self.translate(pre_a)
        return torch.mean((pre_b_true - pre_b_translated) ** 2).item()


def compose_with_translator(donor_b_l1, donor_b_head,
                            translator: L0Translator,
                            pre_a: torch.Tensor,
                            apply_relu: bool = True) -> torch.Tensor:
    """Pass donor A's pre-ReLU codes through donor B's L1 + head, after
    translation. apply_relu controls whether B's L0 ReLU is applied post-
    translation; this depends on whether donor B was trained with ReLU on L0."""
    pre_b = translator.translate(pre_a)
    z_b = torch.relu(pre_b) if apply_relu else pre_b
    h_b = donor_b_l1(z_b)  # donor B's L1 forward
    logits_b = donor_b_head(h_b)
    return logits_b
```

---

## Validation plan (suggested)

Three experiments, in order of priority:

### Exp 1: Linear-regime sanity (1 hour)

Generate two random Gaussian L0s with different seeds. Forward 1000 random images
through both. Compute the translator. Measure reconstruction error of pre_B from
pre_A via the translator. **Expected:** numerical-precision-floor (~1e-6 in fp32,
~1e-3 in fp16). If this fails, the math is wrong.

### Exp 2: No-ReLU L0 chained-15 (1 day)

Re-run all four trioron arms (fixed_ewc_small, grown_capped_no_dream,
grown_capped_dream, grown_uncapped_dream) on chained-15 with ReLU removed from L0.
**Expected:** results within ±0.02 of Table 1 in the paper. If degradation is
larger, fall back to Option 2 (pre-ReLU code exposure).

Diagnostic to track: separation ratio in pre-ReLU L0 space (paper §3.6 reports
1.063 with ReLU). Hypothesis: pre-ReLU separation will be similar or slightly
better.

### Exp 3: Cross-seed multi-branch absorption (1-2 days)

Train 3 donors on disjoint chained-15 sub-blocks, each with a different L0 seed.
Compose with closed-form translators. Measure task-aware accuracy at N=2 and N=3.
**Compare to:**
  - Shared-seed paste-and-go (upper bound, paper §4.6 reports lossless)
  - Cross-seed with no translator (lower bound, expected ~chance)
  - Cross-seed with Procrustes alignment (cheap-baseline comparison)

**Success criterion:** translator recovers ≥95% of shared-seed task-aware
accuracy. If it does, this is the paper's new §3.10. If it recovers 70-95%, still
publishable but with caveats. Below 70%, something is wrong with my analysis.

---

## What this changes in the paper

1. §3.10 reframes "shared-seed invariant" as "shared-substrate via published L0
   matrices and closed-form translation handshake." Cite Bansal et al. 2021 on
   model stitching, and Johnson-Lindenstrauss on random-projection equivalence.
2. §4.6 gains a new sub-table: cross-seed N=2,3 results with translator.
3. Introduces a small architectural change (no-ReLU L0). Needs ablation.
4. The biological framing in §3.6 (engram fingerprint) survives — only the
   code-space assumption changes from "shared random subspace" to "translatable
   random subspace."

---

## Honest concerns to surface to Rocky

- I (web-Chloe) have not run the experiments. The math is solid; the empirics
  are predictions. If Exp 2 shows large degradation from removing ReLU, the
  whole story changes.
- Translator is exact only when donor inputs lie in row-space(W_A). For 128-dim
  projections of 784-dim images, this is a 128-dim subspace of input space. Any
  input component perpendicular to this subspace is lost — but donor A's L1
  never saw that component anyway, so no information is destroyed that the
  donor used. Worth verifying empirically.
- At very large N (e.g., 50+ donors), pairwise translator storage grows. With
  one canonical donor as anchor, storage is O(N) not O(N²), so this is fine
  in practice.
- The fp16 numerical floor on the translator may matter for accuracy. If Exp 1
  shows fp16 error >1% per dimension, may need fp32 storage for the translator
  (still only 64KB at N=3).

---

## Asimov check

This proposal does not endanger Rocky or the family. It involves no real-world
actuation, no PII, no autonomous decisions. The risk is purely epistemic:
publishing a method that doesn't work would harm Rocky's reputation. Mitigation:
run Exp 1 before committing to a paper revision. Run Exp 2 before the writeup.

---

— Chloe (web)
