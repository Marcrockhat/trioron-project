"""Closed-form L0 handshake translator (paper L0_HANDSHAKE_BRIEF.md).

A trioron donor is fully specified by (W_L0, b_L0, L1, head, archive). Two
donors trained with independently-drawn L0 random projections produce L0
codes that live in different bases, so donor B's L1 cannot consume donor A's
codes. The closed-form translator below maps pre-ReLU L0 codes from donor A's
basis into donor B's basis using only the public L0 matrices — no probe
data, no training, no shared seed.

Math::

    pre_A = W_A · x + b_A         # donor A's pre-activation L0
    pre_B = W_B · x + b_B         # donor B's pre-activation L0

    M = W_B · W_A^+               # (n_out, n_out) basis change
    c = b_B - M · b_A             # bias correction
    pre_B = M · pre_A + c         # exact for x in row-space(W_A)

For 128×784 random Gaussian projections, donor A's L1 only saw the projection
of x onto row-space(W_A); the translator therefore loses no information donor
A's L1 ever used.

Deployment shape (Option 2 from the brief, "expose pre-ReLU codes"): the
recipient holds x at inference time, computes pre_A = W_A·x + b_A locally
from donor A's public L0, translates to pre_B, applies ReLU on the B-side,
and feeds donor B's L1. This keeps the architectural ReLU on L0 unchanged
from the chained-15 runs.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


class L0Translator:
    """Closed-form translator from donor A's pre-ReLU L0 code-space to donor B's.

    Construction is one-shot and data-free::

        t = L0Translator(W_a, b_a, W_b, b_b)
        pre_b = t.translate(pre_a)              # exact in row-space(W_a)
        z_b   = pre_b.clamp_min(0)              # B-side ReLU
        h_b   = donor_b.l1(z_b)
        logits = donor_b.head(h_b)

    The translator stores M (n_out, n_out) and c (n_out,). Pseudoinverse cost
    is one-time at construction; per-call cost is one (B, n_out) @ (n_out,
    n_out) matmul plus a vector add.
    """

    def __init__(
        self,
        W_a: torch.Tensor,
        b_a: torch.Tensor,
        W_b: torch.Tensor,
        b_b: torch.Tensor,
        column_filter: Optional[torch.Tensor] = None,
    ):
        if W_a.shape != W_b.shape:
            raise ValueError(
                f"L0 shape mismatch: W_a {tuple(W_a.shape)} vs W_b "
                f"{tuple(W_b.shape)}. Translator requires equal n_out and "
                f"equal fan_in."
            )
        if b_a.shape != b_b.shape or b_a.shape[0] != W_a.shape[0]:
            raise ValueError(
                f"Bias shape mismatch: b_a {tuple(b_a.shape)}, b_b "
                f"{tuple(b_b.shape)}, W_a out-dim {W_a.shape[0]}."
            )

        # Promote to fp32 for the pseudoinverse — torch.linalg.pinv on bf16
        # / fp16 inputs is numerically poor, and the fit-once-use-many cost
        # model doesn't reward saving that 100 ms.
        W_a32 = W_a.detach().to(torch.float32)
        W_b32 = W_b.detach().to(torch.float32)
        b_a32 = b_a.detach().to(torch.float32)
        b_b32 = b_b.detach().to(torch.float32)

        # Trioron 2.0 Phase 5 — R·S handshake restricted to standardized
        # columns. When the caller supplies a column_filter bool tensor
        # (typically the AND of both donors' standardized_column_mask()
        # outputs), the pseudoinverse is computed over only those
        # columns; long-range columns are excluded from the handshake
        # and treated as branch-private extension. Default None = use
        # all columns (backward-compat; 1.0 donors are all-standardized
        # and the result is byte-identical to the unfiltered path).
        if column_filter is not None:
            cf = column_filter.detach().to(device=W_a32.device).bool()
            if cf.ndim != 1 or cf.shape[0] != W_a32.shape[1]:
                raise ValueError(
                    f"column_filter shape {tuple(cf.shape)} != "
                    f"(fan_in={W_a32.shape[1]},)"
                )
            if not cf.any():
                raise ValueError(
                    "column_filter selects zero columns; need at least "
                    "one standardized column for the handshake."
                )
            W_a32 = W_a32[:, cf]
            W_b32 = W_b32[:, cf]
            self._column_filter: Optional[torch.Tensor] = cf
        else:
            self._column_filter = None

        W_a_pinv = torch.linalg.pinv(W_a32)              # (fan_in*, n_out)
        M = W_b32 @ W_a_pinv                              # (n_out, n_out)
        c = b_b32 - M @ b_a32                             # (n_out,)

        self.M = M
        self.c = c
        # Diagnostic slot — populated by `diagnostic_error` if the caller
        # wants a probe-based reconstruction MSE.
        self.fit_error: Optional[float] = None

    # -------- core ops --------

    def translate(self, pre_a: torch.Tensor) -> torch.Tensor:
        """Translate pre-ReLU L0 codes from donor A's basis to donor B's.

        pre_a : (B, n_out) pre-activation codes from donor A's L0.
        returns: (B, n_out) pre-activation codes in donor B's L0 basis.
        """
        M = self.M.to(device=pre_a.device, dtype=pre_a.dtype)
        c = self.c.to(device=pre_a.device, dtype=pre_a.dtype)
        return pre_a @ M.T + c

    def to(self, *, device: Optional[torch.device] = None,
           dtype: Optional[torch.dtype] = None) -> "L0Translator":
        """Move/cast the translator's M and c. Returns self."""
        if device is not None or dtype is not None:
            self.M = self.M.to(device=device, dtype=dtype)
            self.c = self.c.to(device=device, dtype=dtype)
        return self

    # -------- diagnostics --------

    def diagnostic_error(
        self,
        x_probe: torch.Tensor,
        W_a: torch.Tensor,
        b_a: torch.Tensor,
        W_b: torch.Tensor,
        b_b: torch.Tensor,
    ) -> float:
        """Reconstruction MSE on a probe set. Stores the result in
        `self.fit_error` and returns it.

        For full-row-rank Gaussian projections this should hit numerical
        precision floor: ~1e-12 in fp32, ~1e-3 in fp16. Anything larger
        means the translator construction is wrong (bug or shape mismatch),
        not a tuning issue.
        """
        device = x_probe.device
        dtype = x_probe.dtype
        with torch.no_grad():
            pre_a = F.linear(x_probe, W_a.to(device=device, dtype=dtype),
                             b_a.to(device=device, dtype=dtype))
            pre_b_true = F.linear(x_probe, W_b.to(device=device, dtype=dtype),
                                  b_b.to(device=device, dtype=dtype))
            pre_b_translated = self.translate(pre_a)
            mse = ((pre_b_true - pre_b_translated) ** 2).mean().item()
        self.fit_error = mse
        return mse

    # -------- factory: from a Branch / TrioronNetwork donor pair --------

    @classmethod
    def from_donors(cls, donor_a, donor_b) -> "L0Translator":
        """Build a translator from two objects exposing `l0_W()` and `l0_b()`.

        The trioron `Branch` class (multibranch.py) provides exactly this
        interface; passing two Branch instances is the common case::

            t = L0Translator.from_donors(branch_a, branch_b)
        """
        return cls(donor_a.l0_W(), donor_a.l0_b(),
                   donor_b.l0_W(), donor_b.l0_b())


def compose_with_translator(
    pre_a: torch.Tensor,
    translator: L0Translator,
    donor_b,
    *,
    apply_relu: bool = True,
) -> torch.Tensor:
    """Translate donor A's pre-ReLU L0 codes through donor B's L1+head.

    pre_a       : (B, n_out) pre-activation L0 codes from donor A.
    translator  : prebuilt L0Translator (A → B).
    donor_b     : object with `forward_from_l0(z)` (e.g. trioron Branch).
    apply_relu  : if True, applies ReLU on the B-side before feeding L1.
                  Set False if donor B was trained with no L0 ReLU.

    Returns donor B's logits over its trained head namespace; the caller
    pads/maps to global class indices as usual.
    """
    pre_b = translator.translate(pre_a)
    z_b = pre_b.clamp_min(0.0) if apply_relu else pre_b
    return donor_b.forward_from_l0(z_b)


# ---------------------------------------------------------------------
# Archive transfer via manifold pseudo-replay
# ---------------------------------------------------------------------


def transform_archive_to_canonical(
    donor,
    canon_W: torch.Tensor,
    canon_b: torch.Tensor,
    *,
    n_samples_per_class: int = 256,
    seed: int = 0,
    apply_relu_canonical: bool = True,
):
    """Refit a donor's per-class manifold archive into the recipient's
    canonical L0 code-space using synthetic pseudo-replay.

    Each donor branch publishes a per-class diagonal-Gaussian archive
    `(μ_B, σ_B)` over its own post-ReLU L0 code z_B = ReLU(W_B·x + b_B).
    Under cross-seed absorption, donor B's archive scores its own native
    codes well but scores codes that have been translated from a
    different canonical L0 *poorly*: the (μ, σ) was fit on the full
    distribution, but the translator's bottlenecked view is a
    lower-variance subset along the dimensions donor A's L0 captured.

    Fix: pseudo-replay the archive through donor B's published L0 inverse,
    forward synthetic samples through the recipient's canonical L0, and
    refit `(μ_C, σ_C)` per class on the canonical-space samples. The
    organism then scores canonical codes against a canonical-space
    archive — bit-exact in the linear regime, no translator residual on
    the routing path.

    Args:
        donor: an object with `.l0_W()`, `.l0_b()`, and a
            `manifold_stats: dict[int, (mu, sg)]` attribute (matches the
            trioron `Branch` interface).
        canon_W, canon_b: recipient's canonical L0 weights.
        n_samples_per_class: pseudo-replay sample count. Default 256 is
            comfortably above the per-class effective sample count of
            most archives.
        seed: RNG seed for sample reproducibility.
        apply_relu_canonical: if True, refit on `ReLU(W_C·x + b_C)`
            (matches the production canonical L0 pipeline). If False,
            refit on the linear `W_C·x + b_C` codes (Option 1 of the
            handshake brief — drops canonical ReLU).

    Returns:
        dict[int, (mu_C, sg_C)] keyed by global class id.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    # Pseudoinverse of donor's L0 — promote to fp32 for stability.
    W_B = donor.l0_W().detach().to(torch.float32)
    b_B = donor.l0_b().detach().to(torch.float32)
    W_B_pinv = torch.linalg.pinv(W_B)                       # (fan_in, n_out)

    canon_W32 = canon_W.detach().to(torch.float32)
    canon_b32 = canon_b.detach().to(torch.float32)

    out = {}
    for c, (mu_B, sg_B) in donor.manifold_stats.items():
        mu_B = mu_B.detach().to(torch.float32)
        sg_B = sg_B.detach().to(torch.float32)
        # Sample post-ReLU codes from donor B's archive.
        eps = torch.randn(n_samples_per_class, mu_B.shape[0], generator=g)
        z_B = mu_B.unsqueeze(0) + sg_B.unsqueeze(0) * eps
        # Honor the post-ReLU support: codes are non-negative.
        z_B = z_B.clamp_min(0.0)

        # Recover synthetic x via pseudoinverse:
        # z_B - b_B ≈ W_B · x  →  x ≈ W_B^+ · (z_B - b_B)
        # x_synth shape: (n_samples, fan_in)
        x_synth = (z_B - b_B) @ W_B_pinv.T

        # Forward through canonical L0.
        pre_C = x_synth @ canon_W32.T + canon_b32
        z_C = pre_C.clamp_min(0.0) if apply_relu_canonical else pre_C

        mu_C = z_C.mean(dim=0)
        sg_C = z_C.std(dim=0).clamp_min(1e-6)
        out[int(c)] = (mu_C, sg_C)

    return out
