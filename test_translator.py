"""Exp 1 from paper/L0_HANDSHAKE_BRIEF.md as a unit test.

Sanity-check the closed-form L0 translator on randomly-seeded Gaussian
projections. The translator is exact for inputs in row-space(W_A); the
residual on full-rank x is the part of x donor A's L0 never saw, which
is fundamental information loss, not a translator bug.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from trioron.composition import L0Translator


def _make_l0(seed: int, n_in: int = 784, n_out: int = 128,
             dtype: torch.dtype = torch.float32):
    """Donor-style L0: Kaiming-relu init at the given seed, zero bias."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    std = (2.0 / n_in) ** 0.5
    W = torch.randn(n_out, n_in, generator=g, dtype=torch.float32) * std
    b = torch.zeros(n_out, dtype=torch.float32)
    return W.to(dtype), b.to(dtype)


def _project_onto_row_space(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Orthogonal projector P_W onto row-space(W). x: (B, n_in), W: (n_out, n_in).
    Returns (B, n_in)."""
    W_pinv = torch.linalg.pinv(W)               # (n_in, n_out)
    P = W_pinv @ W                                 # (n_in, n_in) projector
    return x @ P.T


def test_translator_exact_in_row_space():
    """For x ∈ row-space(W_A), the translator reconstructs W_B·x + b_B exactly.

    This is the formal correctness statement: the translator is the unique
    closed-form map satisfying W_B·x + b_B = M·(W_A·x + b_A) + c on
    row-space(W_A).
    """
    W_a, b_a = _make_l0(seed=42)
    W_b, b_b = _make_l0(seed=43)
    t = L0Translator(W_a, b_a, W_b, b_b)

    g = torch.Generator(device="cpu").manual_seed(7)
    # Generate inputs that lie in row-space(W_a) by setting x = W_a^T @ alpha.
    alpha = torch.randn(1000, 128, generator=g)
    x_in_row = alpha @ W_a                          # (1000, 784) in row-space

    pre_a = F.linear(x_in_row, W_a, b_a)
    pre_b_true = F.linear(x_in_row, W_b, b_b)
    pre_b_translated = t.translate(pre_a)
    mse = ((pre_b_true - pre_b_translated) ** 2).mean().item()
    assert mse < 1e-8, (
        f"fp32 in-row-space MSE {mse:.3e} exceeds 1e-8 floor — "
        "translator math is wrong"
    )
    print(f"[fp32 in-row-space] reconstruction MSE = {mse:.3e}")


def test_translator_matches_projection_on_full_x():
    """For full-rank x, the translator reconstructs W_B · P_A · x + b_B,
    where P_A is the orthogonal projector onto row-space(W_A).

    This verifies that the translator's residual against W_B·x + b_B is
    *exactly* what donor A's L0 information-bottleneck discards — no extra
    error introduced by the translator itself.
    """
    W_a, b_a = _make_l0(seed=42)
    W_b, b_b = _make_l0(seed=43)
    t = L0Translator(W_a, b_a, W_b, b_b)

    g = torch.Generator(device="cpu").manual_seed(11)
    x = torch.randn(1000, 784, generator=g)

    pre_a = F.linear(x, W_a, b_a)
    pre_b_translated = t.translate(pre_a)

    # Ground truth under information-bottleneck framing: donor B applied to
    # the projection P_A x (the only part of x donor A's L0 ever saw).
    x_proj = _project_onto_row_space(x, W_a)
    pre_b_via_proj = F.linear(x_proj, W_b, b_b)

    bottleneck_mse = ((pre_b_via_proj - pre_b_translated) ** 2).mean().item()
    assert bottleneck_mse < 1e-8, (
        f"translator vs W_B·P_A·x MSE {bottleneck_mse:.3e} exceeds 1e-8 — "
        "translator introduces error beyond donor A's bottleneck"
    )
    print(f"[fp32 vs bottleneck] MSE = {bottleneck_mse:.3e}")

    # For information: the gap to the *full* W_B·x + b_B is the unavoidable
    # info loss. Roughly (1 - 128/784) of the ||W_B x||² magnitude in expectation.
    pre_b_full = F.linear(x, W_b, b_b)
    full_mse = ((pre_b_full - pre_b_translated) ** 2).mean().item()
    print(f"[fp32 vs full]       MSE = {full_mse:.3e} "
          f"(unavoidable bottleneck loss)")


def test_translator_with_nonzero_bias():
    W_a, _ = _make_l0(seed=42)
    W_b, _ = _make_l0(seed=43)
    g = torch.Generator(device="cpu").manual_seed(11)
    b_a = torch.randn(128, generator=g) * 0.1
    b_b = torch.randn(128, generator=g) * 0.1

    t = L0Translator(W_a, b_a, W_b, b_b)
    alpha = torch.randn(500, 128, generator=g)
    x = alpha @ W_a                                 # in row-space(W_a)
    pre_a = F.linear(x, W_a, b_a)
    pre_b_true = F.linear(x, W_b, b_b)
    mse = ((pre_b_true - t.translate(pre_a)) ** 2).mean().item()
    assert mse < 1e-8, f"non-zero-bias MSE {mse:.3e} exceeds 1e-8 floor"


def test_translator_shape_mismatch_raises():
    W_a, b_a = _make_l0(seed=42)
    W_b = torch.randn(64, 784)        # wrong n_out
    b_b = torch.zeros(64)
    try:
        L0Translator(W_a, b_a, W_b, b_b)
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_translator_identity_when_seeds_match():
    """If W_a == W_b and b_a == b_b, the translator should be the identity."""
    W, b = _make_l0(seed=42)
    t = L0Translator(W, b, W, b)
    eye_err = (t.M - torch.eye(128)).abs().max().item()
    c_err = t.c.abs().max().item()
    # W·W^+ on a wide full-row-rank matrix gives I exactly up to fp32 precision.
    assert eye_err < 1e-4, f"M not identity, max abs deviation {eye_err:.3e}"
    assert c_err < 1e-5, f"c not zero, max abs {c_err:.3e}"


def test_translator_fp16_floor_documented():
    """Document fp16 numerical floor for ESP32-class deployment.

    fp16 storage of M plus fp16 matmul gives a materially larger error
    than fp32. Bound is loose; the print is the deliverable.
    """
    W_a, b_a = _make_l0(seed=42, dtype=torch.float32)
    W_b, b_b = _make_l0(seed=43, dtype=torch.float32)
    t_fp16 = L0Translator(W_a, b_a, W_b, b_b).to(dtype=torch.float16)

    g = torch.Generator(device="cpu").manual_seed(7)
    alpha = torch.randn(1000, 128, generator=g, dtype=torch.float16)
    W_a16 = W_a.to(torch.float16); b_a16 = b_a.to(torch.float16)
    W_b16 = W_b.to(torch.float16); b_b16 = b_b.to(torch.float16)
    x = alpha @ W_a16                               # in row-space(W_a) in fp16

    pre_a = F.linear(x, W_a16, b_a16)
    pre_b_true = F.linear(x, W_b16, b_b16)
    mse = ((pre_b_true - t_fp16.translate(pre_a)) ** 2).mean().item()
    assert mse < 1e-1, f"fp16 MSE unexpectedly large: {mse:.3e}"
    print(f"[fp16 in-row-space] reconstruction MSE = {mse:.3e}")


def test_subspace_factored_translator_is_lossless_on_full_x():
    """Trump-card property: when both donors share a protocol-level
    subspace S and only differ by per-donor rotation R, the closed-form
    translator is bit-exact on full-rank x ∈ ℝ^784, not just in
    row-space(W_A). The 656-dim bottleneck disappears because the
    surviving subspace is identical across donors."""
    from trioron.composition import build_factored_l0_weight, PROTOCOL_SEED

    W_a = build_factored_l0_weight(
        784, 128, donor_seed=42, protocol_seed=PROTOCOL_SEED,
    )
    W_b = build_factored_l0_weight(
        784, 128, donor_seed=43, protocol_seed=PROTOCOL_SEED,
    )
    b_a = torch.zeros(128); b_b = torch.zeros(128)
    t = L0Translator(W_a, b_a, W_b, b_b)

    g = torch.Generator(device="cpu").manual_seed(11)
    x = torch.randn(1000, 784, generator=g)
    pre_a = F.linear(x, W_a, b_a)
    pre_b_true = F.linear(x, W_b, b_b)
    pre_b_translated = t.translate(pre_a)
    mse = ((pre_b_true - pre_b_translated) ** 2).mean().item()
    # Same numerical floor as in-row-space on independent W's (fp32).
    assert mse < 1e-8, (
        f"factored translator full-rank MSE {mse:.3e} exceeds 1e-8 — "
        "subspace factorization is supposed to make the translator "
        "lossless on full-rank inputs"
    )
    print(f"[factored translator full-rank] MSE = {mse:.3e}")


if __name__ == "__main__":
    test_translator_exact_in_row_space()
    test_translator_matches_projection_on_full_x()
    test_translator_with_nonzero_bias()
    test_translator_shape_mismatch_raises()
    test_translator_identity_when_seeds_match()
    test_translator_fp16_floor_documented()
    test_subspace_factored_translator_is_lossless_on_full_x()
    print("all tests pass")
