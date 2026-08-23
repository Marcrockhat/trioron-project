"""Chord sense — math-generated word spectra (s060+, Rocky).

Each symbol id is a fixed CHORD: ``k`` Gaussian partials on a ``bins``-wide
spectrum (numbers, not audio; IPA-free). A token occurrence is the chord plus
per-occurrence jitter. This is the sense a Phasecyte Link 0 discovers a word
inventory from (``trioron.organisms.logophosis.ChordLink0``). Row 0 = PAD =
silence.
"""
from __future__ import annotations

import torch


def chord_table(n_symbols: int, *, bins: int = 32, k: int = 3, seed: int = 0) -> torch.Tensor:
    """``[n_symbols, bins]`` unit-norm chord spectra; row 0 (PAD) is zero."""
    g = torch.Generator().manual_seed(seed)
    T = torch.zeros(n_symbols, bins)
    grid = torch.arange(bins).float()
    for w in range(1, n_symbols):
        lines = torch.randperm(bins, generator=g)[:k].float()
        amp = 0.5 + torch.rand(k, generator=g)
        for b, a in zip(lines, amp):          # each partial = Gaussian line, width 0.7 bin
            T[w] += a * torch.exp(-0.5 * ((grid - b) / 0.7) ** 2)
        T[w] /= T[w].norm()
    return T


def chords_of(ids: torch.Tensor, table: torch.Tensor, gen: torch.Generator,
              jitter: float = 0.05) -> torch.Tensor:
    """``[..., bins]`` jittered chord per token id; PAD rows stay zero."""
    X = table[ids]
    return X + jitter * torch.randn(X.shape, generator=gen) * (ids != 0).unsqueeze(-1)
