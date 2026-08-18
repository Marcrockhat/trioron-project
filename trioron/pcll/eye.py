"""The eye — a retina-pooled, foveated, opponent-colour receptor body
(design: docs/design/retina_phasecyte.md; spec §10.11).

A fixed sensory ORGAN in front of a Phasecyte leaf, built the way simple
organisms build perception — WITHOUT convolution (no learned weights
shared across positions). Its three translation mechanisms are nature's:

  1. genotypic replication — the same receptive-field TYPE (difference-
     of-Gaussians centre−surround, ON and OFF) is stamped at every
     retinal position: many independent linear functionals of the pixel
     sheet, identical by gene, not by shared parameters;
  2. time-multiplexing — one foveated retina is MOVED over the scene
     (fixations / saccades; ±1 px microsaccade jitter for tolerance);
  3. pooling that grows with eccentricity — fovea at full acuity,
     parafovea 2×2, periphery 4×4.

Everything here is a fixed linear map W [n_features, C·H·W] followed by
per-channel contrast normalisation and ON/OFF rectification. Nothing is
learned. Colour is opponent (Y luminance, L−M red-green, S−(L+M) blue-
yellow); chroma is sampled only in fovea + parafovea (P stream), the
periphery is luminance-only (M stream), as in the primate retina.

Two streams per fixation:
  P (parvo): fovea (1 px regions) + parafovea (2×2), channels Y/RG/BY.
  M (magno): fovea pooled 2×2, parafovea pooled 4×4, periphery 4×4,
             channel Y only.
Each stream is a ``sense`` callable: x [N, 3, H, W] (or [N, 3·H·W]) ->
features [N, D] ready to feed a PhasecyteLeaf (RECEPTOR quantisation
downstream, spec §10.2). ``Eye.positions(stream)`` returns the imposed
retinotopic (x, y, scale) per feature so a leaf can carry positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

# opponent colour: rows = [Y, RG, BY], cols = [R, G, B]
OPPONENT = torch.tensor([[0.299, 0.587, 0.114],
                         [1.0, -1.0, 0.0],
                         [-0.5, -0.5, 1.0]])
CHANNELS_P = ("Y", "RG", "BY")
CHANNELS_M = ("Y",)


@dataclass
class Region:
    """One pooled receptor: member pixels (row-major on the H×W sheet),
    surround pixels for the DoG, ring name, scale (px), centre (row, col)."""
    members: List[int]
    surround: List[int]
    ring: str
    scale: int
    centre: Tuple[float, float]


def _pixels(r0: int, r1: int, c0: int, c1: int, H: int, W: int) -> List[int]:
    out = []
    for r in range(max(r0, 0), min(r1, H)):
        for c in range(max(c0, 0), min(c1, W)):
            out.append(r * W + c)
    return out


def _blocks(r0: int, r1: int, c0: int, c1: int, s: int, H: int, W: int,
            exclude: Tuple[int, int, int, int] | None, ring: str
            ) -> List[Region]:
    """Tile [r0,r1)×[c0,c1) with s×s blocks, skipping blocks fully inside
    ``exclude`` (an inner window) — the annulus construction."""
    regs = []
    for r in range(r0, r1, s):
        for c in range(c0, c1, s):
            if exclude is not None:
                er0, er1, ec0, ec1 = exclude
                if r >= er0 and r + s <= er1 and c >= ec0 and c + s <= ec1:
                    continue
            mem = _pixels(r, r + s, c, c + s, H, W)
            if not mem:                       # outside the visual field
                continue
            sur = [p for p in _pixels(r - s, r + 2 * s, c - s, c + 2 * s, H, W)
                   if p not in set(mem)]
            regs.append(Region(mem, sur, ring, s,
                               (r + s / 2 - 0.5, c + s / 2 - 0.5)))
    return regs


def retina_layout(centre: Tuple[int, int], *, H: int = 32, W: int = 32,
                  fovea: int = 8, parafovea: int = 16, field: int = 32,
                  m_scales: Tuple[int, int, int] = (2, 4, 4)
                  ) -> Dict[str, List[Region]]:
    """Regions of one fixation at ``centre`` (row, col) for both streams.
    Windows: fovea (fovea×fovea), parafovea annulus out to
    parafovea×parafovea, periphery annulus out to field×field, all
    centred on ``centre`` and clipped to the image."""
    cr, cc = centre

    def win(size):
        return (cr - size // 2, cr - size // 2 + size,
                cc - size // 2, cc - size // 2 + size)

    fw, pw, ow = win(fovea), win(parafovea), win(field)
    P = (_blocks(*fw, 1, H, W, None, "fovea")
         + _blocks(*pw, 2, H, W, fw, "parafovea"))
    M = (_blocks(*fw, m_scales[0], H, W, None, "fovea")
         + _blocks(*pw, m_scales[1], H, W, fw, "parafovea")
         + _blocks(*ow, m_scales[2], H, W, pw, "periphery"))
    return {"P": P, "M": M}


class Eye:
    """A fixed retina at one fixation. ``sense_P`` / ``sense_M`` are the
    stream callables; ``jitter`` > 0 enables microsaccades (the fixation
    centre is displaced by a random integer offset in [-jitter, jitter]²
    per call, drawn from ``generator``)."""

    def __init__(self, centre: Tuple[int, int] = (16, 16), *,
                 H: int = 32, W: int = 32, fovea: int = 8,
                 parafovea: int = 16, field: int = 32,
                 jitter: int = 0, eps: float = 1e-3, rectify: bool = True,
                 generator: torch.Generator | None = None) -> None:
        self.centre = tuple(centre)
        self.H, self.W = H, W
        self.fovea, self.parafovea, self.field = fovea, parafovea, field
        self.jitter = jitter
        self.eps = eps
        # rectify=True: ON/OFF half-wave pairs (retina-faithful). False:
        # one SIGNED DoG receptor per region/channel — for downstream
        # quantisers that treat exact zeros as silence (Phasecyte q=0 mask,
        # spec §10.3) the signed form keeps every receptor evidential.
        self.rectify = rectify
        self.gen = generator
        self._W: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._layout: Dict[Tuple[int, int], Dict[str, List[Region]]] = {}
        # build the canonical (jitter 0) maps eagerly; jittered on demand
        self._build((0, 0))

    # ── construction ────────────────────────────────────────────────

    def _build(self, off: Tuple[int, int]) -> None:
        c = (self.centre[0] + off[0], self.centre[1] + off[1])
        lay = retina_layout(c, H=self.H, W=self.W, fovea=self.fovea,
                            parafovea=self.parafovea, field=self.field)
        self._layout[off] = lay
        n_pix = self.H * self.W
        for stream, chans in (("P", CHANNELS_P), ("M", CHANNELS_M)):
            regs = lay[stream]
            # DoG rows over the luminance/opponent sheet: [R, n_pix]
            D = torch.zeros(len(regs), n_pix)
            for i, r in enumerate(regs):
                D[i, r.members] = 1.0 / len(r.members)
                if r.surround:
                    D[i, r.surround] -= 1.0 / len(r.surround)
            # per channel: opponent mix then DoG → [n_chan·R, 3·n_pix]
            rows = []
            for ch in chans:
                mix = OPPONENT[list(("Y", "RG", "BY")).index(ch)]   # [3]
                # x is [N, 3, H, W] flattened channel-major: [R.., G.., B..]
                rows.append(torch.cat([D * mix[k] for k in range(3)], dim=1))
            self._W[(off[0], off[1], stream)] = torch.cat(rows, dim=0)

    def _offset(self) -> Tuple[int, int]:
        if self.jitter <= 0:
            return (0, 0)
        g = self.gen
        o = torch.randint(-self.jitter, self.jitter + 1, (2,), generator=g)
        off = (int(o[0]), int(o[1]))
        if (off[0], off[1], "P") not in self._W:
            self._build(off)
        return off

    # ── queries ─────────────────────────────────────────────────────

    def regions(self, stream: str) -> List[Region]:
        return self._layout[(0, 0)][stream]

    def n_features(self, stream: str) -> int:
        chans = CHANNELS_P if stream == "P" else CHANNELS_M
        return (2 if self.rectify else 1) * len(chans) * len(self.regions(stream))

    def positions(self, stream: str) -> torch.Tensor:
        """Imposed retinotopic (x, y, scale) ∈ [0,1]³ per feature, ordered
        like the feature vector: [ON: chan-major, region] then [OFF...]."""
        regs = self.regions(stream)
        chans = CHANNELS_P if stream == "P" else CHANNELS_M
        base = torch.tensor([[r.centre[1] / max(self.W - 1, 1),
                              r.centre[0] / max(self.H - 1, 1),
                              r.scale / max(self.H, self.W)] for r in regs])
        one = base.repeat(len(chans), 1)
        return torch.cat([one, one], dim=0) if self.rectify else one

    # ── the sense ───────────────────────────────────────────────────

    def _apply(self, x: torch.Tensor, stream: str) -> torch.Tensor:
        if x.dim() == 4:
            x = x.reshape(x.shape[0], -1)
        off = self._offset()
        Wm = self._W[(off[0], off[1], stream)]
        z = x @ Wm.T                                            # [N, C·R]
        n_reg = len(self._layout[off][stream])
        chans = CHANNELS_P if stream == "P" else CHANNELS_M
        z = z.view(x.shape[0], len(chans), n_reg)
        # contrast normalisation per sample per channel (divisive)
        rms = z.pow(2).mean(dim=2, keepdim=True).sqrt()
        z = z / (rms + self.eps)
        z = z.reshape(x.shape[0], -1)
        if not self.rectify:
            return z
        return torch.cat([z.clamp(min=0), (-z).clamp(min=0)], dim=1)   # ON, OFF

    def sense_P(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(x, "P")

    def sense_M(self, x: torch.Tensor) -> torch.Tensor:
        return self._apply(x, "M")

    def sense(self, x: torch.Tensor) -> torch.Tensor:
        """Both streams concatenated [P ; M] (one leaf per fixation)."""
        return torch.cat([self.sense_P(x), self.sense_M(x)], dim=1)


def fixations(n: int = 5, *, H: int = 32, W: int = 32, spread: int = 6
              ) -> List[Tuple[int, int]]:
    """Fixation centres: 1 = image centre; 5 = centre + 4 diagonal at
    ±spread; 9 = 3×3 grid at ±spread."""
    cr, cc = H // 2, W // 2
    if n == 1:
        return [(cr, cc)]
    if n == 5:
        return [(cr, cc), (cr - spread, cc - spread), (cr - spread, cc + spread),
                (cr + spread, cc - spread), (cr + spread, cc + spread)]
    if n == 9:
        return [(cr + dr, cc + dc) for dr in (-spread, 0, spread)
                for dc in (-spread, 0, spread)]
    raise ValueError("fixations: n must be 1, 5 or 9")
