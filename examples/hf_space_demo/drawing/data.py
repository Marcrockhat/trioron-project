"""Image preprocessing + MNIST loader for the drawing live-learn tab.

Two distinct paths:
  - `load_mnist_subset` builds a small per-class slice of MNIST, used
    once at pretrain time to fit the 5-digit donor and at extend time
    when the user reaches their teach threshold (we mix in real MNIST
    samples of the target class to stabilise the new manifold against
    the user's possibly-quirky 3 sketches).
  - `sketch_to_tensor` converts whatever the Gradio Sketchpad emits
    (RGBA dict, bare PIL, or numpy) into a single 784-dim float32
    tensor in MNIST orientation (white digit on black background, [0,1]).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


IMAGE_DIM = 28 * 28


# ---------------------------------------------------------------------
# Sketchpad → 784-d tensor
# ---------------------------------------------------------------------

def _coerce_to_pil(payload) -> Image.Image:
    """Gradio's Sketchpad returns one of:
      - a dict like {"composite": <ndarray>, "background": ..., "layers": ...}
        when `type='numpy'` and brush is enabled (Gradio 5+);
      - a bare ndarray when `type='numpy'` without layered editor;
      - a PIL Image when `type='pil'`;
      - a path string when `type='filepath'`.
    Normalise to a single PIL Image."""
    if payload is None:
        raise ValueError("empty sketchpad input")
    if isinstance(payload, dict):
        for key in ("composite", "background", "image"):
            if key in payload and payload[key] is not None:
                payload = payload[key]
                break
        else:
            raise ValueError(f"unrecognised sketchpad dict keys: {list(payload)}")
    if isinstance(payload, str) and os.path.exists(payload):
        return Image.open(payload)
    if isinstance(payload, np.ndarray):
        if payload.ndim == 2:
            return Image.fromarray(payload.astype(np.uint8), mode="L")
        if payload.ndim == 3:
            mode = {3: "RGB", 4: "RGBA"}.get(payload.shape[-1])
            if mode is None:
                raise ValueError(f"unexpected channel count: {payload.shape}")
            return Image.fromarray(payload.astype(np.uint8), mode=mode)
        raise ValueError(f"unexpected ndarray shape: {payload.shape}")
    if isinstance(payload, Image.Image):
        return payload
    raise TypeError(f"unsupported sketchpad payload type: {type(payload)}")


def sketch_to_tensor(payload, invert_if_light_bg: bool = True) -> torch.Tensor:
    """Convert a Gradio sketchpad payload to a 784-dim float32 tensor in
    MNIST orientation. Returns shape (784,) on cpu.

    MNIST is white digit on black background, [0,1]. Gradio's sketchpad
    typically gives black ink on a white canvas, so we invert by default.
    """
    pil = _coerce_to_pil(payload)
    pil = pil.convert("RGBA")
    # Composite onto pure white so transparent pixels (where the user
    # hasn't drawn) read as background.
    bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(bg, pil).convert("L")
    # MNIST-style framing: crop to the inked content's bounding box,
    # pad to square, then resize to 28x28. This mirrors the centring
    # the original MNIST authors did when digitising postal envelopes.
    if invert_if_light_bg:
        flat = ImageOps.invert(flat)
    arr = np.asarray(flat, dtype=np.uint8)
    if arr.max() > 0:
        ys, xs = np.where(arr > 30)
        if len(ys) and len(xs):
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            cropped = flat.crop((x0, y0, x1, y1))
            # Pad to square with margin so the digit doesn't touch edges.
            w, h = cropped.size
            side = max(w, h)
            pad = side // 5
            target = side + 2 * pad
            sq = Image.new("L", (target, target), 0)
            sq.paste(cropped, ((target - w) // 2, (target - h) // 2))
            flat = sq
    flat = flat.resize((28, 28), Image.LANCZOS)
    arr = np.asarray(flat, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.flatten())


# ---------------------------------------------------------------------
# MNIST subset
# ---------------------------------------------------------------------

# Cache the loaded MNIST tensors once per process. Subsetting + slicing
# is cheap; the load itself is the expensive bit (downloads ~12 MB on
# first run, then reads ~50 MB into memory).
_MNIST_TRAIN_CACHE: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


def _load_mnist_train(root: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    global _MNIST_TRAIN_CACHE
    if _MNIST_TRAIN_CACHE is not None:
        return _MNIST_TRAIN_CACHE
    from experiments.datasets import _load_split, DEFAULT_DATA_ROOT
    images, labels = _load_split("mnist", train=True,
                                 root=root or DEFAULT_DATA_ROOT)
    _MNIST_TRAIN_CACHE = (images, labels)
    return images, labels


def load_mnist_subset(
    classes: List[int],
    n_per_class: int = 200,
    train_split: float = 0.8,
    seed: int = 0,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Return a per-class dict with `X_train`, `y_train`, `X_test`,
    `y_test` tensors.

    Used by both the one-shot pretrain (donor over digits 0..4) and the
    extend path (mixing real samples into the user's small sketch buffer
    so the new manifold is stable). Labels are returned in the GLOBAL
    head space (i.e., digit 5 → class id 5)."""
    images, labels = _load_mnist_train()
    g = torch.Generator().manual_seed(seed)
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for c in classes:
        idx = (labels == c).nonzero(as_tuple=True)[0]
        if len(idx) < n_per_class:
            raise RuntimeError(
                f"only {len(idx)} samples for digit {c}; asked for {n_per_class}")
        perm = torch.randperm(len(idx), generator=g)
        idx = idx[perm[:n_per_class]]
        n_tr = max(1, int(train_split * n_per_class))
        out[c] = {
            "X_train": images[idx[:n_tr]],
            "y_train": torch.full((n_tr,), c, dtype=torch.long),
            "X_test":  images[idx[n_tr:]],
            "y_test":  torch.full((n_per_class - n_tr,), c, dtype=torch.long),
        }
    return out
