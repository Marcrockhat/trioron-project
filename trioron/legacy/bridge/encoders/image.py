"""Image encoder backed by open-clip-torch.

Model default: ViT-B-32 / openai (~600 MB on disk, 512-dim image
embedding). Pre-trained CLIP-style joint image-text embedding; we use
only the image tower.

Install:
    pip install trioron[bridge-image]
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Union

import torch


_INSTALL_HINT = (
    "Image encoder requires open-clip-torch and Pillow. Install with:\n"
    "    pip install trioron[bridge-image]\n"
    "  or\n"
    "    pip install open-clip-torch Pillow"
)


class ImageEncoder:
    """Wrap an open-clip image tower as a frozen Encoder.

    Output is L2-normalized (CLIP convention). ``encode_dim`` is the
    model's image-embedding dim (512 for ViT-B-32, 768 for ViT-L-14).
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        try:
            import open_clip
            from PIL import Image  # noqa: F401  (just to validate install)
        except ImportError as e:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from e
        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize
        self.device = device or "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        model = model.to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        self._model = model
        self._preprocess = preprocess
        # Probe encode_dim by a dry forward.
        with torch.no_grad():
            dummy = self._preprocess(self._dummy_image()).unsqueeze(0).to(self.device)
            feat = self._model.encode_image(dummy)
        self.encode_dim: int = int(feat.shape[-1])

    @staticmethod
    def _dummy_image():
        from PIL import Image
        return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def __call__(
        self,
        batch,
    ) -> torch.Tensor:
        """Accepts a PIL.Image, a path string, or a sequence of either.
        Returns (B, encode_dim)."""
        from PIL import Image
        single = not isinstance(batch, (list, tuple))
        items = [batch] if single else list(batch)
        imgs = []
        for it in items:
            if isinstance(it, str):
                img = Image.open(it).convert("RGB")
            else:
                img = it.convert("RGB") if it.mode != "RGB" else it
            imgs.append(self._preprocess(img))
        x = torch.stack(imgs, dim=0).to(self.device)
        with torch.no_grad():
            feat = self._model.encode_image(x)
            if self.normalize:
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feat.cpu()


__all__ = ["ImageEncoder"]
