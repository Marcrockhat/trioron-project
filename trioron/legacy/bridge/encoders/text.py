"""Text encoder backed by sentence-transformers.

Model default: all-MiniLM-L6-v2 (~80 MB on disk, 384-dim output, runs
on CPU at ~1k sentences/sec). Reasonable trade-off between size and
quality for an edge-deployment encoder.

Install:
    pip install trioron[bridge-text]
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Union

import torch


_INSTALL_HINT = (
    "Text encoder requires sentence-transformers. Install with:\n"
    "    pip install trioron[bridge-text]\n"
    "  or\n"
    "    pip install sentence-transformers"
)


class TextEncoder:
    """Wrap a sentence-transformer model as a frozen Encoder.

    Output is L2-normalized so cosine-distance and Euclidean-distance
    routings agree up to a monotone transform. ``encode_dim`` is the
    model's hidden dim (384 for the default).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(_INSTALL_HINT) from e
        self.model_name = model_name
        self.normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._model.eval()
        # sentence-transformers exposes the embedding dimension directly.
        self.encode_dim: int = int(self._model.get_sentence_embedding_dimension())

    def __call__(
        self,
        batch: Union[str, Sequence[str]],
    ) -> torch.Tensor:
        """Returns (B, encode_dim). Accepts a single string or a list."""
        single = isinstance(batch, str)
        texts: List[str] = [batch] if single else list(batch)
        with torch.no_grad():
            emb = self._model.encode(
                texts, convert_to_tensor=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )
        if not torch.is_tensor(emb):  # fallback for older versions
            emb = torch.as_tensor(emb)
        return emb.cpu()


__all__ = ["TextEncoder"]
