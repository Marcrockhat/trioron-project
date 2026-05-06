"""Trioron-as-compressed-book-memory: head modules for the 80 Days demo.

The pipeline is:

    query text
      ─► tokenize via the small LLM's tokenizer
      ─► embed via the small LLM's frozen input-embedding layer
      ─► mean-pool over query tokens to a single (embed_dim,) vector
      ─► BookMemory  (a TrioronNetwork: embed_dim → hidden → hidden)
      ─► SoftPromptHead  (linear: hidden → n_soft_tokens × embed_dim)
      ─► continuous prefix prepended to the LLM's input embeddings
      ─► frozen LLM produces an answer conditioned on the prefix

Training: cross-entropy on the answer tokens, with grads flowing only into
BookMemory + SoftPromptHead (LLM is frozen).

The demo's pitch is that the trained head — typically O(100 KB)–O(1 MB) of
weights — replaces a vector-DB-of-chunks RAG store, while still letting a
512-token-context LLM answer questions about a 67k-word book.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from trioron.network import TrioronNetwork


class BookMemory(nn.Module):
    """Trioron-backed compressed book memory.

    Maps a pooled query embedding (e.g., mean-pooled LLM input embeddings
    over the query tokens) to a feature vector that conditions the
    SoftPromptHead.

    The internal TrioronNetwork is the substrate that absorbs the book
    during training and emits a query-specific feature at inference. Its
    EWC / dream / extension machinery is available if we want to extend
    the memory to additional books later without forgetting this one.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 32,
        n_layers: int = 2,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {n_layers}")

        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)

        specs = [(self.embed_dim, self.hidden_dim, "relu")]
        for _ in range(n_layers - 1):
            specs.append((self.hidden_dim, self.hidden_dim, "tanh"))
        self.net = TrioronNetwork(specs)

    def forward(self, query_emb: torch.Tensor) -> torch.Tensor:
        return self.net(query_emb)


class SoftPromptHead(nn.Module):
    """Project trioron features to a continuous prefix in the LLM's
    embedding space.

    Output shape: (batch, n_soft_tokens, embed_dim) — to be prepended to
    the LLM's `inputs_embeds` before the question's own embeddings.
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int,
        n_soft_tokens: int = 16,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.n_soft_tokens = int(n_soft_tokens)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)

        self.proj = nn.Linear(hidden_dim, self.n_soft_tokens * self.embed_dim)
        nn.init.normal_(self.proj.weight, std=init_scale)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        out = self.proj(hidden)
        return out.view(-1, self.n_soft_tokens, self.embed_dim)


class BookMemoryHead(nn.Module):
    """End-to-end module: pooled query embedding → soft prompt.

    Owns BookMemory + SoftPromptHead so callers can save / load /
    parameter-count the whole compressed-book-memory in one place.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 32,
        n_soft_tokens: int = 16,
        n_trioron_layers: int = 2,
    ):
        super().__init__()
        self.memory = BookMemory(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layers=n_trioron_layers,
        )
        self.head = SoftPromptHead(
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_soft_tokens=n_soft_tokens,
        )

    def forward(self, query_emb: torch.Tensor) -> torch.Tensor:
        h = self.memory(query_emb)
        return self.head(h)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def pool_query_embeddings(
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean-pool the LLM's input embeddings over query tokens.

    input_embeds: (batch, seq, embed_dim) — output of the LLM's
        `model.get_input_embeddings()(input_ids)`.
    attention_mask: (batch, seq) of 0/1 — optional; if given, padding
        tokens are excluded from the mean.

    Returns: (batch, embed_dim).
    """
    if attention_mask is None:
        return input_embeds.mean(dim=1)

    mask = attention_mask.to(input_embeds.dtype).unsqueeze(-1)
    summed = (input_embeds * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def build_head_from_ckpt(ckpt: dict, device) -> "BookMemoryHead":
    """Reconstruct a BookMemoryHead matching whatever shapes are in the saved
    state_dict. Handles bio-trained heads whose layers grew asymmetrically
    (e.g., layer 0 stays 32 wide, layer 1 grew to 57 wide).

    Used by both eval.py (load for inference) and train_bio.py (load for
    --resume training on a second book).
    """
    from trioron.network import TrioronNetwork

    cfg = ckpt["config"]
    sd = ckpt["state_dict"]

    # Infer trioron layer shapes from saved W tensors.
    layer_indices = sorted({
        int(k.split(".")[3]) for k in sd
        if k.startswith("memory.net.layers.") and k.endswith(".W")
    })
    layer_shapes = []  # list of (fan_in, n_nodes)
    for i in layer_indices:
        W = sd[f"memory.net.layers.{i}.W"]
        layer_shapes.append((W.shape[1], W.shape[0]))

    # Infer SoftPromptHead.proj input dim from saved weight.
    proj_in = sd["head.proj.weight"].shape[1]

    head = BookMemoryHead(
        embed_dim=cfg["embed_dim"],
        hidden_dim=layer_shapes[-1][1],
        n_soft_tokens=cfg["n_soft_tokens"],
        n_trioron_layers=len(layer_shapes),
    )

    activations = ["relu"] + ["tanh"] * (len(layer_shapes) - 1)
    specs = [(layer_shapes[i][0], layer_shapes[i][1], activations[i])
             for i in range(len(layer_shapes))]
    head.memory.net = TrioronNetwork(specs)
    head.memory.hidden_dim = layer_shapes[-1][1]

    if head.head.proj.in_features != proj_in:
        new_proj = nn.Linear(proj_in, head.head.proj.out_features)
        head.head.proj = new_proj
        head.head.hidden_dim = proj_in

    head = head.to(device)
    head.load_state_dict(sd)
    return head


if __name__ == "__main__":
    # Smoke test: shapes line up.
    embed_dim = 576  # SmolLM2-135M hidden_size
    head = BookMemoryHead(embed_dim=embed_dim, hidden_dim=32, n_soft_tokens=16)
    print(f"BookMemoryHead params: {head.n_parameters():,}")
    print(f"  ~{head.n_parameters() * 4 / 1024:.1f} KB at fp32")

    fake_query_emb = torch.randn(3, embed_dim)  # batch=3
    soft_prompt = head(fake_query_emb)
    assert soft_prompt.shape == (3, 16, embed_dim), soft_prompt.shape
    print(f"Output shape OK: {tuple(soft_prompt.shape)}")
