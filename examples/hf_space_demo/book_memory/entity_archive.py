"""Entity archive for the Book Memory demo.

A lightweight per-entity manifold archive that lets the trioron route
rare-name questions through in-context-hint forcing of the LLM, instead
of asking the LLM to traverse a 4-piece BPE corridor under soft-prompt
distillation alone.

Cosine retrieval over learned per-entity keys in pooled-LLM-embedding
space. Each entity carries its display string + the BPE pieces of that
string, ready to be spliced into an in-context hint at inference.

Design notes:
- Keys are CENTROIDS of training-question pooled embeddings whose gold
  answer auto-annotated as that entity. No dedicated training pass —
  this matches the trioron's existing per-class manifold-archive
  mechanism (μ centroid in L0 space).
- Cosine threshold τ is calibrated on the training set so that
  entity-flagged questions score >τ and non-entity questions score <τ.
  A separate NIL threshold catches the long tail of descriptive
  questions where forcing would pollute the answer.
- The archive is append-only. Adding a book = extract its entities,
  compute centroids, append rows to K + metadata. No retraining.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Auto-annotation: which Q/A pairs route through the entity archive?
# ---------------------------------------------------------------------

# Rule prioritizes precision over recall. Better to miss a few entity
# questions and let them fall through the LLM-only path than to force
# "Passepartout" into an unrelated descriptive answer.
_WORD_RE = re.compile(r"\S+")


def is_entity_answer(answer: str, tokenizer) -> bool:
    """True if `answer` looks like a rare-named entity worth forcing.

    Heuristics, all must hold:
      1. Word count ≤ 4 (entities are short, not paragraphs)
      2. Tokenizes to ≥ 2 BPE pieces (rare BPE corridor — the case where
         the LLM struggles)
      3. Contains a capitalized non-stopword (proper-noun signal)
    """
    a = (answer or "").strip()
    if not a:
        return False
    words = _WORD_RE.findall(a)
    if len(words) == 0 or len(words) > 4:
        return False
    pieces = tokenizer(a, add_special_tokens=False)["input_ids"]
    if len(pieces) < 2:
        return False
    has_caps = any(
        w[0].isupper() and w.lower() not in {"the", "a", "an", "of", "in", "on", "at"}
        for w in words
    )
    return has_caps


def extract_pairs(
    jsonl_path: Path, tokenizer,
) -> Tuple[List[dict], List[dict]]:
    """Read a Q/A jsonl, split into entity-flagged + descriptive pairs."""
    entity, descriptive = [], []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            (entity if is_entity_answer(row["a"], tokenizer)
             else descriptive).append(row)
    return entity, descriptive


# ---------------------------------------------------------------------
# Encoding helper
# ---------------------------------------------------------------------

@torch.no_grad()
def pool_question(question: str, embed_layer, tokenizer) -> torch.Tensor:
    """Mean-pool the LLM's input embeddings over question tokens.
    Same as `pool_query_embeddings` in model.py, packaged here for
    convenience during archive build."""
    prompt = f"Q: {question}\nA: "
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    emb = embed_layer(ids)
    m = mask.to(emb.dtype).unsqueeze(-1)
    summed = (emb * m).sum(dim=1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return (summed / denom).squeeze(0).float().cpu()  # (D,)


# ---------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------

@dataclass
class EntityArchive:
    """Cosine-retrieval archive over per-question keys, each tagged with
    its entity. Lookup is `argmax_k cos(z, K[k])`, return the entity of
    the winning key — same shape as a 1990s retrieval-based chatbot
    where every stored (input, response) pair is a separate searchable
    record. Paraphrase tolerance comes from there being multiple keys
    per entity, not from any single key being broad.

    K shape: (N_keys, D) — one row per stored question.
    key_to_entity[k]: index into entity_text/entity_pieces for K[k].
    """
    K: torch.Tensor                  # (N_keys, D) L2-normalized
    key_to_entity: List[int]         # len = N_keys
    entity_text: List[str]           # display strings, len = N_entities
    entity_pieces: List[List[int]]   # BPE ids, len = N_entities
    book_label: List[str]            # len = N_entities
    n_keys_per_entity: List[int]     # len = N_entities
    threshold: float = 0.95          # cosine threshold
    embed_dim: int = 0

    def __post_init__(self):
        if self.embed_dim == 0 and self.K.numel():
            self.embed_dim = int(self.K.shape[1])

    @torch.no_grad()
    def lookup(self, z: torch.Tensor) -> Optional[Tuple[int, float]]:
        """Argmax cosine over all keys. Returns (entity_id, score) of the
        winning key, or None if no key scores above threshold."""
        if self.K.numel() == 0:
            return None
        z = z.float().cpu().view(-1)
        z = F.normalize(z, dim=0)
        scores = self.K @ z                          # (N_keys,)
        s, k = scores.max(dim=0)
        s = float(s)
        if s < self.threshold:
            return None
        return self.key_to_entity[int(k)], s

    def text_of(self, ent_id: int) -> str:
        return self.entity_text[ent_id]

    def __len__(self) -> int:
        return len(self.entity_text)

    def n_keys(self) -> int:
        return int(self.K.shape[0])

    def save(self, path: Path) -> None:
        torch.save({
            "K": self.K,
            "key_to_entity": self.key_to_entity,
            "entity_text": self.entity_text,
            "entity_pieces": self.entity_pieces,
            "book_label": self.book_label,
            "n_keys_per_entity": self.n_keys_per_entity,
            "threshold": self.threshold,
            "embed_dim": self.embed_dim,
        }, path)

    @classmethod
    def load(cls, path: Path) -> "EntityArchive":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            K=d["K"],
            key_to_entity=d["key_to_entity"],
            entity_text=d["entity_text"],
            entity_pieces=d["entity_pieces"],
            book_label=d["book_label"],
            n_keys_per_entity=d["n_keys_per_entity"],
            threshold=float(d["threshold"]),
            embed_dim=int(d["embed_dim"]),
        )


# ---------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------

def _normalize_entity_key(text: str) -> str:
    """Case-insensitive dedup key for collapsing 'Passepartout' / 'passepartout'
    / 'Passepartout.' into one entity. Keeps the first-seen casing for display."""
    return re.sub(r"[.,;:!?\"']+$", "", text.strip()).lower()


def build_archive(
    book_pairs: Sequence[Tuple[str, Sequence[dict]]],
    embed_layer,
    tokenizer,
    threshold: float = 0.85,
    min_questions_per_entity: int = 1,
) -> Tuple[EntityArchive, dict]:
    """Compute per-entity centroid keys from (Q, A) supervision.

    book_pairs: sequence of (book_label, rows) where each row has
        keys "q" and "a" (other keys are ignored).

    Returns (archive, stats_dict).
    """
    grouped: Dict[str, dict] = {}  # norm_key → {text, pieces, book, vecs[]}

    for book_label, rows in book_pairs:
        for row in rows:
            if not is_entity_answer(row["a"], tokenizer):
                continue
            norm = _normalize_entity_key(row["a"])
            if norm not in grouped:
                pieces = tokenizer(row["a"], add_special_tokens=False)["input_ids"]
                grouped[norm] = {
                    "text": row["a"].strip().rstrip(".,;:!?\"'"),
                    "pieces": pieces,
                    "book": book_label,
                    "vecs": [],
                }
            v = pool_question(row["q"], embed_layer, tokenizer)
            grouped[norm]["vecs"].append(v)

    keep = [g for g in grouped.values()
            if len(g["vecs"]) >= min_questions_per_entity]

    if not keep:
        raise RuntimeError(
            "No entities passed extraction filters. Check the heuristics.")

    # Per-question keys, not centroids. Each stored Q gets its own row in
    # K, tagged with the entity it points at. Lookup returns the entity
    # of whichever stored Q the user's query is closest to.
    all_keys: List[torch.Tensor] = []
    key_to_entity: List[int] = []
    entity_text: List[str] = []
    entity_pieces: List[List[int]] = []
    book_label: List[str] = []
    n_keys_per_entity: List[int] = []
    for ent_id, g in enumerate(keep):
        entity_text.append(g["text"])
        entity_pieces.append(g["pieces"])
        book_label.append(g["book"])
        n_keys_per_entity.append(len(g["vecs"]))
        for v in g["vecs"]:
            all_keys.append(v)
            key_to_entity.append(ent_id)

    K = F.normalize(torch.stack(all_keys), dim=-1)

    archive = EntityArchive(
        K=K,
        key_to_entity=key_to_entity,
        entity_text=entity_text,
        entity_pieces=entity_pieces,
        book_label=book_label,
        n_keys_per_entity=n_keys_per_entity,
        threshold=float(threshold),
        embed_dim=int(K.shape[1]),
    )

    stats = {
        "n_entities": len(keep),
        "n_keys": int(K.shape[0]),
        "embed_dim": int(K.shape[1]),
        "keys_min": min(n_keys_per_entity),
        "keys_max": max(n_keys_per_entity),
        "keys_mean": sum(n_keys_per_entity) / len(n_keys_per_entity),
        "by_book": {
            b: sum(1 for x in book_label if x == b)
            for b in set(book_label)
        },
    }
    return archive, stats
