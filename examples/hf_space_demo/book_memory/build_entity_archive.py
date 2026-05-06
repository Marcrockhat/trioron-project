"""Build the entity archive from the two book Q/A jsonls.

Usage:
    python3 -m examples.hf_space_demo.book_memory.build_entity_archive
        [--threshold 0.85] [--out entity_archive.pt]

Loads the SmolLM2 tokenizer + embedding layer, extracts entity-flagged
Q/A pairs from both books, computes per-entity centroid keys in
pooled-LLM-embedding space, saves a single .pt for inference.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_HERE.parent))

from book_memory.entity_archive import build_archive  # noqa: E402

LLM_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_BOOKS = [
    ("80Days", _HERE / "qa_pairs.jsonl"),
    ("Alice",  _HERE / "alice_qa_pairs.jsonl"),
]
# Preset eval question sets are ALSO (Q, A) supervision and contribute
# their entity-flagged pairs to the archive. They were previously labelled
# "test" but the trioron should still remember any (Q, entity) it's been
# shown — there's no train/test holdout principle for an explicit
# memory archive.
DEFAULT_PRESETS = [
    ("80Days", _HERE / "questions.json"),
    ("Alice",  _HERE / "alice_questions.json"),
]
# Hand-written paraphrases per entity — merge into each centroid so the
# archive tolerates natural question rewording, not just the exact preset
# phrasing. See paraphrases.json for the data; the question list is
# unioned into the same entity's training pool.
DEFAULT_PARAPHRASES = _HERE / "paraphrases.json"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--out", default=str(_HERE / "entity_archive.pt"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    print(f"[build] loading {LLM_NAME} (tokenizer + embed layer only)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(LLM_NAME, dtype=torch.float32)
    llm.eval()
    embed_layer = llm.get_input_embeddings()
    for prm in llm.parameters():
        prm.requires_grad_(False)

    print(f"[build] loading {len(DEFAULT_BOOKS)} jsonl + "
          f"{len(DEFAULT_PRESETS)} preset files...")
    book_pairs = []
    for label, path in DEFAULT_BOOKS:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print(f"  jsonl  {label}: {len(rows)} rows")
        book_pairs.append((label, rows))
    for label, path in DEFAULT_PRESETS:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        # questions.json has shape {questions: [{question, answer, ...}]}
        rows = [{"q": q["question"], "a": q["answer"]}
                for q in data.get("questions", [])]
        print(f"  preset {label}: {len(rows)} rows")
        book_pairs.append((label, rows))

    if DEFAULT_PARAPHRASES.exists():
        pdata = json.loads(DEFAULT_PARAPHRASES.read_text())
        para_rows = []
        for entry in pdata.get("entries", []):
            entity = entry["entity"]
            for q in entry.get("questions", []):
                para_rows.append({"q": q, "a": entity})
        print(f"  paraphrases: {len(para_rows)} rows "
              f"({len(pdata.get('entries', []))} entities)")
        book_pairs.append(("paraphrase", para_rows))

    print(f"[build] extracting + centroiding...")
    archive, stats = build_archive(
        book_pairs=book_pairs,
        embed_layer=embed_layer,
        tokenizer=tok,
        threshold=args.threshold,
    )

    print(f"[build] archive: {stats['n_entities']} entities, "
          f"{stats['n_keys']} stored keys, embed_dim={stats['embed_dim']}")
    print(f"  keys per entity: min={stats['keys_min']} "
          f"max={stats['keys_max']} mean={stats['keys_mean']:.1f}")
    print(f"  by book: {stats['by_book']}")

    out_path = Path(args.out)
    archive.save(out_path)
    size_kb = out_path.stat().st_size / 1024
    print(f"[build] saved → {out_path}  ({size_kb:.1f} KB)")

    print("\n[build] sample entries:")
    for i in range(min(15, len(archive))):
        n_k = archive.n_keys_per_entity[i]
        bl = archive.book_label[i]
        txt = archive.entity_text[i]
        n_pieces = len(archive.entity_pieces[i])
        print(f"  [{bl}] {txt!r:40s}  ({n_k} keys, {n_pieces} BPE pieces)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
