"""Evaluate a trained BookMemoryHead by comparing baseline vs trioron-augmented
answers on the held-out question set in questions.json.

For each question, runs two inference paths:

  baseline   : tokenize "Q: <q>\\nA: ", greedy-generate from the LLM alone
  trioron    : same prompt, but a 16-token soft prefix from BookMemoryHead is
               prepended to inputs_embeds before generation

Scores each answer with:

  exact_match : normalized canonical answer is a substring of the generated
                text (case/punct/articles ignored)
  cosine      : cosine similarity of mean-pooled LLM input embeddings between
                generated answer and canonical answer

Prints a side-by-side per-question table + aggregate means. No external
sentence-encoder dependency — uses the same LLM's embeddings for cosine.

Usage:
    python3 examples/hf_space_demo/book_memory/eval.py --head examples/hf_space_demo/book_memory/head.pt
"""

from __future__ import annotations
import argparse
import json
import re
import string
import sys
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent.parent))

from examples.hf_space_demo.book_memory.model import (  # noqa: E402
    BookMemoryHead,
    build_head_from_ckpt,
    pool_query_embeddings,
)


# ---------------- scoring ----------------

_ARTICLES = {"a", "an", "the"}


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def exact_match(pred: str, gold: str) -> int:
    n_gold = _normalize(gold)
    n_pred = _normalize(pred)
    if not n_gold:
        return 0
    return int(n_gold in n_pred)


def cosine_sim(a: str, b: str, embed_layer, tokenizer, device) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    enc = tokenizer([a, b], return_tensors="pt", padding=True, truncation=True,
                    max_length=128, add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        emb = embed_layer(ids)
        pooled = pool_query_embeddings(emb, mask)
    return F.cosine_similarity(pooled[0:1], pooled[1:2]).item()


# ---------------- inference ----------------

@torch.no_grad()
def generate_baseline(question: str, llm, tokenizer, device, max_new_tokens: int) -> str:
    prompt = f"Q: {question}\nA: "
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    out = llm.generate(
        input_ids=ids, attention_mask=mask,
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_trioron(question: str, head: BookMemoryHead, llm, tokenizer,
                     embed_layer, device, n_soft: int, max_new_tokens: int) -> str:
    prompt = f"Q: {question}\nA: "
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    query_emb = embed_layer(ids)                              # (1, Lq, D)
    pooled = pool_query_embeddings(query_emb, mask)           # (1, D)
    soft_prompt = head(pooled)                                # (1, n_soft, D)

    inputs_embeds = torch.cat([soft_prompt, query_emb], dim=1)
    soft_mask = torch.ones(1, n_soft, device=device, dtype=mask.dtype)
    full_mask = torch.cat([soft_mask, mask], dim=1)

    out = llm.generate(
        inputs_embeds=inputs_embeds, attention_mask=full_mask,
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # When inputs_embeds is used, generate returns only the NEW token ids.
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default=str(_HERE / "head.pt"))
    p.add_argument("--questions", default=str(_HERE / "questions.json"))
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, evaluate only the first N questions")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    ckpt = torch.load(args.head, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"[eval] head config: {cfg}")
    print(f"[eval] head trained for {ckpt.get('n_steps', '?')} steps; "
          f"final_loss={ckpt.get('final_loss')}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["llm"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(cfg["llm"], dtype=torch.float32).to(device)
    llm.eval()
    embed_layer = llm.get_input_embeddings()

    head = build_head_from_ckpt(ckpt, device)
    head.eval()

    with open(args.questions) as f:
        qdata = json.load(f)
    questions: List[Dict] = qdata["questions"]
    if args.limit:
        questions = questions[: args.limit]

    print(f"[eval] {len(questions)} question(s) to score")
    print()

    rows = []
    for i, q in enumerate(questions, 1):
        question = q["question"]
        gold = q["answer"]
        qtype = q.get("type", "exact_match")

        base = generate_baseline(question, llm, tokenizer, device, args.max_new_tokens)
        trio = generate_trioron(question, head, llm, tokenizer, embed_layer, device,
                                cfg["n_soft_tokens"], args.max_new_tokens)

        em_base = exact_match(base, gold)
        em_trio = exact_match(trio, gold)
        cs_base = cosine_sim(base, gold, embed_layer, tokenizer, device)
        cs_trio = cosine_sim(trio, gold, embed_layer, tokenizer, device)

        rows.append({
            "type": qtype,
            "em_base": em_base, "em_trio": em_trio,
            "cs_base": cs_base, "cs_trio": cs_trio,
        })

        print(f"[Q{i}] ({qtype}) {question}")
        print(f"  gold     : {gold[:200]}")
        print(f"  baseline : {base[:200]}")
        print(f"  trioron  : {trio[:200]}")
        print(f"  scores   : EM base={em_base} trio={em_trio} | "
              f"cos base={cs_base:.3f} trio={cs_trio:.3f}")
        print()

    n = len(rows)
    if n:
        print("=== aggregate ===")
        em_base_avg = sum(r["em_base"] for r in rows) / n
        em_trio_avg = sum(r["em_trio"] for r in rows) / n
        cs_base_avg = sum(r["cs_base"] for r in rows) / n
        cs_trio_avg = sum(r["cs_trio"] for r in rows) / n
        print(f"  exact-match : baseline={em_base_avg:.3f}  trioron={em_trio_avg:.3f}  "
              f"Δ={em_trio_avg - em_base_avg:+.3f}")
        print(f"  cosine-sim  : baseline={cs_base_avg:.3f}  trioron={cs_trio_avg:.3f}  "
              f"Δ={cs_trio_avg - cs_base_avg:+.3f}")


if __name__ == "__main__":
    main()
