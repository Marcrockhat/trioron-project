"""Train the trioron-backed BookMemoryHead on Around-the-World-in-80-Days Q/A.

Pipeline per training step:
  1. Tokenize "Q: <question>\nA: " (query) and "<answer><eos>" (target).
  2. Embed query tokens via the frozen LLM's input-embedding layer.
  3. Mean-pool query embeddings → (B, embed_dim).
  4. BookMemoryHead → (B, n_soft, embed_dim) continuous prefix.
  5. Concat [soft_prompt | query_embeds | answer_embeds] and forward through
     the frozen LLM with `inputs_embeds=...`.
  6. Cross-entropy on the answer-token positions only.
  7. Backprop only into BookMemoryHead.

Run from the repo root:
    python3 examples/hf_space_demo/book_memory/train.py --max-steps 50
"""

from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Allow running as a script from the repo root.
import sys
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent.parent))

from examples.hf_space_demo.book_memory.model import (  # noqa: E402
    BookMemoryHead,
    pool_query_embeddings,
)


# ---------------- data ----------------

class QADataset(Dataset):
    def __init__(self, path: Path):
        self.pairs: List[Tuple[str, str]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.pairs.append((d["q"], d["a"]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate(batch, tokenizer, max_query_len: int, max_answer_len: int):
    """Tokenize a list of (q, a) pairs into padded tensors.

    Returns a dict with:
      query_ids        (B, Lq) int
      query_mask       (B, Lq) int
      answer_ids       (B, La) int
      answer_mask      (B, La) int
    """
    queries = [f"Q: {q}\nA: " for q, _ in batch]
    answers = [f"{a}{tokenizer.eos_token}" for _, a in batch]

    q = tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_query_len,
        add_special_tokens=False,
    )
    a = tokenizer(
        answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_answer_len,
        add_special_tokens=False,
    )
    return {
        "query_ids": q["input_ids"],
        "query_mask": q["attention_mask"],
        "answer_ids": a["input_ids"],
        "answer_mask": a["attention_mask"],
    }


# ---------------- training step ----------------

def step_loss(
    batch: dict,
    head: BookMemoryHead,
    llm,
    embed_layer,
    n_soft_tokens: int,
) -> torch.Tensor:
    """Compute per-batch cross-entropy on answer tokens."""
    device = next(llm.parameters()).device
    query_ids = batch["query_ids"].to(device)
    query_mask = batch["query_mask"].to(device)
    answer_ids = batch["answer_ids"].to(device)
    answer_mask = batch["answer_mask"].to(device)

    B = query_ids.size(0)

    # Embeddings (frozen LLM input embedding lookup).
    with torch.no_grad():
        query_emb = embed_layer(query_ids)        # (B, Lq, D)
        answer_emb = embed_layer(answer_ids)      # (B, La, D)

    # Pooled query → soft prompt.
    pooled = pool_query_embeddings(query_emb, query_mask)
    soft_prompt = head(pooled)                    # (B, n_soft, D)

    # Concat: [soft | query | answer].
    inputs_embeds = torch.cat([soft_prompt, query_emb, answer_emb], dim=1)

    Lq = query_ids.size(1)
    La = answer_ids.size(1)
    soft_mask = torch.ones(B, n_soft_tokens, device=device, dtype=query_mask.dtype)
    full_mask = torch.cat([soft_mask, query_mask, answer_mask], dim=1)

    out = llm(inputs_embeds=inputs_embeds, attention_mask=full_mask)
    logits = out.logits  # (B, n_soft + Lq + La, V)

    # Predict answer tokens. Position predicting answer[t] is the prior position
    # (answer-position − 1) in the concatenated stream.
    answer_start = n_soft_tokens + Lq
    pred_logits = logits[:, answer_start - 1 : answer_start - 1 + La, :]  # (B, La, V)

    # CE only on real answer tokens.
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        answer_ids.reshape(-1),
        reduction="none",
    ).reshape(B, La)
    loss = (loss * answer_mask).sum() / answer_mask.sum().clamp(min=1.0)
    return loss


# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--qa-path", default=str(_HERE / "qa_pairs.jsonl"))
    p.add_argument("--out", default=str(_HERE / "head.pt"))
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-soft", type=int, default=16)
    p.add_argument("--n-trioron-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=0,
                   help="If > 0, cap total optimizer steps (smoke test).")
    p.add_argument("--max-query-len", type=int, default=64)
    p.add_argument("--max-answer-len", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import
    print(f"[train] loading {args.llm}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        args.llm, dtype=torch.float32,
    ).to(device)
    llm.eval()
    for p_ in llm.parameters():
        p_.requires_grad_(False)

    embed_layer = llm.get_input_embeddings()
    embed_dim = embed_layer.weight.shape[1]
    print(f"[train] embed_dim={embed_dim}")

    head = BookMemoryHead(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        n_soft_tokens=args.n_soft,
        n_trioron_layers=args.n_trioron_layers,
    ).to(device)
    n_params = head.n_parameters()
    print(f"[train] head params: {n_params:,} (~{n_params * 4 / 1024:.1f} KB fp32)")

    ds = QADataset(Path(args.qa_path))
    print(f"[train] dataset: {len(ds)} pairs from {args.qa_path}")

    def collate_fn(batch):
        return collate(batch, tokenizer, args.max_query_len, args.max_answer_len)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    step = 0
    losses: List[float] = []
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            opt.zero_grad()
            loss = step_loss(batch, head, llm, embed_layer, args.n_soft)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            step += 1
            if step % 10 == 0 or step == 1:
                window = losses[-50:]
                print(f"[train] epoch={epoch} step={step} "
                      f"loss={loss.item():.4f} "
                      f"avg50={sum(window)/len(window):.4f} "
                      f"elapsed={time.time()-t0:.1f}s")

            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": {
                "embed_dim": embed_dim,
                "hidden_dim": args.hidden_dim,
                "n_soft_tokens": args.n_soft,
                "n_trioron_layers": args.n_trioron_layers,
                "llm": args.llm,
            },
            "final_loss": losses[-1] if losses else None,
            "n_steps": step,
        },
        args.out,
    )
    print(f"[train] saved {args.out}; final_loss={losses[-1]:.4f}; total {step} steps")


if __name__ == "__main__":
    main()
