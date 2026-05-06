"""Biological training: chapter curriculum + frustration-triggered growth +
per-chapter dream consolidation. Uses the trioron public API only — no core
modifications.

Per pair (forward CE on answer tokens, head + EWC backward):
  • If batch loss > frustration_threshold → sediment the batch into a manifold
    buffer marked "hard" and increment a frustration counter.
  • Else → ordinary AdamW step on the head.
  • If frustration counter ≥ grow_trigger → call net.grow_layer(...), extend
    the SoftPromptHead projection if the last layer grew, rebuild optimizer.

Per chapter (after all pairs in chapter consumed):
  • Dream phase: replay the manifold buffer for `dream_steps` steps with the
    EWC penalty active, so consolidation happens against the current anchor.
  • Estimate Fisher on the chapter's pairs at the post-replay weights.
  • net.update_lambda_all() → net.anchor_all().
  • Manifold buffer optionally pruned (kept across chapters by default —
    older sediment still gets replayed during later dreams).

Usage:
    python3 examples/hf_space_demo/book_memory/train_bio.py
"""

from __future__ import annotations
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent.parent))

from examples.hf_space_demo.book_memory.model import (  # noqa: E402
    BookMemoryHead, build_head_from_ckpt, pool_query_embeddings,
)


# ---------------- data ----------------

def load_pairs_by_chapter(path: Path) -> dict:
    by_ch = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            by_ch[d["chapter"]].append((d["q"], d["a"]))
    return dict(by_ch)


def tokenize_batch(pairs, tokenizer, max_query_len: int, max_answer_len: int):
    queries = [f"Q: {q}\nA: " for q, _ in pairs]
    answers = [f"{a}{tokenizer.eos_token}" for _, a in pairs]
    q = tokenizer(queries, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_query_len, add_special_tokens=False)
    a = tokenizer(answers, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_answer_len, add_special_tokens=False)
    return {
        "query_ids": q["input_ids"], "query_mask": q["attention_mask"],
        "answer_ids": a["input_ids"], "answer_mask": a["attention_mask"],
    }


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


# ---------------- core forward ----------------

def step_ce_loss(batch, head, llm, embed_layer, n_soft) -> torch.Tensor:
    device = next(llm.parameters()).device
    qids = batch["query_ids"]
    qmask = batch["query_mask"]
    aids = batch["answer_ids"]
    amask = batch["answer_mask"]
    B = qids.size(0)
    Lq = qids.size(1)
    La = aids.size(1)

    with torch.no_grad():
        q_emb = embed_layer(qids)
        a_emb = embed_layer(aids)

    pooled = pool_query_embeddings(q_emb, qmask)
    soft = head(pooled)

    inputs_embeds = torch.cat([soft, q_emb, a_emb], dim=1)
    soft_mask = torch.ones(B, n_soft, device=device, dtype=qmask.dtype)
    full_mask = torch.cat([soft_mask, qmask, amask], dim=1)

    out = llm(inputs_embeds=inputs_embeds, attention_mask=full_mask)
    logits = out.logits

    answer_start = n_soft + Lq
    pred_logits = logits[:, answer_start - 1 : answer_start - 1 + La, :]
    loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.size(-1)),
        aids.reshape(-1), reduction="none",
    ).reshape(B, La)
    return (loss * amask).sum() / amask.sum().clamp(min=1.0)


# ---------------- growth ----------------

def grow_head_node(head: BookMemoryHead, layer_idx: int, task_idx: int):
    """Grow one node in the head's TrioronNetwork. If the grown layer is the
    last one, extend the SoftPromptHead projection's input dim by 1.

    Returns (new_node_idx, new_hidden_dim).
    """
    net = head.memory.net
    n_layers = len(net.layers)
    new_idx = net.grow_layer(layer_idx, task_idx=task_idx)

    new_hidden = head.memory.hidden_dim
    if layer_idx == n_layers - 1:
        # Last layer's output dim grew → SoftPromptHead.proj input dim must too.
        proj = head.head.proj
        old_in = proj.weight.size(1)
        new_in = old_in + 1
        device = proj.weight.device
        new_proj = nn.Linear(new_in, proj.weight.size(0), bias=True).to(device)
        with torch.no_grad():
            new_proj.weight[:, :old_in].copy_(proj.weight)
            new_proj.weight[:, old_in:].zero_()
            new_proj.bias.copy_(proj.bias)
        head.head.proj = new_proj
        head.head.hidden_dim = new_in
        head.memory.hidden_dim = new_in
        new_hidden = new_in
    return new_idx, new_hidden


# ---------------- consolidation ----------------

def consolidate(head, llm, embed_layer, fisher_batches, n_soft):
    """Estimate Fisher on `fisher_batches` at current weights, then update
    lambda and anchor."""
    net = head.memory.net
    net.reset_fisher_all()
    for batch in fisher_batches:
        for p in head.parameters():
            p.grad = None
        ce = step_ce_loss(batch, head, llm, embed_layer, n_soft)
        ce.backward()
        net.update_fisher_all()
    net.update_lambda_all()
    net.anchor_all()
    for p in head.parameters():
        p.grad = None


def dream_replay(manifold, head, llm, embed_layer, n_soft, n_steps, opt,
                 ewc_weight: float, anchored: bool):
    """Replay sedimented hard samples for n_steps. EWC penalty active iff
    `anchored` is True (i.e., we have an anchor from a previous chapter)."""
    if not manifold or n_steps <= 0:
        return None
    last = None
    for _ in range(n_steps):
        batch = random.choice(manifold)
        opt.zero_grad()
        ce = step_ce_loss(batch, head, llm, embed_layer, n_soft)
        loss = ce
        if anchored:
            loss = loss + ewc_weight * head.memory.net.ewc_penalty()
        loss.backward()
        opt.step()
        last = ce.item()
    return last


# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--qa-path", default=str(_HERE / "qa_pairs.jsonl"))
    p.add_argument("--out", default=str(_HERE / "head_bio.pt"))
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--n-soft", type=int, default=16)
    p.add_argument("--n-trioron-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-query-len", type=int, default=64)
    p.add_argument("--max-answer-len", type=int, default=128)
    p.add_argument("--ewc-weight", type=float, default=10.0)
    p.add_argument("--frustration-threshold", type=float, default=4.0,
                   help="Batch CE > this counts as 'frustrating'.")
    p.add_argument("--grow-trigger", type=int, default=3,
                   help="Frustrations needed to fire grow_layer().")
    p.add_argument("--grow-layer-idx", type=int, default=1,
                   help="Which trioron layer to grow (default last layer).")
    p.add_argument("--dream-steps", type=int, default=20,
                   help="Replay steps per chapter dream phase.")
    p.add_argument("--manifold-cap", type=int, default=200,
                   help="Max sedimented samples (FIFO).")
    p.add_argument("--max-chapters", type=int, default=0,
                   help="If > 0, stop after this many chapters (smoke test).")
    p.add_argument("--resume", default=None,
                   help="Path to head.pt to resume from. Loads weights, "
                        "EWC anchor, Fisher; offsets task_idx so new chapter "
                        "indices don't collide with the previous book's.")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bio] device={device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[bio] loading {args.llm}")
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm, dtype=torch.float32).to(device)
    llm.eval()
    for p_ in llm.parameters():
        p_.requires_grad_(False)
    embed_layer = llm.get_input_embeddings()
    embed_dim = embed_layer.weight.shape[1]
    print(f"[bio] embed_dim={embed_dim}")

    task_idx_offset = 0
    anchored = False
    if args.resume:
        print(f"[bio] resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        head = build_head_from_ckpt(ckpt, device)
        task_idx_offset = ckpt.get("stats", {}).get("n_dream_events", 0)
        anchored = True  # head was anchored at the end of the prior book
        # Restore the manifold buffer so dream replay still touches prior
        # frustrations. Without this, EWC alone fights forgetting and loses.
        prior_manifold = ckpt.get("manifold", [])
        prior_manifold = [
            {k: v.to(device) for k, v in batch.items()} for batch in prior_manifold
        ]
        print(f"[bio] resumed: hidden={head.memory.hidden_dim} "
              f"params={head.n_parameters():,} "
              f"task_idx_offset={task_idx_offset} anchored=True "
              f"prior_manifold={len(prior_manifold)}",
              flush=True)
    else:
        prior_manifold = []
        head = BookMemoryHead(
            embed_dim=embed_dim, hidden_dim=args.hidden_dim,
            n_soft_tokens=args.n_soft,
            n_trioron_layers=args.n_trioron_layers,
        ).to(device)
        print(f"[bio] head params (start): {head.n_parameters():,}")

    by_ch = load_pairs_by_chapter(Path(args.qa_path))
    chapters = sorted(by_ch.keys())
    if args.max_chapters:
        chapters = chapters[: args.max_chapters]
    print(f"[bio] chapters: {chapters[0]}–{chapters[-1]} ({len(chapters)} total)")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    manifold: List[dict] = list(prior_manifold)
    frustration_counter = 0
    n_grow_events = 0
    n_dream_events = 0
    # `anchored` and `task_idx_offset` set above based on --resume

    t0 = time.time()
    global_step = 0
    for ch_idx in chapters:
        pairs = by_ch[ch_idx][:]
        random.shuffle(pairs)
        ch_batches = []
        for i in range(0, len(pairs), args.batch_size):
            chunk = pairs[i : i + args.batch_size]
            batch = to_device(
                tokenize_batch(chunk, tokenizer, args.max_query_len, args.max_answer_len),
                device,
            )
            ch_batches.append(batch)

        ch_losses = []
        for batch in ch_batches:
            opt.zero_grad()
            ce = step_ce_loss(batch, head, llm, embed_layer, args.n_soft)
            loss = ce
            if anchored:
                loss = loss + args.ewc_weight * head.memory.net.ewc_penalty()
            loss.backward()
            opt.step()
            ch_losses.append(ce.item())
            global_step += 1

            # Frustration check.
            if ce.item() > args.frustration_threshold:
                manifold.append(batch)
                if len(manifold) > args.manifold_cap:
                    manifold.pop(0)
                frustration_counter += 1

            # Growth.
            if frustration_counter >= args.grow_trigger:
                new_idx, new_hidden = grow_head_node(
                    head, args.grow_layer_idx,
                    task_idx=ch_idx + task_idx_offset,
                )
                opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
                n_grow_events += 1
                frustration_counter = 0
                print(f"[bio] ch {ch_idx} step {global_step}: "
                      f"GROW layer={args.grow_layer_idx} new_idx={new_idx} "
                      f"hidden={new_hidden} (event #{n_grow_events}); "
                      f"head params: {head.n_parameters():,}",
                      flush=True)

        # Dream phase.
        last_dream_ce = dream_replay(
            manifold, head, llm, embed_layer, args.n_soft,
            args.dream_steps, opt, args.ewc_weight, anchored,
        )
        n_dream_events += 1

        # Consolidate.
        consolidate(head, llm, embed_layer, ch_batches, args.n_soft)
        anchored = True

        avg_ce = sum(ch_losses) / max(1, len(ch_losses))
        print(f"[bio] ch={ch_idx:2d} pairs={len(pairs):2d} "
              f"avg_ce={avg_ce:.3f} dream_ce={last_dream_ce} "
              f"manifold={len(manifold)} grows={n_grow_events} "
              f"dreams={n_dream_events} "
              f"hidden={head.memory.hidden_dim} "
              f"params={head.n_parameters():,} "
              f"elapsed={time.time()-t0:.0f}s",
              flush=True)

    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": {
                "embed_dim": embed_dim,
                "hidden_dim": head.memory.hidden_dim,  # post-growth
                "n_soft_tokens": args.n_soft,
                "n_trioron_layers": args.n_trioron_layers,
                "llm": args.llm,
            },
            "stats": {
                "n_steps": global_step,
                "n_grow_events": n_grow_events,
                "n_dream_events": n_dream_events,
                "manifold_size": len(manifold),
                "final_hidden_dim": head.memory.hidden_dim,
                "final_params": head.n_parameters(),
            },
            # Persist the frustration manifold so a future --resume can replay
            # past-book frustrations during the new book's dream phase.
            "manifold": [
                {k: v.cpu() for k, v in batch.items()} for batch in manifold
            ],
        },
        args.out,
    )
    print(f"[bio] saved {args.out}")
    print(f"[bio] final: hidden_dim={head.memory.hidden_dim} "
          f"grows={n_grow_events} dreams={n_dream_events} "
          f"params={head.n_parameters():,}")


if __name__ == "__main__":
    main()
