# Trioron: a growing substrate

This is the **public manual** for trioron — a neural substrate whose networks are not fixed architectures but **populations of cells** that divide, differentiate, lock, and rehearse, learning many tasks in sequence while minimising catastrophic forgetting under a hard parameter envelope.

It is written as a **textbook**: each chapter builds one idea, states the formulas, and walks a runnable example step by step. Tutorial chapters are Jupyter notebooks — download them (button at the top of the page) and run them as your scratchpad; every output shown was produced by actually executing the page.

## How to read this book
1. [What is a trioron?](book/01-what-is-trioron.md) — the cell, the triparametric node $(w, \lambda, u)$, and why forgetting is the enemy.
2. [Your first substrate](book/02-first-substrate.ipynb) — construct, train, inspect. Start here if you learn by doing.
3. [Growth](book/03-growth.md) — division, the stall→grow protocol, lock-after-settlement.
4. [Continual learning](book/04-continual-learning.md) — the epigenetic lock, manifold replay, dreaming.
5. [Organisms](book/05-organisms.md) — composing substrates into saveable creatures (Logophosis, Kinopsis).

The [API reference](reference/index.md) documents every public call in the same anatomy throughout: signature, parameters, returns, source.

## Requirements
Python ≥ 3.10, PyTorch ≥ 2.x, CPU is enough — the substrate is designed for small machines (Phase 1 contract: 50K parameters, CPU-only).
