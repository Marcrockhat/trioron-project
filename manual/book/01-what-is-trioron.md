# 1. What is a trioron?
`status: draft — v1 of this chapter; formulas final, prose will grow`

## The problem: catastrophic forgetting
Train a network on task A, then task B, and its weights move to serve B — the network *forgets* A. Standard fixes either freeze parts of the network (losing plasticity), replay stored data (losing privacy/memory budget), or penalise weight motion. Trioron's answer is **structural**: let the network be a population of cells that grows where it is frustrated, locks where it has settled, and rehearses from a compressed memory of what it has seen.

## The cell and the triparametric node
The name **tri·oron** = *three coupled state variables per node*. Every cell carries:

| symbol | name | role |
|---|---|---|
| $w$ | weight | ordinary learnable edge weights (gradient descent) |
| $\lambda$ | **epigenetic lock** | a per-cell **plasticity gate**: high $\lambda$ stiffens the cell against drift, low $\lambda$ = plastic |
| $u$ | utility | an EMA of the cell's contribution, $u \leftarrow \beta u + (1-\beta)\,\mathrm{sign}(r)\,|a \cdot g|$ — drives pruning and growth triggers |

$\lambda$ is the namesake and the anti-forgetting mechanism. It is driven by an **importance signal**: trioron's native choice is the saliency $|w \cdot g|$ (weight × gradient, an OBD-style first-order term), accumulated as an EMA over training and **row-summed** into a per-cell lock:

$$\lambda_i \;=\; \max\!\Big(\;\sum_{j \in \mathrm{in}(i)} F_{ij} \;+\; F^{b}_i,\;\; \lambda_{\min}\Big)$$

where $F$ is the accumulated importance and $\lambda_{\min}$ is a floor that keeps the lock from silently zeroing. Why $|w\cdot g|$ and not Fisher $g^2$? At convergence $g \to 0$, so empirical Fisher collapses to the floor and the lock goes uniform (dead); $|w \cdot g|$ keeps the weight-magnitude factor, so large settled weights stay protected as gradients shrink.

The lock acts as a quadratic pull toward an **anchor** $\hat w$ (a snapshot taken at a task boundary):

$$\mathcal{L}_{\mathrm{lock}} \;=\; \tfrac{1}{2}\sum_i \lambda_i \Big( \sum_j (w_{ij} - \hat w_{ij})^2 + (b_i - \hat b_i)^2 \Big)$$

added to the task loss with a strength of your choosing. $\lambda$ can also be driven directly by *any* per-cell signal — reward, an environment sense, attention — which is what makes it *epigenetic* rather than a fixed statistic.

## The quad nonlinearity
Interior cells use the quadratic activation

$$\sigma(z) = z + z^2$$

which makes a stack of $k$ layers a polynomial of degree $2^k$ — depth buys expressive *degree*, and problems like $k$-bit parity need provably $\lceil \log_2 k \rceil$ layers. This gives growth a measurable reason to add depth.

## The six ideas in one paragraph
A substrate is **constructed** from a base blueprint into an *arena* (flat tensors of cells and edges — chapter 2). It **trains** like any torch module. When frustrated (loss plateau while capacity is engaged) it **grows** — cells divide, new capacity joins mid-task (chapter 3). When a region settles, credit **locks** it ($\lambda$ rises; growth never builds on unsettled ground — chapter 3). Old tasks survive through the **lock** and through **manifold replay** — per-class Gaussian summaries $(\mu_c, \sigma_c)$ of latent activity replayed instead of stored data (chapter 4). Substrates compose into **organisms** — saveable creatures with a stable API (chapter 5).

*Next:* [build one](02-first-substrate.ipynb).
