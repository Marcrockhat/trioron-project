# 4. Continual learning
`status: skeleton — formulas final; worked notebook to come`

## The epigenetic lock
At a task boundary, **anchor** the weights ($\hat w \leftarrow w$), accumulate importance as the saliency EMA

$$F \leftarrow \beta F + (1-\beta)\,|w \cdot g|,$$

row-sum into the per-cell lock $\lambda_i = \max(\sum_j F_{ij} + F^b_i,\ \lambda_{\min})$, and add the pull

$$\mathcal{L}_{\mathrm{lock}} = \tfrac12 \sum_i \lambda_i \Big(\sum_j (w_{ij}-\hat w_{ij})^2 + (b_i - \hat b_i)^2\Big)$$

to the next task's loss. API: `trioron.learning.epigenetic_lock` — `accumulate_saliency`, `refresh_lambda`, `anchor`, `ewc_penalty`, `set_lambda` (drive $\lambda$ from *any* signal: reward, environment, attention).

## Manifold replay
Instead of storing data, store per class $c$ a Gaussian summary of latent activity, $(\mu_c, \sigma_c)$, and rehearse by sampling

$$\tilde z \sim \mathcal{N}(\mu_c, \operatorname{diag}(\sigma_c^2))$$

during later tasks. ~30 KB replaces thousands of stored examples; the per-class $\sigma$ is load-bearing (a $\mu$-only variant measurably loses). API: `trioron.learning.manifold`.

## Two lessons the formulas don't show
- **Protect the symbol layer first.** In stacked systems the dominant forgetting channel can be *upstream drift* (a gradient-free layer re-adapting its representation), which destroys everything downstream before a single gradient step. Lock/replay applied downstream of a drifting representation is worse than nothing.
- **Locks are situational.** A lock protects fragile structure at an acquisition cost; settled, local structure often protects itself. Measure before locking everything.
