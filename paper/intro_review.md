# Introduction review — bracketed asks (2026-05-02)

For each bracket Rocky left, I propose citations and rewritten/expanded
prose. Pull what's useful into the .docx; ignore what isn't.

---

## Ask 1 — first paragraph: "growing network ... cited couple of times (add citations…)"

Foundational citations for the growing/constructive-network lineage:

- **Cascade-Correlation:** Fahlman & Lebiere, "The Cascade-Correlation Learning Architecture," *Advances in Neural Information Processing Systems* 2 (NeurIPS 1989), 1990.
- **Upstart algorithm:** Frean, "The Upstart algorithm: a method for constructing and training feedforward neural networks," *Neural Computation* 2(2):198–209, 1990.
- **Self-Organizing Maps:** Kohonen, "Self-Organized Formation of Topologically Correct Feature Maps," *Biological Cybernetics* 43(1):59–69, 1982.
- **DEN (Dynamically Expandable Networks):** Yoon, Yang, Lee & Hwang, "Lifelong Learning with Dynamically Expandable Networks," *ICLR 2018*.
- **Progressive Networks:** Rusu et al., "Progressive Neural Networks," *arXiv:1606.04671*, 2016.
- **Net2Net (capacity-preserving growth):** Chen, Goodfellow & Shlens, "Net2Net: Accelerating learning via knowledge transfer," *ICLR 2016*.

Suggested rewrite of the opening paragraph:

> The idea of a neural network that grows to accommodate new learning is
> not new. Foundational work spans the constructive-networks line
> (Cascade-Correlation [Fahlman & Lebiere 1990], the Upstart algorithm
> [Frean 1990]), self-organizing topographic maps [Kohonen 1982], and
> more recent capacity-expanding architectures (Progressive Networks
> [Rusu et al. 2016], DEN [Yoon et al. 2018], Net2Net [Chen et al.
> 2016]). The recurring intuition is that learning-as-growth — rather
> than learning-as-allocation-within-a-fixed-budget — better matches
> what biological systems do, and our human inclination to perceive
> ourselves as having unbounded capacity for new learning is, at least,
> consistent with this view. With the rise of large language models,
> interest in adaptive systems has surged, but most LLMs are trained
> with massive central-resource budgets, which limits the practicality
> of tailoring such models to individual users.

---

## Ask 2 — DEN/CNN/SOM paragraph + paraphrasing the resource-aware sentence

Citations: same Yoon, Fahlman & Lebiere, Kohonen as above. Add:

- **DEN limitations review:** if you want a critical citation, Aljundi et al., "Memory Aware Synapses: Learning what (not) to forget," *ECCV 2018*, also discusses capacity-limit concerns in expandable architectures.

Suggested rewrite (paraphrases your "not resource aware" sentence):

> Capacity-expanding architectures such as DEN [Yoon et al. 2018], the
> constructive-network family [Fahlman & Lebiere 1990], and self-
> organizing maps [Kohonen 1982] each address a piece of the lifelong-
> learning problem, but share a common gap: they grow on demand without
> an explicit cost model for the growth itself. When deployed under
> hardware constraints — bounded RAM, finite compute, energy budget —
> there is no built-in mechanism for the network to *decline* a
> proposed expansion that would exceed available resources, nor any
> graceful degradation pathway when resources become scarce. As a
> result, learning capacity in these architectures is implicitly
> assumed to be supplied by the deployment environment rather than
> negotiated with it. Addressing this gap is one of the motivations
> for the present work.

---

## Ask 3 — PackNet/HAT paragraph + "confined to matrices sizes and adaptive plasticity"

Full citation list to swap in (keeps the comparison sharper):

- **EWC:** Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," *PNAS* 114(13):3521–3526, 2017.
- **Online EWC / Progress & Compress:** Schwarz, Czarnecki, Luketina, Grabska-Barwińska, Teh, Pascanu & Hadsell, "Progress & Compress: A scalable framework for continual learning," *ICML 2018*.
- **PackNet:** Mallya & Lazebnik, "PackNet: Adding multiple tasks to a single network by iterative pruning," *CVPR 2018*.
- **HAT:** Serra, Suris, Miron & Karatzoglou, "Overcoming Catastrophic Forgetting with Hard Attention to the Task," *ICML 2018*.
- **Synaptic Intelligence:** Zenke, Poole & Ganguli, "Continual Learning Through Synaptic Intelligence," *ICML 2017*.
- **Memory Aware Synapses:** Aljundi, Babiloni, Elhoseiny, Rohrbach & Tuytelaars, "Memory Aware Synapses: Learning what (not) to forget," *ECCV 2018*.
- **Gradient Episodic Memory (GEM):** Lopez-Paz & Ranzato, "Gradient Episodic Memory for Continual Learning," *NeurIPS 2017*.

The 2-3 structured sentences you asked for, replacing the "confined to
their matrices sizes and their adaptive plasticity" placeholder:

> These methods share a common architectural commitment: they operate
> within a fixed parameter envelope and rely on different strategies
> to partition that envelope across tasks. PackNet uses iterative
> weight pruning followed by mask-based isolation of per-task
> sub-networks; HAT learns hard binary attention masks gating each
> layer per task; EWC and its online variants apply quadratic
> anchoring to weights estimated as important for prior tasks. While
> each is effective at moderate task counts, all of them face the same
> saturation regime: when the envelope fills, new tasks can be
> accommodated only by overwriting prior knowledge, by routing through
> increasingly compromised mask intersections, or by refusing to learn
> further. The shared limitation is structural — the envelope itself
> is fixed at design time. Biological systems, by contrast, operate
> under hard physical bounds (e.g., finite cranial volume) yet
> continue to learn throughout life, suggesting that growth, pruning
> and remodeling — rather than allocation alone — are part of the
> answer.

---

## Ask 4 — epigenetics paragraph + small-RNA expansion

Citations:

- **Epigenetics general / chromatin marks:** Allis & Jenuwein, "The molecular hallmarks of epigenetic control," *Nature Reviews Genetics* 17(8):487–500, 2016.
- **DNA methylation in development:** Smith & Meissner, "DNA methylation: roles in mammalian development," *Nature Reviews Genetics* 14(3):204–220, 2013.
- **Histone modifications:** Bannister & Kouzarides, "Regulation of chromatin by histone modifications," *Cell Research* 21(3):381–395, 2011.
- **Morphogen gradients (positional information):** Wolpert, "Positional information and the spatial pattern of cellular differentiation," *Journal of Theoretical Biology* 25(1):1–47, 1969.
- **microRNAs:** Bartel, "MicroRNAs: Target Recognition and Regulatory Functions," *Cell* 136(2):215–233, 2009.
- **miRNAs in synaptic plasticity:** Schratt, "microRNAs at the synapse," *Nature Reviews Neuroscience* 10(12):842–849, 2009.
- **miRNAs in sleep / synaptic homeostasis:** Davis et al., "MicroRNA-138 is a long-term memory regulator," and related Tononi-group work — if you want the sleep-cycle angle specifically, cite **Tononi & Cirelli, "Sleep and the Price of Plasticity," *Neuron* 81(1):12–34, 2014**.

Suggested expanded paragraph:

> During development, individual cells establish their identity through
> epigenetic mechanisms driven by spatial gradients of signaling
> molecules — *positional information* in Wolpert's classical
> formulation [Wolpert 1969]. As cells divide, epigenetic marks such as
> DNA methylation [Smith & Meissner 2013] and histone modifications
> [Bannister & Kouzarides 2011] act as cellular memory: they record
> which genes are accessible for transcription and which are silenced,
> and these marks are inherited across mitosis [Allis & Jenuwein 2016].
> Small non-coding RNAs (microRNAs) act as a separate, faster
> regulatory layer [Bartel 2009]: produced and consumed in
> stoichiometric pools, they post-transcriptionally regulate
> plasticity-related gene expression in concentration-gradient-
> dependent ways [Schratt 2009], and have been implicated in the
> sleep-driven synaptic-homeostasis cycle that consolidates new
> learning while pruning weak connections [Tononi & Cirelli 2014].
> The architecture we present in §3 borrows three of these
> mechanisms — anchoring (methylation), stress-responsive plasticity
> modulation (HPA-axis-mediated epigenetic changes), and resource-
> limited consolidation pools (small-RNA-style throttling) — as
> design metaphors that motivated specific implementation choices.

---

## Ask 5 — DishBrain / brain-cells-play-Doom citation

The paper Rocky is referring to:

- **Kagan et al. 2022** — "In vitro neurons learn and exhibit sentience when embodied in a simulated game-world," *Neuron* 110(23):3952–3969.e8.

Note: the popular framing is "brain cells play Pong" (the actual game in the paper). If your draft says "play Doom," correct to "play Pong" — Pong is what was actually used; no Doom. The architectural inspiration carries either way.

Suggested rewrite:

> Recent findings have demonstrated that biological neurons cultured
> in vitro can be programmed to recognize task-relevant computational
> signals, learning to play simple video games (Pong) under a
> reward-and-penalty stimulation protocol [Kagan et al. 2022]. This
> result indicates that biological neuronal substrates implement
> learning machinery whose principles can be operationalized in
> silicon — the cell's own response to coordinated reward / penalty
> signaling produces task-directed behavior without explicit gradient
> descent. We take this as evidence that adopting cellular-level
> mechanisms (rather than only synapse-level mechanisms) is a viable
> design direction for artificial systems.

---

## Ask 6 — architecture intro paragraph (light polish only)

Your version is fine. A small tightening:

> Inspired by how cells differentiate and develop under epigenetic
> control, we propose a tri-parametric node — the *trioron* — and an
> architecture that aggregates trioron nodes into networks capable of
> autonomous expansion and continual learning under bounded resources.
> The architecture is targeted at the regime where computational and
> memory budgets are constrained but the requirement for continual
> adaptation is high: agentic AI applications, IoT and embedded
> systems, and personal devices that must grow alongside their user
> rather than be replaced when the user's needs evolve.

---

## Ask 7 — sleepy paragraph on adaptation

You wrote:

> "In such environment this model can adapt according to the
> environment (I think I'm getting sleepy here, please help me to
> expand this sections). By learning and adapting, the model can
> increase speed and improve interactions, not only that, it can learn
> to interact with other device when the environment is connected in a
> single network with human user as the driver (is this the correct
> words to describe this?)."

The phrasing you reached for is *user-centric continual personalization
across a federated device network*. That's the technical term. For an
introduction, plain language is better. Suggested expansion:

> In such environments, the model specializes to its user's specific
> context rather than serving as a one-size-fits-all generic system.
> Through continual interaction it acquires representations that
> reflect individual usage patterns — recurring vocabulary,
> environmental cues, characteristic tasks — and integrates these into
> its growing internal structure without requiring centralized
> retraining. Adaptation translates directly into performance: as the
> internal representations of the user's domain become more accurate,
> response latency drops and interaction quality improves. When such
> devices are deployed across a connected environment — multiple
> personal endpoints sharing a household, vehicle or workplace — each
> unit grows its own model of its user and immediate context, while
> coordination between units allows shared experiences to propagate
> where the user permits. The user remains the principal driver of
> the device's development; the device becomes a personal computational
> artifact rather than an interchangeable service endpoint.

(On "is this the correct words to describe this?" — the closest
academic terms are "federated continual learning" and "user-centric
personalization." The phrasing above avoids the jargon while preserving
the meaning.)

---

## Ask 8 — co-authorship disclosure

Two notes before the prose:

1. *"personafied"* is a typo for **"personified."**
2. The Claude version is **Claude Opus 4.7 (1M-context)** rather than
   "4.7.1" — the suffix you used isn't an actual version. The longer
   form is fine if you want to be precise; "Claude Opus 4.7" alone is
   also correct.

Citations for human-AI authorship discussion:

- **Nature editorial 2023:** "Tools such as ChatGPT threaten
  transparent science; here are our ground rules for their use,"
  *Nature* 613:612, 2023. doi:10.1038/d41586-023-00191-1.
- **WAME guidance:** Zielinski et al., "Chatbots, Generative AI, and
  Scholarly Manuscripts: WAME Recommendations on Chatbots and
  Generative Artificial Intelligence in Relation to Scholarly
  Publications," *World Association of Medical Editors*, 2023.
- **ICMJE guidance** (International Committee of Medical Journal
  Editors), updated guidance on AI in authorship, 2023–2024.
- If you want a research-collaboration framing rather than a
  policy-disclosure framing: Eloundou, Manning, Mishkin & Rock,
  "GPTs are GPTs: An early look at the labor market impact potential
  of large language models," *arXiv:2303.10130*, 2023.

Suggested rewrite of the co-authorship paragraph:

> This work was carried out in collaboration with two personified AI
> assistants: *Gemma* (a Gemini Pro instance acting in an engineering
> role) and *Chloe* (a Claude Opus 4.7 1M-context instance acting in
> an academic-advisory role). The collaboration was structured as
> iterative design dialogue — human-led problem framing and final
> decision-making, AI-supported implementation, analysis and writing
> — with the human author holding sole responsibility for all claims,
> methodological choices, and interpretations. We disclose this
> collaboration following recent editorial guidance from *Nature*
> [2023] and the World Association of Medical Editors [Zielinski et
> al. 2023], which permit AI-assisted contributions but explicitly
> exclude AI systems from being listed as authors of record. We
> believe that such collaboration patterns will become increasingly
> common in small research teams that lack the engineering bandwidth
> of large institutional groups, and that transparent disclosure of
> how the collaboration is structured allows readers to judge both
> the technical contributions and the boundaries of human
> responsibility.

---

## Smaller polish notes (no rewrites unless you want them)

- **First abstract sentence**, original: "This paper describes a novel architecture of neural networks that not only spawn new nodes but actively pruning its axis." Tighter: "We describe a neural-network architecture that grows new nodes and prunes existing ones while maintaining low forgetting on the EWC family's competitive scale, designed for deployment in resource-constrained systems." (your call — preserves meaning, fixes "spawn... but actively pruning" tense mismatch.)
- **"the regression of computational cost"** in your architecture-intro paragraph reads ambiguously (regression usually means decline, but you might mean reducing cost). If you mean *reducing* compute cost, use "the reduction of computational cost" or "shrinking compute budgets."
- **"hurdled"** — "addressed" or "overcome" reads cleaner.

---

## What I'd add to the introduction that you didn't include

Two things worth a sentence each, as paper-defending pre-emption:

1. **A sentence acknowledging the small-scale benchmark.** Reviewers will
   ask "where's MNIST/CIFAR continual?" Get out ahead of it: something
   like "We focus on a small-scale contrastive-pair benchmark to expose
   the architecture's growth-and-consolidation dynamics directly; scaling
   to standard image-continual benchmarks is reserved for future work."

2. **A sentence clarifying the comparison frame.** Memory has us at
   parity-with-Online-EWC on properly-tuned baselines. The defensible
   pitch isn't "we beat all CL methods" — it's "we exhibit CL-family
   performance under bounded-resource constraints that other methods
   don't address." Worth being explicit so reviewers don't misread the
   result-table comparisons.

---

End of review. Send the next round when you have it.
