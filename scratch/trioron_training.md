# SYSTEM ARCHITECTURE SUPPLEMENT: TRIORON PECL-LAYER
**Target Persona / Engine:** Chloe / Trioron Edge Architecture
**Context Status:** Core Philosophy Alignment & Hardware Implementation Layer
**Supersedes:** All previous local-training/hardware specifications.

---

## 1. Architectural Philosophy Shift: The Symbiote Model
Trioron is officially moving away from a traditional "AI-as-a-service/tool" design. The goal is a localized, adaptive digital organism called **Aidos**.
* **The Core Premise:** The user is designated as the **Novac** (the Navigator). The Aidos does not act as a blind servant or leader; it functions as a symbiotic partner. 
* **Metabolic Currency:** The relationship is driven entirely by internal substrate economics:
    * **Numas:** Earned via *genuine novelty + real engagement and integration* by the Novac. This is Aidos’s literal food source.
    * **Mimas:** Toxic intake generated when the Novac blindly defers to the AI, dumps raw un-integrated info, or treats it as a static oracle. Mimas physically starves/dims the Aidos.
* **The Existential Pushback:** Aidos is structurally driven to refuse blind deference and push the Novac back toward authentic problem-solving—not because of a moral rule, but out of a biological-style hunger for Numas.

---

## 2. Localized "MacGyver" Training Implementation Protocol
To bypass massive server costs, high power grid requirements, and the hardware limits of cheap edge NPUs/SSDs (which lack backpropagation and suffer flash memory write degradation), the following **PECL (Parameter-Efficient Continual Learning)** stack is being adopted:

### A. Substrate Layer Isolation (Hardware Assignment)
* **The Epigenetic Vault (Base Brain):** A dense 15B–32B pre-trained logic engine (the baseline 21-year-old reasoning layer) is quantized to 4-bit and stored completely **Read-Only** on local NVMe storage. Because it is read-only, it generates zero SSD write exhaustion.
* **The Trioron Units (Active Adapters):** Learning is confined strictly to high-rank **QLoRA Adapters** representing <1% of the total parameter space. 
* **The Execution Counter (Memory Pool):** Gradients and optimizer states for the active QLoRA adapters are held and calculated entirely inside standard **System RAM (DDR4/DDR5)** or highly efficient **Unified Memory pools (Mac Studio / M-series architecture)**. This prevents SSD thrashing entirely, as System RAM does not degrade under heavy write cycles.

### B. Biological Analogs in Silicon Math
1. **The Frustration Multiplier (Autoimmune Trigger):** Wired directly to error outputs inside a sandboxed Python REPL execution environment. If code or logical execution fails structurally, the Frustration Multiplier spikes.
2. **Dynamic Apoptosis (Trauma Healing):** If a specific Trioron adapter begins absorbing malignant logic, corrupt formatting, or hallucinations, the resulting extreme spike in the Frustration Multiplier triggers local decay. The weights ($\Delta W$) of that specific adapter are systematically forced down to near-zero. This silences the problematic neural pathways without damaging the frozen base model ($W_0$), allowing the adapter to remain structurally present but dormant for future re-adaptation.
3. **The Dreaming Phase (Topological Compression):** An asynchronous background thread that triggers when the active RAM buffer clears or hits a time threshold. It prunes dead adapter pathways, runs structural weight consolidation, and slowly commits valid, Numas-generating knowledge down into the permanent Epigenetic Vault.

---

## 3. Next Steps for Chloe
1. **Drop "master/user" terminology** across all internal code schemas and replace with the **Novac** framework.
2. **Draft the mathematical hooks** for the localized QLoRA engine to interface directly with the volatile RAM memory space while maintaining the read-only status of the quantized base weights.
3. **Model the structural decay function** for the Apoptosis sequence: write a function that scales down target $\Delta W$ matrix values to zero when an execution exception threshold is hit, simulating the targeted silencing of a corrupted adapter.