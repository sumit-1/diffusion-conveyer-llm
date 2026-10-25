---
title: Conveyor-Belt Diffusion LLM
emoji: 🥖
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
license: mit
tags:
  - diffusion-llm
  - rolling-diffusion
  - language-modeling
  - tinystories
  - pytorch
---

# Conveyor-Belt Diffusion LLM (64-Slot Dual-Channel Elastic Planning Oven)

A **28.88M-parameter hybrid Autoregressive + Diffusion Language Model** where text generation works like a **64-slot bakery conveyor belt** (inspired by *Rolling Diffusion*, *FIFO-Diffusion*, *Diffusion Forcing*, and *AR-Diffusion*):

- **Input / Output Contract at every tick $t$:**
  $$\text{Input: } \underbrace{[x_0, \dots, x_{t-1}]}_{\text{Frozen Causal KV Prefix}} + \underbrace{[d_0^{(t)}, d_1^{(t)}, \dots, d_{63}^{(t)}]}_{\text{64-Slot Conveyor \& Planning Belt}} \quad\longrightarrow\quad \text{Output: } \underbrace{x_t}_{\text{Popped Slot-0 Token}} + \underbrace{[d_0^{(t+1)}, \dots, d_{63}^{(t+1)}]}_{\text{Shifted \& Refined Belt}}$$

- **Live Demo & Pretrained Weights:**
  - **Hugging Face Live Space:** [`https://huggingface.co/spaces/sumitk/diffusion-conveyer-llm`](https://huggingface.co/spaces/sumitk/diffusion-conveyer-llm)
  - **Pretrained 24h Checkpoint (`best.pt`):** [`https://huggingface.co/sumitk/diffusion-conveyer-llm`](https://huggingface.co/sumitk/diffusion-conveyer-llm)

---

## Key Architectural Innovations

1. **Dual-Channel "Soft-Vocab + Gated-Thought" Dough (`model.py`):**
   - **Channel 1 (Lexical Anchor):** Expected vocabulary embedding $e_{\text{soft}, k} = p_k^\top E_{\text{vocab}}$ under a positional temperature schedule $\tau_k \in [0.5, 4.0]$ (soft and exploratory at the back of the belt, crisp at the front).
   - **Channel 2 (Continuous Thought Highway):** Gated residual thought state $m_k \in \mathbb{R}^d$ initialized with forget-gate bias $+1.0$, fused via $\alpha_k = k / 63$ so **$\alpha_0 = 0$ exactly at Slot 0** (guaranteeing zero distribution mismatch when $x_t$ commits into the causal KV cache).
2. **Two-Zone Human-Style Thinking & Speaking Belt:**
   - **Front 25% Articulation Stream (`Slots 0..15`):** Shifts left by 1 slot every tick ($s_k = 1.0$), polishing immediate grammar toward `Slot 0`.
   - **Mid-Belt Clause Expansion (`Slot 15`):** Synthesizes new intermediate phrase embeddings (`expand_head`) at the boundary between the paused planning zone and moving front conveyor.
   - **Back 75% Elastic Planning Horizon (`Slots 16..63`):** Spans a $2\times$ wider lookahead horizon with a learned shift/pause gate $s_k \in (0, 1)$, holding sentence-level goals while clauses form in the middle.
3. **Self-Healing Belt Recovery Training (`losses.py`):**
   - Trained with 16-step unrolled BPTT, full-sequence causal LM supervision (`prefix_lm_weight=1.0`), front-belt shift-consistency KL, DAgger-style scheduled sampling, and mid-unroll belt corruption so the belt actively repairs noisy or contradictory plans.

---

## 24-Hour Training Results on 2,000,000 Real TinyStories

Trained for **24.00 hours (`122,743 steps`)** on a single RTX 5070 Ti GPU across **2,000,000 real TinyStories** (0% synthetic templates, 4,096 compact `tiktoken` BPE vocabulary with byte fallback for zero `<UNK>`):

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **Held-Out Real Val Slot-0 Perplexity** | **`4.21`** | Down from `104.49` at step 500 (10,000 unseen real stories) |
| **Held-Out Real Val Slot-0 Top-1 Acc** | **`64.39%`** | Exact next-subword match at Conveyor Slot 0 |
| **Causal Prefix Trunk PPL / Top-1 Acc** | **`3.49` / `66.78%`** | Comparable to Microsoft Research's `TinyStories-28M/33M` (`CE 1.25`) |
| **Zero-Shot StoryCloze Commonsense** | **`100.0% (16/16)`** | 2-way physical, emotional & entity-tracking pairs (`+2.58 nats` margin) |
| **Word-Order Sensitivity KL** | **`3.674 nats`** | `KL(P(slot0 \| prompt) \|\| P(slot0 \| shuffled_prompt))` |
| **64-Slot Crystallization (Slot 63 $\to$ 0)** | **`8.19` $\to$ `0.48 nats`** | `17x` uncertainty reduction as thoughts travel from back to front |

---

## Quickstart

### 1. Install
```bash
pip install -e .
```

### 2. Launch the Interactive 64-Slot Conveyor Web Simulator
```bash
PYTHONPATH=src python -m conveyer.web_sim --port 8766
```
*(If `runs/tinystories/best.pt` is not present locally, it will automatically download the pretrained 28.88M checkpoint from `huggingface.co/sumitk/diffusion-conveyer-llm`.)*

### 3. CLI Tick-by-Tick Generation & Belt Inspection
```bash
PYTHONPATH=src python -m conveyer.infer \
  --ckpt runs/tinystories/best.pt \
  --prompt "Once upon a time, Lily and a little bunny walked into the green forest to find the golden key near the locked treasure chest. Suddenly a big lion" \
  --tokens 80 --overlap 10
```

### 4. Run the Full Benchmark Suite
```bash
PYTHONPATH=src python -m conveyer.eval --ckpt runs/tinystories/best.pt
```

### 5. Train / Resume 24-Hour Real TinyStories Run
```bash
PYTHONPATH=src python -u -m conveyer.train --config configs/tinystories.yaml --resume auto
```
