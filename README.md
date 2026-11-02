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

# Conveyor-Belt Diffusion LLM (Curious Research Test: Content-Only Rolling Conveyer + Duplex-SimMT Stacking)

An **experimental ~30M-parameter hybrid Autoregressive + Rolling-Conveyer Language Model** built as a curious research test to explore what happens when a language model maintains a **multi-slot rolling "dough-ball" conveyer of upcoming content words** alongside its causal KV prefix (inspired by *Rolling Diffusion*, *FIFO-Diffusion*, *Diffusion Forcing*, and [`sumit-1/duplex-simmt-en-de`](https://github.com/sumit-1/duplex-simmt-en-de)):

- **Input / Output Contract at every tick $t$:**
  $$\text{Input: } \underbrace{[x_0, \dots, x_{t-1}]}_{\text{Duplex-Stacked Causal KV Prefix}} + \underbrace{[d_0^{(t)}, \dots, d_{4}^{(t)}]}_{\text{Last-5 Stopword/Grammar Filling}} + \underbrace{[d_5^{(t)}, \dots, d_{95}^{(t)}]}_{\text{91-Slot Content-Only Conveyer}} \quad\longrightarrow\quad \text{Output: } \underbrace{x_t}_{\text{Popped Slot-0 Token}} + \underbrace{[d_0^{(t+1)}, \dots, d_{95}^{(t+1)}]}_{\text{Shifted \& Refined Belt}}$$

- **Live Demo & Pretrained Weights:**
  - **Hugging Face Live Space:** [`https://huggingface.co/spaces/sumitk/diffusion-conveyer-llm`](https://huggingface.co/spaces/sumitk/diffusion-conveyer-llm)
  - **Pretrained Checkpoint (`best.pt`):** [`https://huggingface.co/sumitk/diffusion-conveyer-llm`](https://huggingface.co/sumitk/diffusion-conveyer-llm)

---

## What This Experiment Tests

Rather than claiming to beat standard autoregressive Transformers, this repo is an interactive testbed for inspecting how a small LLM behaves when future token planning is split into two distinct zones on a rolling belt:

1. **Content-Only Rolling Conveyer (`Slots 5..95`, 91 slots in `model.py` & `losses.py`):**
   - Stopwords (`the`, `is`, `are`, `to`, `and`, etc.) and punctuation are masked out of `Slots 5..95`, leaving only the **3,611 content words** (nouns, lexical verbs, adjectives, character names) in the 4,096 compact `tiktoken` vocabulary.
   - These slots act as "unbaked dough balls" of upcoming content concepts that roll forward (`~45%` shift speed per tick, matching the ~1.9x ratio of full tokens to content words) and self-heal when perturbed.
2. **Last-5 Stopword & Grammar Filling Zone (`Slots 0..4`, `front_conveyer_slots: 5`):**
   - As content dough balls roll across the boundary from `Slot 5` into `Slots 4 -> 3 -> 2 -> 1 -> 0`, the last 5 slots have access to the full 4,096 vocabulary and insert the grammatical "filling" (articles, prepositions, auxiliary verbs, and punctuation) right before each token pops out of `Slot 0` ($\alpha_0 = 0$).
3. **Duplex-SimMT Frame Stacking (`embed_duplex_frame` & `conveyer_to_frame`):**
   - Borrows the frame-stacking design from [`duplex-simmt-en-de`](https://github.com/sumit-1/duplex-simmt-en-de): each committed token in the prefix stream is stacked with $K=4$ content conveyer slots via `duplex_slot_proj`, and the active head of the content conveyer (`Slots 5..8`) is fused directly into `Slot 0` (`conveyer_to_frame`) so the content conveyer actively steers what gets spoken.

---

## Empirical Observations & 24-Hour Training Results (`TinyStories`)

We trained two 24-hour runs from scratch on a single laptop RTX 5070 Ti GPU across **2,000,000 real TinyStories** (`seq_len = 256`, 4,096 compact `tiktoken` BPE vocabulary):
- **Run A (64-Slot Full-Vocab Elastic Belt, 28.88M params, 122,743 steps)**
- **Run B (96-Slot Content-Only Conveyer + Last-5 Stopword Filling + Duplex Stacking, 30.47M params, 106,507 steps)**

| Metric | Run B: 96-Slot Content-Conveyer + Duplex (`best.pt`) | Run A: 64-Slot Full-Vocab Belt | Notes |
| :--- | :---: | :---: | :--- |
| **Held-Out Real Val Slot-0 Perplexity** | **`5.12`** (`5.27` full eval) | `4.21` | 10,000 unseen real TinyStories |
| **Held-Out Real Val Slot-0 Top-1 Acc** | **`61.10%`** (`68.75%` closed-loop Pos-0) | `64.39%` | Exact next-subword match at Slot 0 |
| **Duplex / Causal Prefix Trunk PPL / Acc** | **`4.56` / `61.53%`** | `3.49` / `66.78%` | Evaluated across full 256-token sequences |
| **Zero-Shot StoryCloze Commonsense** | **`93.75% (15/16)`** (`87.50%` Slot-0) | `100.0% (16/16)` | 2-way physical, emotional & entity-tracking pairs (`+2.35 nats` margin) |
| **Word-Order Sensitivity KL** | **`4.045 nats`** | `3.674 nats` | `KL(P(slot0 \| prompt) \|\| P(slot0 \| shuffled_prompt))` |
| **Open-Ended Lexical Diversity** | **`Distinct-2: 89.4%` / `Distinct-3: 98.5%`** | `Distinct-2: 86.2%` | `0.25%` 4-gram repetition rate across 6 novel prompts |

### Honest Takeaways from the Experiment
- **Warmup & Belt Depth Matter (`K=96, W_pre=16` vs `K=16, W_pre=0`):** On closed-loop story continuation, warming up the 96-slot conveyer for 16 ticks on the prompt (`W_pre=16`) boosts first-token accuracy from **`34.38%` (`K=16, W=0`) $\to$ `68.75%` (`K=96, W=16`)**.
- **Conveyer Word Injection ("Dragon Test"):** Because the model is trained with mid-unroll self-healing (`belt_corrupt_prob = 0.20`), a single out-of-context word dropped into one slot is usually "healed" away by surrounding slots. However, when a content word like `'dragon'` is persistently injected into the head of the content conveyer (`Slots 5..8`) across several ticks, the model weaves a dragon into a prompt that ended with `"Suddenly a big lion"`:
  - *Unsteered `[SAMPLE 200t]`:* `...Suddenly a big lion -> appeared and said, "What are you doing here, little bunny?" The little bunny replied, "It's a treasure chest!..."`
  - *With `'dragon'` injected into `Slots 5..8`:* `...Suddenly a big lion -> , the lion was stareled and a dragon of a dragon. The dragon was behind them and said, "You can't have one!" ... He hid in a cave and hid behind the dragon.`
- **Limitations:** Separating content words (`Slots 5..95`) from function-word filling (`Slots 0..4`) makes the conveyer harder to optimize than a standard causal LM (perplexity is ~1 point higher than a pure causal pass), and because training unrolls are $U=16$ ticks long, long rollouts (>100 tokens) benefit from periodic belt state refreshing every 16 ticks so the recurrent state does not drift.

---

## Quickstart

### 1. Install
```bash
pip install -e .
```

### 2. Launch the Interactive Conveyor Web Simulator
```bash
PYTHONPATH=src python -m conveyer.web_sim --port 8766
```
*(If `runs/tinystories/best.pt` is not present locally, it will automatically download the pretrained checkpoint from `huggingface.co/sumitk/diffusion-conveyer-llm`.)*

### 3. Standalone Pure-PyTorch Runner & Tkinter Inspector (`conveyer-runner-ui.py`)
For a self-contained single-file implementation with zero imports from `src/conveyer/` (written in the style of [`llama-runner-ui.py`](https://github.com/sumit-1/llama-runner/blob/main/llama-runner-ui.py)):
```bash
python conveyer-runner-ui.py --cli --tokens 24
```

### 4. CLI Tick-by-Tick Generation & Belt Inspection
```bash
PYTHONPATH=src python -m conveyer.infer \
  --ckpt runs/tinystories/best.pt \
  --prompt "Once upon a time, Lily and a little bunny walked into the green forest to find the golden key near the locked treasure chest. Suddenly a big lion" \
  --tokens 80 --overlap 16 --bake-steps 96
```

### 5. Run the Full Benchmark Suite
```bash
PYTHONPATH=src python -m conveyer.eval --ckpt runs/tinystories/best.pt --bake-steps 16 96 --overlaps 0 16
```
