"""
conveyer-runner-ui.py — Self-Contained Pure PyTorch Runner & GUI for Conveyor-Belt Diffusion LLM

Inspired by https://github.com/sumit-1/llama-runner/blob/main/llama-runner-ui.py
Zero imports from `src/conveyer/`: every tensor operation, RoPE rotation, attention head,
SwiGLU FFN, Dual-Channel Dough fusion, and Elastic 64-Slot Conveyor Shift is written out
explicitly against the raw `state_dict` weight dictionary (`best.pt`).

Usage:
  python conveyer-runner-ui.py                  # Launches the Tkinter GUI
  python conveyer-runner-ui.py --cli            # Runs a step-by-step trace in the terminal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

import tiktoken
import torch
import torch.nn.functional as F

# Special token IDs in the compact 4096 vocabulary
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3


# ==============================================================================
# 1. COMPACT TIKTOKEN BPE TOKENIZER (4096 subwords + 256 byte fallback)
# ==============================================================================
class CompactTokenizer:
    """Maps OpenAI GPT-2 tiktoken BPE tokens into the model's 4096 compact vocabulary."""

    def __init__(self, tok_json_path: str) -> None:
        data = json.loads(Path(tok_json_path).read_text(encoding="utf-8"))
        self.compact_to_gpt2: list[int] = data["compact_to_gpt2"]
        self.gpt2_to_compact: dict[int, int] = {
            gid: cid for cid, gid in enumerate(self.compact_to_gpt2) if gid >= 0
        }
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = len(self.compact_to_gpt2)

        # Byte fallback so any UTF-8 character is 100% representable without <UNK>
        self.byte_to_compact: dict[int, int] = {}
        for b in range(256):
            gid = self.enc.encode_single_token(bytes([b]))
            if gid in self.gpt2_to_compact:
                self.byte_to_compact[b] = self.gpt2_to_compact[gid]

    def encode(self, text: str) -> torch.Tensor:
        """Encodes text into [BOS_TOKEN_ID, ...compact_token_ids...] as a 1D long tensor."""
        ids = [BOS_TOKEN_ID]
        for gid in self.enc.encode_ordinary(text):
            cid = self.gpt2_to_compact.get(gid)
            if cid is not None:
                ids.append(cid)
            else:
                for b in self.enc.decode_single_token_bytes(gid):
                    ids.append(self.byte_to_compact.get(b, UNK_TOKEN_ID))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of compact token IDs back into a string."""
        gids = []
        for cid in ids:
            if cid in (PAD_TOKEN_ID, BOS_TOKEN_ID, UNK_TOKEN_ID):
                continue
            if cid == EOS_TOKEN_ID:
                break
            if 0 <= cid < len(self.compact_to_gpt2):
                gid = self.compact_to_gpt2[cid]
                if gid >= 0:
                    gids.append(gid)
        return self.enc.decode(gids)

    def id_to_piece(self, cid: int) -> str:
        if cid == PAD_TOKEN_ID:
            return "<PAD>"
        if cid == BOS_TOKEN_ID:
            return "<BOS>"
        if cid == EOS_TOKEN_ID:
            return "<EOS>"
        if cid == UNK_TOKEN_ID or not (0 <= cid < len(self.compact_to_gpt2)):
            return "<UNK>"
        gid = self.compact_to_gpt2[cid]
        return self.enc.decode_single_token_bytes(gid).decode("utf-8", errors="replace")


# ==============================================================================
# 2. ATTENTION & TRANSFORMER MATH HELPERS (Pure PyTorch Tensor Operations)
# ==============================================================================
class ConveyerMathHelper:
    """Explicit tensor math for RMSNorm, Half-Split RoPE, and SwiGLU FFN."""

    @staticmethod
    def rms_norm(tensor: torch.Tensor, norm_weights: torch.Tensor, norm_eps: float = 1e-6) -> torch.Tensor:
        """
        Applies Root Mean Square Normalization:
          x_norm = x * rsqrt(mean(x^2) + eps) * weight
        tensor: (seq_len, d_model)
        """
        squared_mean = tensor.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = tensor.float() * torch.rsqrt(squared_mean + norm_eps)
        return (normalized * norm_weights).type_as(tensor)

    @staticmethod
    def calculate_inv_freq(head_dim: int, rope_theta: float = 10000.0, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """
        Precomputes inverse frequencies for Rotary Position Embedding (RoPE):
          inv_freq[i] = 1.0 / (theta ** (2i / head_dim))   for i in 0 .. (head_dim/2 - 1)
        """
        even_indices = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        return 1.0 / (rope_theta ** (even_indices / head_dim))

    @staticmethod
    def apply_rope(tensor_per_head: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        """
        Applies Rotary Position Embedding (RoPE) to a single head's Q or K tensor.
          tensor_per_head: (seq_len, head_dim)
          positions:       (seq_len,) integer sequence positions for each token/slot
          inv_freq:        (head_dim // 2,)

        Note: Unlike Llama's adjacent-pair complex view, Conveyer splits the head dimension
        into first-half `x1 = x[:, :32]` and second-half `x2 = x[:, 32:]`:
          out1 = x1 * cos - x2 * sin
          out2 = x1 * sin + x2 * cos
        """
        freqs = torch.outer(positions.float(), inv_freq)  # (seq_len, head_dim // 2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        half_d = tensor_per_head.shape[-1] // 2
        x1 = tensor_per_head[:, :half_d].float()
        x2 = tensor_per_head[:, half_d:].float()

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.cat([out1, out2], dim=-1).type_as(tensor_per_head)

    @staticmethod
    def swiglu_ffn(normalized_embedding: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU Feed-Forward Network:
          FFN(x) = (SiLU(x @ W_gate^T) * (x @ W_up^T)) @ W_down^T
        Shapes in TinyStories checkpoint:
          w_gate: (1408, 512), w_up: (1408, 512), w_down: (512, 1408)
        """
        gate_act = F.silu(torch.matmul(normalized_embedding, w_gate.T))
        up_proj = torch.matmul(normalized_embedding, w_up.T)
        return torch.matmul(gate_act * up_proj, w_down.T)


# ==============================================================================
# 3. EXPLICIT HEAD-BY-HEAD ATTENTION LAYER (Llama-Runner Style)
# ==============================================================================
class ConveyerAttentionLayer(torch.nn.Module):
    """
    Executes one Transformer layer's Multi-Head Attention head-by-head so you can
    inspect how the 64 Conveyor Belt slots attend to BOTH:
      1. The frozen causal Prefix KV Cache (already-emitted tokens x_{0..t-1})
      2. All 64 active slots on the Conveyor Belt (bidirectional belt planning!)
    """

    def __init__(self, layer: int, weights: dict[str, torch.Tensor], n_heads: int, d_model: int, inv_freq: torch.Tensor) -> None:
        super().__init__()
        self.layer = layer
        self.weights = weights
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.inv_freq = inv_freq

    def forward(
        self,
        current_embedding: torch.Tensor,
        positions: torch.Tensor,
        prefix_kv_for_layer: tuple[torch.Tensor, torch.Tensor] | None = None,
        is_causal_prefill: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        current_embedding:   (L, d_model) — either prefix tokens during prefill OR the 64 belt slots
        positions:           (L,) integer positions (for belt slots: `tick + target_offsets`)
        prefix_kv_for_layer: optional (K_cache, V_cache) each of shape (n_heads, P, head_dim)
        is_causal_prefill:   True when pre-filling prefix tokens (causal mask),
                             False when running the 64-slot belt (bidirectional across all 64 slots!)
        """
        L = current_embedding.shape[0]

        # 1. Pre-Attention RMSNorm
        norm_weight = self.weights[f"blocks.{self.layer}.norm1.weight"]
        normed_x = ConveyerMathHelper.rms_norm(current_embedding, norm_weight)

        # 2. Split fused QKV weight (3 * d_model, d_model) -> (1536, 512) into Q, K, V
        qkv_weight = self.weights[f"blocks.{self.layer}.attn.qkv.weight"]
        wq, wk, wv = qkv_weight.chunk(3, dim=0)  # each is (512, 512)

        # Reshape into per-head slices: (n_heads=8, head_dim=64, d_model=512)
        wq_heads = wq.view(self.n_heads, self.head_dim, self.d_model)
        wk_heads = wk.view(self.n_heads, self.head_dim, self.d_model)
        wv_heads = wv.view(self.n_heads, self.head_dim, self.d_model)
        wo = self.weights[f"blocks.{self.layer}.attn.proj.weight"]  # (512, 512)

        qkv_attention_store = []
        new_k_per_head = []
        new_v_per_head = []

        # 3. Iterate through each of the 8 attention heads explicitly
        for head in range(self.n_heads):
            q_head = torch.matmul(normed_x, wq_heads[head].T)  # (L, 64)
            k_head = torch.matmul(normed_x, wk_heads[head].T)  # (L, 64)
            v_head = torch.matmul(normed_x, wv_heads[head].T)  # (L, 64)

            # Apply RoPE using the exact sequence positions (tick + offset for belt slots)
            q_rot = ConveyerMathHelper.apply_rope(q_head, positions, self.inv_freq)
            k_rot = ConveyerMathHelper.apply_rope(k_head, positions, self.inv_freq)

            new_k_per_head.append(k_rot)
            new_v_per_head.append(v_head)

            # Prepend frozen Prefix KV cache if present: K becomes (P + L, 64)
            if prefix_kv_for_layer is not None:
                k_total = torch.cat([prefix_kv_for_layer[0][head], k_rot], dim=0)
                v_total = torch.cat([prefix_kv_for_layer[1][head], v_head], dim=0)
            else:
                k_total = k_rot
                v_total = v_head

            # Scaled Dot-Product Attention: (L, P + L)
            scores = torch.matmul(q_rot, k_total.T) / (self.head_dim ** 0.5)

            # During prefix prefill, apply upper-triangular causal mask.
            # During 64-slot belt forward pass (`is_causal_prefill=False`), NO mask is applied:
            # every belt slot 0..63 can see the entire prefix AND all 64 belt slots bidirectionally!
            if is_causal_prefill and L > 1:
                P_prev = k_total.shape[0] - L
                causal_mask = torch.full_like(scores, float("-inf"))
                causal_mask = torch.triu(causal_mask, diagonal=P_prev + 1)
                scores = scores + causal_mask

            attn_probs = F.softmax(scores.float(), dim=-1).type_as(v_total)
            head_out = torch.matmul(attn_probs, v_total)  # (L, 64)
            qkv_attention_store.append(head_out)

        # 4. Concatenate all 8 heads -> (L, 512) and project through W_o
        concat_heads = torch.cat(qkv_attention_store, dim=-1)
        embedding_delta = torch.matmul(concat_heads, wo.T)

        # Stack new (K, V) across heads -> (n_heads, L, 64) in case caller commits them to KV cache
        layer_new_kv = (torch.stack(new_k_per_head, dim=0), torch.stack(new_v_per_head, dim=0))
        return embedding_delta, layer_new_kv


# ==============================================================================
# 4. PURE PYTORCH CONVEYOR-BELT DIFFUSION ENGINE
# ==============================================================================
class PureConveyerEngine:
    """
    Implements the complete 64-Slot Dual-Channel Elastic Conveyor-Belt Diffusion LLM
    directly on raw weight tensors (`weights: dict[str, Tensor]`).

    ===========================================================================
    HOW THE ARCHITECTURE WORKS (Follow the 5 methods below in order):
    ===========================================================================
    1. `prefill_prefix_tokens(token_ids)`:
       Committed prefix tokens `x_{0..t-1}` are embedded with `tok_emb[id] + slot_emb[0]`,
       normalized via `dough_norm`, and run causally through the 8 Transformer layers to build
       the frozen `prefix_kv_cache`.

    2. `init_belt_with_prompt_overlap(prompt_ids, overlap=12)`:
       Instead of starting the belt cold at the end of the prompt `n`, we step back `W=12` tokens
       to `t_start = n - 12`:
         - Tokens `0 .. t_start-1` go into the frozen `prefix_kv_cache`.
         - The last 12 prompt tokens are clamped onto the front of the conveyor belt (`e_soft`),
           warming up the continuous thought highway `m_thought` before the first new token pops!

    3. `execute_belt_forward(e_soft, m_thought, tick)`:
       Composes the 64-slot input "dough":
         `dough[k] = RMSNorm( e_soft[k] + alpha[k] * m_thought[k] + slot_emb[k] )`
       where `alpha[k] = k / 63` (`alpha[0] = 0.0` at Slot 0 so Slot 0 has zero thought-vector
       distortion right when it pops out!).
       Runs all 64 slots through the 8 Transformer layers (attending to `prefix_kv_cache` + all 64 slots),
       then computes 3 parallel heads from the final hidden state `h` (64, 512):
         - Channel 1 (Lexical Anchor): `logits = h @ tok_emb.T`, `probs = softmax(logits / tau[k])`,
           `e_soft_new = probs @ tok_emb` (temperature `tau[k]` goes from `0.5` at Slot 0 to `4.0` at Slot 63).
         - Channel 2 (Thought Highway): `m_updated = gate * m_thought + (1 - gate) * cand`.
         - Elastic Planning & Expansion: `shift_prob[k]` (1.0 in front slots 0..15, ~0.25 in back slots 16..63)
           and `expand_cand` at the boundary between Slot 15 and Slot 16.

    4. `pop_slot0_and_append_kv(slot0_logits, ...)`:
       Pops the crisp token `x_t` from `slot0_logits` (Slot 0) and appends its K, V into `prefix_kv_cache`.

    5. `shift_belt_left(...)`:
       Advances the conveyor belt from `tick` to `tick + 1`:
         - Front 25% (Slots 0..14) shift strictly left from Slots 1..15.
         - Mid-Belt Slot 15 blends Slot 16's plan with the newly synthesized clause `expand_cand`.
         - Back 75% (Slots 16..62) elastically interpolate between staying in place `(1 - s_k)` and
           advancing from `k+1` `(s_k)`.
         - Slot 63 blends in fresh raw dough `(mean_tok_emb, m_init)`.
    """

    def __init__(self, weights: dict[str, torch.Tensor], config: dict, device: torch.device) -> None:
        self.weights = {k: v.to(device) for k, v in weights.items()}
        self.config = config
        self.device = device

        self.vocab_size: int = config["vocab_size"]              # 4096
        self.d_model: int = config["d_model"]                    # 512
        self.n_heads: int = config["n_heads"]                    # 8
        self.n_layers: int = config["n_layers"]                  # 8
        self.head_dim: int = self.d_model // self.n_heads        # 64
        self.belt_size: int = config["belt_size"]                # 64
        self.front_slots: int = config["front_conveyer_slots"]   # 16
        self.stride: int = config.get("plan_horizon_stride", 2)  # 2

        self.inv_freq = ConveyerMathHelper.calculate_inv_freq(
            self.head_dim, config.get("rope_theta", 10000.0), device=device
        )

        # Precompute the 4 fixed belt schedules across slots k = 0 .. 63:
        B = self.belt_size
        F_slots = self.front_slots
        ratio = torch.arange(B, device=device, dtype=torch.float32) / max(B - 1, 1)

        # (a) Positional temperature schedule tau[k]: 0.5 at Slot 0 -> 4.0 at Slot 63
        tau_min, tau_max, tau_gamma = config["tau_min"], config["tau_max"], config["tau_gamma"]
        self.tau_schedule = (tau_min + (ratio ** tau_gamma) * (tau_max - tau_min)).unsqueeze(-1)  # (64, 1)

        # (b) Continuous thought mixing weight alpha[k]: 0.0 at Slot 0 -> 1.0 at Slot 63
        self.alpha_schedule = ratio.unsqueeze(-1)  # (64, 1)

        # (c) RoPE target sequence offsets delta[k]:
        #     Slots 0..15 -> [0, 1, ..., 15] (1-to-1 immediate words)
        #     Slots 16..63 -> [16, 18, 20, ..., 110] (2x wider sentence-planning horizon)
        offsets = torch.arange(B, device=device, dtype=torch.long)
        offsets[F_slots:] = F_slots + torch.arange(B - F_slots, device=device, dtype=torch.long) * self.stride
        self.target_offsets = offsets  # (64,)

        # (d) Elastic shift prior bias: +6.0 in front (always shifts), -0.3 .. -1.5 in back (pauses ~70%)
        shift_prior = torch.zeros(B, device=device, dtype=torch.float32)
        shift_prior[:F_slots] = 6.0
        shift_prior[F_slots:] = -0.3 - 1.2 * torch.linspace(0.0, 1.0, B - F_slots, device=device)
        self.shift_prior = shift_prior  # (64,)

    # --------------------------------------------------------------------------
    # STEP 1: Prefix KV Cache Prefill (Causal Transformer Pass)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def append_tokens_to_kv_cache(
        self,
        token_ids: torch.Tensor,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None,
        start_pos: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Encodes 1 or more committed tokens into the frozen causal Prefix KV Cache.
        Notice that committed tokens receive `slot_emb[0]` before `dough_norm` so they
        match Slot 0's exact representation when popped off the conveyor belt!
        """
        L = token_ids.shape[0]
        if L == 0:
            return kv_cache or []

        positions = torch.arange(start_pos, start_pos + L, device=self.device, dtype=torch.long)

        # Embed token + Slot-0 oven embedding + RMSNorm
        tok_embeddings = self.weights["tok_emb.weight"][token_ids]       # (L, 512)
        slot0_embedding = self.weights["slot_emb.weight"][0].unsqueeze(0) # (1, 512)
        x = ConveyerMathHelper.rms_norm(
            tok_embeddings + slot0_embedding,
            self.weights["dough_norm.weight"],
        )

        updated_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(self.n_layers):
            prev_layer_kv = kv_cache[layer] if kv_cache is not None else None
            attn_layer = ConveyerAttentionLayer(layer, self.weights, self.n_heads, self.d_model, self.inv_freq)

            attn_delta, new_kv = attn_layer(
                x, positions=positions, prefix_kv_for_layer=prev_layer_kv, is_causal_prefill=True
            )
            x = x + attn_delta

            # SwiGLU Feed-Forward Network
            norm2_x = ConveyerMathHelper.rms_norm(x, self.weights[f"blocks.{layer}.norm2.weight"])
            ffn_delta = ConveyerMathHelper.swiglu_ffn(
                norm2_x,
                self.weights[f"blocks.{layer}.ffn.w_gate.weight"],
                self.weights[f"blocks.{layer}.ffn.w_up.weight"],
                self.weights[f"blocks.{layer}.ffn.w_down.weight"],
            )
            x = x + ffn_delta

            if prev_layer_kv is not None:
                k_cat = torch.cat([prev_layer_kv[0], new_kv[0]], dim=1)
                v_cat = torch.cat([prev_layer_kv[1], new_kv[1]], dim=1)
                updated_kv_cache.append((k_cat, v_cat))
            else:
                updated_kv_cache.append(new_kv)

        return updated_kv_cache

    # --------------------------------------------------------------------------
    # STEP 2: Initialize the 64-Slot Belt (with Prompt-Overlap Pre-Baking)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def init_session(self, prompt_ids: torch.Tensor, overlap: int = 12) -> dict:
        """
        Sets up the KV cache and 64-slot conveyor belt for a prompt of length `n`:
          - Splits prompt at `t_start = n - overlap`
          - Prefills `prompt_ids[:t_start]` into `kv_cache`
          - Spawns raw dough `(e_soft, m_thought)` across all 64 slots and clamps known
            prompt tokens onto slots where `t_start + target_offsets[k] < n`.
        """
        n = prompt_ids.shape[0]
        w_pre = max(0, min(n - 1 if n > 1 else 0, overlap, self.front_slots))
        t_start = n - w_pre

        # 1. Prefill frozen KV cache on x_{0 .. t_start-1}
        kv_cache = self.append_tokens_to_kv_cache(prompt_ids[:t_start], kv_cache=None, start_pos=0)

        # 2. Spawn raw unbaked dough across all 64 slots:
        #    Channel 1 (e_soft) starts as the mean vocabulary embedding
        #    Channel 2 (m_thought) starts as the learned vector `m_init`
        mean_tok_emb = self.weights["tok_emb.weight"].mean(dim=0, keepdim=True)  # (1, 512)
        m_init = self.weights["m_init"].unsqueeze(0)                             # (1, 512)
        e_soft = mean_tok_emb.expand(self.belt_size, self.d_model).clone()       # (64, 512)
        m_thought = m_init.expand(self.belt_size, self.d_model).clone()          # (64, 512)

        # 3. Clamp overlapping prompt tokens onto the front of the belt
        if w_pre > 0:
            slot_positions = t_start + self.target_offsets  # (64,)
            in_prompt_mask = slot_positions < n
            clamped_indices = slot_positions.clamp(max=n - 1)
            e_soft[in_prompt_mask] = self.weights["tok_emb.weight"][prompt_ids[clamped_indices]][in_prompt_mask]

        return {
            "prompt_ids": prompt_ids,
            "prompt_len": n,
            "tick": t_start,
            "kv_cache": kv_cache,
            "e_soft": e_soft,
            "m_thought": m_thought,
            "emitted_ids": [],
        }

    # --------------------------------------------------------------------------
    # STEP 3: Execute 1 Full Tick of the 64-Slot Conveyor Belt
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def step_tick(self, state: dict, temperature: float = 0.0) -> dict:
        """
        Runs ONE tick `t` of the Conveyor-Belt Diffusion LLM:
          1. Fuses Dual-Channel Dough (`e_soft + alpha * m_thought + slot_emb`)
          2. Runs the 8 Transformer layers over all 64 slots simultaneously
          3. Computes Slot Logits, Soft Vocab Embeddings, Gated Thought Update, & Elastic Shift Gate
          4. Pops confirmed token `x_t` from Slot 0 into the KV cache
          5. Shifts the 64-slot belt left for tick `t + 1`
        """
        t: int = state["tick"]
        n: int = state["prompt_len"]
        is_warmup: bool = t < n
        e_soft: torch.Tensor = state["e_soft"]       # (64, 512)
        m_thought: torch.Tensor = state["m_thought"] # (64, 512)
        kv_cache = state["kv_cache"]

        # --- A. Compose Dual-Channel Dough Input (64, 512) ---
        # Notice alpha_schedule[0] == 0.0, so Slot 0 is purely lexical (e_soft[0] + slot_emb[0])!
        slot_embeddings = self.weights["slot_emb.weight"]  # (64, 512)
        fused_dough = e_soft + self.alpha_schedule * m_thought + slot_embeddings
        h = ConveyerMathHelper.rms_norm(fused_dough, self.weights["dough_norm.weight"])

        # Target sequence positions for RoPE: t + [0, 1, ..., 15, 16, 18, 20, ..., 110]
        belt_positions = t + self.target_offsets  # (64,)

        # --- B. Pass all 64 Belt Slots through the 8 Transformer Layers ---
        for layer in range(self.n_layers):
            attn_layer = ConveyerAttentionLayer(layer, self.weights, self.n_heads, self.d_model, self.inv_freq)
            attn_delta, _ = attn_layer(
                h,
                positions=belt_positions,
                prefix_kv_for_layer=kv_cache[layer] if kv_cache else None,
                is_causal_prefill=False,  # Bidirectional across all 64 belt slots!
            )
            h = h + attn_delta

            norm2_h = ConveyerMathHelper.rms_norm(h, self.weights[f"blocks.{layer}.norm2.weight"])
            ffn_delta = ConveyerMathHelper.swiglu_ffn(
                norm2_h,
                self.weights[f"blocks.{layer}.ffn.w_gate.weight"],
                self.weights[f"blocks.{layer}.ffn.w_up.weight"],
                self.weights[f"blocks.{layer}.ffn.w_down.weight"],
            )
            h = h + ffn_delta

        # Final RMSNorm across the 64 refined belt slots
        h = ConveyerMathHelper.rms_norm(h, self.weights["final_norm.weight"])  # (64, 512)

        # --- C. Channel 1: Lexical Anchor (Soft Probability Projection) ---
        # Weight-tied LM head: logits = h @ tok_emb.weight.T -> (64, 4096)
        logits = torch.matmul(h, self.weights["tok_emb.weight"].T)
        probs = F.softmax((logits / self.tau_schedule).float(), dim=-1).type_as(h)
        e_soft_updated = torch.matmul(probs, self.weights["tok_emb.weight"])  # (64, 512)

        # --- D. Channel 2: Continuous Thought Highway (Gated Residual Update) ---
        # cand = h @ W_cand.T ; gate = sigmoid(h @ W_gate.T + b_gate)
        cand_thought = torch.matmul(h, self.weights["thought_cand.weight"].T)
        gate = torch.sigmoid(
            torch.matmul(h, self.weights["thought_gate.weight"].T) + self.weights["thought_gate.bias"]
        )
        m_updated = gate * m_thought + (1.0 - gate) * cand_thought  # (64, 512)

        # --- E. Elastic Planning Shift/Pause Gate & Mid-Belt Clause Expansion ---
        F_slots = self.front_slots  # 16
        raw_shift = torch.matmul(h, self.weights["shift_head.weight"].T).squeeze(-1) + self.shift_prior
        shift_prob = torch.sigmoid(raw_shift)  # (64,)
        shift_prob[:F_slots] = 1.0             # Front 16 slots always shift at 100% speed

        # Synthesize new clause thought at the boundary between Slot 15 (front) and Slot 16 (planning)
        boundary_pair = torch.cat([h[F_slots - 1], h[F_slots]], dim=-1)  # (1024,)
        expand_cand = torch.matmul(boundary_pair, self.weights["expand_head.weight"].T)  # (512,)

        # --- F. Pop Confirmed Token at Slot 0 & Commit to Frozen Prefix KV Cache ---
        slot0_logits = logits[0].clone()
        slot0_logits[[PAD_TOKEN_ID, BOS_TOKEN_ID, UNK_TOKEN_ID]] = -1e9

        # Prevent 3-in-a-row identical token stammers
        recent = state["emitted_ids"]
        if len(recent) >= 2 and recent[-1] == recent[-2]:
            slot0_logits[recent[-1]] -= 3.0

        if temperature <= 1e-5:
            pred_slot0_id = int(torch.argmax(slot0_logits).item())
        else:
            top_vals, top_idx = torch.topk(slot0_logits / temperature, k=24)
            sampled = torch.multinomial(F.softmax(top_vals, dim=-1), num_samples=1)
            pred_slot0_id = int(top_idx[sampled].item())

        if is_warmup:
            emitted_id = int(state["prompt_ids"][t].item())
        else:
            emitted_id = pred_slot0_id
            state["emitted_ids"].append(emitted_id)

        # Append the popped token x_t into the frozen Prefix KV Cache at position `t`
        state["kv_cache"] = self.append_tokens_to_kv_cache(
            torch.tensor([emitted_id], device=self.device, dtype=torch.long),
            kv_cache=kv_cache,
            start_pos=t,
        )

        # --- G. Two-Zone Elastic Belt Shift (Prepare Belt for Tick t + 1) ---
        B = self.belt_size
        e_spawn = self.weights["tok_emb.weight"].mean(dim=0, keepdim=True)  # (1, 512)
        m_spawn = self.weights["m_init"].unsqueeze(0)                       # (1, 512)

        # 1. Front Articulation Zone (Slots 0..14): strict 1-slot left shift from Slots 1..15
        e_front = e_soft_updated[1:F_slots]   # (15, 512)
        m_front = m_updated[1:F_slots]        # (15, 512)

        # 2. Mid-Belt Clause Expansion Slot (Slot 15): blends Slot 16 with `expand_cand`
        s_mid = shift_prob[F_slots].unsqueeze(-1)  # (1,)
        e_mid = e_soft_updated[F_slots : F_slots + 1]
        m_mid = (s_mid * m_updated[F_slots] + (1.0 - s_mid) * expand_cand).unsqueeze(0)

        # 3. Back 75% Elastic Planning Zone (Slots 16..62): interpolates between staying (k) and advancing (k+1)
        s_back = shift_prob[F_slots : B - 1].unsqueeze(-1)  # (47, 1)
        e_back = s_back * e_soft_updated[F_slots + 1 : B] + (1.0 - s_back) * e_soft_updated[F_slots : B - 1]
        m_back = s_back * m_updated[F_slots + 1 : B] + (1.0 - s_back) * m_updated[F_slots : B - 1]

        # 4. Tail Slot (Slot 63): blends fresh raw dough into the back of the oven
        s_tail = shift_prob[B - 1].unsqueeze(-1)
        e_tail = s_tail * e_spawn + (1.0 - s_tail) * e_soft_updated[B - 1 : B]
        m_tail = s_tail * m_spawn + (1.0 - s_tail) * m_updated[B - 1 : B]

        e_next = torch.cat([e_front, e_mid, e_back, e_tail], dim=0)  # (64, 512)
        m_next = torch.cat([m_front, m_mid, m_back, m_tail], dim=0)  # (64, 512)

        # If still in warmup (t + 1 < prompt_len), clamp remaining prompt tokens onto the shifted belt
        next_t = t + 1
        if next_t < n:
            next_positions = next_t + self.target_offsets
            in_prompt = next_positions < n
            clamped_idx = next_positions.clamp(max=n - 1)
            e_next[in_prompt] = self.weights["tok_emb.weight"][state["prompt_ids"][clamped_idx]][in_prompt]

        state["e_soft"] = e_next
        state["m_thought"] = m_next
        state["tick"] = next_t

        # Compute telemetry for UI/CLI inspection
        top_probs, top_ids = probs.topk(5, dim=-1)
        p_clamped = probs.clamp(min=1e-8)
        entropies = -(p_clamped * p_clamped.log()).sum(dim=-1)

        return {
            "tick": t,
            "is_warmup": is_warmup,
            "emitted_id": emitted_id,
            "pred_slot0_id": pred_slot0_id,
            "slot0_logits": slot0_logits,
            "top_ids": top_ids,           # (64, 5)
            "top_probs": top_probs,       # (64, 5)
            "entropies": entropies,       # (64,)
            "gate_means": gate.mean(-1),  # (64,)
            "shift_probs": shift_prob,    # (64,)
        }


# ==============================================================================
# 5. TKINTER GUI (Modeled after llama-runner-ui.py + Live 64-Slot Conveyor View)
# ==============================================================================
class ConveyerRunnerGUI:
    def __init__(self, tokenizer: CompactTokenizer, engine: PureConveyerEngine, prompt: str) -> None:
        self.tokenizer = tokenizer
        self.engine = engine
        self.prompt = prompt
        self.state: dict | None = None
        self.is_generating = False
        self.dropdown_window: tk.Toplevel | None = None

        self.window = tk.Tk()
        self.window.title(
            f"Conveyer-Belt Diffusion LLM Runner (Pure PyTorch, {engine.n_layers}L/{engine.n_heads}H/{engine.d_model}D, device={engine.device})"
        )
        self.window.geometry("1220x760")
        self._build_ui()
        self.reset_belt_from_editor()

    def _build_ui(self) -> None:
        # Top PanedWindow: Left = Story Text Editor, Right = Token IDs & 64-Slot Conveyor Inspector
        paned = tk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Left Panel: Prompt / Generated Story Editor
        left_frame = tk.LabelFrame(paned, text="Story Text (Committed Prefix + Popped Slot-0 Tokens)", padx=6, pady=6)
        self.text_editor = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=("Consolas", 12), width=52)
        self.text_editor.pack(fill=tk.BOTH, expand=True)
        self.text_editor.insert(tk.END, self.prompt)
        paned.add(left_frame)

        # Right Panel: Token IDs + Live 64-Slot Conveyor Belt Table
        right_frame = tk.Frame(paned)

        tok_box = tk.LabelFrame(right_frame, text="Committed Token IDs (4096 Compact BPE)", padx=6, pady=4)
        tok_box.pack(fill=tk.X, pady=(0, 6))
        self.token_editor = scrolledtext.ScrolledText(tok_box, wrap=tk.WORD, height=4, font=("Consolas", 10))
        self.token_editor.pack(fill=tk.X)

        belt_box = tk.LabelFrame(
            right_frame,
            text="Live 64-Slot Conveyor & Elastic Planning Belt (Slot 0 Pops Out -> Slot 63 Raw Dough)",
            padx=6,
            pady=6,
        )
        belt_box.pack(fill=tk.BOTH, expand=True)
        self.belt_viewer = scrolledtext.ScrolledText(belt_box, wrap=tk.NONE, font=("Consolas", 10))
        self.belt_viewer.pack(fill=tk.BOTH, expand=True)
        paned.add(right_frame)

        # Bottom Control Bar (matching llama-runner-ui.py controls + Belt Step)
        ctrl = tk.Frame(self.window, padx=8, pady=6)
        ctrl.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Button(ctrl, text="Reset / Pre-Bake Belt", command=self.reset_belt_from_editor, width=18).pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Step 1 Tick (Bake)", command=self.step_one_tick, width=16, bg="#d1e7dd").pack(side=tk.LEFT, padx=4)
        self.play_btn = tk.Button(ctrl, text="Play", command=self.toggle_play, width=12)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        tk.Button(ctrl, text="Step In (Top Choices @ Slot 0)", command=self.on_step_in, width=26).pack(side=tk.LEFT, padx=4)

        tk.Label(ctrl, text="Top-K:").pack(side=tk.LEFT, padx=(12, 2))
        self.top_k_entry = tk.Entry(ctrl, width=5)
        self.top_k_entry.insert(0, "10")
        self.top_k_entry.pack(side=tk.LEFT)

        tk.Label(ctrl, text="Overlap W:").pack(side=tk.LEFT, padx=(12, 2))
        self.overlap_entry = tk.Entry(ctrl, width=5)
        self.overlap_entry.insert(0, "12")
        self.overlap_entry.pack(side=tk.LEFT)

        self.status_label = tk.Label(ctrl, text="Ready", anchor=tk.W, font=("Consolas", 10))
        self.status_label.pack(side=tk.RIGHT, padx=8)

    def reset_belt_from_editor(self) -> None:
        self.is_generating = False
        self.play_btn.config(text="Play")
        text = self.text_editor.get("1.0", "end-1c").strip()
        prompt_ids = self.tokenizer.encode(text).to(self.engine.device)
        overlap = int(self.overlap_entry.get() or 12)

        t0 = time.time()
        self.state = self.engine.init_session(prompt_ids, overlap=overlap)
        # Fast-forward through the prompt-overlap warmup ticks so the next click emits a brand-new token!
        last_snap = None
        while self.state["tick"] < self.state["prompt_len"]:
            last_snap = self.engine.step_tick(self.state, temperature=0.0)
        dt = time.time() - t0

        self._refresh_token_display()
        if last_snap is not None:
            self._render_belt_snapshot(last_snap, dt)
        self.status_label.config(
            text=f"Prefilled {len(prompt_ids)} tokens (overlap={overlap}) in {dt:.3f}s | Next Tick={self.state['tick']}"
        )

    def step_one_tick(self) -> dict | None:
        if self.state is None:
            self.reset_belt_from_editor()
        assert self.state is not None

        t0 = time.time()
        snap = self.engine.step_tick(self.state, temperature=0.0)
        dt = time.time() - t0

        if not snap["is_warmup"]:
            piece = self.tokenizer.id_to_piece(snap["emitted_id"])
            if snap["emitted_id"] != EOS_TOKEN_ID:
                self.text_editor.insert("end-1c", piece)
                self.text_editor.see(tk.END)

        self._refresh_token_display()
        self._render_belt_snapshot(snap, dt)
        return snap

    def on_step_in(self) -> None:
        """Just like llama-runner-ui.py's `Step In`: shows a popup of Top-K candidate words at Slot 0."""
        if self.state is None:
            self.reset_belt_from_editor()
        assert self.state is not None

        # Run a non-mutating preview of the belt at the current tick to inspect Slot 0 choices
        preview_copy = {
            "prompt_ids": self.state["prompt_ids"],
            "prompt_len": self.state["prompt_len"],
            "tick": self.state["tick"],
            "kv_cache": [(k.clone(), v.clone()) for k, v in self.state["kv_cache"]],
            "e_soft": self.state["e_soft"].clone(),
            "m_thought": self.state["m_thought"].clone(),
            "emitted_ids": list(self.state["emitted_ids"]),
        }
        snap = self.engine.step_tick(preview_copy, temperature=0.0)
        k = max(1, min(50, int(self.top_k_entry.get() or 10)))
        probs = F.softmax(snap["slot0_logits"].float(), dim=-1)
        top_vals, top_idx = torch.topk(probs, k=k)

        if self.dropdown_window:
            self.dropdown_window.destroy()

        self.dropdown_window = tk.Toplevel(self.window)
        self.dropdown_window.title(f"Slot 0 Top-{k} Candidates (Space/Double-Click to Commit)")
        self.dropdown_window.geometry(f"320x260+{self.window.winfo_pointerx()}+{self.window.winfo_pointery()}")

        listbox = tk.Listbox(self.dropdown_window, font=("Consolas", 11))
        listbox.pack(fill=tk.BOTH, expand=True)

        choices: list[tuple[int, str, float]] = []
        for i in range(k):
            tid = int(top_idx[i].item())
            w = self.tokenizer.id_to_piece(tid)
            p = float(top_vals[i].item())
            choices.append((tid, w, p))
            listbox.insert(tk.END, f"{repr(w):<14} ({p * 100:5.1f}%) [id={tid}]")

        def _commit_selected(_event: object = None) -> None:
            sel = listbox.curselection()
            if not sel:
                return
            chosen_tid, chosen_word, _ = choices[sel[0]]
            self.text_editor.insert("end-1c", chosen_word)
            if self.dropdown_window:
                self.dropdown_window.destroy()
            self.reset_belt_from_editor()

        listbox.bind("<Double-Button-1>", _commit_selected)
        listbox.bind("<space>", _commit_selected)

    def toggle_play(self) -> None:
        if not self.is_generating:
            self.is_generating = True
            self.play_btn.config(text="Pause")
            threading.Thread(target=self._play_loop, daemon=True).start()
        else:
            self.is_generating = False
            self.play_btn.config(text="Play")

    def _play_loop(self) -> None:
        while self.is_generating:
            snap = self.step_one_tick()
            if snap is None or snap["emitted_id"] == EOS_TOKEN_ID:
                self.is_generating = False
                self.play_btn.config(text="Play")
                break
            self.window.update()

    def _refresh_token_display(self) -> None:
        text = self.text_editor.get("1.0", "end-1c")
        tokens = self.tokenizer.encode(text).tolist()
        self.token_editor.delete("1.0", tk.END)
        self.token_editor.insert(tk.END, " ".join(str(t) for t in tokens))

    def _render_belt_snapshot(self, snap: dict, dt: float) -> None:
        popped_piece = repr(self.tokenizer.id_to_piece(snap["emitted_id"]))
        lines = [
            f"Tick t={snap['tick']} ({'WARMUP' if snap['is_warmup'] else 'EMIT'}) | "
            f"Popped Slot 0: {popped_piece} | Step Time: {dt*1000:.1f} ms",
            "=" * 78,
            f"{'Slot':<6} | {'Zone':<16} | {'Top-1 Token':<14} | {'Prob':>6} | {'Entropy':>7} | {'Shift s_k':>9} | {'Top-3 Candidates'}",
            "-" * 78,
        ]
        # Display key slots across the 64-slot belt (Front 0..15, Mid 15..18, Back Planning 20..63)
        slots_to_show = list(range(0, 17)) + [18, 20, 24, 28, 32, 40, 48, 56, 60, 62, 63]
        for k in slots_to_show:
            if k >= self.engine.belt_size:
                continue
            zone = (
                "POP (alpha=0)"
                if k == 0
                else ("Front Conveyor" if k < 15 else ("Mid Expand" if k == 15 else f"Back Plan (+{int(self.engine.target_offsets[k])})"))
            )
            t1_id = int(snap["top_ids"][k, 0].item())
            t1_p = float(snap["top_probs"][k, 0].item())
            ent = float(snap["entropies"][k].item())
            s_k = float(snap["shift_probs"][k].item())
            top3_str = ", ".join(
                f"{repr(self.tokenizer.id_to_piece(int(snap['top_ids'][k, j])))}({float(snap['top_probs'][k, j])*100:.0f}%)"
                for j in range(3)
            )
            lines.append(
                f"k={k:<3d}  | {zone:<16} | {repr(self.tokenizer.id_to_piece(t1_id)):<14} | "
                f"{t1_p*100:5.1f}% | {ent:6.2f}n | {s_k*100:7.1f}%  | {top3_str}"
            )

        self.belt_viewer.delete("1.0", tk.END)
        self.belt_viewer.insert(tk.END, "\n".join(lines))
        self.status_label.config(text=f"Tick {snap['tick']} | Emitted {popped_piece} in {dt*1000:.1f} ms")

    def run(self) -> None:
        self.window.mainloop()


# ==============================================================================
# 6. ENTRY POINT (Supports both GUI and `--cli` terminal trace)
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Pure PyTorch Runner for Conveyor-Belt Diffusion LLM")
    parser.add_argument("--ckpt", type=str, default="runs/tinystories/best.pt", help="Path to best.pt")
    parser.add_argument("--tokenizer", type=str, default="data/tinystories/tiktoken_compact_4096.json")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, Lily and a little bunny walked into the green forest to find the golden key.",
    )
    parser.add_argument("--cli", action="store_true", help="Run in terminal mode instead of launching Tkinter GUI")
    parser.add_argument("--tokens", type=int, default=24, help="Tokens to generate in --cli mode")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading checkpoint from {args.ckpt} onto {device}...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    weights: dict[str, torch.Tensor] = ckpt["model_state"]
    model_cfg: dict = ckpt["config"]["model"]

    tokenizer = CompactTokenizer(args.tokenizer)
    engine = PureConveyerEngine(weights=weights, config=model_cfg, device=device)

    if args.cli:
        prompt_ids = tokenizer.encode(args.prompt.strip()).to(device)
        state = engine.init_session(prompt_ids, overlap=12)
        print(f"Prompt ({len(prompt_ids)} tokens): {args.prompt}\n" + "-" * 95)
        for _ in range(12 + args.tokens):
            snap = engine.step_tick(state, temperature=0.0)
            mode = "WARMUP" if snap["is_warmup"] else "EMIT  "
            pop_w = repr(tokenizer.id_to_piece(snap["emitted_id"]))
            front_preview = " ".join(
                tokenizer.id_to_piece(int(snap["top_ids"][k, 0])).strip() for k in range(1, 6)
            )
            back_pause = 100.0 * (1.0 - float(snap["shift_probs"][16:].mean().item()))
            print(
                f"Tick {snap['tick']:3d} [{mode}] | Slot 0 Pop: {pop_w:<12} | "
                f"Slots 1..5: {front_preview:<28} | Back 75% Paused: {back_pause:4.1f}%"
            )
        print("-" * 95)
        print("FINAL STORY:", args.prompt.strip() + tokenizer.decode(state["emitted_ids"]))
    else:
        gui = ConveyerRunnerGUI(tokenizer=tokenizer, engine=engine, prompt=args.prompt)
        gui.run()


if __name__ == "__main__":
    main()
