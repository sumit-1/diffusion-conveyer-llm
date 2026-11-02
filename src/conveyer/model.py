from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from conveyer.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm + self.eps)
        return (self.weight * x_normed).type_as(x)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        self.max_seq_len = max(self.max_seq_len, seq_len)
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim // 2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def get_cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """positions: (L,) or (N, L) integer positions."""
        max_pos = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if max_pos > self.cos_cached.size(0):
            self._build_cache(max_pos + 256)
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: (N, H, L, D)
    cos, sin: (L, D//2) or (N, L, D//2)
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D//2)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)  # (N, 1, L, D//2)
        sin = sin.unsqueeze(1)
    d2 = x.shape[-1] // 2
    x1 = x[..., :d2]
    x2 = x[..., d2:]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1).type_as(x)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


@dataclass
class PrefixKVCache:
    """Stores committed prefix Keys and Values per layer: list of (K, V) each (N, H, P, D_h)."""
    k_cache: list[torch.Tensor]
    v_cache: list[torch.Tensor]
    seq_len: int

    @classmethod
    def empty(cls, n_layers: int, batch_size: int, n_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> "PrefixKVCache":
        k_cache = [torch.empty((batch_size, n_heads, 0, head_dim), device=device, dtype=dtype) for _ in range(n_layers)]
        v_cache = [torch.empty((batch_size, n_heads, 0, head_dim), device=device, dtype=dtype) for _ in range(n_layers)]
        return cls(k_cache=k_cache, v_cache=v_cache, seq_len=0)

    def clone(self) -> "PrefixKVCache":
        return PrefixKVCache(
            k_cache=[k.clone() for k in self.k_cache],
            v_cache=[v.clone() for v in self.v_cache],
            seq_len=self.seq_len,
        )

    def detach(self) -> "PrefixKVCache":
        return PrefixKVCache(
            k_cache=[k.detach() for k in self.k_cache],
            v_cache=[v.detach() for v in self.v_cache],
            seq_len=self.seq_len,
        )


@dataclass
class BeltState:
    """Holds the 64-slot Dual-Channel + Elastic Planning dough state on the conveyor belt."""
    e_soft: torch.Tensor      # (N, B, D) lexical anchor soft embeddings
    m_thought: torch.Tensor   # (N, B, D) continuous thought highway vectors
    tick: int                 # current absolute sequence position of Slot 0
    probs: Optional[torch.Tensor] = None       # (N, B, V) last probability distributions
    logits: Optional[torch.Tensor] = None      # (N, B, V) last raw logits
    gates: Optional[torch.Tensor] = None       # (N, B, D) last update gate activations
    shift_probs: Optional[torch.Tensor] = None # (N, B) elastic shift vs. pause probability s_k

    def detach(self) -> "BeltState":
        return BeltState(
            e_soft=self.e_soft.detach(),
            m_thought=self.m_thought.detach(),
            tick=self.tick,
            probs=self.probs.detach() if self.probs is not None else None,
            logits=self.logits.detach() if self.logits is not None else None,
            gates=self.gates.detach() if self.gates is not None else None,
            shift_probs=self.shift_probs.detach() if self.shift_probs is not None else None,
        )


@dataclass
class BeltStepOutput:
    logits: torch.Tensor          # (N, B, V) raw vocabulary logits at all 64 slots
    probs: torch.Tensor           # (N, B, V) temperature-scaled probabilities p_k
    e_soft: torch.Tensor          # (N, B, D) expected vocab embeddings p_k^T E_vocab
    cand_thought: torch.Tensor    # (N, B, D) candidate thought vectors c_k
    gate: torch.Tensor            # (N, B, D) update gate g_k
    m_updated: torch.Tensor       # (N, B, D) updated thought vectors before index shift
    shift_prob: torch.Tensor      # (N, B) elastic shift probability s_k (1.0 in front conveyor, <1 in back 75% planning)
    expand_cand: torch.Tensor     # (N, D) mid-belt expansion thought vector for forming new sentence clauses
    slot0_logits: torch.Tensor    # (N, V) crisp logits at Position 0


class ConveyerAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_new_kv: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        x: (N, L, D)
        layer_kv: optional (K_prev, V_prev) each (N, H, P, D_h)
        """
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        new_kv = (k, v) if return_new_kv else None

        if layer_kv is not None and layer_kv[0].size(2) > 0:
            k = torch.cat([layer_kv[0].type_as(k), k], dim=2)
            v = torch.cat([layer_kv[1].type_as(v), v], dim=2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if is_causal else attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.proj(out), new_kv


class ConveyerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = ConveyerAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_new_kv: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv = self.attn(
            self.norm1(x),
            cos=cos,
            sin=sin,
            attn_mask=attn_mask,
            layer_kv=layer_kv,
            return_new_kv=return_new_kv,
            is_causal=is_causal,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


class ConveyerDiffusionLM(nn.Module):
    """
    64-Slot Dual-Channel Elastic Conveyor-Belt Diffusion Language Model.

    Input / Output Contract at every tick t:
      - Input : [All Committed Text Prefix x_{0..t-1}] + [64-Slot Conveyor/Planning Belt]
      - Output: [Popped-out Confirmed Token x_t at Slot 0] + [Updated 64-Slot Conveyor/Planning Belt]

    Two-Zone Human-Style Thinking & Speaking Belt:
      - Front 25% (Slots 0 .. F-1, e.g. 0..15): Immediate Articulation Stream.
        Shifts left by 1 slot every tick (s_k = 1.0), polishing immediate grammar toward Slot 0 (alpha_0 = 0).
      - Back 75% (Slots F .. B-1, e.g. 16..63): Elastic Sentence-Planning Horizon.
        Can pause in-place (s_k < 1.0) to hold long-range sentence goals in mind while new sentence clauses
        bud and expand in the middle of the belt (around Slot F).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.belt_size = cfg.belt_size
        self.front_slots = min(cfg.front_conveyer_slots, cfg.belt_size)
        self.d_model = cfg.d_model
        self.vocab_size = cfg.vocab_size

        # Token embedding & weight-tied LM head
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # Learned oven slot embeddings for positions k = 0 .. B-1 on the conveyor belt
        self.slot_emb = nn.Embedding(cfg.belt_size, cfg.d_model)
        # Learned initial thought vector for newly spawned dough at Slot B-1
        self.m_init = nn.Parameter(torch.zeros(cfg.d_model))

        # Dough input RMSNorm
        self.dough_norm = RMSNorm(cfg.d_model)

        # Transformer trunk
        self.rope = RotaryEmbedding(
            head_dim=cfg.d_model // cfg.n_heads,
            max_seq_len=cfg.max_seq_len,
            theta=cfg.rope_theta,
        )
        self.blocks = nn.ModuleList([ConveyerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)

        # Channel 2: Continuous Thought Highway heads (RecurrentCubeLM style)
        self.thought_cand = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.thought_gate = nn.Linear(cfg.d_model, cfg.d_model, bias=True)

        # Elastic Planning & Mid-Belt Sentence Expansion Heads
        # shift_head predicts whether each back-75% planning slot shifts left or pauses in place
        self.shift_head = nn.Linear(cfg.d_model, 1, bias=True)
        # expand_head synthesizes a new phrase/clause embedding at the boundary between the
        # paused 75% planning zone and the moving 25% front conveyor
        self.expand_head = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)

        # Duplex-SimMT Frame Stacking (https://github.com/sumit-1/duplex-simmt-en-de):
        # Stacks [prefill+generated text token] + [content-only conveyer slots (k_slots)] into a single frame:
        #   frame_t = dough_norm( duplex_slot_proj(concat_k src_emb(conveyer_slots[t, k])) + tok_emb(prev_t) )
        self.duplex_k_slots = max(1, int(getattr(cfg, "duplex_k_slots", 4)))
        self.duplex_slot_proj = nn.Linear(self.duplex_k_slots * cfg.d_model, cfg.d_model, bias=False)
        # Also projects active content-conveyer dough states into Slot 0 so the conveyer directly drives generation
        self.conveyer_to_frame = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)

        # Precompute positional temperature schedule tau_k, thought mixing alpha_k, and target horizon offsets
        B = cfg.belt_size
        F_slots = self.front_slots
        k_idx = torch.arange(B, dtype=torch.float32)
        ratio = k_idx / max(B - 1, 1)
        tau_schedule = cfg.tau_min + (ratio ** cfg.tau_gamma) * (cfg.tau_max - cfg.tau_min)
        alpha_schedule = ratio.clone()  # alpha_0 = 0.0, alpha_{B-1} = 1.0

        # Target sequence offsets delta_k from current tick t:
        # - Front slots k = 0 .. F-1 (e.g. Last 5 slots 0..4) map 1-to-1 to immediate full-vocab tokens t + k
        # - Back content conveyer slots k = F .. B-1 (e.g. Slots 5..95) map to the ordered stream of content dough balls
        stride = max(1, cfg.plan_horizon_stride) if cfg.use_elastic_planning else 1
        target_offsets = torch.arange(B, dtype=torch.long)
        if B > F_slots and stride > 1:
            back_idx = torch.arange(B - F_slots, dtype=torch.long)
            target_offsets[F_slots:] = F_slots + back_idx * stride

        # Prior bias for shift_head:
        # - Front 5 slots (0..4) always shift by 1 every tick (+6.0 -> s_k = 1.0)
        # - Content dough-ball conveyer (Slots 5..95) rolls forward at ~55%-70% speed (matching the ~1.9x content-to-full-token ratio!)
        shift_prior = torch.zeros(B, dtype=torch.float32)
        shift_prior[:F_slots] = 6.0
        if B > F_slots:
            back_ratio = torch.linspace(0.0, 1.0, B - F_slots)
            shift_prior[F_slots:] = 0.8 - 0.8 * back_ratio  # sigmoid(+0.8 .. 0.0) = 69% -> 50% rolling shift!

        self.register_buffer("tau_schedule", tau_schedule, persistent=False)
        self.register_buffer("alpha_schedule", alpha_schedule, persistent=False)
        self.register_buffer("target_offsets", target_offsets, persistent=False)
        self.register_buffer("shift_prior", shift_prior, persistent=False)
        self.register_buffer("content_vocab_mask", torch.ones(cfg.vocab_size, dtype=torch.bool), persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.slot_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.m_init, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.lm_head:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.thought_gate.bias, self.cfg.gate_bias_init)
        nn.init.zeros_(self.shift_head.weight)
        nn.init.zeros_(self.shift_head.bias)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def get_slot_target_offsets(self, active_slots: Optional[int] = None) -> torch.Tensor:
        B = active_slots if active_slots is not None else self.belt_size
        return self.target_offsets[:B]

    def init_prefix_cache(self, batch_size: int, device: torch.device, dtype: Optional[torch.dtype] = None) -> PrefixKVCache:
        if dtype is None:
            dtype = self.tok_emb.weight.dtype
        return PrefixKVCache.empty(
            n_layers=self.cfg.n_layers,
            batch_size=batch_size,
            n_heads=self.cfg.n_heads,
            head_dim=self.cfg.d_model // self.cfg.n_heads,
            device=device,
            dtype=dtype,
        )

    def extract_duplex_conveyer_slots(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """
        CAUSAL extraction of the K=`duplex_k_slots` most recent content words up to and including position `t`
        (`seq_tokens[:, :t+1]`) so the Duplex-SimMT frame at tick `t` NEVER leaks future targets `>= t+1`!
        Returns `(bsz, seq_len, duplex_k_slots)` long tensor.
        """
        bsz, seq_len = seq_tokens.shape
        device = seq_tokens.device
        K = self.duplex_k_slots
        if bool(self.content_vocab_mask.all()):
            idx = (torch.arange(seq_len, device=device).unsqueeze(1) - torch.arange(0, K, device=device).unsqueeze(0)).clamp(min=0)
            return torch.gather(seq_tokens, 1, idx.unsqueeze(0).expand(bsz, -1, -1))

        is_c = self.content_vocab_mask[seq_tokens]  # (bsz, seq_len)
        if seq_len > 1:
            not_dup = torch.ones_like(is_c)
            not_dup[:, 1:] = seq_tokens[:, 1:] != seq_tokens[:, :-1]
            is_c = is_c & not_dup

        # Causal backward window of W=28 tokens at or before position t (t - W + 1 .. t)
        W = min(28, seq_len)
        t_idx = torch.arange(seq_len, device=device).unsqueeze(1)   # (L, 1)
        w_off = torch.arange(0, W, device=device).unsqueeze(0)      # (1, W): 0 = t, 1 = t-1, ...
        win_pos = (t_idx - w_off).clamp(min=0)                      # (L, W)
        in_bounds = (t_idx - w_off) >= 0                            # (L, W)

        win_toks = seq_tokens[:, win_pos]                           # (bsz, L, W)
        win_c = is_c[:, win_pos] & in_bounds.unsqueeze(0)           # (bsz, L, W)
        priority = torch.arange(W, 0, -1, device=device).view(1, 1, W)
        scores = torch.where(win_c, priority, torch.full_like(priority, -1))
        k_take = min(K, W)
        top_scores, top_i = scores.topk(k_take, dim=-1, largest=True, sorted=True)  # (bsz, L, k_take)
        chosen = torch.gather(win_toks, -1, top_i)
        chosen = torch.where(top_scores > 0, chosen, torch.zeros_like(chosen))
        if k_take < K:
            pad_z = torch.zeros((bsz, seq_len, K - k_take), dtype=torch.long, device=device)
            chosen = torch.cat([chosen, pad_z], dim=-1)
        return chosen

    def embed_duplex_frame(
        self,
        token_ids: torch.Tensor,
        conveyer_slot_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Duplex-SimMT Frame Embedding (https://github.com/sumit-1/duplex-simmt-en-de):
          `frame_t = dough_norm( duplex_slot_proj(concat_k tok_emb(conveyer_slots[t, k])) + tok_emb(token_ids[t]) + slot0_bias )`
        Stacks the prefill/generated text stream (`token_ids`) with the content-only conveyer (`conveyer_slot_ids`).
        """
        slot0_bias = self.slot_emb.weight[0].view(1, 1, -1)
        tgt_e = self.tok_emb(token_ids) + slot0_bias  # (bsz, L, D)
        if not getattr(self.cfg, "use_duplex_frame", False):
            return self.dough_norm(tgt_e)
        if conveyer_slot_ids is None:
            conveyer_slot_ids = self.extract_duplex_conveyer_slots(token_ids)
        bsz, seq_len, K = conveyer_slot_ids.shape
        src_e = self.tok_emb(conveyer_slot_ids).reshape(bsz, seq_len, K * self.d_model)
        return self.dough_norm(self.duplex_slot_proj(src_e) + tgt_e)

    def prefill_prefix(
        self,
        prefix_tokens: torch.Tensor,
        kv_cache: Optional[PrefixKVCache] = None,
        conveyer_slot_ids: Optional[torch.Tensor] = None,
    ) -> PrefixKVCache:
        """
        Encodes committed prefix tokens x_{t_start .. t_end} (stacked with Duplex-SimMT conveyer slots)
        into the causal PrefixKVCache.
        prefix_tokens: (N, P) long tensor.
        """
        bsz, plen = prefix_tokens.shape
        device = prefix_tokens.device
        if kv_cache is None:
            kv_cache = self.init_prefix_cache(bsz, device)
        if plen == 0:
            return kv_cache

        start_pos = kv_cache.seq_len
        positions = torch.arange(start_pos, start_pos + plen, device=device, dtype=torch.long)
        cos, sin = self.rope.get_cos_sin(positions)

        x = self.embed_duplex_frame(prefix_tokens, conveyer_slot_ids=conveyer_slot_ids)

        total_k = start_pos + plen
        use_is_causal = start_pos == 0 and plen > 1
        if use_is_causal or plen == 1:
            attn_mask = None
        else:
            causal_mask = torch.ones((plen, total_k), device=device, dtype=torch.bool).tril(diagonal=start_pos)
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        new_k_list: list[torch.Tensor] = []
        new_v_list: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            layer_kv = (kv_cache.k_cache[i], kv_cache.v_cache[i])
            x, new_kv = block(
                x,
                cos=cos,
                sin=sin,
                attn_mask=attn_mask,
                layer_kv=layer_kv,
                return_new_kv=True,
                is_causal=use_is_causal,
            )
            assert new_kv is not None
            new_k_list.append(torch.cat([kv_cache.k_cache[i].type_as(new_kv[0]), new_kv[0]], dim=2))
            new_v_list.append(torch.cat([kv_cache.v_cache[i].type_as(new_kv[1]), new_kv[1]], dim=2))

        return PrefixKVCache(k_cache=new_k_list, v_cache=new_v_list, seq_len=total_k)

    def forward_causal_and_cache(
        self,
        seq_tokens: torch.Tensor,
        prefix_len: int,
    ) -> tuple[torch.Tensor, PrefixKVCache]:
        """
        Runs a full causal pass over `seq_tokens` (N, L) through the shared transformer blocks:
          1. Computes next-token logits `(N, L, V)` at every position so all ~160 tokens per story
             receive direct causal language-modeling supervision (10x more signal per step).
          2. Extracts the exact causal KV cache for `seq_tokens[:, :prefix_len]` (invariant to tokens >= prefix_len)
             to seed the conveyor belt unroll without running a duplicate prefill pass.
        """
        bsz, seq_len = seq_tokens.shape
        device = seq_tokens.device
        positions = torch.arange(0, seq_len, device=device, dtype=torch.long)
        cos, sin = self.rope.get_cos_sin(positions)

        slot0_bias = self.slot_emb.weight[0].view(1, 1, -1)
        x = self.embed_duplex_frame(seq_tokens)

        k_list: list[torch.Tensor] = []
        v_list: list[torch.Tensor] = []
        for block in self.blocks:
            x, new_kv = block(
                x,
                cos=cos,
                sin=sin,
                attn_mask=None,
                layer_kv=None,
                return_new_kv=True,
                is_causal=(seq_len > 1),
            )
            assert new_kv is not None
            k_list.append(new_kv[0][:, :, :prefix_len, :].contiguous())
            v_list.append(new_kv[1][:, :, :prefix_len, :].contiguous())

        h = self.final_norm(x)
        causal_logits = self.lm_head(h)
        kv_cache = PrefixKVCache(k_cache=k_list, v_cache=v_list, seq_len=prefix_len)
        return causal_logits, kv_cache

    def append_committed_token(
        self,
        token_ids: torch.Tensor,
        kv_cache: PrefixKVCache,
        belt: Optional[BeltState] = None,
        committed_seq: Optional[torch.Tensor] = None,
    ) -> PrefixKVCache:
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(1)
        conveyer_slot_ids: Optional[torch.Tensor] = None
        if getattr(self.cfg, "use_duplex_frame", False):
            if committed_seq is not None:
                full_slots = self.extract_duplex_conveyer_slots(committed_seq)
                conveyer_slot_ids = full_slots[:, -token_ids.size(1) :, :]
            elif belt is not None and belt.logits is not None:
                K = self.duplex_k_slots
                F_slots = min(self.front_slots, belt.logits.size(1))
                if belt.logits.size(1) >= F_slots + K:
                    conveyer_slot_ids = belt.logits[:, F_slots : F_slots + K, :].argmax(dim=-1).unsqueeze(1)
        return self.prefill_prefix(token_ids, kv_cache=kv_cache, conveyer_slot_ids=conveyer_slot_ids)

    def spawn_raw_dough(
        self,
        batch_size: int,
        num_slots: int,
        device: torch.device,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_emb = self.tok_emb.weight.mean(dim=0, keepdim=True).unsqueeze(0)
        m_base = self.m_init.view(1, 1, -1)
        if deterministic or self.cfg.spawn_noise_std <= 0.0:
            e_new = mean_emb.expand(batch_size, num_slots, self.d_model).clone()
            m_new = m_base.expand(batch_size, num_slots, self.d_model).clone()
        else:
            eps_e = torch.randn((batch_size, num_slots, self.d_model), device=device, dtype=mean_emb.dtype)
            eps_m = torch.randn((batch_size, num_slots, self.d_model), device=device, dtype=mean_emb.dtype)
            e_new = mean_emb + self.cfg.spawn_noise_std * 0.02 * eps_e
            m_new = m_base + self.cfg.thought_noise_std * 0.02 * eps_m
        return e_new, m_new

    def init_empty_belt(
        self,
        batch_size: int,
        tick: int,
        device: torch.device,
        deterministic: bool = False,
    ) -> BeltState:
        e_soft, m_thought = self.spawn_raw_dough(
            batch_size=batch_size,
            num_slots=self.belt_size,
            device=device,
            deterministic=deterministic,
        )
        return BeltState(e_soft=e_soft, m_thought=m_thought, tick=tick)

    def compose_dough_inputs(self, belt: BeltState, active_slots: Optional[int] = None) -> torch.Tensor:
        B = active_slots if active_slots is not None else self.belt_size
        e_soft = belt.e_soft[:, :B, :]
        m_thought = belt.m_thought[:, :B, :]
        slots = torch.arange(B, device=e_soft.device, dtype=torch.long)
        s_emb = self.slot_emb(slots).unsqueeze(0)

        if self.cfg.use_thought_highway:
            alpha = self.alpha_schedule[:B].view(1, B, 1)  # alpha_0 == 0.0
            fused = e_soft + alpha * m_thought + s_emb
        else:
            fused = e_soft + s_emb

        # Duplex-SimMT Stacking on the Belt:
        # Stack the content-only conveyer (`Slots F_slots : F_slots + K`) directly into the Front Articulation
        # slots (`Slots 0 .. F_slots - 1`) via `duplex_slot_proj`, so the conveyer decides the generated output!
        F_slots = min(self.front_slots, B)
        K = self.duplex_k_slots
        if getattr(self.cfg, "use_duplex_frame", False) and B >= F_slots + K:
            conv_slots = e_soft[:, F_slots : F_slots + K, :].reshape(e_soft.size(0), 1, K * self.d_model)
            conv_drive = self.duplex_slot_proj(conv_slots)  # (N, 1, D)
            front_fused = fused[:, :F_slots, :] + conv_drive
            fused = torch.cat([front_fused, fused[:, F_slots:, :]], dim=1)

        return self.dough_norm(fused)

    def _build_belt_mask(
        self,
        belt_len: int,
        prefix_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        total_k = prefix_len + belt_len
        if not self.cfg.causal_belt_only:
            return torch.ones((1, 1, belt_len, total_k), device=device, dtype=torch.bool)
        mask = torch.ones((belt_len, total_k), device=device, dtype=torch.bool).tril(diagonal=prefix_len)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward_belt(
        self,
        belt: BeltState,
        kv_cache: Optional[PrefixKVCache] = None,
        active_slots: Optional[int] = None,
    ) -> BeltStepOutput:
        """
        Single forward pass across all active slots on the belt:
        Input : [Committed Prefix KV Cache x_{0..t-1}] + [Belt Dough 0..B-1]
        Output: [Slot 0 Token Logits + Updated Belt State (Lexical + Thought + Elastic Shift Gate)]
        """
        B = active_slots if active_slots is not None else self.belt_size
        dough = self.compose_dough_inputs(belt, active_slots=B)
        device = dough.device

        prefix_len = kv_cache.seq_len if kv_cache is not None else 0
        offsets = self.get_slot_target_offsets(active_slots=B)
        positions = belt.tick + offsets
        cos, sin = self.rope.get_cos_sin(positions)

        attn_mask = (
            None
            if not self.cfg.causal_belt_only
            else self._build_belt_mask(belt_len=B, prefix_len=prefix_len, device=device)
        )

        h = dough
        use_ckpt = self.training and self.cfg.grad_checkpointing
        for i, block in enumerate(self.blocks):
            layer_kv = (kv_cache.k_cache[i], kv_cache.v_cache[i]) if kv_cache is not None else None
            if use_ckpt:
                def _run_block(inp: torch.Tensor, blk: ConveyerBlock = block, lkv: Optional[tuple[torch.Tensor, torch.Tensor]] = layer_kv) -> torch.Tensor:
                    out_h, _ = blk(inp, cos=cos, sin=sin, attn_mask=attn_mask, layer_kv=lkv, return_new_kv=False)
                    return out_h
                h = checkpoint(_run_block, h, use_reentrant=False)
            else:
                h, _ = block(
                    h,
                    cos=cos,
                    sin=sin,
                    attn_mask=attn_mask,
                    layer_kv=layer_kv,
                    return_new_kv=False,
                )

        h = self.final_norm(h)  # (N, B, D)

        # Channel 1: Lexical Anchor (Soft Probability Projection)
        logits = self.lm_head(h)  # (N, B, V)
        F_slots = min(self.front_slots, B)
        if getattr(self.cfg, "use_thought_reservoir", False) and B > F_slots:
            non_content = ~self.content_vocab_mask.view(1, 1, -1)
            res_logits = logits[:, F_slots:, :].masked_fill(non_content, -1e9)
            logits = torch.cat([logits[:, :F_slots, :], res_logits], dim=1)
        tau = self.tau_schedule[:B].view(1, B, 1).type_as(logits)
        probs = F.softmax((logits / tau).float(), dim=-1).type_as(self.tok_emb.weight)
        e_soft = torch.matmul(probs, self.tok_emb.weight)  # (N, B, D)

        # Channel 2: Continuous Thought Highway (Gated Residual Update)
        cand = self.thought_cand(h)
        gate = torch.sigmoid(self.thought_gate(h))
        m_prev = belt.m_thought[:, :B, :]
        if self.cfg.use_thought_highway:
            m_updated = gate * m_prev + (1.0 - gate) * cand
        else:
            m_updated = torch.zeros_like(m_prev)

        # Elastic Planning Shift/Pause Gate & Mid-Belt Sentence Expansion
        if self.cfg.use_elastic_planning and B > F_slots:
            prior = self.shift_prior[:B].view(1, B).type_as(h)
            raw_shift = self.shift_head(h).squeeze(-1) + prior  # (N, B)
            shift_prob = torch.sigmoid(raw_shift)
            # Front conveyor slots (0 .. F-1, e.g. 0..4) always shift left at full speed (1.0)
            front_ones = torch.ones((h.size(0), F_slots), device=device, dtype=shift_prob.dtype)
            shift_prob = torch.cat([front_ones, shift_prob[:, F_slots:]], dim=1)
            # Mid-belt expansion thought synthesized from the boundary between Front (F-1) and Content Conveyer (F)
            boundary_pair = torch.cat([h[:, F_slots - 1, :], h[:, F_slots, :]], dim=-1)
            expand_cand = self.expand_head(boundary_pair)  # (N, D)
        else:
            shift_prob = torch.ones((h.size(0), B), device=device, dtype=h.dtype)
            expand_cand = h[:, -1, :]

        # Duplex-SimMT Conveyer-Decides-Result Fusion at Slot 0:
        # Fuses Slot 0's hidden state h[:, 0] with BOTH the incoming conveyer dough `belt.e_soft[:, F_slots : F_slots + K]`
        # AND the refined conveyer dough `e_soft[:, F_slots : F_slots + K]` so any content word on the conveyer
        # (including words injected into `belt.e_soft` at test time!) directly drives the generated token y_t!
        K = self.duplex_k_slots
        if getattr(self.cfg, "use_duplex_frame", False) and B >= F_slots + K:
            conv_blend = 0.5 * belt.e_soft[:, F_slots : F_slots + K, :] + 0.5 * e_soft[:, F_slots : F_slots + K, :]
            conv_summary = self.duplex_slot_proj(conv_blend.reshape(h.size(0), K * self.d_model))
            h_slot0_fused = self.final_norm(h[:, 0, :] + self.conveyer_to_frame(torch.cat([h[:, 0, :], conv_summary], dim=-1)))
            slot0_logits = self.lm_head(h_slot0_fused)
            logits = torch.cat([slot0_logits.unsqueeze(1), logits[:, 1:, :]], dim=1)
            # When the token emitted at Slot 0 is a content word that matches the head of the content conveyer (Slot F_slots),
            # force the front of the content conveyer to shift left by 100% so that dough ball is consumed and the NEXT
            # content word advances into Slot F_slots!
            s0_top = slot0_logits.argmax(dim=-1)                # (N,)
            cf_top = logits[:, F_slots, :].argmax(dim=-1)       # (N,)
            is_s0_content = self.content_vocab_mask[s0_top]     # (N,)
            consumed = (is_s0_content & (s0_top == cf_top)).unsqueeze(1)  # (N, 1)
            advance_ones = torch.ones_like(shift_prob[:, F_slots:])
            shift_prob = torch.cat([shift_prob[:, :F_slots], torch.where(consumed, advance_ones, shift_prob[:, F_slots:])], dim=1)
        else:
            slot0_logits = logits[:, 0, :]

        return BeltStepOutput(
            logits=logits,
            probs=probs,
            e_soft=e_soft,
            cand_thought=cand,
            gate=gate,
            m_updated=m_updated,
            shift_prob=shift_prob,
            expand_cand=expand_cand,
            slot0_logits=slot0_logits,
        )

    def shift_belt_left(
        self,
        step_out: BeltStepOutput,
        current_tick: int,
        deterministic_spawn: bool = False,
        clamp_tokens: Optional[torch.Tensor] = None,
        clamp_mask: Optional[torch.Tensor] = None,
    ) -> BeltState:
        """
        Updates the belt for tick t + 1:
          1. Front Conveyor Zone (Slots 0 .. F-2):
             Always shifts left by 1 slot (`1..F-1 -> 0..F-2`) so fully-baked tokens flow smoothly out of Slot 0.
          2. Mid-Belt Sentence Expansion Zone (Slot F-1, e.g. Slot 15):
             Blends the advancing plan from Slot F with the newly expanded clause (`expand_cand`),
             allowing new sentences and words to form in the middle while the back 75% pauses!
          3. Back 75% Elastic Planning Zone (Slots F .. B-2, e.g. 16..62):
             Each slot k updates elastically using its learned shift probability `s_k`:
               e'_{k} = s_k * e_{k+1} + (1 - s_k) * e_k
               m'_{k} = s_k * m_{k+1} + (1 - s_k) * m_k
             When `s_k` is low (~0.25), the back 75% of the belt pauses in-place holding the long-range
             sentence plan steady while the front conveyor articulates the current sentence!
          4. Slot B-1 (Slot 63):
             Blends fresh raw dough into the tail of the planning belt.
        """
        bsz, B, _ = step_out.e_soft.shape
        device = step_out.e_soft.device
        F_slots = min(self.front_slots, B)

        e_spawn, m_spawn = self.spawn_raw_dough(
            batch_size=bsz,
            num_slots=1,
            device=device,
            deterministic=deterministic_spawn,
        )

        if B <= 1:
            e_next = e_spawn
            m_next = m_spawn
            probs_next = torch.full((bsz, 1, self.vocab_size), 1.0 / self.vocab_size, device=device, dtype=step_out.probs.dtype)
            logits_next = torch.zeros((bsz, 1, self.vocab_size), device=device, dtype=step_out.logits.dtype)
            gates_next = torch.ones((bsz, 1, self.d_model), device=device, dtype=step_out.gate.dtype)
        elif (not self.cfg.use_elastic_planning) or B <= F_slots:
            # Strict 1-by-1 shift across all B slots
            e_next = torch.cat([step_out.e_soft[:, 1:, :], e_spawn], dim=1)
            m_next = torch.cat([step_out.m_updated[:, 1:, :], m_spawn], dim=1)
            probs_next = torch.cat(
                [
                    step_out.probs[:, 1:, :],
                    torch.full((bsz, 1, self.vocab_size), 1.0 / self.vocab_size, device=device, dtype=step_out.probs.dtype),
                ],
                dim=1,
            )
            logits_next = torch.cat(
                [
                    step_out.logits[:, 1:, :],
                    torch.zeros((bsz, 1, self.vocab_size), device=device, dtype=step_out.logits.dtype),
                ],
                dim=1,
            )
            gates_next = torch.cat(
                [
                    step_out.gate[:, 1:, :],
                    torch.ones((bsz, 1, self.d_model), device=device, dtype=step_out.gate.dtype),
                ],
                dim=1,
            )
        else:
            # Elastic Two-Zone Update (Rolling Content Dough Balls in F..B-1 -> Stopword/Grammar Filling in 0..F-1):
            # (a) Front slots 0 .. F-2 (Slots 0..3) shift strictly from 1 .. F-1 (Slots 1..4)
            e_front = step_out.e_soft[:, 1:F_slots, :]
            m_front = step_out.m_updated[:, 1:F_slots, :]

            # (b) Mid-Belt Expansion Slot (F-1 = Slot 4): where content dough ball at Slot 5 expands with stopwords/grammar!
            s_mid = step_out.shift_prob[:, F_slots : F_slots + 1].unsqueeze(-1)  # (N, 1, 1)
            e_FromPlan = step_out.e_soft[:, F_slots : F_slots + 1, :]
            m_FromPlan = step_out.m_updated[:, F_slots : F_slots + 1, :]
            m_expand = step_out.expand_cand.unsqueeze(1)  # (N, 1, D)
            e_mid = e_FromPlan
            m_mid = s_mid * m_FromPlan + (1.0 - s_mid) * m_expand

            # (c) Content Dough-Ball Conveyer Slots (F .. B-2 = Slots 5..94): roll forward as predictions advance!
            s_back = step_out.shift_prob[:, F_slots : B - 1].unsqueeze(-1)  # (N, B - 1 - F, 1)
            e_stay = step_out.e_soft[:, F_slots : B - 1, :]
            e_adv = step_out.e_soft[:, F_slots + 1 : B, :]
            e_back = s_back * e_adv + (1.0 - s_back) * e_stay

            m_stay = step_out.m_updated[:, F_slots : B - 1, :]
            m_adv = step_out.m_updated[:, F_slots + 1 : B, :]
            m_back = s_back * m_adv + (1.0 - s_back) * m_stay

            # (d) Slot B-1 (Slot 95, Tail of content conveyer): blends fresh raw dough ball into the tail
            s_tail = step_out.shift_prob[:, B - 1 : B].unsqueeze(-1)
            e_tail = s_tail * e_spawn + (1.0 - s_tail) * step_out.e_soft[:, B - 1 : B, :]
            m_tail = s_tail * m_spawn + (1.0 - s_tail) * step_out.m_updated[:, B - 1 : B, :]

            e_next = torch.cat([e_front, e_mid, e_back, e_tail], dim=1)
            m_next = torch.cat([m_front, m_mid, m_back, m_tail], dim=1)

            probs_next = torch.cat(
                [
                    step_out.probs[:, 1:, :],
                    torch.full((bsz, 1, self.vocab_size), 1.0 / self.vocab_size, device=device, dtype=step_out.probs.dtype),
                ],
                dim=1,
            )
            logits_next = torch.cat(
                [
                    step_out.logits[:, 1:, :],
                    torch.zeros((bsz, 1, self.vocab_size), device=device, dtype=step_out.logits.dtype),
                ],
                dim=1,
            )
            gates_next = torch.cat(
                [
                    step_out.gate[:, 1:, :],
                    torch.ones((bsz, 1, self.d_model), device=device, dtype=step_out.gate.dtype),
                ],
                dim=1,
            )

        if clamp_tokens is not None and clamp_mask is not None:
            clamped_emb = self.tok_emb(clamp_tokens)
            mask_exp = clamp_mask.unsqueeze(-1)
            e_next = torch.where(mask_exp, clamped_emb, e_next)

        return BeltState(
            e_soft=e_next,
            m_thought=m_next,
            tick=current_tick + 1,
            probs=probs_next,
            logits=logits_next,
            gates=gates_next,
            shift_probs=step_out.shift_prob,
        )
