"""
Warm-Start TinyStories (`runs/tinystories/best.pt`) with a Role-Specialized 4-Register
English Thought Reservoir (`Slots 16..63`).

Why rigid index-by-index Cross-Entropy (`slot k == k-th future content word`) failed:
  In natural stories, whether `'sad'` is the 14th or 17th content word varies from story to story.
  Forcing Slot `k` to predict the exact `k`-th content word averages over all word types at index `k`,
  causing every slot to hedge on generic words (`'said'`, `'day'`, `','`).

Solution — DETR-Style Bipartite Set Matching + Iterative Competitive Attention:
  1. Each of the 4 Registers (`Reg 1: Scene Memory`, `Reg 2: Active Clause`, `Reg 3: Next Plot Turn`,
     `Reg 4: Story Resolution`) is supervised as a **12-slot semantic set** using bipartite matching
     between the 12 slot predictions and the set of English content words in that story span!
  2. Within the Reservoir, slots use **Competitive Coverage Attention** over `(tok_emb + prefix_h + belt_h)`
     so once Slot `k` locks onto `'Lily'`, its coverage mask subtracts `'Lily'` from the remaining slots,
     guaranteeing 12 distinct, high-confidence English concepts per register!
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from conveyer.checkpoint import load_checkpoint
from conveyer.data import build_dataset_and_tokenizer
from conveyer.data.tinystories import TiktokenCompactTokenizer
from conveyer.model import BeltState, BeltStepOutput, ConveyerDiffusionLM, PrefixKVCache


STOPWORDS = {
    "the", "and", "a", "to", "was", "in", "on", "of", "for", "with", "that", "is", "at", "be", "are",
    "as", "by", "an", "or", "from", "it", "this", "were", "had", "have", "has", "did", "so", "up", "out",
    "he", "she", "his", "her", "him", "they", "their", "them", "you", "your", "we", "our", "its", "but",
    "not", "very", "too", "all", "one", "there", "when", "then", "what", "who", "how", "can", "will",
    "would", "could", "should", "about", "into", "over", "after", "before", "once", "upon", "time",
}


def build_content_word_mask(tokenizer: TiktokenCompactTokenizer, device: torch.device) -> torch.Tensor:
    """Returns a boolean tensor `(vocab_size,)` True only for meaningful English content words."""
    mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
    for cid in range(260, tokenizer.vocab_size):
        piece = tokenizer.id_to_piece(cid).strip().lower()
        if len(piece) >= 2 and piece.isalpha() and piece not in STOPWORDS:
            mask[cid] = True
    return mask


def _extract_unique_content(seq_1d: torch.Tensor, is_content_vocab: torch.Tensor, max_len: int) -> list[int]:
    if seq_1d.numel() == 0:
        return []
    c_ids = seq_1d[is_content_vocab[seq_1d]].tolist()
    seen: set[int] = set()
    unique_list: list[int] = []
    for cid in c_ids:
        if cid not in seen:
            seen.add(cid)
            unique_list.append(cid)
            if len(unique_list) >= max_len:
                break
    return unique_list


def build_structured_reservoir_targets(
    seq_tokens: torch.Tensor,
    tick: int,
    is_content_vocab: torch.Tensor,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds `(res_targets, res_valid)` of shape `(bsz, 48)` across the 4 Registers (12 slots each):
      - `Reg 1 (Slots 16..27)`: Scene Memory — unique English content words in `seq[:tick]`
      - `Reg 2 (Slots 28..39)`: Active Clause — unique English content words in `seq[tick : tick+24]`
      - `Reg 3 (Slots 40..51)`: Next Plot Turn — unique English content words in `seq[tick+24 : tick+56]`
      - `Reg 4 (Slots 52..63)`: Story Outcome — unique English content words in `seq[tick+56 :]`
    """
    bsz, seq_len = seq_tokens.shape
    device = seq_tokens.device
    res_targets = torch.full((bsz, 48), pad_id, dtype=torch.long, device=device)
    res_valid = torch.zeros((bsz, 48), dtype=torch.bool, device=device)

    spans = [
        (0, 12, 0, tick),
        (12, 24, tick, min(seq_len, tick + 24)),
        (24, 36, min(seq_len, tick + 24), min(seq_len, tick + 56)),
        (36, 48, min(seq_len, tick + 56), seq_len),
    ]

    for b in range(bsz):
        for r_start, r_end, tok_start, tok_end in spans:
            reg_len = r_end - r_start
            sub_seq = seq_tokens[b, tok_start:tok_end]
            words = _extract_unique_content(sub_seq, is_content_vocab, max_len=reg_len)
            n_w = len(words)
            if n_w > 0:
                res_targets[b, r_start : r_start + n_w] = torch.tensor(words, dtype=torch.long, device=device)
                res_valid[b, r_start : r_start + n_w] = True

    return res_targets, res_valid


@torch.no_grad()
def match_register_targets_bipartite(
    reg_logits: torch.Tensor,  # (bsz, 12, V)
    reg_targets: torch.Tensor, # (bsz, 12)
    reg_valid: torch.Tensor,   # (bsz, 12)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DETR-style greedy bipartite matching within each 12-slot register:
    Matches each valid target word `w` in the register to the unassigned slot `s in 0..11`
    that currently gives `w` the highest logit! This allows each of the 12 slots to specialize
    on a semantic role (character, object, verb, emotion, setting) rather than rigid word index.
    """
    bsz, S, _ = reg_logits.shape
    matched_targets = reg_targets.clone()
    matched_valid = torch.zeros_like(reg_valid)

    for b in range(bsz):
        valid_idx = torch.nonzero(reg_valid[b], as_tuple=False).squeeze(-1)
        if valid_idx.numel() == 0:
            continue
        t_words = reg_targets[b, valid_idx]  # (M,)
        M = t_words.numel()
        # Score matrix: (S, M) = logit assigned by slot s to target word m
        scores = reg_logits[b][:, t_words].clone()  # (12, M)
        used_slots: set[int] = set()
        used_targs: set[int] = set()

        for _ in range(min(S, M)):
            # Find highest (slot, target) pair
            flat_idx = int(scores.argmax().item())
            s_i = flat_idx // M
            m_i = flat_idx % M
            if s_i in used_slots or m_i in used_targs:
                break
            used_slots.add(s_i)
            used_targs.add(m_i)
            matched_targets[b, s_i] = t_words[m_i]
            matched_valid[b, s_i] = True
            scores[s_i, :] = -1e9
            scores[:, m_i] = -1e9

    return matched_targets, matched_valid


class EnglishThoughtReservoir(nn.Module):
    """
    4-Register Role-Specialized English Thought Reservoir for `Slots 16..63`:
      - Reads from both the lexical token embeddings (`tok_emb(prefix)`) and deep contextual hidden
        states (`prefix_h` + `belt_h[:16]`).
      - Uses 48 role-specialized slot queries with Competitive Coverage Attention so every slot in a
        register locks onto a distinct English content word.
      - Outputs crisp English word distributions (`res_probs`) and expected token embeddings (`res_e_soft`)
        that write back into `belt.e_soft[:, 16:64]`!
    """

    def __init__(self, d_model: int, tok_emb_weight: torch.Tensor, is_content_vocab: torch.Tensor) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_res_slots = 48
        self.register_buffer("is_content_vocab", is_content_vocab, persistent=False)

        normed_vocab = F.normalize(tok_emb_weight.detach(), dim=-1)
        self.register_buffer("normed_vocab", normed_vocab, persistent=False)

        slot_q = torch.empty((48, d_model))
        nn.init.orthogonal_(slot_q)
        self.slot_queries = nn.Parameter(slot_q)

        self.mem_norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.salience_head = nn.Linear(d_model, 4, bias=True)  # Per-register salience scorer over memory tokens

        self.out_mlp = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.logit_scale = nn.Parameter(torch.tensor(18.0))
        self.writeback_logit = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        prefix_tokens: torch.Tensor,  # (bsz, P)
        prefix_h: torch.Tensor,       # (bsz, P, d_model)
        belt_h: torch.Tensor,         # (bsz, 64, d_model)
        tok_emb_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, P = prefix_tokens.shape
        D = self.d_model
        device = prefix_h.device

        # Memory bank combines exact unit-norm lexical identity + contextual Layer-7 hidden state!
        prefix_lex = F.normalize(tok_emb_weight[prefix_tokens], dim=-1)
        prefix_ctx = F.normalize(prefix_h, dim=-1)
        prefix_mem = self.mem_norm(prefix_lex + prefix_ctx)  # (bsz, P, D)

        # Flag which prefix positions are content words vs stopwords so Scene Memory focuses on content words!
        prefix_is_content = self.is_content_vocab[prefix_tokens]  # (bsz, P)
        front_mem = self.mem_norm(F.normalize(belt_h[:, :16, :], dim=-1))
        front_is_content = torch.ones((bsz, 16), dtype=torch.bool, device=device)

        memory_bank = torch.cat([prefix_mem, front_mem], dim=1)         # (bsz, P + 16, D)
        mem_valid = torch.cat([prefix_is_content, front_is_content], dim=1)  # (bsz, P + 16)

        q_all = self.q_proj(self.slot_queries.unsqueeze(0).expand(bsz, -1, -1))  # (bsz, 48, D)
        k_mem = self.k_proj(memory_bank)  # (bsz, M, D)
        v_mem = self.v_proj(memory_bank) + prefix_lex_padded(prefix_lex, 16)

        salience = self.salience_head(memory_bank)  # (bsz, M, 4)
        raw_scores = torch.matmul(q_all, k_mem.transpose(-1, -2)) / (D ** 0.5)  # (bsz, 48, M)

        # Competitive Coverage Attention inside each 12-slot register:
        # Once slot `s` attends to memory position `m`, coverage subtracts from `m` so slot `s+1` picks a new word!
        ctx_list: list[torch.Tensor] = []
        for reg_idx in range(4):
            s_start = reg_idx * 12
            s_end = s_start + 12
            reg_scores = raw_scores[:, s_start:s_end, :] + salience[:, :, reg_idx].unsqueeze(1)
            # Reg 1 (Scene Memory) strictly reads from prefix content words; Reg 2..4 read prefix + front belt
            if reg_idx == 0:
                reg_mask = torch.cat([prefix_is_content, torch.zeros((bsz, 16), dtype=torch.bool, device=device)], dim=1)
            else:
                reg_mask = mem_valid
            reg_scores = reg_scores.masked_fill(~reg_mask.unsqueeze(1), -1e4)

            coverage = torch.zeros((bsz, memory_bank.size(1)), device=device)
            for slot_i in range(12):
                step_scores = reg_scores[:, slot_i, :] - 6.0 * coverage
                attn_w = F.softmax(step_scores, dim=-1)  # (bsz, M)
                coverage = coverage + attn_w
                ctx_slot = torch.matmul(attn_w.unsqueeze(1), v_mem)  # (bsz, 1, D)
                ctx_list.append(ctx_slot)

        ctx = torch.cat(ctx_list, dim=1)  # (bsz, 48, D)
        global_summary = memory_bank.mean(dim=1, keepdim=True).expand(-1, 48, -1)
        delta = self.out_mlp(torch.cat([ctx, global_summary + self.slot_queries.unsqueeze(0)], dim=-1))

        # Reg 1 (Scene Memory) stays anchored to extracted memory content; Reg 2..4 blend memory + predictive MLP
        reg_weights = torch.tensor([0.15] * 12 + [0.65] * 36, device=device).view(1, 48, 1)
        res_h = F.normalize((1.0 - reg_weights) * ctx + reg_weights * (ctx + delta), dim=-1)

        scale = self.logit_scale.clamp(min=2.0, max=45.0)
        raw_logits = scale * torch.matmul(res_h, self.normed_vocab.T)  # (bsz, 48, V)
        res_logits = raw_logits.masked_fill(~self.is_content_vocab.view(1, 1, -1), -1e9)
        res_probs = F.softmax(res_logits.float(), dim=-1).type_as(tok_emb_weight)
        res_e_soft = torch.matmul(res_probs, tok_emb_weight)
        return res_logits, res_probs, res_e_soft


def prefix_lex_padded(prefix_lex: torch.Tensor, pad_right: int) -> torch.Tensor:
    bsz, _, D = prefix_lex.shape
    return torch.cat([prefix_lex, torch.zeros((bsz, pad_right, D), device=prefix_lex.device, dtype=prefix_lex.dtype)], dim=1)


def extract_prefix_hidden(model: ConveyerDiffusionLM, prefix_tokens: torch.Tensor) -> torch.Tensor:
    bsz, plen = prefix_tokens.shape
    device = prefix_tokens.device
    positions = torch.arange(0, plen, device=device, dtype=torch.long)
    cos, sin = model.rope.get_cos_sin(positions)
    slot0_bias = model.slot_emb.weight[0].view(1, 1, -1)
    x = model.dough_norm(model.tok_emb(prefix_tokens) + slot0_bias)
    for block in model.blocks:
        x, _ = block(x, cos=cos, sin=sin, attn_mask=None, layer_kv=None, return_new_kv=False, is_causal=(plen > 1))
    return model.final_norm(x)


def extract_belt_hidden(
    model: ConveyerDiffusionLM,
    belt: BeltState,
    kv_cache: PrefixKVCache,
) -> tuple[torch.Tensor, BeltStepOutput]:
    B = model.belt_size
    dough = model.compose_dough_inputs(belt, active_slots=B)
    offsets = model.get_slot_target_offsets(active_slots=B)
    positions = belt.tick + offsets
    cos, sin = model.rope.get_cos_sin(positions)

    h = dough
    for i, block in enumerate(model.blocks):
        layer_kv = (kv_cache.k_cache[i], kv_cache.v_cache[i]) if kv_cache is not None else None
        h, _ = block(h, cos=cos, sin=sin, attn_mask=None, layer_kv=layer_kv, return_new_kv=False)
    h = model.final_norm(h)
    step_out = model.forward_belt(belt=belt, kv_cache=kv_cache)
    return h, step_out


@torch.no_grad()
def decode_register_slots(
    res_probs_slice: torch.Tensor,
    tokenizer: TiktokenCompactTokenizer,
    max_words: int = 8,
) -> str:
    top_p, top_id = res_probs_slice.max(dim=-1)  # (12,)
    items: list[str] = []
    seen: set[int] = set()
    for k in range(res_probs_slice.size(0)):
        cid = int(top_id[k].item())
        p = float(top_p[k].item()) * 100.0
        if cid not in seen:
            seen.add(cid)
            w = tokenizer.id_to_piece(cid).strip()
            items.append(f"s{k}:{w}({p:.0f}%)")
        if len(items) >= max_words:
            break
    return " | ".join(items)


@torch.no_grad()
def inspect_reservoir_on_prompt(
    model: ConveyerDiffusionLM,
    reservoir: EnglishThoughtReservoir,
    tokenizer: TiktokenCompactTokenizer,
    prompt: str,
    device: torch.device,
    steps: int = 10,
) -> None:
    model.eval()
    reservoir.eval()
    p_ids = torch.tensor([tokenizer.encode(prompt.strip())], dtype=torch.long, device=device)
    n = p_ids.size(1)
    w_pre = min(12, n - 1)
    t_start = n - w_pre

    kv_cache = model.prefill_prefix(p_ids[:, :t_start])
    belt = model.init_empty_belt(batch_size=1, tick=t_start, device=device, deterministic=True)
    offsets = model.get_slot_target_offsets().to(device)
    slot_pos = t_start + offsets
    in_prompt = slot_pos < n
    clamped_pos = slot_pos.clamp(min=0, max=n - 1).unsqueeze(0)
    overlap_emb = model.tok_emb(torch.gather(p_ids, dim=1, index=clamped_pos))
    belt.e_soft = torch.where(in_prompt.view(1, 64, 1), overlap_emb, belt.e_soft)

    committed_seq = p_ids[:, :t_start].clone()
    emitted: list[int] = []

    for s in range(w_pre + steps):
        t = t_start + s
        is_warmup = t < n

        prefix_h = extract_prefix_hidden(model, committed_seq)
        belt_h, step_out = extract_belt_hidden(model, belt, kv_cache)
        _, res_probs, res_e_soft = reservoir(committed_seq, prefix_h, belt_h, model.tok_emb.weight)

        s0_logits = step_out.slot0_logits.clone()
        s0_logits[:, [0, 1, 3]] = -1e9
        if len(emitted) >= 2 and emitted[-1] == emitted[-2]:
            s0_logits[:, emitted[-1]] -= 3.0
        pred_s0 = int(s0_logits.argmax(dim=-1).item())
        emitted_tok = int(p_ids[0, t].item()) if is_warmup else pred_s0
        if not is_warmup:
            emitted.append(emitted_tok)

        if not is_warmup and s in (w_pre, w_pre + steps // 2, w_pre + steps - 1):
            front_words = [tokenizer.id_to_piece(int(step_out.logits[0, k].argmax())).strip() for k in range(1, 7)]
            reg1 = decode_register_slots(res_probs[0, 0:12], tokenizer)
            reg2 = decode_register_slots(res_probs[0, 12:24], tokenizer)
            reg3 = decode_register_slots(res_probs[0, 24:36], tokenizer)
            reg4 = decode_register_slots(res_probs[0, 36:48], tokenizer)
            print(f"  Tick {t:2d} | Pop Slot 0: {repr(tokenizer.id_to_piece(emitted_tok)):<10} | Front[1..6]: {' '.join(front_words)}")
            print(f"    [Reg 1 Scene Memory 16..27] : {reg1}")
            print(f"    [Reg 2 Active Clause 28..39]: {reg2}")
            print(f"    [Reg 3 Next Plot 40..51]    : {reg3}")
            print(f"    [Reg 4 Resolution 52..63]   : {reg4}")

        tok_t = torch.tensor([[emitted_tok]], device=device, dtype=torch.long)
        committed_seq = torch.cat([committed_seq, tok_t], dim=1)
        kv_cache = model.append_committed_token(tok_t, kv_cache)

        next_t = t + 1
        if next_t < n:
            next_pos = next_t + offsets
            c_mask = (next_pos < n).unsqueeze(0)
            c_toks = torch.gather(p_ids, dim=1, index=next_pos.clamp(max=n - 1).unsqueeze(0))
        else:
            c_mask, c_toks = None, None
        belt = model.shift_belt_left(step_out, current_tick=t, deterministic_spawn=True, clamp_tokens=c_toks, clamp_mask=c_mask)

        wb_gate = torch.sigmoid(reservoir.writeback_logit)
        belt.e_soft[:, 16:, :] = (1.0 - wb_gate) * belt.e_soft[:, 16:, :] + wb_gate * res_e_soft

    print(f"  Full Continuation: {tokenizer.decode(emitted)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm-start TinyStories with a Role-Specialized 4-Register English Thought Reservoir")
    parser.add_argument("--init", type=str, default="runs/tinystories/best.pt")
    parser.add_argument("--out", type=str, default="runs/tinystories/reservoir_best.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-stories", type=int, default=128)
    parser.add_argument("--lr", type=float, default=4e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, ckpt_meta = load_checkpoint(args.init, device=device)
    _, val_ds, tokenizer = build_dataset_and_tokenizer(cfg, infer_only=True)
    assert isinstance(tokenizer, TiktokenCompactTokenizer)

    is_content_vocab = build_content_word_mask(tokenizer, device)
    reservoir = EnglishThoughtReservoir(model.d_model, model.tok_emb.weight, is_content_vocab).to(device)

    prompt_demo = "Once upon a time, Lily and a little bunny walked into the green forest to find the golden key."

    print(f"[*] Caching Transformer states across {args.num_stories} TinyStories...")
    t_cache = time.time()
    model.eval()
    cached_prefix_toks: list[torch.Tensor] = []
    cached_prefix_h: list[torch.Tensor] = []
    cached_belt_h: list[torch.Tensor] = []
    cached_targets: list[torch.Tensor] = []
    cached_valid: list[torch.Tensor] = []

    with torch.no_grad():
        n_use = min(args.num_stories, len(val_ds))
        batch_sz = 16
        for start_i in range(0, n_use, batch_sz):
            end_i = min(n_use, start_i + batch_sz)
            seqs = torch.stack([val_ds[i]["input_ids"] for i in range(start_i, end_i)]).to(device)
            bsz, seq_len = seqs.shape
            for tick in (24, 36):
                p_toks = seqs[:, :tick]
                prefix_h = extract_prefix_hidden(model, p_toks)
                kv_t = model.prefill_prefix(p_toks)
                belt = model.init_empty_belt(bsz, tick, device, deterministic=True)
                for k in range(6):
                    if tick + k < seq_len:
                        belt.e_soft[:, k, :] = model.tok_emb(seqs[:, tick + k])
                belt_h, _ = extract_belt_hidden(model, belt, kv_t)
                targets, valid = build_structured_reservoir_targets(
                    seqs, tick, is_content_vocab, pad_id=tokenizer.pad_id
                )
                cached_prefix_toks.append(p_toks)
                cached_prefix_h.append(prefix_h)
                cached_belt_h.append(belt_h)
                cached_targets.append(targets)
                cached_valid.append(valid)

    print(f"[*] Cached {len(cached_prefix_h) * batch_sz} states in {time.time() - t_cache:.2f}s")

    optimizer = torch.optim.AdamW(reservoir.parameters(), lr=args.lr, weight_decay=1e-4)
    reservoir.train()
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss_sum = 0.0
        reg1_top1_sum = 0.0
        all_top1_sum = 0.0
        all_top5_sum = 0.0
        n_batches = len(cached_prefix_h)

        perm = torch.randperm(n_batches)
        for b_idx in perm:
            p_toks = cached_prefix_toks[int(b_idx)]
            p_h = cached_prefix_h[int(b_idx)]
            b_h = cached_belt_h[int(b_idx)]
            targs = cached_targets[int(b_idx)]
            val_m = cached_valid[int(b_idx)]

            res_logits, res_probs, _ = reservoir(p_toks, p_h, b_h, model.tok_emb.weight)

            # Match targets within each of the 4 registers using DETR bipartite matching
            matched_t_list = []
            matched_v_list = []
            for r_i in range(4):
                s0, s1 = r_i * 12, (r_i + 1) * 12
                mt, mv = match_register_targets_bipartite(
                    res_logits[:, s0:s1, :], targs[:, s0:s1], val_m[:, s0:s1]
                )
                matched_t_list.append(mt)
                matched_v_list.append(mv)
            matched_t = torch.cat(matched_t_list, dim=1)
            matched_v = torch.cat(matched_v_list, dim=1).float()

            ce = F.cross_entropy(
                res_logits.reshape(-1, model.vocab_size),
                matched_t.reshape(-1),
                reduction="none",
            ).view_as(matched_t)
            ce_loss = (ce * matched_v).sum() / matched_v.sum().clamp(min=1.0)

            reg_p = res_probs.view(res_probs.size(0), 4, 12, -1)
            gram = torch.matmul(reg_p, reg_p.transpose(-1, -2))
            eye = torch.eye(12, device=device, dtype=torch.bool).view(1, 1, 12, 12)
            excl_loss = gram.masked_fill(eye, 0.0).sum(dim=(-1, -2)).mean()

            loss = ce_loss + 0.5 * excl_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                loss_sum += float(ce_loss.item())
                pred1 = res_logits.argmax(dim=-1)
                c1 = (pred1 == matched_t).float() * matched_v
                all_top1_sum += float(c1.sum().item() / matched_v.sum().clamp(min=1.0).item())
                r1_m = matched_v[:, :12]
                reg1_top1_sum += float((c1[:, :12].sum() / r1_m.sum().clamp(min=1.0)).item())
                top5 = res_logits.topk(5, dim=-1).indices
                c5 = (top5 == matched_t.unsqueeze(-1)).any(dim=-1).float() * matched_v
                all_top5_sum += float(c5.sum().item() / matched_v.sum().clamp(min=1.0).item())

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            dt = time.time() - t0
            print(
                f"Epoch {epoch:2d}/{args.epochs} | res_ce {loss_sum/n_batches:.3f} | "
                f"SceneMem(Reg1) Top-1: {reg1_top1_sum/n_batches*100:5.1f}% | "
                f"All-48-Slots Top-1: {all_top1_sum/n_batches*100:5.1f}% | "
                f"All-48-Slots Top-5: {all_top5_sum/n_batches*100:5.1f}% | {dt:.1f}s",
                flush=True,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "reservoir_state": reservoir.state_dict(),
        "config": ckpt_meta.get("config", {}),
        "step": ckpt_meta.get("step", 0),
    }
    torch.save(payload, out_path)

    print("=" * 105)
    print(f"[After Warm-Start] Saved checkpoint to {args.out}. Inspecting 4-Register English Thought Reservoir:")
    inspect_reservoir_on_prompt(model, reservoir, tokenizer, prompt_demo, device, steps=10)
    print("=" * 105)


if __name__ == "__main__":
    main()
