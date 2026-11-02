from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import torch
import torch.nn.functional as F

from conveyer.model import BeltState, BeltStepOutput, ConveyerDiffusionLM, PrefixKVCache


@dataclass
class TickSnapshot:
    """Complete telemetry of a single conveyor-belt oven tick (used for live CLI & Web UI)."""
    tick: int                         # absolute sequence index at Slot 0
    is_warmup: bool                   # True during prompt-overlap pre-baking (t < prompt_len)
    emitted_token: int                # token popped out at Slot 0 (clamped if is_warmup)
    predicted_slot0_token: int        # model's raw top-1 prediction at Slot 0
    slot_top1_tokens: list[int]       # (B,) top-1 token ID at each slot k=0..B-1
    slot_top1_probs: list[float]      # (B,) top-1 probability at each slot k=0..B-1
    slot_entropies: list[float]       # (B,) Shannon entropy at each slot k=0..B-1
    slot_gate_means: list[float]      # (B,) mean thought-highway gate activation g_k
    slot_shift_probs: list[float]     # (B,) elastic shift probability s_k (1.0 in front, <1 in back 75% planning)
    slot_top3_tokens: list[list[int]] # (B, 3) top-3 candidate token IDs per slot
    slot_top3_probs: list[list[float]]# (B, 3) top-3 candidate probabilities per slot


@torch.no_grad()
def sample_slot0_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    recent_tokens: Optional[list[int]] = None,
    repetition_penalty: float = 1.25,
) -> torch.Tensor:
    """
    logits: (N, V)
    Masks out special tokens (<PAD>=0, <BOS>=1, <UNK>=3) and applies mild repetition penalty
    on recently emitted tokens before greedy or top-k sampling.
    """
    adj_logits = logits.clone()
    if adj_logits.size(-1) > 4:
        adj_logits[:, 0] = -1e9
        adj_logits[:, 1] = -1e9
        adj_logits[:, 3] = -1e9
    # In the 4096 compact BPE vocabulary, IDs 4..259 are single-byte UTF-8 fallback tokens.
    # Mask out raw single-byte character/letter fragments (except standard ASCII punctuation
    # '!'=37, '"'=38, "'"=43, ','=48, '-'=49, '.'=50, ':'=62, ';'=63, '?'=67) so a stray space
    # byte (ID 36 = ' ') never triggers character-by-character non-word spelling on long rollouts!
    if adj_logits.size(-1) >= 4096:
        allowed_punct = {37, 38, 43, 48, 49, 50, 62, 63, 67}
        byte_mask = [cid for cid in range(4, 260) if cid not in allowed_punct] + [496, 627]
        adj_logits[:, byte_mask] = -1e9

    if recent_tokens:
        if repetition_penalty > 1.0:
            for tid in set(recent_tokens[-16:]):
                if 0 <= tid < adj_logits.size(-1):
                    val = adj_logits[:, tid]
                    adj_logits[:, tid] = torch.where(val > 0, val / repetition_penalty, val * repetition_penalty)
        # Block immediate 2-in-a-row identical token stammers
        if len(recent_tokens) >= 1:
            last_tid = recent_tokens[-1]
            if 0 <= last_tid < adj_logits.size(-1):
                adj_logits[:, last_tid] -= 4.0
        # Block repeating 3-gram phrase loops (without banning normal 2-word noun phrases like 'a shiny key' or 'the forest')
        n_rec = len(recent_tokens)
        if n_rec >= 5:
            last_trigram = (recent_tokens[-3], recent_tokens[-2], recent_tokens[-1])
            for idx in range(max(0, n_rec - 40), n_rec - 3):
                if (recent_tokens[idx], recent_tokens[idx + 1], recent_tokens[idx + 2]) == last_trigram:
                    banned_next = recent_tokens[idx + 3]
                    if 0 <= banned_next < adj_logits.size(-1):
                        adj_logits[:, banned_next] -= 6.0

    if temperature <= 1e-5:
        return adj_logits.argmax(dim=-1)
    scaled = adj_logits / temperature
    k = min(24, scaled.size(-1))
    top_vals, _ = torch.topk(scaled, k, dim=-1)
    min_top = top_vals[:, -1].unsqueeze(-1)
    scaled = torch.where(scaled < min_top, torch.full_like(scaled, -1e9), scaled)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _build_inference_warmup_clamp(
    model: ConveyerDiffusionLM,
    prompt_tokens: torch.Tensor,
    tick: int,
    active_slots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds `(clamp_tokens, clamp_mask)` of shape `(bsz, active_slots)` for tick `t < prompt_len`:
      - Front Slots `0 .. F_slots-1`: clamped to exact prompt tokens `prompt_tokens[:, t : t + F_slots]`
      - Content Conveyer Slots `F_slots .. B-1`: clamped to the upcoming content words in `prompt_tokens[:, t + F_slots :]`
        so the entire content conveyer is warmed up on the prompt's nouns/verbs/adjectives before tick `n`!
    """
    bsz, n = prompt_tokens.shape
    device = prompt_tokens.device
    B = active_slots
    F_slots = min(model.front_slots, B)
    offsets = model.get_slot_target_offsets(active_slots=B).to(device)

    slot_pos = tick + offsets  # (B,)
    in_prompt = slot_pos < n
    if getattr(model.cfg, "use_thought_reservoir", False) and B > F_slots:
        in_prompt = in_prompt & (torch.arange(B, device=device) < F_slots)
    clamped_pos = slot_pos.clamp(min=0, max=n - 1).unsqueeze(0).expand(bsz, -1)
    clamp_tokens = torch.gather(prompt_tokens, dim=1, index=clamped_pos)
    clamp_mask = in_prompt.unsqueeze(0).expand(bsz, -1).clone()

    if getattr(model.cfg, "use_thought_reservoir", False) and B > F_slots and not bool(model.content_vocab_mask.all()):
        tok_start = min(n, tick + F_slots)
        res_slots = min(B - F_slots, n - tok_start)
        if res_slots > 0:
            sub_toks = prompt_tokens[:, tok_start:]  # (bsz, L_rem)
            is_c = model.content_vocab_mask[sub_toks]
            if sub_toks.size(1) > 1:
                not_dup = torch.ones_like(is_c)
                not_dup[:, 1:] = sub_toks[:, 1:] != sub_toks[:, :-1]
                is_c = is_c & not_dup
            L_rem = sub_toks.size(1)
            idx_score = torch.arange(L_rem, 0, -1, device=device).unsqueeze(0)
            scores = torch.where(is_c, idx_score, torch.full_like(idx_score, -1))
            top_scores, top_idx = scores.topk(res_slots, dim=1, largest=True, sorted=True)
            chosen_toks = torch.gather(sub_toks, 1, top_idx)
            chosen_valid = top_scores > 0
            clamp_tokens[:, F_slots : F_slots + res_slots] = torch.where(
                chosen_valid, chosen_toks, clamp_tokens[:, F_slots : F_slots + res_slots]
            )
            clamp_mask[:, F_slots : F_slots + res_slots] = chosen_valid

    return clamp_tokens, clamp_mask


@torch.no_grad()
def stream_conveyer_generation(
    model: ConveyerDiffusionLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 64,
    prompt_overlap: int = 12,
    bake_steps: Optional[int] = None,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
    deterministic_spawn: bool = False,
    injected_conveyer_tokens: Optional[list[int]] = None,
) -> Iterator[tuple[torch.Tensor, TickSnapshot]]:
    """
    Streams generation tick-by-tick using the `n - W_pre` Prompt-Overlap Pre-Baking protocol:
      - Input at every tick t : [Committed Prefix x_{0..t-1}] + [96-Slot Conveyor/Planning Belt]
      - Output at every tick t: [Popped-out Confirmed Token x_t at Slot 0] + [Updated 96-Slot Belt]
    """
    model.eval()
    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)

    bsz, n = prompt_tokens.shape
    device = prompt_tokens.device
    B = min(bake_steps, model.belt_size) if bake_steps is not None else model.belt_size
    B = max(1, B)

    # Allow full prompt-overlap warmup (e.g. 12-16 ticks) so BOTH Front Slots (0..4) AND the
    # Content Dough-Ball Conveyer (Slots 5..95) warm up across the prompt before generation starts!
    max_overlap = min(24, B - 1) if B > 1 else 0
    w_pre = max(0, min(n - 1 if n > 1 else 0, prompt_overlap, max_overlap))
    t_start = n - w_pre

    # 1. Prefill prefix KV cache on x_{0 : t_start - 1}
    kv_cache: PrefixKVCache = model.prefill_prefix(prompt_tokens[:, :t_start])

    # 2. Initialize conveyor belt at t_start
    belt: BeltState = model.init_empty_belt(
        batch_size=bsz,
        tick=t_start,
        device=device,
        deterministic=deterministic_spawn,
    )
    if B < model.belt_size:
        belt = BeltState(
            e_soft=belt.e_soft[:, :B, :],
            m_thought=belt.m_thought[:, :B, :],
            tick=t_start,
        )

    # Clamp both Front Slots (0..4) AND Content Conveyer Slots (5..95) at tick t_start
    if w_pre > 0:
        init_c_toks, init_c_mask = _build_inference_warmup_clamp(model, prompt_tokens, t_start, B)
        overlap_emb = model.tok_emb(init_c_toks)
        e_init = torch.where(init_c_mask.unsqueeze(-1), overlap_emb, belt.e_soft)
        belt = BeltState(e_soft=e_init, m_thought=belt.m_thought, tick=t_start)

    total_ticks = w_pre + max_new_tokens
    recent_emitted: list[int] = []
    committed_seq = prompt_tokens[:, :t_start].clone()
    F_slots = min(model.front_slots, B)
    inj_emb: Optional[torch.Tensor] = None
    if injected_conveyer_tokens and B > F_slots:
        inj_t = torch.tensor(injected_conveyer_tokens, dtype=torch.long, device=device)
        inj_emb = model.tok_emb(inj_t).unsqueeze(0)  # (1, L_inj, D)

    for step_idx in range(total_ticks):
        t = t_start + step_idx
        is_warmup = t < n

        # Optional test-time conveyer word injection (e.g. inserting 'dragon' into the content conveyer
        # at Slots F_slots..F_slots+3 until the injected word is emitted into the story!)
        if inj_emb is not None and (not is_warmup) and (t - n) < 16:
            already_emitted = any(tid in injected_conveyer_tokens for tid in recent_emitted) if injected_conveyer_tokens else False
            if not already_emitted:
                L_inj = min(inj_emb.size(1), B - F_slots, 4)
                e_mod = belt.e_soft.clone()
                e_mod[:, F_slots : F_slots + L_inj, :] = inj_emb[:, :L_inj, :]
                belt = BeltState(
                    e_soft=e_mod,
                    m_thought=belt.m_thought,
                    tick=belt.tick,
                    probs=belt.probs,
                    logits=belt.logits,
                    gates=belt.gates,
                    shift_probs=belt.shift_probs,
                )

        # Single forward pass across all active slots on the belt
        step_out: BeltStepOutput = model.forward_belt(belt=belt, kv_cache=kv_cache, active_slots=B)

        raw_pred_tok = sample_slot0_token(
            step_out.slot0_logits,
            temperature=temperature,
            recent_tokens=recent_emitted if not is_warmup else None,
            repetition_penalty=1.12,
        )
        if is_warmup:
            emitted_tok = prompt_tokens[:, t]
        else:
            emitted_tok = raw_pred_tok
            recent_emitted.append(int(emitted_tok[0].item()))

        probs0 = step_out.probs[0]  # (B, V)
        p_clamped = probs0.clamp(min=1e-8)
        entropies = -(p_clamped * p_clamped.log()).sum(dim=-1)  # (B,)
        k_top = min(3, model.vocab_size)
        top_probs, top_ids = probs0.topk(k_top, dim=-1)
        gate_means = step_out.gate[0].mean(dim=-1)  # (B,)
        shift_probs = step_out.shift_prob[0]        # (B,)

        snapshot = TickSnapshot(
            tick=t,
            is_warmup=is_warmup,
            emitted_token=int(emitted_tok[0].item()),
            predicted_slot0_token=int(raw_pred_tok[0].item()),
            slot_top1_tokens=top_ids[:, 0].tolist(),
            slot_top1_probs=top_probs[:, 0].tolist(),
            slot_entropies=entropies.tolist(),
            slot_gate_means=gate_means.tolist(),
            slot_shift_probs=shift_probs.tolist(),
            slot_top3_tokens=top_ids.tolist(),
            slot_top3_probs=top_probs.tolist(),
        )

        yield emitted_tok, snapshot

        if (not is_warmup) and eos_token_id is not None and int(emitted_tok[0].item()) == eos_token_id:
            break

        # Commit popped-out token x_t (stacked with Duplex-SimMT conveyer slots) to frozen prefix KV cache
        tok_2d = emitted_tok.unsqueeze(1) if emitted_tok.dim() == 1 else emitted_tok
        committed_seq = torch.cat([committed_seq, tok_2d], dim=1)
        belt_with_logits = BeltState(
            e_soft=step_out.e_soft,
            m_thought=step_out.m_updated,
            tick=t,
            probs=step_out.probs,
            logits=step_out.logits,
            gates=step_out.gate,
            shift_probs=step_out.shift_prob,
        )
        # During normal unsteered generation, use exact causal Duplex frame from committed_seq (matching training!).
        # When testing external conveyer injection, let the belt's injected slots also steer the frame until emitted!
        use_causal_seq = committed_seq if (not injected_conveyer_tokens or any(tid in injected_conveyer_tokens for tid in recent_emitted)) else None
        kv_cache = model.append_committed_token(
            emitted_tok, kv_cache, belt=belt_with_logits, committed_seq=use_causal_seq
        )

        # Shift belt left and clamp remaining prompt tokens on Front Slots (0..4) AND Content Conveyer (5..95) at t + 1
        next_t = t + 1
        if next_t < n:
            clamp_tokens, clamp_mask = _build_inference_warmup_clamp(model, prompt_tokens, next_t, B)
        else:
            clamp_tokens = None
            clamp_mask = None

        belt = model.shift_belt_left(
            step_out=step_out,
            current_tick=t,
            deterministic_spawn=deterministic_spawn,
            clamp_tokens=clamp_tokens,
            clamp_mask=clamp_mask,
        )
        # During long open-ended generation (> 16 ticks), training unrolls were U=16 steps long.
        # Gently refresh the recurrent belt state every 16 ticks after warmup so 100+ token rollouts
        # never accumulate out-of-distribution recurrent drift!
        if (not is_warmup) and ((t - n + 1) % 16 == 0):
            fresh = model.init_empty_belt(bsz, belt.tick, device, deterministic=True)
            if B < model.belt_size:
                fresh = BeltState(
                    e_soft=fresh.e_soft[:, :B, :],
                    m_thought=fresh.m_thought[:, :B, :],
                    tick=belt.tick,
                )
            belt = BeltState(
                e_soft=0.35 * belt.e_soft + 0.65 * fresh.e_soft,
                m_thought=0.35 * belt.m_thought + 0.65 * fresh.m_thought,
                tick=belt.tick,
                probs=belt.probs,
                logits=belt.logits,
                gates=belt.gates,
                shift_probs=belt.shift_probs,
            )


@torch.no_grad()
def generate_conveyer(
    model: ConveyerDiffusionLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 64,
    prompt_overlap: int = 12,
    bake_steps: Optional[int] = None,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
    deterministic_spawn: bool = False,
    injected_conveyer_tokens: Optional[list[int]] = None,
) -> tuple[torch.Tensor, list[TickSnapshot]]:
    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)

    generated: list[torch.Tensor] = []
    snapshots: list[TickSnapshot] = []

    for tok, snap in stream_conveyer_generation(
        model=model,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        prompt_overlap=prompt_overlap,
        bake_steps=bake_steps,
        temperature=temperature,
        eos_token_id=eos_token_id,
        deterministic_spawn=deterministic_spawn,
        injected_conveyer_tokens=injected_conveyer_tokens,
    ):
        snapshots.append(snap)
        if not snap.is_warmup:
            generated.append(tok.unsqueeze(1))

    if len(generated) > 0:
        gen_tensor = torch.cat(generated, dim=1)
        full_seq = torch.cat([prompt_tokens, gen_tensor], dim=1)
    else:
        full_seq = prompt_tokens.clone()

    return full_seq, snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI Test & Tick-by-Tick Trace for Conveyor-Belt Diffusion LLM")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default=None, help="Initial text prompt to continue")
    parser.add_argument("--tokens", type=int, default=32, help="Number of new tokens to bake and emit")
    parser.add_argument("--overlap", type=int, default=12, help="Prompt-overlap pre-baking window (n - W)")
    parser.add_argument("--bake-steps", type=int, default=64, help="Active conveyor belt length K (1..64)")
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature at Slot 0 (0 = greedy, 0.7 = creative)")
    args = parser.parse_args()

    from conveyer.checkpoint import load_checkpoint
    from conveyer.data import build_dataset_and_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt
    if ckpt_path is None:
        ckpt_path = (
            "runs/synthetic/best.pt"
            if (not Path("runs/tinystories/best.pt").exists() and Path("runs/synthetic/best.pt").exists())
            else "runs/tinystories/best.pt"
        )

    model, cfg, _ = load_checkpoint(ckpt_path, device=device)
    _, val_ds, tokenizer = build_dataset_and_tokenizer(cfg, infer_only=True)
    if hasattr(tokenizer, "content_vocab_mask"):
        model.content_vocab_mask.copy_(getattr(tokenizer, "content_vocab_mask").to(device))

    if args.prompt is not None:
        # A trailing space becomes a lone ' ' BPE token that never ends a prefix in training.
        prompt_ids = tokenizer.encode(args.prompt.strip())
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    else:
        sample = val_ds[0]
        p_len = int(sample["prompt_len"])
        prompt_tensor = sample["input_ids"][:p_len].unsqueeze(0).to(device)

    prompt_decoded = tokenizer.decode(prompt_tensor[0].tolist())
    print("=" * 108)
    print(f"Checkpoint : {ckpt_path} ({model.num_parameters()/1e6:.2f}M params, device={device})")
    print(f"Input Text : {prompt_decoded}")
    print("-" * 108)
    print(
        f"{'tick':>4} | {'mode':<6} | {'pop(Slot 0)':<14} | {'front conveyor [1..4]':<30} | "
        f"{'mid expand [15..18]':<24} | {'back 75% plan [58..63] (pause%)'}"
    )
    print("-" * 108)

    emitted_ids: list[int] = []
    for tok, snap in stream_conveyer_generation(
        model=model,
        prompt_tokens=prompt_tensor,
        max_new_tokens=args.tokens,
        prompt_overlap=args.overlap,
        bake_steps=args.bake_steps,
        temperature=args.temp,
        eos_token_id=tokenizer.eos_id,
    ):
        if not snap.is_warmup:
            emitted_ids.append(int(tok[0].item()))
        mode_str = "WARMUP" if snap.is_warmup else "EMIT  "
        pop_str = repr(tokenizer.id_to_piece(snap.emitted_token).strip() or tokenizer.id_to_piece(snap.emitted_token))
        B = len(snap.slot_top1_tokens)
        front_str = " ".join(tokenizer.id_to_piece(t).strip() for t in snap.slot_top1_tokens[1 : min(5, B)])
        mid_str = " ".join(tokenizer.id_to_piece(t).strip() for t in snap.slot_top1_tokens[min(15, B - 1) : min(19, B)])
        back_str = " ".join(tokenizer.id_to_piece(t).strip() for t in snap.slot_top1_tokens[max(0, B - 5) : B])
        F_slots = min(model.front_slots, B)
        back_pause = (
            100.0 * (1.0 - sum(snap.slot_shift_probs[F_slots:]) / max(1, B - F_slots))
            if B > F_slots
            else 0.0
        )
        print(
            f"{snap.tick:4d} | {mode_str} | {pop_str:<14} | {front_str[:30]:<30} | "
            f"{mid_str[:24]:<24} | {back_str[:22]:<22} ({back_pause:2.0f}% paused)"
        )

    print("-" * 108)
    print(f"FULL OUTPUT: {prompt_decoded}{tokenizer.decode(emitted_ids)}")
    print("=" * 108)


if __name__ == "__main__":
    main()
