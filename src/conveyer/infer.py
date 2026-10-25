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

    if recent_tokens and repetition_penalty > 1.0:
        for tid in set(recent_tokens[-6:]):
            if 0 <= tid < adj_logits.size(-1):
                val = adj_logits[:, tid]
                adj_logits[:, tid] = torch.where(val > 0, val / repetition_penalty, val * repetition_penalty)
        # Only block 3-in-a-row identical token stammers (so legitimate words after prompt or across clauses are never clipped!)
        if len(recent_tokens) >= 2 and recent_tokens[-1] == recent_tokens[-2]:
            last_tid = recent_tokens[-1]
            if 0 <= last_tid < adj_logits.size(-1):
                adj_logits[:, last_tid] -= 3.0

    if temperature <= 1e-5:
        return adj_logits.argmax(dim=-1)
    scaled = adj_logits / temperature
    k = min(24, scaled.size(-1))
    top_vals, _ = torch.topk(scaled, k, dim=-1)
    min_top = top_vals[:, -1].unsqueeze(-1)
    scaled = torch.where(scaled < min_top, torch.full_like(scaled, -1e9), scaled)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


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
) -> Iterator[tuple[torch.Tensor, TickSnapshot]]:
    """
    Streams generation tick-by-tick using the `n - W_pre` Prompt-Overlap Pre-Baking protocol:
      - Input at every tick t : [Committed Prefix x_{0..t-1}] + [64-Slot Conveyor/Planning Belt]
      - Output at every tick t: [Popped-out Confirmed Token x_t at Slot 0] + [Updated 64-Slot Belt]
    """
    model.eval()
    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)

    bsz, n = prompt_tokens.shape
    device = prompt_tokens.device
    B = min(bake_steps, model.belt_size) if bake_steps is not None else model.belt_size
    B = max(1, B)

    max_overlap = min(model.front_slots, B - 1) if B > 1 else 0
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

    offsets = model.get_slot_target_offsets(active_slots=B).to(device)  # (B,)

    # Clamp any belt slots at tick t_start that fall inside the known input prompt (t_start + offsets[k] < n)
    if w_pre > 0:
        slot_pos = t_start + offsets  # (B,)
        in_prompt = slot_pos < n
        clamped_pos = slot_pos.clamp(min=0, max=n - 1).unsqueeze(0).expand(bsz, -1)
        prompt_gathered = torch.gather(prompt_tokens, dim=1, index=clamped_pos)
        overlap_emb = model.tok_emb(prompt_gathered)
        e_init = torch.where(in_prompt.view(1, B, 1), overlap_emb, belt.e_soft)
        belt = BeltState(e_soft=e_init, m_thought=belt.m_thought, tick=t_start)

    total_ticks = w_pre + max_new_tokens
    recent_emitted: list[int] = []

    for step_idx in range(total_ticks):
        t = t_start + step_idx
        is_warmup = t < n

        # Single forward pass across all active slots on the belt
        step_out: BeltStepOutput = model.forward_belt(belt=belt, kv_cache=kv_cache, active_slots=B)

        raw_pred_tok = sample_slot0_token(
            step_out.slot0_logits,
            temperature=temperature,
            recent_tokens=recent_emitted if not is_warmup else None,
            repetition_penalty=1.08 if temperature > 1e-5 else 1.0,
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

        # Commit popped-out token x_t to frozen prefix KV cache
        kv_cache = model.append_committed_token(emitted_tok, kv_cache)

        # Shift belt left (with elastic 75% back planning zone) and clamp any remaining prompt tokens at t + 1
        next_t = t + 1
        if next_t < n:
            next_pos = next_t + offsets  # (B,)
            clamp_mask = (next_pos < n).unsqueeze(0).expand(bsz, -1)  # (N, B)
            clamped_idx = next_pos.clamp(min=0, max=n - 1).unsqueeze(0).expand(bsz, -1)
            clamp_tokens = torch.gather(prompt_tokens, dim=1, index=clamped_idx)
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
