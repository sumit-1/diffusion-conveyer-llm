from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

from conveyer.config import ConveyerConfig
from conveyer.model import BeltState, ConveyerDiffusionLM, PrefixKVCache


@dataclass
class UnrollMetrics:
    loss: torch.Tensor
    slot0_loss: float
    slot0_acc: float
    belt_mean_loss: float
    belt_mean_acc: float
    per_slot_acc: torch.Tensor      # (B,) accuracy at each belt position k=0..B-1
    per_slot_entropy: torch.Tensor  # (B,) entropy at each belt position k=0..B-1
    mean_gate: float
    mean_plan_pause: float = 0.0    # Mean pause rate (1 - s_k) on the back 75% planning belt
    prefix_lm_loss: float = 0.0     # Causal next-token CE loss across the prompt/full sequence
    prefix_lm_acc: float = 0.0      # Causal next-token accuracy across the prompt/full sequence


def build_horizon_weights(
    belt_size: int,
    w_min: float,
    w_beta: float,
    device: torch.device,
) -> torch.Tensor:
    k = torch.arange(belt_size, device=device, dtype=torch.float32)
    ratio = k / max(belt_size - 1, 1)
    w = w_min + (1.0 - w_min) * ((1.0 - ratio) ** w_beta)
    return w / w.sum()


def gather_belt_targets(
    seq_tokens: torch.Tensor,
    tick: int,
    offsets: torch.Tensor,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gathers target tokens `x_{tick + offsets[k]}` for each belt slot `k = 0..B-1`:
      - Front 25% Conveyor Slots (k = 0 .. F-1): `offsets[k] = k` (immediate token stream `t + k`)
      - Back 75% Planning Slots (k = F .. B-1): `offsets[k] = F + (k - F) * stride` (longer-range sentence plan!)
    Returns `(targets, in_bounds_mask)` of shape `(N, B)`.
    """
    bsz, seq_len = seq_tokens.shape
    device = seq_tokens.device
    pos = tick + offsets.to(device)  # (B,)
    in_bounds = (pos >= 0) & (pos < seq_len)  # (B,)
    clamped_pos = pos.clamp(min=0, max=seq_len - 1).unsqueeze(0).expand(bsz, -1)  # (N, B)
    gathered = torch.gather(seq_tokens, dim=1, index=clamped_pos)
    pad_tensor = torch.full_like(gathered, pad_token_id)
    targets = torch.where(in_bounds.unsqueeze(0), gathered, pad_tensor)
    return targets, in_bounds.unsqueeze(0).expand(bsz, -1)


def seed_corrupted_belt(
    model: ConveyerDiffusionLM,
    target_tokens: torch.Tensor,
    tick_start: int,
    sigma_max: float = 3.5,
    sigma_eta: float = 1.25,
    cold_drop_prob: float = 0.85,
    prompt_lengths: Optional[torch.Tensor] = None,
) -> BeltState:
    bsz, B = target_tokens.shape
    device = target_tokens.device
    V = model.vocab_size

    k = torch.arange(B, device=device, dtype=torch.float32)
    ratio = k / max(B - 1, 1)

    signal_strength = 3.0 * ((1.0 - ratio) ** (sigma_eta * 2.0))
    signal_strength[-1] = 0.0
    sigma_k = 0.5 + sigma_max * (ratio ** sigma_eta)

    one_hot = F.one_hot(target_tokens.clamp(min=0, max=V - 1), num_classes=V).float()
    noise = torch.randn((bsz, B, V), device=device, dtype=torch.float32)
    tau = model.tau_schedule[:B].view(1, B, 1)

    noisy_logits = signal_strength.view(1, B, 1) * one_hot + sigma_k.view(1, B, 1) * noise
    tilde_p = F.softmax(noisy_logits / tau, dim=-1).type_as(model.tok_emb.weight)
    e_soft = torch.matmul(tilde_p, model.tok_emb.weight)

    _, m_thought = model.spawn_raw_dough(batch_size=bsz, num_slots=B, device=device)

    effective_cold_prob = 1.0 if (not model.training and cold_drop_prob > 0.0) else cold_drop_prob
    if effective_cold_prob > 0.0:
        if effective_cold_prob >= 1.0:
            cold_mask = torch.ones((bsz, 1, 1), device=device, dtype=torch.bool)
        else:
            cold_mask = torch.rand((bsz, 1, 1), device=device) < effective_cold_prob
        e_cold, m_cold = model.spawn_raw_dough(batch_size=bsz, num_slots=B, device=device)
        e_soft = torch.where(cold_mask, e_cold, e_soft)
        m_thought = torch.where(cold_mask, m_cold, m_thought)
        uniform_p = torch.full_like(tilde_p, 1.0 / V)
        tilde_p = torch.where(cold_mask, uniform_p, tilde_p)

    if prompt_lengths is not None:
        offsets = model.get_slot_target_offsets(active_slots=B).to(device)
        pos = tick_start + offsets.unsqueeze(0)  # (1, B)
        is_prompt = pos < prompt_lengths.unsqueeze(1)  # (N, B)
        prompt_emb = model.tok_emb(target_tokens.clamp(min=0, max=V - 1))
        e_soft = torch.where(is_prompt.unsqueeze(-1), prompt_emb, e_soft)

    return BeltState(
        e_soft=e_soft,
        m_thought=m_thought,
        tick=tick_start,
        probs=tilde_p,
        logits=noisy_logits,
    )


def apply_belt_recovery_perturbation(
    model: ConveyerDiffusionLM,
    belt: BeltState,
    corrupt_rate: float = 0.20,
    prompt_mask: Optional[torch.Tensor] = None,
) -> BeltState:
    """
    Belt Recovery Training:
    Randomly corrupts a subset of slots on the active conveyor belt mid-unroll (replacing their
    soft embeddings with random vocabulary embeddings or noise), forcing the next forward pass
    to actively repair and re-synchronize the belt from surrounding slots and the prefix!
    """
    if corrupt_rate <= 0.0 or not model.training:
        return belt
    bsz, B, _ = belt.e_soft.shape
    device = belt.e_soft.device
    # Select ~corrupt_rate of slots per batch item to perturb
    slot_drop = torch.rand((bsz, B), device=device) < corrupt_rate
    if prompt_mask is not None:
        slot_drop = slot_drop & (~prompt_mask)

    rand_toks = torch.randint(0, model.vocab_size, (bsz, B), device=device)
    rand_emb = model.tok_emb(rand_toks)
    e_noise, m_noise = model.spawn_raw_dough(batch_size=bsz, num_slots=B, device=device)
    corrupted_e = 0.5 * rand_emb + 0.5 * e_noise

    mask_exp = slot_drop.unsqueeze(-1)
    e_repaired = torch.where(mask_exp, corrupted_e, belt.e_soft)
    m_repaired = torch.where(mask_exp, m_noise, belt.m_thought)
    return BeltState(
        e_soft=e_repaired,
        m_thought=m_repaired,
        tick=belt.tick,
        probs=belt.probs,
        logits=belt.logits,
        gates=belt.gates,
        shift_probs=belt.shift_probs,
    )


def compute_unrolled_conveyer_loss(
    model: ConveyerDiffusionLM,
    seq_tokens: torch.Tensor,
    cfg: ConveyerConfig,
    prompt_lengths: Optional[torch.Tensor] = None,
    tick_start: Optional[int] = None,
    pad_token_id: int = 0,
) -> UnrollMetrics:
    bsz, seq_len = seq_tokens.shape
    device = seq_tokens.device
    B = model.belt_size
    F_slots = min(model.front_slots, B)
    U = min(cfg.train.unroll_steps, max(1, seq_len - F_slots))
    offsets = model.get_slot_target_offsets(active_slots=B)

    if tick_start is None:
        max_t0 = max(1, seq_len - F_slots - U + 1)
        if prompt_lengths is not None:
            min_prompt = int(prompt_lengths.min().item())
            max_prompt = int(prompt_lengths.max().item())
            low = max(1, min_prompt - min(U - 2, cfg.infer.prompt_overlap))
            if torch.rand((), device=device).item() < 0.5:
                high = max(low + 1, min(max_t0 + 1, max_prompt + 16))
            else:
                high = max(low + 1, max_t0 + 1)
            tick_start = int(torch.randint(low, high, (1,), device=device).item())
        else:
            tick_start = int(torch.randint(1, max_t0 + 1, (1,), device=device).item())

    # 1. Causal pass over the full sequence + extract causal KV cache
    prefix_lm_weight = getattr(cfg.train, "prefix_lm_weight", 1.0)
    full_kv: Optional[PrefixKVCache] = None
    if prefix_lm_weight > 0.0 and seq_len > 1:
        causal_logits, full_kv = model.forward_causal_and_cache(seq_tokens, prefix_len=seq_len)
        shift_logits = causal_logits[:, :-1, :].contiguous()
        shift_targets = seq_tokens[:, 1:].contiguous()
        prefix_lm_loss = F.cross_entropy(
            shift_logits.view(-1, model.vocab_size),
            shift_targets.view(-1),
            ignore_index=pad_token_id,
        )
        with torch.no_grad():
            valid_tok = shift_targets != pad_token_id
            n_valid = valid_tok.float().sum().clamp(min=1.0)
            pred_tok = shift_logits.argmax(dim=-1)
            prefix_lm_acc_t = ((pred_tok == shift_targets) & valid_tok).float().sum() / n_valid
        kv_cache = PrefixKVCache(
            k_cache=[k[:, :, :tick_start, :] for k in full_kv.k_cache],
            v_cache=[v[:, :, :tick_start, :] for v in full_kv.v_cache],
            seq_len=tick_start,
        )
    else:
        prefix_tokens = seq_tokens[:, :tick_start]
        kv_cache = model.prefill_prefix(prefix_tokens)
        prefix_lm_loss = torch.tensor(0.0, device=device)
        prefix_lm_acc_t = torch.tensor(0.0, device=device)

    # 2. Seed initial 64-slot belt at tick_start using slot target offsets
    init_targets, _ = gather_belt_targets(seq_tokens, tick_start, offsets, pad_token_id=pad_token_id)

    belt = seed_corrupted_belt(
        model=model,
        target_tokens=init_targets,
        tick_start=tick_start,
        sigma_max=cfg.train.seed_sigma_max,
        sigma_eta=cfg.train.seed_sigma_eta,
        cold_drop_prob=0.85,
        prompt_lengths=prompt_lengths,
    )

    horizon_w = build_horizon_weights(
        belt_size=B,
        w_min=cfg.train.loss_w_min,
        w_beta=cfg.train.loss_w_beta,
        device=device,
    )

    total_loss = torch.tensor(0.0, device=device)
    prev_logits: Optional[torch.Tensor] = None
    slot0_loss_sum_t = torch.tensor(0.0, device=device)
    slot0_acc_sum_t = torch.tensor(0.0, device=device)
    s0_valid_ticks_t = torch.tensor(0.0, device=device)
    belt_loss_sum_t = torch.tensor(0.0, device=device)
    belt_acc_sum_t = torch.tensor(0.0, device=device)
    gate_sum_t = torch.tensor(0.0, device=device)
    pause_sum_t = torch.tensor(0.0, device=device)
    valid_ticks = 0
    dagger_diverged = False

    per_slot_correct = torch.zeros(B, device=device)
    per_slot_count = torch.zeros(B, device=device)
    per_slot_entropy_sum = torch.zeros(B, device=device)

    for u in range(U):
        t = tick_start + u
        if t >= seq_len:
            break

        targets, in_bounds = gather_belt_targets(seq_tokens, t, offsets, pad_token_id=pad_token_id)
        slot_positions = t + offsets.to(device).unsqueeze(0)  # (1, B)

        # Mid-unroll Belt Recovery Training: occasionally inject wrong/noisy thoughts onto the belt
        # so the next forward pass learns to actively repair the belt!
        if u > 0 and model.training and cfg.train.belt_corrupt_prob > 0.0:
            if torch.rand(()) < 0.35:
                p_mask = (slot_positions < prompt_lengths.unsqueeze(1)) if prompt_lengths is not None else None
                belt = apply_belt_recovery_perturbation(
                    model=model,
                    belt=belt,
                    corrupt_rate=cfg.train.belt_corrupt_prob,
                    prompt_mask=p_mask,
                )

        step_out = model.forward_belt(belt=belt, kv_cache=kv_cache)

        valid_mask = in_bounds & (targets != pad_token_id)  # (N, B)
        if prompt_lengths is not None:
            is_gen = slot_positions >= prompt_lengths.unsqueeze(1)
            valid_mask = valid_mask & is_gen

        if cfg.model.causal_belt_only:
            ar_mask = torch.zeros_like(valid_mask)
            ar_mask[:, 0] = valid_mask[:, 0]
            valid_mask = ar_mask

        ce_per_token = F.cross_entropy(
            step_out.logits.reshape(-1, model.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).view(bsz, B)

        w_broadcast = horizon_w.unsqueeze(0) * valid_mask.float()
        w_denom = w_broadcast.sum().clamp(min=1e-6)
        tick_ce_loss = (ce_per_token * w_broadcast).sum() / w_denom

        probs_clamped = step_out.probs.clamp(min=1e-8)
        entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
        if cfg.train.entropy_reg > 0.0 and not cfg.model.causal_belt_only:
            back_weights = (torch.arange(B, device=device, dtype=torch.float32) / max(B - 1, 1)).unsqueeze(0)
            ent_bonus = (entropy * back_weights * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
            tick_loss = tick_ce_loss - cfg.train.entropy_reg * ent_bonus
        else:
            tick_loss = tick_ce_loss

        # Shift-Consistency on the Front Conveyor Stream (0 .. F_slots - 2)
        if u > 0 and prev_logits is not None and not cfg.model.causal_belt_only and F_slots > 1:
            p_prev = F.softmax(prev_logits[:, 1:F_slots, :].detach(), dim=-1)
            log_p_curr = F.log_softmax(step_out.logits[:, : F_slots - 1, :], dim=-1)
            kl_shift = F.kl_div(log_p_curr, p_prev, reduction="none").sum(dim=-1)
            cons_mask = valid_mask[:, : F_slots - 1].float()
            cons_loss = (kl_shift * cons_mask).sum() / cons_mask.sum().clamp(min=1.0)
            tick_loss = tick_loss + 0.08 * cons_loss

        prev_logits = step_out.logits
        total_loss = total_loss + tick_loss
        valid_ticks += 1

        with torch.no_grad():
            valid_f = valid_mask.float()
            preds = step_out.logits.argmax(dim=-1)
            correct = (preds == targets).float() * valid_f

            s0_valid = valid_f[:, 0]
            s0_sum = s0_valid.sum()
            has_s0 = (s0_sum > 0.0).float()
            s0_denom = s0_sum.clamp(min=1.0)
            slot0_loss_sum_t = slot0_loss_sum_t + has_s0 * ((ce_per_token[:, 0] * s0_valid).sum() / s0_denom)
            slot0_acc_sum_t = slot0_acc_sum_t + has_s0 * (correct[:, 0].sum() / s0_denom)
            s0_valid_ticks_t = s0_valid_ticks_t + has_s0

            all_denom = valid_f.sum().clamp(min=1.0)
            belt_loss_sum_t = belt_loss_sum_t + (ce_per_token * valid_f).sum() / all_denom
            belt_acc_sum_t = belt_acc_sum_t + correct.sum() / all_denom
            gate_sum_t = gate_sum_t + step_out.gate.mean()
            if B > F_slots:
                pause_sum_t = pause_sum_t + (1.0 - step_out.shift_prob[:, F_slots:]).mean()

            per_slot_correct += correct.sum(dim=0)
            per_slot_count += valid_f.sum(dim=0)
            per_slot_entropy_sum += entropy.mean(dim=0)

        if u < U - 1 and (t + 1) < seq_len:
            # Prefix Recovery (DAgger-style Scheduled Sampling on final ticks, or fast slice from full_kv):
            use_dagger = (
                model.training
                and (u >= max(0, U - 4) or full_kv is None)
                and (torch.rand(()) < cfg.train.recovery_prob)
            )
            if use_dagger:
                pred_s0 = step_out.slot0_logits.argmax(dim=-1)
                if prompt_lengths is not None:
                    is_p = t < prompt_lengths
                    committed_x_t = torch.where(is_p, seq_tokens[:, t], pred_s0)
                else:
                    committed_x_t = pred_s0
                dagger_diverged = True

            if not dagger_diverged and full_kv is not None:
                next_len = t + 1
                kv_cache = PrefixKVCache(
                    k_cache=[k[:, :, :next_len, :] for k in full_kv.k_cache],
                    v_cache=[v[:, :, :next_len, :] for v in full_kv.v_cache],
                    seq_len=next_len,
                )
            else:
                if not use_dagger:
                    committed_x_t = seq_tokens[:, t]
                kv_cache = model.append_committed_token(committed_x_t, kv_cache)

            next_targets, _ = gather_belt_targets(seq_tokens, t + 1, offsets, pad_token_id=pad_token_id)
            if prompt_lengths is not None:
                next_pos = (t + 1) + offsets.to(device).unsqueeze(0)
                clamp_mask = next_pos < prompt_lengths.unsqueeze(1)
            else:
                clamp_mask = None

            belt = model.shift_belt_left(
                step_out=step_out,
                current_tick=t,
                deterministic_spawn=False,
                clamp_tokens=next_targets if clamp_mask is not None else None,
                clamp_mask=clamp_mask,
            )
            if cfg.train.no_belt_bptt:
                belt = belt.detach()
                kv_cache = kv_cache.detach()

    denom = max(valid_ticks, 1)
    s0_denom_ticks = float(s0_valid_ticks_t.item()) if float(s0_valid_ticks_t.item()) > 0.0 else 1.0
    per_slot_acc = per_slot_correct / per_slot_count.clamp(min=1.0)
    per_slot_entropy = per_slot_entropy_sum / denom
    conveyer_loss = total_loss / denom
    combined_loss = conveyer_loss + prefix_lm_weight * prefix_lm_loss

    return UnrollMetrics(
        loss=combined_loss,
        slot0_loss=float(slot0_loss_sum_t.item()) / s0_denom_ticks,
        slot0_acc=float(slot0_acc_sum_t.item()) / s0_denom_ticks,
        belt_mean_loss=float(belt_loss_sum_t.item()) / denom,
        belt_mean_acc=float(belt_acc_sum_t.item()) / denom,
        per_slot_acc=per_slot_acc,
        per_slot_entropy=per_slot_entropy,
        mean_gate=float(gate_sum_t.item()) / denom,
        mean_plan_pause=float(pause_sum_t.item()) / denom,
        prefix_lm_loss=float(prefix_lm_loss.item()),
        prefix_lm_acc=float(prefix_lm_acc_t.item()),
    )
