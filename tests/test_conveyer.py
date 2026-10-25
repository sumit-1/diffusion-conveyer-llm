from __future__ import annotations

import torch

from conveyer.config import ConveyerConfig, ModelConfig, load_config
from conveyer.data.synthetic import SyntheticLookaheadDataset
from conveyer.infer import generate_conveyer, stream_conveyer_generation
from conveyer.losses import build_horizon_weights, compute_unrolled_conveyer_loss, seed_corrupted_belt
from conveyer.model import BeltState, ConveyerDiffusionLM


def _small_model_cfg(
    belt_size: int = 64,
    use_thought_highway: bool = True,
    causal_belt_only: bool = False,
) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_seq_len=256,
        dropout=0.0,
        belt_size=belt_size,
        tau_min=0.5,
        tau_max=4.0,
        tau_gamma=1.5,
        gate_bias_init=1.0,
        use_thought_highway=use_thought_highway,
        causal_belt_only=causal_belt_only,
    )


def test_prefix_kv_cache_chunked_equivalence() -> None:
    """
    Verifies that encoding a prefix in one shot (`prefill_prefix(x_{0:P})`) produces the exact same
    KV cache and downstream belt logits as incremental token-by-token `append_committed_token`.
    """
    torch.manual_seed(123)
    cfg = _small_model_cfg(belt_size=16)
    model = ConveyerDiffusionLM(cfg).eval()

    prefix = torch.randint(1, 60, (2, 10), dtype=torch.long)
    belt = model.init_empty_belt(batch_size=2, tick=10, device=prefix.device, deterministic=True)

    # Full prefill in one shot
    kv_full = model.prefill_prefix(prefix)
    out_full = model.forward_belt(belt, kv_cache=kv_full)

    # Incremental prefill: first 4 tokens, then 1 by 1
    kv_inc = model.prefill_prefix(prefix[:, :4])
    for t in range(4, 10):
        kv_inc = model.append_committed_token(prefix[:, t], kv_inc)
    out_inc = model.forward_belt(belt, kv_cache=kv_inc)

    assert kv_full.seq_len == kv_inc.seq_len == 10
    for l_idx in range(cfg.n_layers):
        assert torch.allclose(kv_full.k_cache[l_idx], kv_inc.k_cache[l_idx], atol=1e-5)
        assert torch.allclose(kv_full.v_cache[l_idx], kv_inc.v_cache[l_idx], atol=1e-5)
    assert torch.allclose(out_full.logits, out_inc.logits, atol=1e-5)


def test_slot0_lexical_anchoring_alpha_zero() -> None:
    """
    Verifies that alpha_0 == 0.0 at Position 0:
    Perturbing `m_thought[:, 0, :]` at Slot 0 must have ZERO effect on the composed dough input at Slot 0,
    forcing Slot 0 to rely 100% on its crisp lexical projection `e_soft[:, 0, :]` before popping out.
    Meanwhile, perturbing `m_thought[:, k, :]` for k > 0 MUST change `dough[:, k, :]`.
    """
    torch.manual_seed(42)
    cfg = _small_model_cfg(belt_size=64, use_thought_highway=True)
    model = ConveyerDiffusionLM(cfg).eval()

    belt1 = model.init_empty_belt(batch_size=2, tick=5, device=torch.device("cpu"), deterministic=True)
    m_mod = belt1.m_thought.clone()
    m_mod[:, 0, :] += 100.0  # Massive perturbation at Slot 0 thought vector
    belt2 = BeltState(e_soft=belt1.e_soft.clone(), m_thought=m_mod, tick=5)

    dough1 = model.compose_dough_inputs(belt1)
    dough2 = model.compose_dough_inputs(belt2)

    assert float(model.alpha_schedule[0].item()) == 0.0
    assert float(model.alpha_schedule[-1].item()) == 1.0
    assert torch.allclose(dough1[:, 0, :], dough2[:, 0, :], atol=1e-6)

    # Now perturb Slot 63 thought vector -> dough[:, 63, :] must change
    m_mod63 = belt1.m_thought.clone()
    m_mod63[:, 63, :] += 10.0
    belt3 = BeltState(e_soft=belt1.e_soft.clone(), m_thought=m_mod63, tick=5)
    dough3 = model.compose_dough_inputs(belt3)
    assert not torch.allclose(dough1[:, 63, :], dough3[:, 63, :], atol=1e-4)


def test_bidirectional_belt_attention_vs_causal_ar() -> None:
    """
    Verifies that in Conveyor mode (`causal_belt_only=False`), Slot 0's output logits actively depend on
    downstream dough balls at the back of the belt (e.g., Slot 63), whereas in AR baseline mode
    (`causal_belt_only=True`), Slot 0 is completely blind to Slots 1..63.
    """
    torch.manual_seed(7)
    # 1. Conveyor bidirectional belt mode
    cfg_bi = _small_model_cfg(belt_size=64, causal_belt_only=False)
    model_bi = ConveyerDiffusionLM(cfg_bi).eval()

    belt_a = model_bi.init_empty_belt(batch_size=1, tick=4, device=torch.device("cpu"), deterministic=True)
    belt_b = BeltState(e_soft=belt_a.e_soft.clone(), m_thought=belt_a.m_thought.clone(), tick=4)
    belt_b.e_soft[:, 63, :] += 5.0  # Perturb back of the belt (Slot 63)

    out_bi_a = model_bi.forward_belt(belt_a)
    out_bi_b = model_bi.forward_belt(belt_b)
    # In bidirectional conveyor mode, Slot 0 uses the mental anchor from Slot 63
    assert not torch.allclose(out_bi_a.slot0_logits, out_bi_b.slot0_logits, atol=1e-5)

    # 2. Causal AR baseline mode
    cfg_ar = _small_model_cfg(belt_size=64, causal_belt_only=True)
    model_ar = ConveyerDiffusionLM(cfg_ar).eval()
    out_ar_a = model_ar.forward_belt(belt_a)
    out_ar_b = model_ar.forward_belt(belt_b)
    # In causal AR mode, Slot 0 cannot see Slot 63
    assert torch.allclose(out_ar_a.slot0_logits, out_ar_b.slot0_logits, atol=1e-6)


def test_left_shift_mechanics_and_elastic_planning_zone() -> None:
    """
    Verifies both:
    1. Strict 1-by-1 conveyor shift (`use_elastic_planning=False`), where slots 1..63 shift into 0..62.
    2. Elastic Two-Zone Planning Belt (`use_elastic_planning=True`), where the front 25% (`0..14`) shifts
       strictly from `1..15`, Slot 15 blends mid-belt sentence expansion (`expand_cand`), and the back 75%
       (`16..63`) pauses elastically via `shift_prob < 1.0`.
    """
    torch.manual_seed(99)
    # 1. Strict conveyor mode
    cfg_strict = _small_model_cfg(belt_size=64)
    cfg_strict.use_elastic_planning = False
    model_strict = ConveyerDiffusionLM(cfg_strict).eval()

    belt = model_strict.init_empty_belt(batch_size=1, tick=0, device=torch.device("cpu"), deterministic=True)
    step_out = model_strict.forward_belt(belt)
    shifted = model_strict.shift_belt_left(step_out, current_tick=0, deterministic_spawn=True)

    assert shifted.tick == 1
    assert shifted.e_soft.shape == (1, 64, cfg_strict.d_model)
    assert torch.allclose(shifted.e_soft[:, :63, :], step_out.e_soft[:, 1:, :], atol=1e-6)
    assert torch.allclose(shifted.m_thought[:, :63, :], step_out.m_updated[:, 1:, :], atol=1e-6)

    # 2. Elastic Two-Zone Planning mode (Front 25% = 16 slots, Back 75% = 48 slots)
    cfg_elastic = _small_model_cfg(belt_size=64)
    cfg_elastic.use_elastic_planning = True
    cfg_elastic.front_conveyer_slots = 16
    model_elastic = ConveyerDiffusionLM(cfg_elastic).eval()

    step_el = model_elastic.forward_belt(belt)
    shifted_el = model_elastic.shift_belt_left(step_el, current_tick=0, deterministic_spawn=True)

    # Front 15 slots (0..14) shift 100% from (1..15)
    assert torch.allclose(shifted_el.e_soft[:, :15, :], step_el.e_soft[:, 1:16, :], atol=1e-6)
    assert torch.allclose(shifted_el.m_thought[:, :15, :], step_el.m_updated[:, 1:16, :], atol=1e-6)
    # Back 75% planning slots have shift_prob < 0.5 (paused > 50% in-place)
    assert float(step_el.shift_prob[:, 16:].mean().item()) < 0.50


def test_prompt_overlap_prebaking_clamps_input_and_emits() -> None:
    """
    Verifies the `n - W_pre` cold-start solution:
    When `prompt_overlap = 6` on a prompt of length `n = 10`:
      - Warm-up runs for ticks `t = 4, 5, 6, 7, 8, 9` (6 warm-up ticks), clamping emitted tokens to `prompt[t]`.
      - Generation runs for `max_new_tokens = 8` ticks (`t = 10..17`), returning a sequence of length `10 + 8 = 18`.
    """
    torch.manual_seed(11)
    cfg = _small_model_cfg(belt_size=32)
    model = ConveyerDiffusionLM(cfg).eval()

    prompt = torch.arange(5, 15, dtype=torch.long).unsqueeze(0)  # length n = 10
    full_seq, snapshots = generate_conveyer(
        model=model,
        prompt_tokens=prompt,
        max_new_tokens=8,
        prompt_overlap=6,
        bake_steps=32,
        temperature=0.0,
        deterministic_spawn=True,
    )

    assert full_seq.shape == (1, 18)
    assert torch.equal(full_seq[:, :10], prompt)
    assert len(snapshots) == 6 + 8

    # First 6 snapshots must be warm-up ticks at t = 4..9 with clamped prompt tokens
    for i in range(6):
        assert snapshots[i].is_warmup is True
        assert snapshots[i].tick == 4 + i
        assert snapshots[i].emitted_token == int(prompt[0, 4 + i].item())

    # Next 8 snapshots must be live emissions at t = 10..17
    for j in range(8):
        snap = snapshots[6 + j]
        assert snap.is_warmup is False
        assert snap.tick == 10 + j
        assert snap.emitted_token == int(full_seq[0, 10 + j].item())


def test_unrolled_bptt_gradient_flow_and_slot63_zero_leakage() -> None:
    """
    Verifies that:
    1. `seed_corrupted_belt` puts zero target signal at Slot 63 (pure noise).
    2. `compute_unrolled_conveyer_loss` backpropagates non-zero gradients into both the lexical embeddings
       and the continuous thought highway (`thought_cand`, `thought_gate`, `slot_emb`, `m_init`).
    """
    torch.manual_seed(21)
    full_cfg = ConveyerConfig(model=_small_model_cfg(belt_size=16, use_thought_highway=True))
    full_cfg.train.unroll_steps = 4
    model = ConveyerDiffusionLM(full_cfg.model).train()

    # Check Slot B-1 has zero target signal in seed_corrupted_belt
    torch.manual_seed(55)
    t_a = torch.ones((1, 16), dtype=torch.long) * 5
    b_a = seed_corrupted_belt(model, t_a, tick_start=2, cold_drop_prob=0.0)
    torch.manual_seed(55)
    t_b = torch.ones((1, 16), dtype=torch.long) * 5
    t_b[:, -1] = 42  # Change target at the last slot (B-1)
    b_b = seed_corrupted_belt(model, t_b, tick_start=2, cold_drop_prob=0.0)
    assert torch.allclose(b_a.e_soft[:, -1, :], b_b.e_soft[:, -1, :], atol=1e-6)

    # Check gradient flow across unrolled BPTT
    seq = torch.randint(4, 50, (3, 40), dtype=torch.long)
    plen = torch.tensor([8, 8, 8], dtype=torch.long)
    metrics = compute_unrolled_conveyer_loss(
        model=model,
        seq_tokens=seq,
        cfg=full_cfg,
        prompt_lengths=plen,
        tick_start=6,
    )
    metrics.loss.backward()

    assert model.tok_emb.weight.grad is not None and model.tok_emb.weight.grad.abs().sum() > 0
    assert model.slot_emb.weight.grad is not None and model.slot_emb.weight.grad.abs().sum() > 0
    assert model.thought_cand.weight.grad is not None and model.thought_cand.weight.grad.abs().sum() > 0
    assert model.thought_gate.weight.grad is not None and model.thought_gate.weight.grad.abs().sum() > 0


def test_synthetic_lookahead_dataset_and_config_loader() -> None:
    cfg = load_config("configs/synthetic.yaml", ["train.max_steps=10", "data.num_train=60"])
    assert cfg.train.max_steps == 10
    assert cfg.data.num_train == 60

    ds = SyntheticLookaheadDataset(num_samples=12, seq_len=96, task="all", vocab_size=128)
    assert len(ds) == 12
    for i in range(6):
        item = ds[i]
        assert item["input_ids"].shape == (96,)
        assert 4 < item["prompt_len"] < 60
        assert item["task_id"] in (0, 1, 2)
