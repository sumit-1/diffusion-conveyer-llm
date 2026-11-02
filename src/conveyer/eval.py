from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from conveyer.checkpoint import load_checkpoint
from conveyer.config import ConveyerConfig
from conveyer.data import build_dataset_and_tokenizer, collate_conveyer_batch
from conveyer.infer import generate_conveyer
from conveyer.losses import compute_unrolled_conveyer_loss
from conveyer.model import ConveyerDiffusionLM


# ---------------------------------------------------------------------------
# Benchmark 3: Zero-Shot Commonsense / Story-Coherence Multiple Choice (2-way)
# ---------------------------------------------------------------------------
# Model scores log P(option | context) on unseen story pairs testing:
#   - Physical commonsense (heavy vs light, hot vs cold, broken vs whole)
#   - Emotional / social cause-and-effect (lost toy -> sad, gift -> happy)
#   - Narrative entity & object permanence (tracking who holds what, where they are)
COMMONSENSE_CLOZE_PAIRS: list[tuple[str, str, str, str]] = [
    (
        "emotion_cause",
        "Once upon a time, Tom lost his favorite red ball in the dark woods. Because he lost his ball, Tom felt very",
        " sad and started to cry.",
        " happy and started to laugh.",
    ),
    (
        "emotion_cause",
        "Lily worked all morning to bake a sweet cake for her mom. When her mom smiled and hugged her, Lily felt very",
        " proud and happy.",
        " angry and mean.",
    ),
    (
        "physical_weight",
        "The little mouse tried to lift the huge heavy rock, but the rock was much too",
        " heavy for the tiny mouse to move.",
        " light and floated into the sky.",
    ),
    (
        "physical_temperature",
        "Ben went outside in the snow without his coat or gloves. Soon his hands felt very",
        " cold and freezing.",
        " hot and burning.",
    ),
    (
        "physical_temperature",
        "The soup had just come off the stove and was boiling. Mom told Mia to wait because the soup was too",
        " hot to eat right now.",
        " cold and frozen like ice.",
    ),
    (
        "physical_fragile",
        "The glass cup fell off the high table onto the hard stone floor. With a loud crash, the glass",
        " broke into many small pieces.",
        " bounced high like a rubber ball.",
    ),
    (
        "object_use",
        "The wooden door was locked tight. To unlock the door, Anna took out her small brass",
        " key and turned it in the lock.",
        " spoon and stirred the soup.",
    ),
    (
        "object_use",
        "It started to rain hard outside. Before walking to school, Max opened his big",
        " umbrella to stay dry.",
        " pillow to go to sleep.",
    ),
    (
        "biological_need",
        "The little puppy had been running in the park all day and had not eaten anything. His tummy rumbled because he was very",
        " hungry and wanted some food.",
        " sleepy and wanted a haircut.",
    ),
    (
        "biological_need",
        "After playing all day until night came and the stars shone in the sky, the tired children went to bed to",
        " sleep until morning.",
        " swim in the cold ocean.",
    ),
    (
        "entity_tracking",
        "Sam had a blue car and Lucy had a yellow doll. Sam gave his blue car to Lucy. Now Lucy had the yellow doll and the",
        " blue car.",
        " green bicycle.",
    ),
    (
        "entity_tracking",
        " Lily put her shiny coin inside the small red box and closed the lid. When she opened the red box later, she saw her",
        " shiny coin inside.",
        " big brown horse inside.",
    ),
    (
        "danger_reaction",
        "Lily and a little bunny walked into the green forest. Suddenly a big fierce lion jumped out and roared loudly. The little bunny was",
        " scared and hid behind a tree.",
        " bored and read a newspaper.",
    ),
    (
        "social_repair",
        "Tom accidentally knocked over Mia's tall block tower. Tom felt bad about the accident, so he said,",
        ' "I am sorry, Mia. Let me help you build it again."',
        ' "Thank you for the delicious birthday cake."',
    ),
    (
        "spatial_logic",
        "The bird wanted to reach the nest at the very top of the tall tree, so it flapped its wings and",
        " flew up high into the branches.",
        " dug a deep hole under the mud.",
    ),
    (
        "weather_visibility",
        "At midnight all the lights went out in the room. Because it was pitch dark, Ben could not",
        " see anything around him.",
        " hear the loud thunder outside.",
    ),
]


NOVEL_STORY_PROMPTS: list[tuple[str, list[str]]] = [
    (
        "Once upon a time, Lily and a little bunny walked into the green forest to find the golden key near the locked treasure chest. Suddenly a big lion",
        ["lion", "Lily", "bunny", "key", "chest", "forest", "scared", "roar", "said"],
    ),
    (
        "Once upon a time, there was a little robot named Beep who lived in a noisy workshop. One rainy morning, Beep found a tiny lost kitten by the door. Beep",
        ["Beep", "kitten", "robot", "door", "rain", "warm", "milk", "help"],
    ),
    (
        "Sam and his sister Mia built a huge sandcastle on the beach near the waves. Suddenly, a big wave splashed toward their castle. Sam quickly",
        ["Sam", "Mia", "sand", "castle", "wave", "beach", "water", "wall"],
    ),
    (
        "In a quiet village, an old baker dropped his round loaf of bread and it rolled out the door down the hill. A hungry puppy saw the bread and",
        ["bread", "puppy", "baker", "hill", "chased", "ran", "ate", "mouth"],
    ),
    (
        "Lucy had a box of bright crayons, but she could not find the blue crayon to color the sky. She looked under her bed and",
        ["Lucy", "crayon", "blue", "bed", "sky", "found", "draw", "paper"],
    ),
    (
        "One winter day, Max made a snowman in the yard with a carrot nose. When the warm sun came out in the afternoon, the snowman began to",
        ["Max", "snowman", "sun", "melt", "carrot", "snow", "warm", "water"],
    ),
]


@torch.no_grad()
def evaluate_zero_shot_commonsense(
    model: ConveyerDiffusionLM,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluates zero-shot 2-way multiple-choice accuracy (BLiMP / StoryCloze style) using both:
      1. Causal LM log-probability per token over the continuation.
      2. Conveyor Slot-0 first-token preference on the continuation.
    """
    model.eval()
    correct_lm = 0
    correct_s0 = 0
    margin_sum = 0.0
    details: list[dict[str, Any]] = []

    for category, ctx, good_cont, bad_cont in COMMONSENSE_CLOZE_PAIRS:
        ctx_ids = tokenizer.encode(ctx.strip())
        good_ids = tokenizer.encode(ctx.strip() + good_cont)
        bad_ids = tokenizer.encode(ctx.strip() + bad_cont)
        plen = len(ctx_ids)

        def _score_seq(full_ids: list[int]) -> float:
            inp = torch.tensor([full_ids], dtype=torch.long, device=device)
            logits, _ = model.forward_causal_and_cache(inp, prefix_len=len(full_ids))
            log_probs = F.log_softmax(logits[0, :-1, :].float(), dim=-1)
            targets = inp[0, 1:]
            # Mean log-prob over the continuation tokens [plen-1 : len-1]
            start_idx = max(0, plen - 1)
            cont_lp = log_probs[start_idx:, :].gather(1, targets[start_idx:].unsqueeze(1)).squeeze(1)
            return float(cont_lp.mean().item())

        score_good = _score_seq(good_ids)
        score_bad = _score_seq(bad_ids)
        is_ok = score_good > score_bad
        correct_lm += int(is_ok)
        margin_sum += score_good - score_bad

        # Also check Conveyor Slot-0 log-prob on the first continuation token
        if len(good_ids) > plen and len(bad_ids) > plen:
            p_tensor = torch.tensor([ctx_ids], dtype=torch.long, device=device)
            kv = model.prefill_prefix(p_tensor)
            belt = model.init_empty_belt(1, tick=plen, device=device, deterministic=True)
            out = model.forward_belt(belt, kv_cache=kv)
            s0_lp = F.log_softmax(out.slot0_logits[0].float(), dim=-1)
            s0_good = float(s0_lp[good_ids[plen]].item())
            s0_bad = float(s0_lp[bad_ids[plen]].item())
            correct_s0 += int(s0_good > s0_bad)

        details.append(
            {
                "category": category,
                "correct": is_ok,
                "logprob_good": round(score_good, 3),
                "logprob_bad": round(score_bad, 3),
            }
        )

    n = max(1, len(COMMONSENSE_CLOZE_PAIRS))
    return {
        "commonsense_lm_acc": correct_lm / n,
        "commonsense_slot0_acc": correct_s0 / n,
        "mean_logprob_margin": margin_sum / n,
        "num_pairs": len(COMMONSENSE_CLOZE_PAIRS),
        "details": details,
    }


@torch.no_grad()
def evaluate_prefix_sensitivity_and_generation_quality(
    model: ConveyerDiffusionLM,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int = 80,
) -> dict[str, Any]:
    """
    Measures:
      1. Prefix Sensitivity KL(P(slot0 | prompt) || P(slot0 | shuffled_prompt)):
         In the broken template model this was 0.03 nats (ignored word order).
         A real language model should have KL >> 0.5 nats.
      2. Open-Ended Generation Quality on 6 unseen prompts:
         - Distinct-2 / Distinct-3 n-gram diversity (1.0 = no repetition, <0.5 = looping)
         - 4-gram repetition rate (0.0 = clean prose)
         - Prompt entity/topic retention rate (% of continuations that reference the prompt's new entities/events)
    """
    model.eval()
    kl_list: list[float] = []
    dist2_list: list[float] = []
    dist3_list: list[float] = []
    rep4_list: list[float] = []
    entity_hits = 0
    samples: list[dict[str, str]] = []

    g = torch.Generator().manual_seed(123)
    for prompt_text, expected_keywords in NOVEL_STORY_PROMPTS:
        ids = tokenizer.encode(prompt_text.strip())
        p_true = torch.tensor([ids], dtype=torch.long, device=device)
        kv_true = model.prefill_prefix(p_true)
        belt_true = model.init_empty_belt(1, tick=len(ids), device=device, deterministic=True)
        d_true = F.softmax(model.forward_belt(belt_true, kv_cache=kv_true).slot0_logits[0].float(), dim=-1)

        perm = torch.randperm(len(ids) - 1, generator=g) + 1
        shuf_ids = [ids[0]] + [ids[i] for i in perm.tolist()]
        p_shuf = torch.tensor([shuf_ids], dtype=torch.long, device=device)
        kv_shuf = model.prefill_prefix(p_shuf)
        belt_shuf = model.init_empty_belt(1, tick=len(ids), device=device, deterministic=True)
        d_shuf = F.softmax(model.forward_belt(belt_shuf, kv_cache=kv_shuf).slot0_logits[0].float(), dim=-1)

        kl = float((d_true * (d_true.clamp_min(1e-9).log() - d_shuf.clamp_min(1e-9).log())).sum().item())
        kl_list.append(kl)

        out_ids, _ = generate_conveyer(
            model=model,
            prompt_tokens=p_true,
            max_new_tokens=max_new_tokens,
            prompt_overlap=16,
            temperature=0.0,
            eos_token_id=tokenizer.eos_id,
            deterministic_spawn=True,
        )
        gen_toks = out_ids[0, len(ids) :].tolist()
        gen_text = tokenizer.decode(gen_toks).strip()

        words = [w.strip(".,!?;:\"'()").lower() for w in gen_text.split() if w.strip()]
        if len(words) >= 4:
            bigrams = list(zip(words[:-1], words[1:]))
            trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
            fourgrams = list(zip(words[:-3], words[1:-2], words[2:-1], words[3:]))
            dist2_list.append(len(set(bigrams)) / max(1, len(bigrams)))
            dist3_list.append(len(set(trigrams)) / max(1, len(trigrams)))
            c4 = Counter(fourgrams)
            repeated_4 = sum(v - 1 for v in c4.values() if v > 1)
            rep4_list.append(repeated_4 / max(1, len(fourgrams)))

        low_gen = gen_text.lower()
        hit = any(kw.lower() in low_gen for kw in expected_keywords[:4])
        entity_hits += int(hit)
        samples.append({"prompt": prompt_text, "continuation": gen_text})

    return {
        "prefix_word_order_kl": sum(kl_list) / max(1, len(kl_list)),
        "distinct_2": sum(dist2_list) / max(1, len(dist2_list)),
        "distinct_3": sum(dist3_list) / max(1, len(dist3_list)),
        "repeat_4gram_rate": sum(rep4_list) / max(1, len(rep4_list)),
        "prompt_entity_retention": entity_hits / max(1, len(NOVEL_STORY_PROMPTS)),
        "samples": samples,
    }


@torch.no_grad()
def evaluate_crystallization_curve(
    model: ConveyerDiffusionLM,
    cfg: ConveyerConfig,
    val_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    pad_id: int = 0,
    max_batches: int = 20,
) -> dict[str, Any]:
    """
    Computes the Per-Slot Accuracy, Entropy, and Pause Rate across the 64-slot oven belt
    plus held-out real TinyStories Perplexity (Causal LM & Conveyor Slot-0).
    """
    model.eval()
    B = model.belt_size
    acc_sum = torch.zeros(B, device=device)
    ent_sum = torch.zeros(B, device=device)
    s0_loss_sum = 0.0
    lm_loss_sum = 0.0
    lm_acc_sum = 0.0
    pause_sum = 0.0
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        seq = batch["input_ids"].to(device)
        plen = batch["prompt_lengths"].to(device)
        use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if use_amp else torch.float32, enabled=use_amp):
            m = compute_unrolled_conveyer_loss(
                model=model,
                seq_tokens=seq,
                cfg=cfg,
                prompt_lengths=plen,
                tick_start=int(plen.min().item()),
                pad_token_id=pad_id,
            )
        acc_sum += m.per_slot_acc
        ent_sum += m.per_slot_entropy
        s0_loss_sum += m.slot0_loss
        lm_loss_sum += m.prefix_lm_loss
        lm_acc_sum += m.prefix_lm_acc
        pause_sum += m.mean_plan_pause
        count += 1

    denom = max(count, 1)
    acc_curve = (acc_sum / denom).cpu().tolist()
    ent_curve = (ent_sum / denom).cpu().tolist()
    s0_ce = s0_loss_sum / denom
    lm_ce = lm_loss_sum / denom
    return {
        "slot0_ce": s0_ce,
        "slot0_ppl": math.exp(min(s0_ce, 20.0)),
        "slot0_acc": acc_curve[0],
        "causal_lm_ce": lm_ce,
        "causal_lm_ppl": math.exp(min(lm_ce, 20.0)),
        "causal_lm_acc": lm_acc_sum / denom,
        "mean_plan_pause": pause_sum / denom,
        "per_slot_acc": acc_curve,
        "per_slot_entropy": ent_curve,
    }


@torch.no_grad()
def evaluate_closed_loop_generation(
    model: ConveyerDiffusionLM,
    val_ds: Any,
    tokenizer: Any,
    device: torch.device,
    num_episodes: int = 32,
    prompt_overlap: int = 12,
    bake_steps: int = 64,
    max_new_tokens: int = 24,
) -> dict[str, float]:
    model.eval()
    n_eval = min(num_episodes, len(val_ds))
    first_tok_correct = 0
    total_tok_correct = 0
    total_tok_count = 0
    exact_match = 0

    for i in range(n_eval):
        item = val_ds[i]
        seq = item["input_ids"].to(device)
        plen = int(item["prompt_len"])
        prompt = seq[:plen].unsqueeze(0)

        gt_full = seq[plen:].tolist()
        gt_target: list[int] = []
        for tid in gt_full:
            if tid == tokenizer.pad_id:
                break
            gt_target.append(tid)
            if tid == tokenizer.eos_id:
                break

        if not gt_target:
            continue

        gen_len = min(max_new_tokens, len(gt_target))
        full_out, _ = generate_conveyer(
            model=model,
            prompt_tokens=prompt,
            max_new_tokens=gen_len,
            prompt_overlap=prompt_overlap,
            bake_steps=bake_steps,
            temperature=0.0,
            eos_token_id=tokenizer.eos_id,
            deterministic_spawn=True,
        )
        pred_ids = full_out[0, plen : plen + gen_len].tolist()
        gt_slice = gt_target[:gen_len]

        if len(pred_ids) > 0 and pred_ids[0] == gt_slice[0]:
            first_tok_correct += 1

        matches = sum(1 for p, g in zip(pred_ids, gt_slice) if p == g)
        total_tok_correct += matches
        total_tok_count += len(gt_slice)
        if pred_ids == gt_slice:
            exact_match += 1

    return {
        "first_token_acc": first_tok_correct / max(1, n_eval),
        "token_acc": total_tok_correct / max(1, total_tok_count),
        "exact_seq_acc": exact_match / max(1, n_eval),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Suite for Conveyor-Belt Diffusion LLM")
    parser.add_argument("--ckpt", type=str, default="runs/tinystories/best.pt", help="Path to checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="auto", help="'cpu', 'cuda', or 'auto'")
    parser.add_argument("--episodes", type=int, default=32, help="Number of closed-loop episodes")
    parser.add_argument("--bake-steps", type=int, nargs="+", default=[16, 64], help="Bake depths K to evaluate")
    parser.add_argument("--overlaps", type=int, nargs="+", default=[0, 10], help="Prompt-overlap windows W_pre")
    parser.add_argument("--out-json", type=str, default=None, help="Optional path to save evaluation JSON")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, cfg, ckpt_meta = load_checkpoint(args.ckpt, device=device)
    _, val_ds, tokenizer = build_dataset_and_tokenizer(cfg)
    if hasattr(tokenizer, "content_vocab_mask"):
        model.content_vocab_mask.copy_(getattr(tokenizer, "content_vocab_mask").to(device))
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_conveyer_batch(b, pad_id=tokenizer.pad_id),
    )

    step_num = ckpt_meta.get("step", "?")
    elapsed_h = float(ckpt_meta.get("elapsed_sec", 0.0)) / 3600.0

    cryst = evaluate_crystallization_curve(model, cfg, val_loader, device, pad_id=tokenizer.pad_id)
    acc_c = cryst["per_slot_acc"]
    ent_c = cryst["per_slot_entropy"]
    B = len(acc_c)

    print("=" * 92)
    print(
        f"CONVEYOR-BELT DIFFUSION LLM BENCHMARK SUITE | Ckpt: {args.ckpt} (step {step_num}, {elapsed_h:.2f}h) "
        f"| Params: {model.num_parameters()/1e6:.2f}M | Device: {device}"
    )
    print("=" * 92)
    print(
        f"[1] Held-Out Real TinyStories LM Metrics (10,000 unseen stories, 0% templates):\n"
        f"    - Conveyor Slot-0 Perplexity : {cryst['slot0_ppl']:6.2f}  | Top-1 Next-Token Acc: {cryst['slot0_acc']*100:5.2f}%\n"
        f"    - Causal Prefix LM Perplexity: {cryst['causal_lm_ppl']:6.2f}  | Top-1 Next-Token Acc: {cryst['causal_lm_acc']*100:5.2f}%\n"
        f"    - Back-75% Planning Pause    : {cryst['mean_plan_pause']*100:5.1f}%"
    )
    print("-" * 92)
    print("[2] Per-Slot Crystallization Curve along the 64-Slot Oven Belt (Back k=63 -> Front k=0):")
    sample_slots = [B - 1, (3 * B) // 4, B // 2, B // 4, min(8, B - 1), min(3, B - 1), 0]
    seen: set[int] = set()
    for k in sample_slots:
        if k in seen:
            continue
        seen.add(k)
        zone = "Back (75% Plan)   " if k >= 16 else "Front (25% Stream)"
        print(f"    Slot {k:2d} [{zone}] | Top-1 Acc: {acc_c[k]*100:6.2f}% | Entropy: {ent_c[k]:.3f} nats")

    if cfg.data.dataset == "tinystories":
        cs = evaluate_zero_shot_commonsense(model, tokenizer, device)
        gen_q = evaluate_prefix_sensitivity_and_generation_quality(model, tokenizer, device)
        print("-" * 92)
        print(
            f"[3] Zero-Shot StoryCloze / Commonsense 2-Way Multiple Choice ({cs['num_pairs']} unseen pairs, random=50%):\n"
            f"    - Causal LM Choice Accuracy  : {cs['commonsense_lm_acc']*100:6.2f}% (mean log-prob margin: {cs['mean_logprob_margin']:+.3f} nats)\n"
            f"    - Conveyor Slot-0 Choice Acc : {cs['commonsense_slot0_acc']*100:6.2f}%"
        )
        print("-" * 92)
        print(
            f"[4] Prompt Responsiveness & Open-Ended Generation Quality (6 novel story prompts):\n"
            f"    - Word-Order Sensitivity KL  : {gen_q['prefix_word_order_kl']:6.3f} nats (broken model was 0.032; >0.50 = strong)\n"
            f"    - Prompt Entity Retention    : {gen_q['prompt_entity_retention']*100:6.1f}%\n"
            f"    - Lexical Diversity          : Distinct-2 = {gen_q['distinct_2']*100:5.1f}% | Distinct-3 = {gen_q['distinct_3']*100:5.1f}%\n"
            f"    - 4-gram Repetition Rate     : {gen_q['repeat_4gram_rate']*100:5.2f}% (lower is better; 0% = zero loops)"
        )
        print("-" * 92)
        print("[5] Live Novel Prompt Continutations:")
        for s in gen_q["samples"][:3]:
            print(f"    PROMPT : {s['prompt']}")
            print(f"    OUTPUT : {s['continuation']}\n")
    else:
        cs = {}
        gen_q = {}

    print("-" * 92)
    print(f"{'Bake Depth K':<15} | {'Prompt Overlap W':<18} | {'Pos-0 Acc':<12} | {'Token Acc':<12} | {'Exact Seq Acc':<14}")
    print("-" * 92)

    sweep_results: list[dict[str, Any]] = []
    for k_bake in args.bake_steps:
        for w_ov in args.overlaps:
            res = evaluate_closed_loop_generation(
                model=model,
                val_ds=val_ds,
                tokenizer=tokenizer,
                device=device,
                num_episodes=args.episodes,
                prompt_overlap=w_ov,
                bake_steps=k_bake,
            )
            print(
                f"K = {k_bake:<11d} | W_pre = {w_ov:<10d} | "
                f"{res['first_token_acc']*100:9.2f}%  | {res['token_acc']*100:9.2f}%  | {res['exact_seq_acc']*100:10.2f}%"
            )
            sweep_results.append({"bake_steps": k_bake, "prompt_overlap": w_ov, **res})

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "step": step_num,
                    "crystallization": cryst,
                    "commonsense": cs,
                    "generation_quality": gen_q,
                    "closed_loop": sweep_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
