from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Optional
import torch
from torch.utils.data import DataLoader, Subset

from conveyer.checkpoint import load_checkpoint_payload, save_checkpoint
from conveyer.config import ConveyerConfig, load_config
from conveyer.data import TokenizerProtocol, build_dataset_and_tokenizer, collate_conveyer_batch
from conveyer.infer import generate_conveyer
from conveyer.losses import compute_unrolled_conveyer_loss
from conveyer.model import ConveyerDiffusionLM


def _query_gpu_temp_c() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            timeout=2.0,
            stderr=subprocess.DEVNULL,
        )
        return int(out.decode("utf-8", errors="ignore").strip().splitlines()[0])
    except Exception:
        return None


def get_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate_validation(
    model: ConveyerDiffusionLM,
    val_loader: DataLoader[dict[str, torch.Tensor]],
    cfg: ConveyerConfig,
    device: torch.device,
    pad_id: int = 0,
    max_batches: int = 25,
) -> dict[str, Any]:
    model.eval()
    B = model.belt_size
    loss_sum = 0.0
    s0_loss_sum = 0.0
    s0_acc_sum = 0.0
    lm_loss_sum = 0.0
    lm_acc_sum = 0.0
    belt_acc_sum = 0.0
    per_slot_acc_sum = torch.zeros(B, device=device)
    per_slot_ent_sum = torch.zeros(B, device=device)
    count = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        seq = batch["input_ids"].to(device)
        plen = batch["prompt_lengths"].to(device)
        t_eval = max(1, int(plen.min().item()) - min(4, cfg.train.unroll_steps // 2))
        use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if use_amp else torch.float32, enabled=use_amp):
            m = compute_unrolled_conveyer_loss(
                model=model,
                seq_tokens=seq,
                cfg=cfg,
                prompt_lengths=plen,
                tick_start=t_eval,
                pad_token_id=pad_id,
            )
        loss_sum += float(m.loss.item())
        s0_loss_sum += m.slot0_loss
        s0_acc_sum += m.slot0_acc
        lm_loss_sum += m.prefix_lm_loss
        lm_acc_sum += m.prefix_lm_acc
        belt_acc_sum += m.belt_mean_acc
        per_slot_acc_sum += m.per_slot_acc
        per_slot_ent_sum += m.per_slot_entropy
        count += 1

    model.train()
    denom = max(count, 1)
    s0_ce = s0_loss_sum / denom
    lm_ce = lm_loss_sum / denom
    return {
        "val_loss": loss_sum / denom,
        "val_slot0_ce": s0_ce,
        "val_slot0_ppl": math.exp(min(s0_ce, 20.0)),
        "val_slot0_acc": s0_acc_sum / denom,
        "val_lm_ce": lm_ce,
        "val_lm_ppl": math.exp(min(lm_ce, 20.0)),
        "val_lm_acc": lm_acc_sum / denom,
        "val_belt_acc": belt_acc_sum / denom,
        "val_per_slot_acc": (per_slot_acc_sum / denom).cpu().tolist(),
        "val_per_slot_entropy": (per_slot_ent_sum / denom).cpu().tolist(),
    }


def _prune_old_milestones(out_dir: Path, keep_last: int = 8) -> None:
    milestones = sorted(out_dir.glob("ckpt_step_*.pt"), key=lambda p: p.stat().st_mtime)
    if len(milestones) > keep_last:
        for old_p in milestones[: len(milestones) - keep_last]:
            try:
                old_p.unlink()
                prev_p = old_p.with_suffix(old_p.suffix + ".prev")
                if prev_p.exists():
                    prev_p.unlink()
            except OSError:
                pass


@torch.no_grad()
def _sample_preview(
    model: ConveyerDiffusionLM,
    tokenizer: TokenizerProtocol,
    device: torch.device,
    prompt_text: str,
    max_new_tokens: int = 200,
    injected_word: Optional[str] = None,
) -> str:
    model.eval()
    p_ids = torch.tensor([tokenizer.encode(prompt_text.strip())], dtype=torch.long, device=device)
    inj_ids: Optional[list[int]] = None
    if injected_word is not None:
        enc_w = tokenizer.encode(" " + injected_word.strip())[1:]  # skip <BOS>
        if enc_w:
            inj_ids = (enc_w * 8)[:8]
    out_ids, _ = generate_conveyer(
        model=model,
        prompt_tokens=p_ids,
        max_new_tokens=max_new_tokens,
        prompt_overlap=16,
        temperature=0.0,
        eos_token_id=tokenizer.eos_id,
        injected_conveyer_tokens=inj_ids,
    )
    gen_part = tokenizer.decode(out_ids[0, p_ids.size(1) :].tolist())
    model.train()
    clean = gen_part.replace("\n", " ").replace("\ufffd", "").strip()
    return clean.encode("ascii", errors="ignore").decode("ascii")


def _sanitize_jsonl(log_path: Path) -> None:
    """Strips any trailing NUL bytes left by an abrupt OS crash so the log remains valid JSONL."""
    if not log_path.exists():
        return
    raw = log_path.read_bytes().replace(b"\x00", b"")
    lines = [ln for ln in raw.splitlines() if ln.strip().startswith(b"{") and ln.strip().endswith(b"}")]
    log_path.write_bytes(b"\n".join(lines) + (b"\n" if lines else b""))


def train_model(
    cfg: ConveyerConfig,
    init_ckpt: str | Path | None = None,
    resume_ckpt: str | Path | None = None,
) -> dict[str, Any]:
    torch.manual_seed(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds, tokenizer = build_dataset_and_tokenizer(cfg)
    if cfg.train.overfit_batches > 0:
        n_sub = min(len(train_ds), cfg.train.overfit_batches * cfg.train.batch_size)
        train_ds = Subset(train_ds, list(range(n_sub)))
        val_ds = train_ds

    collate_fn = lambda b: collate_conveyer_batch(b, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=(cfg.train.overfit_batches == 0),
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = ConveyerDiffusionLM(cfg.model).to(device)
    if hasattr(tokenizer, "content_vocab_mask"):
        model.content_vocab_mask.copy_(getattr(tokenizer, "content_vocab_mask").to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
    )

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "train_log.jsonl"
    _sanitize_jsonl(log_jsonl)

    start_step = 1
    best_val_loss = float("inf")
    prior_elapsed = 0.0

    # Resolve resume vs warm-start initialization (with automatic fallback to .prev or ckpt_step_*.pt)
    resume_target: Optional[Path] = None
    if resume_ckpt is not None:
        if str(resume_ckpt).lower() == "auto":
            resume_target = out_dir / "last.pt"
        else:
            resume_target = Path(resume_ckpt)

    if resume_target is not None:
        try:
            ckpt_data, used_path = load_checkpoint_payload(resume_target, map_location="cpu")
            model.load_state_dict(ckpt_data["model_state"], strict=False)
            if "optimizer_state" in ckpt_data:
                try:
                    optimizer.load_state_dict(ckpt_data["optimizer_state"])
                except Exception as e:
                    print(f"[ConveyerTrain] Warning: could not restore optimizer state ({e})", flush=True)
            start_step = int(ckpt_data.get("step", 0)) + 1
            prev_cfg = ckpt_data.get("config", {}).get("model", {})
            if getattr(cfg.model, "use_thought_reservoir", False) and not prev_cfg.get("use_thought_reservoir", False):
                best_val_loss = float("inf")
                print("[ConveyerTrain] Transitioning from Conveyor Belt to English Thought Reservoir mode; resetting best_val_loss.", flush=True)
            else:
                best_val_loss = float(
                    ckpt_data.get("best_val_loss") or ckpt_data.get("metrics", {}).get("val_loss", float("inf"))
                )
            prior_elapsed = float(ckpt_data.get("elapsed_sec", 0.0))
            print(
                f"[ConveyerTrain] Resumed from {used_path} at step {start_step} "
                f"(best_val_loss={best_val_loss:.4f}, prior_elapsed={prior_elapsed/3600:.2f}h)",
                flush=True,
            )
            del ckpt_data
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[ConveyerTrain] No readable resume checkpoint in {out_dir} ({e}); starting from step 1.", flush=True)
    elif init_ckpt is not None and Path(init_ckpt).exists():
        ckpt_data, used_path = load_checkpoint_payload(init_ckpt, map_location="cpu")
        model.load_state_dict(ckpt_data["model_state"], strict=False)
        print(f"[ConveyerTrain] Warm-started weights from {used_path}", flush=True)
        del ckpt_data

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    mode_desc = (
        "1-Pass AR Baseline"
        if cfg.model.causal_belt_only
        else (
            f"{cfg.model.belt_size}-Slot Duplex-SimMT Content-Only Conveyer + Articulation (K={getattr(cfg.model, 'duplex_k_slots', 4)})"
            if getattr(cfg.model, "use_thought_reservoir", False) and getattr(cfg.model, "use_duplex_frame", False)
            else (
                f"{cfg.model.belt_size}-Slot English Thought Reservoir (In-Place 4-Register + Front Articulation)"
                if getattr(cfg.model, "use_thought_reservoir", False)
                else (
                    "64-Slot Conveyor (Soft-Vocab Only)"
                    if not cfg.model.use_thought_highway
                    else "64-Slot Dual-Channel Conveyor (Soft-Vocab + Gated Thought)"
                )
            )
        )
    )
    max_hours = float(getattr(cfg.train, "max_hours", 0.0))
    save_every = max(1, int(getattr(cfg.train, "save_every", 500)))
    milestone_every = save_every * 2

    print(
        f"[ConveyerTrain] Mode: {mode_desc} | Params: {model.num_parameters()/1e6:.2f}M | "
        f"Device: {device} | Train stories: {len(train_ds):,} | Val stories: {len(val_ds):,} | "
        f"Unroll U={cfg.train.unroll_steps} | Belt B={cfg.model.belt_size} | "
        f"max_steps={cfg.train.max_steps:,} | max_hours={max_hours:.1f}h | save_every={save_every}",
        flush=True,
    )

    train_iter = iter(train_loader)
    last_val_metrics: dict[str, Any] = {}
    t0 = time.time()

    model.train()
    for step in range(start_step, cfg.train.max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr = get_lr(step - 1, cfg.train.max_steps, cfg.train.lr, cfg.train.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        seq = batch["input_ids"].to(device)
        plen = batch["prompt_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            metrics = compute_unrolled_conveyer_loss(
                model=model,
                seq_tokens=seq,
                cfg=cfg,
                prompt_lengths=plen,
                pad_token_id=tokenizer.pad_id,
            )
        metrics.loss.backward()
        if cfg.train.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

        cooldown_ms = int(getattr(cfg.train, "step_cooldown_ms", 0))
        if cooldown_ms > 0:
            if device.type == "cuda":
                torch.cuda.synchronize()
            time.sleep(cooldown_ms / 1000.0)

        total_elapsed = prior_elapsed + (time.time() - t0)

        if step % cfg.train.log_every == 0 or step == start_step:
            acc_back = float(metrics.per_slot_acc[-1].item())
            acc_mid = float(metrics.per_slot_acc[model.belt_size // 2].item())
            steps_done = max(1, step - start_step + 1)
            sec_per_step = (time.time() - t0) / steps_done
            gpu_temp = _query_gpu_temp_c() if device.type == "cuda" else None
            temp_str = f" | {gpu_temp}C" if gpu_temp is not None else ""
            print(
                f"step {step:6d}/{cfg.train.max_steps} | loss {metrics.loss.item():.4f} | "
                f"lm_ce {metrics.prefix_lm_loss:.3f} (acc {metrics.prefix_lm_acc:.3f}) | "
                f"s0_ce {metrics.slot0_loss:.3f} (acc {metrics.slot0_acc:.3f}) | "
                f"belt(0/m/63) {metrics.slot0_acc:.2f}/{acc_mid:.2f}/{acc_back:.2f} | "
                f"pause {metrics.mean_plan_pause*100:.0f}% | lr {lr:.2e} | "
                f"{sec_per_step:.2f}s/it{temp_str} | {total_elapsed/3600:.2f}h",
                flush=True,
            )
            max_temp = int(getattr(cfg.train, "max_gpu_temp_c", 80))
            if gpu_temp is not None and max_temp > 0 and gpu_temp >= max_temp:
                torch.cuda.synchronize()
                while True:
                    print(
                        f"  [THERMAL GUARD] GPU at {gpu_temp}C (>= {max_temp}C limit); pausing 8s to let fans cool GPU...",
                        flush=True,
                    )
                    time.sleep(8.0)
                    gpu_temp = _query_gpu_temp_c()
                    if gpu_temp is None or gpu_temp <= max_temp - 5:
                        print(f"  [THERMAL GUARD] GPU cooled to {gpu_temp}C; resuming training.", flush=True)
                        break

        is_time_up = max_hours > 0.0 and (total_elapsed >= max_hours * 3600.0)
        is_eval_step = (step % cfg.train.eval_every == 0) or (step == cfg.train.max_steps) or is_time_up
        is_save_step = (step % save_every == 0) or is_eval_step

        if is_eval_step:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_m = evaluate_validation(
                model=model,
                val_loader=val_loader,
                cfg=cfg,
                device=device,
                pad_id=tokenizer.pad_id,
            )
            last_val_metrics = val_m
            curve = val_m["val_per_slot_acc"]
            B = len(curve)
            print(
                f"  [VAL step {step:6d} | {total_elapsed/3600:.2f}h] "
                f"val_loss {val_m['val_loss']:.4f} | "
                f"real_s0_ppl {val_m['val_slot0_ppl']:.2f} | real_s0_acc {val_m['val_slot0_acc']:.3f} | "
                f"real_lm_ppl {val_m['val_lm_ppl']:.2f} | real_lm_acc {val_m['val_lm_acc']:.3f} | "
                f"crystallization(63->32->8->0): "
                f"{curve[B-1]:.2f}->{curve[B//2]:.2f}->{curve[min(8, B-1)]:.2f}->{curve[0]:.2f}",
                flush=True,
            )
            if cfg.data.dataset == "tinystories":
                sample_p = (
                    "Once upon a time, Lily and a little bunny walked into the green forest "
                    "to find the golden key near the locked treasure chest. Suddenly a big lion"
                )
                preview = _sample_preview(model, tokenizer, device, sample_p, max_new_tokens=200)
                preview_dragon = _sample_preview(
                    model, tokenizer, device, sample_p, max_new_tokens=200, injected_word="dragon"
                )
                print(f"    [SAMPLE 200t] ...Suddenly a big lion -> {preview}", flush=True)
                print(f"    [DRAGON INJECT] ...Suddenly a big lion (+ 'dragon' in conveyer) -> {preview_dragon}", flush=True)
            else:
                preview = ""
                preview_dragon = ""

            if val_m["val_loss"] <= best_val_loss:
                best_val_loss = val_m["val_loss"]
                save_checkpoint(
                    out_dir / "best.pt",
                    model,
                    cfg,
                    step=step,
                    optimizer=None,  # Keep best.pt lightweight (~115 MB) for fast inference & fewer disk writes
                    metrics=val_m,
                    best_val_loss=best_val_loss,
                    elapsed_sec=total_elapsed,
                )

            log_entry = {
                "step": step,
                "elapsed_hours": round(total_elapsed / 3600.0, 4),
                "val_loss": round(val_m["val_loss"], 4),
                "val_slot0_ce": round(val_m["val_slot0_ce"], 4),
                "val_slot0_ppl": round(val_m["val_slot0_ppl"], 2),
                "val_slot0_acc": round(val_m["val_slot0_acc"], 4),
                "val_lm_ce": round(val_m["val_lm_ce"], 4),
                "val_lm_ppl": round(val_m["val_lm_ppl"], 2),
                "val_lm_acc": round(val_m["val_lm_acc"], 4),
                "val_belt_acc": round(val_m["val_belt_acc"], 4),
                "sample_lion": preview,
                "sample_dragon_conveyer": preview_dragon,
            }
            with open(log_jsonl, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry) + "\n")
                lf.flush()
                os.fsync(lf.fileno())

        if is_save_step:
            save_checkpoint(
                out_dir / "last.pt",
                model,
                cfg,
                step=step,
                optimizer=optimizer,
                metrics=last_val_metrics,
                best_val_loss=best_val_loss,
                elapsed_sec=total_elapsed,
            )
            if step % milestone_every == 0:
                save_checkpoint(
                    out_dir / f"ckpt_step_{step:06d}.pt",
                    model,
                    cfg,
                    step=step,
                    optimizer=optimizer,
                    metrics=last_val_metrics,
                    best_val_loss=best_val_loss,
                    elapsed_sec=total_elapsed,
                )
                _prune_old_milestones(out_dir, keep_last=8)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if is_time_up:
            print(
                f"[ConveyerTrain] Reached max_hours={max_hours:.2f}h budget at step {step}. "
                f"Checkpoints saved to {out_dir}.",
                flush=True,
            )
            break

    return last_val_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conveyor-Belt Diffusion LLM")
    parser.add_argument("--config", type=str, default="configs/synthetic.yaml", help="Path to YAML config")
    parser.add_argument("--init", type=str, default=None, help="Optional checkpoint (.pt) to warm-start weights from")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path (.pt) or 'auto' to resume full training state (model + optimizer + step) after a crash",
    )
    parser.add_argument("--max_hours", type=float, default=None, help="Wall-clock training budget in hours (e.g. 24)")
    parser.add_argument("--overfit", type=int, default=0, help="Overfit on N fixed batches for verification")
    parser.add_argument("--no_thought_highway", action="store_true", help="Disable continuous thought highway (alpha_k = 0)")
    parser.add_argument("--no_belt_bptt", action="store_true", help="Detach conveyor belt state between unrolled ticks (W=1)")
    parser.add_argument("--ar_baseline", action="store_true", help="Train 1-pass causal Autoregressive baseline")
    parser.add_argument("overrides", nargs="*", help="Config overrides in section.key=value format")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.max_hours is not None:
        cfg.train.max_hours = args.max_hours
    if args.overfit > 0:
        cfg.train.overfit_batches = args.overfit
    if args.no_thought_highway:
        cfg.train.no_thought_highway = True
        cfg.model.use_thought_highway = False
    if args.no_belt_bptt:
        cfg.train.no_belt_bptt = True
    if args.ar_baseline:
        cfg.train.ar_baseline = True
        cfg.model.causal_belt_only = True

    train_model(cfg, init_ckpt=args.init, resume_ckpt=args.resume)


if __name__ == "__main__":
    main()
