from __future__ import annotations

import argparse
import sys
import time
import torch

from conveyer.checkpoint import load_checkpoint
from conveyer.config import load_config
from conveyer.data import build_dataset_and_tokenizer
from conveyer.infer import TickSnapshot, stream_conveyer_generation
from conveyer.model import ConveyerDiffusionLM

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
GRAY = "\033[90m"


def _format_slot_badge(slot_idx: int, piece: str, prob: float) -> str:
    clean = piece.replace("\n", " ")[:9]
    if slot_idx == 0:
        return f"{BOLD}{GREEN}[0:{clean} {prob*100:2.0f}%]{RESET}"
    elif slot_idx < 15:
        return f"{GREEN}{clean}({prob*100:2.0f}%){RESET}"
    elif slot_idx < 40:
        return f"{YELLOW}{clean}({prob*100:2.0f}%){RESET}"
    else:
        return f"{GRAY}{clean}{RESET}"


def render_frame(
    prompt_str: str,
    committed_pieces: list[str],
    snap: TickSnapshot,
    tokenizer: Any,
    clear_screen: bool = True,
) -> str:
    B = len(snap.slot_top1_tokens)
    lines: list[str] = []
    if clear_screen:
        lines.append("\033[2J\033[H")
    lines.append("=" * 98)
    mode_tag = (
        f"{CYAN}PRE-BAKING WARMUP (Clamping Input -> Preparing Conveyor Belt){RESET}"
        if snap.is_warmup
        else f"{GREEN}LIVE CONVEYOR BAKING (Serving Slot 0 -> Shifting 1..63 -> Spawning 63){RESET}"
    )
    lines.append(f"{BOLD}64-SLOT DUAL-CHANNEL CONVEYOR OVEN LLM{RESET} | Tick t={snap.tick:3d} | {mode_tag}")
    lines.append("-" * 98)
    lines.append(f"{BOLD}Input Prompt :{RESET} {prompt_str}")
    committed_str = " ".join(committed_pieces) if committed_pieces else "(warming up belt...)"
    lines.append(f"{BOLD}Served Output:{RESET} {BOLD}{GREEN}{committed_str}{RESET}")
    lines.append("-" * 98)

    # Zone 1: Front of Belt (0 .. 14) - Final Grammatical Polish
    front_items = [
        _format_slot_badge(k, tokenizer.id_to_piece(snap.slot_top1_tokens[k]), snap.slot_top1_probs[k])
        for k in range(min(15, B))
    ]
    lines.append(f"{BOLD}{GREEN}FRONT [00..14] (Final Polish, alpha->0):{RESET}")
    lines.append("  <- POP " + " | ".join(front_items[:8]))
    if len(front_items) > 8:
        lines.append("         " + " | ".join(front_items[8:15]))

    # Zone 2: Middle of Belt (15 .. 39) - Firming Up Phrasing
    if B > 15:
        mid_indices = list(range(15, min(40, B), 2))
        mid_items = [
            _format_slot_badge(k, tokenizer.id_to_piece(snap.slot_top1_tokens[k]), snap.slot_top1_probs[k])
            for k in mid_indices[:10]
        ]
        lines.append(f"{BOLD}{YELLOW}MIDDLE [15..39] (Firming Phrasing, Dual-Channel):{RESET}")
        lines.append("         " + " | ".join(mid_items))

    # Zone 3: Back of Belt (40 .. 63) - Rough Ideas & Fresh Raw Dough
    if B > 40:
        back_indices = list(range(40, B, 2))
        if (B - 1) not in back_indices:
            back_indices.append(B - 1)
        back_items = [
            f"[{k}:{_format_slot_badge(k, tokenizer.id_to_piece(snap.slot_top1_tokens[k]), snap.slot_top1_probs[k])}]"
            for k in back_indices[-8:]
        ]
        lines.append(f"{BOLD}{MAGENTA}BACK   [40..63] (Rough Ideas <- Slot 63 Raw Dough Spawn):{RESET}")
        lines.append("         " + " ".join(back_items) + " <- [NEW DOUGH]")

    # Telemetry bar: Entropy & Gate Retention
    ent_front = sum(snap.slot_entropies[: min(8, B)]) / max(1, min(8, B))
    ent_mid = sum(snap.slot_entropies[min(15, B - 1) : min(40, B)]) / max(1, min(40, B) - min(15, B - 1))
    ent_back = sum(snap.slot_entropies[max(0, B - 16) : B]) / max(1, B - max(0, B - 16))
    gate_avg = sum(snap.slot_gate_means) / max(1, B)
    lines.append("-" * 98)
    lines.append(
        f"Entropy (Front/Mid/Back): {ent_front:.2f} / {ent_mid:.2f} / {ent_back:.2f} nats | "
        f"Mean Thought Gate g_k: {gate_avg:.3f}"
    )
    lines.append("=" * 98)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Terminal Visualizer for the 64-Slot Conveyor-Belt Diffusion LLM")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (.pt). If omitted, initializes from config")
    parser.add_argument("--config", type=str, default="configs/synthetic.yaml", help="Config path if --ckpt is not provided")
    parser.add_argument("--prompt", type=str, default=None, help="Custom input prompt")
    parser.add_argument("--tokens", type=int, default=24, help="Number of output tokens to bake")
    parser.add_argument("--overlap", type=int, default=12, help="Prompt-overlap pre-baking window (n - W)")
    parser.add_argument("--bake-steps", type=int, default=64, help="Conveyor belt size (1..64)")
    parser.add_argument("--hz", type=float, default=10.0, help="Refresh rate in ticks/sec (use 0 for no delay)")
    parser.add_argument("--no-clear", action="store_true", help="Print frames sequentially without ANSI screen clear")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.ckpt is not None:
        model, cfg, _ = load_checkpoint(args.ckpt, device=device)
    else:
        cfg = load_config(args.config)
        model = ConveyerDiffusionLM(cfg.model).to(device)

    _, val_ds, tokenizer = build_dataset_and_tokenizer(cfg)

    if args.prompt is not None:
        prompt_ids = tokenizer.encode(args.prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    else:
        sample = val_ds[0]
        p_len = int(sample["prompt_len"])
        prompt_tensor = sample["input_ids"][:p_len].unsqueeze(0).to(device)

    prompt_str = tokenizer.decode(prompt_tensor[0].tolist())
    committed: list[str] = []
    delay = (1.0 / args.hz) if args.hz > 0 else 0.0

    for tok, snap in stream_conveyer_generation(
        model=model,
        prompt_tokens=prompt_tensor,
        max_new_tokens=args.tokens,
        prompt_overlap=args.overlap,
        bake_steps=args.bake_steps,
        temperature=0.0,
        eos_token_id=tokenizer.eos_id,
    ):
        if not snap.is_warmup:
            committed.append(tokenizer.id_to_piece(int(tok[0].item())))
        frame = render_frame(
            prompt_str=prompt_str,
            committed_pieces=committed,
            snap=snap,
            tokenizer=tokenizer,
            clear_screen=(not args.no_clear),
        )
        sys.stdout.write(frame + "\n")
        sys.stdout.flush()
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    main()
