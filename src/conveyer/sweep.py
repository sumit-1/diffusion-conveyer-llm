from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import torch

from conveyer.checkpoint import load_checkpoint
from conveyer.config import load_config
from conveyer.data import build_dataset_and_tokenizer
from conveyer.eval import evaluate_closed_loop_generation
from conveyer.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head-to-Head Benchmark: 1-Pass AR Baseline vs. 64-Pass Conveyor Diffusion LLM & Ablations"
    )
    parser.add_argument("--config", type=str, default="configs/synthetic.yaml", help="Base config file")
    parser.add_argument("--steps", type=int, default=400, help="Training steps per variant")
    parser.add_argument("--episodes", type=int, default=64, help="Closed-loop evaluation episodes")
    parser.add_argument("--out-dir", type=str, default="runs/sweep", help="Output directory for sweep")
    args = parser.parse_args()

    variants = [
        {
            "name": "1_ar_baseline",
            "label": "1-Pass Causal AR Baseline (B=1)",
            "overrides": ["train.ar_baseline=true", "train.unroll_steps=8"],
            "bake_steps": 1,
            "overlap": 0,
        },
        {
            "name": "2_conveyer_soft_only",
            "label": "64-Slot Conveyor (Soft-Vocab Only, alpha_k=0)",
            "overrides": ["train.no_thought_highway=true"],
            "bake_steps": 64,
            "overlap": 16,
        },
        {
            "name": "3_conveyer_nobptt",
            "label": "64-Slot Conveyor (Detached W=1, No Belt BPTT)",
            "overrides": ["train.no_belt_bptt=true"],
            "bake_steps": 64,
            "overlap": 16,
        },
        {
            "name": "4_conveyer_dual_full",
            "label": "64-Slot Dual-Channel Conveyor (Soft-Vocab + Gated Thought + BPTT)",
            "overrides": [],
            "bake_steps": 64,
            "overlap": 16,
        },
    ]

    summary: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for var in variants:
        run_dir = Path(args.out_dir) / var["name"]
        overrides = [
            f"train.max_steps={args.steps}",
            f"train.eval_every={max(50, args.steps // 2)}",
            f"train.out_dir={run_dir.as_posix()}",
        ] + var["overrides"]

        print("\n" + "=" * 90)
        print(f"Training Variant: {var['label']}")
        print("=" * 90)
        cfg = load_config(args.config, overrides)
        val_m = train_model(cfg)

        ckpt_path = run_dir / "best.pt"
        model, loaded_cfg, _ = load_checkpoint(ckpt_path, device=device)
        _, val_ds, tokenizer = build_dataset_and_tokenizer(loaded_cfg)

        cl_metrics = evaluate_closed_loop_generation(
            model=model,
            val_ds=val_ds,
            tokenizer=tokenizer,
            device=device,
            num_episodes=args.episodes,
            prompt_overlap=var["overlap"],
            bake_steps=var["bake_steps"],
        )
        row = {
            "name": var["name"],
            "label": var["label"],
            "val_slot0_acc": val_m.get("val_slot0_acc", 0.0),
            "val_slot0_ppl": val_m.get("val_slot0_ppl", 0.0),
            **cl_metrics,
        }
        summary.append(row)

    print("\n" + "=" * 100)
    print(f"{'Model / Ablation Variant':<56} | {'Pos-0 Acc':<10} | {'Token Acc':<10} | {'Exact Seq':<10}")
    print("-" * 100)
    for r in summary:
        print(
            f"{r['label']:<56} | {r['first_token_acc']*100:8.2f}% | "
            f"{r['token_acc']*100:8.2f}% | {r['exact_seq_acc']*100:8.2f}%"
        )
    print("=" * 100)

    out_path = Path(args.out_dir) / "sweep_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
