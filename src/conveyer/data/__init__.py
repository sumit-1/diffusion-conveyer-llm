from __future__ import annotations

from typing import Any, Protocol
import torch
from torch.utils.data import Dataset

from conveyer.config import ConveyerConfig


class TokenizerProtocol(Protocol):
    vocab_size: int
    pad_id: int
    bos_id: int
    eos_id: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def id_to_piece(self, token_id: int) -> str: ...


def build_dataset_and_tokenizer(
    cfg: ConveyerConfig,
    infer_only: bool = False,
) -> tuple[Dataset[dict[str, Any]], Dataset[dict[str, Any]], TokenizerProtocol]:
    if cfg.data.dataset == "synthetic":
        from conveyer.data.synthetic import build_synthetic_datasets

        return build_synthetic_datasets(cfg)
    elif cfg.data.dataset == "tinystories":
        from conveyer.data.tinystories import build_tinystories_datasets

        return build_tinystories_datasets(cfg, infer_only=infer_only)
    else:
        raise ValueError(f"Unknown dataset '{cfg.data.dataset}'. Choose 'synthetic' or 'tinystories'.")


def collate_conveyer_batch(batch: list[dict[str, Any]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    prompt_lengths = torch.zeros(bsz, dtype=torch.long)
    task_ids = torch.zeros(bsz, dtype=torch.long)

    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : seq.size(0)] = seq
        prompt_lengths[i] = int(item.get("prompt_len", 1))
        task_ids[i] = int(item.get("task_id", 0))

    return {
        "input_ids": input_ids,
        "prompt_lengths": prompt_lengths,
        "task_ids": task_ids,
    }
