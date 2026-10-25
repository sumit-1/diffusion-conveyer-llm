from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Optional
import urllib.request
import torch

from conveyer.config import ConveyerConfig, _merge_dataclass
from conveyer.model import ConveyerDiffusionLM

HF_DEFAULT_CKPT_URL = "https://huggingface.co/sumitk/diffusion-conveyer-llm/resolve/main/best.pt"


def _move_to_cpu(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(device="cpu", non_blocking=False)
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(v) for v in obj)
    return obj


def _atomic_write_payload(path: Path, payload: dict[str, Any]) -> None:
    """
    Crash-safe atomic checkpoint writer on Windows NTFS:
      1. If `path` already exists, rotates `path -> path.prev` BEFORE writing so an abrupt BSOD/power loss
         never leaves both the current and previous checkpoint with un-flushed NTFS data blocks.
      2. Writes `payload` to `path.tmp`, calls `f.flush()` + `os.fsync(f.fileno())` to force NTFS data blocks
         to physical NVMe storage BEFORE updating the directory metadata via `os.replace`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_p = path.with_suffix(path.suffix + ".tmp")
    prev_p = path.with_suffix(path.suffix + ".prev")

    with open(tmp_p, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())

    if path.exists():
        try:
            os.replace(path, prev_p)
        except OSError:
            pass

    os.replace(tmp_p, path)


def save_checkpoint(
    path: str | Path,
    model: ConveyerDiffusionLM,
    cfg: ConveyerConfig,
    step: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metrics: Optional[dict[str, Any]] = None,
    best_val_loss: Optional[float] = None,
    elapsed_sec: float = 0.0,
) -> None:
    """
    Copies `model` and `optimizer` state dicts to CPU RAM first (avoiding 350 MB of synchronous
    WDDM GPU->CPU DMA transfers inside `torch.save` that can trigger `dxgmms2.sys` BSODs),
    then writes atomically with `os.fsync` and `.prev` backup rotation.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    cpu_model_state = _move_to_cpu(model.state_dict())
    payload: dict[str, Any] = {
        "model_state": cpu_model_state,
        "config": asdict(cfg),
        "step": step,
        "metrics": metrics or {},
        "best_val_loss": best_val_loss,
        "elapsed_sec": elapsed_sec,
    }
    if optimizer is not None:
        payload["optimizer_state"] = _move_to_cpu(optimizer.state_dict())

    _atomic_write_payload(Path(path), payload)


def load_checkpoint_payload(path: str | Path, map_location: Any = "cpu") -> tuple[dict[str, Any], Path]:
    """
    Loads checkpoint dict from `path`, automatically falling back to `path.prev` or the latest
    valid `ckpt_step_*.pt` in the same directory if `path` was corrupted by a power loss or BSOD.
    """
    p = Path(path)
    candidates: list[Path] = [p, p.with_suffix(p.suffix + ".prev")]
    if p.parent.exists():
        milestones = sorted(p.parent.glob("ckpt_step_*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        for m in milestones:
            if m not in candidates:
                candidates.append(m)
            m_prev = m.with_suffix(m.suffix + ".prev")
            if m_prev.exists() and m_prev not in candidates:
                candidates.append(m_prev)

    last_err: Optional[Exception] = None
    for cand in candidates:
        if not cand.exists() or cand.stat().st_size == 0:
            continue
        try:
            data = torch.load(cand, map_location=map_location, weights_only=False)
            if isinstance(data, dict) and "model_state" in data:
                return data, cand
        except Exception as e:
            last_err = e
            print(f"[Checkpoint] Warning: skipping unreadable checkpoint {cand.name} ({e})", flush=True)

    if p.name == "best.pt" and "tinystories" in str(p).replace("\\", "/"):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp_dl = p.with_suffix(".pt.dl")
            print(f"[Checkpoint] Downloading pretrained checkpoint from {HF_DEFAULT_CKPT_URL} ...", flush=True)
            req = urllib.request.Request(HF_DEFAULT_CKPT_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_dl, "wb") as f:
                while True:
                    chunk = resp.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp_dl, p)
            data = torch.load(p, map_location=map_location, weights_only=False)
            if isinstance(data, dict) and "model_state" in data:
                return data, p
        except Exception as e:
            last_err = e

    raise RuntimeError(f"No valid checkpoint found for {p} (last error: {last_err})")


def load_checkpoint(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> tuple[ConveyerDiffusionLM, ConveyerConfig, dict[str, Any]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, used_path = load_checkpoint_payload(path, map_location=device)
    if used_path != Path(path):
        print(f"[Checkpoint] Recovered from fallback checkpoint {used_path}", flush=True)
    cfg = ConveyerConfig()
    if "config" in ckpt:
        _merge_dataclass(cfg, ckpt["config"])
    model = ConveyerDiffusionLM(cfg.model).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg, ckpt
