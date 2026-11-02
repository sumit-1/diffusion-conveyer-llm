from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import json
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


@dataclass
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 688
    max_seq_len: int = 512
    dropout: float = 0.0
    rope_theta: float = 10000.0
    # Conveyor belt parameters
    belt_size: int = 96
    front_conveyer_slots: int = 5   # Last 5 slots (0..4) develop stopwords/grammar filling; Slots 5..95 are Content Dough Balls!
    duplex_k_slots: int = 4         # Number of content-only conveyer slots stacked per tick in Duplex-SimMT frame
    plan_horizon_stride: int = 2    # Back planning slots span 2x longer horizon while sentences expand in middle
    tau_min: float = 0.5
    tau_max: float = 1.0
    tau_gamma: float = 1.0
    gate_bias_init: float = 1.0
    spawn_noise_std: float = 0.5
    thought_noise_std: float = 0.25
    # Ablation & efficiency switches
    use_thought_highway: bool = True
    use_elastic_planning: bool = True  # Enables paused/elastic planning belt with mid-belt expansion
    use_thought_reservoir: bool = True # Content-only conveyer (strips stopwords/punctuation)
    use_duplex_frame: bool = True      # Stacks [prefill+generated text (prev_t)] + [content conveyer slots] via Duplex-SimMT frame
    causal_belt_only: bool = False     # If True, disables bidirectional belt attention (AR baseline mode)
    grad_checkpointing: bool = False   # Enables activation checkpointing for long unroll horizons (U=24..32)


@dataclass
class TrainConfig:
    out_dir: str = "runs/synthetic"
    seed: int = 42
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 1200
    max_hours: float = 0.0           # Optional wall-clock training budget in hours (0.0 = use max_steps only)
    warmup_steps: int = 80
    eval_every: int = 100
    save_every: int = 500            # Periodic crash-safe checkpoint interval (steps)
    log_every: int = 25
    unroll_steps: int = 16
    prefix_lm_weight: float = 1.0    # Weight on causal next-token LM loss across the prompt/sequence
    step_cooldown_ms: int = 0        # Optional GPU thermal pacing sleep per step in ms
    max_gpu_temp_c: int = 80         # Automatic thermal throttle threshold in Celsius
    # Corrupted-belt seeding & recovery parameters
    seed_sigma_max: float = 3.5
    seed_sigma_eta: float = 1.25
    recovery_prob: float = 0.35      # DAgger / self-conditioned prefix + belt recovery probability
    belt_corrupt_prob: float = 0.20  # Mid-unroll belt perturbation rate to train active belt recovery
    # Horizon loss weights: w_k = w_min + (1 - w_min) * (1 - k / (B - 1))^w_beta
    loss_w_min: float = 0.05
    loss_w_beta: float = 1.5
    entropy_reg: float = 0.001
    # Ablations
    no_belt_bptt: bool = False
    no_thought_highway: bool = False
    ar_baseline: bool = False
    overfit_batches: int = 0


@dataclass
class InferConfig:
    prompt_overlap: int = 12
    bake_steps: int = 64
    temperature: float = 0.0  # 0.0 = greedy argmax at slot 0
    max_new_tokens: int = 64


@dataclass
class DataConfig:
    dataset: str = "synthetic"  # "synthetic" or "tinystories"
    task: str = "all"           # "reverse", "path", "constraint", "all", "lm"
    seq_len: int = 192
    num_train: int = 24000
    num_val: int = 1024
    data_dir: str = "data/synthetic"
    vocab_size: int = 128


@dataclass
class ConveyerConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_scalar(val_str: str) -> Any:
    s = val_str.strip()
    if not s:
        return ""
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        return s


def _simple_yaml_parse(text: str) -> dict[str, Any]:
    """Minimal dependency-free YAML parser for 2-level config files."""
    if yaml is not None:
        return yaml.safe_load(text) or {}
    text_stripped = text.strip()
    if text_stripped.startswith("{"):
        return json.loads(text_stripped)
    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, val_part = stripped.split(":", 1)
        key = key.strip()
        val_part = val_part.strip()
        if indent == 0:
            if val_part == "":
                current_section = {}
                root[key] = current_section
            else:
                root[key] = _parse_scalar(val_part)
                current_section = None
        else:
            if current_section is None:
                raise ValueError(f"Unexpected indentation in YAML line: {raw_line}")
            current_section[key] = _parse_scalar(val_part)
    return root


def _merge_dataclass(dc_instance: Any, updates: dict[str, Any]) -> None:
    valid_fields = {f.name: f for f in fields(dc_instance)}
    for k, v in updates.items():
        if k not in valid_fields:
            continue
        current_val = getattr(dc_instance, k)
        if is_dataclass(current_val) and isinstance(v, dict):
            _merge_dataclass(current_val, v)
        else:
            setattr(dc_instance, k, v)


def _cast_value(raw: str, target_type: Any) -> Any:
    if target_type is bool or raw.lower() in ("true", "false"):
        return raw.lower() in ("true", "1", "yes")
    val = _parse_scalar(raw)
    if target_type is float and isinstance(val, int):
        return float(val)
    if target_type is int and isinstance(val, float):
        return int(val)
    if target_type is str:
        return str(val)
    return val


def apply_overrides(cfg: ConveyerConfig, overrides: list[str]) -> ConveyerConfig:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be in key=value format, got: {item}")
        key_path, raw_val = item.split("=", 1)
        parts = key_path.strip().split(".")
        obj: Any = cfg
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise KeyError(f"Unknown config section '{p}' in '{key_path}'")
            obj = getattr(obj, p)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise KeyError(f"Unknown config key '{leaf}' in '{key_path}'")
        cur = getattr(obj, leaf)
        val = _cast_value(raw_val.strip(), type(cur))
        setattr(obj, leaf, val)
    _sync_config(cfg)
    return cfg


def _sync_config(cfg: ConveyerConfig) -> None:
    cfg.model.vocab_size = cfg.data.vocab_size
    if cfg.train.no_thought_highway:
        cfg.model.use_thought_highway = False
    if cfg.train.ar_baseline:
        cfg.model.causal_belt_only = True


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> ConveyerConfig:
    cfg = ConveyerConfig()
    if path is not None:
        raw = _simple_yaml_parse(Path(path).read_text(encoding="utf-8"))
        _merge_dataclass(cfg, raw)
    if overrides:
        apply_overrides(cfg, overrides)
    _sync_config(cfg)
    return cfg
