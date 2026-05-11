"""YAML configuration loader with dataclass schemas."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FrameworkConfig:
    num_rounds: int = 20
    num_clients: int = 10
    fraction_fit: float = 0.5
    fraction_eval: float = 0.0
    min_available_clients: int = 2
    local_epochs: int = 5
    local_batch_size: int = 32
    optimizer: str = "sgd_momentum"
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    strategy: str = "fedadam"
    server_lr: float = 1.0e-2
    server_momentum: float = 0.9
    server_beta1: float = 0.9
    server_beta2: float = 0.99
    server_tau: float = 1.0e-3
    convergence_threshold: float = 0.80
    num_cpus_per_client: int = 1
    num_gpus_per_client: float = 0.0


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    partition: str = "dirichlet"
    dirichlet_alpha: float = 0.1
    num_classes: int = 2
    max_seq_len: int = 64
    num_samples_per_client: int = 256
    data_root: str = "./.cache/data"
    test_size: float = 0.2


@dataclass
class ModelConfig:
    method: str = "prompt_tuning"
    backbone: str = "tiny_fallback"
    hidden_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    vocab_size: int = 8192
    prompt_length: int = 20
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    log_every: int = 1


@dataclass
class AblationConfig:
    dirichlet_alphas: List[float] = field(default_factory=list)
    include_iid: bool = False


@dataclass
class Config:
    run_name: str = "run"
    seed: int = 42
    output_dir: str = "results"
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ablation: Optional[AblationConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.ablation is None:
            d.pop("ablation", None)
        return d


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path, fast: bool = False) -> Config:
    """Load a YAML config, layering it on top of `configs/base.yaml`.

    `fast=True` shrinks every dimension so a smoke test finishes in seconds.
    """
    path = Path(path)
    base_path = path.parent / "base.yaml"
    raw: Dict[str, Any] = {}
    if base_path.exists() and base_path != path:
        raw = _load_yaml(base_path)
    raw = _deep_update(raw, _load_yaml(path))

    if fast:
        raw = _deep_update(
            raw,
            {
                "run_name": raw.get("run_name", "run") + "_fast",
                "framework": {
                    "num_rounds": 3,
                    "num_clients": 4,
                    "fraction_fit": 0.5,
                    "local_epochs": 1,
                    "local_batch_size": 16,
                },
                "data": {
                    "dataset": "synthetic",
                    "num_samples_per_client": 64,
                    "max_seq_len": 16,
                },
                "model": {"backbone": "tiny_fallback", "prompt_length": 4},
            },
        )

    ablation_raw = raw.pop("ablation", None)
    cfg = Config(
        run_name=raw.get("run_name", "run"),
        seed=raw.get("seed", 42),
        output_dir=raw.get("output_dir", "results"),
        framework=FrameworkConfig(**raw.get("framework", {})),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        logging=LoggingConfig(**raw.get("logging", {})),
        ablation=AblationConfig(**ablation_raw) if ablation_raw else None,
    )
    return cfg
