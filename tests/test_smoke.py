"""Pipeline smoke test — verifies every key path in a few seconds."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Config, DataConfig, FrameworkConfig, ModelConfig  # noqa: E402
from src.data import dirichlet_partition, load_federated_data            # noqa: E402
from src.models import build_model, trainable_arrays, set_trainable_arrays  # noqa: E402
from src.strategies import make_strategy                                  # noqa: E402
from src.seed import set_seed                                             # noqa: E402
from src.server import run_simulation                                     # noqa: E402


@pytest.fixture(autouse=True)
def _seeded():
    set_seed(0)


def test_dirichlet_partition_balances():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
    parts = dirichlet_partition(y, num_clients=4, alpha=0.5, rng=rng)
    assert len(parts) == 4
    assert sum(len(p) for p in parts) == 200


def test_model_factories_run():
    for method in ["prompt_tuning", "lora", "full"]:
        cfg = ModelConfig(method=method, backbone="tiny_fallback",
                          hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                          prompt_length=4, lora_rank=4)
        bundle = build_model(cfg, num_classes=2, vocab_size=128)
        out = bundle.model(
            input_ids=torch.randint(0, 128, (2, 16)),
            attention_mask=torch.ones(2, 16, dtype=torch.long),
            labels=torch.tensor([0, 1]),
        )
        assert out["loss"].dim() == 0
        assert out["logits"].shape == (2, 2)


def test_trainable_array_roundtrip():
    cfg = ModelConfig(method="prompt_tuning", backbone="tiny_fallback",
                      hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                      prompt_length=4)
    bundle = build_model(cfg, num_classes=2, vocab_size=128)
    arrs = trainable_arrays(bundle.model)
    arrs[0] = np.zeros_like(arrs[0])
    set_trainable_arrays(bundle.model, arrs)
    arrs2 = trainable_arrays(bundle.model)
    assert np.allclose(arrs2[0], 0)


def test_strategies_aggregate_shapes():
    g = [np.zeros((3, 3), dtype=np.float32), np.zeros((5,), dtype=np.float32)]
    updates = [
        ([np.ones_like(g[0]), np.ones_like(g[1])], 10),
        ([2 * np.ones_like(g[0]), 2 * np.ones_like(g[1])], 30),
    ]
    for name in ["fedavg", "fedadam", "fedyogi", "fedadagrad"]:
        s = make_strategy(name, server_lr=1e-2)
        out = s.aggregate(g, updates)
        assert all(o.shape == gi.shape for o, gi in zip(out, g))


def test_end_to_end_synthetic(tmp_path):
    cfg = Config(
        run_name="pytest_run",
        seed=0,
        output_dir=str(tmp_path),
        framework=FrameworkConfig(
            num_rounds=2, num_clients=4, fraction_fit=0.5,
            local_epochs=1, local_batch_size=8, strategy="fedadam",
            convergence_threshold=2.0,  # never triggers
        ),
        data=DataConfig(
            dataset="synthetic", partition="dirichlet", dirichlet_alpha=0.5,
            num_classes=2, max_seq_len=16, num_samples_per_client=32,
            data_root=str(tmp_path / "data"),
        ),
        model=ModelConfig(
            method="prompt_tuning", backbone="tiny_fallback",
            hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
            prompt_length=4,
        ),
    )
    fed = load_federated_data(cfg.data, num_clients=cfg.framework.num_clients,
                              backbone=cfg.model.backbone, seed=cfg.seed)
    bundle = build_model(cfg.model, num_classes=fed.num_classes, vocab_size=fed.vocab_size)
    res = run_simulation(cfg, fed, bundle)
    assert len(res.logger.rounds) == 2
    assert res.summary["total_communication_mb"] > 0
