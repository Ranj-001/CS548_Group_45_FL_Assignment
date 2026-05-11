"""Flower-style federated client.

Each client owns:
  * a local PyTorch model whose **trainable** parameters are the only thing
    exchanged with the server (PEFT contract);
  * a local train and test ``TextClassificationDataset``;
  * a local optimiser rebuilt every round (stateless across rounds, as in the
    FedPepTAO baseline).

The class follows Flower's :class:`flwr.client.NumPyClient` shape — ``get_parameters``
/ ``set_parameters`` / ``fit`` / ``evaluate`` — so it can be plugged into a real
Flower simulation if desired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import FrameworkConfig
from .models import set_trainable_arrays, trainable_arrays

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def _build_optimizer(model: nn.Module, cfg: FrameworkConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd_momentum":
        return torch.optim.SGD(
            params, lr=cfg.learning_rate, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


@dataclass
class TrainStats:
    loss: float
    accuracy: float
    num_examples: int


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> TrainStats:
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        out = model(input_ids, attn_mask, labels)
        loss = out["loss"]
        loss.backward()
        # Gradient clipping keeps prompt/LoRA updates well-behaved.
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
        )
        optimizer.step()
        preds = out["logits"].argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)
    return TrainStats(
        loss=total_loss / max(total, 1),
        accuracy=total_correct / max(total, 1),
        num_examples=total,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, int]:
    """Returns ``(avg_loss, accuracy, num_examples)``."""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids, attn_mask, labels)
        total_loss += out["loss"].item() * labels.size(0)
        preds = out["logits"].argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        return 0.0, 0.0, 0
    return total_loss / total, total_correct / total, total


# ---------------------------------------------------------------------------
class FederatedClient:
    """In-process Flower-NumPyClient-style federated client."""

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_set: Dataset,
        test_set: Dataset,
        framework_cfg: FrameworkConfig,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.cfg = framework_cfg
        self.device = device

    # ------------------------------------------------------------------
    # NumPyClient API
    # ------------------------------------------------------------------
    def get_parameters(self) -> List[np.ndarray]:
        return trainable_arrays(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_trainable_arrays(self.model, parameters)

    def fit(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.to(self.device)
        loader = DataLoader(
            self.train_set, batch_size=self.cfg.local_batch_size, shuffle=True,
        )
        optimizer = _build_optimizer(self.model, self.cfg)
        last: TrainStats = TrainStats(loss=0.0, accuracy=0.0, num_examples=len(self.train_set))
        for _ in range(self.cfg.local_epochs):
            last = _train_one_epoch(self.model, loader, optimizer, self.device)
        return (
            self.get_parameters(),
            last.num_examples,
            {"train_loss": last.loss, "train_accuracy": last.accuracy},
        )

    def evaluate(self, parameters: List[np.ndarray]) -> Tuple[float, int, Dict[str, float]]:
        self.set_parameters(parameters)
        self.model.to(self.device)
        loader = DataLoader(self.test_set, batch_size=self.cfg.local_batch_size)
        loss, acc, n = evaluate_model(self.model, loader, self.device)
        return loss, n, {"accuracy": acc}
