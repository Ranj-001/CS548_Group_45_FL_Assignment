"""Federated server / simulator.

Drives the FL training loop in a single process. Each round:

1. Sample ``fraction_fit × num_clients`` clients.
2. Broadcast the current global trainable parameters.
3. Each sampled client trains locally for ``local_epochs`` epochs.
4. The :class:`~strategies.Strategy` aggregates client updates.
5. Evaluate the new global model on the (centralised) global test set.
6. Persist the metrics (CSV + plots) and check the convergence threshold.

Mirrors Flower's :func:`flwr.simulation.start_simulation` flow but stays
single-process for simplicity and reproducibility.
"""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .client import FederatedClient, evaluate_model
from .config import Config
from .data import FederatedData
from .metrics import MetricLogger
from .models import ModelBundle, set_trainable_arrays, trainable_arrays
from .strategies import Strategy, make_strategy

log = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    summary: dict
    logger: MetricLogger


# ---------------------------------------------------------------------------
def _build_strategy(cfg: Config) -> Strategy:
    fw = cfg.framework
    if fw.strategy == "fedavg":
        return make_strategy("fedavg")
    return make_strategy(
        fw.strategy,
        server_lr=fw.server_lr,
        beta1=fw.server_beta1,
        beta2=fw.server_beta2,
        tau=fw.server_tau,
    )


def _build_clients(
    fed_data: FederatedData,
    global_bundle: ModelBundle,
    cfg: Config,
    device: torch.device,
) -> List[FederatedClient]:
    """Spawn one client per partition. Each client holds its own model deep-copy."""
    clients: List[FederatedClient] = []
    for cid in range(cfg.framework.num_clients):
        # Deep-copy the model once per client so they have separate state, but
        # share the (cheap) trainable-only parameter contract with the server.
        model_copy = copy.deepcopy(global_bundle.model)
        clients.append(
            FederatedClient(
                client_id=cid,
                model=model_copy,
                train_set=fed_data.client_train[cid],
                test_set=fed_data.client_test[cid],
                framework_cfg=cfg.framework,
                device=device,
            )
        )
    return clients


# ---------------------------------------------------------------------------
def run_simulation(
    cfg: Config,
    fed_data: FederatedData,
    global_bundle: ModelBundle,
) -> SimulationResult:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Running on device: %s", device)

    strategy = _build_strategy(cfg)
    clients = _build_clients(fed_data, global_bundle, cfg, device)

    global_model = global_bundle.model.to(device)
    global_params = trainable_arrays(global_model)
    trainable_params = sum(int(p.size) for p in global_params)

    logger = MetricLogger(run_name=cfg.run_name, output_dir=cfg.output_dir)
    threshold = cfg.framework.convergence_threshold

    eval_loader = DataLoader(fed_data.global_test, batch_size=128)

    rng = random.Random(cfg.seed)

    for r in range(1, cfg.framework.num_rounds + 1):
        # ----- 1. Sample clients --------------------------------------
        m = max(1, int(round(cfg.framework.fraction_fit * cfg.framework.num_clients)))
        sampled_ids = rng.sample(range(cfg.framework.num_clients), k=m)
        log.info("Round %d: sampling %d/%d clients %s",
                 r, m, cfg.framework.num_clients, sampled_ids)

        # ----- 2. Local training --------------------------------------
        updates: List[Tuple[List[np.ndarray], int]] = []
        for cid in sampled_ids:
            client = clients[cid]
            new_params, n_examples, metrics = client.fit(global_params)
            updates.append((new_params, n_examples))
            logger.log_client(
                round_idx=r, client_id=cid,
                train_loss=metrics["train_loss"],
                train_accuracy=metrics["train_accuracy"],
                num_examples=n_examples,
            )

        # ----- 3. Aggregate ------------------------------------------
        global_params = strategy.aggregate(global_params, updates)

        # ----- 4. Central evaluation ---------------------------------
        set_trainable_arrays(global_model, global_params)
        test_loss, test_acc, _ = evaluate_model(global_model, eval_loader, device)

        rec = logger.log_round(
            round_idx=r,
            test_accuracy=test_acc,
            test_loss=test_loss,
            sampled_clients=m,
            trainable_params=trainable_params,
            threshold=threshold,
            extras={
                "trainable_param_ratio": global_bundle.trainable_ratio,
            },
        )
        log.info(
            "Round %d | acc=%.4f loss=%.4f cum_MB=%.4f%s",
            r, test_acc, test_loss, rec.cumulative_mb,
            f" (converged @ round {logger.convergence_round})" if logger.convergence_round == r else "",
        )

    summary = logger.write()
    return SimulationResult(summary=summary, logger=logger)
