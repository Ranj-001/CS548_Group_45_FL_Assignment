"""Metric tracking + CSV logging + communication-cost accounting."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Bytes-per-element assumption (float32 == 4 bytes). Communication cost is
# computed as `param_count × 4` for both upload and download per sampled
# client, summed across all rounds. PEFT methods only transmit trainable
# params — this is the whole point.
BYTES_PER_PARAM = 4
BYTES_PER_MB = 1024 * 1024


@dataclass
class RoundMetric:
    round: int
    test_accuracy: float
    test_loss: float
    sampled_clients: int
    trainable_params: int
    upload_mb: float
    download_mb: float
    cumulative_mb: float
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClientMetric:
    round: int
    client_id: int
    train_loss: float
    train_accuracy: float
    num_examples: int


class MetricLogger:
    """Accumulates round + client metrics and persists them to CSV."""

    def __init__(self, run_name: str, output_dir: str | Path) -> None:
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.rounds: List[RoundMetric] = []
        self.clients: List[ClientMetric] = []
        self.convergence_round: Optional[int] = None
        self.cumulative_mb: float = 0.0

    # ------------------------------------------------------------------
    def log_round(
        self,
        *,
        round_idx: int,
        test_accuracy: float,
        test_loss: float,
        sampled_clients: int,
        trainable_params: int,
        threshold: float,
        extras: Optional[Dict[str, float]] = None,
    ) -> RoundMetric:
        per_client_mb = trainable_params * BYTES_PER_PARAM / BYTES_PER_MB
        # upload + download per client, summed over sampled clients
        round_mb = 2 * sampled_clients * per_client_mb
        self.cumulative_mb += round_mb

        if self.convergence_round is None and test_accuracy >= threshold:
            self.convergence_round = round_idx

        rec = RoundMetric(
            round=round_idx,
            test_accuracy=test_accuracy,
            test_loss=test_loss,
            sampled_clients=sampled_clients,
            trainable_params=trainable_params,
            upload_mb=sampled_clients * per_client_mb,
            download_mb=sampled_clients * per_client_mb,
            cumulative_mb=self.cumulative_mb,
            extras=extras or {},
        )
        self.rounds.append(rec)
        return rec

    def log_client(
        self,
        *,
        round_idx: int,
        client_id: int,
        train_loss: float,
        train_accuracy: float,
        num_examples: int,
    ) -> None:
        self.clients.append(
            ClientMetric(
                round=round_idx,
                client_id=client_id,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                num_examples=num_examples,
            )
        )

    # ------------------------------------------------------------------
    def write(self) -> dict:
        rounds_csv = self.metrics_dir / f"{self.run_name}.csv"
        with rounds_csv.open("w", newline="") as fh:
            extra_keys = sorted({k for r in self.rounds for k in r.extras.keys()})
            fieldnames = [
                "round", "test_accuracy", "test_loss", "sampled_clients",
                "trainable_params", "upload_mb", "download_mb", "cumulative_mb",
            ] + extra_keys
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.rounds:
                row = {
                    "round": r.round,
                    "test_accuracy": r.test_accuracy,
                    "test_loss": r.test_loss,
                    "sampled_clients": r.sampled_clients,
                    "trainable_params": r.trainable_params,
                    "upload_mb": r.upload_mb,
                    "download_mb": r.download_mb,
                    "cumulative_mb": r.cumulative_mb,
                }
                row.update({k: r.extras.get(k, "") for k in extra_keys})
                writer.writerow(row)

        clients_csv = self.metrics_dir / f"{self.run_name}_clients.csv"
        with clients_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["round", "client_id", "train_loss", "train_accuracy", "num_examples"],
            )
            writer.writeheader()
            for c in self.clients:
                writer.writerow({
                    "round": c.round, "client_id": c.client_id,
                    "train_loss": c.train_loss, "train_accuracy": c.train_accuracy,
                    "num_examples": c.num_examples,
                })

        summary = {
            "run_name": self.run_name,
            "rounds_csv": str(rounds_csv),
            "clients_csv": str(clients_csv),
            "convergence_round": self.convergence_round,
            "final_test_accuracy": self.rounds[-1].test_accuracy if self.rounds else None,
            "final_test_loss": self.rounds[-1].test_loss if self.rounds else None,
            "total_communication_mb": self.cumulative_mb,
        }
        log.info("Wrote metrics: %s", summary)
        return summary
