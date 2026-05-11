"""Command-line entry point: ``python -m src.main --config <yaml>``."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

from .config import Config, load_config
from .data import load_federated_data
from .models import build_model
from .plotting import (
    plot_accuracy_loss,
    plot_client_distribution,
    plot_communication,
    plot_compare_runs,
)
from .seed import set_seed
from .server import run_simulation


# ---------------------------------------------------------------------------
def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _run_one(cfg: Config) -> dict:
    set_seed(cfg.seed)
    fed_data = load_federated_data(
        cfg.data,
        num_clients=cfg.framework.num_clients,
        backbone=cfg.model.backbone,
        seed=cfg.seed,
    )
    bundle = build_model(
        cfg.model,
        num_classes=fed_data.num_classes,
        vocab_size=fed_data.vocab_size,
    )

    res = run_simulation(cfg, fed_data, bundle)

    rounds_csv = res.summary["rounds_csv"]
    clients_csv = res.summary["clients_csv"]
    plots_dir = Path(cfg.output_dir) / "plots"
    plot_accuracy_loss(rounds_csv, plots_dir, cfg.run_name)
    plot_communication(rounds_csv, plots_dir, cfg.run_name)
    plot_client_distribution(clients_csv, plots_dir, cfg.run_name)

    summary_path = Path(cfg.output_dir) / "metrics" / f"{cfg.run_name}_summary.json"
    summary_path.write_text(json.dumps(res.summary, indent=2))

    return res.summary


def _run_ablation(cfg: Config) -> dict:
    """Iterate over ``cfg.ablation.dirichlet_alphas`` (+ optional IID)."""
    assert cfg.ablation is not None
    base_name = cfg.run_name
    summaries = []
    runs_for_compare = []

    alphas = list(cfg.ablation.dirichlet_alphas)
    if cfg.ablation.include_iid:
        alphas = alphas + ["iid"]

    for a in alphas:
        sub = copy.deepcopy(cfg)
        sub.ablation = None
        if a == "iid":
            sub.data.partition = "iid"
            sub.run_name = f"{base_name}_iid"
        else:
            sub.data.partition = "dirichlet"
            sub.data.dirichlet_alpha = float(a)
            sub.run_name = f"{base_name}_alpha{a}"
        s = _run_one(sub)
        summaries.append(s)
        runs_for_compare.append((sub.run_name, s["rounds_csv"]))

    plot_compare_runs(
        runs_for_compare,
        Path(cfg.output_dir) / "plots",
        out_name=f"{base_name}_alpha_sweep",
    )
    plot_compare_runs(
        runs_for_compare,
        Path(cfg.output_dir) / "plots",
        out_name=f"{base_name}_alpha_sweep_comm",
        metric="cumulative_mb",
        ylabel="Cumulative communication (MB)",
    )
    return {"ablation": summaries}


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Federated LLM tuning (FedPepTAO / FATE-LLM).")
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--fast", action="store_true", help="Tiny smoke-test run.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, fast=args.fast)
    _setup_logging(args.log_level)
    logging.info("Run config: %s", json.dumps(cfg.to_dict(), indent=2))

    if cfg.ablation is not None:
        out = _run_ablation(cfg)
    else:
        out = _run_one(cfg)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
