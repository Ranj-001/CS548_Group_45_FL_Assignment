"""Matplotlib plotting helpers (≥300 DPI, 11pt fonts, PNG + PDF)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger(__name__)


_BASE_STYLE = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
plt.rcParams.update(_BASE_STYLE)


def _save(fig, base_path: Path) -> List[Path]:
    """Save a figure as both `<base>.png` and `<base>.pdf`.

    We append extensions as strings (instead of `Path.with_suffix`) so that
    stems containing dots — e.g. ``alpha0.01_accuracy`` — are preserved.
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png = Path(str(base_path) + ".png")
    pdf = Path(str(base_path) + ".pdf")
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    log.info("Wrote %s and %s", png, pdf)
    return [png, pdf]


def plot_accuracy_loss(
    csv_path: str | Path, output_dir: str | Path, run_name: str,
) -> List[Path]:
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["round"], df["test_accuracy"], marker="o", linewidth=1.5)
    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Global test accuracy")
    axes[0].set_title(f"{run_name} — accuracy")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["round"], df["test_loss"], marker="o", color="C3", linewidth=1.5)
    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Global test loss")
    axes[1].set_title(f"{run_name} — loss")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, Path(output_dir) / f"{run_name}_accuracy")


def plot_communication(
    csv_path: str | Path, output_dir: str | Path, run_name: str,
) -> List[Path]:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["round"], df["cumulative_mb"], marker="s", color="C2", linewidth=1.5)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Cumulative communication (MB)")
    ax.set_title(f"{run_name} — communication cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, Path(output_dir) / f"{run_name}_communication")


def plot_compare_runs(
    runs: Sequence[tuple[str, str | Path]],   # (label, csv_path)
    output_dir: str | Path,
    out_name: str,
    metric: str = "test_accuracy",
    ylabel: str = "Global test accuracy",
) -> List[Path]:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, csv_path in runs:
        df = pd.read_csv(csv_path)
        ax.plot(df["round"], df[metric], marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.set_title(out_name)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, Path(output_dir) / out_name)


def plot_client_distribution(
    clients_csv: str | Path, output_dir: str | Path, run_name: str,
) -> List[Path]:
    df = pd.read_csv(clients_csv)
    if df.empty:
        return []
    by_client = df.groupby("client_id")["num_examples"].max()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(by_client.index.astype(str), by_client.values, color="C0")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("# local training samples")
    ax.set_title(f"{run_name} — client data heterogeneity")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, Path(output_dir) / f"{run_name}_clients")
