#!/usr/bin/env python
"""Cross-method comparison: FedPepTAO vs FATE-LLM (LoRA) on Sent140.

Produces a single side-by-side figure (accuracy + communication cost)
suitable for the IEEE report.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.plotting import _save  # type: ignore  # noqa: E402


PAIRS = [
    ("FedPepTAO (prompt)", "results/metrics/fedpeptao_sent140.csv", "C0"),
    ("FATE-LLM (LoRA)",    "results/metrics/fatellm_lora_sent140.csv", "C3"),
]


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))

    for label, csv_path, color in PAIRS:
        path = ROOT / csv_path
        if not path.exists():
            print(f"WARNING: {csv_path} missing — skipping")
            continue
        df = pd.read_csv(path)
        axes[0].plot(df["round"], df["test_accuracy"], marker="o",
                     linewidth=1.5, color=color, label=label)
        axes[1].plot(df["round"], df["cumulative_mb"], marker="s",
                     linewidth=1.5, color=color, label=label)

    axes[0].set_xlabel("Communication round")
    axes[0].set_ylabel("Global test accuracy")
    axes[0].set_title("Sent140 — accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("Communication round")
    axes[1].set_ylabel("Cumulative communication (MB)")
    axes[1].set_title("Sent140 — communication cost")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    out = ROOT / "results" / "plots" / "sent140_peft_comparison"
    _save(fig, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
