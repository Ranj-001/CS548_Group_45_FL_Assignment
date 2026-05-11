"""Interactive analysis notebook (run as a script).

Loads every CSV in ``results/metrics/`` and produces:
  * A summary table to stdout.
  * A heads-up cross-method comparison plot.

Usage:
    python notebooks/analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.plotting import plot_compare_runs  # noqa: E402


def main() -> int:
    metrics_dir = ROOT / "results" / "metrics"
    plots_dir = ROOT / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    runs_for_compare = []
    skip_stems = {"all_runs_summary"}
    for csv_path in sorted(metrics_dir.glob("*.csv")):
        if csv_path.stem.endswith("_clients") or csv_path.stem in skip_stems:
            continue
        df = pd.read_csv(csv_path)
        if df.empty or "round" not in df.columns:
            continue
        summary_rows.append({
            "run": csv_path.stem,
            "rounds": int(df["round"].max()),
            "final_acc": df["test_accuracy"].iloc[-1],
            "best_acc": df["test_accuracy"].max(),
            "best_round": int(df.loc[df["test_accuracy"].idxmax(), "round"]),
            "final_loss": df["test_loss"].iloc[-1],
            "total_comm_mb": df["cumulative_mb"].iloc[-1],
            "trainable_params": int(df["trainable_params"].iloc[-1]),
        })
        runs_for_compare.append((csv_path.stem, csv_path))

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(metrics_dir / "all_runs_summary.csv", index=False)
    print(f"\nWrote {metrics_dir / 'all_runs_summary.csv'}")

    if len(runs_for_compare) >= 2:
        plot_compare_runs(runs_for_compare, plots_dir, "all_runs_accuracy")
        plot_compare_runs(
            runs_for_compare, plots_dir, "all_runs_communication",
            metric="cumulative_mb", ylabel="Cumulative communication (MB)",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
