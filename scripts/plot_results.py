#!/usr/bin/env python
"""Standalone plot helpers — invoked from `scripts/run_all.sh` for cross-run
comparison plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script even though we live under ./scripts
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.plotting import plot_compare_runs  # noqa: E402


def _compare(args: argparse.Namespace) -> int:
    if len(args.label) != len(args.csv):
        raise SystemExit("--label and --csv must occur the same number of times")
    runs = list(zip(args.label, args.csv))
    out_dir = Path(args.out).parent
    out_name = Path(args.out).name
    plot_compare_runs(runs, out_dir, out_name=out_name, metric=args.metric, ylabel=args.ylabel)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--label", action="append", required=True)
    cmp_.add_argument("--csv", action="append", required=True)
    cmp_.add_argument("--out", required=True)
    cmp_.add_argument("--metric", default="test_accuracy")
    cmp_.add_argument("--ylabel", default="Global test accuracy")
    args = ap.parse_args()
    return {"compare": _compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
