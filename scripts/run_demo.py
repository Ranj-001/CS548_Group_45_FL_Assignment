#!/usr/bin/env python
"""Tiny smoke test — runs the synthetic-data pipeline in seconds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["--config", "configs/base.yaml", "--fast"]))
