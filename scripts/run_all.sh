#!/usr/bin/env bash
# Reproduce every reported experiment.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.main --config configs/base.yaml
python -m src.main --config configs/fedpeptao_sent140.yaml
python -m src.main --config configs/fatellm_lora_sent140.yaml
python -m src.main --config configs/fedpeptao_shakespeare.yaml
python -m src.main --config configs/ablation_dirichlet.yaml

# Cross-method comparison plot
python scripts/plot_results.py compare \
  --label "FedPepTAO (prompt)" --csv results/metrics/fedpeptao_sent140.csv \
  --label "FATE-LLM (LoRA)"   --csv results/metrics/fatellm_lora_sent140.csv \
  --out results/plots/sent140_peft_comparison

echo "All experiments completed. Artifacts in ./results/"
