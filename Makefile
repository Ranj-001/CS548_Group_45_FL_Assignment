# Convenience Makefile. Assumes you've created and activated a venv first:
#     python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

PY ?= python

.PHONY: help install test smoke demo fedpeptao lora ablation shakespeare all plots analysis clean

help:
	@echo "Targets:"
	@echo "  install    pip install -r requirements.txt"
	@echo "  test       run pytest"
	@echo "  smoke      run synthetic-data smoke test (~1 min)"
	@echo "  demo       alias for smoke"
	@echo "  fedpeptao  FedPepTAO on Sent140"
	@echo "  lora       FATE-LLM (LoRA) on Sent140"
	@echo "  shakespeare FedPepTAO on Shakespeare"
	@echo "  ablation   Dirichlet-α sweep on synthetic"
	@echo "  all        run every experiment + aggregate plots"
	@echo "  analysis   regenerate cross-run summary + comparison plots"
	@echo "  clean      drop ./.cache, .pytest_cache, __pycache__"

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest tests/test_smoke.py -q

smoke demo:
	$(PY) -m src.main --config configs/base.yaml --fast

fedpeptao:
	$(PY) -m src.main --config configs/fedpeptao_sent140.yaml

lora:
	$(PY) -m src.main --config configs/fatellm_lora_sent140.yaml

shakespeare:
	$(PY) -m src.main --config configs/fedpeptao_shakespeare.yaml

ablation:
	$(PY) -m src.main --config configs/ablation_dirichlet.yaml

all: fedpeptao lora shakespeare ablation analysis

analysis:
	$(PY) notebooks/analysis.py

plots: analysis

clean:
	rm -rf .cache .pytest_cache **/__pycache__ src/__pycache__ tests/__pycache__
