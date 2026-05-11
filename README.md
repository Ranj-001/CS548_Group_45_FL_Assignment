# Federated Learning of Large Language Models

**CS548 Federated Learning Term Paper · Group 45 · IIT Patna · 2026.**

* Compiled IEEE paper: [`report/Group45_CS548_FL_Term_Paper.pdf`](report/Group45_CS548_FL_Term_Paper.pdf)
* YouTube walkthrough: *<paste link after upload>*
* GitHub: <https://github.com/Ranj-001/CS548_Group_45_FL_Assignment>

End-to-end implementation of **parameter-efficient federated fine-tuning of LLMs**,
based on two research papers:

1. **FedPepTAO** — *Federated Learning of Large Language Models with Parameter-Efficient
   Prompt Tuning and Adaptive Optimization* (Che et al., EMNLP 2023).
   [paper](https://openreview.net/forum?id=WuuxbObghx)
2. **FATE-LLM** — *An Industrial Grade Federated Learning Framework for Large Language
   Models* (Fan et al., 2023). [paper](https://arxiv.org/abs/2310.10049)

The codebase realises both ideas as Flower simulations:
* **Soft prompt tuning** with **server-side adaptive optimisation** (FedAdam/FedYogi/FedAdagrad).
* **LoRA fine-tuning** of a frozen pretrained transformer, FATE-LLM style.
* Dirichlet non-IID data partitioning across configurable client counts (10/50/100).
* Communication-cost accounting (only PEFT deltas leave each client).
* Full metric logging (accuracy/loss curves, convergence round, per-client stats).

---

## 1. Repository layout

```
FL_Assignment/
├── README.md
├── requirements.txt
├── configs/                 # YAML experiment configs
│   ├── base.yaml
│   ├── fedpeptao_sent140.yaml
│   ├── fedpeptao_shakespeare.yaml
│   ├── fatellm_lora_sent140.yaml
│   └── ablation_dirichlet.yaml
├── src/
│   ├── __init__.py
│   ├── main.py              # CLI entry point — runs a Flower simulation
│   ├── client.py            # Flower NumPyClient (PEFT only)
│   ├── server.py            # Strategy factory + evaluation hooks
│   ├── strategies.py        # FedAvg / FedAdam / FedYogi / FedAdagrad
│   ├── prompt_tuning.py     # Soft-prompt wrapper (FedPepTAO)
│   ├── lora.py              # LoRA adapter layers (FATE-LLM)
│   ├── models.py            # Model factories (HF + tiny fallback)
│   ├── data.py              # Loaders + Dirichlet / IID partitioner
│   ├── metrics.py           # CSV writer, communication-cost accounting
│   ├── plotting.py          # Matplotlib plot helpers
│   ├── seed.py              # Deterministic seeding
│   └── config.py            # YAML loader & dataclasses
├── scripts/
│   ├── run_all.sh           # Reproduce every reported experiment
│   ├── run_demo.py          # Tiny synthetic-data smoke test
│   └── plot_results.py
├── results/
│   ├── metrics/             # *.csv per run
│   └── plots/               # *.png/pdf
├── tests/
│   └── test_smoke.py
└── report/
    └── report.md            # IEEE-format skeleton
```

---

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code is pinned to **Python 3.10**, **PyTorch >= 2.0**, **Flower >= 1.7**, and
**transformers >= 4.36**. Everything runs CPU-only by default; GPU is auto-detected.

---

## 3. Quick start

The fastest way to verify the pipeline (no internet, no HF download, ~1 minute on
CPU) — uses a tiny transformer and synthetic text data:

```bash
python -m src.main --config configs/base.yaml --fast
```

A full FedPepTAO run on Sent140 with 10 clients, Dirichlet(α=0.1):

```bash
python -m src.main --config configs/fedpeptao_sent140.yaml
```

LoRA federated fine-tuning (FATE-LLM style):

```bash
python -m src.main --config configs/fatellm_lora_sent140.yaml
```

Every run writes:
* `results/metrics/<run_name>.csv`            — per-round global metrics
* `results/metrics/<run_name>_clients.csv`    — per-client local stats
* `results/plots/<run_name>_accuracy.png`     — accuracy/loss curves

Reproduce every experiment from the report:

```bash
bash scripts/run_all.sh
```

---

## 4. Configuration reference

Every YAML file inherits the keys in `configs/base.yaml`. Selected fields:

| Field                          | Type    | Default                          | Notes |
| ------------------------------ | ------- | -------------------------------- | ----- |
| `seed`                         | int     | 42                               | Applied to `random`, `numpy`, `torch`. |
| `framework.num_rounds`         | int     | 20                               | Federated rounds. |
| `framework.num_clients`        | int     | 10                               | Total simulated clients. |
| `framework.fraction_fit`       | float   | 0.5                              | Client sampling fraction. |
| `framework.local_epochs`       | int     | 5                                |  |
| `framework.local_batch_size`   | int     | 32                               |  |
| `framework.optimizer`          | str     | sgd_momentum                     | `sgd_momentum` \| `adamw`. |
| `framework.learning_rate`      | float   | 0.01                             |  |
| `framework.strategy`           | str     | fedadam                          | `fedavg` \| `fedadam` \| `fedyogi` \| `fedadagrad`. |
| `framework.server_lr`          | float   | 1e-2                             | Server-side adaptive LR. |
| `data.dataset`                 | str     | sent140                          | `sent140` \| `shakespeare` \| `agnews` \| `synthetic`. |
| `data.partition`               | str     | dirichlet                        | `iid` \| `dirichlet`. |
| `data.dirichlet_alpha`         | float   | 0.1                              |  |
| `model.method`                 | str     | prompt_tuning                    | `prompt_tuning` \| `lora` \| `full`. |
| `model.backbone`               | str     | prajjwal1/bert-tiny              | Any HF checkpoint or `tiny_fallback`. |
| `model.prompt_length`          | int     | 20                               | Soft-prompt tokens. |
| `model.lora_rank`              | int     | 8                                |  |
| `model.lora_alpha`             | int     | 16                               |  |

---

## 5. Metrics

For every run we log, per round:

* Global test **accuracy** and **loss**.
* **Convergence round** — first round where accuracy ≥ 0.80 (configurable).
* **Communication cost (MB)** — *upload + download* of trainable params per
  sampled client, summed over all rounds. Frozen backbones are never transmitted.
* Per-client local training loss / accuracy / sample count.

Additional metrics for the LLM-adaptation category:

* **Trainable parameter ratio** (PEFT params / total backbone params).
* **PEFT-throughput speedup** vs. full fine-tuning (estimated).

All metrics are persisted as CSV and re-loaded by `scripts/plot_results.py`.

---

## 6. Reproducibility

Every run sets `random.seed`, `numpy.random.seed`, and `torch.manual_seed` to
`seed` (default `42`). The Flower simulation is itself deterministic given a
fixed seed and a single worker. We pin `transformers`, `torch`, and `flwr` in
`requirements.txt` to keep numeric drift bounded.

---

## 7. Deliverables checklist

- [x] Code repository with prescribed structure.
- [x] YAML configs for every reported experiment.
- [x] `results/metrics/*.csv` for every run.
- [x] Plots at ≥300 DPI (PDF + PNG) in `results/plots/`.
- [x] IEEE-format report skeleton in `report/`.
- [x] `requirements.txt` with pinned versions.
- [x] `scripts/run_all.sh` to re-run every experiment.
- [x] Smoke test (`pytest tests/test_smoke.py`).

---

## 8. References

* Che, T., Liu, J., Zhou, Y. et al. *Federated Learning of Large Language Models
  with Parameter-Efficient Prompt Tuning and Adaptive Optimization.* EMNLP 2023.
* Fan, T., Kang, Y., Ma, G. et al. *FATE-LLM: An Industrial Grade Federated
  Learning Framework for Large Language Models.* arXiv:2310.10049, 2023.
* Beutel, D.J. et al. *Flower: A Friendly Federated Learning Research Framework.*
  arXiv:2007.14390.
* Hu, E.J. et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
* Lester, B., Al-Rfou, R., Constant, N. *The Power of Scale for Parameter-Efficient
  Prompt Tuning.* EMNLP 2021.
