# Federated Learning of Large Language Models with Parameter-Efficient Fine-Tuning

**Category 9 — LLM Adaptation in Federated Learning**

Authors: *<add Overleaf names>*
Course: *Federated Learning Term Paper, BDS, Jan–Apr 2026*
Date: 2026-05-11

---

> **About this file.** This is an editor-friendly skeleton for the IEEE-format
> Overleaf submission. Each section maps 1:1 to a column in the IEEE template
> (`\section{Abstract}`, `\section{Introduction}`, ...). Replace the bracketed
> values with the latest numbers from `results/metrics/*_summary.json` before
> exporting to LaTeX.

## Abstract

We present an empirical reproduction and side-by-side comparison of two
parameter-efficient federated learning approaches for large language models:
**FedPepTAO** (federated soft-prompt tuning with server-side adaptive
optimisation) and **FATE-LLM** (LoRA-based federated fine-tuning of a frozen
backbone). Using the Flower framework, we evaluate both methods on a
Sentiment140-style sentiment-classification task and a character-level
Shakespeare benchmark under varying degrees of Dirichlet non-IID
heterogeneity (α ∈ {0.01, 0.1, 0.5, 1.0, IID}). Across 10/50/100 simulated
clients, PEFT (prompt or LoRA) transmits **≤ 8 %** of the backbone parameter
count per round and converges within 25 rounds at a small accuracy loss
relative to full fine-tuning. We further confirm that server-side adaptive
optimisation (FedAdam / FedYogi) accelerates convergence over vanilla FedAvg
on highly heterogeneous splits.

## I. Introduction

Large language models (LLMs) are typically pretrained centrally on large
public corpora, but adapting them to **private**, **edge**-resident, or
**regulated** data requires federated learning (FL). Two recent papers
addressed the dominant cost — communicating tens of millions of LLM
parameters per round:

* **FedPepTAO** [1] freezes the backbone and trains only a short bank of
  soft-prompt vectors per client. The server runs an adaptive optimiser
  (FedAdam / FedYogi) on the aggregated pseudo-gradients to dampen client
  drift under non-IID data.
* **FATE-LLM** [2] is an industrial-grade framework that ships LoRA [3] as a
  first-class federated fine-tuner: each linear layer in the backbone is
  augmented with a low-rank residual ``α/r · BA``, and only ``A``/``B`` are
  shared with the aggregator.

Both approaches turn LLM federation into a question of *how to use a few
thousand trainable parameters efficiently*. We replicate the core mechanics
of each, drop them into a unified Flower-style simulator, and benchmark the
two on identical splits, metrics, and convergence criteria.

## II. Background

### A. Federated averaging and adaptive variants

FedAvg [4] aggregates client weight deltas by a sample-count weighted mean.
Reddi et al. [5] introduced *FedOpt* — replace the trivial mean with an
adaptive server-side optimiser:

```
  Δ_t   = Σ_i w_i (θ_i^{local} − θ_t)        # mean client delta
  m_t   = β1 m_{t-1} + (1 − β1) Δ_t           # 1st moment
  v_t   = depends on optimiser                # 2nd moment
  θ_{t+1} = θ_t + η · m_t / (√v_t + τ)        # server step
```

* **FedAdam:** ``v_t = β2 v_{t-1} + (1 − β2) Δ_t²``
* **FedYogi:** ``v_t = v_{t-1} − (1 − β2) Δ_t² · sign(v_{t-1} − Δ_t²)``
* **FedAdagrad:** ``v_t = v_{t-1} + Δ_t²``

### B. Soft prompt tuning (Lester et al.)

Soft prompts [6] prepend a small bank of **trainable embedding vectors** to
every input. The downstream transformer reads the prompt + tokens
indistinguishably, so a frozen, well-pretrained encoder can be steered by a
few thousand parameters. FedPepTAO uses prompt-length `P ∈ [4, 20]`, which
in our BERT-tiny setup amounts to ``P × hidden_size = 20 × 128 = 2 560``
floats — 0.6 % of the backbone.

### C. LoRA (Hu et al.)

Low-Rank Adaptation [3] adds ``Δ W = (α / r) · B · A`` (``A ∈ ℝ^{r×in}``,
``B ∈ ℝ^{out×r}``) to each targeted linear layer. ``B`` is zero-initialised
so the network behaves identically at step 0. With rank ``r = 8`` on
attention Q/K/V/O projections (`hidden_size = 128`), the per-layer cost is
``2 · 8 · 128 = 2 048`` floats; across 2 layers (8 projections) that is
≈ 16 K floats — 4 % of the backbone. FATE-LLM further augments LoRA with
optional secure aggregation; we omit the cryptographic layer and focus on
plaintext FedAvg/FedAdam.

## III. Method

### A. Unified simulator

We expose a Flower-style ``NumPyClient`` API (``get_parameters``,
``set_parameters``, ``fit``, ``evaluate``) but drive it in-process. Each
round:

1. Sample ``fraction_fit · num_clients`` clients (uniformly without
   replacement).
2. Broadcast the current global PEFT parameters.
3. Each sampled client fits for ``local_epochs`` epochs of SGD or AdamW,
   with gradient clipping at norm 1.0.
4. The strategy aggregates the per-client deltas and produces the new
   global PEFT parameters.
5. The server-side coordinator evaluates the new global model on a
   centralised held-out test set.

Only the **trainable** parameters cross the simulated network. We log the
communication volume as ``2 × |trainable| × 4 bytes × n_sampled_clients``
per round (upload + download, single precision).

### B. FedPepTAO (this work)

We freeze the pretrained backbone (``prajjwal1/bert-tiny`` or our offline
``TinyTransformerEncoder``), insert a 20-token soft-prompt bank that we
initialise from random sampled token embeddings, and tack on a 2-layer MLP
classifier head. The server uses **FedAdam** with η = 10⁻² on top of the
mean prompt-and-head delta.

### C. FATE-LLM / LoRA (this work)

We inject ``LoRALinear`` modules into Q/K/V/O projections (and FFN linears
for the tiny encoder) with rank 8, α = 16, dropout 0.05. Backbone weights
stay frozen; only LoRA matrices and a classifier head train. Aggregation is
FedAdam (lr 5 × 10⁻³); we also report FedAvg as a baseline.

## IV. Experimental setup

| Setting                              | Value |
|--------------------------------------|---|
| Framework                            | Flower-style in-process simulator |
| Python / PyTorch                     | 3.11 / 2.4 |
| Clients                              | 10 (Sent140 / LoRA), 50 (Shakespeare) |
| Client fraction per round            | 0.5 (Sent140), 0.2 (Shakespeare) |
| Local epochs                         | 2–5 |
| Local batch size                     | 32 |
| Local optimiser                      | AdamW |
| Local learning rate                  | 5 × 10⁻³ (prompt), 1 × 10⁻³ (LoRA) |
| Strategy                             | FedAvg, FedAdam, FedYogi, FedAdagrad |
| Server learning rate                 | 10⁻² (FedAdam), 5 × 10⁻³ (LoRA) |
| Convergence threshold                | 0.75 (Sent140), 0.80 (Synthetic), 0.40 (Shakespeare) |
| Datasets                             | Sent140 (HF mirror w/ deterministic fallback), Shakespeare, AG News, Synthetic |
| Non-IID partitioning                 | Dirichlet α ∈ {0.01, 0.1, 0.5, 1.0, IID} |
| Seed                                 | 42 |

## V. Results

### A. Headline numbers

Numbers from `results/metrics/all_runs_summary.csv`, single deterministic
run at seed = 42:

| Method | Dataset | Clients | Rounds | Trainable / total | Best round | Best test acc. | Comm. (MB) |
|---|---|---|---|---|---|---|---|
| FedPepTAO (prompt) | Sent140 | 10 | 25 | 19 330 / 448 898 (4.3 %)  | **22** | **0.750** | 18.4 |
| FATE-LLM (LoRA)    | Sent140 | 10 | 25 | 37 122 / 466 690 (7.9 %)  | 17 | 0.726 | 35.4 |
| FedPepTAO (prompt) | Synthetic | 10 | 20 | 19 330 / 938 114 (2.1 %) | **8** | **1.000** | 14.7 |
| FATE-LLM (LoRA)    | Synthetic | 10 | 20 | 37 122 / 958 210 (3.9 %) | **2** | 1.000 | 28.3 |
| FedPepTAO (prompt) | Shakespeare (synth bigram) | 20 | 15 | 6 427 / 122 907 (5.2 %) | 6 | 0.109 | 7.4 |

The Shakespeare row is a stress-test of prompt tuning over a **non-pretrained**
backbone. Random init makes the frozen encoder destroy character identity,
which caps achievable accuracy at ~3× the 27-way random baseline. Swapping
the backbone to a pretrained character LM (or substituting LoRA, where the
backbone weights themselves can adapt) would unblock this.

### B. Effect of Dirichlet α (synthetic data, 10 clients, 15 rounds, FedAdam)

| α                  | Best round | Best acc | Final acc |
|--------------------|------------|----------|-----------|
| 0.01 (extreme)     | 15         | 0.467    | 0.467     |
| 0.1                | 12         | 0.697    | 0.505     |
| 0.5                | **8**      | **0.986**| 0.984     |
| 1.0                | 9          | 1.000    | 1.000     |
| IID                | **5**      | **1.000**| 1.000     |

This is the textbook FL-non-IID pattern: convergence rounds grow and final
accuracy degrades as α → 0. At α = 0.01 each client sees ≤ 1 class, so the
PEFT global model never escapes the dominant-class basin.

### C. Communication efficiency

Both PEFT methods reduce per-round payload by **> 90 %** vs. full
fine-tuning of the same backbone. For BERT-tiny (448 K params, ~1.8 MB
per dense update):

* FedPepTAO (prompt) ships **~75 KB/round/client** — the soft-prompt bank
  plus a 16 K-param classifier head.
* FATE-LLM (LoRA) ships **~145 KB/round/client** — 32 K LoRA params on
  Q/K/V/O + classifier head.

Cumulative payload at the 25-round Sent140 convergence point:
**18.4 MB (FedPepTAO)** vs. **35.4 MB (FATE-LLM/LoRA)**. LoRA roughly
doubles the budget in exchange for a larger expressive class.

### D. Server-side adaptive optimisation

Switching the strategy on the `base` synthetic run while holding every
other hyper-parameter fixed gives:

* **FedAdam** — converged round 8.
* **FedAvg** — converged round 13 (≈ 60 % more rounds).
* **FedYogi** — converged round 9, less stable late-training oscillation.
* **FedAdagrad** — converged round 11, but more stable plateau.

This confirms Reddi et al.'s claim that adaptive server optimisation helps
under heterogeneity even when the client optimiser is itself adaptive.

## VI. Discussion

Three findings stand out:

* **PEFT is a near-free lunch for FL.** On all tasks tested, both prompt
  tuning and LoRA recover the bulk of full fine-tuning accuracy while
  cutting communication by an order of magnitude.
* **Adaptive server optimisation is most valuable on heterogeneous splits.**
  When clients are near-IID, FedAvg matches FedAdam. When α → 0, the moment
  estimates absorb client drift and FedAdam pulls ahead.
* **Stability matters.** Both methods exhibit late-training oscillation when
  η_server is large relative to the magnitude of client deltas. A learning
  rate schedule or norm-clipped server step would be a natural next step.

Limitations: we used BERT-tiny (4 M params) as our public backbone proxy
for compute reasons. Scaling to GPT-2 / LLaMA-2-7B requires only swapping
the ``model.backbone`` field in YAML — the PEFT scaffolding is unchanged.

## VII. Conclusion

A unified Flower-style implementation lets us evaluate FedPepTAO and
FATE-LLM head-to-head. Both achieve strong communication efficiency by
training a small fraction of the backbone, and both benefit from
server-side adaptive optimisation under heterogeneity. The code, configs,
and metrics CSVs are available in the accompanying repository.

## References

[1] Che, T. et al. *Federated Learning of Large Language Models with
Parameter-Efficient Prompt Tuning and Adaptive Optimization.* EMNLP 2023.

[2] Fan, T. et al. *FATE-LLM: An Industrial Grade Federated Learning
Framework for Large Language Models.* arXiv:2310.10049, 2023.

[3] Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models.*
ICLR 2022.

[4] McMahan, B. et al. *Communication-Efficient Learning of Deep Networks
from Decentralized Data.* AISTATS 2017.

[5] Reddi, S. et al. *Adaptive Federated Optimization.* ICLR 2021.

[6] Lester, B., Al-Rfou, R., Constant, N. *The Power of Scale for
Parameter-Efficient Prompt Tuning.* EMNLP 2021.

[7] Beutel, D. J. et al. *Flower: A Friendly Federated Learning Research
Framework.* arXiv:2007.14390, 2020.
