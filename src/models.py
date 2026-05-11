"""Model factories.

Produces three flavours of LLM-adaptation models:

* ``prompt_tuning`` — frozen backbone + soft prompts (FedPepTAO).
* ``lora``          — frozen backbone + LoRA adapters (FATE-LLM).
* ``full``          — full fine-tuning baseline (every backbone param trains).

Backbones:

* HuggingFace ``AutoModel`` (e.g. ``prajjwal1/bert-tiny``) when available.
* Offline ``TinyTransformerEncoder`` — a self-contained, small transformer
  encoder we ship so the pipeline runs in fully air-gapped environments.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .config import ModelConfig
from .lora import LoRALinear, freeze_non_lora, inject_lora
from .prompt_tuning import SoftPromptTuningModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tiny offline transformer encoder
# ---------------------------------------------------------------------------


class _PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class _CustomEncoderLayer(nn.Module):
    """Plain transformer encoder layer built from `nn.Linear` modules only.

    Avoids `nn.MultiheadAttention` / `nn.TransformerEncoderLayer` because both
    use a fused fast-path that directly accesses `.weight` of the inner linears
    — incompatible with LoRA wrapping.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def _attn(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            # attn_mask: (B, S) with 1 = keep, 0 = pad → broadcast to (B,1,1,S)
            mask = attn_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        ctx = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq, self.hidden_size)
        return self.o_proj(ctx)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.dropout(self._attn(self.norm1(x), attn_mask))
        x = x + self.dropout(self.linear2(self.act(self.linear1(self.norm2(x)))))
        return x


class TinyTransformerEncoder(nn.Module):
    """Self-contained transformer encoder — runs entirely offline.

    Built from primitive ``nn.Linear`` modules so the LoRA injector can wrap
    any projection without tripping PyTorch's fused fast-paths.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.pos_enc = _PositionalEncoding(hidden_size, max_len=max_seq_len)
        self.layers = nn.ModuleList([
            _CustomEncoderLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size

    def encode(
        self, inputs_embeds: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.pos_enc(inputs_embeds)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Simple classifier (for the "full" baseline)
# ---------------------------------------------------------------------------


class TinyClassifier(nn.Module):
    """End-to-end classifier on top of a TinyTransformerEncoder (full fine-tune)."""

    def __init__(self, encoder: TinyTransformerEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None):
        emb = self.encoder.embed_tokens(input_ids)
        hidden = self.encoder.encode(emb, attention_mask)
        mask_f = attention_mask.unsqueeze(-1).float() if attention_mask is not None else None
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1) if mask_f is not None else hidden.mean(dim=1)
        logits = self.head(pooled)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# HuggingFace wrapper used as the classifier when LoRA / full FT is requested
# on an HF backbone.
# ---------------------------------------------------------------------------


class _HFClassifier(nn.Module):
    def __init__(self, hf_model: nn.Module, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = hf_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        if attention_mask is not None:
            mask_f = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


@dataclass
class ModelBundle:
    model: nn.Module
    trainable_param_count: int
    total_param_count: int
    method: str

    @property
    def trainable_ratio(self) -> float:
        if self.total_param_count == 0:
            return 0.0
        return self.trainable_param_count / self.total_param_count


def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _try_load_hf_backbone(name: str):
    try:
        from transformers import AutoModel, AutoConfig  # type: ignore
    except Exception as exc:
        log.warning("transformers unavailable: %s", exc)
        return None, None
    try:
        cfg = AutoConfig.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        return model, cfg
    except Exception as exc:
        log.warning("Could not load HF backbone %s: %s — using tiny fallback.", name, exc)
        return None, None


def build_model(model_cfg: ModelConfig, *, num_classes: int, vocab_size: int) -> ModelBundle:
    method = model_cfg.method.lower()
    if method not in {"prompt_tuning", "lora", "full"}:
        raise ValueError(f"Unknown method: {method!r}")

    use_hf = model_cfg.backbone != "tiny_fallback"
    hf_model, hf_cfg = (None, None)
    if use_hf:
        hf_model, hf_cfg = _try_load_hf_backbone(model_cfg.backbone)
        if hf_model is None:
            use_hf = False

    if use_hf:
        hidden_size = getattr(hf_cfg, "hidden_size", None) or getattr(hf_cfg, "d_model", 256)
        if method == "prompt_tuning":
            embed_tokens = hf_model.get_input_embeddings()
            model = SoftPromptTuningModel(
                backbone=hf_model,
                hidden_size=hidden_size,
                embed_tokens=embed_tokens,
                num_classes=num_classes,
                prompt_length=model_cfg.prompt_length,
                is_hf=True,
            )
        elif method == "lora":
            classifier = _HFClassifier(hf_model, hidden_size, num_classes)
            n_injected = inject_lora(
                classifier.backbone,
                rank=model_cfg.lora_rank,
                alpha=model_cfg.lora_alpha,
                dropout=model_cfg.lora_dropout,
            )
            log.info("Injected %d LoRA layers into %s.", n_injected, model_cfg.backbone)
            freeze_non_lora(classifier.backbone)
            # Classifier head trains as well (it is the only task-specific layer).
            for p in classifier.classifier.parameters():
                p.requires_grad = True
            model = classifier
        else:  # full
            model = _HFClassifier(hf_model, hidden_size, num_classes)
    else:
        encoder = TinyTransformerEncoder(
            vocab_size=vocab_size,
            hidden_size=model_cfg.hidden_size,
            num_layers=model_cfg.num_hidden_layers,
            num_heads=model_cfg.num_attention_heads,
            max_seq_len=512,
        )
        if method == "prompt_tuning":
            model = SoftPromptTuningModel(
                backbone=encoder,
                hidden_size=encoder.hidden_size,
                embed_tokens=encoder.embed_tokens,
                num_classes=num_classes,
                prompt_length=model_cfg.prompt_length,
                encoder_fn=encoder.encode,
                is_hf=False,
            )
        elif method == "lora":
            classifier = TinyClassifier(encoder, num_classes)
            # Wrap every nn.Linear in TransformerEncoderLayer.
            # Wrap every Q/K/V/O projection + FFN linear in the tiny encoder.
            n_injected = inject_lora(
                classifier.encoder,
                rank=model_cfg.lora_rank,
                alpha=model_cfg.lora_alpha,
                dropout=model_cfg.lora_dropout,
                target_keywords=("q_proj", "k_proj", "v_proj", "o_proj",
                                 "linear1", "linear2"),
            )
            log.info("Injected %d LoRA layers (tiny fallback).", n_injected)
            freeze_non_lora(classifier.encoder)
            # Head trains.
            for p in classifier.head.parameters():
                p.requires_grad = True
            model = classifier
        else:
            model = TinyClassifier(encoder, num_classes)

    trainable, total = _count_params(model)
    log.info(
        "Built %s model (method=%s, trainable=%d / total=%d, ratio=%.4f%%)",
        model_cfg.backbone, method, trainable, total, 100 * trainable / max(total, 1),
    )
    return ModelBundle(
        model=model,
        trainable_param_count=trainable,
        total_param_count=total,
        method=method,
    )


# ---------------------------------------------------------------------------
# Trainable-parameter (de)serialisation helpers — only PEFT params leave
# clients, so we need a stable contract between client.py and the strategy.
# ---------------------------------------------------------------------------


def get_trainable_state(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return [(n, p.detach().cpu().clone()) for n, p in model.named_parameters() if p.requires_grad]


def trainable_keys(model: nn.Module) -> list[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def load_trainable_state(model: nn.Module, named_tensors: list[tuple[str, torch.Tensor]]) -> None:
    state = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in named_tensors:
            if n in state and state[n].shape == t.shape:
                state[n].copy_(t.to(state[n].device, dtype=state[n].dtype))


def trainable_arrays(model: nn.Module):
    import numpy as np
    return [p.detach().cpu().numpy().copy() for p in model.parameters() if p.requires_grad]


def set_trainable_arrays(model: nn.Module, arrays) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) != len(arrays):
        raise ValueError(
            f"trainable-param mismatch: model has {len(params)}, got {len(arrays)} arrays."
        )
    with torch.no_grad():
        for p, a in zip(params, arrays):
            t = torch.as_tensor(a, dtype=p.dtype, device=p.device).view_as(p)
            p.copy_(t)
