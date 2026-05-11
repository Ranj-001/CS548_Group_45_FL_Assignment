"""LoRA adapters (FATE-LLM style).

Implements the standard Hu et al. low-rank adaptation: each targeted
`nn.Linear` weight ``W`` (shape ``[out, in]``) gets a residual ``α/r · B A``
where ``A ∈ R^{r × in}`` is zero-init free and ``B ∈ R^{out × r}`` is
zero-initialised. Only ``A`` and ``B`` are trainable; ``W`` stays frozen.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with an additive LoRA branch."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Frozen base layer
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA params — A: (r, in), B: (out, r)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


# ---------------------------------------------------------------------------
# Surgery utilities
# ---------------------------------------------------------------------------

DEFAULT_TARGET_MODULE_KEYWORDS: tuple[str, ...] = (
    "query", "key", "value", "q_proj", "k_proj", "v_proj", "out_proj", "dense",
    "attention.self.query", "attention.self.key", "attention.self.value",
)


def _split_qualname(qualname: str) -> tuple[str, str]:
    if "." not in qualname:
        return "", qualname
    parent, _, name = qualname.rpartition(".")
    return parent, name


def _get_submodule(root: nn.Module, qualname: str) -> nn.Module:
    if not qualname:
        return root
    mod = root
    for part in qualname.split("."):
        mod = getattr(mod, part)
    return mod


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_keywords: Iterable[str] = DEFAULT_TARGET_MODULE_KEYWORDS,
) -> int:
    """Recursively replace matching `nn.Linear` modules with `LoRALinear`.

    Returns the number of injected LoRA layers.
    """
    target_keywords = tuple(target_keywords)
    targets: list[tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(kw in name for kw in target_keywords):
            targets.append((name, mod))

    for name, mod in targets:
        parent_name, attr = _split_qualname(name)
        parent = _get_submodule(model, parent_name)
        setattr(parent, attr, LoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout))

    return len(targets)


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze every parameter that is not part of a LoRA branch."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
