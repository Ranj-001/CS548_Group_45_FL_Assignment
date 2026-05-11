"""Soft-prompt tuning (FedPepTAO).

Implements Lester et al.'s "soft prompts" — a small bank of learnable embedding
vectors prepended to every input sequence. The backbone is **frozen**; only the
prompt embeddings (and a tiny classification head) are trainable, so each
client uploads ~O(prompt_length × hidden_dim) parameters per round.

Compatible with two model families:
  * HuggingFace `AutoModel` (BERT-tiny / DistilBERT / ...) — uses
    `inputs_embeds=` to splice prompts in.
  * Our offline `TinyTransformerEncoder` (see :mod:`models`) — exposes the same
    `(embed_tokens, encoder, hidden_size)` contract.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SoftPromptTuningModel(nn.Module):
    """Wraps a frozen backbone with a learnable soft-prompt bank + classifier."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int,
        embed_tokens: nn.Embedding,
        num_classes: int,
        prompt_length: int = 20,
        encoder_fn: Optional[callable] = None,
        is_hf: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_tokens = embed_tokens
        self.hidden_size = hidden_size
        self.prompt_length = prompt_length
        self.is_hf = is_hf
        self._encoder_fn = encoder_fn

        # Soft prompts initialised from random sampled token embeddings —
        # the original Lester et al. trick.
        with torch.no_grad():
            init_ids = torch.randint(
                low=0, high=embed_tokens.num_embeddings, size=(prompt_length,)
            )
            init = embed_tokens(init_ids).detach().clone()
        self.soft_prompts = nn.Parameter(init)

        # Tiny classification head: pooled hidden state → logits.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_classes),
        )

        # Freeze every backbone param. Only soft_prompts + classifier train.
        for p in self.backbone.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        bsz = input_ids.size(0)
        tok_emb = self.embed_tokens(input_ids)                       # (B, L, H)
        prompts = self.soft_prompts.unsqueeze(0).expand(bsz, -1, -1) # (B, P, H)
        inputs_embeds = torch.cat([prompts, tok_emb], dim=1)         # (B, P+L, H)

        if attention_mask is not None:
            prompt_mask = torch.ones(
                bsz, self.prompt_length,
                device=attention_mask.device, dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        if self.is_hf:
            out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            # CLS-position pooling: the prompt sits in front, so [:, 0] is the
            # first prompt token; we pool over the masked positions instead.
            hidden = out.last_hidden_state
            mask_f = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            assert self._encoder_fn is not None
            hidden = self._encoder_fn(inputs_embeds, attention_mask)
            mask_f = attention_mask.unsqueeze(-1).float() if attention_mask is not None else None
            if mask_f is None:
                pooled = hidden.mean(dim=1)
            else:
                pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        logits = self.classifier(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
