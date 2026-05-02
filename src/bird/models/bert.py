from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bird.models.attention import expand_attention_mask
from bird.models.blocks import EncoderBlock
from bird.models.config import BERTConfig
from bird.models.embeddings import InputEmbedding

class BERTModel(nn.Module):
    """
    Encoder-only Transformer for masked language modeling

    Returns:
        {
            "logits": (batch_size, seq_len, vocab_size),
            "loss": scalar or None,
            "hidden_states": (batch_size, seq_len, d_model),
        }
    """

    def __init__(self, config: BERTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.embeddings = InputEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_position_embeddings=config.max_position_embeddings,
            padding_idx=config.padding_idx,
            dropout=config.dropout,
            use_token_type_embeddings=config.use_token_type_embeddings,
            type_vocab_size=config.type_vocab_size,
            use_layer_norm=True,
        )

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.token_embeddings.embedding.weight

    def _build_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build bidirectional encoder attention mask from padding mask

        Returns:
            bool tensor of shape (batch_size, 1, 1, seq_len), or None
        """
        if attention_mask is None:
            return None

        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must have shape (batch_size, seq_len), got {tuple(attention_mask.shape)}."
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape must match input_ids shape {tuple(input_ids.shape)}, "
                f"got {tuple(attention_mask.shape)}."
            )

        return expand_attention_mask(attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        if input_ids.dtype != torch.long:
            raise TypeError(f"input_ids must have dtype torch.long, got {input_ids.dtype}.")
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch_size, seq_len), got {tuple(input_ids.shape)}."
            )

        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        encoder_attention_mask = self._build_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=encoder_attention_mask,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels shape must match input_ids shape {tuple(input_ids.shape)}, "
                    f"got {tuple(labels.shape)}."
                )
            if labels.dtype != torch.long:
                raise TypeError(f"labels must have dtype torch.long, got {labels.dtype}.")

            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
        }