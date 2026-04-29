from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bird.models.attention import expand_attention_mask, make_causal_mask
from bird.models.blocks import DecoderBlock
from bird.models.config import GPTConfig
from bird.models.embeddings import InputEmbedding

class GPTModel(nn.Module):
    """
    Decoder-only Transformer language model for DNA sequences

    Architecture:
        - input embeddings
        - stack of decoder blocks (self-attention only)
        - final layer norm
        - LM head over vocabulary

    Inputs:
        input_ids: LongTensor of shape (batch_size, seq_len)
        attention_mask: optional LongTensor/BoolTensor of shape (batch_size, seq_len)
            1/True = keep token
            0/False = masked/padded token
        labels: optional LongTensor of shape (batch_size, seq_len)

    Returns:
        dict with:
            - logits: (batch_size, seq_len, vocab_size)
            - loss: scalar tensor if labels are provided, else None
            - hidden_states: (batch_size, seq_len, d_model)
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.embeddings = InputEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_position_embeddings=config.max_position_embeddings,
            padding_idx=config.padding_idx,
            dropout=config.dropout,
            use_token_type_embeddings=False,
            use_layer_norm=True,
        )

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    use_cross_attention=False,
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
    ) -> torch.Tensor:
        """
        Build the combined GPT attention mask:
            causal mask AND optional padding mask

        Returns:
            bool tensor of shape (batch_size or 1, 1, seq_len, seq_len)
        """
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch_size, seq_len), got {tuple(input_ids.shape)}."
            )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        causal_mask = make_causal_mask(seq_len=seq_len, device=device)
        # shape: (1, 1, seq_len, seq_len)

        if attention_mask is None:
            return causal_mask

        if attention_mask.ndim != 2:
            raise ValueError(
                f"attention_mask must have shape (batch_size, seq_len), got {tuple(attention_mask.shape)}."
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape must match input_ids shape {tuple(input_ids.shape)}, "
                f"got {tuple(attention_mask.shape)}."
            )

        padding_mask = expand_attention_mask(attention_mask)
        # shape: (batch_size, 1, 1, seq_len)

        combined_mask = padding_mask & causal_mask
        # broadcast result: (batch_size, 1, seq_len, seq_len)

        return combined_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        if input_ids.dtype != torch.long:
            raise TypeError(
                f"input_ids must have dtype torch.long, got {input_ids.dtype}."
            )
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape (batch_size, seq_len), got {tuple(input_ids.shape)}."
            )

        hidden_states = self.embeddings(input_ids)

        combined_attention_mask = self._build_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=combined_attention_mask,
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
                raise TypeError(
                    f"labels must have dtype torch.long, got {labels.dtype}."
                )

            # Standard next-token LM loss:
            # predict labels[:, 1:] from logits[:, :-1, :]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.padding_idx,
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
        }