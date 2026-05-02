from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bird.models.attention import expand_padding_mask
from bird.models.blocks import EncoderBlock, DecoderBlock
from bird.models.config import BARTConfig
from bird.models.embeddings import InputEmbedding

class BARTModel(nn.Module):
    """
    BART-style encoder-decoder Transformer

    Inputs:
        encoder_input_ids: corrupted source sequence
        decoder_input_ids: clean target prefix for teacher forcing

    Labels:
        labels: clean target sequence token ids
                loss is computed autoregressively on decoder outputs

    Returns:
        {
            "logits": (B, T_dec, vocab_size),
            "loss": scalar or None,
            "encoder_hidden_states": (B, T_enc, d_model),
            "decoder_hidden_states": (B, T_dec, d_model),
        }
    """

    def __init__(self, config: BARTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        # Shared token embedding for encoder + decoder
        self.shared_embeddings = InputEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_position_embeddings=config.max_position_embeddings,
            padding_idx=config.padding_idx,
            dropout=config.dropout,
            use_token_type_embeddings=False,
            type_vocab_size=config.type_vocab_size,
            use_layer_norm=True,
        )

        # Encoder stack
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.encoder_final_layer_norm = nn.LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        # Decoder stack
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    use_cross_attention=True,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.decoder_final_layer_norm = nn.LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
        )

        # LM head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.shared_embeddings.token_embeddings.embedding.weight

    def _build_encoder_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build bidirectional encoder attention mask from padding mask

        Returns:
            tensor of shape (B, 1, 1, T_enc), or None
        """
        if attention_mask is None:
            return None

        if attention_mask.ndim != 2:
            raise ValueError(
                f"encoder attention_mask must have shape (batch_size, seq_len), got {tuple(attention_mask.shape)}."
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"encoder attention_mask shape must match encoder_input_ids shape {tuple(input_ids.shape)}, "
                f"got {tuple(attention_mask.shape)}."
            )

        return expand_padding_mask(attention_mask)

    def _build_decoder_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Build decoder padding mask.
        The causal mask itself is handled inside DecoderBlock / attention code

        Returns:
            tensor of shape (B, 1, 1, T_dec), or None
        """
        if attention_mask is None:
            return None

        if attention_mask.ndim != 2:
            raise ValueError(
                f"decoder attention_mask must have shape (batch_size, seq_len), got {tuple(attention_mask.shape)}."
            )
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"decoder attention_mask shape must match decoder_input_ids shape {tuple(input_ids.shape)}, "
                f"got {tuple(attention_mask.shape)}."
            )

        return expand_padding_mask(attention_mask)

    def encode(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_input_ids.dtype != torch.long:
            raise TypeError(
                f"encoder_input_ids must have dtype torch.long, got {encoder_input_ids.dtype}."
            )
        if encoder_input_ids.ndim != 2:
            raise ValueError(
                f"encoder_input_ids must have shape (batch_size, seq_len), got {tuple(encoder_input_ids.shape)}."
            )

        hidden_states = self.shared_embeddings(
            input_ids=encoder_input_ids,
            token_type_ids=None,
        )

        encoder_mask = self._build_encoder_attention_mask(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
        )

        for block in self.encoder_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=encoder_mask,
            )

        hidden_states = self.encoder_final_layer_norm(hidden_states)
        return hidden_states

    def decode(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if decoder_input_ids.dtype != torch.long:
            raise TypeError(
                f"decoder_input_ids must have dtype torch.long, got {decoder_input_ids.dtype}."
            )
        if decoder_input_ids.ndim != 2:
            raise ValueError(
                f"decoder_input_ids must have shape (batch_size, seq_len), got {tuple(decoder_input_ids.shape)}."
            )

        hidden_states = self.shared_embeddings(
            input_ids=decoder_input_ids,
            token_type_ids=None,
        )

        decoder_mask = self._build_decoder_attention_mask(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
        )

        cross_mask = (
            expand_padding_mask(encoder_attention_mask)
            if encoder_attention_mask is not None
            else None
        )

        for block in self.decoder_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=decoder_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=cross_mask,
                use_causal_mask=True,
            )

        hidden_states = self.decoder_final_layer_norm(hidden_states)
        return hidden_states

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        encoder_hidden_states = self.encode(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
        )

        decoder_hidden_states = self.decode(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
        )

        logits = self.lm_head(decoder_hidden_states)

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            if labels.shape != decoder_input_ids.shape:
                raise ValueError(
                    f"labels shape must match decoder_input_ids shape {tuple(decoder_input_ids.shape)}, "
                    f"got {tuple(labels.shape)}."
                )
            if labels.dtype != torch.long:
                raise TypeError(f"labels must have dtype torch.long, got {labels.dtype}.")

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "encoder_hidden_states": encoder_hidden_states,
            "decoder_hidden_states": decoder_hidden_states,
        }