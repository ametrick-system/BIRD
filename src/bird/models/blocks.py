from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from bird.models.attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network

    Applies:
        Linear(d_model -> d_ff)
        Activation
        Dropout
        Linear(d_ff -> d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderBlock(nn.Module):
    """
    Standard Transformer encoder block

    Used for:
        - BERT
        - BART encoder
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        hidden_states = self.norm1(hidden_states)

        # Feed-forward
        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + self.dropout(ffn_output)
        hidden_states = self.norm2(hidden_states)

        return hidden_states

class DecoderBlock(nn.Module):
    """
    Transformer decoder block

    Used for:
        - GPT (no cross-attention)
        - BART decoder (with cross-attention)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        use_cross_attention: bool = False,  # True for BART, False for GPT
    ) -> None:
        super().__init__()

        self.use_cross_attention = use_cross_attention

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        if use_cross_attention:
            self.cross_attn = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.cross_attn = None

        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = False,
    ) -> torch.Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        hidden_states = self.norm1(hidden_states)

        # Cross-attention (BART only)
        if self.use_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states must be provided for cross-attention.")

            cross_output, _ = self.cross_attn(
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                use_causal_mask=False,
            )
            hidden_states = hidden_states + self.dropout(cross_output)
            hidden_states = self.norm2(hidden_states)

            ffn_norm = self.norm3
        else:
            ffn_norm = self.norm2

        # Feed-forward
        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + self.dropout(ffn_output)
        hidden_states = ffn_norm(hidden_states)

        return hidden_states