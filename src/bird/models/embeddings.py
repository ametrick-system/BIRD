from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Standard token embedding lookup

    Input:
        input_ids: LongTensor of shape (batch_size, seq_len)

    Output:
        embeddings: FloatTensor of shape (batch_size, seq_len, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise TypeError(
                f"TokenEmbedding expected input_ids of dtype torch.long, got {input_ids.dtype}"
            )
        return self.embedding(input_ids)

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings

    Positions are assigned as:
        [0, 1, 2, ..., seq_len - 1]

    regardless of token content

    Input:
        input_ids: LongTensor of shape (batch_size, seq_len)

    Output:
        positional_embeddings: FloatTensor of shape (batch_size, seq_len, d_model)
    """

    def __init__(self, max_position_embeddings: int, d_model: int) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model

        self.embedding = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=d_model,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"LearnedPositionalEmbedding expected input_ids with shape (batch_size, seq_len), got shape {tuple(input_ids.shape)}"
            )

        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_position_embeddings={self.max_position_embeddings}"
            )

        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        return self.embedding(position_ids)

class TokenTypeEmbedding(nn.Module):
    """
    Token-type / segment embedding (useful for BERT-style sentence-pair setups)

    Input:
        token_type_ids: LongTensor of shape (batch_size, seq_len)

    Output:
        token_type_embeddings: FloatTensor of shape (batch_size, seq_len, d_model)
    """

    def __init__(self, type_vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.type_vocab_size = type_vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(
            num_embeddings=type_vocab_size,
            embedding_dim=d_model,
        )

    def forward(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        if token_type_ids.dtype != torch.long:
            raise TypeError(
                f"TokenTypeEmbedding expected token_type_ids of dtype torch.long, got {token_type_ids.dtype}"
            )
        return self.embedding(token_type_ids)


class InputEmbedding(nn.Module):
    """
    Full input embedding module shared across GPT, BERT, and BART

    Combines:
        - token embeddings
        - learned positional embeddings
        - optional token type embeddings
        - layer norm
        - dropout

    Inputs:
        input_ids: LongTensor of shape (batch_size, seq_len)
        token_type_ids: optional LongTensor of shape (batch_size, seq_len)

    Output:
        embeddings: FloatTensor of shape (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_position_embeddings: int,
        padding_idx: Optional[int] = None,
        dropout: float = 0.1,
        use_token_type_embeddings: bool = False,
        type_vocab_size: int = 2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_position_embeddings = max_position_embeddings
        self.padding_idx = padding_idx
        self.dropout_prob = dropout
        self.use_token_type_embeddings = use_token_type_embeddings
        self.type_vocab_size = type_vocab_size
        self.use_layer_norm = use_layer_norm

        self.token_embeddings = TokenEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            padding_idx=padding_idx,
        )
        self.position_embeddings = LearnedPositionalEmbedding(
            max_position_embeddings=max_position_embeddings,
            d_model=d_model,
        )

        if use_token_type_embeddings:
            self.token_type_embeddings: Optional[TokenTypeEmbedding] = TokenTypeEmbedding(
                type_vocab_size=type_vocab_size,
                d_model=d_model,
            )
        else:
            self.token_type_embeddings = None

        self.layer_norm = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                f"InputEmbedding expected input_ids with shape (batch_size, seq_len), got shape {tuple(input_ids.shape)}"
            )

        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(input_ids)

        embeddings = token_embeds + position_embeds

        if self.use_token_type_embeddings:
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            elif token_type_ids.shape != input_ids.shape:
                raise ValueError(
                    f"token_type_ids must have shape {tuple(input_ids.shape)}, "
                    f"got {tuple(token_type_ids.shape)}."
                )

            assert self.token_type_embeddings is not None
            type_embeds = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + type_embeds
        else:
            if token_type_ids is not None:
                raise ValueError(
                    "token_type_ids were provided, but use_token_type_embeddings=False"
                )

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings