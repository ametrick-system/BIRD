from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def make_causal_mask(
    seq_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a causal mask for autoregressive attention

    Returns:
        mask of shape (1, 1, seq_len, seq_len), where:
            True  = allowed attention
            False = masked-out attention

    Entry (i, j) is True iff j <= i
    """
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)

def expand_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Expand a 2D attention mask to 4D for multi-head attention

    Input:
        attention_mask: shape (batch_size, seq_len), where:
            1 = keep token
            0 = masked/padded token

    Returns:
        expanded mask of shape (batch_size, 1, 1, seq_len), dtype=bool
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            f"expand_attention_mask expected shape (batch_size, seq_len), got {tuple(attention_mask.shape)}"
        )

    return attention_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(1)

def expand_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Expand a 2D padding mask (B, T) into attention mask (B, 1, 1, T)

    True = keep token
    False = mask out token

    This is just a thin wrapper around expand_attention_mask
    """
    if attention_mask.ndim != 2:
        raise ValueError(
            f"attention_mask must have shape (batch_size, seq_len), got {tuple(attention_mask.shape)}"
        )

    return expand_attention_mask(attention_mask)

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention

    Inputs:
        query: shape (batch_size, num_heads, query_len, head_dim)
        key: shape (batch_size, num_heads, key_len, head_dim)
        value: shape (batch_size, num_heads, key_len, head_dim)

        attention_mask: optional boolean mask broadcastable to
            (batch_size, num_heads, query_len, key_len)
            True  = keep
            False = mask out

    Returns:
        context: shape (batch_size, num_heads, query_len, head_dim)
        attn_weights: shape (batch_size, num_heads, query_len, key_len)
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "ScaledDotProductAttention expected query/key/value to all have shape (batch_size, num_heads, seq_len, head_dim)"
            )

        batch_size, num_heads, query_len, head_dim = query.shape
        _, _, key_len, key_head_dim = key.shape
        _, _, value_len, value_head_dim = value.shape

        if key_head_dim != head_dim or value_head_dim != head_dim:
            raise ValueError("Query, key, and value must have the same head_dim")
        if key_len != value_len:
            raise ValueError("Key and value must have the same sequence length")

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        # scores shape: (batch_size, num_heads, query_len, key_len)

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise TypeError(
                    f"attention_mask must have dtype torch.bool, got {attention_mask.dtype}"
                )
            scores = scores.masked_fill(~attention_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, value)
        # context shape: (batch_size, num_heads, query_len, head_dim)

        return context, attn_weights

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module supporting both self-attention and cross-attention

    Self-attention:
        forward(hidden_states, key_value_states=None, ...)

    Cross-attention:
        forward(hidden_states, key_value_states=encoder_hidden_states, ...)

    Inputs:
        hidden_states: shape (batch_size, query_len, d_model)
        key_value_states: optional, shape (batch_size, key_len, d_model)

        attention_mask:
            optional boolean mask broadcastable to
            (batch_size, num_heads, query_len, key_len)

    Returns:
        output: shape (batch_size, query_len, d_model)
        attn_weights: shape (batch_size, num_heads, query_len, key_len)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attention = ScaledDotProductAttention(dropout=dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _shape_projection(
        self,
        x: torch.Tensor,
        seq_len: int,
        batch_size: int,
    ) -> torch.Tensor:
        """
        Reshape projected tensor from:
            (batch_size, seq_len, d_model)
        to:
            (batch_size, num_heads, seq_len, head_dim)
        """
        return (
            x.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def _merge_heads(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Merge multi-head tensor from:
            (batch_size, num_heads, seq_len, head_dim)
        to:
            (batch_size, seq_len, d_model)
        """
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape (batch_size, seq_len, d_model), got {tuple(hidden_states.shape)}"
            )

        batch_size, query_len, d_model = hidden_states.shape
        if d_model != self.d_model:
            raise ValueError(
                f"Expected hidden_states last dim = {self.d_model}, got {d_model}"
            )

        if key_value_states is None:
            key_value_states = hidden_states

        if key_value_states.ndim != 3:
            raise ValueError(
                f"key_value_states must have shape (batch_size, seq_len, d_model), got {tuple(key_value_states.shape)}"
            )

        kv_batch_size, key_len, kv_d_model = key_value_states.shape
        if kv_batch_size != batch_size:
            raise ValueError("hidden_states and key_value_states must have same batch size")
        if kv_d_model != self.d_model:
            raise ValueError(
                f"Expected key_value_states last dim = {self.d_model}, got {kv_d_model}"
            )

        # projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(key_value_states)
        value = self.v_proj(key_value_states)

        query = self._shape_projection(query, query_len, batch_size)
        key = self._shape_projection(key, key_len, batch_size)
        value = self._shape_projection(value, key_len, batch_size)

        # causal mask
        causal_mask = None
        if use_causal_mask:
            causal_mask = torch.tril(
                torch.ones((query_len, key_len), dtype=torch.bool, device=hidden_states.device)
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,Q,K)

        # combine masks
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise TypeError(
                    f"attention_mask must have dtype torch.bool, got {attention_mask.dtype}."
                )

        if causal_mask is not None:
            attention_mask = (
                causal_mask if attention_mask is None else (attention_mask & causal_mask)
            )

        # attention
        context, attn_weights = self.attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
        )

        output = self._merge_heads(context, batch_size=batch_size, seq_len=query_len)
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        return output, attn_weights