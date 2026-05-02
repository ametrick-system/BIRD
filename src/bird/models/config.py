from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class TokenizerConfig:
    """
    Configuration for DNA tokenization
    """

    tokenizer_type: str = "char"   # "char", "kmer", "bpe"
    vocab_size: int = 12
    max_length: int = 512

    # Special token ids
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4
    bos_token_id: int = 5
    eos_token_id: int = 6

    # More tokenizer options
    kmer_size: Optional[int] = None
    bpe_vocab_size: Optional[int] = None

@dataclass
class ModelConfig:
    """
    Shared Transformer model configuration -- base config used by GPT, BERT, and BART variants
    """

    # Vocabulary / input
    vocab_size: int = 12
    max_position_embeddings: int = 512
    padding_idx: int = 0

    # Core architecture
    d_model: int = 128
    num_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Depth
    num_layers: int = 4

    # Embedding options
    use_token_type_embeddings: bool = False
    type_vocab_size: int = 2

    # General transformer options
    tie_word_embeddings: bool = False

    # Architecture flags
    is_decoder: bool = False
    use_causal_mask: bool = False
    use_cross_attention: bool = False

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {self.max_position_embeddings}"
            )
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be positive, got {self.d_ff}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.padding_idx < 0:
            raise ValueError(f"padding_idx must be nonnegative, got {self.padding_idx}")
        if self.type_vocab_size <= 0:
            raise ValueError(f"type_vocab_size must be positive, got {self.type_vocab_size}")

@dataclass
class GPTConfig(ModelConfig):
    """
    Decoder-only Transformer config
    """

    is_decoder: bool = True
    use_causal_mask: bool = True
    use_cross_attention: bool = False
    use_token_type_embeddings: bool = False


@dataclass
class BERTConfig(ModelConfig):
    """
    Encoder-only Transformer config
    """

    is_decoder: bool = False
    use_causal_mask: bool = False
    use_cross_attention: bool = False
    use_token_type_embeddings: bool = True

@dataclass
class BARTConfig(ModelConfig):
    """
    Encoder-decoder Transformer config.

    BART uses explicit encoder and decoder depths. The inherited `num_layers`
    field is retained for compatibility with the shared base config, but BART
    model construction should use `num_encoder_layers` and `num_decoder_layers`.
    """

    num_encoder_layers: int = 4
    num_decoder_layers: int = 4

    is_decoder: bool = False
    use_causal_mask: bool = False
    use_cross_attention: bool = True
    use_token_type_embeddings: bool = False
    tie_word_embeddings: bool = True

    def validate(self) -> None:
        super().validate()

        if self.num_encoder_layers <= 0:
            raise ValueError(
                f"num_encoder_layers must be positive, got {self.num_encoder_layers}"
            )
        if self.num_decoder_layers <= 0:
            raise ValueError(
                f"num_decoder_layers must be positive, got {self.num_decoder_layers}"
            )

__all__ = [
    "TokenizerConfig",
    "ModelConfig",
    "GPTConfig",
    "BERTConfig",
    "BARTConfig",
]