from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

class BaseDNATokenizer(ABC):
    """
    Abstract base class for DNA tokenizers
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def unk_token(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def cls_token(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def sep_token(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def mask_token(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def bos_token(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def eos_token(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def unk_token_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def cls_token_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def sep_token_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, sequence: str) -> List[str]:
        """
        Convert a raw DNA sequence string into tokens
        """
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        """
        Convert a raw DNA sequence string to token ids
        This method should NOT add model-specific special tokens
        """
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Convert token ids back into a DNA sequence string
        """
        raise NotImplementedError

    def pad_token_ids(
        self,
        sequences: Sequence[Sequence[int]],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = False,
        return_attention_mask: bool = True,
        pad_value: Optional[int] = None,
    ) -> Dict[str, List[List[int]]]:
        """
        Pad or truncate a batch of already-tokenized id sequences

        Args:
            sequences: A batch of token-id sequences
            max_length: If provided, pad/truncate all sequences to this length
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences longer than max_length
            return_attention_mask: Whether to return attention masks
            pad_value:
                - if None, uses self.pad_token_id (default behavior)
                - if set, uses this value instead (e.g. -100 for labels)

        Returns:
            A dict containing:
                - input_ids
                - attention_mask (optionally)
        """

        if pad_value is None:
            pad_value = self.pad_token_id

        input_ids = [list(seq) for seq in sequences]

        if not input_ids:
            output: Dict[str, List[List[int]]] = {"input_ids": []}
            if return_attention_mask:
                output["attention_mask"] = []
            return output

        if padding:
            # set to longest input sequence if no specified max_length
            target_length = max_length if max_length is not None else max(len(seq) for seq in input_ids)
        else:
            target_length = None

        processed: List[List[int]] = []
        attention_masks: List[List[int]] = []

        for seq in input_ids:
            if max_length is not None and len(seq) > max_length:
                if truncation:
                    seq = seq[:max_length] # truncate sequences longer than max_length if given
                else:
                    raise ValueError(
                        f"Sequence length {len(seq)} exceeds max_length={max_length} -- set truncation=True to truncate!"
                    )

            if padding:
                assert target_length is not None
                pad_len = target_length - len(seq)
                if pad_len < 0:
                    raise ValueError("Internal padding error: negative pad length encountered")

                padded_seq = seq + [self.pad_token_id] * pad_len
                processed.append(padded_seq)

                if return_attention_mask:
                    attention_masks.append([1] * len(seq) + [0] * pad_len) # include padding 0's in mask
            else:
                processed.append(seq)
                if return_attention_mask:
                    attention_masks.append([1] * len(seq)) # no padding in mask

        output = {"input_ids": processed}
        if return_attention_mask:
            output["attention_mask"] = attention_masks
        return output

    def batch_encode(
        self,
        sequences: Sequence[str],
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_attention_mask: bool = True,
    ) -> Dict[str, List[List[int]]]:
        """
        Encode a batch of raw DNA sequences without adding model-specific special tokens
        """
        encoded = [
            self.encode(seq, max_length=max_length if (max_length is not None and truncation and not padding) else None,
                        truncation=truncation)
            for seq in sequences
        ]

        return self.pad_token_ids(
            encoded,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
        )

    # -------------------------------------------------------------------------
    # Model-specific input builders (operate on already-tokenized id sequences)
    # -------------------------------------------------------------------------

    def build_bert_inputs(self, token_ids: Sequence[int]) -> List[int]:
        """
        Build a BERT-style sequence:
            [CLS] X [SEP]
        """
        return [self.cls_token_id] + list(token_ids) + [self.sep_token_id]

    def build_gpt_inputs(
        self,
        token_ids: Sequence[int],
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Build a GPT-style sequence
        Options:
            [BOS] X
            [BOS] X [EOS]
            X [EOS]
        """
        output = list(token_ids)
        if add_bos:
            output = [self.bos_token_id] + output
        if add_eos:
            output = output + [self.eos_token_id]
        return output

    def build_bart_encoder_inputs(self, token_ids: Sequence[int]) -> List[int]:
        """
        Build a BART encoder input sequence:
            X [EOS]
        """
        return list(token_ids) + [self.eos_token_id]

    def build_bart_decoder_inputs(
        self,
        token_ids: Sequence[int],
        add_eos: bool = True,
    ) -> List[int]:
        """
        Build a BART decoder-side target/input sequence:
            [BOS] X [EOS]
        """
        output = [self.bos_token_id] + list(token_ids)
        if add_eos:
            output.append(self.eos_token_id)
        return output

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseDNATokenizer":
        raise NotImplementedError

# -------------------
# SPECIFIC TOKENIZERS
# -------------------

class CharacterDNATokenizer(BaseDNATokenizer):
    """
    Character-level DNA tokenizer.

    Default base vocabulary:
        A, C, G, T, N

    Unknown characters are mapped to [UNK]
    """

    DEFAULT_SPECIAL_TOKENS = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[BOS]",
        "[EOS]",
    ]

    DEFAULT_BASE_TOKENS = ["A", "C", "G", "T", "N"]

    def __init__(
        self,
        base_tokens: Optional[Sequence[str]] = None,
        special_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        self._base_tokens = list(base_tokens) if base_tokens is not None else list(self.DEFAULT_BASE_TOKENS)
        self._special_tokens = list(special_tokens) if special_tokens is not None else list(self.DEFAULT_SPECIAL_TOKENS)

        if len(set(self._special_tokens)) != len(self._special_tokens):
            raise ValueError("Special tokens must be unique")
        if len(set(self._base_tokens)) != len(self._base_tokens):
            raise ValueError("Base tokens must be unique")

        overlap = set(self._special_tokens) & set(self._base_tokens)
        if overlap:
            raise ValueError(f"Base tokens and special tokens overlap: {sorted(overlap)}")

        self._id_to_token: List[str] = self._special_tokens + self._base_tokens
        self._token_to_id: Dict[str, int] = {
            token: idx for idx, token in enumerate(self._id_to_token)
        }

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    @property
    def token_to_id(self) -> Dict[str, int]:
        return dict(self._token_to_id)

    @property
    def id_to_token(self) -> List[str]:
        return list(self._id_to_token)

    @property
    def base_tokens(self) -> List[str]:
        return list(self._base_tokens)

    @property
    def special_tokens(self) -> List[str]:
        return list(self._special_tokens)

    @property
    def pad_token(self) -> str:
        return "[PAD]"

    @property
    def unk_token(self) -> str:
        return "[UNK]"

    @property
    def cls_token(self) -> str:
        return "[CLS]"

    @property
    def sep_token(self) -> str:
        return "[SEP]"

    @property
    def mask_token(self) -> str:
        return "[MASK]"

    @property
    def bos_token(self) -> str:
        return "[BOS]"

    @property
    def eos_token(self) -> str:
        return "[EOS]"

    @property
    def pad_token_id(self) -> int:
        return self._token_to_id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self._token_to_id[self.unk_token]

    @property
    def cls_token_id(self) -> int:
        return self._token_to_id[self.cls_token]

    @property
    def sep_token_id(self) -> int:
        return self._token_to_id[self.sep_token]

    @property
    def mask_token_id(self) -> int:
        return self._token_to_id[self.mask_token]

    @property
    def bos_token_id(self) -> int:
        return self._token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self._token_to_id[self.eos_token]

    # remove whitespace, convert full sequence to uppercase
    def _normalize_sequence(self, sequence: str) -> str:
        return sequence.upper().strip()

    def tokenize(self, sequence: str) -> List[str]:
        """
        Character-level tokenization: one nucleotide symbol per token
        """
        normalized = self._normalize_sequence(sequence)
        return list(normalized)

    def encode(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        tokens = self.tokenize(sequence)
        token_ids = [self._token_to_id.get(token, self.unk_token_id) for token in tokens]

        if max_length is not None and len(token_ids) > max_length:
            if truncation:
                token_ids = token_ids[:max_length]
            else:
                raise ValueError(
                    f"Encoded sequence length {len(token_ids)} exceeds max_length={max_length}."
                    "Set truncation=True to truncate."
                )

        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        tokens: List[str] = []
        special_set = set(self._special_tokens)

        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self._id_to_token):
                raise ValueError(f"Invalid token id: {token_id}")

            token = self._id_to_token[token_id]
            if skip_special_tokens and token in special_set:
                continue
            tokens.append(token)

        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "tokenizer_type": "character",
            "base_tokens": self._base_tokens,
            "special_tokens": self._special_tokens,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharacterDNATokenizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        tokenizer_type = payload.get("tokenizer_type")
        if tokenizer_type != "character":
            raise ValueError(
                f"Expected tokenizer_type='character', got {tokenizer_type!r}"
            )

        return cls(
            base_tokens=payload["base_tokens"],
            special_tokens=payload["special_tokens"],
        )

class KmerDNATokenizer(BaseDNATokenizer):
    """
    General k-mer DNA tokenizer

    Supports:
        - overlapping k-mers
        - non-overlapping k-mers

    Examples:
        sequence = "ACGTAC", k=3

        overlapping=True  -> ["ACG", "CGT", "GTA", "TAC"]
        overlapping=False -> ["ACG", "TAC"]

    Notes on decoding:
        - Non-overlapping tokenization is directly decodable by concatenation
        - Overlapping tokenization is reconstructed by taking the first k-mer
          and then appending the last character of each subsequent k-mer.
          This assumes the token sequence came from this tokenizer.
    """

    DEFAULT_SPECIAL_TOKENS = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "[BOS]",
        "[EOS]",
    ]

    DEFAULT_BASE_ALPHABET = ["A", "C", "G", "T", "N"]

    def __init__(
        self,
        k: int,
        overlapping: bool = True,
        base_alphabet: Optional[Sequence[str]] = None,
        special_tokens: Optional[Sequence[str]] = None,
    ) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")

        self.k = k
        self.overlapping = overlapping
        self._base_alphabet = (
            list(base_alphabet) if base_alphabet is not None else list(self.DEFAULT_BASE_ALPHABET)
        )
        self._special_tokens = (
            list(special_tokens) if special_tokens is not None else list(self.DEFAULT_SPECIAL_TOKENS)
        )

        if len(set(self._special_tokens)) != len(self._special_tokens):
            raise ValueError("Special tokens must be unique.")
        if len(set(self._base_alphabet)) != len(self._base_alphabet):
            raise ValueError("Base alphabet tokens must be unique.")

        overlap = set(self._special_tokens) & set(self._base_alphabet)
        if overlap:
            raise ValueError(f"Base alphabet and special tokens overlap: {sorted(overlap)}")

        # Build all possible k-mers from the base alphabet.
        self._kmer_tokens = self._build_all_kmers(self._base_alphabet, self.k)

        self._id_to_token: List[str] = self._special_tokens + self._kmer_tokens
        self._token_to_id: Dict[str, int] = {
            token: idx for idx, token in enumerate(self._id_to_token)
        }

    @staticmethod
    def _build_all_kmers(alphabet: Sequence[str], k: int) -> List[str]:
        if k == 1:
            return list(alphabet)

        kmers = [""]
        for _ in range(k):
            next_kmers = []
            for prefix in kmers:
                for char in alphabet:
                    next_kmers.append(prefix + char)
            kmers = next_kmers
        return kmers

    def _normalize_sequence(self, sequence: str) -> str:
        return sequence.upper().strip()

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    @property
    def token_to_id(self) -> Dict[str, int]:
        return dict(self._token_to_id)

    @property
    def id_to_token(self) -> List[str]:
        return list(self._id_to_token)

    @property
    def base_alphabet(self) -> List[str]:
        return list(self._base_alphabet)

    @property
    def special_tokens(self) -> List[str]:
        return list(self._special_tokens)

    @property
    def kmer_tokens(self) -> List[str]:
        return list(self._kmer_tokens)

    @property
    def pad_token(self) -> str:
        return "[PAD]"

    @property
    def unk_token(self) -> str:
        return "[UNK]"

    @property
    def cls_token(self) -> str:
        return "[CLS]"

    @property
    def sep_token(self) -> str:
        return "[SEP]"

    @property
    def mask_token(self) -> str:
        return "[MASK]"

    @property
    def bos_token(self) -> str:
        return "[BOS]"

    @property
    def eos_token(self) -> str:
        return "[EOS]"

    @property
    def pad_token_id(self) -> int:
        return self._token_to_id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self._token_to_id[self.unk_token]

    @property
    def cls_token_id(self) -> int:
        return self._token_to_id[self.cls_token]

    @property
    def sep_token_id(self) -> int:
        return self._token_to_id[self.sep_token]

    @property
    def mask_token_id(self) -> int:
        return self._token_to_id[self.mask_token]

    @property
    def bos_token_id(self) -> int:
        return self._token_to_id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self._token_to_id[self.eos_token]

    def tokenize(self, sequence: str) -> List[str]:
        normalized = self._normalize_sequence(sequence)

        if len(normalized) < self.k:
            return []

        step = 1 if self.overlapping else self.k
        tokens = [
            normalized[i : i + self.k]
            for i in range(0, len(normalized) - self.k + 1, step)
        ]
        return tokens

    def encode(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        tokens = self.tokenize(sequence)
        token_ids = [self._token_to_id.get(token, self.unk_token_id) for token in tokens]

        if max_length is not None and len(token_ids) > max_length:
            if truncation:
                token_ids = token_ids[:max_length]
            else:
                raise ValueError(
                    f"Encoded sequence length {len(token_ids)} exceeds max_length={max_length}. "
                    "Set truncation=True to truncate."
                )

        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        tokens: List[str] = []
        special_set = set(self._special_tokens)

        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self._id_to_token):
                raise ValueError(f"Invalid token id: {token_id}")

            token = self._id_to_token[token_id]
            if skip_special_tokens and token in special_set:
                continue
            tokens.append(token)

        if not tokens:
            return ""

        if self.overlapping:
            # Reconstruct by overlap:
            # first k-mer + last char of each later k-mer
            sequence = tokens[0]
            for token in tokens[1:]:
                if len(token) != self.k:
                    raise ValueError(
                        "Cannot decode overlapping k-mers: encountered non-k-mer token."
                    )
                sequence += token[-1]
            return sequence

        # Non-overlapping: concatenate all k-mers directly
        return "".join(tokens)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "tokenizer_type": "kmer",
            "k": self.k,
            "overlapping": self.overlapping,
            "base_alphabet": self._base_alphabet,
            "special_tokens": self._special_tokens,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "KmerDNATokenizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        tokenizer_type = payload.get("tokenizer_type")
        if tokenizer_type != "kmer":
            raise ValueError(
                f"Expected tokenizer_type='kmer', got {tokenizer_type!r}"
            )

        return cls(
            k=payload["k"],
            overlapping=payload["overlapping"],
            base_alphabet=payload["base_alphabet"],
            special_tokens=payload["special_tokens"],
        )


class BPEDNATokenizer(BaseDNATokenizer):
    """
    Placeholder for future BPE tokenizer implementation
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("BPEDNATokenizer is not implemented yet.")

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def pad_token(self) -> str:
        raise NotImplementedError

    @property
    def unk_token(self) -> str:
        raise NotImplementedError

    @property
    def cls_token(self) -> str:
        raise NotImplementedError

    @property
    def sep_token(self) -> str:
        raise NotImplementedError

    @property
    def mask_token(self) -> str:
        raise NotImplementedError

    @property
    def bos_token(self) -> str:
        raise NotImplementedError

    @property
    def eos_token(self) -> str:
        raise NotImplementedError

    @property
    def pad_token_id(self) -> int:
        raise NotImplementedError

    @property
    def unk_token_id(self) -> int:
        raise NotImplementedError

    @property
    def cls_token_id(self) -> int:
        raise NotImplementedError

    @property
    def sep_token_id(self) -> int:
        raise NotImplementedError

    @property
    def mask_token_id(self) -> int:
        raise NotImplementedError

    @property
    def bos_token_id(self) -> int:
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int:
        raise NotImplementedError

    def tokenize(self, sequence: str) -> List[str]:
        raise NotImplementedError

    def encode(
        self,
        sequence: str,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:
        raise NotImplementedError

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "BPEDNATokenizer":
        raise NotImplementedError

__all__ = [
    "BaseDNATokenizer",
    "CharacterDNATokenizer",
    "KmerDNATokenizer",
    "BPEDNATokenizer",
]