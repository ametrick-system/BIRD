from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

import re
import matplotlib.pyplot as plt

def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries
    """
    path = Path(path)
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records

def extract_full_sequences(records: list[dict[str, Any]]) -> list[str]:
    """
    Extract the 'full_sequence' field from generated synthetic DNA examples
    """
    sequences: list[str] = []
    for record in records:
        if "full_sequence" not in record:
            raise KeyError("Each record must contain a 'full_sequence' field.")
        sequences.append(str(record["full_sequence"]))
    return sequences


class SequenceDataset(Dataset):
    """
    Generic dataset of raw DNA sequences
    """

    def __init__(self, sequences: list[str]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


class GPTPretrainingCollator:
    """
    Collator for decoder-only causal LM pretraining

    Pipeline:
        raw sequence
            -> tokenizer.encode(sequence)
            -> tokenizer.build_gpt_inputs(...)
            -> tokenizer.pad_token_ids(...)
            -> tensors

    labels are just input_ids.clone(); GPTModel handles shifting internally
    """

    def __init__(
        self,
        tokenizer: Any,
        max_length: Optional[int] = None,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos

    def __call__(self, batch: list[str]) -> dict[str, torch.Tensor]:
        tokenized: list[list[int]] = []

        for sequence in batch:
            token_ids = self.tokenizer.encode(
                sequence,
                max_length=None,
                truncation=False,
            )
            token_ids = self.tokenizer.build_gpt_inputs(
                token_ids,
                add_bos=self.add_bos,
                add_eos=self.add_eos,
            )
            tokenized.append(token_ids)

        padded = self.tokenizer.pad_token_ids(
            tokenized,
            max_length=self.max_length,
            padding=True,
            truncation=self.max_length is not None,
            return_attention_mask=True,
        )

        input_ids = torch.tensor(padded["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(padded["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@torch.no_grad()
def greedy_decode_gpt(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 20,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Greedy decoding for decoder-only GPT

    Args:
        model: GPT model
        input_ids: shape (batch_size, seq_len)
        attention_mask: optional shape (batch_size, seq_len)
        max_new_tokens: number of tokens to generate
        eos_token_id: optional early-stop token id

    Returns:
        generated_ids: shape (batch_size, seq_len + generated_len)
    """
    model.eval()

    generated_ids = input_ids.clone()

    if attention_mask is None:
        current_attention_mask = torch.ones_like(generated_ids, dtype=torch.long)
    else:
        current_attention_mask = attention_mask.clone()

    finished = torch.zeros(
        generated_ids.size(0),
        dtype=torch.bool,
        device=generated_ids.device,
    )

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=current_attention_mask,
        )
        logits = outputs["logits"]                # (B, T, V)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)  # (B,)

        if eos_token_id is not None:
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            finished = finished | (next_token == eos_token_id)

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=1)

        next_attention = torch.ones(
            (generated_ids.size(0), 1),
            dtype=current_attention_mask.dtype,
            device=current_attention_mask.device,
        )
        current_attention_mask = torch.cat(
            [current_attention_mask, next_attention],
            dim=1,
        )

        if eos_token_id is not None and finished.all():
            break

    return generated_ids

@torch.no_grad()
def top_p_decode_gpt(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 20,
    eos_token_id: Optional[int] = None,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling decoding for GPT

    Args:
        model: GPT model
        input_ids: (batch_size, seq_len)
        attention_mask: optional (batch_size, seq_len)
        max_new_tokens: number of tokens to generate
        eos_token_id: optional early stop token
        top_p: cumulative probability threshold
        temperature: softmax temperature

    Returns:
        generated_ids: (batch_size, seq_len + generated_len)
    """
    model.eval()

    generated_ids = input_ids.clone()

    if attention_mask is None:
        current_attention_mask = torch.ones_like(generated_ids, dtype=torch.long)
    else:
        current_attention_mask = attention_mask.clone()

    finished = torch.zeros(
        generated_ids.size(0),
        dtype=torch.bool,
        device=generated_ids.device,
    )

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=generated_ids,
            attention_mask=current_attention_mask,
        )
        logits = outputs["logits"][:, -1, :]  # (B, V)

        # Apply temperature
        logits = logits / temperature

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)  # (B, V)

        # Sort probs descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Compute cumulative probs
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Create mask for top-p
        sorted_mask = cumulative_probs > top_p

        # Shift mask right to keep first token above threshold
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False

        # Zero out tokens outside nucleus
        sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)

        # Renormalize
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        # Sample
        next_token_idx = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1)

        # Map back to original vocab indices
        next_token = torch.gather(sorted_indices, -1, next_token_idx).squeeze(-1)  # (B,)

        # Handle EOS
        if eos_token_id is not None:
            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_token_id),
                next_token,
            )
            finished = finished | (next_token == eos_token_id)

        # Append token
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=1)

        # Update attention mask
        next_attention = torch.ones(
            (generated_ids.size(0), 1),
            dtype=current_attention_mask.dtype,
            device=current_attention_mask.device,
        )
        current_attention_mask = torch.cat(
            [current_attention_mask, next_attention],
            dim=1,
        )

        if eos_token_id is not None and finished.all():
            break

    return generated_ids


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    config: Any,
    tokenizer: Any,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    extra_state: Optional[dict[str, Any]] = None,
) -> None:
    """
    Save model checkpoint + tokenizer.

    Saved files:
        checkpoint_dir/
            model.pt
            tokenizer.json
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__ if hasattr(config, "__dict__") else config,
        "epoch": epoch,
        "step": step,
        "extra_state": extra_state or {},
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, checkpoint_dir / "model.pt")
    tokenizer.save(checkpoint_dir / "tokenizer.json")


def load_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load model checkpoint into an existing model (and optional optimizer).

    Returns the full saved checkpoint dict.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir / "model.pt"

    state = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state

def load_checkpoint_state(
    checkpoint_dir,
    map_location="cpu",
):
    checkpoint_path = Path(checkpoint_dir) / "model.pt"
    state = torch.load(checkpoint_path, map_location=map_location)
    return state

def estimate_perplexity(loss: float) -> float:
    """
    Convert average cross-entropy loss to perplexity
    """
    return float(torch.exp(torch.tensor(loss)).item())

def plot_pretraining_loss_from_log(
    log_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, list[float]]:
    """
    Parse a GPT pretraining log and plot train/val loss over epochs.

    Expected log line:
        Epoch 01/10 | train_loss=... | train_ppl=... | val_loss=... | val_ppl=...
    """
    log_path = Path(log_path)

    if output_path is None:
        output_path = log_path.parent / "pretrain_loss.png"
    else:
        output_path = Path(output_path)

    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
        r"train_loss=([0-9]*\.?[0-9]+)\s+\|\s+"
        r"train_ppl=([0-9]*\.?[0-9]+)\s+\|\s+"
        r"val_loss=([0-9]*\.?[0-9]+)\s+\|\s+"
        r"val_ppl=([0-9]*\.?[0-9]+)"
    )

    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_ppls: list[float] = []
    val_ppls: list[float] = []

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(3)))
                train_ppls.append(float(match.group(4)))
                val_losses.append(float(match.group(5)))
                val_ppls.append(float(match.group(6)))

    if not epochs:
        raise ValueError(f"No epoch loss lines found in log: {log_path}")

    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="train loss")
    plt.plot(epochs, val_losses, marker="o", label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("GPT Pretraining Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return {
        "epochs": epochs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_ppl": train_ppls,
        "val_ppl": val_ppls,
    }