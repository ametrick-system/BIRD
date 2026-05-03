from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from bird.models.bart import BARTModel
from bird.models.config import BARTConfig
from bird.models.tokenizers import KmerDNATokenizer
from bird.pretrain.pretrain_utils import (
    SequenceDataset,
    estimate_perplexity,
    extract_full_sequences,
    load_jsonl,
    save_checkpoint,
)

class BARTDenoisingCollator:
    """
    Collator for BART-style denoising pretraining.

    For each raw DNA sequence:
      1. tokenize clean sequence
      2. corrupt encoder input by replacing some tokens with [MASK]
      3. decoder input is BOS + clean target prefix
      4. labels are clean target tokens, padded with -100
    """

    def __init__(
        self,
        tokenizer,
        max_length: Optional[int] = None,
        mask_probability: float = 0.15,
        random_seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.rng = random.Random(random_seed)

        self.special_token_ids = {
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.mask_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        }

    def _corrupt_tokens(self, token_ids: list[int]) -> list[int]:
        """
        Simple BART-style corruption:
        randomly replace eligible tokens with [MASK].
        """
        corrupted = list(token_ids)

        for i, token_id in enumerate(corrupted):
            if token_id in self.special_token_ids:
                continue
            if self.rng.random() < self.mask_probability:
                corrupted[i] = self.tokenizer.mask_token_id

        return corrupted

    def __call__(self, batch: list[str]) -> dict[str, torch.Tensor]:
        encoder_sequences: list[list[int]] = []
        decoder_sequences: list[list[int]] = []
        label_sequences: list[list[int]] = []

        for sequence in batch:
            clean_token_ids = self.tokenizer.encode(
                sequence,
                max_length=None,
                truncation=False,
            )

            # Target sequence: clean tokens + EOS
            target_ids = list(clean_token_ids)
            if self.tokenizer.eos_token_id is not None:
                target_ids = target_ids + [self.tokenizer.eos_token_id]

            # Encoder gets corrupted clean sequence + EOS
            encoder_ids = self._corrupt_tokens(clean_token_ids)
            if self.tokenizer.eos_token_id is not None:
                encoder_ids = encoder_ids + [self.tokenizer.eos_token_id]

            # Decoder input gets BOS + clean target prefix
            decoder_ids = list(target_ids)
            if self.tokenizer.bos_token_id is not None:
                decoder_ids = [self.tokenizer.bos_token_id] + decoder_ids[:-1]

            encoder_sequences.append(encoder_ids)
            decoder_sequences.append(decoder_ids)
            label_sequences.append(target_ids)

        padded_encoder = self.tokenizer.pad_token_ids(
            encoder_sequences,
            max_length=self.max_length,
            padding=True,
            truncation=self.max_length is not None,
            return_attention_mask=True,
        )
        padded_decoder = self.tokenizer.pad_token_ids(
            decoder_sequences,
            max_length=self.max_length,
            padding=True,
            truncation=self.max_length is not None,
            return_attention_mask=True,
        )
        padded_labels = self.tokenizer.pad_token_ids(
            label_sequences,
            max_length=self.max_length,
            padding=True,
            truncation=self.max_length is not None,
            return_attention_mask=False,
            pad_value=-100,
        )

        encoder_input_ids = torch.tensor(padded_encoder["input_ids"], dtype=torch.long)
        encoder_attention_mask = torch.tensor(
            padded_encoder["attention_mask"], dtype=torch.long
        )

        decoder_input_ids = torch.tensor(padded_decoder["input_ids"], dtype=torch.long)
        decoder_attention_mask = torch.tensor(
            padded_decoder["attention_mask"], dtype=torch.long
        )

        labels = torch.tensor(padded_labels["input_ids"], dtype=torch.long)

        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }

def train_one_epoch(
    model: BARTModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [train]",
        leave=False,
    )

    for batch in progress_bar:
        encoder_input_ids = batch["encoder_input_ids"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]
        assert loss is not None

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        avg_loss = total_loss / total_batches
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / max(total_batches, 1)

@torch.no_grad()
def evaluate(
    model: BARTModel,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [val]",
        leave=False,
    )

    for batch in progress_bar:
        encoder_input_ids = batch["encoder_input_ids"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        loss = outputs["loss"]
        assert loss is not None

        total_loss += loss.item()
        total_batches += 1
        avg_loss = total_loss / total_batches
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / max(total_batches, 1)

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = load_jsonl(args.data_path)
    sequences = extract_full_sequences(records)
    print(f"Loaded {len(sequences)} sequences.")

    dataset = SequenceDataset(sequences)

    val_size = int(len(dataset) * args.val_fraction)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences:   {len(val_dataset)}")

    tokenizer = KmerDNATokenizer(
        k=args.k,
        overlapping=args.overlapping,
    )

    collator = BARTDenoisingCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        mask_probability=args.mask_probability,
        random_seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    config = BARTConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_encoder_layers,  # unused by BART construction but required by base config
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        padding_idx=tokenizer.pad_token_id,
        tie_word_embeddings=args.tie_word_embeddings,
    )
    config.validate()

    model = BARTModel(config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Starting BART pretraining...")
    print(
        f"k={args.k} | overlapping={args.overlapping} | "
        f"batch_size={args.batch_size} | epochs={args.epochs} | "
        f"mask_probability={args.mask_probability}"
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device, epoch)

        train_ppl = estimate_perplexity(train_loss)
        val_ppl = estimate_perplexity(val_loss)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_loss:.4f} | train_ppl={train_ppl:.4f} | "
            f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.4f}"
        )

        epoch_ckpt_dir = Path(args.output_dir) / f"epoch_{epoch:02d}"
        save_checkpoint(
            checkpoint_dir=epoch_ckpt_dir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            optimizer=optimizer,
            epoch=epoch,
            step=None,
            extra_state={
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "k": args.k,
                "overlapping": args.overlapping,
                "mask_probability": args.mask_probability,
            },
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_dir = Path(args.output_dir) / "best"
            save_checkpoint(
                checkpoint_dir=best_ckpt_dir,
                model=model,
                config=config,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch,
                step=None,
                extra_state={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_ppl": train_ppl,
                    "val_ppl": val_ppl,
                    "k": args.k,
                    "overlapping": args.overlapping,
                    "mask_probability": args.mask_probability,
                },
            )
            print(f"New best checkpoint saved at epoch {epoch:02d}")

    print("Training complete.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain BART on synthetic DNA data.")

    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL synthetic data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--overlapping",
        dest="overlapping",
        action="store_true",
        help="Use overlapping k-mers.",
    )
    parser.add_argument(
        "--non_overlapping",
        dest="overlapping",
        action="store_false",
        help="Use non-overlapping k-mers.",
    )
    parser.set_defaults(overlapping=True)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--mask_probability", type=float, default=0.15)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--tie_word_embeddings",
        action="store_true",
        help="Tie LM head weights to shared token embeddings.",
    )

    return parser

if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)