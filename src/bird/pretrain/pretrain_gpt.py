from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from bird.models.config import GPTConfig
from bird.models.gpt import GPTModel
from bird.models.tokenizers import KmerDNATokenizer
from bird.pretrain.pretrain_utils import (
    GPTPretrainingCollator,
    SequenceDataset,
    estimate_perplexity,
    extract_full_sequences,
    load_jsonl,
    save_checkpoint,
)

def train_one_epoch(
    model: GPTModel,
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
    model: GPTModel,
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
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

    collator = GPTPretrainingCollator(
        tokenizer=tokenizer,
        max_length=args.max_length,
        add_bos=True,
        add_eos=True,
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

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
        padding_idx=tokenizer.pad_token_id,
        tie_word_embeddings=args.tie_word_embeddings,
    )
    config.validate()

    model = GPTModel(config).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Starting training...")
    print(
        f"k={args.k} | overlapping={args.overlapping} | "
        f"batch_size={args.batch_size} | epochs={args.epochs}"
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
                },
            )
            print(f"New best checkpoint saved at epoch {epoch:02d}")

    print("Training complete.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain GPT on synthetic DNA data.")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to JSONL synthetic data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints.",
    )

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

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument(
        "--tie_word_embeddings",
        action="store_true",
        help="Tie LM head weights to input token embeddings.",
    )

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main(args)