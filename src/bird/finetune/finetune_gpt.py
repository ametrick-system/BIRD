from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from bird.models.gpt import GPTModel
from bird.models.config import GPTConfig
from bird.models.tokenizers import KmerDNATokenizer

from bird.pretrain.pretrain_utils import (
    load_checkpoint,
    load_checkpoint_state,
    save_checkpoint,
)

from bird.finetune.finetune_utils import (
    GPTForBinaryClassification,
    TaskSFTDataset,
    SFTCollator,
    ClassificationCollator,
    train_sft_epoch,
    evaluate_sft,
)

from bird.tasks.motif_recognition import MotifRecognition
from bird.tasks.mutation_sensitivity import MutationSensitivity
from bird.tasks.exon_splicing import ExonSplicing


def build_task(task_name: str, data_file: str):
    if task_name == "motif":
        return MotifRecognition(filepath=data_file)
    if task_name == "mutation":
        return MutationSensitivity(filepath=data_file)
    if task_name == "splicing":
        return ExonSplicing(filepath=data_file)
    raise ValueError(f"Unknown task: {task_name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, choices=["motif", "mutation", "splicing"])
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on {device} for task: {args.task}")

    # ----------------------------------------
    # Tokenizer + base model config
    # ----------------------------------------
    tokenizer = KmerDNATokenizer(k=args.k, overlapping=True)

    print(f"Loading pretrained config from {args.pretrained_dir}...")
    checkpoint_state = load_checkpoint_state(args.pretrained_dir, map_location="cpu")
    config = GPTConfig(**checkpoint_state["config"])
    config.validate()

    # Safety checks
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab size {tokenizer.vocab_size} does not match "
            f"checkpoint vocab size {config.vocab_size}."
        )

    if args.max_length > config.max_position_embeddings:
        raise ValueError(
            f"Requested max_length={args.max_length} exceeds checkpoint "
            f"max_position_embeddings={config.max_position_embeddings}."
        )

    base_model = GPTModel(config)

    print(f"Loading pretrained weights from {args.pretrained_dir}...")
    load_checkpoint(args.pretrained_dir, base_model)

    # ----------------------------------------
    # Task setup
    # ----------------------------------------
    task_instance = build_task(args.task, args.data_file)

    if args.task in {"motif", "mutation"}:
        task_type = "binary_classification"
        model = GPTForBinaryClassification(base_model, config)
        collator = ClassificationCollator(tokenizer, max_length=args.max_length)
    else:
        task_type = "generative"
        model = base_model
        collator = SFTCollator(tokenizer, max_length=args.max_length)

    model.to(device)

    # ----------------------------------------
    # Dataset split
    # ----------------------------------------
    full_dataset = TaskSFTDataset(task_instance)

    val_size = int(len(full_dataset) * args.val_fraction)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples:   {len(val_dataset)}")

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

    optimizer = AdamW(model.parameters(), lr=args.lr)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ----------------------------------------
    # Training loop
    # ----------------------------------------
    best_val_loss = float("inf")

    print("Evaluating pretrained baseline (epoch 0)...")

    baseline_metrics = evaluate_sft(
        model=model,
        dataloader=val_loader,
        device=device,
        epoch=0,
        task_type=task_type,
    )

    print(
        "Epoch 0 (baseline) | "
        + " | ".join(f"{k}={v:.4f}" for k, v in baseline_metrics.items())
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_sft_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            task_type=task_type,
        )

        val_metrics = evaluate_sft(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            task_type=task_type,
        )

        if task_type == "binary_classification":
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics.get('f1', float('nan')):.4f} | "
                f"val_auroc={val_metrics.get('auroc', float('nan')):.4f}"
            )
        elif task_type == "generative":
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_token_acc={val_metrics.get('token_accuracy', float('nan')):.4f} | "
                f"exact_match={val_metrics.get('exact_match_rate', float('nan')):.4f}"
            )

        epoch_dir = Path(args.output_dir) / f"epoch_{epoch:02d}"
        save_checkpoint(
            checkpoint_dir=epoch_dir,
            model=model,
            config=config,
            tokenizer=tokenizer,
            optimizer=optimizer,
            epoch=epoch,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_dir = Path(args.output_dir) / "best"
            save_checkpoint(
                checkpoint_dir=best_dir,
                model=model,
                config=config,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch,
            )
            print(f"New best checkpoint saved at epoch {epoch:02d}")

    print(f"Finished fine-tuning. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()