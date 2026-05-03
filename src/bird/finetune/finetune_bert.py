from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm.auto import tqdm

from bird.models.bert import BERTModel
from bird.models.config import BERTConfig
from bird.models.tokenizers import KmerDNATokenizer

from bird.pretrain.pretrain_utils import (
    load_checkpoint,
    load_checkpoint_state,
    save_checkpoint,
)

from bird.finetune.finetune_utils import (
    BERTForBinaryClassification,
    BERTForTokenClassification,
    ClassificationCollator,
    TokenClassificationCollator,
    binary_classification_metrics,
)

from bird.tasks.motif_recognition import MotifRecognition
from bird.tasks.mutation_sensitivity import MutationSensitivity


class SplicingTokenDataset(Dataset):
    """
    Builds token-level exon/intron labels directly from the raw JSONL data.

    Output example format:
        {
            "sequence": <full DNA sequence>,
            "token_labels": [0, 1, 1, 0, ...]
        }

    Token labels are aligned to tokenizer.encode(sequence), assuming
    overlapping k-mer tokenization. A token is labeled by the region
    containing its start position.
    """

    def __init__(self, filepath: str, k: int) -> None:
        self.examples: list[dict[str, Any]] = []
        self.k = k

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                sequence = record["full_sequence"]
                structure = record["structure"]

                base_labels = self._build_base_labels(sequence, structure)
                token_labels = self._build_token_labels(base_labels)

                expected_token_len = len(sequence) - self.k + 1
                if expected_token_len < 0:
                    raise ValueError(
                        f"Sequence length {len(sequence)} is shorter than k={self.k}."
                    )
                if len(token_labels) != expected_token_len:
                    raise ValueError(
                        f"Token label length {len(token_labels)} does not match "
                        f"expected {expected_token_len} for sequence length {len(sequence)} and k={self.k}."
                    )

                self.examples.append(
                    {
                        "sequence": sequence,
                        "token_labels": token_labels,
                    }
                )

    def _build_base_labels(self, sequence: str, structure: dict[str, Any]) -> list[int]:
        """
        Build per-base labels: 1 for exon, 0 for everything else.
        """
        base_labels = [0] * len(sequence)
        cursor = 0

        # Walk through the sequence in the same order it was assembled.
        cursor += len(structure["utr_1"])
        cursor += len(structure["enhancer"])
        cursor += len(structure["utr_2"])
        cursor += len(structure["promoter"])
        cursor += len(structure["utr_3"])

        for segment in structure["orf_region"]:
            seg_seq = segment["sequence"]
            seg_len = len(seg_seq)

            if segment["type"] == "exon":
                for i in range(cursor, cursor + seg_len):
                    base_labels[i] = 1

            cursor += seg_len

        if cursor != len(sequence):
            raise ValueError(
                f"Parsed structure length {cursor} does not match sequence length {len(sequence)}."
            )

        return base_labels

    def _build_token_labels(self, base_labels: list[int]) -> list[int]:
        """
        For overlapping k-mers, label token i using the label at the token's start position.
        """
        return [base_labels[i] for i in range(len(base_labels) - self.k + 1)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def build_dataset(task_name: str, data_file: str, k: int) -> tuple[Dataset, str]:
    """
    Returns:
        dataset, task_type
    where task_type is one of:
        - "binary_classification"
        - "token_classification"
    """
    if task_name == "motif":
        return MotifRecognition(filepath=data_file), "binary_classification"
    if task_name == "mutation":
        return MutationSensitivity(filepath=data_file), "binary_classification"
    if task_name == "splicing":
        return SplicingTokenDataset(filepath=data_file, k=k), "token_classification"

    raise ValueError(f"Unknown task: {task_name}")


@torch.no_grad()
def binary_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.sigmoid(logits.view(-1))
    preds = (probs >= 0.5).float()
    gold = labels.view(-1).float()
    return (preds == gold).float().mean().item()


@torch.no_grad()
def token_accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    logits: (B, T, 1)
    labels: (B, T) with values in {0,1} or -100 for ignored positions
    """
    active_mask = labels != -100
    if not active_mask.any():
        return 0.0

    probs = torch.sigmoid(logits.squeeze(-1))
    preds = (probs >= 0.5).long()
    gold = labels.long()

    return (preds[active_mask] == gold[active_mask]).float().mean().item()


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    task_type: str,
) -> dict[str, float]:
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
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
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

    return {"loss": total_loss / max(total_batches, 1)}

@torch.no_grad()
def token_classification_metrics(logits, labels):
    if logits.ndim == 3:
        logits = logits.squeeze(-1)

    active_mask = labels != -100
    if not active_mask.any():
        return {
            "token_accuracy": 0.0,
            "majority_baseline_accuracy": 0.0,
            "exon_precision": 0.0,
            "exon_recall": 0.0,
            "exon_f1": 0.0,
            "non_exon_accuracy": 0.0,
            "exon_accuracy": 0.0,
        }

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    gold = labels.long()

    preds = preds[active_mask]
    gold = gold[active_mask]

    token_accuracy = (preds == gold).float().mean().item()

    num_exon = (gold == 1).sum().item()
    num_non_exon = (gold == 0).sum().item()
    total = len(gold)

    majority_baseline_accuracy = max(num_exon, num_non_exon) / total

    tp = ((preds == 1) & (gold == 1)).sum().item()
    fp = ((preds == 1) & (gold == 0)).sum().item()
    fn = ((preds == 0) & (gold == 1)).sum().item()
    tn = ((preds == 0) & (gold == 0)).sum().item()

    exon_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    exon_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    exon_f1 = (
        2 * exon_precision * exon_recall / (exon_precision + exon_recall)
        if (exon_precision + exon_recall) > 0
        else 0.0
    )

    non_exon_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    exon_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "token_accuracy": token_accuracy,
        "majority_baseline_accuracy": majority_baseline_accuracy,
        "exon_precision": exon_precision,
        "exon_recall": exon_recall,
        "exon_f1": exon_f1,
        "non_exon_accuracy": non_exon_accuracy,
        "exon_accuracy": exon_accuracy,
    }

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    epoch: int,
    task_type: str,
):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    # For binary classification
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_majority = 0.0
    total_pos_rate = 0.0
    total_classification_batches = 0

    # For token classification
    total_token_metrics = {
        "token_accuracy": 0.0,
        "majority_baseline_accuracy": 0.0,
        "exon_precision": 0.0,
        "exon_recall": 0.0,
        "exon_f1": 0.0,
        "non_exon_accuracy": 0.0,
        "exon_accuracy": 0.0,
    }
    total_token_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [val]",
        leave=False,
    )

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

        loss = outputs["loss"]
        assert loss is not None

        total_loss += loss.item()
        total_batches += 1
        avg_loss = total_loss / total_batches

        if task_type == "binary_classification":
            batch_metrics = binary_classification_metrics(outputs["logits"], labels)

            total_accuracy += batch_metrics["accuracy"]
            total_precision += batch_metrics["precision"]
            total_recall += batch_metrics["recall"]
            total_f1 += batch_metrics["f1"]
            total_majority += batch_metrics["majority_baseline_accuracy"]
            total_pos_rate += batch_metrics["positive_rate"]
            total_classification_batches += 1

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{batch_metrics['accuracy']:.4f}",
                f1=f"{batch_metrics['f1']:.4f}",
            )

        elif task_type == "token_classification":
            batch_metrics = token_classification_metrics(outputs["logits"], labels)

            for k in total_token_metrics:
                total_token_metrics[k] += batch_metrics[k]

            total_token_batches += 1

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                tok_acc=f"{batch_metrics['token_accuracy']:.4f}",
                exon_f1=f"{batch_metrics['exon_f1']:.4f}",
            )

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    metrics = {"loss": total_loss / max(total_batches, 1)}

    if task_type == "binary_classification" and total_classification_batches > 0:
        metrics["accuracy"] = total_accuracy / total_classification_batches
        metrics["precision"] = total_precision / total_classification_batches
        metrics["recall"] = total_recall / total_classification_batches
        metrics["f1"] = total_f1 / total_classification_batches
        metrics["majority_baseline_accuracy"] = total_majority / total_classification_batches
        metrics["positive_rate"] = total_pos_rate / total_classification_batches

    elif task_type == "token_classification" and total_token_batches > 0:
        for k in total_token_metrics:
            metrics[k] = total_token_metrics[k] / total_token_batches

    return metrics

def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT on DNA tasks.")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["motif", "mutation", "splicing"],
    )
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning BERT on {device} for task: {args.task}")

    tokenizer = KmerDNATokenizer(k=args.k, overlapping=True)

    print(f"Loading pretrained config from {args.pretrained_dir}...")
    checkpoint_state = load_checkpoint_state(args.pretrained_dir, map_location="cpu")
    config = BERTConfig(**checkpoint_state["config"])
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

    base_model = BERTModel(config)

    print(f"Loading pretrained weights from {args.pretrained_dir}...")
    load_checkpoint(args.pretrained_dir, base_model)

    dataset, task_type = build_dataset(args.task, args.data_file, args.k)

    if task_type == "binary_classification":
        model = BERTForBinaryClassification(base_model, config)
        collator = ClassificationCollator(tokenizer, max_length=args.max_length)

    elif task_type == "token_classification":
        model = BERTForTokenClassification(base_model, config)
        collator = TokenClassificationCollator(tokenizer, max_length=args.max_length)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    model.to(device)

    val_size = int(len(dataset) * args.val_fraction)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
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

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            task_type=task_type,
        )

        val_metrics = evaluate(
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
                f"f1={val_metrics['f1']:.4f} | "
                f"prec={val_metrics['precision']:.4f} | "
                f"rec={val_metrics['recall']:.4f} | "
                f"maj_acc={val_metrics['majority_baseline_accuracy']:.4f} | "
                f"pos_rate={val_metrics['positive_rate']:.4f}"
            )
        elif task_type == "token_classification":
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_token_acc={val_metrics.get('token_accuracy', float('nan')):.4f} | "
                f"majority_acc={val_metrics.get('majority_baseline_accuracy', float('nan')):.4f} | "
                f"exon_prec={val_metrics.get('exon_precision', float('nan')):.4f} | "
                f"exon_rec={val_metrics.get('exon_recall', float('nan')):.4f} | "
                f"exon_f1={val_metrics.get('exon_f1', float('nan')):.4f} | "
                f"non_exon_acc={val_metrics.get('non_exon_accuracy', float('nan')):.4f} | "
                f"exon_acc={val_metrics.get('exon_accuracy', float('nan')):.4f}"
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

    print(f"Finished fine-tuning BERT. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()