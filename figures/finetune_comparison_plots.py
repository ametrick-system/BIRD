#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from bird.models.gpt import GPTModel
from bird.models.bert import BERTModel
from bird.models.bart import BARTModel
from bird.models.config import GPTConfig, BERTConfig, BARTConfig
from bird.models.tokenizers import KmerDNATokenizer

from bird.pretrain.pretrain_utils import load_checkpoint, load_checkpoint_state

from bird.finetune.finetune_utils import (
    GPTForBinaryClassification,
    BERTForBinaryClassification,
    ClassificationCollator,
    TaskSFTDataset,
)

from bird.finetune.finetune_bart import BARTForBinaryClassification

from bird.tasks.motif_recognition import MotifRecognition
from bird.tasks.mutation_sensitivity import MutationSensitivity


TASKS = ["motif", "mutation", "splicing"]
BINARY_TASKS = ["motif", "mutation"]
MODELS = ["GPT", "BERT", "BART"]

EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)/(\d+)\s+\|")

METRIC_PATTERNS = {
    "loss": re.compile(r"val_loss=([0-9]*\.?[0-9]+)"),
    "accuracy": re.compile(r"(?:val_acc|val_token_acc)=([0-9]*\.?[0-9]+)"),
    "auroc": re.compile(r"(?:val_auroc|auroc)=([0-9]*\.?[0-9]+)"),
}


def parse_finetune_log(log_path: str | Path) -> dict[str, list[float]]:
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    metrics = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "auroc": [],
    }

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            epoch_match = EPOCH_PATTERN.search(line)
            if not epoch_match:
                continue

            epoch = int(epoch_match.group(1))
            row = {}

            for metric_name, pattern in METRIC_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    row[metric_name] = float(match.group(1))

            if not row:
                continue

            metrics["epoch"].append(epoch)
            for metric_name in ["loss", "accuracy", "auroc"]:
                metrics[metric_name].append(row.get(metric_name, float("nan")))

    if not metrics["epoch"]:
        raise ValueError(f"No epoch metric lines found in log: {log_path}")

    return metrics


def get_log_path(project_root: Path, model_name: str, task: str) -> Path:
    return project_root / model_name / "logs" / f"finetune_{task}.log"


def plot_metric_over_epochs(
    logs: dict[str, Path],
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    all_epochs = set()

    for model_name, log_path in logs.items():
        parsed = parse_finetune_log(log_path)

        epochs = []
        values = []
        for e, v in zip(parsed["epoch"], parsed[metric]):
            if not np.isnan(v):
                epochs.append(e)
                values.append(v)

        if not epochs:
            print(f"Skipping {model_name}: no {metric} values in {log_path}")
            continue

        all_epochs.update(epochs)
        plt.plot(epochs, values, marker="o", label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)

    if all_epochs:
        plt.xticks(sorted(all_epochs))

    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def build_binary_task(task: str, data_file: Path):
    if task == "motif":
        return MotifRecognition(filepath=str(data_file))
    if task == "mutation":
        return MutationSensitivity(filepath=str(data_file))
    raise ValueError(f"ROC only supported for binary tasks, got: {task}")


def load_binary_model(
    model_name: str,
    checkpoint_dir: Path,
    tokenizer: KmerDNATokenizer,
    device: torch.device,
):
    state = load_checkpoint_state(checkpoint_dir, map_location="cpu")

    if model_name == "GPT":
        config = GPTConfig(**state["config"])
        base = GPTModel(config)
        model = GPTForBinaryClassification(base, config)

    elif model_name == "BERT":
        config = BERTConfig(**state["config"])
        base = BERTModel(config)
        model = BERTForBinaryClassification(base, config)

    elif model_name == "BART":
        config = BARTConfig(**state["config"])
        base = BARTModel(config)
        model = BARTForBinaryClassification(base, config)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"{model_name}: tokenizer vocab size {tokenizer.vocab_size} "
            f"does not match checkpoint vocab size {config.vocab_size}"
        )

    load_checkpoint(checkpoint_dir, model)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_binary_scores(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_scores = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )

        scores = torch.sigmoid(outputs["logits"].view(-1))

        all_scores.append(scores.detach().cpu())
        all_labels.append(labels.view(-1).detach().cpu())

    y_score = torch.cat(all_scores).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)

    return y_true, y_score


def roc_curve_numpy(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    y_true = y_true.astype(int)

    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC requires both positive and negative labels.")

    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    positives = np.sum(y_true_sorted == 1)
    negatives = np.sum(y_true_sorted == 0)

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tpr = tps / positives
    fpr = fps / negatives

    # Add origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def plot_best_checkpoint_roc(
    project_root: Path,
    task: str,
    data_file: Path,
    output_path: Path,
    batch_size: int,
    k: int,
    max_length: int,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = KmerDNATokenizer(k=k, overlapping=True)
    task_instance = build_binary_task(task, data_file)
    dataset = TaskSFTDataset(task_instance)

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    collator = ClassificationCollator(tokenizer, max_length=max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    plt.figure(figsize=(7, 6))

    for model_name in MODELS:
        checkpoint_dir = (
            project_root
            / model_name
            / "finetuned"
            / task
            / "best"
        )

        if not checkpoint_dir.exists():
            print(f"Skipping {model_name}: missing checkpoint {checkpoint_dir}")
            continue

        model = load_binary_model(
            model_name=model_name,
            checkpoint_dir=checkpoint_dir,
            tokenizer=tokenizer,
            device=device,
        )

        y_true, y_score = collect_binary_scores(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        fpr, tpr, auc = roc_curve_numpy(y_true, y_score)

        plt.plot(fpr, tpr, label=f"{model_name} AUROC={auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{task.capitalize()} ROC Curve: Best Checkpoints")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make BIRD finetuning comparison plots."
    )

    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument("--num_seq", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    project_root = Path(args.project_root)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else project_root / "figures" / "finetune"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file = (
        project_root
        / "src"
        / "bird"
        / "data"
        / f"dna_{args.num_seq}_seq.jsonl"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Accuracy over epochs for all tasks
    for task in TASKS:
        logs = {
            model: get_log_path(project_root, model, task)
            for model in MODELS
        }

        plot_metric_over_epochs(
            logs=logs,
            metric="accuracy",
            output_path=output_dir / f"{task}_accuracy_over_epochs.png",
            title=f"{task.capitalize()} Accuracy Over Epochs",
            ylabel="Validation Accuracy",
            ylim=(0.0, 1.0),
        )

    # 2. AUROC over epochs for binary tasks
    for task in BINARY_TASKS:
        logs = {
            model: get_log_path(project_root, model, task)
            for model in MODELS
        }

        plot_metric_over_epochs(
            logs=logs,
            metric="auroc",
            output_path=output_dir / f"{task}_auroc_over_epochs.png",
            title=f"{task.capitalize()} AUROC Over Epochs",
            ylabel="Validation AUROC",
            ylim=(0.0, 1.0),
        )

    # 3. Loss over epochs for all tasks
    for task in TASKS:
        logs = {
            model: get_log_path(project_root, model, task)
            for model in MODELS
        }

        plot_metric_over_epochs(
            logs=logs,
            metric="loss",
            output_path=output_dir / f"{task}_loss_over_epochs.png",
            title=f"{task.capitalize()} Loss Over Epochs",
            ylabel="Validation Loss",
            ylim=None,
        )

    # 4. ROC curves from best checkpoints for binary tasks
    for task in BINARY_TASKS:
        plot_best_checkpoint_roc(
            project_root=project_root,
            task=task,
            data_file=data_file,
            output_path=output_dir / f"{task}_best_checkpoint_roc.png",
            batch_size=args.batch_size,
            k=args.k,
            max_length=args.max_length,
            val_fraction=args.val_fraction,
            seed=args.seed,
            device=device,
        )

    print(f"All figures saved to: {output_dir}")

if __name__ == "__main__":
    main()