from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from bird.models.bart import BARTModel
from bird.models.config import BARTConfig
from bird.models.tokenizers import KmerDNATokenizer

from bird.pretrain.pretrain_utils import (
    load_checkpoint,
    load_checkpoint_state,
    save_checkpoint,
)

from bird.finetune.finetune_utils import (
    TaskSFTDataset,
    ClassificationCollator,
    generative_token_metrics,
    binary_classification_metrics,
    binary_auroc_from_logits,
)

from bird.tasks.motif_recognition import MotifRecognition
from bird.tasks.mutation_sensitivity import MutationSensitivity
from bird.tasks.exon_splicing import ExonSplicing


class BARTForBinaryClassification(nn.Module):
    """
    Wrap a pretrained BART backbone for binary sequence classification.

    Strategy:
        - encode input sequence
        - take hidden state of last non-padding encoder token
        - pass through linear classifier
    """

    def __init__(self, base_bart_model, config):
        super().__init__()
        self.bart = base_bart_model
        self.classifier = nn.Linear(config.d_model, 1)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        token_type_ids=None,   # ignored, kept for compatibility
    ):
        encoder_hidden_states = self.bart.encode(
            encoder_input_ids=input_ids,
            encoder_attention_mask=attention_mask,
        )

        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        sequence_lengths = sequence_lengths.clamp(min=0)

        pooled_states = encoder_hidden_states[
            torch.arange(batch_size, device=input_ids.device),
            sequence_lengths,
        ]

        logits = self.classifier(pooled_states)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return {"loss": loss, "logits": logits}


class BARTGenerativeCollator:
    """
    Collator for encoder-decoder generative finetuning.

    Expects task examples shaped like:
        {
            "messages": [
                {"role": "user", "content": ...},
                {"role": "assistant", "content": ...},
            ]
        }

    Encoder input: tokenized user prompt + EOS
    Decoder input: BOS + assistant target prefix
    Labels: assistant target + EOS, padded with -100
    """

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def __call__(self, batch):
        encoder_sequences = []
        decoder_sequences = []
        label_sequences = []

        for item in batch:
            user_text = item["messages"][0]["content"]
            asst_text = item["messages"][1]["content"]

            encoder_ids = self.tokenizer.encode(user_text)
            target_ids = self.tokenizer.encode(asst_text)

            # encoder input ends with EOS
            encoder_ids = list(encoder_ids) + [self.eos_token_id]

            # labels = target + EOS
            labels = list(target_ids) + [self.eos_token_id]

            # decoder input = BOS + target prefix
            if self.bos_token_id is None:
                raise ValueError("Tokenizer must define bos_token_id for BART finetuning.")
            decoder_ids = [self.bos_token_id] + labels[:-1]

            if len(encoder_ids) > self.max_length:
                encoder_ids = encoder_ids[:self.max_length]
            if len(decoder_ids) > self.max_length:
                decoder_ids = decoder_ids[:self.max_length]
                labels = labels[:self.max_length]

            encoder_sequences.append(encoder_ids)
            decoder_sequences.append(decoder_ids)
            label_sequences.append(labels)

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

        return {
            "input_ids": torch.tensor(padded_encoder["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(padded_encoder["attention_mask"], dtype=torch.long),
            "decoder_input_ids": torch.tensor(padded_decoder["input_ids"], dtype=torch.long),
            "decoder_attention_mask": torch.tensor(padded_decoder["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(padded_labels["input_ids"], dtype=torch.long),
        }


def build_task(task_name: str, data_file: str):
    if task_name == "motif":
        return MotifRecognition(filepath=data_file)
    if task_name == "mutation":
        return MutationSensitivity(filepath=data_file)
    if task_name == "splicing":
        return ExonSplicing(filepath=data_file)
    raise ValueError(f"Unknown task: {task_name}")


def train_bart_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch: int,
    task_type: str,
):
    model.train()
    total_loss = 0.0
    total_batches = 0

    from tqdm.auto import tqdm
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [train]", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        if task_type == "binary_classification":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        elif task_type == "generative":
            encoder_input_ids = batch["input_ids"].to(device)
            encoder_attention_mask = batch["attention_mask"].to(device)
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

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        loss = outputs["loss"]
        assert loss is not None

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        progress_bar.set_postfix(loss=f"{total_loss / total_batches:.4f}")

    return {"loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def evaluate_bart(
    model,
    dataloader,
    device,
    epoch: int,
    task_type: str,
):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    # Binary classification
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_majority = 0.0
    total_pos_rate = 0.0
    total_metric_batches = 0
    all_binary_logits = []
    all_binary_labels = []

    # Generative
    total_gen_token_accuracy = 0.0
    total_gen_exact_match_rate = 0.0
    total_gen_metric_batches = 0

    from tqdm.auto import tqdm
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [val]", leave=False)

    for batch in progress_bar:
        if task_type == "binary_classification":
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            batch_metrics = binary_classification_metrics(outputs["logits"], labels)

            total_accuracy += batch_metrics["accuracy"]
            total_precision += batch_metrics["precision"]
            total_recall += batch_metrics["recall"]
            total_f1 += batch_metrics["f1"]
            total_majority += batch_metrics["majority_baseline_accuracy"]
            total_pos_rate += batch_metrics["positive_rate"]
            total_metric_batches += 1

            all_binary_logits.append(outputs["logits"].detach().cpu())
            all_binary_labels.append(labels.detach().cpu())

            progress_bar.set_postfix(
                loss=f"{(total_loss + outputs['loss'].item()) / (total_batches + 1):.4f}",
                acc=f"{batch_metrics['accuracy']:.4f}",
                f1=f"{batch_metrics['f1']:.4f}",
            )

        elif task_type == "generative":
            encoder_input_ids = batch["input_ids"].to(device)
            encoder_attention_mask = batch["attention_mask"].to(device)
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

            batch_metrics = generative_token_metrics(
                outputs["logits"],
                labels,
                ignore_label=-100,
            )

            total_gen_token_accuracy += batch_metrics["token_accuracy"]
            total_gen_exact_match_rate += batch_metrics["exact_match_rate"]
            total_gen_metric_batches += 1

            progress_bar.set_postfix(
                loss=f"{(total_loss + outputs['loss'].item()) / (total_batches + 1):.4f}",
                tok_acc=f"{batch_metrics['token_accuracy']:.4f}",
                exact=f"{batch_metrics['exact_match_rate']:.4f}",
            )

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        loss = outputs["loss"]
        assert loss is not None
        total_loss += loss.item()
        total_batches += 1

    metrics = {"loss": total_loss / max(total_batches, 1)}

    if task_type == "binary_classification" and total_metric_batches > 0:
        metrics["accuracy"] = total_accuracy / total_metric_batches
        metrics["precision"] = total_precision / total_metric_batches
        metrics["recall"] = total_recall / total_metric_batches
        metrics["f1"] = total_f1 / total_metric_batches
        metrics["majority_baseline_accuracy"] = total_majority / total_metric_batches
        metrics["positive_rate"] = total_pos_rate / total_metric_batches

        binary_logits = torch.cat(all_binary_logits, dim=0)
        binary_labels = torch.cat(all_binary_labels, dim=0)
        metrics["auroc"] = binary_auroc_from_logits(binary_logits, binary_labels)

    elif task_type == "generative" and total_gen_metric_batches > 0:
        metrics["token_accuracy"] = total_gen_token_accuracy / total_gen_metric_batches
        metrics["exact_match_rate"] = total_gen_exact_match_rate / total_gen_metric_batches

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, required=True, choices=["motif", "mutation", "splicing"])
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
    print(f"Fine-tuning BART on {device} for task: {args.task}")

    tokenizer = KmerDNATokenizer(k=args.k, overlapping=True)

    print(f"Loading pretrained config from {args.pretrained_dir}...")
    checkpoint_state = load_checkpoint_state(args.pretrained_dir, map_location="cpu")
    config = BARTConfig(**checkpoint_state["config"])
    config.validate()

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

    base_model = BARTModel(config)

    print(f"Loading pretrained weights from {args.pretrained_dir}...")
    load_checkpoint(args.pretrained_dir, base_model)

    task_instance = build_task(args.task, args.data_file)

    if args.task in {"motif", "mutation"}:
        task_type = "binary_classification"
        model = BARTForBinaryClassification(base_model, config)
        collator = ClassificationCollator(tokenizer, max_length=args.max_length)
    else:
        task_type = "generative"
        model = base_model
        collator = BARTGenerativeCollator(tokenizer, max_length=args.max_length)

    model.to(device)

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

    best_val_loss = float("inf")

    print("Evaluating pretrained baseline (epoch 0)...")

    baseline_metrics = evaluate_bart(
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
        train_metrics = train_bart_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            task_type=task_type,
        )

        val_metrics = evaluate_bart(
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
        else:
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

    print(f"Finished fine-tuning BART. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()