from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# ==================
# 1. MODEL WRAPPERS
# ==================
class GPTForBinaryClassification(nn.Module):
    """
    Wrap a pretrained GPT backbone for binary sequence classification.

    Strategy:
        - run GPT
        - take the hidden state of the last non-padding token
        - pass through a linear classifier
    """

    def __init__(self, base_gpt_model, config):
        super().__init__()
        self.gpt = base_gpt_model

        # Remove LM head for classification usage
        self.gpt.lm_head = nn.Identity()

        self.classifier = nn.Linear(config.d_model, 1)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs["hidden_states"]

        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        sequence_lengths = sequence_lengths.clamp(min=0)

        last_token_states = hidden_states[
            torch.arange(batch_size, device=input_ids.device), sequence_lengths
        ]

        logits = self.classifier(last_token_states)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return {"loss": loss, "logits": logits}

class BERTForBinaryClassification(nn.Module):
    """
    Wrap a pretrained BERT backbone for binary sequence classification.

    Strategy:
        - run BERT
        - take the [CLS] hidden state
        - pass through a linear classifier
    """

    def __init__(self, base_bert_model, config):
        super().__init__()
        self.bert = base_bert_model
        self.classifier = nn.Linear(config.d_model, 1)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs["hidden_states"]   # (B, T, d_model)

        cls_states = hidden_states[:, 0, :]        # [CLS] token
        logits = self.classifier(cls_states)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        return {"loss": loss, "logits": logits}

class BERTForTokenClassification(nn.Module):
    """
    BERT token classification model.

    Predicts one label per token.
    """

    def __init__(self, base_bert_model, config, num_labels: int = 1):
        super().__init__()
        self.bert = base_bert_model
        self.num_labels = num_labels
        self.classifier = nn.Linear(config.d_model, num_labels)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs["hidden_states"]   # (B, T, d_model)
        logits = self.classifier(hidden_states)    # (B, T, 1)

        loss = None
        if labels is not None:
            # labels shape: (B, T), values in {0,1} or -100 for ignore
            active_mask = labels != -100
            if active_mask.any():
                active_logits = logits.squeeze(-1)[active_mask]
                active_labels = labels[active_mask].float()
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = torch.tensor(0.0, device=input_ids.device)

        return {"loss": loss, "logits": logits}

# ==========================================
# 2. DATASET
# ==========================================
class TaskSFTDataset(Dataset):
    """
    Wrap a task object that supports __len__ and __getitem__.
    """

    def __init__(self, task_instance):
        self.task = task_instance

    def __len__(self):
        return len(self.task)

    def __getitem__(self, idx):
        return self.task[idx]


# ==========================================
# 3. COLLATORS
# ==========================================
class SFTCollator:
    """
    Generative collator for tasks like exon splicing.

    Expects examples shaped like:
        {
            "messages": [
                {"role": "user", "content": ...},
                {"role": "assistant", "content": ...},
            ]
        }

    Only the assistant portion contributes to loss.
    """

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.sep_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    def __call__(self, batch):
        batch_input_ids = []
        batch_labels = []

        for item in batch:
            user_text = item["messages"][0]["content"]
            asst_text = item["messages"][1]["content"]

            user_tokens = self.tokenizer.encode(user_text)
            asst_tokens = self.tokenizer.encode(asst_text)

            input_ids = user_tokens + [self.sep_token_id] + asst_tokens + [self.sep_token_id]
            labels = [0] * len(user_tokens) + [0] + asst_tokens + [self.sep_token_id]

            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        input_ids_padded = pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels_padded = pad_sequence(
            batch_labels,
            batch_first=True,
            padding_value=0,
        )
        attention_mask = (input_ids_padded != self.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }


class ClassificationCollator:
    """
    Binary classification collator for tasks like motif recognition / mutation sensitivity.
    """

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        batch_input_ids = []
        batch_labels = []

        for item in batch:
            user_text = item["messages"][0]["content"]
            label_text = item["messages"][1]["content"]

            token_ids = self.tokenizer.encode(user_text)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]

            batch_input_ids.append(torch.tensor(token_ids, dtype=torch.long))
            batch_labels.append(torch.tensor([float(label_text)], dtype=torch.float))

        input_ids_padded = pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = (input_ids_padded != self.pad_token_id).long()
        labels_tensor = torch.stack(batch_labels)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
        }

class TokenClassificationCollator:
    """
    Collator for token-level classification tasks such as exon vs non-exon.

    Expected example format:
        {
            "sequence": "...",
            "token_labels": [0, 1, 1, 0, ...]   # aligned to tokenizer.encode(sequence)
        }
    """

    def __init__(self, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch):
        batch_input_ids = []
        batch_labels = []

        for item in batch:
            sequence = item["sequence"]
            token_labels = item["token_labels"]

            token_ids = self.tokenizer.encode(sequence)

            if len(token_ids) != len(token_labels):
                raise ValueError(
                    f"token label length {len(token_labels)} does not match tokenized sequence length {len(token_ids)}"
                )

            token_ids = self.tokenizer.build_bert_inputs(token_ids)
            labels = [-100] + list(token_labels) + [-100]   # ignore [CLS], [SEP]

            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
                labels = labels[:self.max_length]

            batch_input_ids.append(token_ids)
            batch_labels.append(labels)

        padded_inputs = self.tokenizer.pad_token_ids(
            batch_input_ids,
            max_length=self.max_length,
            padding=True,
            truncation=self.max_length is not None,
            return_attention_mask=True,
        )
        padded_labels = self.tokenizer.pad_token_ids(
            batch_labels,
            max_length=self.max_length,
            padding=True,
            truncation=self.max_length is not None,
            return_attention_mask=False,
            pad_value=-100,
        )

        input_ids = torch.tensor(padded_inputs["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(padded_inputs["attention_mask"], dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        labels = torch.tensor(padded_labels["input_ids"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

# ==========================================
# 4. METRICS
# ==========================================

@torch.no_grad()
def binary_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """
    Compute binary classification metrics from logits.

    Returns:
        accuracy, precision, recall, f1,
        majority_baseline_accuracy,
        positive_rate
    """
    probs = torch.sigmoid(logits.view(-1))
    preds = (probs >= 0.5).long()
    gold = labels.view(-1).long()

    total = len(gold)

    # Accuracy
    accuracy = (preds == gold).float().mean().item()

    # Confusion matrix
    tp = ((preds == 1) & (gold == 1)).sum().item()
    fp = ((preds == 1) & (gold == 0)).sum().item()
    fn = ((preds == 0) & (gold == 1)).sum().item()
    tn = ((preds == 0) & (gold == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Majority baseline
    num_pos = (gold == 1).sum().item()
    num_neg = (gold == 0).sum().item()
    majority_baseline_accuracy = max(num_pos, num_neg) / total if total > 0 else 0.0

    # Debug signal: how often model predicts positive
    positive_rate = (preds == 1).float().mean().item()

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "majority_baseline_accuracy": float(majority_baseline_accuracy),
        "positive_rate": float(positive_rate),
    }

@torch.no_grad()
def token_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """
    logits: (B, T, 1) or (B, T)
    labels: (B, T) with values in {0,1} or -100 for ignored positions
    """
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

    # Overall token accuracy
    token_accuracy = (preds == gold).float().mean().item()

    # Majority baseline
    num_exon = (gold == 1).sum().item()
    num_non_exon = (gold == 0).sum().item()
    total = len(gold)

    majority_baseline_accuracy = max(num_exon, num_non_exon) / total if total > 0 else 0.0

    # Confusion counts for exon=positive class
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
        "token_accuracy": float(token_accuracy),
        "majority_baseline_accuracy": float(majority_baseline_accuracy),
        "exon_precision": float(exon_precision),
        "exon_recall": float(exon_recall),
        "exon_f1": float(exon_f1),
        "non_exon_accuracy": float(non_exon_accuracy),
        "exon_accuracy": float(exon_accuracy),
    }

@torch.no_grad()
def generative_token_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_label: int = 0,
) -> dict[str, float]:
    """
    Compute token-level metrics for GPT generative tasks.

    Assumes:
      - logits shape: (B, T, V)
      - labels shape: (B, T)
      - GPT loss is next-token prediction, so we compare:
            logits[:, :-1, :]  vs labels[:, 1:]
      - labels == ignore_label are ignored (user prompt / padded positions)

    Returns metrics over the supervised assistant portion only.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (B, T, V), got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape (B, T), got {tuple(labels.shape)}")

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    active_mask = shift_labels != ignore_label
    if not active_mask.any():
        return {
            "token_accuracy": 0.0,
            "majority_baseline_accuracy": 0.0,
            "exon_precision": 0.0,
            "exon_recall": 0.0,
            "exon_f1": 0.0,
            "non_exon_accuracy": 0.0,
            "exon_accuracy": 0.0,
            "exact_match_rate": 0.0,
        }

    preds = torch.argmax(shift_logits, dim=-1)

    active_preds = preds[active_mask]
    active_gold = shift_labels[active_mask]

    token_accuracy = (active_preds == active_gold).float().mean().item()

    # Majority baseline over supervised tokens
    unique_vals, counts = torch.unique(active_gold, return_counts=True)
    majority_baseline_accuracy = counts.max().item() / counts.sum().item()

    # For exon/non-exon metrics, assume labels are binary token ids 0/1
    # after task formatting. If that is not true for your splicing target
    # representation, these metrics will not be meaningful.
    tp = ((active_preds == 1) & (active_gold == 1)).sum().item()
    fp = ((active_preds == 1) & (active_gold == 0)).sum().item()
    fn = ((active_preds == 0) & (active_gold == 1)).sum().item()
    tn = ((active_preds == 0) & (active_gold == 0)).sum().item()

    exon_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    exon_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    exon_f1 = (
        2 * exon_precision * exon_recall / (exon_precision + exon_recall)
        if (exon_precision + exon_recall) > 0
        else 0.0
    )

    non_exon_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    exon_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Exact match at sequence level over supervised assistant tokens
    exact_matches = []
    batch_size = shift_labels.size(0)
    for b in range(batch_size):
        row_mask = active_mask[b]
        if row_mask.any():
            row_pred = preds[b][row_mask]
            row_gold = shift_labels[b][row_mask]
            exact_matches.append(float(torch.equal(row_pred, row_gold)))

    exact_match_rate = sum(exact_matches) / len(exact_matches) if exact_matches else 0.0

    return {
        "token_accuracy": float(token_accuracy),
        "majority_baseline_accuracy": float(majority_baseline_accuracy),
        "exon_precision": float(exon_precision),
        "exon_recall": float(exon_recall),
        "exon_f1": float(exon_f1),
        "non_exon_accuracy": float(non_exon_accuracy),
        "exon_accuracy": float(exon_accuracy),
        "exact_match_rate": float(exact_match_rate),
    }

# ==========================================
# 5. TRAIN / EVAL LOOPS
# ==========================================
def train_sft_epoch(
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

        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids.to(device)

        outputs = model(**model_kwargs)

        loss = outputs["loss"]
        assert loss is not None

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1
        avg_loss = total_loss / total_batches

        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    metrics = {"loss": total_loss / max(total_batches, 1)}
    return metrics

@torch.no_grad()
def evaluate_sft(
    model,
    dataloader,
    device,
    epoch: int,
    task_type: str,
):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    # For binary classification metrics
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_majority = 0.0
    total_pos_rate = 0.0
    total_classification_batches = 0

    # For token-classification metrics
    total_token_accuracy = 0.0
    total_majority_baseline_accuracy = 0.0
    total_exon_precision = 0.0
    total_exon_recall = 0.0
    total_exon_f1 = 0.0
    total_non_exon_accuracy = 0.0
    total_exon_accuracy = 0.0
    total_token_metric_batches = 0

    # For generative metrics
    total_gen_token_accuracy = 0.0
    total_gen_majority_baseline_accuracy = 0.0
    total_gen_exon_precision = 0.0
    total_gen_exon_recall = 0.0
    total_gen_exon_f1 = 0.0
    total_gen_non_exon_accuracy = 0.0
    total_gen_exon_accuracy = 0.0
    total_gen_exact_match_rate = 0.0
    total_gen_metric_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [val]",
        leave=False,
    )

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids.to(device)

        outputs = model(**model_kwargs)

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

            total_token_accuracy += batch_metrics["token_accuracy"]
            total_majority_baseline_accuracy += batch_metrics["majority_baseline_accuracy"]
            total_exon_precision += batch_metrics["exon_precision"]
            total_exon_recall += batch_metrics["exon_recall"]
            total_exon_f1 += batch_metrics["exon_f1"]
            total_non_exon_accuracy += batch_metrics["non_exon_accuracy"]
            total_exon_accuracy += batch_metrics["exon_accuracy"]
            total_token_metric_batches += 1

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                tok_acc=f"{batch_metrics['token_accuracy']:.4f}",
                exon_f1=f"{batch_metrics['exon_f1']:.4f}",
            )

        elif task_type == "generative":
            batch_metrics = generative_token_metrics(outputs["logits"], labels)

            total_gen_token_accuracy += batch_metrics["token_accuracy"]
            total_gen_majority_baseline_accuracy += batch_metrics["majority_baseline_accuracy"]
            total_gen_exon_precision += batch_metrics["exon_precision"]
            total_gen_exon_recall += batch_metrics["exon_recall"]
            total_gen_exon_f1 += batch_metrics["exon_f1"]
            total_gen_non_exon_accuracy += batch_metrics["non_exon_accuracy"]
            total_gen_exon_accuracy += batch_metrics["exon_accuracy"]
            total_gen_exact_match_rate += batch_metrics["exact_match_rate"]
            total_gen_metric_batches += 1

            progress_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                tok_acc=f"{batch_metrics['token_accuracy']:.4f}",
                exon_f1=f"{batch_metrics['exon_f1']:.4f}",
                exact=f"{batch_metrics['exact_match_rate']:.4f}",
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

    elif task_type == "token_classification" and total_token_metric_batches > 0:
        metrics["token_accuracy"] = total_token_accuracy / total_token_metric_batches
        metrics["majority_baseline_accuracy"] = (
            total_majority_baseline_accuracy / total_token_metric_batches
        )
        metrics["exon_precision"] = total_exon_precision / total_token_metric_batches
        metrics["exon_recall"] = total_exon_recall / total_token_metric_batches
        metrics["exon_f1"] = total_exon_f1 / total_token_metric_batches
        metrics["non_exon_accuracy"] = (
            total_non_exon_accuracy / total_token_metric_batches
        )
        metrics["exon_accuracy"] = total_exon_accuracy / total_token_metric_batches

    elif task_type == "generative" and total_gen_metric_batches > 0:
        metrics["token_accuracy"] = total_gen_token_accuracy / total_gen_metric_batches
        metrics["majority_baseline_accuracy"] = (
            total_gen_majority_baseline_accuracy / total_gen_metric_batches
        )
        metrics["exon_precision"] = total_gen_exon_precision / total_gen_metric_batches
        metrics["exon_recall"] = total_gen_exon_recall / total_gen_metric_batches
        metrics["exon_f1"] = total_gen_exon_f1 / total_gen_metric_batches
        metrics["non_exon_accuracy"] = (
            total_gen_non_exon_accuracy / total_gen_metric_batches
        )
        metrics["exon_accuracy"] = total_gen_exon_accuracy / total_gen_metric_batches
        metrics["exact_match_rate"] = (
            total_gen_exact_match_rate / total_gen_metric_batches
        )

    return metrics