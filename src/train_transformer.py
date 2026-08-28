"""Fine-tune DistilBERT for 3-class customer sentiment.

The training loop is written explicitly with PyTorch rather than delegating to
``transformers.Trainer``. That is a deliberate choice for a capstone: the loop
below shows the actual mechanics being assessed - batching, the optimiser and
its decoupled weight decay, linear warmup/decay scheduling, gradient clipping,
mixed precision, per-epoch validation and best-checkpoint selection - instead
of hiding them behind one ``trainer.train()`` call.

DistilBERT (66M parameters, 6 layers) is chosen over BERT-base (110M, 12
layers) because it retains ~97% of BERT's language understanding at ~60% of
the compute, which is what makes CPU-only fine-tuning and sub-100ms API
inference realistic.

Run:
    python -m src.train_transformer
    python -m src.train_transformer --epochs 3 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import ID2LABEL, LABEL2ID, NUM_LABELS, PATHS, TRAINING
from src.dataset import load_splits
from src.metrics import compute_metrics, format_metrics, save_model_metrics
from src.preprocessing import clean_for_transformer

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG that participates in training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the best available accelerator.

    ``mps`` is included so the project trains natively on Apple silicon; the
    fallback chain always ends at CPU, which is what this project was
    developed and benchmarked on.
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """Tokenises lazily so memory stays flat regardless of corpus size.

    Sequences are deliberately **not** padded here - padding happens per batch
    in :class:`DynamicPadCollator`.
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.texts[index], truncation=True, max_length=self.max_length
        )
        encoded["labels"] = self.labels[index]
        return encoded


class DynamicPadCollator:
    """Pad each batch to its own longest sequence, not to ``max_length``.

    Padding everything to 128 tokens makes every batch cost the same as the
    longest review in the corpus. Padding per batch means short feedback ("Great
    service!") costs what it should. On mixed-length customer feedback this is
    a substantial throughput win for free, and it changes nothing about the
    maths - attention masks already tell the model to ignore pad positions.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.long)
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


def build_dataset(frame: pd.DataFrame, tokenizer, max_length: int) -> SentimentDataset:
    texts = [clean_for_transformer(text) for text in frame["text"].astype(str)]
    return SentimentDataset(texts, frame["label"].astype(int).tolist(), tokenizer, max_length)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> tuple[dict, float]:
    """Run the model over a loader, returning metrics and mean loss."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss, batches = 0.0, 0

    for batch in loader:
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=labels,
        )
        total_loss += float(outputs.loss.item())
        batches += 1
        all_logits.append(outputs.logits.float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_labels)
    y_prob = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    y_pred = logits.argmax(axis=1)
    return compute_metrics(y_true, y_pred, y_prob), total_loss / max(batches, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    max_length: int | None = None,
    device_preference: str = "auto",
    output_dir: Path | None = None,
    limit: int | None = None,
) -> dict:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    epochs = epochs or TRAINING.epochs
    batch_size = batch_size or TRAINING.batch_size
    learning_rate = learning_rate or TRAINING.learning_rate
    max_length = max_length or TRAINING.max_seq_length
    output_dir = Path(output_dir or PATHS.transformer_dir)

    set_seed(TRAINING.seed)
    device = resolve_device(device_preference)

    splits = load_splits()
    if limit:  # smoke-test path used by CI and the quickstart script
        splits = {name: frame.head(limit) for name, frame in splits.items()}

    print(f"[bert] device={device} | torch={torch.__version__} | {platform.processor() or platform.machine()}")
    print(f"[bert] loading tokenizer + {TRAINING.base_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(TRAINING.base_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRAINING.base_checkpoint,
        num_labels=NUM_LABELS,
        id2label={int(k): v for k, v in ID2LABEL.items()},
        label2id=LABEL2ID,
    ).to(device)

    collator = DynamicPadCollator(tokenizer)
    train_loader = DataLoader(
        build_dataset(splits["train"], tokenizer, max_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        build_dataset(splits["val"], tokenizer, max_length),
        batch_size=TRAINING.eval_batch_size,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        build_dataset(splits["test"], tokenizer, max_length),
        batch_size=TRAINING.eval_batch_size,
        collate_fn=collator,
    )

    # --- Optimiser -----------------------------------------------------------
    # Weight decay is applied to weight matrices only. Applying it to biases
    # and LayerNorm parameters measurably hurts fine-tuning, which is why the
    # parameters are split into two groups here.
    no_decay = ("bias", "LayerNorm.weight")
    grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": TRAINING.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_parameters, lr=learning_rate)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * TRAINING.warmup_ratio),
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[bert] {n_params / 1e6:.1f}M params | {len(train_loader)} steps/epoch "
        f"| {total_steps} total steps | amp={use_amp}"
    )

    history: list[dict] = []
    best_f1, best_epoch = -1.0, -1
    output_dir.mkdir(parents=True, exist_ok=True)
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, seen = 0.0, 0
        epoch_started = time.perf_counter()

        for step, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            # Unscale before clipping, otherwise the norm is computed on
            # scaled gradients and the threshold means nothing.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += float(loss.item())
            seen += 1

            if step % max(1, len(train_loader) // 20) == 0 or step == len(train_loader):
                elapsed = time.perf_counter() - epoch_started
                rate = step / max(elapsed, 1e-9)
                eta = (len(train_loader) - step) / max(rate, 1e-9)
                print(
                    f"\r[bert] epoch {epoch}/{epochs}  step {step}/{len(train_loader)}"
                    f"  loss={epoch_loss / seen:.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                    f"  {rate:.2f} it/s  eta {eta / 60:.1f}m",
                    end="",
                    flush=True,
                )

        print()
        val_metrics, val_loss = evaluate(model, val_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(epoch_loss / max(seen, 1), 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "val_f1_macro": round(val_metrics["f1_macro"], 4),
            "seconds": round(time.perf_counter() - epoch_started, 1),
        }
        history.append(epoch_record)
        print(
            f"[bert] epoch {epoch} | train_loss={epoch_record['train_loss']} "
            f"val_loss={epoch_record['val_loss']} "
            f"val_acc={epoch_record['val_accuracy']} "
            f"val_f1={epoch_record['val_f1_macro']} "
            f"({epoch_record['seconds']}s)"
        )

        # Checkpoint on validation macro-F1, never on training loss: the point
        # is to keep the epoch that generalises best, not the one that fit
        # hardest.
        if val_metrics["f1_macro"] > best_f1:
            best_f1, best_epoch = val_metrics["f1_macro"], epoch
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"[bert] new best (macro-F1 {best_f1:.4f}) -> saved to {output_dir}")

    train_seconds = time.perf_counter() - training_started

    # --- Final test evaluation on the best checkpoint -------------------------
    print(f"[bert] reloading best checkpoint (epoch {best_epoch}) for test evaluation")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_metrics, _ = evaluate(model, test_loader, device)

    latency_ms = _measure_latency(model, tokenizer, device, max_length)

    test_metrics.update(
        {
            "model": f"DistilBERT fine-tuned ({TRAINING.base_checkpoint})",
            "model_key": "distilbert",
            "parameters": int(n_params),
            "best_epoch": best_epoch,
            "val_f1_macro": round(best_f1, 4),
            "history": history,
            "train_seconds": round(train_seconds, 1),
            "latency_ms_per_sample": latency_ms,
            "device": str(device),
            "hyperparameters": {
                **asdict(TRAINING),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_seq_length": max_length,
            },
        }
    )

    save_model_metrics("distilbert", test_metrics)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    print(format_metrics("DISTILBERT FINE-TUNED (test set)", test_metrics))
    print(f"[bert] total training time: {train_seconds / 60:.1f} min")
    print(f"[bert] checkpoint: {output_dir}")
    return test_metrics


@torch.no_grad()
def _measure_latency(model, tokenizer, device, max_length: int, runs: int = 30) -> float:
    """Single-sample inference latency - the number that matters for the API."""
    model.eval()
    sample = "The delivery was late and the packaging was damaged, very disappointing."
    encoded = tokenizer(
        sample, truncation=True, max_length=max_length, padding="max_length",
        return_tensors="pt",
    ).to(device)

    for _ in range(5):  # warm up lazy kernels / allocator
        model(**encoded)

    started = time.perf_counter()
    for _ in range(runs):
        model(**encoded)
    return round((time.perf_counter() - started) / runs * 1000, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for sentiment.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use only N rows per split (fast smoke test)",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device_preference=args.device,
        output_dir=args.output_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
