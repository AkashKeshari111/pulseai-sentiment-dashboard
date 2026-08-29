"""Evaluate a saved checkpoint on the held-out test set and record the result.

`src/train_transformer.py` does this automatically at the end of a run. This
script exists for the case where training was interrupted after a checkpoint was
written but before the final evaluation ran - the weights are on disk and
perfectly usable, they just have no reported numbers yet.

Usage:
    python scripts/evaluate_checkpoint.py
    python scripts/evaluate_checkpoint.py --model-dir models/distilbert-sentiment \\
        --max-length 256 --key distilbert --epochs-completed 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PATHS, TRAINING  # noqa: E402
from src.dataset import load_splits  # noqa: E402
from src.metrics import format_metrics, save_model_metrics  # noqa: E402
from src.train_transformer import (  # noqa: E402
    DynamicPadCollator,
    _measure_latency,
    build_dataset,
    evaluate,
    resolve_device,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/distilbert-sentiment")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--key", default="distilbert",
                        help="key to write under reports/metrics.json -> models")
    parser.add_argument("--label", default=None,
                        help="human-readable model name for the reports")
    parser.add_argument("--epochs-completed", type=int, default=None)
    parser.add_argument("--train-seconds", type=float, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(args.model_dir)
    if not (model_dir / "config.json").exists():
        print(f"No checkpoint at {model_dir}")
        return 1

    device = resolve_device(args.device)
    print(f"[eval] checkpoint : {model_dir}")
    print(f"[eval] device     : {device}")
    print(f"[eval] max_length : {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    test_df = load_splits()["test"]
    loader = DataLoader(
        build_dataset(test_df, tokenizer, args.max_length),
        batch_size=TRAINING.eval_batch_size,
        collate_fn=DynamicPadCollator(tokenizer),
    )

    print(f"[eval] scoring {len(test_df)} held-out test examples...")
    started = time.perf_counter()
    metrics, _ = evaluate(model, loader, device)
    print(f"[eval] done in {time.perf_counter() - started:.0f}s")

    latency = _measure_latency(model, tokenizer, device, args.max_length)

    # Recover the training history the interrupted run had already written.
    history = []
    history_path = model_dir / "training_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))

    n_params = sum(p.numel() for p in model.parameters())
    metrics.update(
        {
            "model": args.label
            or f"DistilBERT fine-tuned (max_seq_length={args.max_length})",
            "model_key": args.key,
            "parameters": int(n_params),
            "history": history,
            "best_epoch": (
                args.epochs_completed
                or (max((h["epoch"] for h in history), default=None))
            ),
            "val_f1_macro": max((h["val_f1_macro"] for h in history), default=None),
            "train_seconds": args.train_seconds
            or sum(h.get("seconds", 0) for h in history),
            "latency_ms_per_sample": latency,
            "device": str(device),
            "hyperparameters": {
                **asdict(TRAINING),
                "max_seq_length": args.max_length,
                "epochs": args.epochs_completed,
            },
        }
    )

    save_model_metrics(args.key, metrics)
    print(format_metrics(f"{metrics['model']} — TEST SET", metrics))
    print(f"[eval] written to {PATHS.metrics_json} under models.{args.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
