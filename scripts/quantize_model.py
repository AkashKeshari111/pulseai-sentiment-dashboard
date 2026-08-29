"""Export the fine-tuned model to ONNX, quantize it to INT8, and measure the cost.

Motivation: the PyTorch API measures **601 MB** resident, which does not fit the
512 MB ceiling common to free container tiers. INT8 quantization stores each
weight in 8 bits instead of 32, so the checkpoint drops roughly 4x and CPU
inference gets faster - at the price of some numerical precision.

"Some precision" is not a number, so this script produces one: it evaluates the
quantized model on the same held-out test set as every other run in this project
and writes the result to reports/metrics.json alongside them. The decision to
ship it or not is then made on evidence.

Usage:
    python scripts/quantize_model.py
    python scripts/quantize_model.py --skip-eval        # export only
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS, PATHS  # noqa: E402
from src.metrics import compute_metrics, format_metrics, save_model_metrics  # noqa: E402
from src.preprocessing import clean_for_transformer  # noqa: E402


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024 / 1024


def export_and_quantize(source: Path, target: Path) -> tuple[float, float]:
    """FP32 ONNX export followed by dynamic INT8 quantization."""
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    fp32_dir = target.with_name(target.name + "-fp32")
    for directory in (fp32_dir, target):
        if directory.exists():
            shutil.rmtree(directory)

    print(f"[onnx] exporting {source} to ONNX...")
    model = ORTModelForSequenceClassification.from_pretrained(source, export=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model.save_pretrained(fp32_dir)
    tokenizer.save_pretrained(fp32_dir)
    fp32_mb = directory_size_mb(fp32_dir)
    print(f"[onnx] fp32 export: {fp32_mb:.0f} MB")

    print("[onnx] quantizing to INT8 (dynamic, per-channel, reduce_range)...")
    quantizer = ORTQuantizer.from_pretrained(fp32_dir)
    # Dynamic quantization computes activation ranges at inference time, so it
    # needs no calibration dataset.
    #
    # avx2 + reduce_range, NOT avx512_vnni, and this is not a detail. A model
    # quantized for AVX-512 VNNI runs on a CPU without those instructions but
    # can *silently* produce garbage: U8S8 accumulation saturates, and the
    # output collapses towards a uniform distribution with no error raised.
    # This was observed in production - identical image and weights gave
    # "negative 98.8%" on a VNNI laptop and "neutral 37%" on the deploy host.
    # reduce_range keeps weights in 7 bits, which removes the saturation and
    # makes the artefact portable across x86 generations.
    config = AutoQuantizationConfig.avx2(
        is_static=False, per_channel=True, reduce_range=True
    )
    quantizer.quantize(save_dir=target, quantization_config=config)
    tokenizer.save_pretrained(target)

    int8_mb = directory_size_mb(target)
    print(f"[onnx] int8 model : {int8_mb:.0f} MB  ({fp32_mb / int8_mb:.1f}x smaller)")
    shutil.rmtree(fp32_dir, ignore_errors=True)
    return fp32_mb, int8_mb


def evaluate_onnx(model_dir: Path, max_length: int, batch_size: int = 32) -> dict:
    """Score the ONNX model on the held-out test set."""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    from src.dataset import load_splits

    model = ORTModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    test_df = load_splits()["test"]
    texts = [clean_for_transformer(t) for t in test_df["text"].astype(str)]
    y_true = test_df["label"].to_numpy()

    print(f"[eval] scoring {len(texts)} test examples...")
    started = time.perf_counter()
    logits: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, truncation=True, max_length=max_length,
            padding=True, return_tensors="np",
        )
        logits.append(model(**encoded).logits)
    elapsed = time.perf_counter() - started

    stacked = np.concatenate(logits)
    exp = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predictions = stacked.argmax(axis=1)

    print(f"[eval] done in {elapsed:.0f}s")
    return compute_metrics(y_true, predictions, probabilities)


def measure_latency(model_dir: Path, max_length: int, runs: int = 30) -> float:
    """Single-sample latency, measured the *same way* as the PyTorch baseline.

    `src/train_transformer.py` pads its timing sample to `max_length`, so this
    must too. Timing an unpadded 15-token input against a padded 256-token one
    makes quantization look ~17x faster when the honest figure is closer to 3x -
    the rest is just the padding difference.
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    model = ORTModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    sample = "The delivery was late and the packaging was damaged, very disappointing."
    encoded = tokenizer(
        sample, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="np",
    )

    for _ in range(5):  # warm up
        model(**encoded)

    started = time.perf_counter()
    for _ in range(runs):
        model(**encoded)
    return round((time.perf_counter() - started) / runs * 1000, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="models/distilbert-sentiment")
    parser.add_argument("--target", default="models/distilbert-sentiment-onnx-int8")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--key", default="distilbert_int8")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    source, target = Path(args.source), Path(args.target)
    if not (source / "config.json").exists():
        print(f"No checkpoint at {source}")
        return 1

    torch_mb = directory_size_mb(source)
    fp32_mb, int8_mb = export_and_quantize(source, target)

    if args.skip_eval:
        return 0

    metrics = evaluate_onnx(target, args.max_length)
    latency = measure_latency(target, args.max_length)

    metrics.update(
        {
            "model": "DistilBERT fine-tuned, ONNX INT8 quantized",
            "model_key": args.key,
            "latency_ms_per_sample": latency,
            "device": "cpu (onnxruntime)",
            "size_mb": round(int8_mb, 1),
            "size_mb_original": round(torch_mb, 1),
            "hyperparameters": {"max_seq_length": args.max_length,
                                "quantization": "dynamic INT8, per-channel, avx512_vnni"},
        }
    )
    save_model_metrics(args.key, metrics)
    print(format_metrics("DISTILBERT — ONNX INT8 (test set)", metrics))

    # The comparison that decides whether this ships.
    from src.metrics import load_model_metrics

    baseline = load_model_metrics()["models"].get("distilbert")
    if baseline:
        drop = baseline["f1_macro"] - metrics["f1_macro"]
        speedup = baseline.get("latency_ms_per_sample", 0) / max(latency, 1e-9)
        print()
        print("=" * 62)
        print("  QUANTIZATION TRADE-OFF")
        print("=" * 62)
        print(f"  size      {torch_mb:>7.0f} MB  ->  {int8_mb:>6.0f} MB"
              f"   ({torch_mb / int8_mb:.1f}x smaller)")
        print(f"  latency   {baseline.get('latency_ms_per_sample', 0):>7.1f} ms  ->"
              f"  {latency:>6.1f} ms   ({speedup:.1f}x faster)")
        print(f"  macro-F1  {baseline['f1_macro']:>7.4f}     ->  {metrics['f1_macro']:>6.4f}"
              f"   ({-drop:+.4f}, {-drop / baseline['f1_macro'] * 100:+.2f}%)")
        print()
        for label in LABELS:
            before = baseline["per_class"][label]["f1"]
            after = metrics["per_class"][label]["f1"]
            print(f"  {label:<10} F1 {before:.3f} -> {after:.3f}  ({after - before:+.3f})")
        print("=" * 62)

    print(f"\n[done] quantized model at {target}")
    print(f"[done] metrics written to {PATHS.metrics_json} under models.{args.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
