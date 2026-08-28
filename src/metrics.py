"""Shared evaluation utilities.

Both the baseline and the transformer report through this module so the
numbers in the notebook, the final report and the dashboard's Model Card page
are computed identically and are therefore directly comparable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src.config import LABELS, PATHS


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], y_prob: np.ndarray | None = None
) -> dict:
    """Accuracy plus macro/weighted F1 and the full per-class breakdown.

    Macro-F1 is the headline number rather than accuracy: the neutral class is
    both the smallest and the hardest, and accuracy would let a model hide a
    total failure on it behind good performance on the two easy classes.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(LABELS)))
        ).tolist(),
        "labels": LABELS,
        "n_samples": int(len(y_true)),
    }

    if y_prob is not None:
        probs = np.asarray(y_prob)
        confidence = probs.max(axis=1)
        correct = y_true == y_pred
        metrics["mean_confidence"] = float(confidence.mean())
        metrics["mean_confidence_correct"] = (
            float(confidence[correct].mean()) if correct.any() else 0.0
        )
        metrics["mean_confidence_incorrect"] = (
            float(confidence[~correct].mean()) if (~correct).any() else 0.0
        )
    return metrics


def format_metrics(name: str, metrics: dict) -> str:
    """Human readable one-screen summary, used by the training CLIs."""
    lines = [
        f"\n{'=' * 62}",
        f"  {name}",
        f"{'=' * 62}",
        f"  accuracy     : {metrics['accuracy']:.4f}",
        f"  f1 (macro)   : {metrics['f1_macro']:.4f}",
        f"  f1 (weighted): {metrics['f1_weighted']:.4f}",
        "",
        f"  {'class':<12}{'precision':>11}{'recall':>10}{'f1':>10}{'support':>10}",
    ]
    for label, scores in metrics["per_class"].items():
        lines.append(
            f"  {label:<12}{scores['precision']:>11.3f}{scores['recall']:>10.3f}"
            f"{scores['f1']:>10.3f}{scores['support']:>10d}"
        )
    lines.append("")
    lines.append("  confusion matrix (rows = true, cols = predicted)")
    header = "".join(f"{label[:4]:>8}" for label in metrics["labels"])
    lines.append(f"  {'':<10}{header}")
    for label, row in zip(metrics["labels"], metrics["confusion_matrix"], strict=True):
        cells = "".join(f"{value:>8d}" for value in row)
        lines.append(f"  {label:<10}{cells}")
    lines.append("=" * 62)
    return "\n".join(lines)


def save_model_metrics(model_key: str, payload: dict, path: Path | None = None) -> Path:
    """Merge one model's results into ``reports/metrics.json``.

    The file accumulates results across runs (baseline, transformer, ...) so
    the dashboard can render a side-by-side comparison from a single fetch.
    """
    path = path or PATHS.metrics_json
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    models = existing.get("models", {})
    models[model_key] = payload
    existing["models"] = models
    existing["labels"] = LABELS

    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path


def load_model_metrics(path: Path | None = None) -> dict:
    path = path or PATHS.metrics_json
    if not path.exists():
        return {"models": {}, "labels": LABELS}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"models": {}, "labels": LABELS}


__all__ = [
    "compute_metrics",
    "format_metrics",
    "load_model_metrics",
    "save_model_metrics",
]
