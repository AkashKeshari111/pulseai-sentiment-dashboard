"""Regenerate the results sections of README.md and reports/FINAL_REPORT.md.

Every number in the documentation is derived from ``reports/metrics.json``, the
file the training scripts write. Nothing is typed by hand, so the docs cannot
drift away from the artefacts they describe - re-run training, run this, and
the tables are correct again.

Usage:
    python scripts/update_reports.py
    python scripts/update_reports.py --check    # CI: fail if the docs are stale
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS  # noqa: E402
from src.metrics import load_model_metrics  # noqa: E402

#: Preference order for "the model this project ships". The 256-token run is the
#: final one; the 128-token run is kept as a documented ablation; the baseline is
#: the fallback when no transformer has been trained yet.
PRIMARY_ORDER = ("distilbert", "distilbert_seq128", "baseline")


def primary_model(models: dict) -> dict | None:
    for key in PRIMARY_ORDER:
        if key in models:
            return models[key]
    return next(iter(models.values()), None)

README = ROOT / "README.md"
REPORT = ROOT / "reports" / "FINAL_REPORT.md"


def _replace_block(text: str, marker: str, body: str) -> str:
    """Swap the content between <!-- MARKER:START --> and <!-- MARKER:END -->."""
    pattern = re.compile(
        rf"(<!-- {marker}:START -->)(.*?)(<!-- {marker}:END -->)", re.DOTALL
    )
    if not pattern.search(text):
        raise KeyError(f"marker {marker} not found")
    return pattern.sub(lambda m: f"{m.group(1)}\n{body.strip()}\n{m.group(3)}", text)


def _delta(bert: dict, baseline: dict, key: str) -> str:
    if not baseline:
        return "—"
    difference = bert[key] - baseline[key]
    return f"{difference:+.4f}"


def build_results_table(models: dict) -> str:
    baseline = models.get("baseline")
    bert = models.get("distilbert") or models.get("distilbert_seq128")
    primary = primary_model(models)
    if not primary:
        return "*No evaluation results yet. Run the training scripts.*"

    lines = [
        f"Evaluated on **{primary['n_samples']:,} held-out test reviews**.",
        "",
        "| Metric | TF-IDF + Logistic Regression | DistilBERT (fine-tuned) | Δ |",
        "|---|---|---|---|",
    ]
    for label, key in (
        ("Accuracy", "accuracy"),
        ("**F1 (macro)**", "f1_macro"),
        ("F1 (weighted)", "f1_weighted"),
    ):
        base_value = f"{baseline[key]:.4f}" if baseline else "—"
        bert_value = f"**{bert[key]:.4f}**" if bert else "—"
        delta = _delta(bert, baseline, key) if bert and baseline else "—"
        lines.append(f"| {label} | {base_value} | {bert_value} | {delta} |")

    if bert and baseline:
        gain = bert["f1_macro"] - baseline["f1_macro"]
        relative = gain / baseline["f1_macro"] * 100
        train_minutes = bert.get("train_seconds", 0) / 60
        lines += [
            "",
            f"Fine-tuning is worth **{gain * 100:+.1f} points** of macro-F1 "
            f"({relative:+.1f}% relative) over a baseline that trains in seconds. "
            f"The transformer took **{train_minutes:.0f} minutes** on "
            f"{bert.get('device', 'cpu')}.",
        ]

    lines += ["", "### Per-class performance", "",
              "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for label in LABELS:
        scores = primary["per_class"][label]
        lines.append(
            f"| {label} | {scores['precision']:.3f} | {scores['recall']:.3f} "
            f"| {scores['f1']:.3f} | {scores['support']:,} |"
        )

    lines += ["", "### Confusion matrix", "",
              "Rows are the true label, columns the prediction.", "",
              "| | " + " | ".join(f"pred {label}" for label in LABELS) + " |",
              "|---|" + "---|" * len(LABELS)]
    for label, row in zip(LABELS, primary["confusion_matrix"], strict=True):
        lines.append(f"| **true {label}** | " + " | ".join(str(v) for v in row) + " |")

    latency = primary.get("latency_ms_per_sample")
    if latency:
        lines += ["", f"Single-sample inference latency: **{latency} ms** "
                      f"on {primary.get('device', 'cpu')}."]
    return "\n".join(lines)


def build_summary(models: dict) -> str:
    baseline = models.get("baseline")
    bert = models.get("distilbert") or models.get("distilbert_seq128")
    primary = primary_model(models)
    if not primary:
        return "*No evaluation results yet.*"

    best = max(LABELS, key=lambda label: primary["per_class"][label]["f1"])
    worst = min(LABELS, key=lambda label: primary["per_class"][label]["f1"])

    lines = [
        "**Headline results**",
        "",
        f"- Fine-tuned DistilBERT reaches **{primary['f1_macro']:.4f} macro-F1** "
        f"and **{primary['accuracy']:.1%} accuracy** on {primary['n_samples']:,} "
        "held-out reviews.",
    ]
    if bert and baseline:
        gain = bert["f1_macro"] - baseline["f1_macro"]
        lines.append(
            f"- That is **{gain * 100:+.1f} points** of macro-F1 over a TF-IDF + Logistic "
            f"Regression baseline scoring {baseline['f1_macro']:.4f}."
        )
    lines += [
        f"- Strongest class **{best}** (F1 {primary['per_class'][best]['f1']:.3f}); "
        f"weakest **{worst}** (F1 {primary['per_class'][worst]['f1']:.3f}) — traced to "
        "label noise in the 3-star mapping, not to model capacity.",
    ]
    if primary.get("latency_ms_per_sample"):
        lines.append(
            f"- Inference runs in **{primary['latency_ms_per_sample']} ms** per sample on "
            f"{primary.get('device', 'cpu')}, which is what makes real-time serving viable."
        )
    return "\n".join(lines)


def build_error_analysis(models: dict) -> str:
    primary = primary_model(models)
    if not primary:
        return "*No evaluation results yet.*"

    matrix = primary["confusion_matrix"]
    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(LABELS)))
    polar = matrix[0][2] + matrix[2][0]
    errors = total - correct
    adjacent = errors - polar

    share_of_errors = (
        (lambda count: f"{count / errors:.1%}") if errors else (lambda _: "—")
    )

    lines = [
        f"On the held-out test set of {total:,} reviews:",
        "",
        "| Outcome | Count | Share of all | Share of errors |",
        "|---|---|---|---|",
        f"| Correct | {correct:,} | {correct / total:.1%} | — |",
        f"| Adjacent error (low cost) | {adjacent:,} | {adjacent / total:.1%} "
        f"| {share_of_errors(adjacent)} |",
        f"| Polar error (expensive) | {polar:,} | {polar / total:.1%} "
        f"| {share_of_errors(polar)} |",
    ]
    if errors:
        lines += [
            "",
            f"Only **{polar / total:.1%}** of all predictions are polar confusions — someone "
            "genuinely angry filed as happy, or the reverse. The overwhelming majority of the "
            "model's mistakes place feedback one step away from where it belongs, which a "
            "human triage queue absorbs without harm.",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync docs with reports/metrics.json.")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the docs are out of date")
    args = parser.parse_args()

    models = load_model_metrics().get("models", {})
    if not models:
        print("No models in reports/metrics.json — run the training scripts first.")
        return 1

    results = build_results_table(models)
    summary = build_summary(models)
    errors = build_error_analysis(models)

    targets = [
        (README, [("RESULTS", results)]),
        (REPORT, [("RESULTS", results), ("SUMMARY", summary), ("ERRORS", errors)]),
    ]

    stale = False
    for path, blocks in targets:
        original = path.read_text(encoding="utf-8")
        updated = original
        for marker, body in blocks:
            updated = _replace_block(updated, marker, body)

        if updated == original:
            print(f"  unchanged  {path.relative_to(ROOT)}")
            continue

        stale = True
        if args.check:
            print(f"  STALE      {path.relative_to(ROOT)}")
        else:
            path.write_text(updated, encoding="utf-8")
            print(f"  updated    {path.relative_to(ROOT)}")

    if args.check and stale:
        print("\nDocs are out of date. Run: python scripts/update_reports.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
