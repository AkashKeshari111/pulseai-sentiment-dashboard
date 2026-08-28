# Model artefacts

This directory holds trained models. Its contents are **gitignored** — checkpoints are
build outputs, not source, and a DistilBERT checkpoint is ~260 MB.

## What lands here

| Path | Produced by | Size |
|---|---|---|
| `distilbert-sentiment/` | `python -m src.train_transformer` | ~260 MB |
| `baseline_tfidf_logreg.joblib` | `python -m src.train_baseline` | ~30 MB |

## Regenerating them

```bash
python -m src.dataset --prepare      # build the train/val/test splits
python -m src.train_baseline         # ~2 minutes
python -m src.train_transformer      # ~2 hours on CPU, minutes on GPU
```

Evaluation results are written to `reports/metrics.json`, which the dashboard's Model Card
page reads at runtime.

## If this directory is empty

The API still starts. `api/inference.py` falls through to the next available backend and
reports which one it is using via `/health` and `/api/model/info`:

1. fine-tuned checkpoint here
2. `FALLBACK_MODEL` from the Hugging Face Hub
3. the TF-IDF baseline
4. a built-in negation-aware lexicon
