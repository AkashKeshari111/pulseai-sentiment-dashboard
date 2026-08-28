"""Classical baseline: TF-IDF features + Logistic Regression.

Why a baseline at all? Because "we fine-tuned BERT and got 0.87 F1" is not a
result - it is a number without a reference point. The baseline answers the
question a reviewer will actually ask: *how much did the transformer buy us
over a model that trains in ten seconds?* It also acts as a fallback
classifier and as a sanity check on the data pipeline (if the baseline scores
at chance level, the labels are wrong, not the model).

Design notes
------------
* Word 1-2 grams **and** character 3-5 grams are unioned. Character n-grams
  absorb typos and morphology that a stemmer misses, which is common in raw
  customer feedback.
* ``class_weight="balanced"`` counteracts the neutral class being the minority
  in most review corpora.
* The hyper-parameter search is small and explicit rather than a giant grid:
  it must stay fast enough that anyone can re-run the whole notebook.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import LABELS, PATHS, TRAINING
from src.dataset import load_splits
from src.metrics import compute_metrics, format_metrics, save_model_metrics
from src.preprocessing import clean_for_classical


def build_pipeline(C: float = 4.0, max_features: int = 60_000):
    """TF-IDF (word + char) -> Logistic Regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        max_features=max_features,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=max_features,
    )
    features = FeatureUnion(
        [("word", word_vec), ("char", char_vec)], transformer_weights={"word": 1.0, "char": 0.6}
    )
    classifier = LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
        random_state=TRAINING.seed,
    )
    return Pipeline([("features", features), ("clf", classifier)])


def _prepare_texts(frame: pd.DataFrame) -> list[str]:
    return [clean_for_classical(text) for text in frame["text"].astype(str)]


def top_features_per_class(pipeline, top_n: int = 15) -> dict[str, list[tuple[str, float]]]:
    """Most influential word n-grams per class.

    This is the baseline's built-in explainability: the learned coefficients
    are directly readable, which is useful evidence that the model latched
    onto sentiment words rather than dataset artefacts.
    """
    features = pipeline.named_steps["features"]
    word_vec = dict(features.transformer_list)["word"]
    names = np.asarray(word_vec.get_feature_names_out())
    n_word = len(names)

    coefs = pipeline.named_steps["clf"].coef_
    output: dict[str, list[tuple[str, float]]] = {}
    for class_index, label in enumerate(LABELS):
        # Only the word-n-gram block is human readable; char n-grams are not.
        weights = coefs[class_index][:n_word]
        order = np.argsort(weights)[::-1][:top_n]
        output[label] = [(str(names[i]), float(weights[i])) for i in order]
    return output


def train(
    tune: bool = True, save_path: Path | None = None, verbose: bool = True
) -> dict:
    """Fit, evaluate on validation + test, persist the model and metrics."""
    splits = load_splits()
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

    if verbose:
        print(f"[baseline] train={len(train_df)} val={len(val_df)} test={len(test_df)}")
        print("[baseline] cleaning text (aggressive profile)...")

    X_train, y_train = _prepare_texts(train_df), train_df["label"].to_numpy()
    X_val, y_val = _prepare_texts(val_df), val_df["label"].to_numpy()
    X_test, y_test = _prepare_texts(test_df), test_df["label"].to_numpy()

    best_pipeline, best_score, best_C = None, -1.0, None
    candidates = [1.0, 4.0, 10.0] if tune else [4.0]

    from sklearn.metrics import f1_score

    for C in candidates:
        started = time.perf_counter()
        pipeline = build_pipeline(C=C)
        pipeline.fit(X_train, y_train)
        score = f1_score(y_val, pipeline.predict(X_val), average="macro", zero_division=0)
        if verbose:
            print(
                f"[baseline] C={C:<5} val macro-F1={score:.4f} "
                f"({time.perf_counter() - started:.1f}s)"
            )
        if score > best_score:
            best_pipeline, best_score, best_C = pipeline, score, C

    assert best_pipeline is not None
    started = time.perf_counter()
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)
    latency_ms = (time.perf_counter() - started) / max(len(X_test), 1) * 1000

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics.update(
        {
            "model": "TF-IDF + Logistic Regression",
            "model_key": "baseline",
            "best_C": best_C,
            "val_f1_macro": float(best_score),
            "latency_ms_per_sample": round(latency_ms, 3),
            "n_features": int(
                best_pipeline.named_steps["clf"].coef_.shape[1]
            ),
            "top_features": top_features_per_class(best_pipeline),
        }
    )

    save_path = save_path or PATHS.baseline_model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, save_path)
    save_model_metrics("baseline", metrics)

    if verbose:
        print(format_metrics("BASELINE - TF-IDF + Logistic Regression (test set)", metrics))
        print(f"[baseline] model saved to {save_path}")
        print(f"[baseline] metrics appended to {PATHS.metrics_json}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the TF-IDF baseline.")
    parser.add_argument("--no-tune", action="store_true", help="skip the C sweep")
    args = parser.parse_args()
    train(tune=not args.no_tune)


if __name__ == "__main__":
    main()
