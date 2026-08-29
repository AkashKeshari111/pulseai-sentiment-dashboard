"""Sentiment inference engine used by the API.

Responsibilities
----------------
* Load the best model available, degrading gracefully:
  1. a quantized ONNX model (``ONNX_MODEL_DIR`` / ``ONNX_MODEL_REPO``) - the
     smallest and fastest path, and the only one that fits a 512 MB container
  2. the fine-tuned DistilBERT checkpoint in ``MODEL_DIR``
  3. a pretrained sentiment model from the Hub (``FALLBACK_MODEL``)
  4. the TF-IDF baseline saved by ``src.train_baseline``
  5. a transparent lexicon heuristic, so the API *always* answers
* Normalise whatever label space the loaded model uses onto our canonical
  negative/neutral/positive ordering.
* Serve single and batched predictions with a token-level explanation.

The model is loaded once at process start and reused. Both the torch and ONNX
paths pin their thread counts, so a small CPU container stays predictable when
uvicorn serves several requests at once.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from src.config import LABELS, PATHS, SERVICE
from src.preprocessing import clean_for_transformer, detect_issue_categories, tokenize

logger = logging.getLogger("pulseai.inference")

#: Inputs longer than this are truncated. Must match the value the model was
#: fine-tuned with, otherwise the positional statistics shift at serving time.
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "128"))

# Some checkpoints publish labels as LABEL_0 / POS / positive / 5 stars. This
# maps every variant we care about onto our canonical names.
_LABEL_ALIASES = {
    "negative": "negative", "neg": "negative", "label_0": "negative",
    "0": "negative", "1 star": "negative", "2 stars": "negative",
    "neutral": "neutral", "neu": "neutral", "label_1": "neutral",
    "1": "neutral", "3 stars": "neutral",
    "positive": "positive", "pos": "positive", "label_2": "positive",
    "2": "positive", "4 stars": "positive", "5 stars": "positive",
}

#: Fallback lexicon. Small on purpose - it exists so the service degrades to
#: something explainable rather than crashing, not to compete with the model.
_POSITIVE_LEXICON = {
    "good", "great", "excellent", "amazing", "love", "loved", "perfect", "best",
    "wonderful", "fantastic", "happy", "awesome", "recommend", "smooth", "fast",
    "friendly", "helpful", "quality", "worth", "brilliant", "satisfied",
}
_NEGATIVE_LEXICON = {
    "bad", "terrible", "awful", "worst", "hate", "poor", "broken", "damaged",
    "slow", "rude", "disappointed", "disappointing", "refund", "useless",
    "late", "delay", "defective", "overpriced", "unacceptable", "never",
}
_NEGATORS = {"not", "no", "never", "cannot", "without", "hardly", "barely"}


@dataclass
class Prediction:
    """One classified piece of feedback."""

    label: str
    confidence: float
    scores: dict[str, float]
    categories: list[str]
    model: str
    latency_ms: float
    explanation: list[dict] | None = None

    def to_dict(self) -> dict:
        payload = {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "categories": self.categories,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.explanation is not None:
            payload["explanation"] = self.explanation
        return payload


class SentimentEngine:
    """Thread-safe, lazily initialised sentiment classifier."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = False
        self.backend: str = "uninitialised"
        self.model_name: str = "none"
        self.model_source: str = "none"
        self._model = None
        self._tokenizer = None
        self._pipeline = None  # sklearn baseline
        self._index_to_label: dict[int, str] = {}
        self._torch = None
        self._np = None
        self._session = None          # onnxruntime InferenceSession
        self._onnx_inputs: set[str] = set()
        self._onnx_origin = "local"
        self._onnx_label = ""
        self._device = "cpu"

    # -- loading ------------------------------------------------------------

    def load(self) -> None:
        """Resolve and load the best available backend (idempotent)."""
        with self._lock:
            if self._ready:
                return
            for loader in (
                self._load_onnx,
                self._load_finetuned,
                self._load_hub_fallback,
                self._load_baseline,
            ):
                try:
                    if loader():
                        self._ready = True
                        logger.info(
                            "sentiment engine ready: backend=%s model=%s",
                            self.backend, self.model_name,
                        )
                        return
                except Exception:  # noqa: BLE001 - try the next backend
                    logger.exception("backend %s failed to load", loader.__name__)

            self.backend = "lexicon"
            self.model_name = "rule-based lexicon"
            self.model_source = "builtin"
            self._ready = True
            logger.warning(
                "No trained model found. Falling back to the lexicon heuristic - "
                "run `python -m src.train_transformer` for real predictions."
            )

    def _load_onnx(self) -> bool:
        """Quantized ONNX model, served without PyTorch.

        Tried first because it is strictly better where it is available: the
        same weights at INT8 precision, measured at 141 MB resident against
        601 MB for the PyTorch path, and ~3.4x faster on CPU for a 0.03%
        macro-F1 difference. That is what makes a 512 MB free container viable.

        Deliberately imports **only** ``onnxruntime`` and ``tokenizers`` - not
        ``optimum``, which pulls in torch and transformers and puts the memory
        saving straight back (measured: 512 MB, i.e. no gain at all).
        """
        directory = self._resolve_onnx_dir()
        if directory is None:
            return False

        model_file = next(
            (p for p in sorted(directory.glob("*.onnx")) if "quantized" in p.name),
            next(iter(sorted(directory.glob("*.onnx"))), None),
        )
        tokenizer_file = directory / "tokenizer.json"
        if model_file is None or not tokenizer_file.exists():
            logger.info("no usable ONNX model in %s", directory)
            return False

        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        options = ort.SessionOptions()
        # One intra-op thread: inference is already parallel across concurrent
        # requests, and letting each session grab every core makes p99 worse.
        options.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(model_file), options, providers=["CPUExecutionProvider"]
        )
        self._onnx_inputs = {i.name for i in self._session.get_inputs()}

        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self._tokenizer = tokenizer

        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self._index_to_label = {
            int(index): _LABEL_ALIASES.get(str(name).strip().lower(), str(name).lower())
            for index, name in (config.get("id2label") or {}).items()
        } or dict(enumerate(LABELS))

        self.backend = "onnx"
        # snapshot_download returns a content-hashed path, so name the model
        # after the repo it came from rather than the cache directory.
        self.model_name = f"DistilBERT INT8 (ONNX) - {self._onnx_label}"
        self.model_source = self._onnx_origin
        self._device = "cpu (onnxruntime)"
        return True

    def _resolve_onnx_dir(self) -> Path | None:
        """Local directory if present, otherwise pull one from the Hub.

        A 65 MB artefact is small enough to fetch at boot, which keeps it out of
        git while still letting a container start with nothing baked in.
        """
        local = Path(os.getenv("ONNX_MODEL_DIR", "models/distilbert-sentiment-onnx-int8"))
        if (local / "config.json").exists():
            self._onnx_origin = "local"
            self._onnx_label = local.name
            return local

        repo = os.getenv("ONNX_MODEL_REPO", "").strip()
        if not repo:
            return None

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.info("ONNX_MODEL_REPO set but huggingface_hub is not installed")
            return None

        logger.info("downloading ONNX model from the Hub: %s", repo)
        path = Path(
            snapshot_download(
                repo_id=repo,
                allow_patterns=["*.onnx", "*.json", "*.txt"],
                cache_dir=os.getenv("HF_HOME"),
            )
        )
        self._onnx_origin = "huggingface-hub"
        self._onnx_label = repo
        return path

    def _load_transformer(self, source: str, label: str, origin: str) -> bool:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        # One inference thread per request keeps p99 latency stable when
        # uvicorn serves several requests at once on a small container.
        torch.set_num_threads(max(1, (torch.get_num_threads() or 2) // 2))

        self._tokenizer = AutoTokenizer.from_pretrained(source)
        self._model = AutoModelForSequenceClassification.from_pretrained(source)
        self._model.eval()

        device = SERVICE.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model.to(device)

        self._index_to_label = self._resolve_label_map(self._model.config)
        self.backend = "transformer"
        self.model_name = label
        self.model_source = origin
        return True

    def _load_finetuned(self) -> bool:
        directory = Path(PATHS.transformer_dir)
        if not (directory / "config.json").exists():
            logger.info("no fine-tuned checkpoint at %s", directory)
            return False
        return self._load_transformer(
            str(directory), f"DistilBERT (fine-tuned) - {directory.name}", "local"
        )

    def _load_hub_fallback(self) -> bool:
        name = SERVICE.fallback_model
        if not name:
            return False
        logger.info("loading fallback model from the Hub: %s", name)
        return self._load_transformer(name, name, "huggingface-hub")

    def _load_baseline(self) -> bool:
        path = PATHS.baseline_model
        if not path.exists():
            return False
        import joblib

        self._pipeline = joblib.load(path)
        self._index_to_label = dict(enumerate(LABELS))
        self.backend = "sklearn"
        self.model_name = "TF-IDF + Logistic Regression (baseline)"
        self.model_source = "local"
        return True

    @staticmethod
    def _resolve_label_map(config) -> dict[int, str]:
        """Translate a checkpoint's id2label onto our canonical label names."""
        raw = getattr(config, "id2label", None) or {}
        resolved: dict[int, str] = {}
        for index, name in raw.items():
            key = str(name).strip().lower()
            resolved[int(index)] = _LABEL_ALIASES.get(key, key)

        if set(resolved.values()) >= set(LABELS):
            return resolved

        logger.warning(
            "checkpoint labels %s are not recognised; assuming canonical order %s",
            list(raw.values()), LABELS,
        )
        return dict(enumerate(LABELS))

    # -- prediction ---------------------------------------------------------

    def predict(self, text: str, explain: bool = False) -> Prediction:
        return self.predict_batch([text], explain=explain)[0]

    def predict_batch(self, texts: list[str], explain: bool = False) -> list[Prediction]:
        if not self._ready:
            self.load()

        started = time.perf_counter()

        # Each backend cleans the *raw* text with the same profile it was
        # trained on. Pre-cleaning once for everyone would feed the sklearn
        # model text that had already been through the transformer profile -
        # a train/serve mismatch, and one that would be invisible in testing.
        if self.backend == "onnx":
            score_rows = self._scores_onnx(
                [clean_for_transformer(t) for t in texts]
            )
        elif self.backend == "transformer":
            score_rows = self._scores_transformer(
                [clean_for_transformer(t) for t in texts]
            )
        elif self.backend == "sklearn":
            score_rows = self._scores_sklearn(texts)
        else:
            score_rows = [
                self._scores_lexicon(clean_for_transformer(t)) for t in texts
            ]

        elapsed_ms = (time.perf_counter() - started) / max(len(texts), 1) * 1000

        predictions: list[Prediction] = []
        for original, scores in zip(texts, score_rows, strict=True):
            label = max(scores, key=scores.get)
            predictions.append(
                Prediction(
                    label=label,
                    confidence=scores[label],
                    scores=scores,
                    categories=detect_issue_categories(original),
                    model=self.model_name,
                    latency_ms=elapsed_ms,
                    explanation=self.explain(original, label) if explain else None,
                )
            )
        return predictions

    def _scores_transformer(self, texts: list[str]) -> list[dict[str, float]]:
        torch = self._torch
        encoded = self._tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            logits = self._model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        return [self._row_to_scores(row) for row in probabilities]

    def _scores_onnx(self, texts: list[str]) -> list[dict[str, float]]:
        """Batched ONNX inference. Softmax is done in numpy - no torch needed."""
        np = self._np
        encodings = self._tokenizer.encode_batch(texts)

        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array(
                [e.attention_mask for e in encodings], dtype=np.int64
            ),
        }
        # Some exports also expect token_type_ids; DistilBERT does not, so only
        # send what this particular graph declares.
        feed = {k: v for k, v in feed.items() if k in self._onnx_inputs}

        logits = self._session.run(None, feed)[0]
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities = exponentiated / exponentiated.sum(axis=-1, keepdims=True)
        return [self._row_to_scores(row) for row in probabilities]

    def _scores_sklearn(self, texts: list[str]) -> list[dict[str, float]]:
        """Takes raw text: the baseline was fitted on the aggressive profile."""
        from src.preprocessing import clean_for_classical

        prepared = [clean_for_classical(t) for t in texts]
        probabilities = self._pipeline.predict_proba(prepared)
        return [self._row_to_scores(row) for row in probabilities]

    def _row_to_scores(self, row) -> dict[str, float]:
        """Fold a raw probability vector into our 3 canonical labels.

        Checkpoints with a wider label space (e.g. 5-star models) collapse by
        summing the probability mass of every index mapped to the same label.
        """
        scores = dict.fromkeys(LABELS, 0.0)
        for index, probability in enumerate(row):
            label = self._index_to_label.get(index)
            if label in scores:
                scores[label] += float(probability)
        total = sum(scores.values()) or 1.0
        return {label: value / total for label, value in scores.items()}

    @staticmethod
    def _scores_lexicon(text: str) -> dict[str, float]:
        """Transparent negation-aware lexicon scorer (last-resort backend)."""
        tokens = tokenize(text)
        score = 0.0
        for i, token in enumerate(tokens):
            polarity = 0.0
            if token in _POSITIVE_LEXICON:
                polarity = 1.0
            elif token in _NEGATIVE_LEXICON:
                polarity = -1.0
            if polarity and any(t in _NEGATORS for t in tokens[max(0, i - 3) : i]):
                polarity *= -1.0
            score += polarity

        # Squash the raw count into a bounded neutral-centred distribution.
        magnitude = min(abs(score) / 3.0, 1.0)
        neutral = max(0.15, 1.0 - magnitude)
        positive = magnitude if score > 0 else 0.0
        negative = magnitude if score < 0 else 0.0
        total = neutral + positive + negative
        return {
            "negative": negative / total,
            "neutral": neutral / total,
            "positive": positive / total,
        }

    # -- explainability -----------------------------------------------------

    def explain(self, text: str, label: str | None = None, max_tokens: int = 40) -> list[dict]:
        """Per-word contribution via leave-one-out occlusion.

        Each word is deleted in turn and the drop in the predicted class
        probability is that word's attribution. It is model-agnostic (works
        for every backend), needs no gradients, and is trivially explainable
        to a non-ML stakeholder - which is exactly what a business dashboard
        needs next to an automated decision.
        """
        words = re.findall(r"\S+", text)[:max_tokens]
        if not words:
            return []

        base = self.predict_batch([text])[0]
        target = label or base.label
        baseline_score = base.scores[target]

        variants = [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
        variant_predictions = self.predict_batch(variants)

        contributions = [
            {
                "token": word,
                "weight": round(baseline_score - prediction.scores[target], 4),
            }
            for word, prediction in zip(words, variant_predictions, strict=True)
        ]
        peak = max((abs(c["weight"]) for c in contributions), default=0.0) or 1.0
        for contribution in contributions:
            contribution["normalised"] = round(contribution["weight"] / peak, 4)
        return contributions

    # -- introspection ------------------------------------------------------

    def info(self) -> dict:
        if not self._ready:
            self.load()
        return {
            "backend": self.backend,
            "model": self.model_name,
            "source": self.model_source,
            "device": self._device,
            "labels": LABELS,
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "explainability": "leave-one-out occlusion",
        }


#: Process-wide singleton. The FastAPI lifespan hook warms it at startup so the
#: first user request does not pay the model-loading cost.
engine = SentimentEngine()
