"""Central configuration for PulseAI.

Every tunable lives here so that the notebook, the training scripts and the
FastAPI service all agree on paths, label ordering and hyper-parameters.
Values are read from the environment (``.env``) with sensible defaults, so the
project runs out of the box and is still 12-factor friendly for deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Project root = one level above this file's package directory.
ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
# The index order is part of the model contract: it must never change once a
# checkpoint has been trained, otherwise the softmax outputs get mislabelled.
LABELS: list[str] = ["negative", "neutral", "positive"]
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}
NUM_LABELS: int = len(LABELS)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. All paths are absolute to keep CWD irrelevant."""

    root: Path = ROOT_DIR
    data_raw: Path = ROOT_DIR / "data" / "raw"
    data_processed: Path = ROOT_DIR / "data" / "processed"
    models: Path = ROOT_DIR / "models"
    reports: Path = ROOT_DIR / "reports"
    figures: Path = ROOT_DIR / "reports" / "figures"

    @property
    def train_csv(self) -> Path:
        return self.data_processed / "train.csv"

    @property
    def val_csv(self) -> Path:
        return self.data_processed / "val.csv"

    @property
    def test_csv(self) -> Path:
        return self.data_processed / "test.csv"

    @property
    def baseline_model(self) -> Path:
        return self.models / "baseline_tfidf_logreg.joblib"

    @property
    def transformer_dir(self) -> Path:
        return Path(os.getenv("MODEL_DIR", self.models / "distilbert-sentiment"))

    @property
    def metrics_json(self) -> Path:
        """Consumed by the dashboard's Model Card page."""
        return self.reports / "metrics.json"

    def ensure(self) -> None:
        for directory in (
            self.data_raw,
            self.data_processed,
            self.models,
            self.reports,
            self.figures,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainingConfig:
    """Hyper-parameters for both the baseline and the transformer."""

    dataset: str = field(default_factory=lambda: os.getenv("DATASET", "yelp_review_full"))
    base_checkpoint: str = "distilbert-base-uncased"

    train_size: int = field(default_factory=lambda: _env_int("TRAIN_SIZE", 12_000))
    val_size: int = field(default_factory=lambda: _env_int("VAL_SIZE", 2_000))
    test_size: int = field(default_factory=lambda: _env_int("TEST_SIZE", 2_000))

    max_seq_length: int = field(default_factory=lambda: _env_int("MAX_SEQ_LENGTH", 128))
    epochs: int = field(default_factory=lambda: _env_int("EPOCHS", 2))
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 16))
    eval_batch_size: int = field(default_factory=lambda: _env_int("EVAL_BATCH_SIZE", 64))
    learning_rate: float = field(default_factory=lambda: _env_float("LEARNING_RATE", 3e-5))
    weight_decay: float = field(default_factory=lambda: _env_float("WEIGHT_DECAY", 0.01))
    warmup_ratio: float = field(default_factory=lambda: _env_float("WARMUP_RATIO", 0.1))
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))


@dataclass(frozen=True)
class ServiceConfig:
    """Runtime settings for the FastAPI service."""

    mongodb_uri: str = field(default_factory=lambda: os.getenv("MONGODB_URI", ""))
    mongodb_db: str = field(default_factory=lambda: os.getenv("MONGODB_DB", "pulseai"))
    fallback_model: str = field(
        default_factory=lambda: os.getenv(
            "FALLBACK_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    )
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", "").strip())

    @property
    def cors_origins(self) -> list[str]:
        raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


PATHS = Paths()
TRAINING = TrainingConfig()
SERVICE = ServiceConfig()

# Collections used in MongoDB.
COLLECTION_FEEDBACK = "feedback"
COLLECTION_RUNS = "model_runs"
