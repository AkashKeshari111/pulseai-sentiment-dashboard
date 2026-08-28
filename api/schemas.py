"""Pydantic request/response models.

These double as the API contract: FastAPI turns them into the OpenAPI schema
served at ``/docs``, which is what the dashboard and any future integration
codes against.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.config import LABELS

Sentiment = Literal["negative", "neutral", "positive"]

#: Where a piece of feedback came from. Kept as a free string in the database
#: (new channels appear all the time) but suggested here for the docs.
KNOWN_SOURCES = (
    "web", "mobile_app", "email", "twitter", "play_store", "app_store",
    "survey", "call_center", "chat", "review_site",
)

MAX_TEXT_LENGTH = 5000
MAX_BATCH_SIZE = 200


class TextIn(BaseModel):
    """Bare prediction request - analyse without persisting."""

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH,
                      examples=["The delivery was late and the box arrived damaged."])
    explain: bool = Field(
        False, description="Return per-word contribution scores (slower)."
    )

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value.strip()


class BatchTextIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def clean(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values if v and v.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty text is required")
        return [v[:MAX_TEXT_LENGTH] for v in cleaned]


class FeedbackIn(BaseModel):
    """A piece of customer feedback to classify *and* store."""

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    source: str = Field("web", max_length=64, examples=list(KNOWN_SOURCES[:4]))
    product: str | None = Field(None, max_length=120)
    customer_id: str | None = Field(None, max_length=120)
    rating: int | None = Field(None, ge=1, le=5,
                               description="Optional star rating supplied by the customer.")
    #: Lets historical feedback be backfilled onto the correct trend bucket.
    created_at: datetime | None = None
    explain: bool = False

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must contain non-whitespace characters")
        return value.strip()


class FeedbackBatchIn(BaseModel):
    items: list[FeedbackIn] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionOut(BaseModel):
    label: Sentiment
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: dict[str, float]
    categories: list[str] = []
    model: str
    latency_ms: float
    explanation: list[dict] | None = None


class FeedbackOut(BaseModel):
    id: str
    text: str
    source: str
    product: str | None = None
    customer_id: str | None = None
    rating: int | None = None
    sentiment: Sentiment
    confidence: float
    scores: dict[str, float]
    categories: list[str] = []
    model: str
    created_at: str
    analyzed_at: str


class PaginatedFeedback(BaseModel):
    items: list[FeedbackOut]
    total: int
    page: int
    page_size: int
    pages: int


class SummaryOut(BaseModel):
    total: int
    counts: dict[str, int]
    distribution: dict[str, float]
    net_sentiment_score: float = Field(
        ..., description="%positive - %negative, ranging from -100 to +100."
    )
    avg_confidence: float


class TrendPoint(BaseModel):
    period: str
    total: int
    negative: int
    neutral: int
    positive: int
    net_sentiment_score: float


class BreakdownRow(BaseModel):
    key: str
    total: int
    negative: int
    neutral: int
    positive: int
    negative_rate: float


class IssueRow(BaseModel):
    category: str
    total: int
    negative: int
    positive: int
    negative_rate: float
    avg_confidence: float
    sample: str


class KeywordOut(BaseModel):
    text: str
    value: int


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    api_version: str
    database: dict
    model: dict
    labels: list[str] = list(LABELS)


class IngestResult(BaseModel):
    inserted: int
    ids: list[str]
    predictions: list[PredictionOut]
