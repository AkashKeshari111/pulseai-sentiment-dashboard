"""Shared FastAPI dependencies: authentication and database gating."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Header, HTTPException, Query, status

from api import db
from src.config import SERVICE


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Protect write endpoints when ``API_KEY`` is configured.

    Left disabled by default so the local demo needs no ceremony; setting the
    variable in a deployed environment is what turns it on. Comparison is
    constant-time to avoid leaking the key through response timing.
    """
    if not SERVICE.api_key:
        return
    import hmac

    if not x_api_key or not hmac.compare_digest(x_api_key, SERVICE.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )


async def require_database() -> None:
    """Assert MongoDB is reachable, for handlers that need to know up front.

    Most endpoints do **not** use this: they simply touch the database and let
    :class:`api.db.DatabaseUnavailable` bubble up to the handler in ``main.py``,
    which keeps parameter validation (422) ahead of availability (503). This
    helper exists for the SSE stream, which has to refuse *before* it starts
    emitting an event stream that it could never populate.
    """
    if not db.is_connected():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "MongoDB is not connected. Set MONGODB_URI in .env and restart "
                f"the API. Last error: {db.last_error() or 'not configured'}"
            ),
        )


class FeedbackFilters:
    """Query parameters shared by the listing and every analytics endpoint."""

    def __init__(
        self,
        sentiment: str | None = Query(None, pattern="^(all|negative|neutral|positive)$"),
        source: str | None = Query(None, max_length=64),
        category: str | None = Query(None, max_length=120),
        search: str | None = Query(None, max_length=200),
        days: int | None = Query(None, ge=1, le=1825,
                                 description="Restrict to the last N days."),
        start: datetime | None = Query(None),
        end: datetime | None = Query(None),
        min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    ) -> None:
        self.sentiment = sentiment
        self.source = source
        self.category = category
        self.search = search
        self.days = days
        # Naive datetimes from the browser are interpreted as UTC so that
        # comparisons against the stored timezone-aware values are valid.
        self.start = _as_utc(start)
        self.end = _as_utc(end)
        self.min_confidence = min_confidence

    def to_query(self) -> dict:
        return db.build_filter(
            sentiment=self.sentiment,
            source=self.source,
            category=self.category,
            search=self.search,
            days=self.days,
            start=self.start,
            end=self.end,
            min_confidence=self.min_confidence,
        )


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
