"""Analytics endpoints backing the dashboard.

Every route accepts the same ``FeedbackFilters`` dependency, so a filter set
in the dashboard header (date range, channel, search term) applies
consistently to the KPI cards, the trend chart and the issue table at once.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api import db
from api.deps import FeedbackFilters
from api.schemas import BreakdownRow, IssueRow, KeywordOut, SummaryOut, TrendPoint

# No router-level database dependency: a missing database is reported by the
# DatabaseUnavailable handler in main.py, which runs *after* query-parameter
# validation so bad input still gets an accurate 422.
router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary", response_model=SummaryOut, summary="Headline KPIs")
async def summary(filters: FeedbackFilters = Depends()) -> dict:
    return await db.summary(filters.to_query())


@router.get("/trends", response_model=list[TrendPoint], summary="Sentiment over time")
async def trends(
    filters: FeedbackFilters = Depends(),
    granularity: str = Query("day", pattern="^(hour|day|week|month)$"),
) -> list[dict]:
    """Time series of volume and Net Sentiment Score per bucket.

    Trend direction is the signal a CX team acts on: a stable 20% negative
    rate is business as usual, the same rate doubling in a week is an incident.
    """
    return await db.trends(filters.to_query(), granularity=granularity)


@router.get("/sources", response_model=list[BreakdownRow], summary="Split by channel")
async def sources(filters: FeedbackFilters = Depends()) -> list[dict]:
    return await db.breakdown(filters.to_query(), field="source")


@router.get("/products", response_model=list[BreakdownRow], summary="Split by product")
async def products(filters: FeedbackFilters = Depends()) -> list[dict]:
    return await db.breakdown(filters.to_query(), field="product")


@router.get("/issues", response_model=list[IssueRow], summary="Top complaint drivers")
async def issues(
    filters: FeedbackFilters = Depends(),
    limit: int = Query(10, ge=1, le=50),
) -> list[dict]:
    """Business areas ranked by how much negative feedback they attract."""
    return await db.issue_ranking(filters.to_query(), limit=limit)


@router.get("/keywords", response_model=list[KeywordOut], summary="Word cloud data")
async def keywords(
    filters: FeedbackFilters = Depends(),
    sentiment: str | None = Query(None, pattern="^(all|negative|neutral|positive)$"),
    limit: int = Query(40, ge=5, le=150),
) -> list[dict]:
    return await db.keyword_cloud(filters.to_query(), sentiment=sentiment, limit=limit)


@router.get("/recent", summary="Most recent feedback")
async def recent(
    filters: FeedbackFilters = Depends(),
    limit: int = Query(10, ge=1, le=50),
) -> list[dict]:
    return await db.recent(filters.to_query(), limit=limit)


@router.get("/filters", summary="Available filter values")
async def filter_options() -> dict:
    """Distinct sources, products and categories, for the dashboard dropdowns.

    Derived from the data rather than hard-coded, so a newly ingested channel
    shows up in the UI without a code change.
    """
    return {
        "sources": await db.distinct_values("source"),
        "products": await db.distinct_values("product"),
        "categories": await db.distinct_values("categories"),
    }
