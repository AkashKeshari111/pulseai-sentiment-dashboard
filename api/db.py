"""MongoDB Atlas integration (async, via Motor).

Holds the connection lifecycle, index management and every aggregation
pipeline the analytics endpoints depend on. Keeping the pipelines here rather
than inline in the routers means the dashboard's numbers have exactly one
definition, and they can be unit-tested without spinning up FastAPI.

Design decisions
----------------
* **Aggregation over application-side loops.** Trends, distributions and issue
  rankings are computed by the database. Pulling 100k documents into Python to
  count them would not survive a real feedback volume.
* **``created_at`` is the business timestamp**, ``analyzed_at`` is the
  processing timestamp. Keeping them separate means backfilled historical
  feedback still lands on the correct day of the trend chart.
* **Graceful degradation.** If ``MONGODB_URI`` is unset or unreachable the API
  still starts and serves ``/api/predict``; only persistence-backed endpoints
  report unavailable. That keeps the live demo working on a flaky network.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import COLLECTION_FEEDBACK, COLLECTION_RUNS, LABELS, SERVICE

logger = logging.getLogger("pulseai.db")

class DatabaseUnavailable(RuntimeError):
    """Raised when a request needs MongoDB and it is not connected.

    Raised from inside the request handler rather than from a dependency, so
    FastAPI validates the query parameters first. A client that sends
    ``days=0`` gets a 422 explaining the real problem, not a misleading 503.
    """


_client: Any = None
_database: Any = None
_connected = False
_last_error: str | None = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def connect() -> bool:
    """Open the Atlas connection and ensure indexes. Never raises."""
    global _client, _database, _connected, _last_error

    if not SERVICE.mongodb_uri or "<username>" in SERVICE.mongodb_uri:
        _last_error = "MONGODB_URI is not configured (see .env.example)"
        logger.warning("MongoDB disabled: %s", _last_error)
        return False

    try:
        from motor.motor_asyncio import AsyncIOMotorClient

        _client = AsyncIOMotorClient(
            SERVICE.mongodb_uri,
            serverSelectionTimeoutMS=8000,
            uuidRepresentation="standard",
        )
        await _client.admin.command("ping")
        _database = _client[SERVICE.mongodb_db]
        await ensure_indexes()
        _connected = True
        _last_error = None
        logger.info("connected to MongoDB database '%s'", SERVICE.mongodb_db)
        return True
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        _last_error = f"{exc.__class__.__name__}: {exc}"
        _connected = False
        logger.error("MongoDB connection failed: %s", _last_error)
        return False


async def close() -> None:
    global _client, _database, _connected
    if _client is not None:
        _client.close()
    _client, _database, _connected = None, None, False


def is_connected() -> bool:
    return _connected


def last_error() -> str | None:
    return _last_error


def get_database():
    if _database is None:
        raise DatabaseUnavailable(
            "MongoDB is not connected. Set MONGODB_URI in .env and restart the API. "
            f"Last error: {_last_error or 'not configured'}"
        )
    return _database


def feedback_collection():
    return get_database()[COLLECTION_FEEDBACK]


async def ensure_indexes() -> None:
    """Create the indexes every query path relies on (idempotent)."""
    collection = _database[COLLECTION_FEEDBACK]
    await collection.create_index([("created_at", -1)], name="created_at_desc")
    await collection.create_index([("sentiment", 1), ("created_at", -1)], name="sentiment_time")
    await collection.create_index([("source", 1)], name="source")
    await collection.create_index([("categories", 1)], name="categories")
    # Text index powers the Explorer's free-text search box.
    await collection.create_index([("text", "text")], name="text_search", default_language="english")
    await _database[COLLECTION_RUNS].create_index([("created_at", -1)], name="runs_time")
    logger.info("MongoDB indexes ensured")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def serialise(document: dict) -> dict:
    """Make a Mongo document JSON-safe (ObjectId + datetime -> str)."""
    output = dict(document)
    if "_id" in output:
        output["id"] = str(output.pop("_id"))
    for key, value in output.items():
        if isinstance(value, datetime):
            output[key] = value.isoformat()
    return output


def build_filter(
    sentiment: str | None = None,
    source: str | None = None,
    category: str | None = None,
    search: str | None = None,
    days: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    min_confidence: float | None = None,
) -> dict:
    """Compose the shared query filter used by list + analytics endpoints.

    Both surfaces must agree: if the Explorer shows 42 negative items for a
    filter, the KPI card has to show 42 too. One builder guarantees that.
    """
    query: dict[str, Any] = {}

    if sentiment and sentiment != "all":
        query["sentiment"] = sentiment
    if source and source != "all":
        query["source"] = source
    if category and category != "all":
        query["categories"] = category
    if min_confidence is not None:
        query["confidence"] = {"$gte": min_confidence}
    if search:
        query["$text"] = {"$search": search}

    time_filter: dict[str, datetime] = {}
    if days:
        time_filter["$gte"] = datetime.now(timezone.utc) - timedelta(days=days)
    if start:
        time_filter["$gte"] = start
    if end:
        time_filter["$lte"] = end
    if time_filter:
        query["created_at"] = time_filter

    return query


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

async def insert_feedback(document: dict) -> str:
    result = await feedback_collection().insert_one(document)
    return str(result.inserted_id)


async def insert_many_feedback(documents: list[dict]) -> list[str]:
    if not documents:
        return []
    result = await feedback_collection().insert_many(documents)
    return [str(i) for i in result.inserted_ids]


async def delete_feedback(feedback_id: str) -> bool:
    from bson import ObjectId
    from bson.errors import InvalidId

    try:
        oid = ObjectId(feedback_id)
    except (InvalidId, TypeError):
        return False
    result = await feedback_collection().delete_one({"_id": oid})
    return result.deleted_count == 1


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

async def list_feedback(
    query: dict, page: int = 1, page_size: int = 25, sort: str = "-created_at"
) -> dict:
    """Paginated listing with a total count for the table footer."""
    collection = feedback_collection()
    direction = -1 if sort.startswith("-") else 1
    field = sort.lstrip("-")

    skip = max(0, (page - 1) * page_size)
    cursor = (
        collection.find(query).sort(field, direction).skip(skip).limit(page_size)
    )
    items = [serialise(document) async for document in cursor]
    total = await collection.count_documents(query)

    return {
        "items": items,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, -(-total // page_size)),
    }


async def summary(query: dict) -> dict:
    """Headline KPIs: volume, sentiment mix, mean confidence, net score."""
    collection = feedback_collection()

    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": "$sentiment",
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"},
            }
        },
    ]
    rows = [row async for row in collection.aggregate(pipeline)]

    counts = dict.fromkeys(LABELS, 0)
    weighted_confidence, total = 0.0, 0
    for row in rows:
        label = row["_id"]
        if label in counts:
            counts[label] = row["count"]
        total += row["count"]
        weighted_confidence += (row.get("avg_confidence") or 0.0) * row["count"]

    positive_share = counts["positive"] / total * 100 if total else 0.0
    negative_share = counts["negative"] / total * 100 if total else 0.0

    return {
        "total": total,
        "counts": counts,
        "distribution": {
            label: round(count / total * 100, 2) if total else 0.0
            for label, count in counts.items()
        },
        # Net Sentiment Score: %positive - %negative, the standard CX metric.
        # Ranges from -100 (everyone unhappy) to +100.
        "net_sentiment_score": round(positive_share - negative_share, 2),
        "avg_confidence": round(weighted_confidence / total, 4) if total else 0.0,
    }


async def trends(query: dict, granularity: str = "day") -> list[dict]:
    """Sentiment volume over time, one row per bucket."""
    date_formats = {"hour": "%Y-%m-%dT%H:00", "day": "%Y-%m-%d", "week": "%G-W%V", "month": "%Y-%m"}
    date_format = date_formats.get(granularity, "%Y-%m-%d")

    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": {
                    "bucket": {
                        "$dateToString": {"format": date_format, "date": "$created_at"}
                    },
                    "sentiment": "$sentiment",
                },
                "count": {"$sum": 1},
            }
        },
        {
            "$group": {
                "_id": "$_id.bucket",
                "buckets": {
                    "$push": {"sentiment": "$_id.sentiment", "count": "$count"}
                },
                "total": {"$sum": "$count"},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    output: list[dict] = []
    async for row in feedback_collection().aggregate(pipeline):
        entry = {"period": row["_id"], "total": row["total"]}
        entry.update(dict.fromkeys(LABELS, 0))
        for bucket in row["buckets"]:
            if bucket["sentiment"] in entry:
                entry[bucket["sentiment"]] = bucket["count"]
        entry["net_sentiment_score"] = round(
            (entry["positive"] - entry["negative"]) / row["total"] * 100, 2
        ) if row["total"] else 0.0
        output.append(entry)
    return output


async def breakdown(query: dict, field: str, limit: int = 12) -> list[dict]:
    """Sentiment split grouped by any scalar field (source, product, ...)."""
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": {"key": f"${field}", "sentiment": "$sentiment"},
                "count": {"$sum": 1},
            }
        },
        {
            "$group": {
                "_id": "$_id.key",
                "buckets": {"$push": {"sentiment": "$_id.sentiment", "count": "$count"}},
                "total": {"$sum": "$count"},
            }
        },
        {"$sort": {"total": -1}},
        {"$limit": limit},
    ]

    output: list[dict] = []
    async for row in feedback_collection().aggregate(pipeline):
        entry = {"key": row["_id"] or "unknown", "total": row["total"]}
        entry.update(dict.fromkeys(LABELS, 0))
        for bucket in row["buckets"]:
            if bucket["sentiment"] in entry:
                entry[bucket["sentiment"]] = bucket["count"]
        entry["negative_rate"] = round(
            entry["negative"] / row["total"] * 100, 2
        ) if row["total"] else 0.0
        output.append(entry)
    return output


async def issue_ranking(query: dict, limit: int = 10) -> list[dict]:
    """Which business areas drive negative feedback.

    ``$unwind`` on the categories array turns one multi-tagged document into
    one row per category, so a review complaining about both delivery *and*
    support counts towards both - which is what an ops team wants to see.
    """
    pipeline = [
        {"$match": query},
        {"$unwind": "$categories"},
        {
            "$group": {
                "_id": "$categories",
                "total": {"$sum": 1},
                "negative": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment", "negative"]}, 1, 0]}
                },
                "positive": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment", "positive"]}, 1, 0]}
                },
                "avg_confidence": {"$avg": "$confidence"},
                "sample": {"$first": "$text"},
            }
        },
        {
            "$addFields": {
                "negative_rate": {
                    "$round": [
                        {"$multiply": [{"$divide": ["$negative", "$total"]}, 100]}, 2
                    ]
                }
            }
        },
        # Rank by absolute negative volume: a 100% negative category with 2
        # mentions is noise, 60% of 400 mentions is a real problem.
        {"$sort": {"negative": -1, "total": -1}},
        {"$limit": limit},
    ]

    output: list[dict] = []
    async for row in feedback_collection().aggregate(pipeline):
        output.append(
            {
                "category": row["_id"],
                "total": row["total"],
                "negative": row["negative"],
                "positive": row["positive"],
                "negative_rate": row.get("negative_rate", 0.0),
                "avg_confidence": round(row.get("avg_confidence") or 0.0, 4),
                "sample": (row.get("sample") or "")[:240],
            }
        )
    return output


async def keyword_cloud(query: dict, sentiment: str | None = None, limit: int = 40) -> list[dict]:
    """Top content words, computed from a capped sample of matching docs."""
    from src.preprocessing import extract_keywords

    scoped = dict(query)
    if sentiment and sentiment != "all":
        scoped["sentiment"] = sentiment

    cursor = (
        feedback_collection()
        .find(scoped, {"text": 1})
        .sort("created_at", -1)
        .limit(2000)  # bounded scan keeps this endpoint fast at any volume
    )
    texts = [document.get("text", "") async for document in cursor]
    return [
        {"text": word, "value": count}
        for word, count in extract_keywords(texts, top_n=limit)
    ]


async def distinct_values(field: str) -> list[str]:
    values = await feedback_collection().distinct(field)
    return sorted(str(v) for v in values if v)


async def recent(query: dict, limit: int = 10) -> list[dict]:
    cursor = feedback_collection().find(query).sort("created_at", -1).limit(limit)
    return [serialise(document) async for document in cursor]


async def count_documents(query: dict | None = None) -> int:
    return await feedback_collection().count_documents(query or {})


async def latest_timestamp() -> datetime | None:
    document = await feedback_collection().find_one({}, sort=[("analyzed_at", -1)])
    return document.get("analyzed_at") if document else None
