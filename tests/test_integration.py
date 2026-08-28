"""Integration tests that need a live MongoDB.

Skipped unless ``MONGODB_TEST_URI`` is set, so the default suite and CI stay
fast and dependency-free.

Run them against a throwaway database:

    # replica set (change streams available - what Atlas gives you)
    docker run -d --name mongo-rs -p 27077:27017 mongo:7 --replSet rs0 --bind_ip_all
    docker exec mongo-rs mongosh --quiet --eval \
        'rs.initiate({_id:"rs0",members:[{_id:0,host:"127.0.0.1:27017"}]})'
    MONGODB_TEST_URI="mongodb://127.0.0.1:27077/?replicaSet=rs0&directConnection=true" \
        pytest tests/test_integration.py -v

    # standalone (no change streams - exercises the polling fallback)
    docker run -d --name mongo-standalone -p 27078:27017 mongo:7
    MONGODB_TEST_URI="mongodb://127.0.0.1:27078/?directConnection=true" \
        pytest tests/test_integration.py -v

These cover what the unit suite structurally cannot: the aggregation pipelines
(they are MongoDB query language, not Python), and the guarantee that the
analytics endpoints and the record listing return consistent totals.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

MONGODB_TEST_URI = os.getenv("MONGODB_TEST_URI", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not MONGODB_TEST_URI, reason="set MONGODB_TEST_URI to run"),
    pytest.mark.asyncio,
]

TEST_DB = "pulseai_pytest"


@pytest.fixture
async def database():
    """A connected, freshly emptied test database."""
    os.environ["MONGODB_URI"] = MONGODB_TEST_URI
    os.environ["MONGODB_DB"] = TEST_DB

    from api import db
    from src import config

    # SERVICE is read at import time, so point it at the test cluster.
    object.__setattr__(config.SERVICE, "mongodb_uri", MONGODB_TEST_URI)
    object.__setattr__(config.SERVICE, "mongodb_db", TEST_DB)

    assert await db.connect(), f"could not connect: {db.last_error()}"
    await db.feedback_collection().delete_many({})
    yield db
    await db.feedback_collection().delete_many({})
    await db.close()


def _document(text, sentiment, *, source="web", days_ago=1, categories=None, confidence=0.9):
    now = datetime.now(timezone.utc)
    scores = {"negative": 0.1, "neutral": 0.1, "positive": 0.1}
    scores[sentiment] = confidence
    return {
        "text": text,
        "source": source,
        "product": "Pulse Checkout",
        "sentiment": sentiment,
        "confidence": confidence,
        "scores": scores,
        "categories": categories or [],
        "model": "test",
        "created_at": now - timedelta(days=days_ago),
        "analyzed_at": now,
    }


@pytest.fixture
async def seeded(database):
    await database.insert_many_feedback([
        _document("Delivery was late again", "negative", categories=["Delivery & Logistics"]),
        _document("Support was rude", "negative", source="email", categories=["Customer Support"]),
        _document("Package arrived damaged", "negative", categories=["Delivery & Logistics"]),
        _document("It was acceptable", "neutral", days_ago=40),
        _document("Nothing special", "neutral", source="email", days_ago=40),
        _document("Absolutely brilliant service", "positive", categories=["Customer Support"]),
        _document("Great value for money", "positive", source="email"),
        _document("Delivery was quick", "positive", categories=["Delivery & Logistics"]),
    ])
    return database


class TestAggregations:
    async def test_summary_counts_every_document_once(self, seeded):
        summary = await seeded.summary({})
        assert summary["total"] == 8
        assert summary["counts"] == {"negative": 3, "neutral": 2, "positive": 3}
        assert sum(summary["counts"].values()) == summary["total"]

    async def test_distribution_sums_to_one_hundred(self, seeded):
        summary = await seeded.summary({})
        assert abs(sum(summary["distribution"].values()) - 100) < 0.5

    async def test_net_sentiment_score_matches_its_definition(self, seeded):
        summary = await seeded.summary({})
        expected = (3 / 8 - 3 / 8) * 100
        assert summary["net_sentiment_score"] == pytest.approx(expected, abs=0.01)

    async def test_empty_result_set_does_not_divide_by_zero(self, database):
        summary = await database.summary({})
        assert summary == {
            "total": 0,
            "counts": {"negative": 0, "neutral": 0, "positive": 0},
            "distribution": {"negative": 0.0, "neutral": 0.0, "positive": 0.0},
            "net_sentiment_score": 0.0,
            "avg_confidence": 0.0,
        }

    @pytest.mark.parametrize("granularity", ["hour", "day", "week", "month"])
    async def test_trend_rows_are_internally_consistent(self, seeded, granularity):
        rows = await seeded.trends({}, granularity=granularity)
        assert rows, "expected at least one bucket"
        for row in rows:
            assert row["negative"] + row["neutral"] + row["positive"] == row["total"]
        assert sum(row["total"] for row in rows) == 8

    async def test_trends_are_ordered_by_period(self, seeded):
        rows = await seeded.trends({}, granularity="day")
        assert [row["period"] for row in rows] == sorted(row["period"] for row in rows)

    async def test_breakdown_partitions_the_whole_set(self, seeded):
        rows = await seeded.breakdown({}, field="source")
        assert sum(row["total"] for row in rows) == 8
        assert {row["key"] for row in rows} == {"web", "email"}

    async def test_issue_ranking_is_sorted_by_negative_volume(self, seeded):
        rows = await seeded.issue_ranking({})
        assert [row["category"] for row in rows][0] == "Delivery & Logistics"
        assert all(
            rows[i]["negative"] >= rows[i + 1]["negative"] for i in range(len(rows) - 1)
        )

    async def test_multi_tagged_documents_count_towards_every_category(self, database):
        await database.insert_many_feedback([
            _document("Late and the agent was rude", "negative",
                      categories=["Delivery & Logistics", "Customer Support"]),
        ])
        rows = await database.issue_ranking({})
        assert {row["category"] for row in rows} == {"Delivery & Logistics", "Customer Support"}
        assert all(row["negative"] == 1 for row in rows)


class TestFilterConsistency:
    """The KPI cards and the record table must never disagree."""

    async def test_sentiment_filter_agrees_across_surfaces(self, seeded):
        query = seeded.build_filter(sentiment="negative")
        summary = await seeded.summary(query)
        listing = await seeded.list_feedback(query, page_size=100)
        assert summary["total"] == listing["total"] == 3

    async def test_source_filter_agrees_across_surfaces(self, seeded):
        query = seeded.build_filter(source="email")
        summary = await seeded.summary(query)
        listing = await seeded.list_feedback(query, page_size=100)
        assert summary["total"] == listing["total"] == 3

    async def test_date_window_excludes_older_documents(self, seeded):
        recent = await seeded.summary(seeded.build_filter(days=7))
        assert recent["total"] == 6  # the two neutral rows are 40 days old
        assert recent["counts"]["neutral"] == 0

    async def test_combined_filters_intersect(self, seeded):
        query = seeded.build_filter(sentiment="negative", source="email")
        assert (await seeded.summary(query))["total"] == 1


class TestPagination:
    async def test_pages_partition_the_result_set(self, seeded):
        first = await seeded.list_feedback({}, page=1, page_size=3)
        second = await seeded.list_feedback({}, page=2, page_size=3)

        assert len(first["items"]) == 3
        assert first["total"] == second["total"] == 8
        assert first["pages"] == 3
        assert not {i["id"] for i in first["items"]} & {i["id"] for i in second["items"]}

    async def test_page_beyond_the_end_is_empty_not_an_error(self, seeded):
        page = await seeded.list_feedback({}, page=99, page_size=10)
        assert page["items"] == [] and page["total"] == 8

    async def test_sorting_by_confidence_descends(self, database):
        await database.insert_many_feedback([
            _document("low", "negative", confidence=0.51),
            _document("high", "negative", confidence=0.99),
            _document("mid", "negative", confidence=0.75),
        ])
        items = (await database.list_feedback({}, sort="-confidence"))["items"]
        assert [i["confidence"] for i in items] == [0.99, 0.75, 0.51]


class TestWrites:
    async def test_insert_then_delete_round_trip(self, database):
        inserted_id = await database.insert_feedback(_document("temporary", "neutral"))
        assert await database.count_documents() == 1

        assert await database.delete_feedback(inserted_id) is True
        assert await database.count_documents() == 0

    async def test_deleting_a_malformed_id_reports_false(self, database):
        assert await database.delete_feedback("not-an-object-id") is False

    async def test_deleting_twice_reports_false_the_second_time(self, database):
        inserted_id = await database.insert_feedback(_document("temporary", "neutral"))
        assert await database.delete_feedback(inserted_id) is True
        assert await database.delete_feedback(inserted_id) is False


class TestKeywords:
    async def test_keywords_are_extracted_from_stored_text(self, seeded):
        words = {row["text"] for row in await seeded.keyword_cloud({}, sentiment="negative")}
        assert "delivery" in words
        # Function words are topics for nobody.
        assert not {"was", "the", "again"} & words
