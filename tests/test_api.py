"""API contract tests.

Run against the ASGI app in-process with no database, so they verify routing,
validation, error shapes and the inference path - the parts that must not break
when the model or the storage layer changes underneath.
"""

from __future__ import annotations

import pytest


class TestSystemEndpoints:
    def test_root_lists_the_endpoints(self, client):
        body = client.get("/").json()
        assert body["service"].startswith("PulseAI")
        assert "predict" in body["endpoints"]

    def test_health_reports_degraded_without_a_database(self, client):
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["database"]["connected"] is False
        assert body["labels"] == ["negative", "neutral", "positive"]

    def test_health_is_200_even_when_degraded(self, client):
        """A missing database must not make the container look dead."""
        assert client.get("/health").status_code == 200

    def test_model_info_describes_the_loaded_backend(self, client):
        body = client.get("/api/model/info").json()
        assert body["backend"] in {"transformer", "sklearn", "lexicon"}
        assert body["labels"] == ["negative", "neutral", "positive"]

    def test_openapi_schema_is_valid(self, client):
        spec = client.get("/openapi.json").json()
        assert "/api/predict" in spec["paths"]
        assert "/api/analytics/summary" in spec["paths"]

    def test_timing_header_is_present(self, client):
        assert "X-Process-Time-Ms" in client.get("/health").headers


class TestPredict:
    def test_returns_a_complete_prediction(self, client):
        response = client.post("/api/predict", json={"text": "This is fantastic!"})
        assert response.status_code == 200

        body = response.json()
        assert body["label"] in {"negative", "neutral", "positive"}
        assert 0.0 <= body["confidence"] <= 1.0
        assert set(body["scores"]) == {"negative", "neutral", "positive"}
        assert body["latency_ms"] >= 0

    def test_probabilities_form_a_distribution(self, client):
        scores = client.post("/api/predict", json={"text": "Terrible."}).json()["scores"]
        assert abs(sum(scores.values()) - 1.0) < 1e-6
        assert all(0.0 <= value <= 1.0 for value in scores.values())

    def test_confidence_equals_the_winning_score(self, client):
        body = client.post("/api/predict", json={"text": "Very poor service"}).json()
        assert body["confidence"] == pytest.approx(max(body["scores"].values()), abs=1e-4)

    def test_issue_categories_are_attached(self, client):
        body = client.post(
            "/api/predict", json={"text": "The delivery was late and damaged"}
        ).json()
        assert "Delivery & Logistics" in body["categories"]

    def test_explanation_is_opt_in(self, client):
        without = client.post("/api/predict", json={"text": "Awful service"}).json()
        assert "explanation" not in without

        with_explanation = client.post(
            "/api/predict", json={"text": "Awful service", "explain": True}
        ).json()
        tokens = with_explanation["explanation"]
        assert len(tokens) == 2
        assert {"token", "weight", "normalised"} <= set(tokens[0])

    @pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}])
    def test_invalid_payloads_are_rejected(self, client, payload):
        assert client.post("/api/predict", json=payload).status_code == 422

    def test_oversized_text_is_rejected(self, client):
        assert client.post("/api/predict", json={"text": "a" * 5001}).status_code == 422

    def test_batch_preserves_input_order_and_length(self, client):
        texts = ["Excellent!", "Terrible.", "It is acceptable."]
        body = client.post("/api/predict/batch", json={"texts": texts}).json()
        assert body["count"] == 3 == len(body["predictions"])

    def test_empty_batch_is_rejected(self, client):
        assert client.post("/api/predict/batch", json={"texts": []}).status_code == 422

    def test_batch_size_is_capped(self, client):
        response = client.post("/api/predict/batch", json={"texts": ["ok"] * 201})
        assert response.status_code == 422


class TestDatabaseGating:
    """Without MongoDB the persistence endpoints must fail clearly, not crash."""

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/api/feedback"),
            ("get", "/api/feedback"),
            ("get", "/api/analytics/summary"),
            ("get", "/api/analytics/trends"),
            ("get", "/api/analytics/issues"),
            ("get", "/api/analytics/filters"),
        ],
    )
    def test_persistence_endpoints_return_503(self, client, method, path):
        call = getattr(client, method)
        response = call(path, json={"text": "hello"}) if method == "post" else call(path)
        assert response.status_code == 503
        assert "MONGODB_URI" in response.json()["detail"]

    def test_stateless_prediction_still_works(self, client):
        """The service degrades - it does not go down."""
        assert client.post("/api/predict", json={"text": "hi"}).status_code == 200


class TestQueryValidation:
    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            ({"sentiment": "bogus"}, 422),
            ({"days": 0}, 422),
            ({"days": 99999}, 422),
            ({"min_confidence": 1.5}, 422),
            ({"page": 0}, 422),
            ({"page_size": 500}, 422),
            ({"sort": "; drop"}, 422),
        ],
    )
    def test_bad_query_parameters_are_rejected_before_the_database(
        self, client, params, expected
    ):
        assert client.get("/api/feedback", params=params).status_code == expected

    @pytest.mark.parametrize("granularity", ["hour", "day", "week", "month"])
    def test_valid_granularities_pass_validation(self, client, granularity):
        # 503 means it reached the database gate, i.e. validation accepted it.
        response = client.get("/api/analytics/trends", params={"granularity": granularity})
        assert response.status_code == 503

    def test_invalid_granularity_is_rejected(self, client):
        response = client.get("/api/analytics/trends", params={"granularity": "decade"})
        assert response.status_code == 422


class TestFilterBuilder:
    """The filter builder is shared by the list and analytics paths, so a bug
    here would silently make the KPI cards disagree with the table."""

    def test_sentinel_values_do_not_become_filters(self):
        from api.db import build_filter

        assert build_filter(sentiment="all", source="all", category="all") == {}

    def test_each_filter_maps_to_its_field(self):
        from api.db import build_filter

        query = build_filter(sentiment="negative", source="web", category="Wait Time")
        assert query == {
            "sentiment": "negative",
            "source": "web",
            "categories": "Wait Time",
        }

    def test_days_becomes_a_lower_bound_on_created_at(self):
        from api.db import build_filter

        query = build_filter(days=7)
        assert "$gte" in query["created_at"]

    def test_search_uses_the_text_index(self):
        from api.db import build_filter

        assert build_filter(search="refund")["$text"] == {"$search": "refund"}


class TestEngine:
    def test_label_map_falls_back_on_unknown_label_spaces(self):
        from api.inference import SentimentEngine

        class Config:
            id2label = {0: "WEIRD_A", 1: "WEIRD_B", 2: "WEIRD_C"}

        assert SentimentEngine._resolve_label_map(Config()) == {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

    def test_label_aliases_are_normalised(self):
        from api.inference import SentimentEngine

        class Config:
            id2label = {0: "NEGATIVE", 1: "Neutral", 2: "POS"}

        assert SentimentEngine._resolve_label_map(Config()) == {
            0: "negative",
            1: "neutral",
            2: "positive",
        }

    def test_lexicon_backend_respects_negation(self):
        from api.inference import SentimentEngine

        plain = SentimentEngine._scores_lexicon("this is good")
        negated = SentimentEngine._scores_lexicon("this is not good")
        assert plain["positive"] > plain["negative"]
        assert negated["negative"] > negated["positive"]

    def test_batch_and_single_predictions_agree(self, engine):
        text = "The refund process was a nightmare"
        single = engine.predict(text)
        batched = engine.predict_batch([text, "unrelated filler"])[0]
        assert single.label == batched.label
        assert single.scores == pytest.approx(batched.scores, abs=1e-6)
