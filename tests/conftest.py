"""Shared pytest fixtures.

The API tests run against the ASGI app in-process with no network and no
database, so they are fast and deterministic. Anything that genuinely needs
MongoDB is marked ``integration`` and skipped unless ``MONGODB_TEST_URI`` is
set - CI should never depend on a live Atlas cluster.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force the deterministic lexicon backend: the unit tests are about routing,
# validation and analytics maths, not about model quality, and downloading a
# checkpoint in CI would make them slow and flaky.
os.environ.setdefault("MODEL_DIR", str(ROOT / "models" / "__unit_test_none__"))
os.environ.setdefault("FALLBACK_MODEL", "")
os.environ.setdefault("MONGODB_URI", "")
# Left unset deliberately: if a quantized ONNX model happens to be present the
# suite runs against it, which is the backend a deployment actually uses. Set
# ONNX_MODEL_DIR to a nonexistent path to force the lower tiers instead.


@pytest.fixture(scope="session")
def app():
    from api.main import app as fastapi_app

    return fastapi_app


@pytest.fixture(scope="session")
def client(app):
    from fastapi.testclient import TestClient

    # Not using TestClient as a context manager: that would run the lifespan,
    # which tries to reach MongoDB. These tests exercise the stateless paths.
    return TestClient(app)


@pytest.fixture(scope="session")
def engine():
    from api.inference import engine as sentiment_engine

    sentiment_engine.load()
    return sentiment_engine


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: requires a live MongoDB (set MONGODB_TEST_URI)"
    )
