"""Health, model introspection and evaluation-metric endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from api import __version__, db
from api.inference import engine
from api.schemas import HealthOut
from src.metrics import load_model_metrics

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthOut, summary="Liveness and dependency status")
async def health() -> dict:
    """Reports *why* the service is degraded, not just that it is.

    Deliberately returns 200 even when MongoDB is down: the container is alive
    and can still classify text, so an orchestrator should not restart it. The
    ``status`` field is what a monitor should alert on.
    """
    model_info = engine.info()
    database_ok = db.is_connected()

    document_count = None
    if database_ok:
        try:
            document_count = await db.count_documents()
        except Exception as exc:  # noqa: BLE001
            database_ok = False
            document_count = f"error: {exc}"

    return {
        "status": "ok" if database_ok else "degraded",
        "api_version": __version__,
        "database": {
            "connected": database_ok,
            "documents": document_count,
            "error": db.last_error(),
        },
        "model": model_info,
    }


@router.get("/api/model/info", summary="Which model is currently serving")
async def model_info() -> dict:
    return engine.info()


@router.get("/api/model/metrics", summary="Offline evaluation results")
async def model_metrics() -> dict:
    """Serves ``reports/metrics.json`` - the numbers behind the Model Card.

    Reading it per request (instead of caching at import) means re-running
    training refreshes the dashboard without restarting the API.
    """
    payload = load_model_metrics()
    payload["available"] = bool(payload.get("models"))
    if not payload["available"]:
        payload["hint"] = (
            "No evaluation results yet. Run `python -m src.train_baseline` and "
            "`python -m src.train_transformer` to generate reports/metrics.json."
        )
    return payload
