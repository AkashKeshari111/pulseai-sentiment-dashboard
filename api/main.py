"""FastAPI application entry point.

Run locally:
    uvicorn api.main:app --reload --port 8000

Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__, db
from api.inference import engine
from api.routers import analytics, feedback, health, predict, stream
from src.config import SERVICE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pulseai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model and open the database connection before serving.

    Loading the transformer eagerly moves a multi-second, once-per-process
    cost off the first user request. Both steps are non-fatal: the service
    starts even if Atlas is unreachable, and reports that through /health.
    """
    logger.info("PulseAI API v%s starting", __version__)

    started = time.perf_counter()
    engine.load()
    logger.info(
        "model ready in %.1fs -> %s", time.perf_counter() - started, engine.model_name
    )

    await db.connect()
    yield

    await db.close()
    logger.info("PulseAI API stopped")


app = FastAPI(
    title="PulseAI - Customer Sentiment Intelligence API",
    description=(
        "Real-time sentiment classification for customer feedback, backed by a "
        "fine-tuned DistilBERT model and MongoDB Atlas.\n\n"
        "- `POST /api/predict` classifies text without storing it\n"
        "- `POST /api/feedback` classifies **and** persists\n"
        "- `GET /api/analytics/*` powers the dashboard\n"
        "- `GET /api/stream` streams new feedback over SSE"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=SERVICE.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Expose server-side latency so the dashboard can display it honestly."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(db.DatabaseUnavailable)
async def database_unavailable_handler(request: Request, exc: db.DatabaseUnavailable):
    """Turn a missing database into an actionable 503.

    Handled here rather than as a route dependency on purpose: dependencies are
    resolved before query parameters are validated, so gating that way would
    answer a request with a bad `days=0` parameter with "database unavailable"
    instead of telling the caller what is actually wrong with their request.
    """
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Return a structured error instead of an HTML stack trace."""
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": exc.__class__.__name__,
            "path": request.url.path,
        },
    )


app.include_router(health.router)
app.include_router(predict.router)
app.include_router(feedback.router)
app.include_router(analytics.router)
app.include_router(stream.router)


@app.get("/", tags=["system"], summary="Service banner")
async def root() -> dict:
    return {
        "service": "PulseAI - Customer Sentiment Intelligence API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /api/predict",
            "ingest": "POST /api/feedback",
            "list": "GET /api/feedback",
            "analytics": "GET /api/analytics/summary",
            "live": "GET /api/stream",
        },
    }


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
