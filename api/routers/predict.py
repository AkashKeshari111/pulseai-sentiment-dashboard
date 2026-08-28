"""Stateless inference endpoints.

Separate from ``/api/feedback`` on purpose: these classify without writing
anything, which is what the dashboard's "Analyze" playground and any external
integration wanting a pure scoring service should call.
"""

from __future__ import annotations

import anyio
from fastapi import APIRouter, Depends

from api.deps import require_api_key
from api.inference import engine
from api.schemas import BatchTextIn, PredictionOut, TextIn

router = APIRouter(prefix="/api", tags=["inference"])


@router.post(
    "/predict",
    response_model=PredictionOut,
    # Without this the key is still serialised as `"explanation": null`; an
    # opt-in field should simply be absent when it was not requested.
    response_model_exclude_none=True,
    summary="Classify one text",
)
async def predict(payload: TextIn) -> dict:
    """Run the model on a single piece of text.

    The forward pass is CPU-bound and releases the GIL only partially, so it
    runs in a worker thread. Without this, one slow inference would block the
    event loop and stall every other in-flight request.
    """
    prediction = await anyio.to_thread.run_sync(
        lambda: engine.predict(payload.text, explain=payload.explain)
    )
    return prediction.to_dict()


@router.post("/predict/batch", summary="Classify many texts in one call")
async def predict_batch(payload: BatchTextIn) -> dict:
    """Batched inference - one padded forward pass instead of N sequential ones."""
    predictions = await anyio.to_thread.run_sync(
        lambda: engine.predict_batch(payload.texts)
    )
    return {
        "count": len(predictions),
        "predictions": [p.to_dict() for p in predictions],
    }


@router.post(
    "/explain",
    summary="Word-level attribution for a prediction",
    dependencies=[Depends(require_api_key)],
)
async def explain(payload: TextIn) -> dict:
    """Leave-one-out occlusion: how much each word moved the predicted class.

    Costs one forward pass per word, hence the separate endpoint and the API
    key guard - it is materially more expensive than ``/predict``.
    """
    prediction = await anyio.to_thread.run_sync(
        lambda: engine.predict(payload.text, explain=True)
    )
    return prediction.to_dict()
