"""Feedback ingestion and retrieval.

Every write goes through the same path: classify with the model, enrich with
issue categories, then persist. Storing the model name and the full score
vector alongside each document means predictions stay auditable after a model
upgrade - you can always tell which model produced a given label.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

import anyio
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from api import db
from api.deps import FeedbackFilters, require_api_key
from api.inference import engine
from api.schemas import (
    MAX_BATCH_SIZE,
    MAX_TEXT_LENGTH,
    FeedbackBatchIn,
    FeedbackIn,
    IngestResult,
    PaginatedFeedback,
)

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


def _build_document(item: FeedbackIn, prediction) -> dict:
    now = datetime.now(timezone.utc)
    created_at = item.created_at or now
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    return {
        "text": item.text,
        "source": item.source,
        "product": item.product,
        "customer_id": item.customer_id,
        "rating": item.rating,
        "sentiment": prediction.label,
        "confidence": round(prediction.confidence, 4),
        "scores": {k: round(v, 4) for k, v in prediction.scores.items()},
        "categories": prediction.categories,
        "model": prediction.model,
        "created_at": created_at,
        "analyzed_at": now,
    }


@router.post(
    "",
    response_model=IngestResult,
    status_code=status.HTTP_201_CREATED,
    summary="Classify and store one piece of feedback",
    dependencies=[Depends(require_api_key)],
)
async def create_feedback(item: FeedbackIn) -> dict:
    prediction = await anyio.to_thread.run_sync(
        lambda: engine.predict(item.text, explain=item.explain)
    )
    document = _build_document(item, prediction)
    inserted_id = await db.insert_feedback(document)
    return {
        "inserted": 1,
        "ids": [inserted_id],
        "predictions": [prediction.to_dict()],
    }


@router.post(
    "/batch",
    response_model=IngestResult,
    status_code=status.HTTP_201_CREATED,
    summary="Classify and store many items in one call",
    dependencies=[Depends(require_api_key)],
)
async def create_feedback_batch(payload: FeedbackBatchIn) -> dict:
    """One padded forward pass over the whole batch, then a single bulk insert.

    Batching both halves is what makes bulk import fast: N round trips to the
    model and N round trips to Atlas collapse into one of each.
    """
    texts = [item.text for item in payload.items]
    predictions = await anyio.to_thread.run_sync(lambda: engine.predict_batch(texts))

    documents = [
        _build_document(item, prediction)
        for item, prediction in zip(payload.items, predictions, strict=True)
    ]
    ids = await db.insert_many_feedback(documents)
    return {
        "inserted": len(ids),
        "ids": ids,
        "predictions": [p.to_dict() for p in predictions],
    }


@router.post(
    "/upload",
    summary="Bulk import feedback from a CSV file",
    dependencies=[Depends(require_api_key)],
)
async def upload_csv(
    file: UploadFile = File(..., description="CSV with a 'text' column."),
    source: str = Query("csv_upload", max_length=64),
) -> dict:
    """Import a CSV export from a review platform or support desk.

    Required column: ``text``. Optional: ``source``, ``product``,
    ``customer_id``, ``rating``, ``created_at``. Unknown columns are ignored so
    a raw vendor export can be dropped in unchanged.
    """
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = raw.decode("latin-1")

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames or "text" not in {f.strip().lower() for f in reader.fieldnames}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CSV must contain a 'text' column. Found: {reader.fieldnames}",
        )

    items: list[FeedbackIn] = []
    skipped = 0
    for row in reader:
        normalised = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        text = normalised.get("text", "")
        if not text:
            skipped += 1
            continue
        try:
            items.append(
                FeedbackIn(
                    text=text[:MAX_TEXT_LENGTH],
                    source=normalised.get("source") or source,
                    product=normalised.get("product") or None,
                    customer_id=normalised.get("customer_id") or None,
                    rating=int(normalised["rating"]) if normalised.get("rating", "").isdigit() else None,
                    created_at=_parse_date(normalised.get("created_at")),
                )
            )
        except Exception:  # noqa: BLE001 - one bad row must not fail the import
            skipped += 1

    if not items:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No usable rows found in the uploaded file.",
        )

    # Chunked so a 10k-row file neither exhausts memory nor blocks for minutes.
    total_inserted, all_ids = 0, []
    for start in range(0, len(items), MAX_BATCH_SIZE):
        chunk = items[start : start + MAX_BATCH_SIZE]
        predictions = await anyio.to_thread.run_sync(
            lambda c=chunk: engine.predict_batch([i.text for i in c])
        )
        documents = [_build_document(i, p) for i, p in zip(chunk, predictions, strict=True)]
        ids = await db.insert_many_feedback(documents)
        total_inserted += len(ids)
        all_ids.extend(ids)

    return {
        "filename": file.filename,
        "inserted": total_inserted,
        "skipped": skipped,
        "sample_ids": all_ids[:5],
    }


@router.get(
    "",
    response_model=PaginatedFeedback,
    summary="List stored feedback with filters and pagination",
)
async def list_feedback(
    filters: FeedbackFilters = Depends(),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    sort: str = Query("-created_at", pattern="^-?(created_at|confidence|analyzed_at)$"),
) -> dict:
    return await db.list_feedback(
        filters.to_query(), page=page, page_size=page_size, sort=sort
    )


@router.delete(
    "/{feedback_id}",
    summary="Delete one feedback document",
    dependencies=[Depends(require_api_key)],
)
async def delete_feedback(feedback_id: str) -> dict:
    deleted = await db.delete_feedback(feedback_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No feedback found with id {feedback_id}",
        )
    return {"deleted": True, "id": feedback_id}


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
