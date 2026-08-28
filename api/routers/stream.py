"""Server-Sent Events feed for the dashboard's live tile.

Two transports, chosen at runtime:

1. **MongoDB change streams** - the database pushes each insert to us. Atlas
   clusters are replica sets, so this works out of the box and new feedback
   reaches the browser in roughly the time it takes to commit.
2. **Timestamp polling** - used when change streams are unavailable (a
   standalone ``mongod``, or a user lacking the ``changeStream`` privilege).
   Each tick asks only for documents newer than the last one seen, so the
   query stays indexed and cheap.

SSE is preferred over WebSockets here because the traffic is strictly
server-to-client, and SSE reconnects automatically in the browser with no
client-side retry logic.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timezone

import orjson
from fastapi import APIRouter, Query, Request
from sse_starlette.sse import EventSourceResponse

from api import db
from api.deps import require_database

logger = logging.getLogger("pulseai.stream")

router = APIRouter(prefix="/api", tags=["stream"])

POLL_INTERVAL_SECONDS = 3.0
HEARTBEAT_SECONDS = 20.0


def _encode(document: dict) -> str:
    return orjson.dumps(db.serialise(document)).decode()


async def _watch_change_stream(request: Request, queue: asyncio.Queue) -> None:
    """Push inserted documents onto the queue via a change stream."""
    pipeline = [{"$match": {"operationType": "insert"}}]
    async with db.feedback_collection().watch(pipeline) as change_stream:
        async for change in change_stream:
            if await request.is_disconnected():
                return
            await queue.put(change["fullDocument"])


async def _poll_inserts(request: Request, queue: asyncio.Queue) -> None:
    """Fallback: repeatedly ask for anything analysed since the last check."""
    cursor_time = await db.latest_timestamp() or datetime.now(timezone.utc)
    while not await request.is_disconnected():
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        collection = db.feedback_collection()
        query = {"analyzed_at": {"$gt": cursor_time}}
        async for document in collection.find(query).sort("analyzed_at", 1).limit(50):
            cursor_time = document.get("analyzed_at", cursor_time)
            await queue.put(document)


@router.get("/stream", summary="Live feed of newly analysed feedback (SSE)")
async def stream(request: Request, replay: int = Query(5, ge=0, le=50)):
    """Stream new feedback to the browser as it is classified.

    ``replay`` seeds the connection with the N most recent items so a freshly
    opened dashboard is never blank while waiting for the first live event.
    """
    await require_database()

    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue(maxsize=500)

        if replay:
            for document in reversed(await db.recent({}, limit=replay)):
                yield {"event": "seed", "data": orjson.dumps(document).decode()}

        producer = asyncio.create_task(_watch_change_stream(request, queue))
        transport = "change_stream"

        # A change stream fails fast if unsupported; fall back rather than
        # leaving the client with a dead connection.
        await asyncio.sleep(0.4)
        if producer.done() and producer.exception() is not None:
            logger.info(
                "change streams unavailable (%s); polling instead",
                producer.exception(),
            )
            producer = asyncio.create_task(_poll_inserts(request, queue))
            transport = "polling"

        yield {"event": "ready", "data": orjson.dumps({"transport": transport}).decode()}

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    document = await asyncio.wait_for(
                        queue.get(), timeout=HEARTBEAT_SECONDS
                    )
                except asyncio.TimeoutError:
                    # Keeps proxies and load balancers from closing an idle
                    # connection.
                    yield {"event": "ping", "data": "{}"}
                    continue
                yield {"event": "feedback", "data": _encode(document)}
        finally:
            # Cancelling the producer is best-effort cleanup on a connection
            # that is already going away; nothing it can raise is actionable.
            # CancelledError is listed explicitly - it derives from
            # BaseException, so suppressing Exception alone would let the
            # expected cancellation escape.
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer

    return EventSourceResponse(event_generator())
