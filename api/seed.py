"""Populate MongoDB with a realistic demo dataset.

The *text* is real: rows are drawn from the held-out test split of the review
corpus prepared by ``src.dataset``, so the dashboard is showing genuine
customer writing rather than lorem ipsum. The *metadata* around it (channel,
product, timestamp) is generated, because the public corpus does not ship
those fields and a trend chart needs a time axis.

Timestamps are drawn from a weekday-weighted distribution with a deliberate
negative spike in one recent week, so the Overview page has something real to
show: a trend that moves, and an issue that can actually be found.

Usage:
    python -m api.seed --count 600
    python -m api.seed --count 600 --reset
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from api import db
from api.inference import engine
from src.config import COLLECTION_FEEDBACK, PATHS

CHANNELS = [
    ("web", 0.26),
    ("mobile_app", 0.22),
    ("play_store", 0.14),
    ("email", 0.12),
    ("twitter", 0.10),
    ("survey", 0.08),
    ("call_center", 0.05),
    ("review_site", 0.03),
]

PRODUCTS = [
    "Pulse Checkout",
    "Pulse Mobile",
    "Pulse Delivery",
    "Pulse Support Plus",
    "Pulse Marketplace",
]

BATCH_SIZE = 128


def _weighted_channel(rng: random.Random) -> str:
    roll = rng.random()
    cumulative = 0.0
    for name, weight in CHANNELS:
        cumulative += weight
        if roll <= cumulative:
            return name
    return CHANNELS[0][0]


def _timestamp(rng: random.Random, days: int, spike_window: tuple[int, int]) -> datetime:
    """Pick a plausible arrival time.

    Feedback volume is not uniform: it clusters on weekdays and during working
    hours. Reproducing that makes the trend chart look like a real product's
    data instead of white noise.
    """
    day_offset = rng.randint(0, days - 1)
    candidate = datetime.now(timezone.utc) - timedelta(days=day_offset)

    # Thin out weekends rather than removing them entirely.
    if candidate.weekday() >= 5 and rng.random() < 0.55:
        candidate -= timedelta(days=rng.randint(1, 3))

    return candidate.replace(
        hour=min(23, max(0, int(rng.gauss(14, 4)))),
        minute=rng.randint(0, 59),
        second=rng.randint(0, 59),
        microsecond=0,
    )


_ESCAPE_SEQUENCES = re.compile(r"\\+[nrt]")


def unescape_corpus_text(text: str) -> str:
    """Undo the corpus's literal escape sequences.

    The source stores newlines and quotes as the two characters ``\\n`` and
    ``\\"`` rather than as the characters themselves. That is an artefact of how
    the dataset was serialised, not something a customer typed - so it is
    repaired here, at the point the corpus is read, rather than in the API
    (which must store exactly what a real client sends).
    """
    text = _ESCAPE_SEQUENCES.sub(" ", str(text))
    text = text.replace('\\"', '"').replace("\\'", "'")
    return re.sub(r"\s+", " ", text).strip()


def _load_corpus(count: int, seed: int) -> pd.DataFrame:
    """Prefer the real held-out reviews; fall back to the synthetic generator."""
    for path in (PATHS.test_csv, PATHS.val_csv, PATHS.train_csv):
        if path.exists():
            frame = pd.read_csv(path)
            frame["text"] = frame["text"].map(unescape_corpus_text)
            print(f"[seed] using {len(frame)} real reviews from {path.name}")
            if len(frame) >= count:
                return frame.sample(count, random_state=seed).reset_index(drop=True)
            # Sample with replacement only when the corpus is genuinely smaller.
            return frame.sample(count, replace=True, random_state=seed).reset_index(drop=True)

    print("[seed] no prepared splits found; generating synthetic feedback")
    from src.dataset import generate_synthetic

    return generate_synthetic(-(-count // 3), seed=seed).head(count)


async def seed(count: int = 600, days: int = 90, reset: bool = False, seed_value: int = 7) -> None:
    rng = random.Random(seed_value)

    if not await db.connect():
        raise SystemExit(
            "Cannot reach MongoDB. Set MONGODB_URI in .env (see .env.example) and retry."
        )

    collection = db.get_database()[COLLECTION_FEEDBACK]
    if reset:
        deleted = (await collection.delete_many({})).deleted_count
        print(f"[seed] cleared {deleted} existing documents")

    print("[seed] loading the sentiment model...")
    engine.load()
    print(f"[seed] model: {engine.model_name}")

    corpus = _load_corpus(count, seed_value)
    # A recent week where negative feedback spikes, so the dashboard has a
    # genuine incident to surface rather than a flat line.
    spike_window = (5, 12)

    inserted = 0
    for start in range(0, len(corpus), BATCH_SIZE):
        chunk = corpus.iloc[start : start + BATCH_SIZE]
        texts = [str(text)[:5000] for text in chunk["text"]]

        predictions = await asyncio.to_thread(engine.predict_batch, texts)

        documents = []
        for text, prediction in zip(texts, predictions, strict=True):
            created_at = _timestamp(rng, days, spike_window)
            day_offset = (datetime.now(timezone.utc) - created_at).days

            # During the spike window, over-sample negatives by re-drawing the
            # timestamp for non-negative items out of that range.
            in_spike = spike_window[0] <= day_offset <= spike_window[1]
            if in_spike and prediction.label != "negative" and rng.random() < 0.55:
                created_at -= timedelta(days=rng.randint(14, days - 1))

            now = datetime.now(timezone.utc)
            documents.append(
                {
                    "text": text,
                    "source": _weighted_channel(rng),
                    "product": rng.choice(PRODUCTS) if rng.random() < 0.75 else None,
                    "customer_id": f"CUST-{rng.randint(1000, 9999)}",
                    "rating": None,
                    "sentiment": prediction.label,
                    "confidence": round(prediction.confidence, 4),
                    "scores": {k: round(v, 4) for k, v in prediction.scores.items()},
                    "categories": prediction.categories,
                    "model": prediction.model,
                    "created_at": created_at,
                    "analyzed_at": min(now, created_at + timedelta(seconds=rng.randint(1, 90))),
                }
            )

        await db.insert_many_feedback(documents)
        inserted += len(documents)
        print(f"[seed] inserted {inserted}/{len(corpus)}", end="\r", flush=True)

    print()
    summary = await db.summary({})
    print(
        f"[seed] done: {summary['total']} documents | "
        f"NSS {summary['net_sentiment_score']} | {summary['counts']}"
    )
    await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed MongoDB with demo feedback.")
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--days", type=int, default=90, help="spread over the last N days")
    parser.add_argument("--reset", action="store_true", help="delete existing documents first")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    asyncio.run(seed(count=args.count, days=args.days, reset=args.reset, seed_value=args.seed))


if __name__ == "__main__":
    main()
