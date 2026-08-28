"""Dataset acquisition and preparation.

Turns a public customer-review corpus into three stratified CSV splits with a
single 3-class label space (negative / neutral / positive).

Sources
-------
``yelp_review_full`` (default)
    650k real customer reviews of businesses, labelled 1-5 stars. Stars are
    mapped to sentiment: 1-2 -> negative, 3 -> neutral, 4-5 -> positive. This
    is the closest public proxy for the "customer feedback" domain the project
    targets, and the star mapping is the standard convention in the
    literature.

``tweet_eval``
    ~60k tweets with *native* 3-class sentiment labels. Useful as a
    domain-shift check: a model trained on long reviews and evaluated on short
    social posts shows how well the representation generalises.

``synthetic``
    A template-based generator. Guarantees the whole pipeline (training,
    API, dashboard) runs on a machine with no internet access, and is what CI
    uses so tests never depend on the Hugging Face Hub.

Everything is streamed and quota-filled per class, so preparing 16k balanced
rows never downloads the full multi-GB corpus.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pandas as pd

from src.config import LABEL2ID, LABELS, PATHS, TRAINING

# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """How to pull a Hub dataset and normalise it into (text, label)."""

    hub_id: str
    config: str | None
    split: str
    text_field: str
    label_field: str
    #: Maps the source's native label to one of our 3 class ids, or None to
    #: drop the row entirely.
    label_map: Callable[[int], int | None]
    description: str


def _yelp_stars_to_sentiment(star_index: int) -> int | None:
    """Yelp labels are 0-4 for 1-5 stars."""
    if star_index <= 1:
        return LABEL2ID["negative"]
    if star_index == 2:
        return LABEL2ID["neutral"]
    return LABEL2ID["positive"]


SOURCES: dict[str, SourceSpec] = {
    "yelp_review_full": SourceSpec(
        hub_id="Yelp/yelp_review_full",
        config=None,
        split="train",
        text_field="text",
        label_field="label",
        label_map=_yelp_stars_to_sentiment,
        description="Yelp customer reviews, 1-5 stars mapped to 3 sentiment classes.",
    ),
    "tweet_eval": SourceSpec(
        hub_id="cardiffnlp/tweet_eval",
        config="sentiment",
        split="train",
        # tweet_eval already uses 0=negative, 1=neutral, 2=positive, which is
        # exactly our LABELS ordering, so the map is the identity.
        text_field="text",
        label_field="label",
        label_map=lambda x: x if 0 <= x < len(LABELS) else None,
        description="Tweets with native 3-class sentiment labels.",
    ),
}


# ---------------------------------------------------------------------------
# Synthetic fallback generator
# ---------------------------------------------------------------------------

_TEMPLATES: dict[str, list[str]] = {
    "negative": [
        "The {noun} was {bad_adj} and the {aspect} was even worse. I would not recommend this to anyone.",
        "Completely disappointed with the {aspect}. My {noun} arrived {bad_adj} and nobody responded to my emails.",
        "I waited over an hour and the {aspect} was still {bad_adj}. Save your money and go somewhere else.",
        "Terrible experience. The {noun} is overpriced for what you get and the {aspect} was {bad_adj}.",
        "This is the third time the {aspect} has failed. The {noun} is simply not worth the price.",
        "Do not buy this. The {noun} broke within a week and the refund process was {bad_adj}.",
        "Support was {bad_adj} and unhelpful. I am still waiting for a resolution after two weeks.",
    ],
    "neutral": [
        "The {noun} is okay. Nothing special about the {aspect}, but it does the job.",
        "It works as described. The {aspect} could be better but the price is reasonable.",
        "Average experience overall. The {noun} met expectations, the {aspect} was fine.",
        "Not bad, not great. The {aspect} is standard and the {noun} is what you would expect.",
        "Delivery took the usual time. The {noun} is decent for everyday use.",
        "Mixed feelings: the {aspect} is {good_adj} but the {noun} feels a bit basic.",
        "It is fine. I have used better and I have used worse, so the {aspect} is acceptable.",
    ],
    "positive": [
        "Absolutely {good_adj}! The {noun} exceeded my expectations and the {aspect} was flawless.",
        "Really happy with this purchase. The {aspect} is {good_adj} and delivery was quick.",
        "The team was {good_adj} and resolved my issue in minutes. Excellent {aspect}.",
        "Best {noun} I have bought this year. The {aspect} is worth every rupee.",
        "Fantastic experience from start to finish. The {aspect} was {good_adj} and the staff were friendly.",
        "Highly recommend. The {noun} is well built and the {aspect} is genuinely {good_adj}.",
        "Great value for money. The {aspect} is {good_adj} and support replied within the hour.",
    ],
}

_SLOTS = {
    "noun": [
        "product", "order", "app", "device", "meal", "service plan", "package",
        "subscription", "laptop stand", "headset", "coffee", "delivery",
    ],
    "aspect": [
        "customer support", "delivery experience", "build quality", "checkout flow",
        "packaging", "refund process", "mobile app", "food quality", "waiting time",
        "billing", "user interface", "after-sales service",
    ],
    "bad_adj": [
        "damaged", "rude", "painfully slow", "confusing", "defective", "careless",
        "unacceptable", "sloppy",
    ],
    "good_adj": [
        "outstanding", "smooth", "responsive", "brilliant", "reliable",
        "impressive", "seamless", "excellent",
    ],
}


def generate_synthetic(n_per_class: int, seed: int = 42) -> pd.DataFrame:
    """Build a balanced, template-generated corpus.

    Each row combines a sentiment template with randomly filled slots, so the
    lexical surface varies while the label stays deterministic.
    """
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for label_name in LABELS:
        templates = _TEMPLATES[label_name]
        for i in range(n_per_class):
            template = templates[i % len(templates)]
            text = template.format(**{k: rng.choice(v) for k, v in _SLOTS.items()})
            # Light jitter so duplicate templates are not byte-identical.
            if rng.random() < 0.3:
                text += " " + rng.choice(
                    ["Thanks.", "Please improve.", "Just my honest opinion.", "FYI."]
                )
            rows.append(
                {"text": text, "label": LABEL2ID[label_name], "label_name": label_name}
            )
    frame = pd.DataFrame(rows)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Streaming, quota-filled download
# ---------------------------------------------------------------------------


def _stream_rows(spec: SourceSpec) -> Iterator[dict]:
    from datasets import load_dataset  # imported lazily: heavy dependency

    stream = load_dataset(
        spec.hub_id, spec.config, split=spec.split, streaming=True
    )
    yield from stream


def download_balanced(
    source: str, n_per_class: int, seed: int = 42, max_scanned: int | None = None
) -> pd.DataFrame:
    """Pull ``n_per_class`` examples of each sentiment from a streamed corpus.

    Streaming keeps the download proportional to what we actually keep. The
    scan is capped so a pathological class distribution cannot loop forever.
    """
    spec = SOURCES[source]
    quota = dict.fromkeys(range(len(LABELS)), n_per_class)
    collected: list[dict[str, object]] = []
    cap = max_scanned or n_per_class * len(LABELS) * 25

    for scanned, row in enumerate(_stream_rows(spec)):
        if scanned >= cap or not any(quota.values()):
            break
        label = spec.label_map(int(row[spec.label_field]))
        if label is None or quota.get(label, 0) <= 0:
            continue
        text = str(row[spec.text_field] or "").strip()
        if len(text) < 15:  # discard fragments that carry no usable signal
            continue
        quota[label] -= 1
        collected.append(
            {"text": text, "label": label, "label_name": LABELS[label]}
        )

    if not collected:
        raise RuntimeError(f"No usable rows streamed from {spec.hub_id}")

    frame = pd.DataFrame(collected)
    return frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitting + persistence
# ---------------------------------------------------------------------------


def stratified_split(
    frame: pd.DataFrame, val_size: int, test_size: int, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split preserving the class ratio in every partition.

    Stratification matters here: with an unstratified split a rare class can
    end up absent from validation, which silently breaks macro-F1.
    """
    from sklearn.model_selection import train_test_split

    holdout = val_size + test_size
    train, rest = train_test_split(
        frame,
        test_size=holdout,
        random_state=seed,
        stratify=frame["label"],
    )
    val, test = train_test_split(
        rest,
        test_size=test_size,
        random_state=seed,
        stratify=rest["label"],
    )
    reset = lambda df: df.reset_index(drop=True)  # noqa: E731
    return reset(train), reset(val), reset(test)


def prepare(
    source: str | None = None,
    train_size: int | None = None,
    val_size: int | None = None,
    test_size: int | None = None,
    seed: int | None = None,
    force_synthetic: bool = False,
) -> dict[str, pd.DataFrame]:
    """End-to-end data preparation, writing CSVs into ``data/processed``."""
    source = source or TRAINING.dataset
    train_size = train_size or TRAINING.train_size
    val_size = val_size or TRAINING.val_size
    test_size = test_size or TRAINING.test_size
    seed = seed if seed is not None else TRAINING.seed

    PATHS.ensure()
    total = train_size + val_size + test_size
    n_per_class = -(-total // len(LABELS))  # ceil division

    if force_synthetic or source == "synthetic":
        print(f"[data] generating {n_per_class * len(LABELS)} synthetic rows")
        frame = generate_synthetic(n_per_class, seed=seed)
        resolved_source = "synthetic"
    else:
        try:
            print(f"[data] streaming '{source}' ({n_per_class} rows per class)...")
            frame = download_balanced(source, n_per_class, seed=seed)
            resolved_source = source
        except Exception as exc:  # noqa: BLE001 - any Hub/network failure
            print(f"[data] '{source}' unavailable ({exc.__class__.__name__}: {exc})")
            print("[data] falling back to the synthetic generator")
            frame = generate_synthetic(n_per_class, seed=seed)
            resolved_source = "synthetic"

    frame["source_dataset"] = resolved_source
    frame = frame.drop_duplicates(subset="text").reset_index(drop=True)

    train, val, test = stratified_split(frame, val_size, test_size, seed=seed)
    for name, split in (("train", train), ("val", val), ("test", test)):
        path = PATHS.data_processed / f"{name}.csv"
        split.to_csv(path, index=False)
        counts = Counter(split["label_name"])
        print(
            f"[data] {name:<5} {len(split):>6} rows -> {path.name} "
            f"({dict(sorted(counts.items()))})"
        )

    return {"train": train, "val": val, "test": test}


def load_splits() -> dict[str, pd.DataFrame]:
    """Read the prepared CSVs; raises a helpful error if they are missing."""
    splits = {}
    for name in ("train", "val", "test"):
        path = PATHS.data_processed / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m src.dataset --prepare"
            )
        splits[name] = pd.read_csv(path)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sentiment datasets.")
    parser.add_argument("--prepare", action="store_true", help="build the CSV splits")
    parser.add_argument("--source", default=None, choices=[*SOURCES, "synthetic"])
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--synthetic", action="store_true", help="skip the Hub, generate data locally"
    )
    args = parser.parse_args()

    prepare(
        source=args.source,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        force_synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
