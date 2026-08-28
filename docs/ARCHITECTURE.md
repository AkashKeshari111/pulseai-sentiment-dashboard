# Architecture

How PulseAI is put together, and why each piece is shaped the way it is.

---

## 1. System overview

Three processes, one data store.

```
┌────────────────────────────────────────────────────────────────────────┐
│  Browser                                                               │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  React SPA                                                       │  │
│  │  ├─ React Query      cache, deduplication, background refresh    │  │
│  │  ├─ Filter context   URL-backed shared filter state              │  │
│  │  ├─ Recharts         validated diverging palette, table fallback │  │
│  │  └─ EventSource      SSE live feed                               │  │
│  └────────────────────────────┬─────────────────────────────────────┘  │
└───────────────────────────────┼────────────────────────────────────────┘
                                │  HTTP (same origin via nginx / Vite proxy)
┌───────────────────────────────▼────────────────────────────────────────┐
│  FastAPI (uvicorn, ASGI)                                               │
│                                                                        │
│   routers/                    inference.py            db.py            │
│   ├─ health    ─────────────▶ SentimentEngine  ────▶  Motor client     │
│   ├─ predict        (stateless)   4-tier loader       aggregation      │
│   ├─ feedback  ─────────────▶     batched forward     pipelines        │
│   ├─ analytics ──────────────────────────────────▶    index-backed     │
│   └─ stream    ──────────────────────────────────▶    change streams   │
│                                                                        │
│   deps.py: shared filter parameters + optional API-key auth            │
└───────────────────────────────┬────────────────────────────────────────┘
                                │  mongodb+srv (TLS)
┌───────────────────────────────▼────────────────────────────────────────┐
│  MongoDB Atlas — `feedback` collection                                 │
│  indexes: created_at · (sentiment, created_at) · source · categories   │
│           · text (full-text)                                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Request flows

### 2.1 Classify and store

```
POST /api/feedback
  │
  ├─▶ Pydantic validation           reject malformed input before any work
  ├─▶ require_api_key               only when API_KEY is configured
  ├─▶ anyio.to_thread(engine)       forward pass off the event loop
  │      ├─ clean_for_transformer   light normalisation
  │      ├─ tokenize + forward      DistilBERT → softmax
  │      └─ detect_issue_categories keyword → business area
  ├─▶ build document                text + label + full score vector + model name
  ├─▶ insert_one                    Atlas
  └─▶ change stream fires ────────▶ SSE subscribers see it within ~1s
```

The forward pass runs in a worker thread deliberately. Torch releases the GIL only
partially, so a synchronous inference call inside the event loop would stall *every*
other in-flight request for its duration.

Storing the **full score vector and the model name** on every document is what keeps
predictions auditable. After a model upgrade you can still tell which model produced any
given label, and re-score historical data without losing the original.

### 2.2 Analytics

```
GET /api/analytics/summary?days=30&source=web
  │
  ├─▶ FeedbackFilters               shared dependency — same builder as the list endpoint
  ├─▶ build_filter()                one query object, used by every surface
  ├─▶ aggregate([$match, $group])   MongoDB computes it
  └─▶ SummaryOut                    typed response
```

Every analytics endpoint takes the *same* `FeedbackFilters` dependency. That is what
guarantees the KPI card and the record table cannot disagree — there is one definition of
"the current filter", not one per panel.

### 2.3 Live feed

```
GET /api/stream  (Server-Sent Events)
  │
  ├─▶ replay the N most recent          a fresh dashboard is never blank
  ├─▶ try: collection.watch()           MongoDB pushes each insert
  │     └─ on failure: poll analyzed_at indexed, bounded, cheap
  ├─▶ heartbeat every 20s               keeps proxies from closing an idle connection
  └─▶ cancel producer on disconnect
```

SSE rather than WebSockets: the traffic is strictly server-to-client, and `EventSource`
reconnects on its own with no client-side retry logic to write or get wrong.

---

## 3. The model loader

`SentimentEngine.load()` walks four backends in order and stops at the first that works:

| # | Backend | When it is used |
|---|---|---|
| 1 | Fine-tuned DistilBERT (`MODEL_DIR`) | Normal operation |
| 2 | Hub model (`FALLBACK_MODEL`) | Before training has been run |
| 3 | TF-IDF + Logistic Regression | Offline, or no transformer available |
| 4 | Negation-aware lexicon | Last resort — always available, always explainable |

The point is that **the API always answers**. A demo that dies because a checkpoint is
missing or a download failed is worse than one that answers with a clearly-labelled weaker
model — and `/health` and `/api/model/info` always report which tier is actually serving.

### Label-space normalisation

Different checkpoints publish different label vocabularies: `LABEL_0`, `POS`, `negative`,
`5 stars`. `_resolve_label_map` translates any of them onto the canonical
`negative / neutral / positive` ordering, and a model with a wider label space (a 5-star
model) collapses by summing the probability mass mapped to each canonical class. Without
this, swapping the fallback model would silently permute every prediction.

---

## 4. Data model

```javascript
// feedback
{
  _id:         ObjectId,
  text:        "The delivery was late and the box arrived damaged.",
  source:      "mobile_app",         // channel
  product:     "Pulse Delivery",     // optional
  customer_id: "CUST-4821",          // optional
  rating:      2,                    // optional, customer-supplied stars

  sentiment:   "negative",           // predicted label
  confidence:  0.9612,               // winning class probability
  scores:      { negative: 0.9612, neutral: 0.0281, positive: 0.0107 },
  categories:  ["Delivery & Logistics", "Product Quality"],
  model:       "DistilBERT (fine-tuned) - distilbert-sentiment",

  created_at:  ISODate,              // when the customer wrote it
  analyzed_at: ISODate               // when the model scored it
}
```

**Why two timestamps.** `created_at` is the business fact; `analyzed_at` is the processing
fact. Backfilling six months of historical feedback would otherwise pile every record onto
today's bucket in the trend chart. The SSE poll fallback keys off `analyzed_at` for exactly
the mirrored reason — it wants "what was processed since I last looked", not "what was
written".

### Indexes

| Index | Serves |
|---|---|
| `created_at: -1` | Default listing order, date-range filters |
| `(sentiment, created_at)` | The most common compound filter |
| `source` | Channel breakdown and filter |
| `categories` (multikey) | Issue filtering and the `$unwind` ranking |
| `text` (full-text) | The Explorer's search box |

---

## 5. Frontend architecture

### State ownership

| State | Owner | Why there |
|---|---|---|
| Server data | React Query | Caching, deduplication and background refresh are its job, not the component's |
| Filters | URL query string | A filtered view is shareable, bookmarkable and survives a refresh |
| Theme | Context + `localStorage` | Applied before first paint by an inline script, so a reload never flashes |
| Ephemeral UI | Local `useState` | Expanded rows, active tab — nothing that needs to outlive the component |

### Chart system

Every chart is built to the same rules:

- **Sentiment is a diverging encoding** — negative pole, achromatic neutral midpoint,
  positive pole. Not a categorical palette, because the three classes are ordered.
- **Red/green was measured and rejected.** ΔE 4.1 under deuteranopia is indistinguishable
  for roughly 1 in 12 men. Red ↔ grey ↔ blue measures ΔE 8.7 light / 8.5 dark.
- **Colour is never the only channel.** Legend always present, direct labels where they
  fit, glyphs on the badges, and a table view behind every chart.
- **Never two y-axes.** The trend chart *switches* between volume and net score rather than
  overlaying two scales, because a crossing point between two arbitrary scales means
  nothing but looks like it does.
- **One hue for magnitude.** The confusion matrix uses a single-hue sequential ramp, row-
  normalised — the question is "of the items that really were negative, where did they
  go?", which is about each row, not the global maximum.

Colours are declared once as CSS custom properties in `index.css` and read back through
`useChartColors()`, so the palette lives in exactly one place and cannot drift between the
stylesheet and the chart code.

---

## 6. Error handling

| Failure | Behaviour |
|---|---|
| Invalid request | `422` with a field-level explanation |
| Database down | `503` with the actual connection error and how to fix it |
| Model missing | Silent fall-through to the next backend; the tier is reported by `/health` |
| Change streams unavailable | Automatic switch to polling; the transport is shown in the UI |
| Unhandled exception | Structured `500` JSON, logged with a stack trace — never an HTML error page |

**Ordering matters.** The database availability check is raised from *inside* the handler
(as `DatabaseUnavailable`, mapped to 503 by an exception handler) rather than as a route
dependency. FastAPI resolves dependencies *before* validating query parameters, so gating
that way would answer a request with a bad `days=0` parameter with "database unavailable"
— hiding the caller's real mistake behind an unrelated one.

---

## 7. Performance

| Concern | Approach |
|---|---|
| Model load cost | Loaded once in the lifespan hook, before the first request arrives |
| Event-loop blocking | Every forward pass runs via `anyio.to_thread` |
| Bulk ingestion | One padded forward pass per batch, one `insert_many` per chunk |
| Training throughput | Dynamic padding — each batch pads to its own longest member, not to 128 |
| Analytics at volume | Aggregation pipelines against indexed fields |
| Keyword extraction | Bounded scan (most recent 2,000 documents) so cost stays flat |
| Dashboard payload | Recharts split into its own chunk; the first paint is not blocked by charts |
| Repeat queries | React Query caches per filter combination |

---

## 8. Security

- **Optional API-key auth** on write endpoints, compared in constant time.
- **Non-root container user**, no build toolchain in the runtime image.
- **Input caps** at every entry point: 5,000 characters per text, 200 items per batch,
  5 MB per upload.
- **Explicit CORS allowlist** — and in the Docker topology the dashboard proxies the API on
  the same origin, so no cross-origin request is made at all.
- **No secrets in the image.** Configuration is environment-only; `.env` is gitignored.

---

## 9. What would change at scale

The current design is honest about its ceiling. Past roughly a million documents or a few
hundred requests per second:

- **Serving** — export to ONNX Runtime with INT8 quantisation, or move to GPU batching with
  a queue in front.
- **Ingestion** — put a message queue (Kafka/SQS) between intake and scoring so a traffic
  spike buffers instead of timing out.
- **Analytics** — pre-aggregate daily rollups into a summary collection; scanning raw
  documents for a 12-month trend stops being viable.
- **Storage** — time-based sharding on `created_at`, with older data rolled up and archived.
- **Model lifecycle** — a registry with versioned checkpoints, shadow deployment and
  automatic rollback on metric regression.
