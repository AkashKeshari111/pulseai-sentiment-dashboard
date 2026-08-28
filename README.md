# PulseAI — Customer Sentiment Intelligence Platform

An end-to-end system that classifies customer feedback as **negative / neutral / positive**
using a fine-tuned DistilBERT model, stores it in MongoDB Atlas, and surfaces sentiment
trends and issue drivers through an interactive dashboard.

> **AI Major Capstone Project — Option 1: AI Customer Sentiment Analysis Dashboard**

```
                    ┌──────────────────────────────────────────┐
   Customer         │  React Dashboard (Vite + Tailwind)       │
   feedback         │  Overview · Explorer · Insights ·        │
      │             │  Analyze · Model Card                    │
      │             └───────────────┬──────────────────────────┘
      │                             │ REST + Server-Sent Events
      ▼                             ▼
┌─────────────┐            ┌────────────────────────────────────┐
│  CSV import │───────────▶│  FastAPI                           │
│  REST API   │            │  ├─ /api/predict     inference     │
│  Live demo  │            │  ├─ /api/feedback    ingest+store  │
└─────────────┘            │  ├─ /api/analytics/* aggregations  │
                           │  └─ /api/stream      live feed     │
                           └────────┬──────────────┬────────────┘
                                    │              │
                    ┌───────────────▼───┐   ┌──────▼──────────────┐
                    │ DistilBERT        │   │  MongoDB Atlas      │
                    │ (fine-tuned)      │   │  aggregation        │
                    │ + TF-IDF fallback │   │  pipelines          │
                    └───────────────────┘   └─────────────────────┘
```

---

## What it does

| Capability | How |
|---|---|
| **NLP preprocessing** | Two purpose-built cleaning profiles + a separate topic-extraction path. No NLTK/spaCy downloads — runs identically offline. |
| **Transformer model** | DistilBERT fine-tuned with a hand-written PyTorch loop (warmup schedule, decoupled weight decay, gradient clipping, dynamic padding, best-checkpoint selection on validation macro-F1). |
| **Baseline comparison** | TF-IDF (word + character n-grams) + Logistic Regression, so the transformer's gain is a measured number rather than an assertion. |
| **Real-time API** | FastAPI with batched inference, word-level explanations, CSV bulk import and an SSE live feed. |
| **Database** | MongoDB Atlas. Every trend, distribution and issue ranking is an aggregation pipeline — no counting in application code. |
| **Dashboard** | React SPA: KPI tiles, trend chart, sentiment mix, channel breakdown, issue ranking, keyword analysis, live feed, model card, and an inference playground. |
| **Explainability** | Leave-one-out occlusion attribution — model-agnostic, gradient-free, and explainable to a non-technical stakeholder. |

---

## Results

Evaluated on a **held-out test set of 2,000 reviews** that neither model saw during
training, and which was not used for checkpoint selection.

<!-- RESULTS:START -->
Evaluated on **2,000 held-out test reviews**.

| Metric | TF-IDF + Logistic Regression | DistilBERT (fine-tuned) | Δ |
|---|---|---|---|
| Accuracy | 0.7250 | **0.7170** | -0.0080 |
| **F1 (macro)** | 0.7237 | **0.7180** | -0.0057 |
| F1 (weighted) | 0.7238 | **0.7180** | -0.0057 |

Fine-tuning is worth **-0.6 points** of macro-F1 (-0.8% relative) over a baseline that trains in seconds. The transformer took **105 minutes** on cpu.

### Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.789 | 0.769 | 0.779 | 667 |
| neutral | 0.599 | 0.622 | 0.610 | 666 |
| positive | 0.769 | 0.760 | 0.765 | 667 |

### Confusion matrix

Rows are the true label, columns the prediction.

| | pred negative | pred neutral | pred positive |
|---|---|---|---|
| **true negative** | 513 | 135 | 19 |
| **true neutral** | 119 | 414 | 133 |
| **true positive** | 18 | 142 | 507 |

Single-sample inference latency: **88.86 ms** on cpu.
<!-- RESULTS:END -->

---

## Quickstart

### Prerequisites

- Python 3.10+ (developed on 3.13)
- Node.js 20+
- A free [MongoDB Atlas](https://cloud.mongodb.com) cluster (M0 tier is enough)

### 1 · Get an Atlas connection string

1. Create a free **M0** cluster at <https://cloud.mongodb.com>
2. **Database Access** → add a database user (username + password)
3. **Network Access** → add IP `0.0.0.0/0` (any IP) so the API can connect
4. **Connect → Drivers** → copy the `mongodb+srv://…` string

### 2 · Set up

**Windows (PowerShell)**

```powershell
.\scripts\quickstart.ps1
```

**macOS / Linux**

```bash
chmod +x scripts/quickstart.sh
./scripts/quickstart.sh
```

Or manually:

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # macOS / Linux

pip install -r requirements.txt
cp .env.example .env             # then paste your MONGODB_URI into .env

python -m src.dataset --prepare  # streams and builds the train/val/test splits
python -m src.train_baseline     # ~2 minutes
python -m src.train_transformer  # ~2 hours on CPU, minutes on GPU

cd dashboard && npm install && cd ..
```

> **No internet, or in a hurry?** `python -m src.dataset --prepare --synthetic`
> generates a corpus locally, and the API runs fine with only the baseline trained.

### 3 · Run

```bash
python -m api.seed --count 600            # populate the dashboard with demo data
uvicorn api.main:app --reload --port 8020   # terminal 1  → http://localhost:8020/docs
cd dashboard && npm run dev               # terminal 2  → http://localhost:5173
```

### 4 · Or run the whole stack in Docker

```bash
docker compose up --build
# dashboard  http://localhost:8080
# API docs   http://localhost:8020/docs
```

---

## Project layout

```
├── src/                         # ML package
│   ├── config.py                #   paths, label space, hyper-parameters
│   ├── preprocessing.py         #   cleaning profiles + issue taxonomy
│   ├── dataset.py               #   streaming download, stratified splits
│   ├── train_baseline.py        #   TF-IDF + Logistic Regression
│   ├── train_transformer.py     #   DistilBERT fine-tuning loop
│   └── metrics.py               #   shared evaluation + reports/metrics.json
│
├── api/                         # FastAPI service
│   ├── main.py                  #   app, lifespan, error handling
│   ├── inference.py             #   4-tier model loading, batching, explanations
│   ├── db.py                    #   Motor client + every aggregation pipeline
│   ├── schemas.py               #   Pydantic request/response contracts
│   ├── deps.py                  #   auth + shared filter parameters
│   ├── seed.py                  #   demo data generator
│   └── routers/                 #   health · predict · feedback · analytics · stream
│
├── dashboard/                   # React + Vite + Tailwind SPA
│   └── src/
│       ├── components/charts/   #   Recharts components (validated palette)
│       ├── pages/               #   Overview · Explorer · Insights · Analyze · Model
│       ├── hooks/               #   React Query data hooks, SSE live feed
│       └── lib/                 #   API client, theme, filter state, formatting
│
├── notebooks/                   # 01_sentiment_analysis_capstone.ipynb
├── tests/                       # pytest suite (85 unit + 23 integration)
├── reports/                     # metrics.json, FINAL_REPORT.md, training logs
└── docker-compose.yml           # API + dashboard
```

---

## API reference

Interactive docs at `http://localhost:8020/docs`.

> The API defaults to **port 8020**, not 8000. 8000 collides with something on
> most machines; `API_PORT` in `.env` controls it and the dashboard's dev proxy
> reads the same value, so changing it in one place is enough.

### Inference (no database required)

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/predict` | Classify one text. `{"explain": true}` adds word-level attribution. |
| `POST` | `/api/predict/batch` | Classify up to 200 texts in one padded forward pass. |
| `POST` | `/api/explain` | Attribution only. |

### Feedback

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/feedback` | Classify **and** store one item. |
| `POST` | `/api/feedback/batch` | Bulk classify + store. |
| `POST` | `/api/feedback/upload` | Import a CSV (needs a `text` column). |
| `GET` | `/api/feedback` | Paginated list with filters, search and sorting. |
| `DELETE` | `/api/feedback/{id}` | Remove one document. |

### Analytics

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/analytics/summary` | Volume, sentiment mix, Net Sentiment Score, mean confidence. |
| `GET` | `/api/analytics/trends` | Time series (`granularity=hour\|day\|week\|month`). |
| `GET` | `/api/analytics/sources` | Split by channel. |
| `GET` | `/api/analytics/products` | Split by product. |
| `GET` | `/api/analytics/issues` | Categories ranked by negative volume. |
| `GET` | `/api/analytics/keywords` | Word-frequency data. |
| `GET` | `/api/analytics/filters` | Distinct values for the dashboard dropdowns. |

### System

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness plus database and model status. |
| `GET` | `/api/model/info` | Which model is currently serving. |
| `GET` | `/api/model/metrics` | Offline evaluation results. |
| `GET` | `/api/stream` | Server-Sent Events feed of newly classified feedback. |

**Example**

```bash
curl -X POST http://localhost:8020/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The delivery was late and the box arrived damaged.","explain":true}'
```

```json
{
  "label": "negative",
  "confidence": 0.9612,
  "scores": { "negative": 0.9612, "neutral": 0.0281, "positive": 0.0107 },
  "categories": ["Delivery & Logistics", "Product Quality"],
  "model": "DistilBERT (fine-tuned) - distilbert-sentiment",
  "latency_ms": 41.8,
  "explanation": [{ "token": "damaged", "weight": 0.2418, "normalised": 1.0 }]
}
```

---

## Design decisions

Eleven choices that shaped this project, and the reasoning behind each.

**1 · Three classes, not two.** Most real feedback is mixed — *"the food was good but
we waited 40 minutes"*. Forcing it to a pole either inflates the complaint rate or hides
it. Keeping neutral separate is what makes the negative count trustworthy enough to
escalate on.

**2 · Macro-F1 is the headline metric.** Neutral is the hardest class. Accuracy would let
a model that fails completely on neutral still look strong by getting the two easy classes
right; macro-F1 does not.

**3 · Negations are never removed.** The standard English stopword list contains `not`,
`no`, `never`. Removing them turns *"not good"* into *"good"* — a label flip introduced by
preprocessing. They are explicitly subtracted from the stopword set.

**4 · Two cleaning profiles, plus a third stopword list.** TF-IDF wants aggressive
normalisation; DistilBERT wants punctuation and casing preserved. Keyword extraction wants
the opposite of both — `not` is an essential *feature* but a useless *topic*.

**5 · A baseline, always.** *"We fine-tuned BERT and got 0.87"* is a number with no
reference point. The baseline turns the transformer's value into a measurement, doubles as
a data sanity check, and serves as a production fallback.

**6 · A hand-written training loop.** `transformers.Trainer` hides exactly the mechanics a
capstone is meant to demonstrate. The loop makes its decisions visible: decoupled weight
decay (never on LayerNorm), linear warmup (a randomly-initialised head would otherwise
damage the pretrained encoder), clipping after unscaling, dynamic padding, and checkpoint
selection on validation macro-F1 rather than training loss.

**7 · The token budget was measured, not assumed — and getting it wrong cost real points.**
The first fine-tuning run used a 128-token window, chosen from the *word*-length
distribution, and **lost to the TF-IDF baseline**. WordPiece splits review text into
roughly 1.4 tokens per word, so 128 tokens covered only ~52% of the corpus: the
transformer was reading half of every long review while the bag-of-words model read all of
it. Re-running at 256 tokens (~84% coverage) with everything else held constant is the
controlled experiment. Both runs are kept in `reports/metrics.json` — the losing one on
purpose, because the gap between them is the clearest evidence here that a preprocessing
decision can outweigh the choice of model.

**8 · Aggregation pipelines, not application-side counting.** Pulling 100k documents into
Python to count them does not survive real volume. Every dashboard number is computed by
MongoDB.

**9 · One filter builder for both surfaces.** The KPI cards and the record table share a
single query builder, so "42 negative" on the Overview and "42 rows" in the Explorer are
guaranteed to be the same query. Disagreeing numbers is what makes a dashboard untrusted.

**10 · Graceful degradation everywhere.** The model loader falls through four backends
(fine-tuned checkpoint → Hub model → TF-IDF baseline → lexicon). A missing database leaves
`/api/predict` working and reports *why* through `/health`. A live demo should never die on
a flaky network.

**11 · The chart palette was measured, not chosen.** Sentiment is a diverging scale, and
the obvious red/green encoding measures **ΔE 4.1** under deuteranopia — indistinguishable
for roughly 1 in 12 men. Red ↔ grey ↔ **blue** measures **ΔE 8.7 (light) / 8.5 (dark)**,
clearing the threshold in both themes. Every chart also carries a legend, direct labels
and a table view, so meaning never depends on colour alone.

---

## Testing

```bash
pytest tests -q                # 85 unit tests; no database or model download needed
ruff check src api tests       # lint
cd dashboard && npm run build  # dashboard compiles
```

The unit suite runs against the **degraded** path — no `MONGODB_URI`, no checkpoint —
because that is the configuration most likely to break silently. It covers the API
contract, validation rules, error ordering, the shared filter builder and the
preprocessing invariants.

A further **23 integration tests** exercise the MongoDB aggregation pipelines directly,
because those are query language rather than Python and cannot be unit-tested. They are
skipped unless a database is provided:

```bash
docker run -d --name mongo-test -p 27017:27017 mongo:7
MONGODB_TEST_URI="mongodb://127.0.0.1:27017/?directConnection=true"   pytest tests/test_integration.py -v
```

CI (`.github/workflows/ci.yml`) runs both suites, builds the dashboard and both Docker
images, and runs the whole ML pipeline end to end on the synthetic generator — so it never
depends on the Hugging Face Hub being reachable.

---

## Configuration

All settings come from `.env` (see `.env.example`).

| Variable | Default | Purpose |
|---|---|---|
| `MONGODB_URI` | — | Atlas connection string. **Required** for persistence. |
| `MONGODB_DB` | `pulseai` | Database name. |
| `MODEL_DIR` | `models/distilbert-sentiment` | Fine-tuned checkpoint. |
| `FALLBACK_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Used when no local checkpoint exists. Set empty to disable. |
| `MAX_SEQ_LENGTH` | `128` | Token budget. Must match training. |
| `API_PORT` | `8020` | Host port for the API. The dashboard's dev proxy follows it. |
| `API_KEY` | *(empty)* | When set, write endpoints require an `X-API-Key` header. |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins. |
| `TRAIN_SIZE` / `EPOCHS` / `BATCH_SIZE` / `LEARNING_RATE` | `12000` / `2` / `16` / `3e-5` | Training hyper-parameters. |

---

## Deployment

**API** — any container host (Render, Railway, Fly.io, Cloud Run):

```bash
docker build -t pulseai-api .
docker run -p 8020:8000 --env-file .env pulseai-api
```

Set `MONGODB_URI`, `CORS_ORIGINS` and `API_KEY` in the platform's environment settings.
One worker per container: each worker loads its own copy of the model, so scale with
replicas rather than `--workers`.

**Dashboard** — any static host (Vercel, Netlify, Cloudflare Pages):

```bash
cd dashboard && npm run build     # outputs dist/
```

Set `VITE_API_BASE` to the deployed API URL if the dashboard is served from a different
origin. When both run behind the bundled nginx config, requests stay same-origin and no
CORS configuration is needed at all.

---

## Limitations

1. **Label noise at the 3-star boundary.** Neutral is defined by a star-rating mapping, and
   3-star reviews genuinely mix praise and complaint. This caps neutral-class performance;
   removing that ceiling requires human annotation, not better modelling.
2. **Domain.** Trained on English business reviews. Short social posts, code-mixed text
   (Hinglish and similar) and technical support tickets are out of distribution and
   unmeasured.
3. **Sarcasm.** *"Great, another week without my order"* remains the known hard case.
4. **Drift.** Language and product concerns change. Without periodic re-evaluation on fresh
   labelled samples the model quietly degrades; the confidence gap between correct and
   incorrect predictions is the cheapest early warning available.
5. **Scale.** Benchmarks are CPU-only at moderate volume. High throughput would want GPU
   serving or an ONNX/quantised export.

**Intended use.** This system belongs in front of a *human* triage workflow — routing and
ranking so a team reads the right 5% of feedback first. It should not take automated action
on an individual customer, and low-confidence predictions should always be reviewed.

---

## Documentation

- **[`notebooks/01_sentiment_analysis_capstone.ipynb`](notebooks/01_sentiment_analysis_capstone.ipynb)** — the full analysis: EDA, preprocessing, both models, evaluation, error analysis, explainability, benchmarks
- **[`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)** — analytical report: performance, insights, business implications
- **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — system design and data flow
- **[`docs/LEARNING_PATH.md`](docs/LEARNING_PATH.md)** — a staged route through the codebase: read it, run it, break it, explain it
- **[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)** — timed shot list for the demonstration video

---

*Built for the LaunchED Global AI Major Capstone Project.*
