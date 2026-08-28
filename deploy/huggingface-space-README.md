---
title: PulseAI Sentiment API
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Real-time customer sentiment classification API (fine-tuned DistilBERT)
---

# PulseAI — Sentiment Intelligence API

Real-time customer sentiment classification backed by a fine-tuned DistilBERT model.

- **Interactive docs:** [`/docs`](./docs)
- **Health and model status:** [`/health`](./health)

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/predict` | Classify one text; `{"explain": true}` adds word-level attribution |
| `POST` | `/api/predict/batch` | Classify up to 200 texts in one pass |
| `POST` | `/api/feedback` | Classify **and** store |
| `GET` | `/api/analytics/summary` | Volume, sentiment mix, Net Sentiment Score |
| `GET` | `/api/stream` | Server-Sent Events feed of newly classified feedback |

## Example

```bash
curl -X POST https://<your-space>.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The delivery was late and the box arrived damaged."}'
```

## Required Space secrets

Set these under **Settings → Variables and secrets**:

| Name | Value |
|---|---|
| `MONGODB_URI` | your MongoDB Atlas connection string |
| `MONGODB_DB` | `pulseai` |
| `FALLBACK_MODEL` | the Hub repo holding the fine-tuned checkpoint |
| `MODEL_DIR` | `/nonexistent` — forces the loader to use `FALLBACK_MODEL` |
| `MAX_SEQ_LENGTH` | `256` |
| `CORS_ORIGINS` | the dashboard's Vercel URL |

Source: <https://github.com/AkashKeshari111/pulseai-sentiment-dashboard>
