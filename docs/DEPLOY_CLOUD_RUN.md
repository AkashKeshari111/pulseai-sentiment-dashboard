# Deploying the API on Google Cloud Run

Cloud Run is the recommended host for this API. It is the only widely available option
that is **genuinely free at demo traffic** *and* gives the container enough memory.

## Why this one

The API's measured resident memory is **601 MB** — PyTorch plus a 67M-parameter
DistilBERT. That single number rules out most free tiers:

| Platform | Free RAM | Fits 601 MB? |
|---|---|---|
| Render free | 512 MB | ❌ OOM |
| Koyeb free | 512 MB | ❌ OOM |
| Northflank free | 512 MB | ❌ OOM |
| Hugging Face Docker Space | — | ❌ requires PRO ($9/mo) |
| **Cloud Run** | configurable to 1–2 GB | ✅ |

Cloud Run's free tier covers **2M requests, 360,000 GiB-seconds of memory and 180,000
vCPU-seconds per month**. A portfolio demo uses a rounding error of that. It also
**scales to zero**, so an idle service costs nothing at all.

> A card is required to enable billing on the Google Cloud account. At this scale
> nothing is charged — but set a budget alert (step 6) so that is guaranteed rather
> than assumed.

---

## One-time setup

### 1 · Create the project

<https://console.cloud.google.com/projectcreate> → name it `pulseai` → **Create**.

Then enable billing: **Billing → Link a billing account** (this is what unlocks the free
tier; it does not start charging).

### 2 · Install the CLI

Download from <https://cloud.google.com/sdk/docs/install> and then:

```bash
gcloud auth login
gcloud config set project pulseai          # or whatever you named it
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 3 · Publish the model to the Hub (already done)

The 260 MB checkpoint is not in git, so the container pulls it at startup:

<https://huggingface.co/akashkeshari111/pulseai-distilbert-sentiment>

### 4 · Deploy

From the repository root:

```bash
gcloud run deploy pulseai-api \
  --source . \
  --region asia-south1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 8 \
  --min-instances 0 \
  --max-instances 3 \
  --set-env-vars "MONGODB_DB=pulseai,MODEL_DIR=/nonexistent,FALLBACK_MODEL=akashkeshari111/pulseai-distilbert-sentiment,MAX_SEQ_LENGTH=256,HF_HOME=/tmp/hf" \
  --set-env-vars "^##^CORS_ORIGINS=https://your-project.vercel.app" \
  --update-secrets "MONGODB_URI=pulseai-mongodb-uri:latest"
```

Notes on the flags that matter:

- **`--memory 2Gi`** — 1 GiB is enough at rest but the model load spikes above it. 2 GiB
  is still inside the free allowance because you are billed for memory *while a request is
  being served*, not for the ceiling you set.
- **`--min-instances 0`** — scales to zero. The trade is a cold start of roughly 30–60 s
  while the model downloads and loads. Acceptable for a demo; set `1` if you want it
  always warm, but that leaves the free tier.
- **`--concurrency 8`** — one container serves 8 simultaneous requests. Inference is
  CPU-bound, so a higher number would just queue.
- **`--timeout 300`** — the first request after a cold start has to wait for the model.
- **`MODEL_DIR=/nonexistent`** — deliberate. It makes the loader fall through to
  `FALLBACK_MODEL` and pull your published checkpoint. See `api/inference.py`.
- **`HF_HOME=/tmp/hf`** — the container filesystem is read-only apart from `/tmp`.

### 5 · Store the database URI as a secret, not an env var

```bash
echo -n "mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority" \
  | gcloud secrets create pulseai-mongodb-uri --data-file=-

gcloud secrets add-iam-policy-binding pulseai-mongodb-uri \
  --member="serviceAccount:$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

Environment variables are visible to anyone with read access to the service. A database
credential belongs in Secret Manager.

### 6 · Set a budget alert

<https://console.cloud.google.com/billing/budgets> → **Create budget** → amount **₹100**
→ alert at 50% and 100%.

You should never reach it. This is how you find out immediately if something unexpected
starts running.

---

## Verify

```bash
URL=$(gcloud run services describe pulseai-api --region asia-south1 --format='value(status.url)')

curl "$URL/health"          # first call may take ~60s (cold start + model download)

curl -X POST "$URL/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"The delivery was late and the box arrived damaged."}'
```

Check in the `/health` response that:

- `status` is `ok`
- `database.connected` is `true`
- `model.source` is `huggingface-hub` — **not** `builtin`, which would mean it fell all
  the way through to the lexicon

---

## Point the dashboard at it

In **Vercel → Settings → Environment Variables**:

```
VITE_API_BASE = https://pulseai-api-xxxxx.a.run.app
```

Then redeploy the dashboard, and set the matching value on the API:

```bash
gcloud run services update pulseai-api --region asia-south1 \
  --set-env-vars "CORS_ORIGINS=https://your-project.vercel.app"
```

Both sides are required. Setting only one produces a dashboard that loads but shows no
data — the most common failure with this split-hosting setup.

---

## If you would rather not use a card at all

Two honest alternatives:

1. **Run the API locally for the demo.** The video is recorded on your machine anyway.
   Deploy only the dashboard, and be upfront in the README that the API runs locally.
   This is what a large share of student projects do, and it costs nothing.

2. **Shrink the model so a 512 MB tier fits.** Export to ONNX and quantize to INT8:
   the weights drop from 268 MB to roughly 67 MB, `onnxruntime` is far lighter than
   PyTorch, and total memory lands near 280 MB. Render's free tier then works. It is real
   work, and it is also a genuinely good thing to be able to talk about.
