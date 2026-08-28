# Deployment

Two pieces deploy separately: the **dashboard** (static, goes on Vercel) and the **API**
(a container that loads a PyTorch model, needs real memory).

---

## The one thing to decide first: where the model lives

The fine-tuned checkpoint is **260 MB** and is deliberately **not** in git — model weights
are build artefacts, and committing them makes every clone pay for them forever.

That leaves the deployed API needing to get the model from somewhere. Three options, in
order of preference:

### Option A — Hugging Face Hub (recommended)

Publish the checkpoint once; every deployment pulls it by name. This is how model
distribution is normally done, it is free, and it keeps the repo small.

```bash
pip install huggingface_hub
huggingface-cli login          # token from https://huggingface.co/settings/tokens

python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = "AkashKeshari111/pulseai-distilbert-sentiment"   # your username
api.create_repo(repo, exist_ok=True)
api.upload_folder(folder_path="models/distilbert-sentiment", repo_id=repo)
print("published:", repo)
PY
```

Then in the deployed environment set:

```
MODEL_DIR=/nonexistent                                     # forces the fallback path
FALLBACK_MODEL=AkashKeshari111/pulseai-distilbert-sentiment
```

`api/inference.py` walks its backends in order, finds no local checkpoint, and loads yours
from the Hub instead. Nothing in the code changes.

### Option B — bake it into the Docker image

Build the image locally *after* training, so `models/` is copied in, then push the image to
a registry. Larger image, slower deploys, but zero runtime download.

### Option C — accept the public fallback model

Leave `FALLBACK_MODEL` at its default. The API works, but it is serving
`cardiffnlp/twitter-roberta-base-sentiment-latest` — a good model, but **not yours**, and
the Model Card page will say so. Fine for a smoke test, wrong for a portfolio piece.

---

## API hosting — read the memory numbers before choosing

| | RAM | Cost | Verdict |
|---|---|---|---|
| **Hugging Face Spaces** | 16 GB | **free** | ✅ Best free option for this project |
| Render Starter | 512 MB | $7/mo | ⚠️ Very tight; may still OOM |
| Render Standard | 2 GB | $25/mo | ✅ Comfortable |
| Railway | 512 MB–8 GB | trial credits | ✅ Works while credits last |
| **Render Free** | 512 MB | free | ❌ **Will be OOM-killed** |

PyTorch plus a 67M-parameter model sits around **800 MB–1 GB** resident. Anything with a
512 MB ceiling is a coin flip at best.

### Recommended: Hugging Face Spaces (free, enough RAM)

1. Create a Space at <https://huggingface.co/new-space> → SDK **Docker** → hardware
   **CPU basic (free)**
2. Push this repo to it:
   ```bash
   git remote add space https://huggingface.co/spaces/AkashKeshari111/pulseai-api
   git push space main
   ```
3. In **Settings → Variables and secrets** add `MONGODB_URI`, `FALLBACK_MODEL`,
   `MAX_SEQ_LENGTH=256`, and `CORS_ORIGINS` (your Vercel URL)
4. Spaces expect the container to listen on **7860**, so add to the `Dockerfile`:
   ```dockerfile
   ENV API_PORT=7860
   EXPOSE 7860
   CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
   ```

### If you use Render anyway

`render.yaml` in the repo root is a working blueprint set to the `starter` plan.

1. <https://dashboard.render.com> → **New → Blueprint** → pick the GitHub repo
2. Add the secret environment variables when prompted
3. First build takes ~10 minutes (PyTorch is a large install)

> Do **not** pick the free instance type. It deploys, passes the build, then dies on the
> first request when the model loads.

---

## Dashboard — Vercel

```bash
cd dashboard
vercel login
vercel --prod
```

Or through the web UI: **New Project → import the GitHub repo → set Root Directory to
`dashboard`**. `vercel.json` already sets the framework, SPA rewrites and cache headers.

### One required environment variable

The dashboard and the API are on different domains, so the same-origin proxy that works
locally does not apply. In **Vercel → Settings → Environment Variables**:

```
VITE_API_BASE = https://your-api-host.example.com
```

Then redeploy. Without it the dashboard calls its own domain for `/api/*` and every request
404s.

### And the matching CORS setting on the API

```
CORS_ORIGINS = https://your-project.vercel.app
```

Both sides must be set. Missing either one produces a dashboard that loads but shows no
data — the single most common deployment mistake with this stack.

---

## Post-deploy checklist

```bash
# 1. API is alive and knows which model it loaded
curl https://your-api-host/health

# 2. Inference works
curl -X POST https://your-api-host/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The delivery was late and the box arrived damaged."}'

# 3. Database is reachable from the deployed host
curl https://your-api-host/api/analytics/summary
```

- [ ] `/health` reports `"status": "ok"` and `database.connected: true`
- [ ] `model.source` is `local` or `huggingface-hub` — **not** the lexicon fallback
- [ ] The dashboard loads data, not empty states
- [ ] Atlas **Network Access** allows `0.0.0.0/0` (the deployed host's IP is not fixed)
- [ ] `API_KEY` is set on the API and `VITE_API_KEY` matches on the dashboard, if you want
      the write endpoints protected in public

---

## Seeding the deployed database

Run it locally against the same Atlas cluster — no need to run it on the server:

```bash
python -m api.seed --count 600 --reset
```

The dashboard reads from Atlas, so whatever is in the database shows up wherever the API is
hosted.

---

## Security before you make the repo public

- [ ] `.env` is **not** committed (`git check-ignore -v .env` must print a match)
- [ ] No connection string anywhere in tracked files:
      `git grep -n "mongodb+srv://" -- . ':!*.example' ':!docs'`
- [ ] Rotate the Atlas password if a connection string was ever pasted into a commit,
      an issue, or a screenshot — rotating is cheap, and a leaked string is not
- [ ] If you publish the demo, set `API_KEY` so strangers cannot write into your database
