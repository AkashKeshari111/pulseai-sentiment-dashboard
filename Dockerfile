# =============================================================================
# PulseAI API - production image
#
# Multi-stage: the build stage compiles wheels (torch and friends need a
# toolchain), the runtime stage copies only the installed packages. That keeps
# the shipped image free of compilers and roughly a third smaller.
# =============================================================================

# ---- build ------------------------------------------------------------------
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# CPU-only torch: the default wheel pulls ~2 GB of CUDA libraries that a CPU
# container can never use.
RUN pip install --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Cache the Hub download in the image layer rather than re-fetching per boot.
    HF_HOME=/app/.cache/huggingface \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Run as a non-root user: a compromised process should not own the filesystem.
RUN useradd --create-home --uid 1000 pulseai

WORKDIR /app
COPY --from=build /install /usr/local

COPY --chown=pulseai:pulseai src/ ./src/
COPY --chown=pulseai:pulseai api/ ./api/
COPY --chown=pulseai:pulseai reports/metrics.json ./reports/metrics.json

# The fine-tuned checkpoint is a build input, not a repo artefact. Train first
# (`python -m src.train_transformer`), then build. If models/ is absent the API
# still starts and falls back - see api/inference.py.
COPY --chown=pulseai:pulseai models/ ./models/

RUN mkdir -p /app/.cache/huggingface && chown -R pulseai:pulseai /app/.cache

USER pulseai
EXPOSE 8000

# Checks the app's own dependency-aware endpoint rather than just "is the port
# open", so a container that cannot reach Atlas is visibly degraded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# One worker by default: each worker loads its own copy of the model into
# memory. Scale with replicas rather than workers unless the host has the RAM.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
