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

COPY --chown=pulseai:pulseai docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

COPY --chown=pulseai:pulseai src/ ./src/
COPY --chown=pulseai:pulseai api/ ./api/
COPY --chown=pulseai:pulseai reports/metrics.json ./reports/metrics.json

# The fine-tuned checkpoint is a build input, not a repo artefact. Train first
# (`python -m src.train_transformer`), then build. If models/ is absent the API
# still starts and falls back - see api/inference.py.
COPY --chown=pulseai:pulseai models/ ./models/

RUN mkdir -p /app/.cache/huggingface && chown -R pulseai:pulseai /app/.cache

USER pulseai

# 8000 locally; Hugging Face Spaces fixes it at 7860 and Render injects $PORT.
# The entrypoint resolves whichever is set, so this is documentation only.
EXPOSE 8000 7860

# Checks the app's own dependency-aware endpoint rather than just "is the port
# open", so a container that cannot reach Atlas is visibly degraded. The port is
# resolved the same way the entrypoint resolves it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,sys,urllib.request; p=os.getenv('PORT') or os.getenv('API_PORT') or '8000'; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=4).status == 200 else 1)"

# One worker by default: each worker loads its own copy of the model into
# memory. Scale with replicas rather than workers unless the host has the RAM.
# The entrypoint picks the port from PORT / API_PORT / the 8000 default, so one
# image runs unchanged on Hugging Face Spaces (7860), Render ($PORT) and locally.
ENTRYPOINT ["./docker-entrypoint.sh"]
