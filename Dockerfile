# =============================================================================
# PulseAI API — production image
#
# Two decisions make this image small enough for a free 512 MB container:
#
#   1. It installs `requirements-serve.txt`, not `requirements.txt`. Training
#      needs torch, transformers, datasets and scikit-learn; serving a quantized
#      ONNX model needs none of them. Measured resident memory:
#
#          torch + transformers ................ 601 MB
#          onnx via optimum (imports torch) .... 512 MB   <- no saving at all
#          onnxruntime + tokenizers ............ 141 MB
#
#   2. Model weights are fetched at boot from the Hub rather than baked in, so
#      neither git nor the image carries a 65 MB binary.
#
# Multi-stage: the build stage keeps pip's tooling out of the final image.
# =============================================================================

# ---- build ------------------------------------------------------------------
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements-serve.txt .

# No build-essential: every serving dependency ships a manylinux wheel.
RUN pip install --prefix=/install -r requirements-serve.txt

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Comments sit above the ENV rather than inside its line continuation, because
# builders disagree about whether a comment may appear mid-continuation.
#   HF_HOME          where the Hub cache lands; must be writable
#   MAX_SEQ_LENGTH   must match the value the model was fine-tuned with
#   ONNX_MODEL_REPO  the 65 MB quantized model, downloaded once at boot
#   ONNX_MODEL_DIR   checked first; point it at a baked-in copy to skip the
#                    download entirely
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    MAX_SEQ_LENGTH=256 \
    ONNX_MODEL_REPO=akashkeshari111/pulseai-distilbert-sentiment-onnx \
    ONNX_MODEL_DIR=/app/models/onnx

# Run as a non-root user: a compromised process should not own the filesystem.
RUN useradd --create-home --uid 1000 pulseai

WORKDIR /app
COPY --from=build /install /usr/local

COPY --chown=pulseai:pulseai docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

COPY --chown=pulseai:pulseai src/ ./src/
COPY --chown=pulseai:pulseai api/ ./api/
COPY --chown=pulseai:pulseai reports/metrics.json ./reports/metrics.json

# Model weights are not copied in — see ONNX_MODEL_REPO above.
RUN mkdir -p /app/.cache/huggingface /app/models \
    && chown -R pulseai:pulseai /app/.cache /app/models

USER pulseai

# 8000 locally; Hugging Face Spaces fixes it at 7860 and Render injects $PORT.
# The entrypoint resolves whichever is set, so this is documentation only.
EXPOSE 8000 7860

# Checks the app's own dependency-aware endpoint rather than just "is the port
# open", so a container that cannot reach Atlas is visibly degraded. The port is
# resolved the same way the entrypoint resolves it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,sys,urllib.request; p=os.getenv('PORT') or os.getenv('API_PORT') or '8000'; sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=4).status == 200 else 1)"

# One worker: each worker would load its own copy of the model. Scale with
# replicas rather than workers unless the host has RAM to spare. The entrypoint
# picks the port from PORT / API_PORT / the 8000 default, so one image runs
# unchanged on Spaces (7860), Render ($PORT) and locally.
ENTRYPOINT ["./docker-entrypoint.sh"]
