#!/usr/bin/env sh
# Start the API on whichever port the host expects.
#
# Every platform names this differently and none of them let a JSON-form
# Dockerfile CMD expand a variable, so the resolution happens here:
#
#   PORT       Render, Railway, Cloud Run, Heroku
#   API_PORT   this project's own .env
#   7860       Hugging Face Spaces (fixed by the platform)
#
# One image therefore runs unmodified on all of them.
set -eu

PORT="${PORT:-${API_PORT:-8000}}"
WORKERS="${WEB_CONCURRENCY:-1}"

echo "[entrypoint] starting uvicorn on 0.0.0.0:${PORT} with ${WORKERS} worker(s)"

# exec so uvicorn becomes PID 1 and receives SIGTERM directly - without it the
# shell swallows the signal and the platform has to SIGKILL on every deploy.
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --workers "${WORKERS}" \
    --proxy-headers \
    --forwarded-allow-ips '*'
