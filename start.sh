#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/workspace/lama/big-lama"
ZIP_URL="https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip"
ZIP_PATH="/workspace/big-lama.zip"

echo "[start.sh] Ensuring Big LaMa model is present at $MODEL_DIR"
if [ ! -d "$MODEL_DIR" ]; then
  mkdir -p /workspace/lama
  echo "[start.sh] Downloading model from $ZIP_URL ..."
  curl -L -o "$ZIP_PATH" "$ZIP_URL"
  echo "[start.sh] Unzipping to /workspace/lama ..."
  unzip -o "$ZIP_PATH" -d /workspace/lama >/dev/null
  # Normalize path to /workspace/lama/big-lama
  if [ ! -d "$MODEL_DIR" ]; then
    FOUND_DIR=$(find /workspace/lama -maxdepth 2 -type d -name "big-lama" | head -n 1 || true)
    if [ -n "${FOUND_DIR:-}" ] && [ "$FOUND_DIR" != "$MODEL_DIR" ]; then
      mv "$FOUND_DIR" "$MODEL_DIR"
    fi
  fi
  rm -f "$ZIP_PATH"
fi

export PYTHONPATH="/workspace/lama:${PYTHONPATH:-}"
export TORCH_HOME="/workspace/lama"

# Default to serverless; set USE_HTTP=1 to run HTTP API
if [ "${USE_HTTP:-0}" = "1" ]; then
  PORT="${PORT:-7860}"
  echo "[start.sh] Starting FastAPI with Uvicorn on 0.0.0.0:${PORT}"
  exec python3 -m uvicorn app:app --app-dir /workspace --host 0.0.0.0 --port "${PORT}"
else
  echo "[start.sh] Starting Runpod Serverless handler"
  exec python3 /workspace/handler.py
fi


