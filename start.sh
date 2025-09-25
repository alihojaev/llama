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

echo "[start.sh] Starting FastAPI with Uvicorn on 0.0.0.0:7860"
exec python3.8 -m uvicorn /workspace/app:app --host 0.0.0.0 --port 7860


