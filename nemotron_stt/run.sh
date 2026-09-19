#!/bin/bash
# Nemotron STT on port 8090, in place of Kyutai's STT (see nemotron_stt/server.py).
# It has its own virtualenv so NeMo's dependencies never touch the backend's.
# The first run installs NeMo and downloads the model: allow 10-15 minutes.
set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
[ -f .env ] && source .env
export HUGGING_FACE_HUB_TOKEN
unset PYTHONPATH

VENV=nemotron_stt/.venv
if [ ! -x "$VENV/bin/python" ]; then
  echo "First run: setting up Nemotron STT (NeMo install, several minutes)..."
  if ! ldconfig -p 2>/dev/null | grep -q libsndfile; then
    apt-get update && apt-get install -y libsndfile1 ffmpeg
  fi
  uv venv --python 3.11 "$VENV"
  uv pip install --python "$VENV/bin/python" Cython packaging
  uv pip install --python "$VENV/bin/python" -r nemotron_stt/requirements.txt
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$VENV/bin/python" -m nemotron_stt.server
