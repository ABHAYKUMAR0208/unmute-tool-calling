#!/bin/bash
set -ex
# Resolve the repo root from this script's own location, so the checkout can
# live at /unmute, /unmute-tool-calling, or anywhere else.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO_DIR"
source .env
export HUGGING_FACE_HUB_TOKEN
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN
vllm serve google/gemma-3-1b-it \
  --host 0.0.0.0 \
  --port 8091 \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.6 \
  --rope-scaling '{"rope_type":"yarn","factor":8.0,"original_max_position_embeddings":8192}'
