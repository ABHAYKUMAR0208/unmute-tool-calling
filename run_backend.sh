#!/bin/bash
set -ex
source .env

export KYUTAI_STT_URL=ws://localhost:8090
export KYUTAI_TTS_URL=ws://localhost:8089
export KYUTAI_LLM_URL=${KYUTAI_LLM_URL:-http://localhost:8091}
export KYUTAI_LLM_API_KEY=${KYUTAI_LLM_API_KEY:-}
export KYUTAI_LLM_MODEL=${KYUTAI_LLM_MODEL:-}
export HUGGING_FACE_HUB_TOKEN

# Tool calling. Set these in .env, or inline: TOOL_SMOKE_TEST=1 ./run_backend.sh
# TOOL_SMOKE_TEST=1 offers a single in-process tool and needs no backend, which
# is how the pipeline is demonstrated before the voice-ai-agent routes exist.
export TOOL_SMOKE_TEST=${TOOL_SMOKE_TEST:-}
export TOOL_API_BASE_URL=${TOOL_API_BASE_URL:-http://localhost:8081}
export TOOL_API_KEY=${TOOL_API_KEY:-}
export TOOL_TIMEOUT_S=${TOOL_TIMEOUT_S:-3.0}
export MAX_TOOL_ITERATIONS=${MAX_TOOL_ITERATIONS:-3}

# Pass your frontend URL so CORS accepts it
export CORS_EXTRA_ORIGINS="$FRONTEND_URL"

uv run uvicorn unmute.main_websocket:app \
  --host 0.0.0.0 \
  --port 8000 \
  --ws-per-message-deflate=false
