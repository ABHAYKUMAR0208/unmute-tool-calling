#!/bin/bash
# Resolve the repo root from this script's own location, so the checkout can
# live at /unmute, /unmute-tool-calling, or anywhere else.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$REPO_DIR"/.env
tmux kill-session -t unmute 2>/dev/null || true
tmux new-session -d -s unmute -x 220 -y 50
tmux rename-window -t unmute:0 LLM
tmux send-keys -t unmute:0 "echo 'Using Groq API - no local LLM needed'" ENTER
tmux new-window -t unmute -n STT
# STT_ENGINE=nemotron (default): NVIDIA Nemotron streaming STT, better on Indian
# English (nemotron_stt/). STT_ENGINE=kyutai: the original Kyutai STT. Both
# serve the same protocol on port 8090, so nothing else changes.
if [ "${STT_ENGINE:-nemotron}" = "kyutai" ]; then
tmux send-keys -t unmute:1 "unset PYTHONPATH; unset LD_LIBRARY_PATH; cd $REPO_DIR && source .env && export HUGGING_FACE_HUB_TOKEN && source $HOME/.cargo/env && cd dockerless && source .venv/bin/activate && export LD_LIBRARY_PATH=\$(python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))') && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd .. && moshi-server worker --config services/moshi-server/configs/stt.toml --port 8090" ENTER
else
tmux send-keys -t unmute:1 "unset LD_LIBRARY_PATH; cd $REPO_DIR && ./nemotron_stt/run.sh" ENTER
fi
sleep 20
tmux new-window -t unmute -n TTS
tmux send-keys -t unmute:2 "unset PYTHONPATH; unset LD_LIBRARY_PATH; cd $REPO_DIR && source .env && export HUGGING_FACE_HUB_TOKEN && source $HOME/.cargo/env && cd dockerless && source .venv/bin/activate && export LD_LIBRARY_PATH=\$(python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))') && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd .. && uv run --locked --project ./dockerless moshi-server worker --config services/moshi-server/configs/tts.toml --port 8089" ENTER
sleep 20
tmux new-window -t unmute -n Backend
tmux send-keys -t unmute:3 "cd $REPO_DIR && ./run_backend.sh" ENTER
sleep 10
tmux new-window -t unmute -n Frontend
tmux send-keys -t unmute:4 "cd $REPO_DIR/frontend && PORT=3000 HOSTNAME=0.0.0.0 node .next/standalone/server.js" ENTER
echo "All services started! Open: https://${POD_ID}-3000.proxy.runpod.net"
