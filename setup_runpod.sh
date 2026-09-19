#!/bin/bash
set -e
# Resolve the repo root from this script's own location, so the checkout can
# live at /unmute, /unmute-tool-calling, or anywhere else.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing system dependencies ==="
apt-get update -qq && apt-get install -y pkg-config libssl-dev build-essential cmake libopus-dev tmux curl
echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
echo "=== Installing Rust/cargo ==="
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH=$HOME/.cargo/bin:$PATH
echo "=== Installing Node.js ==="
curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -
apt-get install -y nodejs
echo "=== Installing pnpm ==="
npm install -g pnpm
echo "=== Fixing vLLM ==="
pip install vllm==0.8.5 -q
pip install transformers==4.51.3 tokenizers==0.21.0 -q
python3 -c "
with open('/usr/local/lib/python3.11/dist-packages/vllm/transformers_utils/config.py', 'r') as f:
    content = f.read()
content = content.replace(\"raise ValueError(\\\"rope_scaling should have a 'rope_type' key\\\")\", \"rope_scaling['rope_type'] = 'yarn'\")
with open('/usr/local/lib/python3.11/dist-packages/vllm/transformers_utils/config.py', 'w') as f:
    f.write(content)
print('vLLM rope_scaling patched!')
"
echo "=== Building moshi-server (15-30 min) ==="
cd "$REPO_DIR"/dockerless
[ -f pyproject.toml ] || wget -q https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/pyproject.toml
[ -f uv.lock ] || wget -q https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/uv.lock
uv venv
source .venv/bin/activate
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export CXXFLAGS="-include cstdint"
cd "$REPO_DIR"
cargo install --features cuda moshi-server@0.6.4
/usr/bin/python3.11 -m pip install moshi -q
echo "=== Building frontend ==="
source "$REPO_DIR"/.env
cd "$REPO_DIR"/frontend
export NEXT_PUBLIC_BACKEND_URL="$BACKEND_URL"
pnpm install --silent
pnpm build
cp -r .next/static .next/standalone/.next/static 2>/dev/null || true
cp -r public .next/standalone/public 2>/dev/null || true
echo "=== SETUP COMPLETE - run: bash $REPO_DIR/start_all.sh ==="
