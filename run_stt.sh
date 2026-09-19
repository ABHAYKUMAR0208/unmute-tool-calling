#!/bin/bash
set -ex
source .env
export HUGGING_FACE_HUB_TOKEN

source $HOME/.cargo/env
cd dockerless

[ -f pyproject.toml ] || wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/pyproject.toml
[ -f uv.lock ]        || wget https://raw.githubusercontent.com/kyutai-labs/moshi/9837ca328d58deef5d7a4fe95a0fb49c902ec0ae/rust/moshi-server/uv.lock

uv venv
source .venv/bin/activate
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
export CXXFLAGS="-include cstdint"

cd ..
cargo install --features cuda moshi-server@0.6.4
moshi-server worker --config services/moshi-server/configs/stt.toml --port 8090
