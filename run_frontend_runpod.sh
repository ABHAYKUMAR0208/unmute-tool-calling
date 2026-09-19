#!/bin/bash
# Resolve the repo root from this script's own location, so the checkout can
# live at /unmute, /unmute-tool-calling, or anywhere else.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -ex
source "$REPO_DIR"/.env

cd "$REPO_DIR"/frontend

export NEXT_PUBLIC_BACKEND_URL="$BACKEND_URL"

pnpm install
pnpm build

cp -r .next/static .next/standalone/.next/static 2>/dev/null || true
cp -r public .next/standalone/public 2>/dev/null || true

PORT=3000 HOSTNAME=0.0.0.0 node .next/standalone/server.js
