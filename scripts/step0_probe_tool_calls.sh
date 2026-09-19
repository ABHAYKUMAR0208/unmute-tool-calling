#!/usr/bin/env bash
# Step 0 — prove the LLM endpoint streams tool_calls before any loop code is written.
#
#   export KYUTAI_LLM_API_KEY=sk-or-...
#   ./scripts/step0_probe_tool_calls.sh
#
# NOTE: KYUTAI_LLM_URL must NOT include /v1 — get_openai_client() appends it.
# This script talks to the raw endpoint, so it adds /v1 itself.
set -uo pipefail

BASE="${KYUTAI_LLM_URL:-https://openrouter.ai/api}"
MODEL="${KYUTAI_LLM_MODEL:-meta-llama/llama-3.3-70b-instruct}"
KEY="${KYUTAI_LLM_API_KEY:-}"

if [ -z "$KEY" ]; then
  echo "FAIL: KYUTAI_LLM_API_KEY is not set." >&2
  exit 1
fi

echo "endpoint : $BASE/v1/chat/completions"
echo "model    : $MODEL"
echo "--- raw stream ---"

RAW=$(curl -sS --max-time 60 "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d "$(cat <<JSON
{
  "model": "$MODEL",
  "stream": true,
  "tool_choice": "auto",
  "messages": [
    {"role": "system", "content": "You are a hotel concierge. Use tools to act."},
    {"role": "user", "content": "Book me a taxi to the airport at 6 PM, room 305."}
  ],
  "tools": [{
    "type": "function",
    "function": {
      "name": "book_taxi",
      "description": "Book a taxi for a hotel guest.",
      "parameters": {
        "type": "object",
        "properties": {
          "room_number": {"type": "string"},
          "destination": {"type": "string"},
          "pickup_time": {"type": "string"}
        },
        "required": ["room_number", "destination", "pickup_time"]
      }
    }
  }]
}
JSON
)")

echo "$RAW" | head -40
echo "--- analysis ---"
echo "$RAW" | python3 - <<'PY'
import json, sys
raw = sys.stdin.read()
frags, name, tid, finish, keepalive, text = [], None, None, None, 0, ""
for line in raw.splitlines():
    line = line.strip()
    if not line.startswith("data:"):
        continue
    payload = line[5:].strip()
    if payload == "[DONE]":
        break
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        continue
    choices = obj.get("choices") or []
    if not choices:
        keepalive += 1
        continue
    ch = choices[0]
    d = ch.get("delta") or {}
    if d.get("content"):
        text += d["content"]
    for tc in (d.get("tool_calls") or []):
        fn = tc.get("function") or {}
        if fn.get("name"):
            name = fn["name"]
        if tc.get("id"):
            tid = tc["id"]
        if fn.get("arguments"):
            frags.append(fn["arguments"])
    if ch.get("finish_reason"):
        finish = ch["finish_reason"]

joined = "".join(frags)
print(f"tool name        : {name}")
print(f"tool call id     : {tid}")
print(f"arg fragments    : {len(frags)}  <-- >1 proves fragment accumulation is required")
print(f"joined arguments : {joined}")
print(f"finish_reason    : {finish}")
print(f"keep-alive chunks: {keepalive}")
print(f"leading text     : {text!r}")
ok = True
if not name:
    print("FAIL: no tool call streamed."); ok = False
else:
    try:
        print(f"parsed args      : {json.loads(joined)}")
    except Exception as e:
        print(f"FAIL: joined arguments are not valid JSON: {e}"); ok = False
if finish != "tool_calls":
    print(f"WARN: finish_reason is {finish!r}, expected 'tool_calls'.")
print("\nSTEP 0:", "GREEN" if ok else "RED")
sys.exit(0 if ok else 1)
PY
