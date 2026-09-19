"""End-to-end smoke test of the tool-calling loop, WITHOUT audio.

Everything a real voice call does except STT and TTS: a real LLM sees the real
tool schema, decides to call it, the real dispatch runs it, and the real LLM
speaks the result. No backend, no Repo A, no HTTP tools.

    export KYUTAI_LLM_URL=https://openrouter.ai/api      # note: no /v1
    export KYUTAI_LLM_MODEL=meta-llama/llama-3.3-70b-instruct
    export KYUTAI_LLM_API_KEY=sk-or-...
    python scripts/smoke_test_tool_call.py

Run this BEFORE the full voice demo: if it fails, the voice call will fail too,
and this tells you why in one screen instead of through a microphone.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Run from anywhere: put the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The smoke tool must be offered; set it before importing the fork's config.
os.environ.setdefault("TOOL_SMOKE_TEST", "1")

from unmute.llm.llm_utils import (  # noqa: E402
    TextDelta,
    ToolAwareLLMStream,
    ToolCallReady,
    get_openai_client,
)
from unmute.llm.tools import default_tool_schemas, dispatch  # noqa: E402

SYSTEM_PROMPT = (
    "You are a hotel concierge speaking with a guest. You can take real "
    "actions by calling tools. Before calling one, say a short natural "
    "lead-in out loud so the guest is not left in silence. Never invent an "
    "answer a tool should give you."
)
USER_TURN = "Hi, what time is it right now?"
MAX_ITERATIONS = 3


async def main() -> int:
    if not os.environ.get("KYUTAI_LLM_API_KEY"):
        print("FAIL: KYUTAI_LLM_API_KEY is not set.", file=sys.stderr)
        return 1

    tools = default_tool_schemas()
    print(f"model  : {os.environ.get('KYUTAI_LLM_MODEL', '(autoselect)')}")
    print(f"tools  : {[t['function']['name'] for t in tools]}")
    print(f"guest  : {USER_TURN}\n")

    client = get_openai_client()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TURN},
    ]

    spoken_all: list[str] = []
    tool_ran = False
    turn_started = time.monotonic()

    for _iteration in range(MAX_ITERATIONS):
        stream = ToolAwareLLMStream(client, temperature=0.3, tools=tools)
        calls: list[ToolCallReady] = []
        spoken = ""
        first_token_at = None

        async for event in stream.chat_completion_events(messages):
            if isinstance(event, TextDelta):
                if first_token_at is None:
                    first_token_at = time.monotonic() - turn_started
                spoken += event.text
            elif isinstance(event, ToolCallReady):
                calls.append(event)

        if spoken.strip():
            label = "agent (lead-in)" if calls else "agent"
            print(f"{label}: {spoken.strip()}")
            spoken_all.append(spoken.strip())
        if first_token_at is not None:
            print(f"          [first token {first_token_at:.2f}s]")

        if not calls:
            break

        # Execute each call exactly ONCE, then reuse the envelope. Calling
        # dispatch twice would double-book with a real side-effecting tool.
        envelopes = []
        for call in calls:
            print(f"\n  -> tool call: {call.name}({call.args})")
            started = time.monotonic()
            envelope = await dispatch(call.name, call.args, call_id=call.id)
            print(f"  <- {envelope}  [{time.monotonic() - started:.3f}s]\n")
            envelopes.append(envelope)
            tool_ran = True

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.args)},
                    }
                    for c in calls
                ],
            }
        )
        for call, envelope in zip(calls, envelopes, strict=True):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(envelope),
                }
            )

    total = time.monotonic() - turn_started
    final = spoken_all[-1] if spoken_all else ""

    print(f"\n{'=' * 60}")
    print(f"turn took {total:.2f}s")
    ok = True
    if not tool_ran:
        print("FAIL: the model never called the tool.")
        print("      Try a blunter prompt, or check the model supports tools.")
        ok = False
    if not final:
        print("FAIL: nothing was spoken.")
        ok = False
    print("SMOKE TEST:", "GREEN" if ok else "RED")
    return 0 if ok else 1


sys.exit(asyncio.run(main()))
