# Live tool calling (fork addition)

This fork adds tool calling at the handler's LLM seam, so the agent can execute
guest requests *during* a call and speak a truthful confirmation, instead of a
post-call worker replaying the transcript.

Everything here is additive. STT, TTS, VAD, turn-taking, barge-in and the
WebSocket protocol are untouched.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `TOOL_API_BASE_URL` | `http://localhost:8081` | Base URL of the voice-ai-agent backend. **No trailing slash, no `/v1`.** |
| `TOOL_API_KEY` | unset | Sent as `X-Tool-Api-Key`. The backend rejects a missing/wrong key with 401. |
| `TOOL_TIMEOUT_S` | `3.0` | Hard client-side timeout per tool call. |
| `MAX_TOOL_ITERATIONS` | `3` | Bound on the pause-act-resume loop. |
| `KYUTAI_LLM_URL` | `http://localhost:8091` | LLM endpoint. For OpenRouter use `https://openrouter.ai/api` — the client appends `/v1` itself. |
| `KYUTAI_LLM_MODEL` | unset | e.g. `meta-llama/llama-3.3-70b-instruct`. |
| `KYUTAI_LLM_API_KEY` | unset | e.g. `sk-or-...`. |

> **Host topology.** If the fork runs on a RunPod GPU pod and the backend runs
> elsewhere, `localhost:8081` is wrong: set `TOOL_API_BASE_URL` to the backend's
> reachable URL and make sure `TOOL_API_KEY` is set on both sides. This is the
> one integration risk of building the two repos in parallel.

## How it works

`_generate_response_task` in `unmute/unmute_handler.py` runs a bounded loop:

1. Stream a completion with `tools=[...]`, `tool_choice="auto"`.
2. `TextDelta` events go down the existing speak path, unchanged
   (`rechunk_to_words` → `strip_unpronounceable` → `tts.send`, with the same
   interruption check). `ToolCallReady` events are collected, never spoken.
3. No tool call this pass → done.
4. Otherwise: cover the latency with filler, POST the calls to the backend,
   append the assistant `tool_calls` message and the `tool` results to a
   **local** message list, and loop.

The agent speaks the backend's `speakable` sentence. It never invents a
confirmation, and on `ok: false` it says so rather than claiming success.

### Strategy 1: tool state stays local

Unmute's `Chatbot` state machine only understands `system`/`user`/`assistant`
with **string** content:

- `conversation_state()` raises `RuntimeError` on an unknown role;
- `add_chat_message_delta` concatenates strings, so `content: None` breaks it;
- `preprocess_messages_for_llm` calls `.replace()` on every `content`.

So tool messages are appended **only** to the local `messages` list inside the
turn, never to `chatbot.chat_history`. Only spoken text reaches the history, by
the normal path. This is what keeps the state machine untouched — don't
"improve" it by teaching `chatbot.py` about tool roles.

### Interruption safety

`interrupt_bot()` removes the `llm` quest, cancelling the task with
`CancelledError`, which can land mid-tool. Side-effecting tools
(`SIDE_EFFECTING_TOOLS` in `unmute/llm/tools.py`) are wrapped in
`asyncio.shield`, so a barge-in stops the *speaking* but the booking still
completes — no half-done state, and nothing is spoken about it. Read-only tools
cancel freely.

### Filler speech

Two layers. The model is asked (via `TOOL_USE_INSTRUCTIONS` in
`system_prompt.py`) to speak a short lead-in before calling a tool, which
streams to the TTS while the HTTP call is in flight. If it calls a tool with no
lead-in at all, a canned phrase from `TOOL_FILLER_PHRASES` covers the gap. The
canned phrases promise nothing, so they stay honest even if the tool then fails.

## Files

| File | Role |
| --- | --- |
| `unmute/llm/tools.py` | Tool schemas + `dispatch()`. The fork's entire knowledge of tools. No business logic, no secrets. |
| `unmute/llm/llm_utils.py` | `ToolAwareLLMStream` yielding `TextDelta` / `ToolCallReady`. `VLLMStream` unchanged as the no-tools fallback. |
| `unmute/unmute_handler.py` | The pause-act-resume loop and `_run_tool_calls`. |
| `scripts/mock_tool_backend.py` | Contract-shaped mock for local testing. |
| `scripts/step0_probe_tool_calls.sh` | Proves the LLM endpoint streams tool calls. |

## Demo without a backend (`TOOL_SMOKE_TEST=1`)

The six real tools all POST to the voice-ai-agent backend. Before those routes
exist, set `TOOL_SMOKE_TEST=1` and the agent is offered exactly one tool,
`get_hotel_time`, which runs **in-process** — no HTTP, no key, no Repo A.

It's a good probe precisely because the model cannot know the time: if the
agent says the right time, the round-trip provably happened.

Dry-run it first, without audio:

```bash
python scripts/smoke_test_tool_call.py
```

Then the full voice call, with `TOOL_SMOKE_TEST=1` set for the backend process:

```bash
TOOL_SMOKE_TEST=1 ./run_backend.sh
```

Ask *"what time is it?"* and the agent should speak a lead-in, pause, and come
back with the real time. Unset the flag to return to the six real tools.

## Testing

Prove the endpoint streams tool calls before anything else:

```bash
export KYUTAI_LLM_URL=https://openrouter.ai/api
export KYUTAI_LLM_MODEL=meta-llama/llama-3.3-70b-instruct
export KYUTAI_LLM_API_KEY=sk-or-...
./scripts/step0_probe_tool_calls.sh
```

Run the loop against the mock backend:

```bash
TOOL_API_KEY=secret python scripts/mock_tool_backend.py
```

```bash
TOOL_API_BASE_URL=http://localhost:8081 TOOL_API_KEY=secret ./run_backend.sh
```

Force the honest-failure path with `MOCK_FAIL=book_taxi` on the mock.

Unit tests:

```bash
pytest tests/test_tool_call_stream.py tests/test_tools.py tests/test_tool_calling_loop.py
```

## Per-session tools

`session.update` now carries an optional `tools` field alongside `instructions`
and `voice`. Omitting it keeps the default hotel tool set; `[]` disables tool
calling for that session.
