"""Client-run tools: the server asks the client over the WebSocket and waits.

The handler pulls in fastrtc and the audio stack, so this exercises the two
methods involved against a minimal stand-in rather than a full handler.
"""

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

if "fastrtc" not in sys.modules:
    # Only the names unmute_handler imports; none of them run in these tests.
    fastrtc = types.ModuleType("fastrtc")
    for name in ("AdditionalOutputs", "AsyncStreamHandler", "CloseStream"):
        setattr(fastrtc, name, type(name, (), {}))
    fastrtc.audio_to_float32 = lambda audio: audio
    fastrtc.wait_for_item = lambda queue, *a, **k: queue.get()
    sys.modules["fastrtc"] = fastrtc

try:
    from unmute import unmute_handler
except (ImportError, SyntaxError) as exc:  # audio deps or Python < 3.12
    pytest.skip(f"unmute_handler not importable here: {exc}", allow_module_level=True)

import unmute.openai_realtime_api_events as ora
from unmute.llm.llm_utils import ToolCallReady

Handler = unmute_handler.UnmuteHandler


def _stand_in():
    stand_in = SimpleNamespace(output_queue=asyncio.Queue(), _client_tool_waiters={})
    stand_in._run_client_tool = Handler._run_client_tool.__get__(stand_in)
    stand_in.resolve_client_tool = Handler.resolve_client_tool.__get__(stand_in)
    return stand_in


@pytest.mark.asyncio
async def test_client_tool_round_trip():
    handler = _stand_in()
    call = ToolCallReady(id="c1", name="submit_guest_requests", args={"requests": []})

    task = asyncio.create_task(handler._run_client_tool(call))
    request = await asyncio.wait_for(handler.output_queue.get(), 1)
    assert isinstance(request, ora.UnmuteToolCallRequest)
    assert (request.call_id, request.name, request.arguments) == ("c1", call.name, call.args)

    handler.resolve_client_tool("c1", {"ok": True, "speakable": "All done.", "data": {}})
    envelope = await asyncio.wait_for(task, 1)
    assert envelope["speakable"] == "All done."
    assert handler._client_tool_waiters == {}  # nothing left waiting


@pytest.mark.asyncio
async def test_malformed_client_result_is_never_spoken_as_success():
    handler = _stand_in()
    task = asyncio.create_task(
        handler._run_client_tool(ToolCallReady(id="c2", name="x", args={}))
    )
    await handler.output_queue.get()
    handler.resolve_client_tool("c2", {"ok": True})  # no speakable
    envelope = await asyncio.wait_for(task, 1)
    assert envelope["ok"] is False and envelope["error_code"] == "MALFORMED_CLIENT_RESULT"


@pytest.mark.asyncio
async def test_late_or_unknown_results_are_ignored():
    handler = _stand_in()
    handler.resolve_client_tool("nobody-asked", {"ok": True, "speakable": "x"})
    task = asyncio.create_task(
        handler._run_client_tool(ToolCallReady(id="c3", name="x", args={}))
    )
    await handler.output_queue.get()
    task.cancel()  # e.g. the outer timeout fired
    with pytest.raises(asyncio.CancelledError):
        await task
    assert handler._client_tool_waiters == {}
    handler.resolve_client_tool("c3", {"ok": True, "speakable": "too late"})  # no error


def test_session_update_carries_client_tools():
    update = ora.SessionUpdate.model_validate(
        {
            "type": "session.update",
            "session": {
                "allow_recording": False,
                "tools": [],
                "client_tools": ["submit_guest_requests"],
            },
        }
    )
    assert update.session.client_tools == ["submit_guest_requests"]
    # Older clients that don't send it keep working.
    old = ora.SessionUpdate.model_validate(
        {"type": "session.update", "session": {"allow_recording": False}}
    )
    assert old.session.client_tools is None
