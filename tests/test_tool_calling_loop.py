"""Tests for the pause-act-resume loop in `_generate_response_task`.

These drive the real handler method with a fake LLM stream and a fake TTS. They
need the full runtime deps (fastrtc, redis, aiofiles...), so they run on the
deployment box rather than on a bare dev checkout.

What they pin down:
  - a tool call round-trips and the RESULT is what gets spoken;
  - Strategy 1: no tool message ever reaches chatbot.chat_history;
  - a failing tool is spoken as a failure, never as a confirmation;
  - the loop is bounded;
  - a barge-in mid-commit still completes the booking but speaks nothing.
"""

import asyncio
import json

import pytest

pytest.importorskip("fastrtc", reason="needs the full runtime deps")

import unmute.unmute_handler as uh  # noqa: E402
from unmute.llm.chatbot import Chatbot  # noqa: E402
from unmute.llm.llm_utils import TextDelta, ToolCallReady  # noqa: E402


class FakeTTS:
    def __init__(self):
        self.spoken: list[str] = []

    async def send(self, message):
        self.spoken.append(message if isinstance(message, str) else "<EOS>")

    def spoken_text(self) -> str:
        return "".join(w for w in self.spoken if w != "<EOS>")


class FakeQuest:
    def __init__(self, tts):
        self._tts = tts

    async def get(self):
        return self._tts


class FakeStream:
    """Yields a scripted list of events on each successive LLM pass."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.messages_seen: list[list[dict]] = []

    def __call__(self, client, temperature=1.0, tools=None):
        self.tools_seen = tools
        return self

    async def chat_completion_events(self, messages):
        self.messages_seen.append([dict(m) for m in messages])
        events = self.script[self.calls] if self.calls < len(self.script) else []
        self.calls += 1
        for event in events:
            yield event


def make_handler(monkeypatch, script, dispatch_impl):
    handler = uh.UnmuteHandler.__new__(uh.UnmuteHandler)
    handler.chatbot = Chatbot()
    handler.chatbot.chat_history = [
        {"role": "system", "content": "You are a concierge."},
        {"role": "user", "content": "Book a taxi to the airport at 6 PM, room 305."},
        {"role": "assistant", "content": ""},
    ]
    handler.output_queue = asyncio.Queue()
    handler.debug_dict = {"timing": {}, "connection": {}, "chatbot": {}}
    handler.tts_voice = "voice"
    handler.openai_client = object()
    handler.tool_schemas = [
        {"type": "function", "function": {"name": "book_taxi", "parameters": {}}}
    ]
    handler.tts_output_stopwatch = uh.Stopwatch(autostart=False)

    tts = FakeTTS()
    handler.tts_fake = tts

    async def start_up_tts(_i):
        return FakeQuest(tts)

    handler.start_up_tts = start_up_tts

    stream = FakeStream(script)
    monkeypatch.setattr(uh, "ToolAwareLLMStream", stream)
    monkeypatch.setattr(uh, "dispatch", dispatch_impl)
    handler.stream_fake = stream
    return handler


@pytest.mark.asyncio
async def test_tool_result_is_what_gets_spoken(monkeypatch):
    dispatched = []

    async def fake_dispatch(name, args, call_id=""):
        dispatched.append((name, args))
        return {
            "ok": True,
            "speakable": "Your taxi is booked, Raju will collect you at 6 PM.",
            "data": {"booking_id": "A1B2"},
        }

    handler = make_handler(
        monkeypatch,
        [
            [
                TextDelta("Sure, "),
                TextDelta("booking that now. "),
                ToolCallReady("c1", "book_taxi", {"room_number": "305"}),
            ],
            [TextDelta("Your taxi is booked, Raju will collect you at 6 PM.")],
        ],
        fake_dispatch,
    )
    await handler._generate_response_task()

    spoken = handler.tts_fake.spoken_text()
    assert dispatched and dispatched[0][0] == "book_taxi"
    assert handler.stream_fake.calls == 2  # pause, act, resume
    assert "Raju" in spoken
    assert "book_taxi" not in spoken  # the tool name is never spoken


@pytest.mark.asyncio
async def test_chat_history_never_sees_tool_messages(monkeypatch):
    """Strategy 1: Unmute's state machine must stay untouched."""

    async def fake_dispatch(name, args, call_id=""):
        return {"ok": True, "speakable": "Booked.", "data": {}}

    handler = make_handler(
        monkeypatch,
        [
            [ToolCallReady("c1", "book_taxi", {"room_number": "305"})],
            [TextDelta("That's booked for you.")],
        ],
        fake_dispatch,
    )
    await handler._generate_response_task()

    history = handler.chatbot.chat_history
    assert all(m["role"] in ("system", "user", "assistant") for m in history)
    assert all(isinstance(m.get("content"), str) for m in history)
    assert not any("tool_calls" in m for m in history)
    handler.chatbot.conversation_state()  # must not raise

    # ...but the LLM did see the tool round-trip, in the local list.
    second_pass = handler.stream_fake.messages_seen[1]
    assert any(m["role"] == "tool" for m in second_pass)
    assert any(m.get("tool_calls") for m in second_pass)
    tool_message = next(m for m in second_pass if m["role"] == "tool")
    assert json.loads(tool_message["content"])["ok"] is True


@pytest.mark.asyncio
async def test_failed_tool_is_spoken_as_a_failure(monkeypatch):
    async def failing_dispatch(name, args, call_id=""):
        return {
            "ok": False,
            "speakable": "I'm sorry, no drivers are free right now.",
            "error_code": "NO_DRIVERS_AVAILABLE",
            "data": {},
        }

    handler = make_handler(
        monkeypatch,
        [
            [ToolCallReady("c1", "book_taxi", {"room_number": "305"})],
            [TextDelta("I'm sorry, no drivers are free right now.")],
        ],
        failing_dispatch,
    )
    await handler._generate_response_task()

    spoken = handler.tts_fake.spoken_text().lower()
    assert "sorry" in spoken
    assert "booked" not in spoken  # never claims success


@pytest.mark.asyncio
async def test_filler_covers_a_silent_tool_call(monkeypatch):
    """If the model gives no lead-in, the guest still hears something."""

    async def fake_dispatch(name, args, call_id=""):
        return {"ok": True, "speakable": "Booked.", "data": {}}

    handler = make_handler(
        monkeypatch,
        [
            [ToolCallReady("c1", "book_taxi", {})],  # no TextDelta at all
            [TextDelta("All set.")],
        ],
        fake_dispatch,
    )
    await handler._generate_response_task()

    spoken = handler.tts_fake.spoken_text()
    assert spoken.strip()
    # The filler is spoken before the result, so something precedes "All set."
    assert spoken.index("All set") > 0


@pytest.mark.asyncio
async def test_plain_turn_is_unchanged(monkeypatch):
    """No tool call means exactly one LLM pass and no HTTP at all."""

    async def unused_dispatch(name, args, call_id=""):
        raise AssertionError("should not dispatch on a plain turn")

    handler = make_handler(monkeypatch, [[TextDelta("Good evening!")]], unused_dispatch)
    await handler._generate_response_task()

    assert handler.stream_fake.calls == 1
    assert "Good evening!" in handler.tts_fake.spoken_text()


@pytest.mark.asyncio
async def test_loop_is_bounded(monkeypatch):
    """A model that keeps calling tools cannot spiral."""

    async def always_dispatch(name, args, call_id=""):
        return {"ok": True, "speakable": "done", "data": {}}

    handler = make_handler(
        monkeypatch,
        [[ToolCallReady(f"c{i}", "book_taxi", {})] for i in range(10)],
        always_dispatch,
    )
    await handler._generate_response_task()

    assert handler.stream_fake.calls <= uh.MAX_TOOL_ITERATIONS


@pytest.mark.asyncio
async def test_barge_in_completes_the_booking_but_speaks_nothing(monkeypatch):
    """A guest interrupting must not leave a half-done booking."""
    committed = asyncio.Event()
    finished: list[str] = []

    async def slow_dispatch(name, args, call_id=""):
        await asyncio.sleep(0.15)
        finished.append(name)
        committed.set()
        return {"ok": True, "speakable": "Booked.", "data": {}}

    handler = make_handler(
        monkeypatch, [[ToolCallReady("c1", "book_taxi", {})]], slow_dispatch
    )
    task = asyncio.create_task(handler._generate_response_task())
    await asyncio.sleep(0.05)
    task.cancel()  # barge-in, mid-commit
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(committed.wait(), timeout=2)
    assert finished == ["book_taxi"]  # the shielded call still completed
    assert "booked" not in handler.tts_fake.spoken_text().lower()


@pytest.mark.asyncio
async def test_filler_is_spoken_before_the_tool_runs(monkeypatch):
    """The filler exists to cover the round-trip, so it must precede it.

    Regression test: the filler used to be emitted after the stream drained,
    which is after the model had already finished deciding -- the guest heard
    the silence first and the apology for it second.
    """
    order: list[str] = []

    async def slow_dispatch(name, args, call_id=""):
        order.append("dispatch")
        await asyncio.sleep(0.05)
        return {"ok": True, "speakable": "It's 2pm.", "data": {}}

    handler = make_handler(
        monkeypatch,
        [
            [ToolCallReady("c1", "get_hotel_time", {})],  # no lead-in
            [TextDelta("It's 2pm.")],
        ],
        slow_dispatch,
    )
    original_send = FakeTTS.send

    async def tracking_send(self, message):
        if isinstance(message, str):
            order.append("spoke")
        await original_send(self, message)

    monkeypatch.setattr(FakeTTS, "send", tracking_send)
    await handler._generate_response_task()

    assert "dispatch" in order
    assert order.index("spoke") < order.index("dispatch")


@pytest.mark.asyncio
async def test_no_filler_when_the_model_speaks_its_own_lead_in(monkeypatch):
    async def fake_dispatch(name, args, call_id=""):
        return {"ok": True, "speakable": "It's 2pm.", "data": {}}

    handler = make_handler(
        monkeypatch,
        [
            [TextDelta("Sure, "), ToolCallReady("c1", "get_hotel_time", {})],
            [TextDelta("It's 2pm.")],
        ],
        fake_dispatch,
    )
    await handler._generate_response_task()

    spoken = handler.tts_fake.spoken_text()
    # Match a distinctive fragment: the first word of a filler ("Sure") also
    # appears in the model's own lead-in.
    assert not any(" ".join(p.split()[1:3]) in spoken for p in uh.TOOL_FILLER_PHRASES)
