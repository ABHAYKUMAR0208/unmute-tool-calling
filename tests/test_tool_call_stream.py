"""Tests for tool-call fragment accumulation in ToolAwareLLMStream.

These use fake chunks rather than a live endpoint, so they run in CI without an
API key. They encode the behaviour that matters: arguments arrive in pieces and
must not be parsed until the stream says they're complete.
"""

from types import SimpleNamespace

import pytest

from unmute.llm.llm_utils import TextDelta, ToolAwareLLMStream, ToolCallReady


def _chunk(content=None, tool_calls=None, finish_reason=None, empty=False):
    if empty:  # OpenRouter keep-alive
        return SimpleNamespace(choices=[])
    calls = None
    if tool_calls is not None:
        calls = [
            SimpleNamespace(
                index=tc.get("index", 0),
                id=tc.get("id"),
                function=SimpleNamespace(
                    name=tc.get("name"), arguments=tc.get("arguments")
                ),
            )
            for tc in tool_calls
        ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=calls),
                finish_reason=finish_reason,
            )
        ]
    )


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _stream_with(chunks) -> ToolAwareLLMStream:
    async def create(**kwargs):
        create.kwargs = kwargs
        return _FakeStream(chunks)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    stream = ToolAwareLLMStream.__new__(ToolAwareLLMStream)
    stream.client = client
    stream.model = "test-model"
    stream.temperature = 1.0
    stream.tools = [{"type": "function", "function": {"name": "book_taxi"}}]
    stream._create = create
    return stream


async def _collect(stream, messages=None):
    return [e async for e in stream.chat_completion_events(messages or [])]


@pytest.mark.asyncio
async def test_arguments_split_across_chunks_are_accumulated():
    """The central case: JSON arrives in fragments that are invalid alone."""
    stream = _stream_with(
        [
            _chunk(tool_calls=[{"id": "call_1", "name": "book_taxi", "arguments": ""}]),
            _chunk(tool_calls=[{"arguments": '{"room_'}]),
            _chunk(tool_calls=[{"arguments": 'number":"3'}]),
            _chunk(tool_calls=[{"arguments": '05","destination":"air'}]),
            _chunk(tool_calls=[{"arguments": 'port"}'}]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    events = await _collect(stream)
    assert events == [
        ToolCallReady(
            id="call_1",
            name="book_taxi",
            args={"room_number": "305", "destination": "airport"},
        )
    ]


@pytest.mark.asyncio
async def test_text_and_tool_call_in_one_turn():
    """Filler text before a tool call is spoken; the call is not."""
    stream = _stream_with(
        [
            _chunk(content="Sure, "),
            _chunk(content="booking that now —"),
            _chunk(
                tool_calls=[{"id": "c1", "name": "book_taxi", "arguments": '{"a":1}'}]
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    events = await _collect(stream)
    assert events[0] == TextDelta(text="Sure, ")
    assert events[1] == TextDelta(text="booking that now —")
    assert events[2] == ToolCallReady(id="c1", name="book_taxi", args={"a": 1})


@pytest.mark.asyncio
async def test_parallel_tool_calls_kept_separate_by_index():
    stream = _stream_with(
        [
            _chunk(
                tool_calls=[
                    {"index": 0, "id": "c0", "name": "book_taxi", "arguments": '{"x'},
                    {
                        "index": 1,
                        "id": "c1",
                        "name": "create_food_order",
                        "arguments": '{"y',
                    },
                ]
            ),
            _chunk(
                tool_calls=[
                    {"index": 0, "arguments": '":1}'},
                    {"index": 1, "arguments": '":2}'},
                ]
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    events = await _collect(stream)
    assert events == [
        ToolCallReady(id="c0", name="book_taxi", args={"x": 1}),
        ToolCallReady(id="c1", name="create_food_order", args={"y": 2}),
    ]


@pytest.mark.asyncio
async def test_keepalive_chunks_are_ignored():
    stream = _stream_with(
        [
            _chunk(empty=True),
            _chunk(content="Hello"),
            _chunk(empty=True),
            _chunk(finish_reason="stop"),
        ]
    )
    assert await _collect(stream) == [TextDelta(text="Hello")]


@pytest.mark.asyncio
async def test_text_only_turn_yields_no_tool_calls():
    stream = _stream_with(
        [
            _chunk(content="Good "),
            _chunk(content="evening."),
            _chunk(finish_reason="stop"),
        ]
    )
    events = await _collect(stream)
    assert all(isinstance(e, TextDelta) for e in events)


@pytest.mark.asyncio
async def test_malformed_arguments_are_dropped_not_guessed():
    """A truncated call must not become a booking with invented fields."""
    stream = _stream_with(
        [
            _chunk(
                tool_calls=[{"id": "c1", "name": "book_taxi", "arguments": '{"room_nu'}]
            ),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    assert await _collect(stream) == []


@pytest.mark.asyncio
async def test_missing_finish_reason_still_flushes():
    """Some providers end without finish_reason='tool_calls'."""
    stream = _stream_with(
        [
            _chunk(
                tool_calls=[{"id": "c1", "name": "book_taxi", "arguments": '{"a":1}'}]
            ),
        ]
    )
    assert await _collect(stream) == [
        ToolCallReady(id="c1", name="book_taxi", args={"a": 1})
    ]


@pytest.mark.asyncio
async def test_empty_arguments_become_empty_dict():
    stream = _stream_with(
        [
            _chunk(tool_calls=[{"id": "c1", "name": "get_menu", "arguments": ""}]),
            _chunk(finish_reason="tool_calls"),
        ]
    )
    assert await _collect(stream) == [ToolCallReady(id="c1", name="get_menu", args={})]


@pytest.mark.asyncio
async def test_tools_and_tool_choice_are_sent():
    stream = _stream_with([_chunk(content="hi"), _chunk(finish_reason="stop")])
    await _collect(stream)
    assert stream._create.kwargs["tool_choice"] == "auto"
    assert stream._create.kwargs["tools"] == stream.tools
    assert stream._create.kwargs["stream"] is True
