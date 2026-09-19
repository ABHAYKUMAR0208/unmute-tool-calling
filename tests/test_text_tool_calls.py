"""Tool calls a provider returns as TEXT must run, never be spoken."""

from types import SimpleNamespace

import pytest

from unmute.llm import llm_utils
from unmute.llm.llm_utils import TextDelta, ToolCallReady, ToolAwareLLMStream, parse_text_tool_calls

TOOLS = ["get_menu", "get_service_price", "submit_guest_requests"]


def test_llama_style_call_is_recovered():
    calls, spoken = parse_text_tool_calls(
        '{"type": "function", "name": "get_menu", "parameters": {"query": "biryani"}}', TOOLS
    )
    assert [(c.name, c.args) for c in calls] == [("get_menu", {"query": "biryani"})]
    assert spoken == ""


def test_mangled_name_and_arguments_string_are_recovered():
    calls, _ = parse_text_tool_calls(
        '{"name": "getmenu", "arguments": "{\\"query\\": \\"tea\\"}"}', TOOLS
    )
    assert [(c.name, c.args) for c in calls] == [("get_menu", {"query": "tea"})]


def test_several_calls_and_python_tag():
    calls, _ = parse_text_tool_calls(
        '<|python_tag|>{"name": "get_menu", "parameters": {}}; '
        '{"function": {"name": "get_service_price", "arguments": {"service": "spa"}}}',
        TOOLS,
    )
    assert [c.name for c in calls] == ["get_menu", "get_service_price"]


def test_unknown_or_broken_json_is_dropped_not_spoken():
    calls, spoken = parse_text_tool_calls('{"name": "book_rocket", "parameters": {}}', TOOLS)
    assert calls == [] and spoken == ""
    calls, spoken = parse_text_tool_calls('{"name": "get_menu", "parameters": {', TOOLS)
    assert calls == [] and spoken == ""


def test_text_around_the_json_is_kept():
    calls, spoken = parse_text_tool_calls(
        '{"name": "get_menu", "parameters": {}} Let me check.', TOOLS
    )
    assert [c.name for c in calls] == ["get_menu"] and spoken.strip() == "Let me check."


def _chunk(content=None, finish=None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


class _FakeClient:
    def __init__(self, chunks, base_url="https://openrouter.ai/api/v1/"):
        self.base_url = base_url
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._chunks = chunks

    async def _create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream(self._chunks)


def _schemas(names):
    return [{"type": "function", "function": {"name": n, "parameters": {}}} for n in names]


@pytest.mark.asyncio
async def test_stream_speaks_the_lead_in_and_runs_the_text_call(monkeypatch):
    # The exact shape from a live call: speech, then the call written as text.
    monkeypatch.setattr(llm_utils, "autoselect_model", lambda: "m")
    client = _FakeClient([
        _chunk("We have a variety of dishes. Let me look. "),
        _chunk('{"type": "function", '),
        _chunk('"name": "get_menu", "parameters": {}}'),
        _chunk(None, "stop"),
    ])
    stream = ToolAwareLLMStream(client, tools=_schemas(TOOLS))
    events = [e async for e in stream.chat_completion_events([])]

    spoken = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert spoken == "We have a variety of dishes. Let me look. "
    assert "{" not in spoken and "get_menu" not in spoken
    assert [e.name for e in events if isinstance(e, ToolCallReady)] == ["get_menu"]
    # OpenRouter is asked to use only providers that support tool calls.
    assert client.kwargs["extra_body"]["provider"]["require_parameters"] is True
    assert client.kwargs["extra_body"]["provider"]["sort"] == "latency"


@pytest.mark.asyncio
async def test_plain_speech_is_untouched_and_no_routing_without_tools(monkeypatch):
    monkeypatch.setattr(llm_utils, "autoselect_model", lambda: "m")
    client = _FakeClient([_chunk("Hello, "), _chunk("welcome!"), _chunk(None, "stop")])
    stream = ToolAwareLLMStream(client, tools=[])
    events = [e async for e in stream.chat_completion_events([])]
    assert [e.text for e in events] == ["Hello, ", "welcome!"]
    assert "extra_body" not in client.kwargs and "tools" not in client.kwargs


def test_null_arguments_mean_no_arguments():
    # Seen live: get_menu with arguments "null" was dropped and Kelly went silent.
    partials = {0: llm_utils._PartialToolCall(id="c1", name="get_menu", arg_fragments=["null"])}
    assert [(c.name, c.args) for c in llm_utils._finalize_tool_calls(partials)] == [("get_menu", {})]
    calls, _ = parse_text_tool_calls('{"name": "get_menu", "parameters": null}', TOOLS)
    assert [(c.name, c.args) for c in calls] == [("get_menu", {})]


def _groq_error():
    import httpx
    from openai import APIError

    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return APIError(
        "Upstream error from Groq: tool call validation failed: attempted to call tool "
        "'get_menu_item_details {\"item\": \"burger\"}' which was not in request.tools",
        request,
        body=None,
    )


class _FlakyClient(_FakeClient):
    """Fails the first `failures` requests the way Groq did live, then streams normally."""

    def __init__(self, chunks, failures):
        super().__init__(chunks)
        self.failures = failures
        self.calls = []

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise _groq_error()
        return _FakeStream(self._chunks)


@pytest.mark.asyncio
async def test_a_provider_error_is_retried_instead_of_ending_the_call(monkeypatch):
    monkeypatch.setattr(llm_utils, "autoselect_model", lambda: "m")
    client = _FlakyClient([_chunk("Sure, one burger. "), _chunk(None, "stop")], failures=1)
    stream = ToolAwareLLMStream(client, tools=_schemas(TOOLS))
    events = [e async for e in stream.chat_completion_events([])]
    assert "".join(e.text for e in events) == "Sure, one burger. "
    assert len(client.calls) == 2 and "tools" in client.calls[1]


@pytest.mark.asyncio
async def test_repeated_errors_fall_back_to_no_tools_then_an_honest_line(monkeypatch):
    monkeypatch.setattr(llm_utils, "autoselect_model", lambda: "m")
    client = _FlakyClient([_chunk("Sure. "), _chunk(None, "stop")], failures=2)
    stream = ToolAwareLLMStream(client, tools=_schemas(TOOLS))
    events = [e async for e in stream.chat_completion_events([])]
    assert "".join(e.text for e in events) == "Sure. "
    assert "tools" not in client.calls[2]  # third try: plain speech can't fail tool validation

    client = _FlakyClient([], failures=99)
    stream = ToolAwareLLMStream(client, tools=_schemas(TOOLS))
    events = [e async for e in stream.chat_completion_events([])]
    assert [e.text for e in events] == [llm_utils.LLM_FAILURE_SPEECH]


@pytest.mark.asyncio
async def test_error_after_speaking_ends_the_reply_quietly(monkeypatch):
    monkeypatch.setattr(llm_utils, "autoselect_model", lambda: "m")

    class _BreaksMidway(_FakeStream):
        async def _gen(self):
            yield _chunk("Sure, one burger ")
            raise _groq_error()

    client = _FakeClient([])
    client.chat.completions.create = lambda **kw: _async(_BreaksMidway([]))
    stream = ToolAwareLLMStream(client, tools=_schemas(TOOLS))
    events = [e async for e in stream.chat_completion_events([])]
    assert "".join(e.text for e in events) == "Sure, one burger "  # no crash, no repeat


async def _async(value):
    return value
