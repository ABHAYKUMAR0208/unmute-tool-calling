import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cache
from logging import getLogger
from typing import Any, AsyncIterator, Protocol, cast

from openai import APIError, AsyncOpenAI, OpenAI

from unmute.kyutai_constants import (
    LLM_PROVIDER_IGNORE,
    LLM_PROVIDER_SORT,
    LLM_REQUIRE_TOOL_PROVIDERS,
    LLM_SERVER,
)

from ..kyutai_constants import KYUTAI_LLM_API_KEY, KYUTAI_LLM_MODEL

logger = getLogger(__name__)

INTERRUPTION_CHAR = "—"  # em-dash
USER_SILENCE_MARKER = "..."


def preprocess_messages_for_llm(
    chat_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    output = []

    for message in chat_history:
        message = deepcopy(message)

        # Sometimes, an interruption happens before the LLM can say anything at all.
        # In that case, we're left with a message with only INTERRUPTION_CHAR.
        # Simplify by removing.
        if message["content"].replace(INTERRUPTION_CHAR, "") == "":
            continue

        # If the llm was interrupted we don't want to insert the INTERRUPTION_CHAR
        # into the context, otherwise the LLM might want to repeat it.
        message["content"] = message["content"].strip().removesuffix(INTERRUPTION_CHAR)

        if output and message["role"] == output[-1]["role"]:
            output[-1]["content"] += " " + message["content"]
        else:
            output.append(message)

    def role_at(index: int) -> str | None:
        if index >= len(output):
            return None
        return output[index]["role"]

    if role_at(0) == "system" and role_at(1) in [None, "assistant"]:
        # Some LLMs, like Gemma, get confused if the assistant message goes before user
        # messages, so add a dummy user message.
        output = [output[0]] + [{"role": "user", "content": "Hello."}] + output[1:]

    for message in output:
        if (
            message["role"] == "user"
            and message["content"].startswith(USER_SILENCE_MARKER)
            and message["content"] != USER_SILENCE_MARKER
        ):
            # This happens when the user is silent but then starts talking again after
            # the silence marker was inserted but before the LLM could respond.
            # There are special instructions in the system prompt about how to handle
            # the silence marker, so remove the marker from the message to not confuse
            # the LLM
            message["content"] = message["content"][len(USER_SILENCE_MARKER) :]

    return output


async def rechunk_to_words(iterator: AsyncIterator[str]) -> AsyncIterator[str]:
    """Rechunk the stream of text to whole words.

    Otherwise the TTS doesn't know where word boundaries are and will mispronounce
    split words.

    The spaces will be included with the next word, so "foo bar baz" will be split into
    "foo", " bar", " baz".
    Multiple space-like characters will be merged to a single space.
    """
    buffer = ""
    space_re = re.compile(r"\s+")
    prefix = ""
    async for delta in iterator:
        buffer = buffer + delta
        while True:
            match = space_re.search(buffer)
            if match is None:
                break
            chunk = buffer[: match.start()]
            buffer = buffer[match.end() :]
            if chunk != "":
                yield prefix + chunk
            prefix = " "

    if buffer != "":
        yield prefix + buffer


class LLMStream(Protocol):
    async def chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Get a chat completion from the LLM."""
        ...


def get_openai_client(
    server_url: str = LLM_SERVER, api_key: str | None = KYUTAI_LLM_API_KEY
) -> AsyncOpenAI:
    # AsyncOpenAI() will complain if the API key is not set, so set a dummy string if it's None.
    # This still makes sense when using vLLM because it doesn't care about the API key.
    return AsyncOpenAI(api_key=api_key or "EMPTY", base_url=server_url + "/v1")


@cache
def autoselect_model() -> str:
    if KYUTAI_LLM_MODEL is not None:
        return KYUTAI_LLM_MODEL
    openai_client = get_openai_client()
    # OpenAI() will complain if the API key is not set, so set a dummy string if it's None.
    # This still makes sense when using vLLM because it doesn't care about the API key.
    client_sync = OpenAI(
        api_key=openai_client.api_key or "EMPTY", base_url=openai_client.base_url
    )
    models = client_sync.models.list()
    if len(models.data) != 1:
        raise ValueError("There are multiple models available. Please specify one.")
    return models.data[0].id


class VLLMStream:
    def __init__(
        self,
        client: AsyncOpenAI,
        temperature: float = 1.0,
    ):
        """
        If `model` is None, it will look at the available models, and if there is only
        one model, it will use that one. Otherwise, it will raise.
        """
        self.client = client
        self.model = autoselect_model()
        self.temperature = temperature

    async def chat_completion(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),  # Cast and hope for the best
            stream=True,
            temperature=self.temperature,
        )

        async with stream:
            async for chunk in stream:
                if len(chunk.choices) == 0:
                    # OpenRouter sometimes does this, some kind of keep-alive chunk with no content. Just ignore it.
                    continue

                chunk_content = chunk.choices[0].delta.content

                if not chunk_content:
                    # This happens on the first message, see:
                    # https://platform.openai.com/docs/guides/streaming-responses#read-the-responses
                    # Also ignore `null` chunks, which is what llama.cpp does:
                    # https://github.com/ggml-org/llama.cpp/blob/6491d6e4f1caf0ad2221865b4249ae6938a6308c/tools/server/tests/unit/test_chat_completion.py#L338
                    continue

                yield chunk_content


# --- Tool calling (fork addition) ------------------------------------------
#
# `VLLMStream` above reads only `.delta.content` and is kept unchanged as the
# no-tools fallback. `ToolAwareLLMStream` below yields *typed events* so the
# caller can tell speech apart from actions.
#
# The subtle part is that tool-call arguments arrive as FRAGMENTS across many
# chunks (`{"room_`, `number":"3`, `05"}`), keyed by `index` because a model may
# emit several tool calls at once. Parsing early gives invalid JSON, so we
# accumulate per index and only `json.loads` at `finish_reason == "tool_calls"`.


@dataclass(frozen=True)
class TextDelta:
    """A piece of speakable text. Goes to the TTS exactly as before."""

    text: str


@dataclass(frozen=True)
class ToolCallReady:
    """A fully accumulated, parsed tool call. Never spoken."""

    id: str
    name: str
    args: dict[str, Any]


LLMEvent = TextDelta | ToolCallReady

# Said when the model can't be reached at all for a reply, so the caller hears
# something instead of silence and the session stays up.
LLM_FAILURE_SPEECH = "Sorry, I didn't quite catch that. Could you say it again?"


@dataclass
class _PartialToolCall:
    """Accumulator for one streamed tool call, keyed by its stream index."""

    id: str = ""
    name: str = ""
    arg_fragments: list[str] = field(default_factory=list)

    def joined_args(self) -> str:
        return "".join(self.arg_fragments)


# --- Tool calls that arrive as TEXT ------------------------------------------
# Llama models write their tool calls as JSON in the reply text, e.g.
#   {"type": "function", "name": "get_menu", "parameters": {}}
# Most OpenRouter providers turn that into a proper `tool_calls` field; some
# don't, and the JSON arrives as ordinary content. Spoken, it becomes gibberish
# ("getmenu parameters"), and the lookup never runs. So content is held back
# from the first "{" or Llama's "<|python_tag|>" on, and at the end of the
# stream it is either turned into a real tool call or dropped. Ordinary speech
# never contains "{", so nothing a guest should hear is delayed.
_TEXT_CALL_START = re.compile(r"\{|<\|python_tag\|>")


def _tool_key(name: str) -> str:
    """'get_menu', 'getmenu', 'Get-Menu' -> 'getmenu'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _as_text_call(obj: Any, known: dict[str, str]) -> tuple[str, dict[str, Any]] | None:
    """One decoded JSON value -> (tool name, args) if it is a call to a known tool."""
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):  # OpenAI-style {"function": {...}}
        obj = obj["function"]
    name = obj.get("name")
    if not isinstance(name, str) or _tool_key(name) not in known:
        return None
    args = obj.get("parameters", obj.get("arguments", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            return None
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return known[_tool_key(name)], args


def parse_text_tool_calls(
    text: str, tool_names: list[str]
) -> tuple[list[ToolCallReady], str]:
    """Split held-back text into (tool calls, leftover text worth speaking).

    JSON that isn't a call to one of `tool_names` is dropped, never spoken:
    reading JSON aloud is always wrong. Text between or around the JSON is kept.
    """
    known = {_tool_key(n): n for n in tool_names}
    body = text.replace("<|python_tag|>", "")
    decoder = json.JSONDecoder()
    calls: list[ToolCallReady] = []
    spoken: list[str] = []
    i = 0
    while i < len(body):
        start = body.find("{", i)
        if start < 0:
            spoken.append(body[i:])
            break
        spoken.append(body[i:start])
        try:
            obj, end = decoder.raw_decode(body, start)
        except json.JSONDecodeError:
            # Unbalanced or broken JSON: nothing after it is safe to speak.
            logger.warning("Dropping unparseable text after '{': %r", body[start:][:200])
            break
        found = _as_text_call(obj, known)
        if found:
            name, args = found
            calls.append(ToolCallReady(id=f"call_text_{name}_{len(calls)}", name=name, args=args))
        else:
            logger.warning("Dropping JSON the model wrote as speech: %r", body[start:end][:200])
        i = end
        while i < len(body) and body[i] in " \t\r\n;,":
            i += 1  # separators between several calls
    leftover = "".join(spoken)
    return calls, (leftover if leftover.strip() else "")


class ToolAwareLLMStream:
    """Streams an OpenAI-compatible completion as typed events.

    Yields `TextDelta` for content and `ToolCallReady` once a tool call's
    arguments are complete and parsed. If `tools` is empty this behaves like
    `VLLMStream`, just wrapped in `TextDelta`.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        temperature: float = 1.0,
        tools: list[dict[str, Any]] | None = None,
    ):
        self.client = client
        self.model = autoselect_model()
        self.temperature = temperature
        self.tools = tools or []

    async def chat_completion_events(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[LLMEvent]:
        """One reply, surviving a failed model request.

        A provider error used to escape from here and take the whole session
        down, e.g. Groq rejecting a garbled tool call mid-stream ("tool call
        validation failed"), which dropped a live phone call. Now: if nothing
        has been said yet, try again; then once more without tools (plain
        speech can't fail tool validation); then an honest line so the caller
        isn't left in silence. A failure after the reply has started just ends
        that reply where it is.
        """
        attempts = [self.tools, self.tools, []] if self.tools else [[], []]
        for number, tools in enumerate(attempts, start=1):
            produced = False
            try:
                async for event in self._events_once(messages, tools):
                    produced = True
                    yield event
                return
            except APIError as exc:
                if produced:
                    logger.warning("LLM stream failed part-way, ending this reply: %s", exc)
                    return
                logger.warning(
                    "LLM request failed (attempt %d/%d, tools=%s): %s",
                    number,
                    len(attempts),
                    bool(tools),
                    exc,
                )
        yield TextDelta(text=LLM_FAILURE_SPEECH)

    async def _events_once(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMEvent]:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            if LLM_REQUIRE_TOOL_PROVIDERS and "openrouter.ai" in str(
                self.client.base_url
            ):
                provider: dict[str, Any] = {"require_parameters": True}
                if LLM_PROVIDER_SORT:
                    provider["sort"] = LLM_PROVIDER_SORT
                if LLM_PROVIDER_IGNORE:
                    provider["ignore"] = LLM_PROVIDER_IGNORE
                kwargs["extra_body"] = {"provider": provider}

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            stream=True,
            temperature=self.temperature,
            **kwargs,
        )

        partials: dict[int, _PartialToolCall] = {}
        saw_tool_calls = False
        tool_names = [
            t.get("function", {}).get("name", "")
            for t in tools
            if t.get("function", {}).get("name")
        ]
        # Content from the first "{" on, held back until the stream ends (see
        # parse_text_tool_calls). None while nothing is held.
        held: list[str] | None = None

        async with stream:
            async for chunk in stream:
                if len(chunk.choices) == 0:
                    # OpenRouter keep-alive chunk, as in VLLMStream. Ignore.
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                content = delta.content
                if content and held is not None:
                    held.append(content)
                elif content:
                    match = _TEXT_CALL_START.search(content) if tool_names else None
                    if match:
                        if match.start():
                            yield TextDelta(text=content[: match.start()])
                        held = [content[match.start() :]]
                    else:
                        yield TextDelta(text=content)

                for tc in delta.tool_calls or []:
                    saw_tool_calls = True
                    partial = partials.setdefault(tc.index, _PartialToolCall())
                    if tc.id:
                        partial.id = tc.id
                    if tc.function is not None:
                        if tc.function.name:
                            partial.name = tc.function.name
                        if tc.function.arguments:
                            # Fragment — accumulate, do NOT parse yet.
                            partial.arg_fragments.append(tc.function.arguments)

                if choice.finish_reason is not None:
                    # Terminal chunk: arguments are complete, safe to parse.
                    if choice.finish_reason == "tool_calls" or saw_tool_calls:
                        for event in _finalize_tool_calls(partials):
                            yield event
                        partials.clear()

        # Some providers end the stream without a `tool_calls` finish_reason.
        # Anything still buffered is complete by definition, so flush it.
        for event in _finalize_tool_calls(partials):
            yield event

        if held is not None:
            calls, leftover = parse_text_tool_calls("".join(held), tool_names)
            if calls:
                logger.warning(
                    "Model wrote %d tool call(s) as text; running them: %s",
                    len(calls),
                    [c.name for c in calls],
                )
            if leftover:
                yield TextDelta(text=leftover)
            for call in calls:
                yield call


def _finalize_tool_calls(
    partials: dict[int, _PartialToolCall],
) -> list[ToolCallReady]:
    """Parse accumulated fragments into complete tool calls, skipping bad ones."""
    events: list[ToolCallReady] = []

    for index in sorted(partials):
        partial = partials[index]
        if not partial.name:
            continue

        raw_args = partial.joined_args().strip() or "{}"
        try:
            args = json.loads(raw_args)
            if args is None:
                # Llama sends `null` for a tool with no required inputs
                # (get_menu to hear the sections). That means "no arguments".
                args = {}
        except json.JSONDecodeError:
            # A truncated/malformed call is dropped rather than guessed at.
            # Acting on half-parsed arguments could book the wrong thing.
            logger.warning(
                "Dropping tool call %r: arguments are not valid JSON: %r",
                partial.name,
                raw_args,
            )
            continue

        if not isinstance(args, dict):
            logger.warning(
                "Dropping tool call %r: arguments are not an object: %r",
                partial.name,
                raw_args,
            )
            continue

        events.append(
            ToolCallReady(
                id=partial.id or f"call_{partial.name}_{index}",
                name=partial.name,
                args=args,
            )
        )

    return events
