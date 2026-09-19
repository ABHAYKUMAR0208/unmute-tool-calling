"""See OpenAI's docs: https://platform.openai.com/docs/api-reference/realtime

https://platform.openai.com/docs/api-reference/realtime-client-events
https://platform.openai.com/docs/api-reference/realtime-server-events
"""

import random
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, model_validator

from unmute.llm.system_prompt import Instructions

T = TypeVar("T", bound=str)


def random_id(prefix: str) -> str:
    """e.g. event_BJhGUIswO2u7vA2Cxw3Jy"""
    n_characters = 21
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return prefix + "_" + "".join(random.choices(alphabet, k=n_characters))


class BaseEvent(BaseModel, Generic[T]):
    type: T = None  # type: ignore - will be set by validator below
    event_id: str = Field(default_factory=lambda: random_id("event"))

    @model_validator(mode="after")
    def set_type_from_generic(self) -> "BaseEvent":
        if type(self) == BaseEvent:  # noqa - we want to use type() == and not isinstance
            raise ValueError("Cannot instantiate BaseEvent directly")

        # e.g. Literal["session.update"]
        type_argument = self.__class__.model_fields["type"].annotation

        if get_origin(type_argument) is not Literal:
            # I don't know how to enforce this in the type system directly
            raise ValueError("Type argument is not a Literal")

        self.type = get_args(type_argument)[0]  # e.g. "session.update"

        return self


class ErrorDetails(BaseModel):
    type: str
    code: str | None = None
    message: str
    param: str | None = None
    # ours, not part of the OpenAI API:
    details: object | None = None


class Error(BaseEvent[Literal["error"]]):
    error: ErrorDetails


class SessionConfig(BaseModel):
    # The "Instructions" object is an Unmute extension
    instructions: Instructions | None = None
    voice: str | None = None
    allow_recording: bool
    # OpenAI-format tool schemas the LLM may call during the session. `None`
    # keeps the server-side default set; `[]` explicitly disables tool calling.
    tools: list[dict[str, Any]] | None = None
    # Names (from `tools`) that the CLIENT executes rather than this server. When
    # the LLM calls one, the server sends unmute.tool_call.request and waits for
    # the client's unmute.tool_call.result. Unmute extension; lets the client use
    # what only it knows (who is calling, which room) without a public endpoint.
    client_tools: list[str] | None = None


class SessionUpdate(BaseEvent[Literal["session.update"]]):
    session: SessionConfig


class SessionUpdated(BaseEvent[Literal["session.updated"]]):
    session: SessionConfig


class InputAudioBufferAppend(BaseEvent[Literal["input_audio_buffer.append"]]):
    audio: str  # Base64-encoded Opus data


class UnmuteInputAudioBufferAppendAnonymized(
    BaseEvent[Literal["unmute.input_audio_buffer.append_anonymized"]]
):
    """
    For recording, an anonymous version of InputAudioBufferAppend that only says
    how many samples were appended, not the actual audio data.
    """

    number_of_samples: int


class InputAudioBufferSpeechStarted(
    BaseEvent[Literal["input_audio_buffer.speech_started"]]
):
    """Speech started according to the STT.

    Note this is not symmetrical with `InputAudioBufferSpeechStopped` because it's based
    on the STT and not the VAD signal. This is because sometimes the VAD will think
    there is speech but then nothing will end up getting transcribed. If we were using
    the VAD for both events we might get a start event without a stop event.
    For VAD interruptions, see `UnmuteInterruptedByVAD`.
    """


class InputAudioBufferSpeechStopped(
    BaseEvent[Literal["input_audio_buffer.speech_stopped"]]
):
    """A pause was detected by the VAD."""


class Response(BaseModel):
    object: Literal["realtime.response"] = "realtime.response"
    # We currently only use in_progress
    status: Literal["in_progress", "completed", "cancelled", "failed", "incomplete"]
    voice: str
    chat_history: list[dict[str, Any]] = Field(default_factory=list)


class ResponseCreated(BaseEvent[Literal["response.created"]]):
    response: Response


class ResponseTextDelta(BaseEvent[Literal["response.text.delta"]]):
    delta: str


class ResponseTextDone(BaseEvent[Literal["response.text.done"]]):
    text: str


class ResponseAudioDelta(BaseEvent[Literal["response.audio.delta"]]):
    delta: str  # Base64-encoded Opus audio data


class ResponseAudioDone(BaseEvent[Literal["response.audio.done"]]):
    pass


class TranscriptLogprob(BaseModel):
    bytes: bytes
    logprob: float
    token: str


class ConversationItemInputAudioTranscriptionDelta(
    BaseEvent[Literal["conversation.item.input_audio_transcription.delta"]]
):
    delta: str
    start_time: float  # Unmute extension


class UnmuteAdditionalOutputs(BaseEvent[Literal["unmute.additional_outputs"]]):
    args: Any


class UnmuteResponseTextDeltaReady(
    BaseEvent[Literal["unmute.response.text.delta.ready"]]
):
    delta: str


class UnmuteResponseAudioDeltaReady(
    BaseEvent[Literal["unmute.response.audio.delta.ready"]]
):
    number_of_samples: int


class UnmuteInterruptedByVAD(BaseEvent[Literal["unmute.interrupted_by_vad"]]):
    """The VAD interrupted the response generation."""


class ConversationItemInputAudioTranscriptionCompleted(
    BaseEvent[Literal["conversation.item.input_audio_transcription.completed"]]
):
    """The final words of the caller's turn, replacing the live deltas sent so far
    (see unmute/stt/refine.py)."""

    transcript: str


class UnmuteToolCallRequest(BaseEvent[Literal["unmute.tool_call.request"]]):
    """The LLM called one of the session's client_tools; the client must run it."""

    call_id: str
    name: str
    arguments: dict[str, Any]


class UnmuteResponseCreate(BaseEvent[Literal["unmute.response.create"]]):
    """Reply now, without waiting for the user to speak. Unmute extension.

    Used after a mid-call session.update (e.g. a call passed from reception to
    the restaurant team) so the new persona greets the caller. `note`, if
    given, is added to the conversation as the user's side first, to tell
    the model why it is speaking.
    """

    note: str | None = None


class UnmuteToolCallResult(BaseEvent[Literal["unmute.tool_call.result"]]):
    """The client's answer to an UnmuteToolCallRequest, as a tool envelope
    ({"ok", "speakable", "data", optional "error_code"})."""

    call_id: str
    result: dict[str, Any]


# Server events (from OpenAI to client)
ServerEvent = Union[
    Error,
    SessionUpdated,
    ResponseTextDelta,
    ResponseTextDone,
    ResponseAudioDelta,
    ResponseAudioDone,
    ResponseCreated,
    ConversationItemInputAudioTranscriptionDelta,
    InputAudioBufferSpeechStarted,
    InputAudioBufferSpeechStopped,
    UnmuteAdditionalOutputs,
    UnmuteResponseTextDeltaReady,
    UnmuteResponseAudioDeltaReady,
    UnmuteInterruptedByVAD,
    UnmuteToolCallRequest,
    ConversationItemInputAudioTranscriptionCompleted,
]

# Client events (from client to OpenAI)
ClientEvent = Union[
    SessionUpdate,
    InputAudioBufferAppend,
    UnmuteToolCallResult,
    UnmuteResponseCreate,
    # Used internally for recording, we're not expecting the user to send this
    UnmuteInputAudioBufferAppendAnonymized,
]

Event = ClientEvent | ServerEvent
