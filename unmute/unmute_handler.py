import asyncio
import json
import math
import os
import random
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Any, AsyncIterator, Literal, cast

import httpx
import numpy as np
import websockets
from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    CloseStream,
    audio_to_float32,
    wait_for_item,
)
from pydantic import BaseModel

import unmute.openai_realtime_api_events as ora
from unmute import metrics as mt
from unmute.audio_input_override import AudioInputOverride
from unmute.exceptions import make_ora_error
from unmute.kyutai_constants import (
    CLIENT_TOOL_TIMEOUT_S,
    FRAME_TIME_SEC,
    MAX_TOOL_ITERATIONS,
    RECORDINGS_DIR,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    TOOL_FILLER_ENABLED,
    TOOL_TIMEOUT_S,
)
from unmute.llm.chatbot import Chatbot
from unmute.llm.llm_utils import (
    INTERRUPTION_CHAR,
    USER_SILENCE_MARKER,
    TextDelta,
    ToolAwareLLMStream,
    ToolCallReady,
    get_openai_client,
    rechunk_to_words,
)
from unmute.llm.tools import (
    CACHEABLE_TOOLS,
    DETAILS_TOOL,
    MAX_TOOL_NOTES,
    SIDE_EFFECTING_TOOLS,
    asked_about_details,
    cache_key,
    default_tool_schemas,
    details_not_requested,
    dispatch,
    from_cache,
    tool_note,
    with_tool_notes,
)
from unmute.quest_manager import Quest, QuestManager
from unmute.recorder import Recorder
from unmute.service_discovery import find_instance
from unmute.stt.refine import (
    TurnAudio,
    has_caller_words,
    refine_enabled,
    replace_last_user_text,
    transcribe,
)
from unmute.stt.speech_to_text import SpeechToText, STTMarkerMessage
from unmute.timer import Stopwatch
from unmute.tts.text_to_speech import (
    TextToSpeech,
    TTSAudioMessage,
    TTSClientEosMessage,
    TTSTextMessage,
)

# TTS_DEBUGGING_TEXT: str | None = "What's 'Hello world'?"
# TTS_DEBUGGING_TEXT: str | None = "What's the difference between a bagel and a donut?"
TTS_DEBUGGING_TEXT = None

# AUDIO_INPUT_OVERRIDE: Path | None = Path.home() / "audio/dog-or-cat-3.mp3"
AUDIO_INPUT_OVERRIDE: Path | None = None
DEBUG_PLOT_HISTORY_SEC = 10.0

# Seconds of caller silence before Kelly checks in ("Are you still there?").
# Was 7: on phone calls people pause longer than that while deciding, and the
# check-ins interrupted them. Override with USER_SILENCE_TIMEOUT_S.
USER_SILENCE_TIMEOUT = float(os.environ.get("USER_SILENCE_TIMEOUT_S", "10"))
FIRST_MESSAGE_TEMPERATURE = 0.7
FURTHER_MESSAGES_TEMPERATURE = 0.3
# For this much time, the VAD does not interrupt the bot. This is needed because at
# least on Mac, the echo cancellation takes a while to kick in, at the start, so the ASR
# sometimes hears a bit of the TTS audio and interrupts the bot. Only happens on the
# first message.
# A word from the ASR can still interrupt the bot.
UNINTERRUPTIBLE_BY_VAD_TIME_SEC = 3

logger = getLogger(__name__)

HandlerOutput = (
    tuple[int, np.ndarray] | AdditionalOutputs | ora.ServerEvent | CloseStream
)
import re

_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)
_STAGE_DIRECTION_PATTERN = re.compile(r"[\(\*][^)\*]{1,30}[\)\*]")


def strip_unpronounceable(text: str) -> str:
    text = _EMOJI_PATTERN.sub("", text)
    text = _STAGE_DIRECTION_PATTERN.sub("", text)
    text = text.replace("*", "").replace("#", "").replace("_", "")
    return text


# Spoken only when the model calls a tool without any lead-in of its own, to
# cover the backend round-trip. Deliberately vague: they promise nothing, so
# they stay honest even if the tool then fails. Off unless TOOL_FILLER_ENABLED:
# the TTS can't voice a short phrase until more text follows it, so in practice
# it arrived after the tool, glued to the answer (see kyutai_constants).
TOOL_FILLER_PHRASES = [
    "One moment, let me take care of that.",
    "Sure, just a second.",
    "Let me sort that out for you.",
    "Right away, one moment.",
]


class GradioUpdate(BaseModel):
    chat_history: list[dict[str, str]]
    debug_dict: dict[str, Any]
    debug_plot_data: list[dict]


class UnmuteHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            input_sample_rate=SAMPLE_RATE,
            # IMPORTANT! If set to a higher value, will lead to choppy audio. 🤷‍♂️
            output_frame_size=480,
            output_sample_rate=SAMPLE_RATE,
        )
        self.n_samples_received = 0  # Used for measuring time
        self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()
        self.recorder = Recorder(RECORDINGS_DIR) if RECORDINGS_DIR else None

        self.quest_manager = QuestManager()

        self.stt_last_message_time: float = 0
        self.stt_end_of_flush_time: float | None = None
        self.stt_flush_timer = Stopwatch()

        self.tts_voice: str | None = None  # Stored separately because TTS is restarted
        self.tts_output_stopwatch = Stopwatch()

        self.chatbot = Chatbot()
        self.openai_client = get_openai_client()

        # Tools offered to the LLM this session. Replaced per-session via
        # update_session (Step 6); defaults to the full hotel tool set so the
        # loop is testable before the client sends any.
        self.tool_schemas: list[dict[str, Any]] = default_tool_schemas()
        # One line per finished tool call this session, fed back to the LLM on
        # later turns so it doesn't repeat lookups (see tools.with_tool_notes).
        self.tool_notes: list[str] = []
        # Tools the client runs (see SessionConfig.client_tools), and the calls
        # currently waiting for the client's unmute.tool_call.result.
        self.client_tools: frozenset[str] = frozenset()
        self._client_tool_waiters: dict[str, asyncio.Future[dict[str, Any]]] = {}
        # Successful read-only lookups this session, by normalised call
        # (see tools.cache_key), so a repeat is answered from the first result.
        self.tool_cache: dict[str, dict[str, Any]] = {}
        # The caller's recent audio, and where their current turn began in it,
        # so the turn can be re-transcribed when it ends (see stt/refine.py).
        self.turn_audio = TurnAudio()
        self._turn_start_sample: int | None = None
        self._refine_client: httpx.AsyncClient | None = None

        self.turn_transition_lock = asyncio.Lock()

        self.debug_dict: dict[str, Any] = {
            "timing": {},
            "connection": {},
            "chatbot": {},
        }
        self.debug_plot_data: list[dict] = []
        self.last_additional_output_update = self.audio_received_sec()

        if AUDIO_INPUT_OVERRIDE is not None:
            self.audio_input_override = AudioInputOverride(AUDIO_INPUT_OVERRIDE)
        else:
            self.audio_input_override = None

    async def cleanup(self):
        if self.recorder is not None:
            await self.recorder.shutdown()
        if self._refine_client is not None:
            await self._refine_client.aclose()

    @property
    def stt(self) -> SpeechToText | None:
        try:
            quest = self.quest_manager.quests["stt"]
        except KeyError:
            return None
        return cast(Quest[SpeechToText], quest).get_nowait()

    @property
    def tts(self) -> TextToSpeech | None:
        try:
            quest = self.quest_manager.quests["tts"]
        except KeyError:
            return None
        return cast(Quest[TextToSpeech], quest).get_nowait()

    def get_gradio_update(self):
        self.debug_dict["conversation_state"] = self.chatbot.conversation_state()
        self.debug_dict["connection"]["stt"] = self.stt.state() if self.stt else "none"
        self.debug_dict["connection"]["tts"] = self.tts.state() if self.tts else "none"
        self.debug_dict["tts_voice"] = self.tts.voice if self.tts else "none"
        self.debug_dict["stt_pause_prediction"] = (
            self.stt.pause_prediction.value if self.stt else -1
        )

        # This gets verbose
        # cutoff_time = self.audio_received_sec() - DEBUG_PLOT_HISTORY_SEC
        # self.debug_plot_data = [x for x in self.debug_plot_data if x["t"] > cutoff_time]

        return AdditionalOutputs(
            GradioUpdate(
                chat_history=[
                    # Not trying to hide the system prompt, just making it less verbose
                    m
                    for m in self.chatbot.chat_history
                    if m["role"] != "system"
                ],
                debug_dict=self.debug_dict,
                debug_plot_data=[],
            )
        )

    async def add_chat_message_delta(
        self,
        delta: str,
        role: Literal["user", "assistant"],
        generating_message_i: int | None = None,  # Avoid race conditions
    ):
        is_new_message = await self.chatbot.add_chat_message_delta(
            delta, role, generating_message_i=generating_message_i
        )

        return is_new_message

    async def _generate_response(self):
        # Empty message to signal we've started responding.
        # Do it here in the lock to avoid race conditions
        await self.add_chat_message_delta("", "assistant")
        quest = Quest.from_run_step("llm", self._generate_response_task)
        await self.quest_manager.add(quest)

    async def _run_tool_calls(
        self, tool_calls: list[ToolCallReady]
    ) -> list[dict[str, Any]]:
        """Execute tool calls and return their contract envelopes, in order.

        Side-effecting tools are wrapped in `asyncio.shield` so that a barge-in
        (which cancels this task via `quest_manager.remove("llm")`) stops the
        speaking but never leaves a half-finished booking or a charge the guest
        was never told about. The shielded call is allowed to finish; the
        CancelledError is re-raised straight after, so nothing is spoken.

        Read-only tools carry no such risk and are cancelled freely.
        """
        results: list[dict[str, Any]] = []

        for call in tool_calls:
            logger.info("Tool call: %s(%r)", call.name, call.args)
            tool_stopwatch = Stopwatch()
            is_client_tool = call.name in self.client_tools
            key = cache_key(call.name, call.args)
            cached = self.tool_cache.get(key) if call.name in CACHEABLE_TOOLS else None
            if call.name == DETAILS_TOOL and not asked_about_details(
                self._last_user_text()
            ):
                logger.info("Not running %s: the guest didn't ask about it", call.name)
                coro = asyncio.sleep(0, result=details_not_requested(call.args))
                timeout_s = TOOL_TIMEOUT_S
            elif cached is not None:
                logger.info("Repeat of %s; answering from the earlier result", key)
                coro = asyncio.sleep(0, result=from_cache(cached))
                timeout_s = TOOL_TIMEOUT_S
            elif is_client_tool:
                coro = self._run_client_tool(call)
                timeout_s = CLIENT_TOOL_TIMEOUT_S
            else:
                coro = dispatch(call.name, call.args, call_id=call.id)
                timeout_s = TOOL_TIMEOUT_S

            try:
                # Client tools may save things (a guest's requests), so they get
                # the same barge-in protection as the side-effecting ones.
                if call.name in SIDE_EFFECTING_TOOLS or is_client_tool:
                    # Shield first, THEN bound it: wait_for would otherwise
                    # cancel the shielded task it is waiting on.
                    task = asyncio.ensure_future(coro)
                    try:
                        envelope = await asyncio.wait_for(
                            asyncio.shield(task), timeout=timeout_s
                        )
                    except asyncio.CancelledError:
                        # Barge-in mid-commit: let the backend finish so the
                        # guest isn't left in a half-done state, then bail.
                        logger.warning(
                            "Interrupted during %s; letting it complete.", call.name
                        )
                        await asyncio.wait([task], timeout=timeout_s)
                        raise
                else:
                    envelope = await asyncio.wait_for(coro, timeout=timeout_s)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                # dispatch() has its own timeout; this is the backstop.
                logger.warning("Tool %s exceeded %.1fs", call.name, timeout_s)
                envelope = {
                    "ok": False,
                    "speakable": (
                        "I'm sorry, that's taking longer than expected. "
                        "Let me have the front desk confirm this for you."
                    ),
                    "error_code": "TOOL_TIMEOUT",
                    "data": {},
                }
            except Exception:
                logger.exception("Tool %s raised", call.name)
                envelope = {
                    "ok": False,
                    "speakable": (
                        "I'm sorry, I couldn't complete that just now. "
                        "Shall I have the front desk follow up with you?"
                    ),
                    "error_code": "TOOL_ERROR",
                    "data": {},
                }

            logger.info(
                "Tool %s -> ok=%s in %.2fs",
                call.name,
                envelope.get("ok"),
                tool_stopwatch.time(),
            )
            self.debug_dict.setdefault("tool_calls", []).append(
                {
                    "name": call.name,
                    "ok": envelope.get("ok"),
                    "error_code": envelope.get("error_code"),
                    "duration_s": round(tool_stopwatch.time(), 3),
                }
            )
            results.append(envelope)
            if (
                call.name in CACHEABLE_TOOLS
                and cached is None
                and envelope.get("ok")
                and not (envelope.get("data") or {}).get("note")
            ):
                self.tool_cache[key] = envelope
            self.tool_notes.append(tool_note(call.name, call.args, envelope))
            del self.tool_notes[:-MAX_TOOL_NOTES]

        return results

    async def create_response(self, note: str | None = None) -> None:
        """Reply now without waiting for the user (unmute.response.create)."""
        async with self.turn_transition_lock:
            if self.chatbot.conversation_state() == "bot_speaking":
                logger.warning("response.create ignored: already replying")
                return
            if note and note.strip():
                self.chatbot.chat_history.append({"role": "user", "content": note.strip()})
            logger.info("Reply requested by the client (note: %r)", note)
            await self._generate_response()

    def _last_user_text(self) -> str:
        for message in reversed(self.chatbot.chat_history):
            if message.get("role") == "user" and str(message.get("content") or "").strip():
                return str(message["content"])
        return ""

    async def _run_client_tool(self, call: ToolCallReady) -> dict[str, Any]:
        """Hand a client tool to the client and wait for its envelope.

        The outer loop bounds the wait; if the client never answers (it hung
        up, or its CRM is down) the timeout backstop gives an honest reply.
        """
        call_id = call.id or f"call_{random.getrandbits(48):x}"
        future: asyncio.Future[dict[str, Any]] = (
            asyncio.get_running_loop().create_future()
        )
        self._client_tool_waiters[call_id] = future
        try:
            await self.output_queue.put(
                ora.UnmuteToolCallRequest(
                    call_id=call_id, name=call.name, arguments=call.args
                )
            )
            return await future
        finally:
            self._client_tool_waiters.pop(call_id, None)

    def resolve_client_tool(self, call_id: str, result: dict[str, Any]) -> None:
        """Called by the receive loop with the client's unmute.tool_call.result."""
        future = self._client_tool_waiters.get(call_id)
        if future is None or future.done():
            logger.warning("Tool result for unknown or finished call %r", call_id)
            return
        if not isinstance(result.get("ok"), bool) or not result.get("speakable"):
            # Same rule as dispatch(): never let the LLM improvise a confirmation.
            logger.error("Client tool %r returned a non-contract envelope", call_id)
            result = {
                "ok": False,
                "speakable": (
                    "I'm sorry, I couldn't complete that just now. "
                    "Shall I have the front desk follow up with you?"
                ),
                "error_code": "MALFORMED_CLIENT_RESULT",
                "data": {},
            }
        future.set_result(result)

    async def _use_better_transcript(self) -> None:
        """Replace the live words of the caller's turn with a better transcription.

        Runs as the reply starts, so the LLM, the chat history and the client's
        transcript all get the better words. Never blocks for longer than
        STT_REFINE_TIMEOUT_S; on any failure the live words stand.
        """
        start = self._turn_start_sample
        if start is None or not refine_enabled() or not has_caller_words(
            self.chatbot.chat_history
        ):
            return
        if self._refine_client is None:
            self._refine_client = httpx.AsyncClient()
        stopwatch = Stopwatch()
        text = await transcribe(self.turn_audio.slice(start), self._refine_client)
        if text is None:
            return
        old = replace_last_user_text(self.chatbot.chat_history, text)
        if old is None:
            return
        logger.info(
            "Better transcript in %.2fs: %r -> %r", stopwatch.time(), old.strip(), text
        )
        await self.output_queue.put(
            ora.ConversationItemInputAudioTranscriptionCompleted(transcript=text)
        )

    async def _generate_response_task(self):
        await self._use_better_transcript()
        generating_message_i = len(self.chatbot.chat_history)

        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice=self.tts_voice or "missing",
                    chat_history=self.chatbot.chat_history,
                )
            )
        )

        llm_stopwatch = Stopwatch()

        quest = await self.start_up_tts(generating_message_i)
        # if generating_message_i is 2, then we have a system prompt + an empty
        # assistant message signalling that we are generating a response.
        temperature = (
            FIRST_MESSAGE_TEMPERATURE
            if generating_message_i == 2
            else FURTHER_MESSAGES_TEMPERATURE
        )

        # A LOCAL copy: tool messages get appended here and are deliberately
        # never written back to chatbot.chat_history.
        messages = with_tool_notes(
            list(self.chatbot.preprocessed_messages()), self.tool_notes
        )

        self.tts_output_stopwatch = Stopwatch(autostart=False)
        tts = None

        response_words = []
        error_from_tts = False
        interrupted = False
        time_to_first_token = None
        num_words_sent = sum(
            # `or ""` because a tool-calling assistant message has content=None.
            len((message.get("content") or "").split())
            for message in messages
        )
        mt.VLLM_SENT_WORDS.inc(num_words_sent)
        mt.VLLM_REQUEST_LENGTH.observe(num_words_sent)
        mt.VLLM_ACTIVE_SESSIONS.inc()

        try:
            # --- Pause-act-resume loop (fork addition) ----------------------
            # Bounded so a model that keeps calling tools can't spiral.
            #
            # Tool messages live ONLY in this local `messages` list. They are
            # never written to chatbot.chat_history, whose state machine
            # understands system/user/assistant with string content only and
            # would raise on a `tool` role. Only spoken text reaches the
            # history, via the normal add_chat_message_delta path.
            # No tools on the opening turn: the greeting must be the first thing
            # the caller hears. With tools on offer the model sometimes looks
            # something up first, and the filler ("One moment...") then lands
            # in front of "Hello, welcome to...".
            is_opening_turn = not any(
                m.get("role") == "assistant" and str(m.get("content") or "").strip()
                for m in self.chatbot.chat_history
            )
            turn_tools = [] if is_opening_turn else self.tool_schemas
            for iteration in range(MAX_TOOL_ITERATIONS):
                tool_calls: list[ToolCallReady] = []
                llm = ToolAwareLLMStream(
                    self.openai_client,
                    temperature=temperature,
                    tools=turn_tools,
                )

                async def _text_only(
                    stream=llm, sink=tool_calls, msgs=messages
                ) -> AsyncIterator[str]:
                    """Split the typed event stream: speak text, collect calls.

                    The filler is yielded here, on the FIRST tool call, rather
                    than after the stream drains. By that point the model has
                    already finished deciding and the guest has heard nothing
                    for the whole round-trip -- the filler would arrive after
                    the silence it exists to cover.
                    """
                    said_something = False
                    filled = False

                    async for event in stream.chat_completion_events(msgs):
                        if isinstance(event, TextDelta):
                            said_something = True
                            yield event.text
                        elif isinstance(event, ToolCallReady):
                            # Never spoken: this is an action, not speech.
                            sink.append(event)
                            if TOOL_FILLER_ENABLED and not said_something and not filled:
                                # The model gave no lead-in of its own, so cover
                                # the tool round-trip that is about to happen.
                                filled = True
                                filler = random.choice(TOOL_FILLER_PHRASES)
                                logger.info(
                                    "No lead-in from model; using filler: %s", filler
                                )
                                yield filler

                async for delta in rechunk_to_words(_text_only()):
                    delta = strip_unpronounceable(delta)

                    await self.output_queue.put(
                        ora.UnmuteResponseTextDeltaReady(delta=delta)
                    )

                    mt.VLLM_RECV_WORDS.inc()
                    response_words.append(delta)

                    if time_to_first_token is None:
                        time_to_first_token = llm_stopwatch.time()
                        self.debug_dict["timing"]["to_first_token"] = (
                            time_to_first_token
                        )
                        mt.VLLM_TTFT.observe(time_to_first_token)
                        logger.info("Sending first word to TTS: %s", delta)

                    self.tts_output_stopwatch.start_if_not_started()
                    try:
                        tts = await quest.get()
                    except Exception:
                        error_from_tts = True
                        raise

                    if len(self.chatbot.chat_history) > generating_message_i:
                        interrupted = True
                        break  # We've been interrupted

                    assert isinstance(delta, str)  # make Pyright happy
                    await tts.send(delta)

                if interrupted or not tool_calls:
                    # Nothing to act on: this was an ordinary spoken turn.
                    break

                if iteration == MAX_TOOL_ITERATIONS - 1:
                    logger.warning(
                        "Hit MAX_TOOL_ITERATIONS=%d, stopping the tool loop.",
                        MAX_TOOL_ITERATIONS,
                    )
                    break

                # Act, then feed the results back so the LLM can speak them.
                results = await self._run_tool_calls(tool_calls)
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.args),
                                },
                            }
                            for call in tool_calls
                        ],
                    }
                )
                # strict=: results are 1:1 with calls, and pairing them wrong
                # would attach one tool's outcome to another's message.
                for call, envelope in zip(tool_calls, results, strict=True):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.name,
                            # The backend owns truth. Hand the model the whole
                            # envelope so it repeats `speakable` rather than
                            # inventing a confirmation of its own.
                            "content": json.dumps(envelope),
                        }
                    )

            await self.output_queue.put(
                # The words include the whitespace, so no need to add it here
                ora.ResponseTextDone(text="".join(response_words))
            )

            if tts is not None:
                logger.info("Sending TTS EOS.")
                await tts.send(TTSClientEosMessage())
        except asyncio.CancelledError:
            mt.VLLM_INTERRUPTS.inc()
            raise
        except Exception:
            if not error_from_tts:
                mt.VLLM_HARD_ERRORS.inc()
            raise
        finally:
            logger.info("End of VLLM, after %d words.", len(response_words))
            mt.VLLM_ACTIVE_SESSIONS.dec()
            mt.VLLM_REPLY_LENGTH.observe(len(response_words))
            mt.VLLM_GEN_DURATION.observe(llm_stopwatch.time())

    def audio_received_sec(self) -> float:
        """How much audio has been received in seconds. Used instead of time.time().

        This is so that we aren't tied to real-time streaming.
        """
        return self.n_samples_received / self.input_sample_rate

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        stt = self.stt
        assert stt is not None
        sr = frame[0]
        assert sr == self.input_sample_rate

        assert frame[1].shape[0] == 1  # Mono
        array = frame[1][0]

        self.n_samples_received += array.shape[0]

        # If this doesn't update, it means the receive loop isn't running because
        # the process is busy with something else, which is bad.
        self.debug_dict["last_receive_time"] = self.audio_received_sec()
        float_audio = audio_to_float32(array)
        self.turn_audio.append(float_audio)

        self.debug_plot_data.append(
            {
                "t": self.audio_received_sec(),
                "amplitude": float(np.sqrt((float_audio**2).mean())),
                "pause_prediction": stt.pause_prediction.value,
            }
        )

        if self.chatbot.conversation_state() == "bot_speaking":
            # Periodically update this not to trigger the "long silence" accidentally.
            self.waiting_for_user_start_time = self.audio_received_sec()

        if TTS_DEBUGGING_TEXT is not None:
            assert self.audio_input_override is None, (
                "Can't use both TTS_DEBUGGING_TEXT and audio input override."
            )

            # Debugging mode: always send a fixed string when it's the user's turn.
            if self.chatbot.conversation_state() == "waiting_for_user":
                logger.info("Using TTS debugging text. Ignoring microphone.")
                self.chatbot.chat_history.append(
                    {"role": "user", "content": TTS_DEBUGGING_TEXT}
                )
                await self._generate_response()
            return

        if (
            len(self.chatbot.chat_history) == 1
            # Wait until the instructions are updated. A bit hacky
            and self.chatbot.get_instructions() is not None
        ):
            logger.info("Generating initial response.")
            await self._generate_response()

        if self.audio_input_override is not None:
            frame = (frame[0], self.audio_input_override.override(frame[1]))

        if self.chatbot.conversation_state() == "user_speaking":
            self.debug_dict["timing"] = {}

        await stt.send_audio(array)
        if self.stt_end_of_flush_time is None:
            await self.detect_long_silence()

            if self.determine_pause():
                logger.info("Pause detected")
                await self.output_queue.put(ora.InputAudioBufferSpeechStopped())

                self.stt_end_of_flush_time = stt.current_time + stt.delay_sec
                self.stt_flush_timer = Stopwatch()
                num_frames = (
                    int(math.ceil(stt.delay_sec / FRAME_TIME_SEC)) + 1
                )  # some safety margin.
                zero = np.zeros(SAMPLES_PER_FRAME, dtype=np.float32)
                for _ in range(num_frames):
                    await stt.send_audio(zero)
            elif (
                self.chatbot.conversation_state() == "bot_speaking"
                and stt.pause_prediction.value < 0.4
                and self.audio_received_sec() > UNINTERRUPTIBLE_BY_VAD_TIME_SEC
            ):
                logger.info("Interruption by STT-VAD")
                await self.interrupt_bot()
                await self.add_chat_message_delta("", "user")
        else:
            # We do not try to detect interruption here, the STT would be processing
            # a chunk full of 0, so there is little chance the pause score would indicate an interruption.
            if stt.current_time > self.stt_end_of_flush_time:
                self.stt_end_of_flush_time = None
                elapsed = self.stt_flush_timer.time()
                rtf = stt.delay_sec / elapsed
                logger.info(
                    "Flushing finished, took %.1f ms, RTF: %.1f", elapsed * 1000, rtf
                )
                await self._generate_response()

    def determine_pause(self) -> bool:
        stt = self.stt
        if stt is None:
            return False
        if self.chatbot.conversation_state() != "user_speaking":
            return False

        # This is how much wall clock time has passed since we received the last ASR
        # message. Assumes the ASR connection is healthy, so that stt.sent_samples is up
        # to date.
        time_since_last_message = (
            stt.sent_samples / self.input_sample_rate
        ) - self.stt_last_message_time
        self.debug_dict["time_since_last_message"] = time_since_last_message

        if stt.pause_prediction.value > 0.6:
            self.debug_dict["timing"]["pause_detection"] = time_since_last_message
            return True
        else:
            return False

    async def emit(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> HandlerOutput | None:
        output_queue_item = await wait_for_item(self.output_queue)

        if output_queue_item is not None:
            return output_queue_item
        else:
            if self.last_additional_output_update < self.audio_received_sec() - 1:
                # If we have nothing to emit, at least update the debug dict.
                # Don't update too often for performance reasons
                self.last_additional_output_update = self.audio_received_sec()
                return self.get_gradio_update()
            else:
                return None

    def copy(self):
        return UnmuteHandler()

    async def __aenter__(self) -> None:
        await self.quest_manager.__aenter__()

    async def start_up(self):
        await self.start_up_stt()
        self.waiting_for_user_start_time = self.audio_received_sec()

    async def __aexit__(self, *exc: Any) -> None:
        return await self.quest_manager.__aexit__(*exc)

    async def start_up_stt(self):
        async def _init() -> SpeechToText:
            return await find_instance("stt", SpeechToText)

        async def _run(stt: SpeechToText):
            await self._stt_loop(stt)

        async def _close(stt: SpeechToText):
            await stt.shutdown()

        quest = await self.quest_manager.add(Quest("stt", _init, _run, _close))
        # We want to be sure to have the STT before starting anything.
        await quest.get()

    async def _stt_loop(self, stt: SpeechToText):
        try:
            async for data in stt:
                if isinstance(data, STTMarkerMessage):
                    # Ignore the marker messages
                    continue

                await self.output_queue.put(
                    ora.ConversationItemInputAudioTranscriptionDelta(
                        delta=data.text,
                        start_time=data.start_time,
                    )
                )

                # The STT sends an empty string as the first message, but we
                # don't want to add that because it can trigger a pause even
                # if the user hasn't started speaking yet.
                if data.text == "":
                    continue

                if self.chatbot.conversation_state() == "bot_speaking":
                    logger.info("STT-based interruption")
                    await self.interrupt_bot()

                self.stt_last_message_time = data.start_time
                is_new_message = await self.add_chat_message_delta(data.text, "user")
                if is_new_message:
                    # The turn began a little before its first word reached us: the
                    # STT runs delay_sec behind, and the word itself takes time.
                    self._turn_start_sample = max(
                        0,
                        self.turn_audio.end - int((stt.delay_sec + 1.0) * SAMPLE_RATE),
                    )
                    # Ensure we don't stop after the first word if the VAD didn't have
                    # time to react.
                    stt.pause_prediction.value = 0.0
                    await self.output_queue.put(ora.InputAudioBufferSpeechStarted())
        except websockets.ConnectionClosed:
            logger.info("STT connection closed while receiving messages.")

    async def start_up_tts(self, generating_message_i: int) -> Quest[TextToSpeech]:
        async def _init() -> TextToSpeech:
            factory = partial(
                TextToSpeech,
                recorder=self.recorder,
                get_time=self.audio_received_sec,
                voice=self.tts_voice,
            )
            sleep_time = 0.05
            sleep_growth = 1.5
            max_sleep = 1.0
            trials = 5
            for trial in range(trials):
                try:
                    tts = await find_instance("tts", factory)
                except Exception:
                    if trial == trials - 1:
                        raise
                    logger.warning("Will sleep for %.4f sec", sleep_time)
                    await asyncio.sleep(sleep_time)
                    sleep_time = min(max_sleep, sleep_time * sleep_growth)
                    error = make_ora_error(
                        type="warning",
                        message="Looking for the resources, expect some latency.",
                    )
                    await self.output_queue.put(error)
                else:
                    return tts
            raise AssertionError("Too many unexpected packets.")

        async def _run(tts: TextToSpeech):
            await self._tts_loop(tts, generating_message_i)

        async def _close(tts: TextToSpeech):
            await tts.shutdown()

        return await self.quest_manager.add(Quest("tts", _init, _run, _close))

    async def _tts_loop(self, tts: TextToSpeech, generating_message_i: int):
        # On interruption, we swap the output queue. This will ensure that this worker
        # can never accidentally push to the new queue if it's interrupted.
        output_queue = self.output_queue
        try:
            audio_started = None

            async for message in tts:
                if audio_started is not None:
                    time_since_start = self.audio_received_sec() - audio_started
                    time_received = tts.received_samples / self.input_sample_rate
                    time_received_yielded = (
                        tts.received_samples_yielded / self.input_sample_rate
                    )
                    assert self.input_sample_rate == SAMPLE_RATE
                    self.debug_dict["tts_throughput"] = {
                        "time_received": round(time_received, 2),
                        "time_received_yielded": round(time_received_yielded, 2),
                        "time_since_start": round(time_since_start, 2),
                        "ratio": round(
                            time_received_yielded / (time_since_start + 0.01), 2
                        ),
                    }

                if len(self.chatbot.chat_history) > generating_message_i:
                    break

                if isinstance(message, TTSAudioMessage):
                    t = self.tts_output_stopwatch.stop()
                    if t is not None:
                        self.debug_dict["timing"]["tts_audio"] = t

                    audio = np.array(message.pcm, dtype=np.float32)
                    assert self.output_sample_rate == SAMPLE_RATE

                    await output_queue.put((SAMPLE_RATE, audio))

                    if audio_started is None:
                        audio_started = self.audio_received_sec()
                elif isinstance(message, TTSTextMessage):
                    await output_queue.put(ora.ResponseTextDelta(delta=message.text))
                    await self.add_chat_message_delta(
                        message.text,
                        "assistant",
                        generating_message_i=generating_message_i,
                    )
                else:
                    logger.warning("Got unexpected message from TTS: %s", message.type)

        except websockets.ConnectionClosedError as e:
            logger.error(f"TTS connection closed with an error: {e}")

        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        await output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        message = self.chatbot.last_message("assistant")
        if message is None:
            logger.warning("No message to send in TTS shutdown.")
            message = ""

        # It's convenient to have the whole chat history available in the client
        # after the response is done, so send the "gradio update"
        await self.output_queue.put(self.get_gradio_update())
        await self.output_queue.put(ora.ResponseAudioDone())

        # Signal that the turn is over by adding an empty message.
        await self.add_chat_message_delta("", "user")

        await asyncio.sleep(1)
        await self.check_for_bot_goodbye()
        self.waiting_for_user_start_time = self.audio_received_sec()

    async def interrupt_bot(self):
        if self.chatbot.conversation_state() != "bot_speaking":
            raise RuntimeError(
                "Can't interrupt bot when conversation state is "
                f"{self.chatbot.conversation_state()}"
            )

        await self.add_chat_message_delta(INTERRUPTION_CHAR, "assistant")

        if self._clear_queue is not None:
            # Clear any audio queued up by FastRTC's emit().
            # Not sure under what circumstatnces this is None.
            self._clear_queue()
        self.output_queue = asyncio.Queue()  # Clear our own queue too

        # Push some silence to flush the Opus state.
        # Not sure that this is actually needed.
        await self.output_queue.put(
            (SAMPLE_RATE, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
        )

        await self.output_queue.put(ora.UnmuteInterruptedByVAD())

        await self.quest_manager.remove("tts")
        await self.quest_manager.remove("llm")

    async def check_for_bot_goodbye(self):
        last_assistant_message = next(
            (
                msg
                for msg in reversed(self.chatbot.chat_history)
                if msg["role"] == "assistant"
            ),
            {"content": ""},
        )["content"]

        # Using function calling would be a more robust solution, but it would make it
        # harder to swap LLMs.
        if last_assistant_message.lower().endswith("bye!"):
            await self.output_queue.put(
                CloseStream("The assistant ended the conversation. Bye!")
            )

    async def detect_long_silence(self):
        """Handle situations where the user doesn't answer for a while."""
        if (
            self.chatbot.conversation_state() == "waiting_for_user"
            and (self.audio_received_sec() - self.waiting_for_user_start_time)
            > USER_SILENCE_TIMEOUT
        ):
            # This will trigger pause detection because it changes the conversation
            # state to "user_speaking".
            # The system prompt has a rule that tells it how to handle the "..."
            # messages.
            logger.info("Long silence detected.")
            await self.add_chat_message_delta(USER_SILENCE_MARKER, "user")

    async def update_session(self, session: ora.SessionConfig):
        if session.instructions:
            self.chatbot.set_instructions(session.instructions)

        if session.voice:
            self.tts_voice = session.voice

        if session.tools is not None:
            # An explicit empty list disables tool calling for this session;
            # omitting the field entirely keeps the default set.
            self.tool_schemas = session.tools
            logger.info(
                "Session tools set: %s",
                [t.get("function", {}).get("name", "?") for t in session.tools]
                or "(none)",
            )

        if session.client_tools is not None:
            self.client_tools = frozenset(session.client_tools)
            logger.info("Client-run tools: %s", sorted(self.client_tools) or "(none)")

        if not session.allow_recording and self.recorder:
            await self.recorder.add_event("client", ora.SessionUpdate(session=session))
            await self.recorder.shutdown(keep_recording=False)
            self.recorder = None
            logger.info("Recording disabled for a session.")
