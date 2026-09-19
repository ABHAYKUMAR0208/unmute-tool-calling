"""Better per-turn transcription (unmute/stt/refine.py), with a fake OpenRouter."""

import base64
import io
import json
import wave

import httpx
import numpy as np
import pytest

from unmute.kyutai_constants import SAMPLE_RATE
from unmute.stt import refine


def test_turn_audio_slices_by_absolute_sample_and_keeps_a_bounded_window():
    audio = refine.TurnAudio(max_sec=1.0)
    for i in range(30):  # 30 x 0.1 s = 3 s, only the last ~1 s is kept
        audio.append(np.full(SAMPLE_RATE // 10, i, dtype=np.float32))
    assert audio.end == 3 * SAMPLE_RATE
    assert audio.end - audio.start <= SAMPLE_RATE + SAMPLE_RATE // 10
    last = audio.slice(audio.end - SAMPLE_RATE // 10)
    assert last.size == SAMPLE_RATE // 10 and np.all(last == 29)
    assert audio.slice(0).size == audio.end - audio.start  # older audio is gone, not an error
    assert audio.slice(audio.end).size == 0


def test_wav_is_16khz_mono_16bit_with_the_right_length():
    data = refine.to_wav(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    with wave.open(io.BytesIO(data)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
        assert abs(wav.getnframes() - 32000) <= 1


def test_whisper_silence_phrases_are_not_used():
    for text in ("Thank you.", "thanks for watching!", "you", "  ", None):
        assert refine.usable(text) is None
    assert refine.usable("  I'm Abhay and   I need a food service. ") == "I'm Abhay and I need a food service."


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_transcribe_sends_one_wav_turn_and_returns_the_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.update(body)
        seen["wav"] = base64.b64decode(body["input_audio"]["data"])
        return httpx.Response(200, json={"text": "Just order me a burger, please."})

    speech = np.random.default_rng(0).normal(0, 0.1, SAMPLE_RATE * 2).astype(np.float32)
    async with _client(handler) as client:
        text = await refine.transcribe(speech, client)
    assert text == "Just order me a burger, please."
    assert seen["model"] == refine.STT_REFINE_MODEL
    assert seen["input_audio"]["format"] == "wav" and seen["wav"][:4] == b"RIFF"
    assert seen.get("language") == "en"


@pytest.mark.asyncio
async def test_short_turns_skip_the_round_trip():
    def handler(request):  # pragma: no cover - must not be called
        raise AssertionError("no request expected for a short turn")

    async with _client(handler) as client:
        assert await refine.transcribe(np.zeros(SAMPLE_RATE // 4, dtype=np.float32), client) is None


@pytest.mark.asyncio
async def test_failures_and_timeouts_keep_the_live_words():
    speech = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.1

    async with _client(lambda r: httpx.Response(502, json={"error": "upstream"})) as client:
        assert await refine.transcribe(speech, client) is None

    def slow(request):
        raise httpx.ReadTimeout("too slow", request=request)

    async with _client(slow) as client:
        assert await refine.transcribe(speech, client) is None

    async with _client(lambda r: httpx.Response(200, content=b"not json")) as client:
        assert await refine.transcribe(speech, client) is None


def test_only_the_callers_latest_turn_is_replaced():
    history = [
        {"role": "system", "content": "You are Kelly."},
        {"role": "assistant", "content": "Welcome to the reception."},
        {"role": "user", "content": "yeah I'm Duda and I need a food service"},
        {"role": "assistant", "content": ""},  # the reply that is starting
    ]
    old = refine.replace_last_user_text(history, "Yeah, I'm Abhay and I need a food service.")
    assert old == "yeah I'm Duda and I need a food service"
    assert history[2]["content"] == "Yeah, I'm Abhay and I need a food service."
    assert history[1]["content"] == "Welcome to the reception."

    # A silence check-in ("...") is not the caller's words.
    silence = history[:2] + [{"role": "user", "content": "..."}, {"role": "assistant", "content": ""}]
    assert refine.replace_last_user_text(silence, "anything") is None
    # The greeting has no caller turn before it.
    assert refine.replace_last_user_text([{"role": "system", "content": "x"}], "anything") is None
    assert not refine.has_caller_words([{"role": "system", "content": "x"}, {"role": "assistant", "content": ""}])
