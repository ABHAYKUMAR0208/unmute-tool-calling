"""Better words for each caller turn, from an OpenRouter transcription model.

Kyutai's streaming STT is kept for what it is good at: knowing, live, when the
caller has stopped talking and when they talk over Kelly. Its words are weak on
Indian English ("I'm Abhay" came out as "Duda", "tap" as "top"), and the whole
call is built on those words: the reply, and the transcript the CRM reads.

So the handler keeps the caller's recent audio (TurnAudio). When a turn ends,
that turn's audio goes to a batch transcription model on OpenRouter (Whisper by
default) and its text replaces Kyutai's before the LLM sees it. If the model is
slow or fails, Kyutai's text stands, so a turn is never lost or held up long.
"""

import base64
import io
import re
import wave
from logging import getLogger
from typing import Any

import httpx
import numpy as np

from unmute.kyutai_constants import (
    SAMPLE_RATE,
    STT_REFINE_API_KEY,
    STT_REFINE_LANGUAGE,
    STT_REFINE_MODEL,
    STT_REFINE_TIMEOUT_S,
    STT_REFINE_URL,
)

logger = getLogger(__name__)

# Whisper-family models expect 16 kHz; it also makes the upload a third smaller.
UPLOAD_SAMPLE_RATE = 16000
# Turns shorter than this are "mm", "yes" and the like: Kyutai handles them fine
# and a round trip isn't worth it.
MIN_TURN_SEC = 0.6
# How much audio to keep. A single turn longer than this keeps its last part.
MAX_KEPT_SEC = 45.0

# What Whisper says for silence or noise. Taken as "no better text".
_HALLUCINATIONS = re.compile(
    r"^\W*(thank you( (so much|for watching))?|thanks for watching|you|bye|"
    r"subtitles? by .*|please subscribe.*)\W*$",
    re.IGNORECASE,
)


def refine_enabled() -> bool:
    return bool(STT_REFINE_MODEL and STT_REFINE_API_KEY)


class TurnAudio:
    """The caller's recent audio at SAMPLE_RATE, addressable by absolute sample index."""

    def __init__(self, max_sec: float = MAX_KEPT_SEC):
        self.max_samples = int(max_sec * SAMPLE_RATE)
        self.chunks: list[np.ndarray] = []
        self.start = 0  # absolute index of the first kept sample
        self.end = 0  # absolute index one past the last sample

    def append(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not audio.size:
            return
        self.chunks.append(audio)
        self.end += audio.size
        while self.chunks and self.end - self.start - self.chunks[0].size >= self.max_samples:
            self.start += self.chunks.pop(0).size

    def slice(self, start: int, end: int | None = None) -> np.ndarray:
        end = self.end if end is None else min(end, self.end)
        start = max(start, self.start)
        if not self.chunks or start >= end:
            return np.zeros(0, dtype=np.float32)
        kept = np.concatenate(self.chunks)
        return kept[start - self.start : end - self.start]


def to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Float audio at `sample_rate` -> 16-bit mono WAV at UPLOAD_SAMPLE_RATE."""
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != UPLOAD_SAMPLE_RATE and audio.size:
        n_out = int(round(audio.size * UPLOAD_SAMPLE_RATE / sample_rate))
        positions = np.linspace(0, audio.size - 1, num=max(n_out, 1))
        audio = np.interp(positions, np.arange(audio.size), audio)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(UPLOAD_SAMPLE_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()


def usable(text: str | None) -> str | None:
    """The model's text if it is worth using, else None (keep Kyutai's)."""
    text = " ".join((text or "").split())
    if not text or _HALLUCINATIONS.match(text):
        return None
    return text


async def transcribe(
    audio: np.ndarray,
    client: httpx.AsyncClient,
    sample_rate: int = SAMPLE_RATE,
) -> str | None:
    """One turn's audio -> text, or None if it can't be had in time."""
    if audio.size < MIN_TURN_SEC * sample_rate:
        return None
    body: dict[str, Any] = {
        "model": STT_REFINE_MODEL,
        "input_audio": {
            "data": base64.b64encode(to_wav(audio, sample_rate)).decode("ascii"),
            "format": "wav",
        },
        "temperature": 0,
    }
    if STT_REFINE_LANGUAGE:
        body["language"] = STT_REFINE_LANGUAGE
    try:
        response = await client.post(
            STT_REFINE_URL,
            json=body,
            headers={"Authorization": f"Bearer {STT_REFINE_API_KEY}"},
            timeout=STT_REFINE_TIMEOUT_S,
        )
        response.raise_for_status()
        return usable(response.json().get("text"))
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Better transcription unavailable, keeping the live one: %r", exc)
        return None


def _last_user_message(chat_history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The caller's latest turn, if the reply now starting answers one."""
    for message in reversed(chat_history):
        if message.get("role") == "assistant" and not str(message.get("content") or "").strip():
            continue  # the empty reply that is just starting
        if message.get("role") != "user":
            return None
        words = str(message.get("content") or "").strip()
        if not words or words == "...":
            return None  # a silence marker, not something the caller said
        return message
    return None


def has_caller_words(chat_history: list[dict[str, Any]]) -> bool:
    return _last_user_message(chat_history) is not None


def replace_last_user_text(chat_history: list[dict[str, Any]], text: str) -> str | None:
    """Swap the words of the caller's latest turn. Returns the old words, or None."""
    message = _last_user_message(chat_history)
    if message is None:
        return None
    old = str(message.get("content") or "")
    message["content"] = text
    return old
