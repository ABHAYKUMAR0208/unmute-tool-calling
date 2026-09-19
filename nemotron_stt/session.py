"""One caller's STT stream, speaking Kyutai's STT protocol (moshi-server).

The backend's STT client (unmute/stt/speech_to_text.py) expects, per websocket:
  - {"type": "Ready"} once, when the connection is up;
  - one {"type": "Step", "step_idx", "prs": [4 floats]} per 80 ms of audio it
    sent, in real time. prs[2] is the chance the caller has finished their
    turn: above 0.6 the backend treats it as a pause and replies; below 0.4
    while Kelly speaks it counts as the caller talking over her;
  - {"type": "Word", "text", "start_time"} for each recognised word;
  - {"type": "Marker", "id"} echoed back after the audio sent before it.

Kyutai's model produces both the words and a learned pause signal. Here the
words come from Nemotron (engine.py) and the pause signal from a voice
activity detector: silence that lasts PAUSE_MS counts as the end of a turn.

Everything in this file is plain Python and runs without NeMo or a GPU, so the
protocol can be tested on its own (tests/test_nemotron_session.py).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

import numpy as np

logger = logging.getLogger("nemotron_stt.session")

CLIENT_SAMPLE_RATE = 24000  # what the backend sends
MODEL_SAMPLE_RATE = 16000  # what Nemotron and Silero expect
STEP_SAMPLES = 1920  # one backend frame: 80 ms at 24 kHz
STEP_SEC = STEP_SAMPLES / CLIENT_SAMPLE_RATE
VAD_WINDOW = 512  # Silero's window at 16 kHz (32 ms)
VAD_WINDOW_SEC = VAD_WINDOW / MODEL_SAMPLE_RATE


class Stream(Protocol):
    """One caller's recogniser state (engine.NemotronStream, or a fake in tests)."""

    chunk_samples: int  # 16 kHz samples per step() call

    def step(self, audio: np.ndarray) -> str:
        """Feed exactly chunk_samples; return the whole transcript since reset()."""
        ...

    def reset(self) -> None: ...


class Vad(Protocol):
    def speech_prob(self, window: np.ndarray) -> float:
        """Probability of speech in one VAD_WINDOW of 16 kHz audio."""
        ...


def to_model_rate(pcm_24k: np.ndarray) -> np.ndarray:
    """24 kHz -> 16 kHz. Frames are 1920 samples, so this is an exact 3:2."""
    from scipy.signal import resample_poly

    return resample_poly(np.asarray(pcm_24k, dtype=np.float32), 2, 3).astype(np.float32)


@dataclass
class PauseTracker:
    """Turns per-window speech probabilities into the backend's pause signal.

    Speech must last MIN_SPEECH_MS before it counts, so a click or a cough on
    the line doesn't interrupt Kelly. Silence then has to last PAUSE_MS before
    the caller is taken to have finished; until then the value stays below the
    backend's 0.6 threshold.
    """

    pause_ms: float = 600.0
    min_speech_ms: float = 160.0
    threshold: float = 0.5
    silence_ms: float = 10_000.0  # start as "not speaking"
    speech_ms: float = 0.0

    def update(self, speech_prob: float, window_ms: float = VAD_WINDOW_SEC * 1000) -> None:
        if speech_prob >= self.threshold:
            self.speech_ms += window_ms
            if self.speech_ms >= self.min_speech_ms:
                self.silence_ms = 0.0
        else:
            self.speech_ms = 0.0
            self.silence_ms += window_ms

    @property
    def speaking(self) -> bool:
        return self.silence_ms == 0.0

    @property
    def turn_over(self) -> bool:
        return self.silence_ms >= self.pause_ms

    @property
    def value(self) -> float:
        """prs[2]: 0 while speaking, rising to 0.6 at pause_ms, then 1."""
        if self.turn_over:
            return 1.0
        return 0.6 * self.silence_ms / self.pause_ms


class WordTracker:
    """Turns the growing transcript into Word messages, each word sent once.

    Nemotron (RNN-T, greedy) only ever appends to its transcript, but the last
    word may still be growing ("bur" -> "burger"), so it is held back until a
    new word starts after it or the turn ends (flush).
    """

    def __init__(self) -> None:
        self.sent = 0  # words already sent from the current transcript

    def update(self, transcript: str) -> list[str]:
        words = transcript.split()
        if self.sent > len(words):  # the model restarted underneath us
            self.sent = len(words)
        stable = len(words) if transcript.endswith((" ", "\n")) else len(words) - 1
        new = words[self.sent : max(stable, self.sent)]
        self.sent += len(new)
        return new

    def flush(self, transcript: str) -> list[str]:
        words = transcript.split()
        new = words[self.sent :]
        self.sent = len(words)
        return new

    def reset(self) -> None:
        self.sent = 0


Send = Callable[[dict[str, Any]], Awaitable[None]]


class SttSession:
    """Protocol state for one websocket. The server feeds it; it calls `send`."""

    def __init__(self, stream: Stream, vad: Vad, send: Send, pause: PauseTracker | None = None):
        self.stream = stream
        self.vad = vad
        self.send = send
        self.pause = pause or PauseTracker()
        self.words = WordTracker()

        self.step_idx = 0
        self._pending_24k = np.zeros(0, dtype=np.float32)  # until a full 80 ms frame
        self._vad_16k = np.zeros(0, dtype=np.float32)  # until a full VAD window
        self._asr_16k = np.zeros(0, dtype=np.float32)  # until a full model chunk
        self._audio_sec = 0.0  # audio received so far
        self._asr_sec = 0.0  # audio the recogniser has consumed
        self._transcript = ""
        # True until the caller first speaks: a turn can only end after speech.
        self._flushed_this_pause = True
        # Model work runs beside the receive loop, so Steps keep their pace.
        self._jobs: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    # --- called by the server's receive loop ------------------------------

    async def on_audio(self, pcm: list[float] | np.ndarray) -> None:
        self._pending_24k = np.concatenate([self._pending_24k, np.asarray(pcm, dtype=np.float32)])
        while self._pending_24k.size >= STEP_SAMPLES:
            frame, self._pending_24k = self._pending_24k[:STEP_SAMPLES], self._pending_24k[STEP_SAMPLES:]
            audio_16k = to_model_rate(frame)
            self._audio_sec += STEP_SEC
            self._update_vad(audio_16k)
            await self.send({"type": "Step", "step_idx": self.step_idx, "prs": [self.pause.value] * 4})
            self.step_idx += 1
            await self._jobs.put(("audio", audio_16k))
            if self.pause.turn_over and not self._flushed_this_pause:
                # End of the caller's turn: send the held-back last word now,
                # so it reaches the backend before it starts replying.
                self._flushed_this_pause = True
                await self._jobs.put(("flush", None))
            elif self.pause.speaking:
                self._flushed_this_pause = False

    async def on_marker(self, marker_id: int) -> None:
        await self._jobs.put(("marker", marker_id))  # echoed once earlier audio is done

    # --- model worker -----------------------------------------------------

    async def run_recogniser(self) -> None:
        """Consume queued audio in model-sized chunks. Runs until cancelled."""
        while True:
            kind, payload = await self._jobs.get()
            if kind == "audio":
                self._asr_16k = np.concatenate([self._asr_16k, payload])
                while self._asr_16k.size >= self.stream.chunk_samples:
                    chunk = self._asr_16k[: self.stream.chunk_samples]
                    self._asr_16k = self._asr_16k[self.stream.chunk_samples :]
                    chunk_start = self._asr_sec
                    self._asr_sec += chunk.size / MODEL_SAMPLE_RATE
                    self._transcript = await asyncio.to_thread(self.stream.step, chunk)
                    await self._send_words(self.words.update(self._transcript), chunk_start)
            elif kind == "flush":
                await self._send_words(self.words.flush(self._transcript), self._asr_sec)
                if self._transcript.strip():
                    logger.info("Heard: %s", self._transcript.strip())
                # Start the next turn fresh: bounded memory and no carry-over.
                await asyncio.to_thread(self.stream.reset)
                self.words.reset()
                self._transcript = ""
            elif kind == "marker":
                await self.send({"type": "Marker", "id": payload})

    # --- internals ----------------------------------------------------------

    def _update_vad(self, audio_16k: np.ndarray) -> None:
        self._vad_16k = np.concatenate([self._vad_16k, audio_16k])
        while self._vad_16k.size >= VAD_WINDOW:
            window, self._vad_16k = self._vad_16k[:VAD_WINDOW], self._vad_16k[VAD_WINDOW:]
            self.pause.update(self.vad.speech_prob(window))

    async def _send_words(self, words: list[str], start_time: float) -> None:
        for word in words:
            await self.send({"type": "Word", "text": word, "start_time": start_time})
            await self.send({"type": "EndWord", "stop_time": self._asr_sec})
