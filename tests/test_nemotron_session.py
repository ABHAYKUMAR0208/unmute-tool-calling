"""The Nemotron STT server's protocol logic, with a fake recogniser and VAD."""

import asyncio

import numpy as np
import pytest

pytest.importorskip("scipy")

from nemotron_stt.session import (  # noqa: E402
    STEP_SAMPLES,
    PauseTracker,
    SttSession,
    WordTracker,
    to_model_rate,
)


class FakeStream:
    """Returns the next scripted transcript for each chunk it is fed."""

    chunk_samples = 2559  # 160 ms at 16 kHz, as the real engine

    def __init__(self, transcripts):
        self.transcripts = list(transcripts)
        self.calls = 0
        self.resets = 0

    def step(self, audio):
        assert audio.size == self.chunk_samples
        text = self.transcripts[min(self.calls, len(self.transcripts) - 1)]
        self.calls += 1
        return text

    def reset(self):
        # Like the real model: a fresh stream starts from an empty transcript.
        self.resets += 1
        self.transcripts, self.calls = [""], 0


class LoudnessVad:
    def speech_prob(self, window):
        return 1.0 if float(np.abs(window).mean()) > 0.01 else 0.0


def speech(seconds):
    return (np.ones(int(24000 * seconds), dtype=np.float32) * 0.2).tolist()


def silence(seconds):
    return np.zeros(int(24000 * seconds), dtype=np.float32).tolist()


async def run(session, *audio_blocks, settle=0.2):
    sent = session.sent
    worker = asyncio.create_task(session.run_recogniser())
    for block in audio_blocks:
        for i in range(0, len(block), STEP_SAMPLES):  # 80 ms at a time, like the backend
            await session.on_audio(block[i : i + STEP_SAMPLES])
            await asyncio.sleep(0.01)  # frames arrive over time; the recogniser keeps up
    await asyncio.sleep(settle)
    worker.cancel()
    return sent


def make_session(transcripts, **pause):
    sent = []

    async def send(message):
        sent.append(message)

    stream = FakeStream(transcripts)
    session = SttSession(stream, LoudnessVad(), send, PauseTracker(**pause))
    session.sent = sent
    return session, stream


def test_resampling_is_exactly_three_to_two():
    assert to_model_rate(np.zeros(STEP_SAMPLES)).size == 1280


def test_word_tracker_holds_back_the_growing_last_word():
    words = WordTracker()
    assert words.update("I") == []
    assert words.update("I want a bur") == ["I", "want", "a"]
    assert words.update("I want a burger") == []
    assert words.update("I want a burger please") == ["burger"]
    assert words.flush("I want a burger please.") == ["please."]
    assert words.update("I want a burger please.") == []


def test_pause_signal_crosses_the_backends_threshold_only_after_pause_ms():
    pause = PauseTracker(pause_ms=600, min_speech_ms=160)
    for _ in range(10):
        pause.update(1.0)  # 320 ms of speech
    assert pause.speaking and pause.value == 0.0
    for _ in range(15):
        pause.update(0.0)  # 480 ms of silence
    assert 0.4 < pause.value < 0.6 and not pause.turn_over
    for _ in range(5):
        pause.update(0.0)  # past 600 ms
    assert pause.turn_over and pause.value == 1.0


def test_a_click_does_not_count_as_speech():
    pause = PauseTracker(pause_ms=600, min_speech_ms=160)
    pause.update(1.0)  # one 32 ms window
    assert not pause.speaking and pause.value == 1.0


@pytest.mark.asyncio
async def test_a_turn_gives_steps_in_real_time_every_word_once_and_a_pause():
    script = ["", "I", "I want", "I want a bur", "I want a burger", "I want a burger please"]
    session, stream = make_session(script)
    sent = await run(session, silence(0.4), speech(1.2), silence(1.6))

    steps = [m for m in sent if m["type"] == "Step"]
    assert len(steps) == int(3.2 * 24000) // STEP_SAMPLES  # one per 80 ms received
    assert [s["step_idx"] for s in steps] == list(range(len(steps)))

    words = [m["text"] for m in sent if m["type"] == "Word"]
    assert words == ["I", "want", "a", "burger", "please"]  # each once, last one at the pause

    pause_values = [s["prs"][2] for s in steps]
    assert min(pause_values) == 0.0  # while speaking: could interrupt Kelly
    assert pause_values[-1] == 1.0  # after the turn: the backend replies
    assert stream.resets >= 1  # the next turn starts fresh
    # After the pause the backend waits 0.5 s (6 steps) before replying: the
    # last word must be in by then or the reply would miss it.
    first_pause = next(i for i, v in enumerate(pause_values) if v > 0.6 and i > 10)
    last_word = max(i for i, m in enumerate(sent) if m["type"] == "Word")
    assert last_word < sent.index(steps[first_pause + 6])


@pytest.mark.asyncio
async def test_marker_is_echoed_after_the_audio_sent_before_it():
    session, _ = make_session(["", "hello there", "hello there friend"])
    sent = session.sent
    worker = asyncio.create_task(session.run_recogniser())
    block = speech(0.8)
    for i in range(0, len(block), STEP_SAMPLES):
        await session.on_audio(block[i : i + STEP_SAMPLES])
    await session.on_marker(7)
    await asyncio.sleep(0.2)
    worker.cancel()
    kinds = [m["type"] for m in sent]
    assert kinds[-1] == "Marker" and sent[-1]["id"] == 7
    assert "Word" in kinds and kinds.index("Word") < kinds.index("Marker")
