"""Nemotron STT server: a drop-in for Kyutai's STT (moshi-server) on port 8090.

Same websocket path and msgpack messages, so the backend needs no change (see
session.py for the protocol). Run with nemotron_stt/run.sh.

Settings (environment):
  NEMOTRON_MODEL           default nvidia/nemotron-speech-streaming-en-0.6b
  NEMOTRON_CHUNK_MS        80, 160 (default), 560 or 1120: latency vs accuracy
  NEMOTRON_PAUSE_MS        silence that ends the caller's turn, default 600
  NEMOTRON_MIN_SPEECH_MS   speech that counts (and can interrupt Kelly), default 160
  NEMOTRON_VAD_THRESHOLD   Silero speech probability threshold, default 0.5
  NEMOTRON_MAX_SESSIONS    concurrent calls, default 8
  NEMOTRON_PORT            default 8090
"""

from __future__ import annotations

import asyncio
import logging
import os

import msgpack
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from nemotron_stt.engine import NemotronModel
from nemotron_stt.session import PauseTracker, SttSession
from nemotron_stt.vad import SileroVad

PATH = "/api/asr-streaming"
logger = logging.getLogger("nemotron_stt.server")

MODEL_NAME = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-speech-streaming-en-0.6b")
CHUNK_MS = int(os.environ.get("NEMOTRON_CHUNK_MS", "160"))
PAUSE_MS = float(os.environ.get("NEMOTRON_PAUSE_MS", "600"))
MIN_SPEECH_MS = float(os.environ.get("NEMOTRON_MIN_SPEECH_MS", "160"))
VAD_THRESHOLD = float(os.environ.get("NEMOTRON_VAD_THRESHOLD", "0.5"))
MAX_SESSIONS = int(os.environ.get("NEMOTRON_MAX_SESSIONS", "8"))
PORT = int(os.environ.get("NEMOTRON_PORT", "8090"))
# Loopback only: the backend always reaches this over ws://localhost:8090
# (see unmute/kyutai_constants.py), so this never needs to be reachable from
# outside the pod. Binding 0.0.0.0 made RunPod's proxy auto-expose it
# publicly, which is what was producing the periodic 426 "Upgrade Required"
# log lines (external probes/health-checks hitting the port with plain HTTP,
# not real call traffic — real traffic is loopback and would never surface
# as a 426). Override with NEMOTRON_HOST if you have a real reason to bind
# wider (e.g. running the backend on a different host/container).
HOST = os.environ.get("NEMOTRON_HOST", "127.0.0.1")

MODEL: NemotronModel | None = None
active_sessions = 0


def _pack(message: dict) -> bytes:
    return msgpack.packb(message, use_bin_type=True, use_single_float=True)


async def handle(ws: ServerConnection) -> None:
    global active_sessions
    if ws.request is None or ws.request.path.split("?")[0] != PATH:
        await ws.close(1008, "unknown path")
        return
    if active_sessions >= MAX_SESSIONS:
        # The backend treats this as "at capacity" and tries again.
        await ws.send(_pack({"type": "Error", "message": "at capacity"}))
        await ws.close()
        return

    active_sessions += 1
    send_lock = asyncio.Lock()

    async def send(message: dict) -> None:
        async with send_lock:
            await ws.send(_pack(message))

    # Ready first: the backend gives up after 0.5 s. Setting up the caller's
    # recogniser and VAD takes a moment, and audio just waits in the socket.
    await send({"type": "Ready"})
    assert MODEL is not None
    stream = await asyncio.to_thread(MODEL.new_stream)
    vad = await asyncio.to_thread(SileroVad)
    session = SttSession(
        stream,
        vad,
        send,
        PauseTracker(pause_ms=PAUSE_MS, min_speech_ms=MIN_SPEECH_MS, threshold=VAD_THRESHOLD),
    )
    worker = asyncio.create_task(session.run_recogniser())
    logger.info("Call connected (%d active)", active_sessions)
    try:
        async for raw in ws:
            if not isinstance(raw, bytes) or raw == b"\0":
                continue
            data = msgpack.unpackb(raw)
            kind = data.get("type")
            if kind == "Audio":
                await session.on_audio(data.get("pcm") or [])
            elif kind == "Marker":
                await session.on_marker(int(data.get("id", 0)))
    except ConnectionClosed:
        pass
    finally:
        worker.cancel()
        active_sessions -= 1
        logger.info("Call ended (%d active)", active_sessions)


async def main() -> None:
    global MODEL
    MODEL = await asyncio.to_thread(NemotronModel, MODEL_NAME, CHUNK_MS)
    async with serve(handle, HOST, PORT, max_size=None, ping_interval=20, ping_timeout=20):
        logger.info("Nemotron STT listening on ws://%s:%d%s", HOST, PORT, PATH)
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main())
