"""Stream a WAV file through the running STT server, in real time, and print
what it hears and where it thinks the turn ends. Works against Kyutai's STT
too, so the two can be compared on the same recording.

    nemotron_stt/.venv/bin/python -m nemotron_stt.check_wav my_call.wav
    nemotron_stt/.venv/bin/python -m nemotron_stt.check_wav my_call.wav ws://localhost:8090
"""

import asyncio
import sys
import time
import wave
from fractions import Fraction

import msgpack
import numpy as np
import websockets
from scipy.signal import resample_poly

FRAME = 1920  # 80 ms at 24 kHz, as the backend sends it


def load_24k(path: str) -> np.ndarray:
    with wave.open(path) as wav:
        if wav.getsampwidth() != 2:
            sys.exit("Please use a 16-bit WAV file.")
        rate, channels = wav.getframerate(), wav.getnchannels()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    ratio = Fraction(24000, rate).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)


async def main(path: str, url: str) -> None:
    audio = load_24k(path)
    audio = np.concatenate([audio, np.zeros(24000 * 2, dtype=np.float32)])  # let the last turn end
    async with websockets.connect(url + "/api/asr-streaming",
                                  additional_headers={"kyutai-api-key": "public_token"}) as ws:
        print("server:", msgpack.unpackb(await ws.recv())["type"])
        start = time.monotonic()
        paused = True

        async def reader():
            nonlocal paused
            line = []
            async for raw in ws:
                msg = msgpack.unpackb(raw)
                if msg["type"] == "Word":
                    line.append(msg["text"])
                    print(f"{time.monotonic() - start:6.2f}s  word: {msg['text']}")
                elif msg["type"] == "Step":
                    is_pause = msg["prs"][2] > 0.6
                    if is_pause and not paused and line:
                        print(f"{time.monotonic() - start:6.2f}s  --- turn ended: {' '.join(line)}")
                        line = []
                    paused = is_pause

        task = asyncio.create_task(reader())
        for i in range(0, audio.size, FRAME):
            await ws.send(msgpack.packb({"type": "Audio", "pcm": audio[i:i + FRAME].tolist()},
                                        use_single_float=True))
            await asyncio.sleep(max(0.0, start + (i + FRAME) / 24000 - time.monotonic()))
        await asyncio.sleep(1.5)
        task.cancel()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8090"))
