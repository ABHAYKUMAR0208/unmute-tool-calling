"""Silero VAD: is the caller speaking in this 32 ms window?

One instance per caller, because the model keeps state between windows.
"""

from __future__ import annotations

import numpy as np


class SileroVad:
    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        self.torch = torch
        self.model = load_silero_vad()

    def speech_prob(self, window: np.ndarray) -> float:
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
            return float(self.model(tensor, 16000).item())
