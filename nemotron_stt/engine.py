"""Nemotron Speech Streaming (NeMo cache-aware FastConformer + RNN-T) for live audio.

Follows NeMo's live-microphone example for cache-aware models
(tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb): the
model is loaded once, and each caller gets a NemotronStream that carries the
encoder caches, the decoder hypothesis and a small cache of previous audio
features between chunks.

Chunk size sets the latency/accuracy trade-off (model card): the right
attention context is 0, 1, 6 or 13 steps of 80 ms, i.e. chunks of 80, 160,
560 or 1120 ms. 160 ms is the default here: fast enough for a phone call.
"""

from __future__ import annotations

import copy
import logging
import threading

import numpy as np

logger = logging.getLogger("nemotron_stt.engine")

ENCODER_STEP_MS = 80
SAMPLE_RATE = 16000
# chunk length (ms) -> right attention context, from the model card
RIGHT_CONTEXT = {80: 0, 160: 1, 560: 6, 1120: 13}
LEFT_CONTEXT = 70


class NemotronModel:
    """The loaded model, shared by every caller."""

    def __init__(self, model_name: str, chunk_ms: int = 160, device: str | None = None):
        import torch
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf, open_dict

        if chunk_ms not in RIGHT_CONTEXT:
            raise ValueError(f"chunk_ms must be one of {sorted(RIGHT_CONTEXT)}, got {chunk_ms}")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading %s on %s (chunks of %d ms)...", model_name, self.device, chunk_ms)

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name, map_location=self.device)
        model.encoder.set_default_att_context_size([LEFT_CONTEXT, RIGHT_CONTEXT[chunk_ms]])

        decoding_cfg = model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.preserve_alignments = False
            if hasattr(model, "joint"):  # RNN-T
                decoding_cfg.greedy.max_symbols = 10
                decoding_cfg.fused_batch_size = -1
        model.change_decoding_strategy(decoding_cfg)
        model.eval()
        self.model = model

        # Feature extraction for small live chunks: no dither or padding, and no
        # per-utterance normalisation (it would change with every chunk).
        cfg = copy.deepcopy(model._cfg)
        OmegaConf.set_struct(cfg.preprocessor, False)
        cfg.preprocessor.dither = 0.0
        cfg.preprocessor.pad_to = 0
        cfg.preprocessor.normalize = "None"
        self.preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(cfg.preprocessor)
        self.preprocessor.to(self.device)

        self.pre_encode_cache_size = model.encoder.streaming_cfg.pre_encode_cache_size[1]
        self.num_channels = model.cfg.preprocessor.features
        # One chunk = the lookahead plus one encoder step (as in NeMo's example).
        self.chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000) - 1
        # The GPU is shared by every call; one step at a time keeps it simple.
        self.lock = threading.Lock()
        logger.info("Model ready: %d samples per chunk", self.chunk_samples)

    def new_stream(self) -> "NemotronStream":
        return NemotronStream(self)


class NemotronStream:
    """One caller's recogniser state."""

    def __init__(self, shared: NemotronModel):
        self.shared = shared
        self.chunk_samples = shared.chunk_samples
        self.reset()

    def reset(self) -> None:
        torch, m = self.shared.torch, self.shared
        with m.lock:
            (self.cache_last_channel, self.cache_last_time, self.cache_last_channel_len) = (
                m.model.encoder.get_initial_cache_state(batch_size=1)
            )
            self.previous_hypotheses = None
            self.pred_out_stream = None
            self.cache_pre_encode = torch.zeros(
                (1, m.num_channels, m.pre_encode_cache_size), device=m.device
            )

    def step(self, audio: np.ndarray) -> str:
        torch, m = self.shared.torch, self.shared
        with m.lock, torch.no_grad():
            signal = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0).to(m.device)
            length = torch.tensor([audio.shape[0]], device=m.device)
            features, feature_len = m.preprocessor(input_signal=signal, length=length)
            features = torch.cat([self.cache_pre_encode, features], dim=-1)
            feature_len = feature_len + self.cache_pre_encode.shape[-1]
            self.cache_pre_encode = features[:, :, -m.pre_encode_cache_size :]

            (
                self.pred_out_stream,
                transcribed,
                self.cache_last_channel,
                self.cache_last_time,
                self.cache_last_channel_len,
                self.previous_hypotheses,
            ) = m.model.conformer_stream_step(
                processed_signal=features,
                processed_signal_length=feature_len,
                cache_last_channel=self.cache_last_channel,
                cache_last_time=self.cache_last_time,
                cache_last_channel_len=self.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=self.previous_hypotheses,
                previous_pred_out=self.pred_out_stream,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        first = transcribed[0] if transcribed else ""
        return str(getattr(first, "text", first) or "")
