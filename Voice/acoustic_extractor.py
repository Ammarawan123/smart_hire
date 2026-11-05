"""
acoustic_extractor.py — smartHIRE Ai
=========================================
All non-deep-learning signal processing:
  - Silero VAD (pretrained, snakers4/silero-vad) for speech/silence gating
    and hesitation/pause/response-latency tracking
  - Parselmouth (Praat) for Jitter, Shimmer, HNR — the correct, standard
    open-source tool for these acoustic-phonetic measures (no legitimate
    pretrained neural model outputs them directly)
  - librosa for Pitch (F0) and its variability

Nothing in this file is trained — everything is either a pretrained
checkpoint (Silero VAD) or a deterministic DSP algorithm (Praat/librosa).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import librosa
import numpy as np
import parselmouth
import torch

import voice_config as config

_vad_model = None
_vad_utils = None


def load_silero_vad():
    """Loads Silero VAD once (pretrained, ~1MB, CPU real-time)."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        _vad_model.eval()
    return _vad_model, _vad_utils


def get_speech_segments(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> list[dict]:
    """Returns Silero VAD speech timestamps (in samples) for this window."""
    model, utils = load_silero_vad()
    get_speech_timestamps = utils[0]
    audio_t = torch.from_numpy(audio.astype(np.float32))
    with torch.inference_mode():
        segments = get_speech_timestamps(
            audio_t, model, sampling_rate=sr, threshold=config.VAD_THRESHOLD
        )
    return segments


# --------------------------------------------------------------------------
# Acoustic bio-markers: pitch, jitter, shimmer, HNR
# --------------------------------------------------------------------------

def extract_acoustic_features(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> dict:
    feats = {}

    # Pitch (F0) via librosa pYIN
    # Speech ke liye 70-400 Hz kaafi hai; chhota range pyin ko kai guna tez karta hai.
    f0, _, _ = librosa.pyin(audio, fmin=70, fmax=400, sr=sr, frame_length=1024)
    f0_voiced = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    feats["pitch_mean_hz"] = float(np.mean(f0_voiced)) if len(f0_voiced) else 0.0
    feats["pitch_std_hz"] = float(np.std(f0_voiced)) if len(f0_voiced) else 0.0

    # Jitter / Shimmer / HNR via Praat
    try:
        snd = parselmouth.Sound(audio, sampling_frequency=sr)
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter_local = parselmouth.praat.call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        shimmer_local = parselmouth.praat.call(
            [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

        feats["jitter_local_pct"] = float(jitter_local) * 100 if jitter_local == jitter_local else 0.0
        feats["shimmer_local_pct"] = float(shimmer_local) * 100 if shimmer_local == shimmer_local else 0.0
        feats["hnr_db"] = float(hnr) if hnr == hnr else 0.0
    except Exception:
        feats["jitter_local_pct"] = 0.0
        feats["shimmer_local_pct"] = 0.0
        feats["hnr_db"] = 0.0

    return feats


# --------------------------------------------------------------------------
# Speech-flow / hesitation tracking (Type A raw signal source)
# --------------------------------------------------------------------------

@dataclass
class SpeechTurnTracker:
    """
    Tracks pauses, hesitation, and response latency across the live stream
    using wall-clock time, so pause/latency numbers stay meaningful even
    though analysis windows overlap.
    """
    stream_start_time: float = field(default_factory=time.time)
    last_speech_end_time: float | None = None
    last_response_latency_sec: float = 0.0

    def update(self, window_start_time: float, speech_segments: list[dict], sr: int, window_dur: float) -> dict:
        if not speech_segments:
            return {
                "speech_ratio": 0.0,
                "num_pauses": 0,
                "mean_pause_sec": 0.0,
                "longest_pause_sec": 0.0,
                "response_latency_sec": self.last_response_latency_sec,
                "speech_rate_proxy": 0.0,
            }

        speech_time = sum((s["end"] - s["start"]) / sr for s in speech_segments)
        speech_ratio = speech_time / window_dur if window_dur > 0 else 0.0

        pauses = []
        for a, b in zip(speech_segments, speech_segments[1:]):
            gap = (b["start"] - a["end"]) / sr
            if gap > config.MIN_PAUSE_SEC:
                pauses.append(gap)

        # response latency: gap between the last time speech ended (globally)
        # and the first speech onset in THIS window — i.e. how long the
        # candidate was silent before starting to speak again.
        first_onset_abs = window_start_time + speech_segments[0]["start"] / sr
        if self.last_speech_end_time is not None:
            gap_since_last_speech = first_onset_abs - self.last_speech_end_time
            if 0 <= gap_since_last_speech <= config.LONG_SILENCE_RESET_SEC:
                self.last_response_latency_sec = gap_since_last_speech

        self.last_speech_end_time = window_start_time + speech_segments[-1]["end"] / sr

        return {
            "speech_ratio": float(speech_ratio),
            "num_pauses": len(pauses),
            "mean_pause_sec": float(np.mean(pauses)) if pauses else 0.0,
            "longest_pause_sec": float(np.max(pauses)) if pauses else 0.0,
            "response_latency_sec": float(self.last_response_latency_sec),
            "speech_rate_proxy": float(speech_ratio / window_dur) if window_dur > 0 else 0.0,
        }
