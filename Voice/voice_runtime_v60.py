"""
voice_runtime_v60.py — plugs the team's heavy voice models into the live interview (they were only used by Voice/live_mic_runner.py before).

  SerRunner      Voice/models_arena.ModelArena (audeering wav2vec2 arousal/dominance/valence, ehcalabres categorical, superb HuBERT) run on up to
                 `max_chunks` chunks of ONE answer, spread evenly over it.  Returns the answer-level mean in the same shape ModelArena.run() uses
                 (so voice_cues_v59.ser_to_cues() works unchanged) plus '_chunks' = [(start_s, end_s, per-chunk result)], which the assistance-pattern
                 detector aligns with gaze and text.  Cost: roughly one model pass per chunk and model; limit it with SMARTHIRE_SER_MAX_CHUNKS (default 4).
  silero_vad_fn  Silero VAD (the same VAD live_mic_runner uses) -> [(start_s, end_s)] for voice_cues_v59 (pauses, latency, articulation rate).
Both are optional and fail soft: if torch/transformers/models are missing the interview continues with the energy-VAD voice cues and says so.
SER arousal/valence describe how a voice SOUNDS (nervous, flat, calm); they are not a lie detector.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _mean_numeric(rows: Sequence[dict]) -> dict:
    out: dict = {}
    for k in {k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}:
        vals = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals and k not in ("latency_ms",):
            out[k] = float(np.mean(vals))
    return out


class SerRunner:
    def __init__(self, arena=None, chunk_s: float = 6.0, max_chunks: Optional[int] = None, min_rms: float = 0.004):
        self.arena = arena
        self.chunk_s = chunk_s
        self.max_chunks = max_chunks if max_chunks is not None else int(os.environ.get("SMARTHIRE_SER_MAX_CHUNKS", "4"))
        self.min_rms = min_rms
        self.errors: List[str] = []
        self.ready = arena is not None

    def load(self) -> bool:
        """Load the arena (heavy: ~1.2 GB per model on CPU).  Returns False (and records why) if it cannot be loaded."""
        if self.arena is not None:
            return True
        try:
            import voice_config as vc
            for k in list(vc.MODEL_RUN_EVERY):
                vc.MODEL_RUN_EVERY[k] = 1                     # every chunk must be scored; the live_mic_runner 'reuse previous result' cache would repeat chunks
            from models_arena import ModelArena
            self.arena = ModelArena()
            self.arena.ensure_loaded()
            self.ready = True
        except Exception as e:
            self.errors.append(f"{type(e).__name__}: {e}")
            self.arena, self.ready = None, False
        return self.ready

    def chunks(self, n_samples: int, sr: int) -> List[Tuple[int, int]]:
        c = int(self.chunk_s * sr)
        spans = [(i, min(i + c, n_samples)) for i in range(0, n_samples, c) if min(i + c, n_samples) - i >= 2 * sr]
        if len(spans) > self.max_chunks > 0:
            idx = np.linspace(0, len(spans) - 1, self.max_chunks).round().astype(int)
            spans = [spans[i] for i in sorted(set(idx.tolist()))]
        return spans

    def __call__(self, audio: np.ndarray, sr: int) -> Optional[dict]:
        if not self.ready:
            return None
        audio = np.asarray(audio, np.float32)
        if sr != 16000:
            raise ValueError("SER models expect 16 kHz audio")
        per: List[Tuple[float, float, dict]] = []
        for a, b in self.chunks(len(audio), sr):
            seg = audio[a:b]
            if float(np.sqrt(np.mean(seg ** 2) + 1e-12)) < self.min_rms:
                continue
            try:
                per.append((a / sr, b / sr, self.arena.run(seg)))
            except Exception as e:
                self.errors.append(f"{type(e).__name__}: {e}")
        if not per:
            return None
        merged: Dict[str, dict] = {}
        for key in {k for _, _, r in per for k in r}:
            merged[key] = _mean_numeric([r[key] for _, _, r in per if key in r])
        merged["_chunks"] = per
        return merged


def silero_vad_fn() -> Optional[Callable]:
    """Callable(audio, sr) -> [(start_s, end_s)] using Silero VAD, or None if torch / the cached Silero repo is unavailable."""
    try:
        import acoustic_extractor as ae
        ae.load_silero_vad()
    except Exception:
        return None

    def vad(audio: np.ndarray, sr: int):
        return [(s["start"] / sr, s["end"] / sr) for s in ae.get_speech_segments(np.asarray(audio, np.float32), sr)]
    return vad
