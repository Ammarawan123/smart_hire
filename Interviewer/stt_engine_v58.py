"""
stt_engine_v58.py — answer input that never lets silence or noise advance the interview.

Backends: 'typed' (default; also the fallback) and 'whisper' (faster-whisper, CPU int8). The whisper path
records with an energy VAD: it calibrates the noise floor, waits for speech, stops after STT_SILENCE_S of
silence, rejects too-short utterances, transcribes, and rejects empty / filler / typical hallucinated
transcripts ("you", "thank you.") as NOT an answer (returns ""), so the interviewer re-prompts instead of
advancing. After 3 consecutive empty captures it offers typed input.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import v58_config as cfg

try:
    import numpy as np
except ImportError:                    # pragma: no cover
    np = None

_HALLUCINATIONS = {"you", "thank you", "thanks", "thanks for watching", "bye", "okay", "ok", "so", "the", "uh", "um", "hmm", "mm"}


def is_meaningful_transcript(text: str, audio_seconds: float = 0.0) -> bool:
    t = re.sub(r"[^a-z0-9' ]", "", (text or "").lower()).strip()
    if len(t) < cfg.STT_MIN_CHARS:
        return False
    if t in _HALLUCINATIONS and audio_seconds < 2.5:
        return False
    return True


def segment_utterance(frames: Iterable, sr: int, frame_s: float = 0.03, silence_s: float = cfg.STT_SILENCE_S,
                      max_s: float = cfg.STT_MAX_UTTERANCE_S, min_s: float = cfg.STT_MIN_UTTERANCE_S,
                      calib_s: float = 0.4) -> Tuple[Optional["np.ndarray"], str]:
    """Consume audio frames (float32 mono) and return (utterance, reason). reason: 'ok' | 'no_speech' | 'too_short' | 'max_length'."""
    buf: List = []
    floor_samples: List[float] = []
    started, silent_frames, speech_frames = False, 0, 0
    need_sil = int(silence_s / frame_s)
    thr = None
    n = 0
    for fr in frames:
        n += 1
        e = float(np.sqrt(np.mean(np.square(fr)))) if len(fr) else 0.0
        if thr is None:
            floor_samples.append(e)
            if len(floor_samples) * frame_s >= calib_s:
                thr = max(0.012, float(np.median(floor_samples)) * 3.0)
            continue
        if e > thr:
            if not started:
                started = True
            speech_frames += 1
            silent_frames = 0
            buf.append(fr)
        elif started:
            silent_frames += 1
            buf.append(fr)
            if silent_frames >= need_sil:
                break
        if started and len(buf) * frame_s >= max_s:
            return np.concatenate(buf), "max_length"
    if not started:
        return None, "no_speech"
    if speech_frames * frame_s < min_s:
        return None, "too_short"
    return np.concatenate(buf), "ok"


class SpeechInput:
    def __init__(self, backend: str = "typed", model_size: str = "base.en", input_fn: Callable[[str], str] = input, tts=None, device=None):
        self.backend, self.model_size, self.input_fn, self.tts, self.device = backend, model_size, input_fn, tts, device
        self._model = None
        self.empty_streak = 0
        self.stats = {"captures": 0, "empty": 0, "typed_fallbacks": 0, "transcribe_s": []}
        self.status = "typed"
        if backend == "whisper":
            self._init_whisper()

    def _init_whisper(self):
        try:
            from faster_whisper import WhisperModel
            import sounddevice  # noqa: F401
            size = self.model_size
            try:
                import config as _root
                size = _root.local_or({"base.en": _root.WHISPER_BASE_DIR, "distil-large-v3": _root.WHISPER_DISTIL_DIR}.get(size, Path(size)), size)
            except Exception:
                pass
            self._model = WhisperModel(size, device="cpu", compute_type="int8")
            self.status = f"whisper:{self.model_size}"
        except Exception as e:
            self.backend, self.status = "typed", f"typed (whisper unavailable: {type(e).__name__}: {e})"

    def listen(self, prompt: str = "You: ") -> Optional[str]:
        """Return the candidate's answer text; '' means 'nothing usable heard' (caller must not advance); None = end of input."""
        if self.tts:
            self.tts.wait()                     # never record the interviewer's own voice
        self.stats["captures"] += 1
        if self.backend == "whisper" and self._model is not None and self.empty_streak < 3:
            text = self._listen_whisper()
            if text:
                self.empty_streak = 0
                print(f"{prompt}{text}")
                return text
            self.empty_streak += 1
            self.stats["empty"] += 1
            if self.empty_streak < 3:
                return ""
            print("  (I could not hear you clearly — you can type your answer instead.)")
            self.stats["typed_fallbacks"] += 1
        try:
            return self.input_fn(prompt).strip()
        except EOFError:
            return None

    def _listen_whisper(self) -> str:
        import sounddevice as sd
        sr, frame = 16000, int(16000 * 0.03)

        def frames():
            with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=frame, device=self.device) as st:
                while True:
                    data, _ = st.read(frame)
                    yield data[:, 0].copy()
        print("  🎤 listening…", end="\r")
        audio, why = segment_utterance(frames(), sr)
        if audio is None:
            return ""
        t0 = time.time()
        segs, _ = self._model.transcribe(audio, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False)
        segs = list(segs)
        self.stats["transcribe_s"].append(round(time.time() - t0, 2))
        if segs and sum(getattr(s, "no_speech_prob", 0) for s in segs) / len(segs) > 0.6:
            return ""
        text = " ".join(s.text.strip() for s in segs).strip()
        return text if is_meaningful_transcript(text, len(audio) / sr) else ""
