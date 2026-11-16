"""
live_session_v59.py — capture layer for a real interview: audio with the interviewer's clock, camera segments, signal hand-over.

UNTESTED ON REAL HARDWARE in this package (no microphone, camera, models or sounddevice here).  The timing/ordering logic is exercised with fakes in
test_live_v59.py; run `python run_interview_v59.py --stt whisper --debug` on your machine and read the 'capture check' it prints before trusting any number.

SpeechInputV59   V58 SpeechInput + keeps the answer audio and the time.monotonic() at which that audio started (needed for TTS-end based latency).
VisionAdapter    wraps the Vision v3 pipeline: begin_segment at question start (the FIRST answer is the baseline segment), end_segment at answer end,
                 returns the vision-fusion/1 record and the frame samples for vision_extras_v59.
CameraThread     reads cv2.VideoCapture and feeds pipeline.process_frame(); optional.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

import numpy as np

from interviewer_bridge_v59 import Signals

try:
    from stt_engine_v58 import SpeechInput, segment_utterance, is_meaningful_transcript
except Exception:                                      # pragma: no cover - lets tests import without V58
    SpeechInput = object


class SpeechInputV59(SpeechInput):                     # type: ignore[misc]
    """Adds .last_audio, .last_sr, .last_capture_t0 (monotonic).  Typed answers leave them None, so no voice cue is invented for typed input."""

    def __init__(self, *a, clock: Callable[[], float] = time.monotonic, **k):
        super().__init__(*a, **k)
        self.clock = clock
        self.last_audio: Optional[np.ndarray] = None
        self.last_sr = 16000
        self.last_capture_t0: Optional[float] = None
        self.last_words: Optional[list] = None          # [(start_s, end_s, word)] relative to last_audio (needs word_timestamps)
        self.last_segments: Optional[list] = None       # [(start_s, end_s, corrected text)]
        self.last_corrections: list = []                # audit trail of profile-based transcript corrections
        self.corrector = None                           # NLP/transcript_fix_v60.TranscriptCorrector (from candidate_profile.json)
        self.on_listen_start: Optional[Callable[[], None]] = None      # called right before the microphone opens (vision segment starts here)

    def listen(self, prompt: str = "You: "):
        self.last_audio, self.last_capture_t0 = None, None
        self.last_words, self.last_segments, self.last_corrections = None, None, []
        text = super().listen(prompt) if self.on_listen_start is None else self._listen_with_hook(prompt)
        return text

    def _listen_with_hook(self, prompt: str):
        # V58's listen() first waits for the TTS queue, then opens the microphone; we start the vision segment when the question has finished playing.
        tts = getattr(self, "tts", None)
        if tts is not None and hasattr(tts, "wait"):
            tts.wait()
        try:
            self.on_listen_start()
        except Exception:
            pass
        return super().listen(prompt)

    def _listen_whisper(self) -> str:
        import sounddevice as sd
        sr, frame = 16000, int(16000 * 0.03)
        frame_s = frame / sr
        stamps: List[float] = []                       # monotonic time at which each frame finished arriving

        def frames():
            with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=frame, device=self.device) as st:
                while True:
                    data, _ = st.read(frame)
                    stamps.append(self.clock())
                    yield data[:, 0].copy()
        print("  🎤 listening…", end="\r")
        audio, why = segment_utterance(frames(), sr)
        if audio is None:
            return ""
        n = int(round(len(audio) / frame))
        if n and len(stamps) >= n:                      # the returned audio is the last n frames read
            self.last_capture_t0 = stamps[len(stamps) - n] - frame_s
            self.last_audio, self.last_sr = audio, sr
        t0 = time.time()
        kw = {"language": "en", "beam_size": 1, "vad_filter": True, "condition_on_previous_text": False, "word_timestamps": True}
        cor = self.corrector
        if cor is not None:
            kw["initial_prompt"] = cor.whisper_prompt()          # candidate's own vocabulary biases the decoder
            kw["hotwords"] = cor.hotwords()
        try:
            segs, _ = self._model.transcribe(audio, **kw)
            segs = list(segs)
        except TypeError:                                         # older faster-whisper without hotwords / word_timestamps
            kw.pop("hotwords", None)
            try:
                segs, _ = self._model.transcribe(audio, **kw)
            except TypeError:
                kw.pop("word_timestamps", None)
                segs, _ = self._model.transcribe(audio, **kw)
            segs = list(segs)
        self.stats["transcribe_s"].append(round(time.time() - t0, 2))
        if segs and sum(getattr(s, "no_speech_prob", 0) for s in segs) / len(segs) > 0.6:
            return ""
        fixed_segments, words, changes = [], [], []
        for s in segs:
            txt = s.text.strip()
            if cor is not None:
                txt, ch = cor.correct(txt)
                changes += ch
            if hasattr(s, "start") and hasattr(s, "end"):
                fixed_segments.append((float(s.start), float(s.end), txt))
            for w in (getattr(s, "words", None) or []):
                try:
                    words.append((float(w.start), float(w.end), str(w.word).strip()))
                except Exception:
                    pass
        if fixed_segments:
            text = " ".join(x[2] for x in fixed_segments).strip()
        else:                                                     # no timing information (older faster-whisper / test stubs)
            text = " ".join((cor.correct(s.text.strip())[0] if cor is not None else s.text.strip()) for s in segs).strip()
        if self.last_audio is not None:
            self.last_segments = fixed_segments or None
            self.last_words = words or None
            self.last_corrections = changes
        if changes:
            print("  transcript corrected: " + "; ".join(f"{c['from']} → {c['to']}" for c in changes[:6]))
        return text if is_meaningful_transcript(text, len(audio) / sr) else ""


class VisionAdapter:
    def __init__(self, pipeline):
        self.p = pipeline
        self._n = 0
        self.errors: List[str] = []

    def begin(self, label: str):
        try:
            self.p.begin_segment(label, kind="baseline" if self._n == 0 else "answer", phase="answer")
            self._n += 1
        except Exception as e:
            self.errors.append(f"begin: {e}")

    @staticmethod
    def clock_offset() -> float:
        """time.monotonic() = FrameSample.t + offset (Vision v3 stamps frames with a wall-clock driven by time.monotonic)."""
        try:
            import vision_pipeline_v3 as V
            return float(V._T0_MONO - V._T0_WALL)
        except Exception:
            return 0.0

    def end(self):
        """-> (fusion record | None, [FrameSample])"""
        try:
            seg = getattr(self.p.behavior, "_seg", None)
            samples = list(seg["samples"]) if seg else []
            feats = self.p.end_segment()
            return (feats.to_fusion_record() if feats is not None else None), samples
        except Exception as e:
            self.errors.append(f"end: {e}")
            return None, []


class CameraThread(threading.Thread):
    def __init__(self, pipeline, index: int = 0):
        super().__init__(daemon=True)
        self.p, self.index, self.stop_flag, self.error = pipeline, index, False, None

    def run(self):
        try:
            import cv2
            cap = cv2.VideoCapture(self.index)
            while not self.stop_flag:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                self.p.process_frame(frame, draw=False)
            cap.release()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"


class SignalCollector:
    """Glue: `collect()` is called when an answer has arrived.  tts_end is read LAZILY (the end of the question that this answer responds to);
    reading it earlier would give the end of the PREVIOUS question.  The vision segment starts when listening starts (SpeechInputV59.on_listen_start)."""

    def __init__(self, stt: Optional[SpeechInputV59] = None, tts=None, vision: Optional[VisionAdapter] = None, ser_fn: Optional[Callable] = None, vad_fn: Optional[Callable] = None):
        self.stt, self.tts, self.vision, self.ser_fn, self.vad_fn = stt, tts, vision, ser_fn, vad_fn
        self._n = 0
        self.errors: List[str] = []
        if stt is not None:
            stt.on_listen_start = self.listening_started

    def attach_stt(self, stt):
        self.stt = stt
        stt.on_listen_start = self.listening_started

    def listening_started(self):
        self._n += 1
        if self.vision:
            self.vision.begin(f"q{self._n}")

    def collect(self) -> Signals:
        s = Signals()
        if self.stt is not None and getattr(self.stt, "last_audio", None) is not None:
            s.audio, s.sr, s.capture_t0 = self.stt.last_audio, self.stt.last_sr, self.stt.last_capture_t0
            s.tts_end = getattr(self.tts, "last_end", None) if self.tts is not None else None
            s.words, s.segments = getattr(self.stt, "last_words", None), getattr(self.stt, "last_segments", None)
            s.corrections = list(getattr(self.stt, "last_corrections", []) or []) or None
            if self.vad_fn is not None:
                try:
                    s.speech_segments = self.vad_fn(s.audio, s.sr)          # Silero VAD (Voice/acoustic_extractor), seconds relative to the audio start
                except Exception as e:
                    self.errors.append(f"vad: {type(e).__name__}: {e}")
            if self.ser_fn is not None:
                try:
                    r = self.ser_fn(s.audio, s.sr)
                    s.ser_chunks = r.pop("_chunks", None) if isinstance(r, dict) else None
                    s.ser_results = r
                except Exception as e:
                    s.ser_results = None
                    self.errors.append(f"ser: {type(e).__name__}: {e}")
        if self.vision:
            s.vision_record, s.vision_samples = self.vision.end()
            s.vision_offset = self.vision.clock_offset()
        return s
