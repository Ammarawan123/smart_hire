"""
tts_timed_v59.py — gives the interviewer an exact 'question finished' timestamp, and adds Kokoro as a drop-in alternative to Piper.

TimedTTS(inner)   wraps ANY engine that has speak()/wait()/ready (V58 TTSEngine or KokoroTTS below). After `say()` returns, `last_end`
                  is the time.monotonic() at which playback finished (accuracy: tens of ms, the queue-join latency).  This timestamp is what
                  the Voice module uses for response latency.  With no audio engine (text-only run) `last_end` is None, so latency is never
                  invented from reading time.
KokoroTTS         Kokoro-82M through kokoro-onnx (model Apache-2.0, kokoro-onnx MIT; ~300 MB, ~80 MB quantised; CPU near real-time).
                  API assumed from the kokoro-onnx README: Kokoro(model_path, voices_path).create(text, voice=, speed=, lang=) -> (samples, sr).
                  NOT run here (no model files); the queue/timing logic is unit-tested with fakes.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional


class TimedTTS:
    def __init__(self, inner=None, clock: Callable[[], float] = time.monotonic):
        self.inner, self.clock = inner, clock
        self.last_start: Optional[float] = None
        self.last_end: Optional[float] = None
        self.log: List[dict] = []

    @property
    def ready(self) -> bool:
        return bool(self.inner is not None and getattr(self.inner, "ready", False))

    def say(self, text: str) -> Optional[float]:
        """Speak `text` and block until playback ends. Returns the end timestamp (None if there is no audio engine)."""
        if not self.ready:
            self.last_start, self.last_end = self.clock(), None
            return None
        self.last_start = self.clock()
        self.inner.speak(text, block=False)
        self.inner.wait()
        self.last_end = self.clock()
        self.log.append({"chars": len(text), "start": self.last_start, "end": self.last_end})
        return self.last_end

    # pass-through so the V58 runner can use it unchanged
    def speak(self, text: str, block: bool = False):
        if block:
            self.say(text)
        elif self.ready:
            self.last_start = self.clock()
            self.inner.speak(text, block=False)

    def wait(self):
        if self.ready:
            self.inner.wait()
            self.last_end = self.clock()

    def close(self):
        if self.inner is not None:
            self.inner.close()

    def __getattr__(self, name):                       # stats, initialize, diagnose ... of the wrapped engine
        return getattr(self.inner, name)


@dataclass
class KokoroStats:
    backend: str = "none"
    voice: str = ""
    device: str = ""
    spoken: int = 0
    synth_ms: List[float] = field(default_factory=list)
    first_audio_ms: List[float] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    verified: bool = False


class KokoroTTS:
    def __init__(self, model_path: str = "kokoro-v1.0.onnx", voices_path: str = "voices-v1.0.bin", voice: str = "af_sarah", speed: float = 1.0,
                 lang: str = "en-us", device=None, enabled: bool = True, kokoro_factory: Optional[Callable] = None, sd_module=None):
        self.model_path, self.voices_path, self.voice, self.speed, self.lang = model_path, voices_path, voice, speed, lang
        self.device, self.enabled = device, enabled
        self._factory, self._sd, self._k = kokoro_factory, sd_module, None
        self.stats = KokoroStats(backend="kokoro", voice=voice)
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self.ready = False
        self.checks: List[tuple] = []

    def initialize(self, verbose=print, **_) -> bool:
        if not self.enabled:
            self.stats.backend = "disabled"
            return False
        try:
            if self._sd is None:
                import sounddevice as sd
                self._sd = sd
            if self._factory is not None:
                self._k = self._factory()
            else:
                from kokoro_onnx import Kokoro
                self._k = Kokoro(self.model_path, self.voices_path)
            self.checks.append(("kokoro load", True, self.model_path))
        except Exception as e:
            self.checks.append(("kokoro load", False, f"{type(e).__name__}: {e}"))
            self.stats.failures.append(str(e))
            return False
        self.ready = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        return True

    def _run(self):
        while True:
            text = self._q.get()
            try:
                if text is None:
                    return
                t0 = time.time()
                samples, sr = self._k.create(text, voice=self.voice, speed=self.speed, lang=self.lang)
                self.stats.synth_ms.append((time.time() - t0) * 1000)
                self._sd.play(samples, sr, device=self.device)
                self._sd.wait()
                self.stats.spoken += 1
            except Exception as e:
                self.stats.failures.append(f"{type(e).__name__}: {e}")
            finally:
                self._q.task_done()

    def speak(self, text: str, block: bool = False):
        if self.ready and text and text.strip():
            self._q.put(text.strip())
            if block:
                self.wait()

    def wait(self):
        if self.ready:
            self._q.join()

    def close(self):
        if self.ready:
            self.wait()
            self._q.put(None)
        self.ready = False
