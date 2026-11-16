"""
tts_engine_v58.py — text-to-speech that is VERIFIED, not assumed.

V45 printed "[TTS] ready" without proving anything was audible. V58:
  * finds the Piper voice (.onnx + .onnx.json) and loads it;
  * synthesises a test phrase and checks the audio is non-empty and not silent (peak/RMS);
  * checks an output device exists (sounddevice) and resamples if the device rejects the rate;
  * plays through sounddevice and checks playback took ~the audio's duration;
  * handles the piper-tts API differences (1.3+ AudioChunk generator, older synthesize_stream_raw / wav APIs);
  * falls back to Windows SAPI (System.Speech via PowerShell, offline) if Piper fails;
  * never blocks the interview on failure — the caller always prints the text first, and every failure is
    recorded in `stats.failures`;
  * `--test-tts` runs `diagnose()` and prints every check.
Speech is queued on a worker thread so the interviewer's text appears immediately; call wait() before
listening so the microphone never records the interviewer's own voice.
"""
from __future__ import annotations

import io
import os
import queue
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import v58_config as cfg

try:
    import numpy as np
except ImportError:                    # pragma: no cover
    np = None


@dataclass
class TTSStats:
    backend: str = "none"
    voice: str = ""
    device: str = ""
    spoken: int = 0
    synth_ms: List[float] = field(default_factory=list)
    first_audio_ms: List[float] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    verified: bool = False


def _voice_dirs() -> List[Path]:
    h = cfg.HERE
    import config as _root_cfg
    dirs = [_root_cfg.PIPER_VOICE.parent, Path.cwd(), h, h / "voices", h / "voice", h / "models", h / "Models", h.parent, h.parent / "voices", h.parent / "models",
            Path.home() / ".local" / "share" / "piper", Path.home() / "piper"]
    if os.name == "nt":
        dirs += [Path(r"C:\piper"), Path(r"C:\piper\voices")]
    return dirs


def find_voice(hint: str = "") -> Optional[str]:
    for h in (hint, cfg.PIPER_VOICE_HINT):
        if h and Path(h).is_file():
            return str(Path(h))
        if h and Path(h).is_dir():
            c = [p for p in Path(h).glob("*.onnx") if Path(str(p) + ".json").exists()]
            if c:
                return str(sorted(c)[0])
    cands: List[Path] = []
    for d in _voice_dirs():
        try:
            cands += [p for p in d.glob("*.onnx") if Path(str(p) + ".json").exists()]
        except OSError:
            pass
    pref = lambda p: (0 if p.name.startswith("en_") else 1, 0 if any(x in p.name for x in ("lessac", "amy", "ryan", "libritts")) else 1, p.stat().st_size)
    cands.sort(key=pref)
    return str(cands[0]) if cands else None


def find_piper_exe(hint: str = "") -> Optional[str]:
    import shutil
    for h in (hint, os.environ.get("SMARTHIRE_PIPER_EXE", "")):
        if h and Path(h).is_file():
            return h
    w = shutil.which("piper.exe" if os.name == "nt" else "piper")
    if w:
        return w
    for d in _voice_dirs():
        for n in ("piper.exe", "piper"):
            if (d / n).is_file():
                return str(d / n)
    return None


class TTSEngine:
    def __init__(self, voice_hint: str = "", piper_exe_hint: str = "", device: Optional[str] = None, enabled: bool = True,
                 sd_module=None, voice_factory: Optional[Callable[[str], object]] = None):
        self.voice_hint, self.piper_exe_hint, self.enabled = voice_hint, piper_exe_hint, enabled
        self.device = int(device) if isinstance(device, str) and device.isdigit() else device
        self.stats = TTSStats()
        self._sd = sd_module
        self._voice_factory = voice_factory
        self._voice = None
        self._exe: Optional[str] = None
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self.ready = False
        self.checks: List[Tuple[str, bool, str]] = []

    # ── setup ──────────────────────────────────────────────────────
    def _note(self, name: str, ok: bool, detail: str = ""):
        self.checks.append((name, ok, detail))
        return ok

    def _get_sd(self):
        if self._sd is None:
            try:
                import sounddevice as sd
                self._sd = sd
            except Exception as e:
                self._note("sounddevice import", False, str(e))
                return None
        return self._sd

    def initialize(self, verbose=print, verify_synthesis: bool = True) -> bool:
        if not self.enabled:
            self.stats.backend = "disabled"
            return False
        self.checks.clear()
        sd = self._get_sd()
        dev_ok = False
        if sd is not None:
            try:
                devs = sd.query_devices()
                outs = [i for i, d in enumerate(devs) if d.get("max_output_channels", 0) > 0] if hasattr(devs, "__iter__") else []
                dev_ok = self._note("audio output device", bool(outs), f"{len(outs)} output device(s); default={self._default_out(sd)}")
                self.stats.device = str(self._default_out(sd))
            except Exception as e:
                self._note("audio output device", False, str(e))
        model = find_voice(self.voice_hint)
        self._note("piper voice model (.onnx + .onnx.json)", bool(model), model or "not found — pass --voice <path> or set SMARTHIRE_PIPER_MODEL")
        if model and self._load_piper_py(model):
            self.stats.backend, self.stats.voice = "piper", Path(model).name
        else:
            self._exe = find_piper_exe(self.piper_exe_hint) if model else None
            if model and self._exe:
                self._note("piper executable", True, self._exe)
                self._model_path = model
                self.stats.backend, self.stats.voice = "piper_exe", Path(model).name
            elif os.name == "nt" and self._sapi_available():
                self._note("Windows SAPI fallback", True, "System.Speech via PowerShell")
                self.stats.backend = "sapi"
            else:
                self._note("any TTS backend", False, "Piper unavailable and no SAPI fallback; text-only mode")
                self.stats.backend = "none"
                return False
        if verify_synthesis and self.stats.backend in ("piper", "piper_exe"):
            try:
                audio, sr = self._synth("SmartHire speech check.")
                peak = int(np.max(np.abs(audio))) if len(audio) else 0
                rms = float(np.sqrt(np.mean(audio.astype("float64") ** 2))) if len(audio) else 0.0
                ok = len(audio) > sr * 0.3 and peak > 300
                self._note("synthesis produced audible audio", ok, f"{len(audio) / sr:.2f}s @ {sr} Hz, peak={peak}, rms={rms:.0f}")
                self.stats.verified = ok
                if not ok:
                    self.stats.failures.append("synthesis silent/empty")
                    return False
            except Exception as e:
                self._note("synthesis", False, f"{type(e).__name__}: {e}")
                self.stats.failures.append(f"synthesis: {e}")
                return False
        if self.stats.backend in ("piper", "piper_exe") and not dev_ok:
            self._note("playback path", False, "no audio output device — text-only mode")
            return False
        self.ready = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        verbose(f"  TTS: {self.stats.backend} ready" + (f" (voice {self.stats.voice}, verified audible)" if self.stats.verified else ""))
        return True

    def _default_out(self, sd):
        try:
            d = sd.default.device
            return d[1] if isinstance(d, (list, tuple)) else d
        except Exception:
            return "?"

    def _load_piper_py(self, model: str) -> bool:
        try:
            if self._voice_factory:
                self._voice = self._voice_factory(model)
            else:
                from piper import PiperVoice
                self._voice = PiperVoice.load(model, config_path=model + ".json")
            self._note("piper python module + voice load", True, Path(model).name)
            return True
        except Exception as e:
            self._note("piper python module + voice load", False, f"{type(e).__name__}: {e}")
            return False

    @staticmethod
    def _sapi_available() -> bool:
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", "Add-Type -AssemblyName System.Speech; 'ok'"], capture_output=True, text=True, timeout=20)
            return "ok" in r.stdout
        except Exception:
            return False

    # ── synthesis ──────────────────────────────────────────────────
    def _synth(self, text: str):
        """-> (int16 numpy array, sample_rate). Tries the piper-tts 1.3+ API first, then older ones."""
        t0 = time.time()
        if self.stats.backend == "piper_exe" or (self._voice is None and self._exe):
            audio, sr = self._synth_exe(text)
        else:
            audio, sr = self._synth_py(text)
        self.stats.synth_ms.append(round((time.time() - t0) * 1000, 1))
        return audio, sr

    def _synth_py(self, text: str):
        v = self._voice
        errs = []
        try:                                                         # piper-tts >= 1.3: generator of AudioChunk
            chunks = list(v.synthesize(text))
            if chunks and hasattr(chunks[0], "audio_int16_bytes"):
                audio = np.concatenate([np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks])
                return audio, int(chunks[0].sample_rate)
        except TypeError as e:
            errs.append(str(e))
        try:                                                         # older raw stream API
            raw = b"".join(v.synthesize_stream_raw(text))
            return np.frombuffer(raw, dtype=np.int16), int(v.config.sample_rate)
        except Exception as e:
            errs.append(str(e))
        try:                                                         # wav-file APIs
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                (v.synthesize_wav if hasattr(v, "synthesize_wav") else v.synthesize)(text, wf)
            buf.seek(0)
            with wave.open(buf, "rb") as wf:
                return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16), wf.getframerate()
        except Exception as e:
            errs.append(str(e))
        raise RuntimeError("piper synthesis failed with every known API: " + " | ".join(errs))

    def _synth_exe(self, text: str):
        cfgp = Path(self._model_path + ".json")
        import json
        sr = int(json.loads(cfgp.read_text(encoding="utf-8"))["audio"]["sample_rate"])
        r = subprocess.run([self._exe, "--model", self._model_path, "--output_raw"], input=text.encode("utf-8"), capture_output=True, timeout=120)
        if r.returncode != 0 or not r.stdout:
            raise RuntimeError(f"piper.exe failed: {r.stderr.decode('utf-8', 'replace')[-200:]}")
        return np.frombuffer(r.stdout, dtype=np.int16), sr

    # ── playback ───────────────────────────────────────────────────
    def _play(self, audio, sr: int) -> float:
        sd = self._get_sd()
        if sd is None:
            raise RuntimeError("sounddevice unavailable")
        dev = self.device
        try:
            sd.check_output_settings(device=dev, samplerate=sr)
        except Exception:
            info = sd.query_devices(dev, "output") if dev is not None else sd.query_devices(kind="output")
            new_sr = int(info["default_samplerate"])
            x = np.arange(len(audio))
            xi = np.linspace(0, len(audio) - 1, int(len(audio) * new_sr / sr))
            audio, sr = np.interp(xi, x, audio.astype("float64")).astype(np.int16), new_sr
        t0 = time.time()
        sd.play(audio, sr, device=dev)
        sd.wait()
        return time.time() - t0

    def _speak_sapi(self, text: str):
        ps = "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak([Console]::In.ReadToEnd())"
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=text, text=True, timeout=120)

    def _run(self):
        while True:
            text = self._q.get()
            if text is None:
                self._q.task_done()
                return
            try:
                t0 = time.time()
                if self.stats.backend == "sapi":
                    self._speak_sapi(text)
                else:
                    audio, sr = self._synth(text)
                    if len(audio) == 0 or int(np.max(np.abs(audio))) < 100:
                        raise RuntimeError("synthesised audio was silent")
                    self.stats.first_audio_ms.append(round((time.time() - t0) * 1000, 1))
                    dur = len(audio) / sr
                    took = self._play(audio, sr)
                    if took < 0.5 * dur:
                        self.stats.failures.append(f"playback returned early ({took:.1f}s of {dur:.1f}s) — wrong/unavailable output device?")
                self.stats.spoken += 1
            except Exception as e:
                self.stats.failures.append(f"{type(e).__name__}: {e}")
            finally:
                self._q.task_done()

    # ── public ─────────────────────────────────────────────────────
    def speak(self, text: str, block: bool = False):
        if not (self.ready and text and text.strip()):
            return
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

    def diagnose(self, play: bool = True, ask=None, verbose=print) -> bool:
        """Visible end-to-end check for `--test-tts`."""
        verbose("TTS diagnostic")
        ok = self.initialize(verbose=lambda *_: None, verify_synthesis=True)
        try:
            import piper
            self._note("piper package version", True, getattr(piper, "__version__", "installed"))
        except Exception:
            pass
        sd = self._get_sd()
        if sd is not None:
            try:
                for i, d in enumerate(sd.query_devices()):
                    if d.get("max_output_channels", 0) > 0:
                        verbose(f"    output device [{i}] {d['name']}  (default rate {int(d['default_samplerate'])} Hz)")
            except Exception:
                pass
        for name, good, detail in self.checks:
            verbose(f"  [{'OK ' if good else 'FAIL'}] {name}: {detail}")
        if ok and play:
            verbose("  Playing: 'Hello, this is Smarty, your interviewer. If you can hear this, text to speech is working.'")
            n0 = len(self.stats.failures)
            self.speak("Hello, this is Smarty, your interviewer. If you can hear this, text to speech is working.", block=True)
            new = self.stats.failures[n0:]
            verbose("  [OK ] playback finished with no errors" if not new else "  [FAIL] playback problems: " + "; ".join(new))
            ok = ok and not new
            if ask:
                heard = ask("  Did you actually hear it? [y/n]: ").strip().lower().startswith("y")
                verbose("  [OK ] you confirmed audio" if heard else "  [FAIL] you did not hear it — check --audio-device (list above) and system volume")
                ok = ok and heard
        verbose("TTS diagnostic result: " + ("PASS" if ok else "FAIL"))
        return ok
