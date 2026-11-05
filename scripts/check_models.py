"""Real-model smoke checks.  python scripts/check_models.py paths|vision|voice|nlp|interviewer|all [options]
These load the REAL models from the paths in config.py and report PASS / FAIL / SKIP with the reason.
They prove 'loads and produces sane output', NOT that a model is accurate. Not run against real models in the
migration session (none were available there), so report any failure message back.
"""
from __future__ import annotations
import argparse, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap  # noqa: E402
bootstrap()
import config as C  # noqa: E402
C.apply_runtime_env(offline=True)
import numpy as np  # noqa: E402

R = {"PASS": 0, "FAIL": 0, "SKIP": 0}


def res(status, name, detail=""):
    R[status] += 1
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))


def run(name, fn):
    try:
        out = fn()
        if out is None or out is True:
            res("PASS", name)
        elif isinstance(out, str) and out.startswith("SKIP"):
            res("SKIP", name, out[5:])
        else:
            res("PASS", name, str(out))
    except Exception as e:
        res("FAIL", name, f"{type(e).__name__}: {e}")
        if "-v" in sys.argv:
            traceback.print_exc()


def has_files(d: Path, names):
    d = Path(d)
    return d.is_dir() and all(any(d.rglob(n)) for n in names)


# ───────────────────────── paths ─────────────────────────
def check_paths(a):
    print("PATHS")
    need = {
        "sbert folder": (C.SBERT_DIR, ["config.json", "modules.json"]),
        "whisper base.en": (C.WHISPER_BASE_DIR, ["model.bin", "config.json"]),
        "whisper distil-large-v3": (C.WHISPER_DISTIL_DIR, ["model.bin", "config.json"]),
        "grammar T5": (C.GRAMMAR_T5_DIR, ["config.json"]),
    }
    import os
    hub = Path(os.environ.get("HF_HUB_CACHE", str(C.HF_CACHE / "hub")))
    for label, local, repo in (("SER audeering", C.SER_AUDEERING_DIR, "audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim"),
                               ("SER ehcalabres", C.SER_EHCALABRES_DIR, "ehcalabres--wav2vec2-lg-xlsr-en-speech-emotion-recognition"),
                               ("SER hubert", C.SER_HUBERT_DIR, "superb--hubert-large-superb-er")):
        def f(local=local, repo=repo):
            if has_files(local, ["config.json"]):
                return f"plain folder {local}"
            snap = hub / f"models--{repo}" / "snapshots"
            if snap.is_dir() and any(snap.rglob("config.json")):
                return f"HF cache layout in {hub}"
            raise FileNotFoundError(f"neither plain folder {local} nor HF cache {hub}\\models--{repo}\\snapshots")
        run(label, f)
    for n, (d, files) in need.items():
        if n == "grammar T5" and not Path(d).exists():
            res("SKIP", n + f"  ({d})", "optional (only NLP/stt_realtime_v5.py post-session grammar pass); set SMARTHIRE_GRAMMAR_DIR if you have it")
            continue
        run(n + f"  ({d})", lambda d=d, files=files: True if has_files(d, files) else (_ for _ in ()).throw(FileNotFoundError(f"missing folder or files {files}")))
    for n, p in {"Qwen GGUF": C.QWEN_GGUF, "llama-server.exe": C.LLAMA_SERVER, "piper.exe": C.PIPER_EXE, "piper voice .onnx": C.PIPER_VOICE,
                 "piper voice .json": Path(str(C.PIPER_VOICE) + ".json"), "NuExtract GGUF": C.NUEXTRACT_GGUF, "face_landmarker.task": C.FACE_LANDMARKER,
                 "gaze onnx": C.GAZE_ONNX, "emotion onnx (resemotenet)": C.EMOTION_ONNX, "yolov8n.onnx": C.PHONE_ONNX}.items():
        run(f"{n}  ({p})", lambda p=p: True if Path(p).is_file() and Path(p).stat().st_size > 0 else (_ for _ in ()).throw(FileNotFoundError("file missing/empty")))


# ───────────────────────── vision ─────────────────────────
def check_vision(a):
    print("VISION  (backend=%s, gaze=%s)" % (a.backend, a.gaze))
    import vision_pipeline_v3 as V
    holder = {}

    def build():
        holder["p"] = V.VisionPipeline(face_backend=a.backend, face_model_path=str(C.FACE_LANDMARKER) if a.backend == "tasks" else None,
                                       gaze_model_path=str(C.GAZE_ONNX) if a.gaze else None)
        return f"VisionPipeline built (emotion model: {V.EMOTION_MODEL_PATH})"
    run("build VisionPipeline (loads emotion ONNX + MediaPipe)", build)
    if "p" not in holder:
        return
    import cv2
    p = holder["p"]

    def synthetic():
        t = []
        for _ in range(20):
            f = (np.random.rand(480, 640, 3) * 255).astype("uint8")
            t0 = time.time(); p.process_frame(f, draw=False); t.append(time.time() - t0)
        return f"20 synthetic frames ok, mean {1000*np.mean(t):.0f} ms/frame (a noise image has no face; this only proves no crash)"
    run("process synthetic frames", synthetic)

    def camera():
        cap = cv2.VideoCapture(a.camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else 0)
        if not cap.isOpened():
            return "SKIP camera not available (use --no-camera to hide this)"
        n_ok = n_face = 0; t = []; end = time.time() + a.seconds; last = None
        while time.time() < end:
            ok, fr = cap.read()
            if not ok:
                continue
            t0 = time.time(); out = p.process_frame(fr, draw=False); t.append(time.time() - t0); n_ok += 1; last = out
        cap.release()
        if n_ok == 0:
            raise RuntimeError("camera opened but no frames")
        feats = p.get_window_features() if hasattr(p, "get_window_features") else {}
        return f"{n_ok} frames in {a.seconds}s, {1000*np.mean(t):.0f} ms/frame (~{1/np.mean(t):.1f} fps processing). Window features keys: {list(feats)[:12] if isinstance(feats, dict) else type(feats).__name__}"
    if not a.no_camera:
        print("    (look at the camera, then look away for a moment, during the next few seconds)")
        run(f"live camera {a.seconds}s", camera)
    p.release() if hasattr(p, "release") else None


# ───────────────────────── voice ─────────────────────────
def _tone_speech(sr=16000, secs=4.0):
    t = np.arange(int(sr * secs)) / sr
    sig = np.zeros_like(t)
    for s, e in ((0.5, 1.5), (2.2, 3.4)):                      # two 'voiced' bursts with harmonics + vibrato
        m = (t >= s) & (t < e); f0 = 140 + 8 * np.sin(2 * np.pi * 5 * t[m])
        ph = 2 * np.pi * np.cumsum(f0) / sr
        sig[m] = sum(np.sin(k * ph) / k for k in range(1, 8)) * 0.3
    return (sig + 0.002 * np.random.randn(len(t))).astype("float32"), sr


def check_voice(a):
    print("VOICE")
    import acoustic_extractor as ae
    audio, sr = _tone_speech()

    def vad():
        ae.load_silero_vad()
        seg = ae.get_speech_segments(audio, sr)
        return f"{len(seg)} speech segment(s) found in a synthetic 2-burst signal (VAD may reject non-speech tones: 0 is not necessarily a bug; confirm with --mic)"
    run("Silero VAD loads + segments", vad)
    run("acoustic features (Praat/librosa)", lambda: f"keys: {list(ae.extract_acoustic_features(audio, sr))[:12]}")

    if a.mic:
        def mic():
            import sounddevice as sd
            print(f"    speak normally for {a.mic}s ...")
            rec = sd.rec(int(a.mic * 16000), samplerate=16000, channels=1, dtype="float32"); sd.wait()
            x = rec[:, 0]; seg = ae.get_speech_segments(x, 16000); feats = ae.extract_acoustic_features(x, 16000)
            return f"peak {np.abs(x).max():.3f}, {len(seg)} speech segments, features: { {k: (round(v, 3) if isinstance(v, (int, float)) else v) for k, v in list(feats.items())[:8]} }"
        run("live microphone", mic)
    if a.ser:
        import models_arena as ma
        def ser():
            arena = ma.ModelArena(); arena.ensure_loaded()
            out = arena.run(audio)
            return f"models loaded: {list(arena._loaded)} skipped: {sorted(arena._invalid)}; output keys: {list(out)[:8]}"
        run("speech-emotion arena (heavy: ~1.2 GB per model; close other apps)", ser)
    else:
        res("SKIP", "speech-emotion models", "add --ser to load them (heavy)")


# ───────────────────────── nlp ─────────────────────────
def check_nlp(a):
    print("NLP")
    from sbert_analyzer import SBERTAnalyzer
    def sb():
        s = SBERTAnalyzer()
        q = "Explain how you would reduce overfitting in a neural network."
        good = "I would add dropout and weight decay, use early stopping and get more training data."
        bad = "My favourite food is biryani and I enjoy cricket on weekends."
        g = s.question_answer_similarity(q, good); b = s.question_answer_similarity(q, bad)
        if not g > b:
            raise AssertionError(f"relevant {g:.2f} not above irrelevant {b:.2f}")
        return f"relevant={g:.2f} > irrelevant={b:.2f}"
    run("SBERT loads and ranks relevant > irrelevant", sb)

    def wh(path, label):
        def f():
            from faster_whisper import WhisperModel
            m = WhisperModel(str(path), device="cpu", compute_type="int8")
            if a.wav:
                segs, info = m.transcribe(a.wav, language="en"); txt = " ".join(s.text for s in segs).strip()
                return f"transcript: {txt[:200]!r}"
            return "loaded (add --wav file.wav to test transcription)"
        run(label, f)
    wh(C.WHISPER_BASE_DIR, "faster-whisper base.en")
    if a.big_whisper:
        wh(C.WHISPER_DISTIL_DIR, "faster-distil-whisper-large-v3 (heavy)")
    if a.grammar:
        def g():
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            t = AutoTokenizer.from_pretrained(str(C.GRAMMAR_T5_DIR), local_files_only=True)
            m = AutoModelForSeq2SeqLM.from_pretrained(str(C.GRAMMAR_T5_DIR), local_files_only=True)
            ids = t("grammar: i has two year experience in python", return_tensors="pt")
            return repr(t.decode(m.generate(**ids, max_length=40)[0], skip_special_tokens=True))
        run("T5 grammar correction", g)


# ───────────────────────── interviewer ─────────────────────────
def check_interviewer(a):
    print("INTERVIEWER")
    def piper():
        from tts_engine_v58 import TTSEngine
        t = TTSEngine(str(C.PIPER_VOICE), str(C.PIPER_EXE)); 
        t.speak("SmartHire voice check. If you hear this, text to speech works."); t.wait()
        return f"backend={getattr(t.stats, 'backend', '?')} (you should have heard the sentence)"
    if a.no_audio_out:
        res("SKIP", "Piper speak", "--no-audio-out")
    else:
        run("Piper speaks", piper)
    def llm():
        import llm_engine_v58 as L, v58_config as cfg
        import inspect
        names = [n for n, o in vars(L).items() if inspect.isclass(o) and "LLM" in n.upper()]
        return f"classes: {names}. Real LLM check = run: python main.py interview --resume R --jd J --stt typed --no-tts   (the runner prints the llama-server startup status)"
    run("LLM engine importable", llm)


def check_cues(a):
    print("CUES  (text/voice/vision cue extraction + fusion on the scripted SYNTHETIC interview)")
    import json, subprocess, tempfile
    def f():
        out = Path(tempfile.mkdtemp()) / "sim.json"
        r = subprocess.run([sys.executable, str(ROOT / "main.py"), "simulate", "--report-json", str(out)], capture_output=True, text=True, cwd=str(ROOT))
        if r.returncode:
            raise RuntimeError(r.stderr[-400:])
        b = json.loads(out.read_text(encoding="utf-8"))["behavior"]
        t0 = b["turns"][0]
        return f"{len(b['turns'])} turns scored; summary={b['summary']}; cue names on turn 1: {list(t0['cues'])}; z: {list(t0['z'])}"
    run("cue extraction + fusion machinery", f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="paths", choices=["paths", "vision", "voice", "nlp", "interviewer", "cues", "all"])
    ap.add_argument("--backend", default="legacy", choices=["legacy", "tasks"]); ap.add_argument("--gaze", action="store_true")
    ap.add_argument("--camera-index", type=int, default=0); ap.add_argument("--seconds", type=int, default=8); ap.add_argument("--no-camera", action="store_true")
    ap.add_argument("--mic", type=int, default=0, help="seconds to record from the mic"); ap.add_argument("--ser", action="store_true")
    ap.add_argument("--wav"); ap.add_argument("--big-whisper", action="store_true"); ap.add_argument("--grammar", action="store_true")
    ap.add_argument("--no-audio-out", action="store_true"); ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    for name, fn in (("paths", check_paths), ("vision", check_vision), ("voice", check_voice), ("nlp", check_nlp), ("interviewer", check_interviewer), ("cues", check_cues)):
        if a.what in (name, "all"):
            fn(a)
    print(f"\nSUMMARY: {R}")
    return 1 if R["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
