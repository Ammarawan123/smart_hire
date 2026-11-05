"""
selftest_v59.py — checks that EVERY cue and model of SmartHire works on YOUR machine.  Run it from the V58 folder (V59 files copied in).

  python selftest_v59.py                      # A environment + B model files + C every cue on synthetic data + D unit tests
  python selftest_v59.py --models             # + E real models: SBERT, NLI, Silero, SER folders, Whisper
  python selftest_v59.py --mic                # + F microphone check (you will be asked to read a sentence)
  python selftest_v59.py --camera             # + G camera check (look at screen / look right / nod)
Each line prints PASS / FAIL / WARN / SKIP with a fix hint.  A JSON copy is written to selftest_report_v59.json.
Stages C and D were run by the package author on synthetic data; stages E, F, G were NOT (no models, mic or camera there) — so they are the ones that matter for you.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

import shared.model_registry as M

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.paths import bootstrap  # noqa: E402
bootstrap()
sys.path.insert(0, str(HERE))
M.apply_env()
RESULTS = []


def rec(stage, name, status, detail="", fix=""):
    RESULTS.append({"stage": stage, "name": name, "status": status, "detail": detail, "fix": fix})
    icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "SKIP": "⏭️ "}[status]
    print(f"{icon} [{stage}] {name:44} {detail}" + (f"\n      fix: {fix}" if fix and status in ("FAIL", "WARN") else ""))


def check(stage, name, fn, fix=""):
    try:
        ok, detail = fn()
        rec(stage, name, "PASS" if ok else "FAIL", detail, fix)
    except Exception as e:
        rec(stage, name, "FAIL", f"{type(e).__name__}: {e}", fix)


# ───────────────────────── A: environment ─────────────────────────
PKGS = [("numpy", True, "pip install numpy"), ("scipy", True, "pip install scipy"), ("cv2", True, "pip install opencv-python==4.10.0.84"),
        ("mediapipe", True, "pip install mediapipe==0.10.21"), ("onnxruntime", True, "pip install onnxruntime"), ("sounddevice", True, "pip install sounddevice"),
        ("faster_whisper", True, "pip install faster-whisper"), ("sentence_transformers", True, "pip install sentence-transformers"), ("transformers", True, "pip install transformers"),
        ("torch", True, "pip install torch torchaudio  (CPU build: --index-url https://download.pytorch.org/whl/cpu)"), ("librosa", False, "pip install librosa"),
        ("parselmouth", False, "pip install praat-parselmouth"), ("silero_vad", False, "pip install silero-vad"), ("piper", False, "pip install piper-tts"),
        ("kokoro_onnx", False, "pip install kokoro-onnx"), ("sklearn", False, "pip install scikit-learn"), ("huggingface_hub", True, "pip install huggingface_hub"),
        ("fitz", True, "pip install pymupdf")]


def stage_a():
    print("\n== A. Environment ==")
    v = sys.version_info
    rec("A", "python 3.10-3.12", "PASS" if (3, 10) <= v[:2] <= (3, 12) else "WARN", f"{v.major}.{v.minor}.{v.micro}", "mediapipe 0.10.21 is tested on 3.10-3.12")
    for mod, required, fix in PKGS:
        try:
            m = importlib.import_module(mod)
            rec("A", f"import {mod}", "PASS", str(getattr(m, "__version__", "")))
        except Exception as e:
            rec("A", f"import {mod}", "FAIL" if required else "WARN", f"{type(e).__name__}", fix)
    try:
        import numpy
        major = int(numpy.__version__.split(".")[0])
        rec("A", "numpy < 2 (mediapipe 0.10.21 pin)", "PASS" if major < 2 else "WARN", numpy.__version__, "pip install 'numpy<2'" if major >= 2 else "")
    except Exception:
        pass
    ls = M.MODELS_DIR
    rec("A", "models folder exists", "PASS" if ls.exists() else "WARN", str(ls), f"mkdir {ls}  (or set SMARTHIRE_MODELS)")


# ───────────────────────── B: model files ─────────────────────────
def _hf_cached(repo: str) -> bool:
    p = Path(__import__("os").environ["HF_HUB_CACHE"]) / ("models--" + repo.replace("/", "--"))
    return p.exists() and any(p.rglob("*.safetensors")) or (p.exists() and any(p.rglob("pytorch_model.bin")))


def stage_b():
    print("\n== B. Model files ==")
    for a in M.ASSETS:
        if a.group in ("nuextract",) or a.kind == "manual" and a.key in ("resemotenet",):
            pass
        if a.kind == "hf":
            ok = _hf_cached(a.source) or (a.key == "sbert" and (Path(__import__("os").environ["SENTENCE_TRANSFORMERS_HOME"])).exists())
            rec("B", a.key, "PASS" if ok else "WARN", a.source, f"python download_models_v59.py --group {a.group}")
        elif a.kind == "faster_whisper":
            p = Path(__import__("os").environ["HF_HUB_CACHE"])
            ok = any(p.glob(f"*faster-whisper-{a.source}*")) or any(p.glob(f"*faster-distil-whisper*")) if "distil" in a.source else any(p.glob(f"*faster-whisper-{a.source}*"))
            rec("B", a.key, "PASS" if ok else "WARN", a.source, f"python download_models_v59.py --group core")
        else:
            p = M.path_of(a)
            rec("B", a.key, "PASS" if p.exists() else "WARN", str(p), a.note or f"python download_models_v59.py --group {a.group}")


# ───────────────────────── C: every cue on synthetic input ─────────────────────────
def stage_c():
    print("\n== C. Every cue on synthetic input (known answers) ==")
    import voice_cues_v59 as V, vision_extras_v59 as X, text_cues_v59 as T
    sr = 16000
    rng = np.random.default_rng(0)
    voiced = lambda d, f0: (0.3 * (lambda x: x / np.max(np.abs(x)))(sum(np.sin(2 * np.pi * f0 * k * (np.arange(int(d * sr)) / sr)) / k for k in range(1, 8)))).astype(np.float32)
    sil = lambda d: rng.normal(0, .002, int(d * sr)).astype(np.float32)
    a = np.concatenate([sil(1.6), voiced(2.0, 150), sil(0.5), voiced(2.5, 150), sil(.3)])
    v = V.voice_cues(a, sr, 100.0, 99.6, n_words=14)
    check("C", "voice: response latency from TTS clock (true 2.0 s)", lambda: (abs(v.response_latency_s - 2.0) < 0.15, f"{v.response_latency_s:.2f}s"))
    check("C", "voice: pause count (true 1)", lambda: (v.n_pauses == 1, f"{v.n_pauses}"))
    check("C", "voice: articulation rate (true 14 words / 4.5 s)", lambda: (abs(v.articulation_rate_wps - 14 / 4.5) < 0.4, f"{v.articulation_rate_wps:.2f} w/s"))
    check("C", "voice: pitch in semitones (true +7.02)", lambda: (abs(v.f0_mean_st - 7.02) < 0.5, f"{v.f0_mean_st:.2f}"))
    check("C", "voice: noise is rejected as unusable", lambda: (not V.voice_cues(rng.normal(0, .01, sr * 5).astype(np.float32), sr, 0, None, 5).usable, "unusable"))
    bi = V.voice_cues(voiced(4, 150), sr, 100.0, 101.5, 10)
    check("C", "voice: barge-in detected and not scored", lambda: (bi.barge_in and bi.fusion_cues()["response_latency_s"] is None, "barge-in"))
    same, diff = np.concatenate([voiced(4, 120), voiced(4, 120)]), np.concatenate([voiced(4, 120), voiced(4, 230)])
    ds, dd = V.second_voice_drift(same, sr, V.energy_vad(same, sr)), V.second_voice_drift(diff, sr, V.energy_vad(diff, sr))
    check("C", "voice: second-voice drift larger for another source", lambda: (dd > ds + 0.02, f"same {ds:.3f} vs different {dd:.3f}"))
    fs, Tn = 30, 20
    t = np.arange(0, Tn, 1 / fs)
    def mk(dx, dy=None, yaw=None, pit=None):
        z = np.zeros(len(t))
        return [NS(t=t[i], face_ok=True, gaze_valid=True, gaze_dx=dx[i], gaze_dy=(z if dy is None else dy)[i], yaw_dev=(z if yaw is None else yaw)[i], pitch_dev=(z if pit is None else pit)[i], looking=abs(dx[i]) < 1) for i in range(len(t))]
    ph = (t % 2.5) / 2.5
    saw = np.where(ph < .92, ph / .92, 1 - (ph - .92) / .08) * 1.2 - .6
    rd = X.vision_extras(mk(saw + rng.normal(0, .03, len(t)), rng.normal(0, .05, len(t)))).reading_score
    free = np.clip(np.cumsum(rng.normal(0, .06, len(t))), -.9, .9)
    for g in (60, 200, 350, 500): free[g:g + 20] += .9
    fr = X.vision_extras(mk(free + rng.normal(0, .05, len(t)))).reading_score
    check("C", "vision: reading pattern high, free gaze low", lambda: (rd >= 0.6 and fr < 0.6, f"reading {rd:.2f} vs free {fr:.2f}"))
    fix = np.zeros(len(t)); fix[100:300] = 1.3
    check("C", "vision: off-screen fixation (true 6.6 s)", lambda: (abs(X.vision_extras(mk(fix)).offscreen_fixation_s - 6.6) < 0.4, f"{X.vision_extras(mk(fix)).offscreen_fixation_s:.1f}s"))
    nod = 4 * np.sin(2 * np.pi * 1.2 * t) * (t > 5) * (t < 12)
    check("C", "vision: head nods detected", lambda: (X.vision_extras(mk(np.zeros(len(t)), pit=nod)).head_nod_rate_per_min > 15, f"{X.vision_extras(mk(np.zeros(len(t)), pit=nod)).head_nod_rate_per_min:.0f}/min"))
    check("C", "vision: head shake from |yaw| detected", lambda: (X.vision_extras(mk(np.zeros(len(t)), yaw=np.abs(nod))).head_shake_rate_per_min > 12, "ok"))
    rich = T.text_cues("Um, I built it in PyTorch for two years, we deployed it in 2023 and it crashed because of a leak, so I profiled it and cut latency by 40%.", ["pytorch"])
    poor = T.text_cues("I did some work on it and it was good and I learned many things from the project overall.")
    check("C", "text: detail density rich > 3x poor", lambda: (rich.detail_density > 3 * poor.detail_density, f"{rich.detail_density:.1f} vs {poor.detail_density:.1f}"))
    check("C", "text: filler counted", lambda: (rich.filler_rate_per_100w > 0, f"{rich.filler_rate_per_100w:.1f}/100w"))
    check("C", "text: quantity conflict (5 yr vs 6 mo)", lambda: (T.quantity_conflict("5 years of Python experience", "I used python for six months")[0] == 1.0, "conflict"))
    check("C", "text: no false conflict (2 yr vs 18 mo)", lambda: (T.quantity_conflict("2 years of Python", "about 18 months of python")[0] == 0.0, "no conflict"))
    check("C", "text: language 'C' does not match every sentence", lambda: (not T.has_term("I made a classifier", "c") and T.has_term("I know C, Python", "c"), "ok"))
    import fusion_v59 as F, sim_study_v59 as S, eval_harness_v59 as E
    m = F.FusionModel()
    check("C", "fusion: face/voice alone never flags", lambda: (m.score({k: 3 for k in ("response_latency_s", "f0_mean_st", "ser_arousal", "self_adaptor_rate_per_min", "pause_rate_per_min")}).flag == "none", "no flag"))
    check("C", "fusion: hard content conflict flags", lambda: (m.score({}, {"quantity_conflict": 1.0}).flag == "verify", "verify"))
    rows = S.simulate(50, 10, shift={}, seed=3)
    r = E.evaluate(rows, B=150)
    check("C", "eval: no-effect data gives AUC CI containing 0.5", lambda: (r["auc_ci95"][0] <= 0.5 <= r["auc_ci95"][1], f"{r['auc']:.2f} {tuple(round(x, 2) for x in r['auc_ci95'])}"))


# ───────────────────────── D: unit tests ─────────────────────────
def stage_d():
    print("\n== D. Unit tests ==")
    for modname in ("test_v59", "test_live_v59"):
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            rec("D", modname, "FAIL", f"{type(e).__name__}: {e}")
            continue
        res = unittest.TextTestRunner(stream=open("nul" if sys.platform == "win32" else "/dev/null", "w"), verbosity=0).run(unittest.defaultTestLoader.loadTestsFromModule(mod))
        status = "PASS" if res.wasSuccessful() else "FAIL"
        rec("D", modname, status, f"{res.testsRun} tests, {len(res.failures)} failures, {len(res.errors)} errors, {len(res.skipped)} skipped",
            "run: python -m unittest " + modname + " -v   (skips mean the V58 files are not on the path)")


# ───────────────────────── E: real models ─────────────────────────
def stage_e(whisper_size: str, wav: str):
    print("\n== E. Real models (needs the downloads) ==")
    def sbert():
        from sentence_transformers import SentenceTransformer, util
        m = SentenceTransformer("all-MiniLM-L6-v2")
        e = m.encode(["I trained a CNN for image classification", "I built a convolutional network to classify pictures", "My favourite food is biryani"])
        s1, s2 = float(util.cos_sim(e[0], e[1])), float(util.cos_sim(e[0], e[2]))
        return s1 > s2 + 0.2, f"paraphrase {s1:.2f} vs unrelated {s2:.2f}"
    check("E", "SBERT paraphrase > unrelated", sbert, "python download_models_v59.py --group core")
    def nli():
        import text_cues_v59 as T
        n = T.NLIChecker()
        if not n.available:
            return False, getattr(n, "error", "model not available")
        c = n.contradiction_prob([("I used PyTorch for two years.", "I have never used PyTorch."), ("I used PyTorch for two years.", "I trained models in PyTorch for 24 months."), ("I used PyTorch for two years.", "The weather is nice today.")])
        return c[0] > 0.8 and c[1] < 0.3 and c[2] < 0.3, f"contradiction {c[0]:.2f} | paraphrase {c[1]:.2f} | unrelated {c[2]:.2f}"
    check("E", "NLI: contradiction vs paraphrase vs unrelated", nli, "python download_models_v59.py --group nli ; pip install sentence-transformers")
    def silero():
        from silero_vad import load_silero_vad, get_speech_timestamps
        import torch
        m = load_silero_vad()
        silence = torch.zeros(16000 * 3)
        return len(get_speech_timestamps(silence, m, sampling_rate=16000)) == 0, "silence -> no speech"
    check("E", "Silero VAD loads, silence has no speech", silero, "pip install silero-vad")
    for a in M.by_group(["ser"]):
        def f(a=a):
            from transformers import AutoConfig
            AutoConfig.from_pretrained(a.source, local_files_only=True)
            return True, "config found in cache"
        check("E", f"SER model cached: {a.key}", f, f"python download_models_v59.py --group ser")
    def whisper():
        from faster_whisper import WhisperModel
        m = WhisperModel(whisper_size, device="cpu", compute_type="int8")
        if not wav:
            return True, f"{whisper_size} loaded (pass --wav file.wav to also test transcription)"
        import wave
        w = wave.open(wav); aud = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        t0 = time.time(); segs, _ = m.transcribe(aud, language="en", beam_size=1); text = " ".join(s.text for s in segs).strip()
        return bool(text), f"{time.time() - t0:.1f}s -> “{text[:70]}”"
    check("E", f"faster-whisper {whisper_size}", whisper, "python download_models_v59.py --group core")


# ───────────────────────── F: microphone ─────────────────────────
def stage_f():
    print("\n== F. Microphone (read the sentence when the beep text appears) ==")
    try:
        import sounddevice as sd
    except Exception as e:
        rec("F", "sounddevice", "FAIL", str(e), "pip install sounddevice")
        return
    import voice_cues_v59 as V
    sr = 16000
    print("  Stay SILENT for 4 seconds…")
    sil = sd.rec(int(4 * sr), samplerate=sr, channels=1, dtype="float32"); sd.wait()
    v0 = V.voice_cues(sil[:, 0], sr, 0.0, None, 0)
    rec("F", "silence is not treated as speech", "PASS" if not v0.usable else "FAIL", "; ".join(v0.quality_reasons) or "usable?!", "room too noisy or mic gain too high" if v0.usable else "")
    print("  In 2 seconds, READ ALOUD: 'I built a small project using Python and tested it with my friends last month.'  (recording 10 s)")
    time.sleep(2)
    t0 = time.monotonic()
    a = sd.rec(int(10 * sr), samplerate=sr, channels=1, dtype="float32"); sd.wait()
    v = V.voice_cues(a[:, 0], sr, t0, t0, n_words=16)
    rec("F", "speech detected and usable", "PASS" if v.usable else "FAIL", f"speech {v.speaking_s:.1f}s SNR {v.snr_db and round(v.snr_db)} dB; " + "; ".join(v.quality_reasons), "speak louder / move the mic / reduce noise")
    rec("F", "pitch plausible (70-400 Hz)", "PASS" if v.f0_mean_st is not None and -6 < v.f0_mean_st < 24 else "WARN", f"{v.f0_mean_st}", "")
    rec("F", "articulation rate plausible (1.5-5 words/s)", "PASS" if v.articulation_rate_wps and 1.5 < v.articulation_rate_wps < 5 else "WARN", f"{v.articulation_rate_wps}", "")
    rec("F", "latency ~2 s (you started 2 s after the prompt)", "PASS" if v.response_latency_s is not None and 1.0 < v.response_latency_s < 3.5 else "WARN", f"{v.response_latency_s}", "clock/VAD problem if far from 2 s")


# ───────────────────────── G: camera ─────────────────────────
def stage_g():
    print("\n== G. Camera (3 tasks of 6 s) ==")
    try:
        from vision_pipeline_v3 import VisionPipeline
        from live_session_v59 import CameraThread, VisionAdapter
        import vision_extras_v59 as X
    except Exception as e:
        rec("G", "vision pipeline import", "FAIL", f"{type(e).__name__}: {e}", "copy vision_pipeline_v3.py (+ models/face_landmarker.task) next to these files")
        return
    pipe = VisionPipeline(); th = CameraThread(pipe); th.start(); ad = VisionAdapter(pipe)
    time.sleep(2)
    def task(label, msg, n=6):
        print(f"  {msg} ({n} s)…"); ad.begin(label); time.sleep(n); return ad.end()
    rec_, s = task("look", "LOOK AT THE SCREEN CENTRE, stay natural")
    r = X.vision_extras(s)
    rec("G", "face found (valid frames)", "PASS" if r.usable else "FAIL", f"{r.n_valid} valid frames; {r.notes}", "lighting / camera index / move closer")
    rec("G", "eye contact high when looking at screen", "PASS" if (r.eye_contact_ratio or 0) > 0.6 else "WARN", f"{r.eye_contact_ratio}", "run the Vision calibration first")
    _, s = task("right", "LOOK FAR TO YOUR RIGHT, hold still")
    r = X.vision_extras(s)
    rec("G", "off-screen fixation detected", "PASS" if r.offscreen_fixation_s >= 2.5 else "WARN", f"{r.offscreen_fixation_s:.1f}s", "gaze calibration/tolerance may be too loose")
    _, s = task("nod", "NOD YOUR HEAD about 5 times")
    r = X.vision_extras(s)
    rec("G", "head nods detected", "PASS" if (r.head_nod_rate_per_min or 0) >= 15 else "WARN", f"{r.head_nod_rate_per_min}/min", "nod bigger/slower; amplitude threshold is 2 degrees")
    th.stop_flag = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true"); ap.add_argument("--mic", action="store_true"); ap.add_argument("--camera", action="store_true")
    ap.add_argument("--whisper", default="base.en"); ap.add_argument("--wav", default="")
    a = ap.parse_args()
    stage_a(); stage_b(); stage_c(); stage_d()
    if a.models:
        stage_e(a.whisper, a.wav)
    if a.mic:
        stage_f()
    if a.camera:
        stage_g()
    counts = {k: sum(1 for r in RESULTS if r["status"] == k) for k in ("PASS", "FAIL", "WARN", "SKIP")}
    print(f"\nSUMMARY  PASS {counts['PASS']}  FAIL {counts['FAIL']}  WARN {counts['WARN']}")
    Path("selftest_report_v59.json").write_text(json.dumps({"results": RESULTS, "counts": counts}, indent=1), encoding="utf-8")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
