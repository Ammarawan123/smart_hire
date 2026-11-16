"""SmartHire orchestrator.  Thin by design: it verifies the environment, runs module tests, and launches the
interviewer. It does not re-implement any module.

  python main.py verify                 environment + model presence check (no model loading, no network)
  python main.py models                 list the model inventory and where each model lives
  python main.py test [-m MODULE]       run unit tests: interviewer | fusion | vision | voice | nlp | integration | all
  python main.py simulate [...]         end-to-end interview with a scripted SYNTHETIC candidate; no mic/camera/models needed
  python main.py check <paths|vision|voice|nlp|interviewer|all> [opts]   load the REAL models and smoke-test them (scripts/check_models.py)
  python main.py interview [...]        real interview (typed or Whisper STT, Piper TTS, optional camera); extra args go to the runner
Run `python main.py <mode> -h` for options.  Student/HR modes are CLI-level only: no web frontend or database exists in this repo.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap  # noqa: E402

bootstrap()
import config as C  # noqa: E402
from shared import model_registry as M  # noqa: E402

FIX = ROOT / "Interviewer" / "tests" / "fixtures"
SAMPLE_RESUME = FIX / "v51" / "fixture_resume_clean.txt"
SAMPLE_JD = FIX / "contour_jd.txt"
RUNNER = ROOT / "Interviewer" / "run_interview_v59.py"
TEST_DIRS = {
    "interviewer": ROOT / "Interviewer" / "tests", "fusion": ROOT / "fusion" / "tests", "vision": ROOT / "vision" / "tests",
    "voice": ROOT / "Voice" / "tests", "nlp": ROOT / "NLP" / "tests", "integration": ROOT / "tests",
}
CORE_PKGS = ["numpy", "scipy"]
OPTIONAL_PKGS = ["cv2", "mediapipe", "onnxruntime", "librosa", "parselmouth", "sounddevice", "soundfile", "faster_whisper",
                 "sentence_transformers", "torch", "transformers", "silero_vad", "fitz", "reportlab"]


def cmd_verify(a) -> int:
    py = sys.version_info
    out = {"python": platform.python_version(), "python_ok": (3, 10) <= py[:2] < (3, 13), "platform": platform.platform(),
           "preferred_python": "3.10.11", "offline_flags_set_at_runtime": True, "packages": {}, "models": []}
    for p in CORE_PKGS + OPTIONAL_PKGS:
        out["packages"][p] = importlib.util.find_spec(p) is not None
    out["models"] = [{k: r[k] for k in ("key", "owner", "required", "present", "location")} for r in M.verify_all()]
    missing_req = [m["key"] for m in out["models"] if m["required"] and not m["present"]]
    out["required_models_missing"] = missing_req
    out["can_run_simulation"] = all(out["packages"][p] for p in CORE_PKGS)
    out["can_run_real_interview"] = out["can_run_simulation"] and not missing_req
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"Python {out['python']} (preferred 3.10.11)  OS: {out['platform']}")
        print("Packages:  " + "  ".join(f"{k}={'ok' if v else 'MISSING'}" for k, v in out["packages"].items()))
        for m in out["models"]:
            print(f"  [{'ok' if m['present'] else '--'}] {m['owner']:<11} {m['key']:<24} {'(required) ' if m['required'] else ''}{m['location']}")
        print(f"\nSimulation possible: {out['can_run_simulation']}   Real interview possible: {out['can_run_real_interview']}")
        if missing_req:
            print("Missing required models: " + ", ".join(missing_req) + "  -> see MODEL_DOWNLOAD_GUIDE.md")
    return 0 if out["can_run_simulation"] else 1


def cmd_models(a) -> int:
    for r in M.verify_all():
        print(f"{r['key']:<24} {r['owner']:<11} {r['size']:>8}  {r['licence'] or '-':<30} present={r['present']}  {r['purpose']}")
    return 0


def _run_tests(name: str) -> int:
    d = TEST_DIRS[name]
    if not d.exists() or not any(d.glob("test_*.py")):
        print(f"[{name}] no tests found in {d}")
        return 0
    print(f"=== {name} tests ({d.relative_to(ROOT)}) ===")
    return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", str(d), "-p", "test_*.py"], cwd=str(ROOT))


def cmd_test(a) -> int:
    names = list(TEST_DIRS) if a.module == "all" else [a.module]
    rc = 0
    for n in names:
        rc |= 1 if _run_tests(n) else 0
    return rc


def _runner(extra: list[str]) -> int:
    return subprocess.call([sys.executable, str(RUNNER), *extra], cwd=str(ROOT))


def cmd_simulate(a) -> int:
    print("SIMULATION: scripted synthetic candidate; deterministic fallback (no LLM); no microphone, camera or models. "
          "Output is a machinery check, not an assessment of any real person.")
    out = a.report_json or str(C.OUTPUTS_DIR / "simulation_report.json")
    return _runner(["--resume", a.resume or str(SAMPLE_RESUME), "--jd", a.jd or str(SAMPLE_JD), "--simulate", "--no-llm",
                    "--no-tts", "--report-json", out])


def cmd_interview(a) -> int:
    if not a.resume or not a.jd:
        print("interview needs --resume and --jd", file=sys.stderr)
        return 2
    extra = ["--resume", a.resume, "--jd", a.jd, "--stt", a.stt]
    if a.no_llm:
        extra.append("--no-llm")
    if a.no_tts:
        extra.append("--no-tts")
    if a.llama_server is None and C.LLAMA_SERVER.exists():
        extra += ["--llama-server", str(C.LLAMA_SERVER)]
    if C.QWEN_GGUF.exists() and not a.no_llm:
        extra += ["--model", str(C.QWEN_GGUF)]
    if C.PIPER_VOICE.exists() and not a.no_tts:
        extra += ["--voice", str(C.PIPER_VOICE)]
    if C.PIPER_EXE.exists() and not a.no_tts:
        extra += ["--piper-exe", str(C.PIPER_EXE)]
    import os
    if a.backend:
        os.environ["SMARTHIRE_FACE_BACKEND"] = a.backend
    if a.gaze:
        os.environ["SMARTHIRE_USE_GAZE"] = "1"
    if a.vision:
        import os
        os.environ["SMARTHIRE_VISION"] = "1"
    if a.full:
        a.vision = a.gaze = a.ser = a.nli = True
        os.environ["SMARTHIRE_FACE_BACKEND"] = a.backend or "tasks"
        os.environ["SMARTHIRE_USE_GAZE"] = "1"
        os.environ["SMARTHIRE_VISION"] = "1"
    if a.ser:
        os.environ["SMARTHIRE_SER"] = "1"
    if a.nli:
        os.environ["SMARTHIRE_NLI"] = "1"
    if a.no_nuextract:
        os.environ["SMARTHIRE_NUEXTRACT"] = "off"
    if a.no_cue_log:
        os.environ["SMARTHIRE_CUELOG"] = "0"
    return _runner(extra + a.rest)


def cmd_check(a) -> int:
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "check_models.py"), *a.rest], cwd=str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("verify"); p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("models"); p.set_defaults(fn=cmd_models)
    p = sub.add_parser("test"); p.add_argument("-m", "--module", default="all", choices=[*TEST_DIRS, "all"]); p.set_defaults(fn=cmd_test)
    p = sub.add_parser("check"); p.add_argument("rest", nargs=argparse.REMAINDER); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("simulate"); p.add_argument("--resume"); p.add_argument("--jd"); p.add_argument("--report-json"); p.set_defaults(fn=cmd_simulate)
    p = sub.add_parser("interview")
    p.add_argument("--resume"); p.add_argument("--jd"); p.add_argument("--stt", choices=["typed", "whisper"], default="typed")
    p.add_argument("--no-llm", action="store_true"); p.add_argument("--no-tts", action="store_true"); p.add_argument("--vision", action="store_true")
    p.add_argument("--llama-server"); p.add_argument("--backend", choices=["legacy", "tasks"]); p.add_argument("--gaze", action="store_true")
    p.add_argument("--ser", action="store_true", help="run the heavy voice models (audeering/ehcalabres/HuBERT) + Silero VAD on every answer")
    p.add_argument("--nli", action="store_true", help="NLI contradiction check between earlier and current statements")
    p.add_argument("--full", action="store_true", help="vision(tasks)+gaze+ser+nli: use every model")
    p.add_argument("--no-nuextract", action="store_true", help="do not start NuExtract for skill/project extraction")
    p.add_argument("--no-cue-log", action="store_true", help="do not write outputs/cues/<run>/*.txt")
    p.set_defaults(fn=cmd_interview)
    a, extra = ap.parse_known_args(argv)      # unknown flags (e.g. --debug, --max-questions) pass through to the interview runner
    if extra:
        if a.mode != "interview":
            ap.error("unrecognized arguments: " + " ".join(extra))
        a.rest = list(getattr(a, "rest", None) or []) + extra
    elif not hasattr(a, "rest"):
        a.rest = []
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
