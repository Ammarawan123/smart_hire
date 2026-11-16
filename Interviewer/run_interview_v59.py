"""
run_interview_v59.py — V58 interviewer wired to the behaviour layer.
Adds SBERT question-answer and Qwen2.5 reference-answer similarity.
"""

import os
import sys
import json

import sys as _s
from pathlib import Path as _P

_s.path.insert(0, str(_P(__file__).resolve().parents[1]))

from shared.paths import bootstrap
bootstrap()

import config as _cfg
_cfg.apply_runtime_env(offline=True)

import run_interview_v58_interactive as R
import interviewer_bridge_v59 as B
from live_session_v59 import CameraThread, SignalCollector, SpeechInputV59, VisionAdapter
from tts_timed_v59 import KokoroTTS, TimedTTS

COLLECTOR = SignalCollector()
_vision = {"thread": None}

_V60 = {
    "bundle": None,
    "corrector": None,
    "relevance": None,
    "monitor": None,
    "cuelog": None,
    "nli": None,
    "ser": None
}

# SBERT model is loaded only once
_SBERT_ANALYZER = None


def _make_tts(*args, **kwargs):
    if os.environ.get("SMARTHIRE_TTS", "").lower() == "kokoro":
        inner = KokoroTTS(
            os.environ.get("KOKORO_MODEL", "kokoro-v1.0.onnx"),
            os.environ.get("KOKORO_VOICES", "voices-v1.0.bin"),
            os.environ.get("KOKORO_VOICE", "af_sarah"),
            device=args[2] if len(args) > 2 else None,
            enabled=kwargs.get("enabled", True)
        )
    else:
        inner = _orig_tts(*args, **kwargs)

    t = TimedTTS(inner)
    COLLECTOR.tts = t
    return t


def _make_stt(*args, **kwargs):
    s = SpeechInputV59(*args, **kwargs)
    s.corrector = _V60["corrector"]
    COLLECTOR.attach_stt(s)
    return s


class LiveEngine(B.BehaviorAwareEngine):

    def __init__(self, resume, job, llm=None, *a, **k):
        nux = None

        if os.environ.get("SMARTHIRE_NUEXTRACT_URL"):
            from claim_extractor_v59 import NuExtractClient
            nux = NuExtractClient(
                os.environ["SMARTHIRE_NUEXTRACT_URL"],
                os.environ.get("SMARTHIRE_NUEXTRACT_STYLE", "1.5")
            )

        super().__init__(
            resume,
            job,
            llm,
            *a,
            nuextract=nux,
            use_llm_claims=os.environ.get("SMARTHIRE_CLAIMS_LLM") == "1",
            nli=_V60["nli"],
            relevance=_V60["relevance"],
            monitor=_V60["monitor"],
            cuelog=_V60["cuelog"],
            **k
        )

        if (
            os.environ.get("SMARTHIRE_VISION") == "1"
            and COLLECTOR.vision is None
        ):
            try:
                from vision_pipeline_v3 import VisionPipeline
                import config as _c

                _gaze = (
                    str(_c.GAZE_ONNX)
                    if os.environ.get("SMARTHIRE_USE_GAZE") == "1"
                    else None
                )

                pipe = VisionPipeline(
                    face_backend=os.environ.get(
                        "SMARTHIRE_FACE_BACKEND", "legacy"
                    ),
                    face_model_path=str(_c.FACE_LANDMARKER),
                    gaze_model_path=_gaze
                )

                COLLECTOR.vision = VisionAdapter(pipe)

                th = CameraThread(pipe)
                th.start()
                _vision["thread"] = th

                print("  Vision: camera thread started")

            except Exception as e:
                print(
                    f"  Vision: unavailable "
                    f"({type(e).__name__}: {e}) — continuing without it"
                )

    def step(self, answer, signals=None):
        sig = signals if signals is not None else COLLECTOR.collect()
        return super().step(answer, sig)


def _patched_say_hook():
    """V58 TTS/STT timing hook."""


_orig_tts = R.TTSEngine
_orig_build = R.build_report
_orig_render = R.render_text


def _build(engine, *a, **k):
    global _SBERT_ANALYZER

    rep = _orig_build(engine, *a, **k)

    if isinstance(engine, B.BehaviorAwareEngine):
        rep["behavior"] = B.behavior_report(engine)

        used = [
            m for m in ("voice_usable", "vision_usable")
            if any(t.quality.get(m) for t in engine.behavior.turns)
        ]

        rep["behavioural_note"] = (
            "Behaviour layer ran with: text"
            + "".join(", " + m.split("_")[0] for m in used)
            + ". Cues are advisory and person-relative; see the "
              "'behavior' section. Never an automated verdict."
        )

    b = _V60["bundle"]

    if b is not None:
        rep["candidate_profile"] = {
            "file": str(_V60.get("profile_path", "")),
            "extraction_log": b.profile.get("extraction_log"),
            "vocabulary_terms": len(b.profile["vocabulary"]["terms"])
        }

    if _V60["corrector"] is not None:
        rep["transcript_corrections"] = [
            c
            for t in getattr(
                getattr(engine, "behavior", None), "turns", []
            )
            for c in (getattr(t, "corrections", None) or [])
        ]

    if _V60["cuelog"] is not None:
        rep["cue_log"] = _V60["cuelog"].summary()

    # SBERT + Qwen2.5 reference-answer analysis
    try:
        from NLP.sbert_analyzer import SBERTAnalyzer
        from NLP.reference_answer_generator import generate_all, build_profile_context, is_closing_question

        if _SBERT_ANALYZER is None:
            _SBERT_ANALYZER = SBERTAnalyzer()

        ref_path = (
            _P(__file__).resolve().parents[1]
            / "NLP"
            / "reference_answers.json"
        )

        with open(ref_path, "r", encoding="utf-8") as f:
            reference_answers = json.load(f)

        # Collect unique interview questions
        questions = list(dict.fromkeys(
            str(t.question).strip()
            for t in engine.transcript
            if getattr(t, "question", None)
            and str(t.question).strip()
        ))

        # Generate reference answers using the existing Qwen2.5 instance
        llm = getattr(engine, "llm", None)

        bundle = _V60.get("bundle")
        profile_context = build_profile_context(
            getattr(bundle, "profile", {}) if bundle is not None else {}
        )

        generated = generate_all(
            llm,
            questions,
            profile_context
        )

        # Generated Qwen answers override static fallback answers
        reference_answers.update(generated)

        sbert_results = []

        for t in engine.transcript:
            q = str(t.question or "").strip()
            ans = str(t.answer or "").strip()

            if not q or not ans:
                continue

            if is_closing_question(q):
                sbert_results.append({
                    "turn": t.turn,
                    "question": q,
                    "answer": ans,
                    "analysis_type": "closing_question",
                    "scoring_excluded": True
                })
                continue

            ideal = reference_answers.get(q)

            result = {
                "turn": t.turn,
                "question": q,
                "answer": ans,
                "question_answer_similarity":
                    _SBERT_ANALYZER.question_answer_similarity(q, ans)
            }

            if ideal:
                result["reference_answer"] = ideal

                result["reference_answer_source"] = (
                    "qwen2.5_generated"
                    if q in generated
                    else "static_fallback"
                )

                result["ideal_answer_similarity"] = (
                    _SBERT_ANALYZER.answer_ideal_similarity(
                        ans, ideal
                    )
                )

            sbert_results.append(result)

        rep["sbert_analysis"] = sbert_results
        rep["qwen_reference_answers_generated"] = len(generated)

    except Exception as e:
        rep["sbert_error"] = f"{type(e).__name__}: {e}"

    return rep


def _render(rep):
    txt = _orig_render(rep)

    b = rep.get("behavior")

    if b:
        s = b["summary"]

        txt += (
            f"\n\nBEHAVIOUR LAYER (advisory)\n"
            f"  answers analysed: {s['answers']} | "
            f"flagged for verification: {s['flagged']} | "
            f"probes: {s['probes']} "
            f"(verified {s['verified']}, "
            f"unresolved {s['unresolved']}, "
            f"inconsistent {s['inconsistent']})\n"
            f"  {b['disclaimer']}"
        )

    cp = rep.get("candidate_profile")

    if cp:
        x = cp.get("extraction_log") or {}

        txt += (
            f"\n\nCANDIDATE PROFILE: {cp['file']}\n"
            f"  skills by source: {x.get('skills_by_source')} | "
            f"learned this run: "
            f"{[a['name'] for a in x.get('accepted', [])]} "
            f"| dropped: {len(x.get('dropped', []))} | "
            f"vocabulary terms for transcription: "
            f"{cp['vocabulary_terms']}"
        )

    if rep.get("cue_log"):
        txt += (
            f"\n\nCUE FILES (vision/voice/text/integrity, timestamped): "
            f"{rep['cue_log'].get('folder')}"
        )

    return txt


R.TTSEngine = _make_tts
R.SpeechInput = _make_stt
R.InterviewEngine = LiveEngine
R.build_report = _build
R.render_text = _render


def _pull(argv, name, default=""):
    """Remove --name value / --name=value from argv."""

    out, val, i = [], default, 0

    while i < len(argv):
        a = argv[i]

        if a == name and i + 1 < len(argv):
            val = argv[i + 1]
            i += 2
            continue

        if a.startswith(name + "="):
            val = a.split("=", 1)[1]
            i += 1
            continue

        out.append(a)
        i += 1

    return out, val


def main():
    import time as _t

    argv = sys.argv[1:]

    argv, profile_json = _pull(argv, "--profile-json", "")
    resume_p = _pull(argv, "--resume")[1]
    jd_p = _pull(argv, "--jd")[1]
    company = _pull(argv, "--company")[1]
    role_key = _pull(argv, "--role-key")[1]

    t0 = _t.monotonic()

    if (
        resume_p
        and jd_p
        and "--test-tts" not in argv
        and os.environ.get("SMARTHIRE_PROFILE", "1") != "0"
    ):
        from profile_builder_v60 import build_profile

        print(
            "=" * 60
            + "\nSMART HIRE — building candidate profile "
              "(skill bank + NuExtract fallback)\n"
            + "=" * 60
        )

        out = profile_json or str(_cfg.PROFILE_JSON)

        nux = (
            "off"
            if os.environ.get("SMARTHIRE_NUEXTRACT", "auto").lower()
            in ("off", "0")
            else "auto"
        )

        try:
            b = build_profile(
                resume_p,
                jd_p,
                company,
                role_key,
                out_path=out,
                nuextract=nux
            )

        except Exception as e:
            import traceback

            print(
                f"  profile builder failed "
                f"({type(e).__name__}: {e}); "
                f"falling back to the plain V58 extractors\n"
                f"{traceback.format_exc(limit=3)}"
            )
            b = None

        if b is not None:
            _V60.update(bundle=b, profile_path=out)

            res, job = b.resume, b.job

            R.extract_resume = lambda *a, **k: res
            R.extract_jd = lambda *a, **k: job

            _orig_cached = R.cached

            R.cached = lambda kind, path, build: (
                (res, False)
                if kind == "resume"
                else (job, False)
                if kind.startswith("jd_")
                else _orig_cached(kind, path, build)
            )

            try:
                from transcript_fix_v60 import TranscriptCorrector
                _V60["corrector"] = TranscriptCorrector.from_profile(
                    b.profile
                )

            except Exception as e:
                print(
                    f"  transcript corrector unavailable: "
                    f"{type(e).__name__}: {e}"
                )

            try:
                from relevance_v60 import ResumeRelevance
                _V60["relevance"] = ResumeRelevance.from_profile(
                    b.profile
                )

            except Exception as e:
                print(
                    f"  resume relevance unavailable: "
                    f"{type(e).__name__}: {e}"
                )

    if os.environ.get("SMARTHIRE_CUELOG", "1") != "0":
        try:
            from cue_log_v60 import CueLog
            from cheat_detector_v60 import CheatMonitor

            _V60["cuelog"] = CueLog(session_t0=t0)

            _V60["monitor"] = CheatMonitor(
                cuelog=_V60["cuelog"],
                relevance=_V60["relevance"],
                session_t0=t0
            )

        except Exception as e:
            print(
                f"  cue log / assist-pattern monitor unavailable: "
                f"{type(e).__name__}: {e}"
            )

    if os.environ.get("SMARTHIRE_SER") == "1":
        try:
            from voice_runtime_v60 import SerRunner, silero_vad_fn

            sr_ = SerRunner()

            print(
                "  Voice models: loading SER arena "
                "(audeering / ehcalabres / HuBERT)…"
            )

            if sr_.load():
                COLLECTOR.ser_fn = sr_
                _V60["ser"] = sr_

            else:
                print(
                    f"  SER unavailable: "
                    f"{sr_.errors[-1] if sr_.errors else '?'} "
                    f"— continuing with acoustic cues only"
                )

            COLLECTOR.vad_fn = silero_vad_fn()

            print(
                "  Voice: Silero VAD "
                + (
                    "ready"
                    if COLLECTOR.vad_fn
                    else "unavailable (energy VAD used)"
                )
            )

        except Exception as e:
            print(
                f"  voice runtime unavailable: "
                f"{type(e).__name__}: {e}"
            )

    if os.environ.get("SMARTHIRE_NLI") == "1":
        try:
            from text_cues_v59 import NLIChecker

            name = (
                str(_cfg.NLI_DIR)
                if os.path.isdir(str(_cfg.NLI_DIR))
                else "cross-encoder/nli-deberta-v3-xsmall"
            )

            n = NLIChecker(name)
            _V60["nli"] = n if n.available else None

            print(
                "  NLI contradiction check: "
                + ("ready" if n.available else "unavailable")
            )

        except Exception as e:
            print(
                f"  NLI unavailable: {type(e).__name__}: {e}"
            )

    sys.argv = [sys.argv[0]] + argv

    try:
        R.main()

    finally:
        if _V60["cuelog"] is not None:
            _V60["cuelog"].close()


if __name__ == "__main__":
    main()