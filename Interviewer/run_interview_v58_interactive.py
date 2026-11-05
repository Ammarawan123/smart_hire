"""
SmartHire V58 — evidence-driven adaptive interviewer (typed or voice).

Example (same inputs as the V51 run):
  python run_interview_v58_interactive.py --resume $RESUME --jd $JD --company "Contour Software" --role-key "Trainee Software Developer" ^
      --llama-server $LLAMA --model $GGUF --voice $VOICE --debug

Flags: --no-llm (fully deterministic) | --no-tts | --test-tts | --stt whisper | --simulate | --answers-file answers.txt | --dump-profile | --stop-server
The text/voice runner is independent of the vision/audio pipelines; typed mode always works.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.paths import bootstrap  # noqa: E402
bootstrap()
sys.path.insert(0, str(HERE))

import concepts_v58 as K                       # noqa: E402
import v58_config as cfg                       # noqa: E402
from interview_engine_v58 import InterviewEngine   # noqa: E402
from jd_extractor_v58 import extract_jd         # noqa: E402
from llm_engine_v58 import LocalLLM             # noqa: E402
from report_engine_v58 import build_report, render_text, save_json   # noqa: E402
from resume_extractor_v58 import extract_resume  # noqa: E402
from stt_engine_v58 import SpeechInput          # noqa: E402
from tts_engine_v58 import TTSEngine            # noqa: E402


def _hash(path: str) -> str:
    h = hashlib.sha256()
    h.update(cfg.PARSER_VERSION.encode())
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def cached(kind: str, path: str, build):
    Path(cfg.CACHE_DIR).mkdir(parents=True, exist_ok=True)
    f = Path(cfg.CACHE_DIR) / f"{kind}_{_hash(path)}.pkl"
    if f.exists():
        try:
            return pickle.loads(f.read_bytes()), True
        except Exception:
            pass
    obj = build()
    try:
        f.write_bytes(pickle.dumps(obj))
    except Exception:
        pass
    return obj, False


def blueprint(engine: InterviewEngine):
    g = engine.graph
    sup, gaps, claims = [], [], []
    for k in g.required_keys():
        c = g.comp[k]
        (sup if c.resume_level in ("used_in_project", "strong") else gaps if c.resume_level == "none" else claims).append(c.name)
    return sup, gaps, claims


def main():
    ap = argparse.ArgumentParser(description="SmartHire V58 adaptive interviewer")
    ap.add_argument("--resume"); ap.add_argument("--jd"); ap.add_argument("--company", default=""); ap.add_argument("--role-key", default="")
    ap.add_argument("--llama-server", default=""); ap.add_argument("--model", default=""); ap.add_argument("--llama-port", type=int, default=cfg.LLAMA_PORT)
    ap.add_argument("--no-llm", action="store_true"); ap.add_argument("--stop-server", action="store_true", help="stop llama-server at exit (default: leave it running for the next interview)")
    ap.add_argument("--voice", default=""); ap.add_argument("--piper-exe", default=""); ap.add_argument("--audio-device", default=None)
    ap.add_argument("--no-tts", action="store_true"); ap.add_argument("--test-tts", action="store_true")
    ap.add_argument("--stt", choices=["typed", "whisper"], default="typed"); ap.add_argument("--whisper-model", default="base.en")
    ap.add_argument("--debug", action="store_true"); ap.add_argument("--dump-profile", action="store_true")
    ap.add_argument("--profile-json", default="", help="write the parsed candidate (resume) + job profile to this JSON file; combine with --dump-profile to stop after parsing")
    ap.add_argument("--simulate", action="store_true"); ap.add_argument("--answers-file", default="")
    ap.add_argument("--max-questions", type=int, default=cfg.MAX_QUESTIONS); ap.add_argument("--minutes", type=float, default=cfg.TARGET_MINUTES)
    ap.add_argument("--report-json", default=str(HERE.parent / "outputs" / "interview_report.json"))
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if a.test_tts:
        tts = TTSEngine(a.voice, a.piper_exe, a.audio_device)
        ok = tts.diagnose(play=True, ask=(input if sys.stdin.isatty() else None))
        sys.exit(0 if ok else 1)
    if not a.resume or not a.jd:
        ap.error("--resume and --jd are required")

    print("=" * 60 + "\nSMART HIRE V58 — EVIDENCE-DRIVEN ADAPTIVE INTERVIEWER\n" + "=" * 60)
    print("\n[1/5] Reading résumé and job description…")
    t0 = time.time()
    resume, c1 = cached("resume", a.resume, lambda: extract_resume(a.resume, filename=a.resume))
    job, c2 = cached("jd_" + (a.role_key or "").replace(" ", "_") + "_" + (a.company or "").replace(" ", "_"), a.jd, lambda: extract_jd(a.jd, False, a.company, a.role_key))
    print(f"  Résumé : {resume.summary_line()}{' (cached)' if c1 else ''}")
    for w in resume.warnings:
        print(f"    ! {w}")
    print(f"  Projects: {[p.short_name for p in resume.projects]}")
    print(f"  JD     : {job.summary_line()}{' (cached)' if c2 else ''}")
    for w in job.warnings:
        print(f"    ! {w}")
    print(f"  parse time: {time.time() - t0:.2f}s")
    if a.profile_json:
        Path(a.profile_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.profile_json).write_text(json.dumps({"resume": resume.to_dict(), "job": job.to_dict()}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"  candidate profile written: {a.profile_json}")
    if a.dump_profile:
        print(json.dumps({"resume": resume.to_dict(), "job": job.to_dict()}, indent=2, ensure_ascii=False, default=str))
        return

    print("\n[2/5] Local LLM…")
    llm = LocalLLM(a.llama_server, a.model, port=a.llama_port, enabled=not a.no_llm and not a.simulate, keep_running=not a.stop_server)
    st = llm.start()
    if st.ok:
        print(f"  LLM: ready ({st.reason}, {st.seconds:.1f}s) — {Path(st.model).name if st.model else llm.base}")
        llm.warm_up()                                  # warm the prompt cache while the greeting is spoken
    else:
        print(f"  LLM: not used ({st.reason}) — {st.detail}\n  Interview continues fully deterministic (all core logic is LLM-free).")
    engine = InterviewEngine(resume, job, llm if st.ok else None, a.minutes, a.max_questions)

    print("\n[3/5] Blueprint (résumé ↔ role) — résumé evidence only, NOT interview proof")
    sup, gaps, claims = blueprint(engine)
    print(f"  Résumé shows use in a project : {', '.join(sup) or '—'}")
    print(f"  Listed only (claims to verify): {', '.join(claims) or '—'}")
    print(f"  No résumé support             : {', '.join(gaps) or '—'}")
    if job.company_context:
        print(f"  Company text mentions (not requirements): {', '.join(job.company_context)}")
    print("  Projects by JD value          : " + ", ".join(f"{n} ({v})" for v, n, _ in engine.planner.rank))

    print("\n[4/5] Voice…")
    tts = TTSEngine(a.voice, a.piper_exe, a.audio_device, enabled=not a.no_tts and not a.simulate)
    if a.no_tts or a.simulate:
        print("  TTS: off")
    elif not tts.initialize():
        print("  TTS: unavailable — text only")
        for n, ok, d in tts.checks:
            print(f"    {'ok ' if ok else 'FAIL'} {n} {d}")
    else:
        print(f"  TTS: {tts.stats.backend} ready")
    stt = SpeechInput(a.stt, a.whisper_model, tts=tts if tts.ready else None, device=a.audio_device)
    print(f"  STT: {stt.status}")

    scripted = None
    if a.answers_file:
        scripted = [l.rstrip("\n") for l in open(a.answers_file, encoding="utf-8") if l.strip()]
    if a.simulate:
        import sim_candidate_v58 as sim

    print("\n[5/5] Starting interview…\n")

    def say(text: str):
        print(f"Interviewer: {text}")
        tts.speak(text)

    out = engine.start()
    say(out.greeting + " " + out.text)
    idx = 0
    while True:
        if a.simulate:
            ans = sim.answer(engine.ctx, out.text)
            print(f"You: {ans}")
        elif scripted is not None:
            if idx >= len(scripted):
                break
            ans = scripted[idx]
            idx += 1
            print(f"You: {ans}")
        else:
            ans = stt.listen("You: ")
            if ans is None:
                break
            if not ans:
                continue
        try:
            out = engine.step(ans)
        except KeyboardInterrupt:
            break
        if a.debug:
            print("\n".join("   ┆ " + l for l in out.debug.splitlines()))
        if out.finished:
            if out.text:
                print(f"Interviewer: {out.text}")
            say(out.farewell)
            break
        say(out.text)
        if a.debug:
            m = out.metrics
            print(f"   ⏱ understanding {m.get('understand_ms', 0):.0f} ms | planning {m.get('plan_ms', 0):.0f} ms | total {m.get('total_ms', 0):.0f} ms")
    tts.close()
    print("\n" + "=" * 60 + "\nINTERVIEW COMPLETE\n" + "=" * 60)
    rep = build_report(engine, tts.stats.__dict__ if (tts.ready or tts.stats.backend not in ("none", "disabled")) else None, stt.status, llm.summary() if st.ok else {"mode": "deterministic (LLM not used)"}, st if st.ok else None)
    print(render_text(rep))
    save_json(rep, a.report_json)
    print(f"\nFull JSON report: {a.report_json}")
    llm.stop(force=a.stop_server)


if __name__ == "__main__":
    main()
