"""report_engine_v58.py — honest report: résumé evidence, candidate claims and interview evidence are kept apart."""
from __future__ import annotations

import json
from typing import Dict, List

import concepts_v58 as K
from evidence_engine_v58 import Competency
from models_v58 import *            # noqa: F401,F403


def _resume_text(c: Competency) -> str:
    return {"none": "none", "listed": "listed only", "used_in_project": "used in project", "strong": "multiple projects"}.get(c.resume_level, c.resume_level)


def build_report(engine, tts_stats=None, stt_status="typed", llm_summary=None, startup=None) -> Dict:
    g, r, j, pl = engine.graph, engine.resume, engine.job, engine.planner
    cov = g.coverage()
    comps = []
    for k in g.required_keys():
        c = g.comp[k]
        order = {"practical": 0, "theoretical": 1, "learning": 2, "claim": 3, "no_experience": 4, "unknown": 5}
        win = sorted([i for i in c.direct_items() if i.quote], key=lambda i: (order.get(i.stance, 9), -i.weight))
        q = [i.quote for i in win][:2]
        comps.append({"key": k, "name": c.name, "importance": c.importance, "required": c.required, "expected_depth": c.expected_depth, "resume_evidence": _resume_text(c),
                      "resume_sources": c.resume_sources[:3], "interview_status": c.status, "confidence": c.confidence, "addressed": c.addressed, "resolved": g.resolved(c),
                      "validated": g.validated(c), "validation_basis": g.validation_basis(c), "questions": c.questions_asked, "facets_covered": sorted(c.facets_covered), "candidate_claim": c.candidate_claim,
                      "said_unknown": c.said_unknown, "contradicts_resume": c.contradicts_resume, "implied_only": bool(c.interview) and not c.direct_items(), "quotes": q, "notes": c.notes})
    soft = [{"key": k, "name": s.name, "strength": round(s.strength, 2), "asked": s.asked} for k, s in g.soft.items()]
    listed_untouched = [s.name for s in r.all_skills() if s.listed and not s.key.startswith("custom:") and not (g.get(s.key) and (g.get(s.key).interview or g.get(s.key).addressed)) and not s.in_projects]
    rep = {
        "version": "V58",
        "candidate": {"name": r.name, "education": [f"{e.degree} {e.field_of_study} — {e.institution} ({e.dates}{', CGPA ' + e.gpa if e.gpa else ''})".strip() for e in r.education],
                      "projects": [p.short_name for p in r.projects], "experience": [f"{x.title} @ {x.organization}" for x in r.experience],
                      "parse": r.summary_line(), "warnings": r.warnings},
        "role": {"title": j.title, "company": j.company, "seniority": j.seniority, "parse": j.summary_line(), "company_context_not_requirements": j.company_context},
        "interview": {"questions": pl.turn, "minutes_virtual": round(pl.clock / 60, 1), "coverage": cov, "projects_sampled": f"{len(pl.projects_done)}/{len(r.projects)}",
                      "soft_areas_with_evidence": cov["soft_with_evidence"]},
        "competencies": comps, "soft_skills": soft,
        "summary": {
            "validated_hands_on": [c["name"] for c in comps if c["validation_basis"] == "hands_on"],
            "validated_conceptual_awareness_level": [c["name"] for c in comps if c["validation_basis"] == "conceptual"],
            "practical_unvalidated": [c["name"] for c in comps if c["interview_status"] == PRACTICAL and not c["validated"] and not c["implied_only"]],
            "theoretical": [c["name"] for c in comps if c["interview_status"] in (THEORETICAL, PARTIALLY_VALIDATED) and not c["validated"]],
            "learning_only": [c["name"] for c in comps if c["interview_status"] == LEARNING_ONLY],
            "explicit_no_experience": [c["name"] for c in comps if c["interview_status"] == EXPLICIT_NO_EXPERIENCE],
            "said_dont_know": [c["name"] for c in comps if c["said_unknown"]],
            "contradicted": [c["name"] for c in comps if c["interview_status"] == CONTRADICTED],
            "implied_only_not_asked": [c["name"] for c in comps if c["implied_only"] and not c["addressed"]],
            "not_asked": [c["name"] for c in comps if not c["addressed"] and not c["implied_only"]],
            "resume_claims_not_probed": listed_untouched,
        },
        "behavioural_note": "Behavioural scores come only from the candidate's own examples in this interview; absence of a score means 'not covered', not 'weak'.",
        "multimodal": "No audio/video behavioural analysis was run by this text/voice runner (advisory signals are never an automated verdict).",
        "transcript": [{"turn": t.turn, "phase": t.phase, "interviewer": t.interviewer, "answer": t.answer, "kind": t.kind, "targets": t.targets, "why": t.why, "source": t.source,
                        "interpretation": t.interpretation, "evidence_changes": t.evidence_changes} for t in engine.transcript],
        "system": {"llm": llm_summary or {"mode": "not used"}, "llm_startup": (startup.__dict__ if startup else None), "tts": tts_stats, "stt": stt_status,
                   "per_turn_ms_avg": _avg(engine.timings)},
    }
    return rep


def _avg(rows):
    if not rows:
        return {}
    keys = {k for r in rows for k in r}
    return {k: round(sum(r.get(k, 0) for r in rows) / len(rows), 1) for k in keys}


def render_text(rep: Dict) -> str:
    L = ["=" * 70, "SMARTHIRE V58 — INTERVIEW REPORT", "=" * 70, ""]
    c = rep["candidate"]
    L += [f"CANDIDATE  {c['name']}", *[f"  Education : {e}" for e in c["education"]], f"  Projects  : {', '.join(c['projects']) or '—'}", f"  Experience: {', '.join(c['experience']) or '—'}", f"  Parse     : {c['parse']}"]
    L += [f"  ! {w}" for w in c["warnings"]]
    ro = rep["role"]
    L += ["", f"ROLE  {ro['title']} @ {ro['company']} | seniority: {ro['seniority']}", f"  {ro['parse']}"]
    if ro["company_context_not_requirements"]:
        L.append(f"  Company text mentions (NOT treated as requirements): {', '.join(ro['company_context_not_requirements'])}")
    iv = rep["interview"]
    cv = iv["coverage"]
    L += ["", f"INTERVIEW  {iv['questions']} questions | {iv['minutes_virtual']} min (speaking+answering estimate)",
          f"  Required competencies: {cv['required']} | asked {cv['addressed']} | resolved {cv['resolved']} | validated {cv['validated']}",
          f"  Weighted coverage: asked {cv['weighted_addressed_pct']}% | resolved {cv['weighted_resolved_pct']}% | validated {cv['weighted_validated_pct']}%   (résumé alone adds 0%)",
          f"  Résumé projects sampled: {iv['projects_sampled']} | behavioural areas with evidence: {cv['soft_with_evidence']}/{cv['soft_total']}", "",
          "TECHNICAL COMPETENCY REPORT (interview evidence; résumé shown separately)",
          f"  {'Competency':34}{'Imp':4}{'Résumé':18}{'Interview status':26}{'Conf':6}{'Asked':6}{'Valid'}"]
    for x in rep["competencies"]:
        st = x["interview_status"] + (" (implied)" if x["implied_only"] else "") + (" [concept]" if x["validation_basis"] == "conceptual" else "") + (" (dont-know)" if x["said_unknown"] and x["interview_status"] == "UNKNOWN" else "")
        L.append(f"  {x['name'][:33]:34}{x['importance']:<4}{x['resume_evidence']:18}{st:26}{x['confidence']:<6}{'yes' if x['addressed'] else 'no':6}{'yes' if x['validated'] else 'no'}")
        for q in x["quotes"][:1]:
            L.append(f"      ↳ “{q}”")
        if x["facets_covered"]:
            L.append(f"      · evidence areas shown: {', '.join(x['facets_covered'])}")
        if x["contradicts_resume"]:
            L.append("      ! conflicts with résumé/earlier answer")
    L += ["", "SOFT SKILLS (only from the candidate's own examples)"]
    for s in rep["soft_skills"]:
        L.append(f"  {s['name']:30} strength={s['strength']:.2f}  {'covered' if s['strength'] >= 0.35 else ('asked, weak evidence' if s['asked'] else 'not covered')}")
    sm = rep["summary"]
    L += ["", "SUMMARY"]
    for lab, key in [("Validated — hands-on (described doing it)", "validated_hands_on"), ("Validated — conceptual (JD asks only for exposure)", "validated_conceptual_awareness_level"), ("Practical, not yet validated", "practical_unvalidated"), ("Theoretical / conceptual only", "theoretical"),
                     ("Learning only", "learning_only"), ("Explicit no experience", "explicit_no_experience"), ("Said they don't know", "said_dont_know"), ("Contradicted", "contradicted"),
                     ("Implied by other answers, not asked", "implied_only_not_asked"), ("Not asked (unresolved)", "not_asked"), ("Résumé claims never probed", "resume_claims_not_probed")]:
        L.append(f"  {lab:40}: {', '.join(sm[key]) or '—'}")
    L += ["", rep["behavioural_note"], rep["multimodal"], "", "SYSTEM", f"  LLM: {json.dumps(rep['system']['llm'])}", f"  per-turn ms (avg): {rep['system']['per_turn_ms_avg']}", f"  TTS: {rep['system']['tts']}  STT: {rep['system']['stt']}"]
    return "\n".join(L)


def save_json(rep: Dict, path: str):
    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False, default=str)
