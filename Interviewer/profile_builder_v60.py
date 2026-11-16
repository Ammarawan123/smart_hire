"""
profile_builder_v60.py — résumé + JD -> ONE candidate_profile.json that the whole system shares.

  skill bank (static ontology + learned skills)  ->  rule-based résumé/JD extraction
        -> terms still unresolved?  NuExtract (grounded + gated)  ->  new skills added to the bank  ->  re-extract
        -> projects missing?  NuExtract project pass (grounded + validated like every rule-based project)
        -> candidate_profile.json

The profile is then used for
  * the INTERVIEW itself (the same Resume/JobProfile objects drive the question planner),
  * TRANSCRIPT CORRECTION (vocabulary.terms -> Whisper initial_prompt/hotwords + post-correction, NLP/transcript_fix_v60.py),
  * SBERT RELEVANCE (relevance_chunks -> resume-support / unlisted-technology cues, NLP/relevance_v60.py),
  * the report (extraction_log says which skills came from the static bank, the learned bank, or NuExtract, and what was dropped and why).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import concepts_v58 as K
from jd_extractor_v58 import extract_jd
from resume_extractor_v58 import extract_resume
from skill_bank_v60 import (JD_EXAMPLE, JD_TEMPLATE, NuExtractRunner, NuExtractServer, SkillBank, SkillExtractionLog, learn_skills, norm, nuextract_projects,
                            unresolved_resume_terms)

SCHEMA = "smarthire-candidate-profile/1"


@dataclass
class ProfileBundle:
    resume: object
    job: object
    profile: dict
    bank: SkillBank
    log: SkillExtractionLog


def _file_hash(path: str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _vocabulary(resume, job, company: str, role: str) -> List[str]:
    """Terms worth telling the speech recogniser about, most specific first."""
    terms: List[str] = []

    def add(t: str):
        t = (t or "").strip(" .,;:()[]\"'")
        if 2 <= len(t) <= 40 and len(t.split()) <= 4 and t not in terms and t.lower() not in {x.lower() for x in terms}:
            terms.append(t)
    for p in resume.projects:
        add(p.short_name)
    for s in resume.all_skills():
        if not s.key.startswith("custom:"):
            add(s.name)
    custom = set()
    for p in resume.projects:
        for t in p.technologies:
            add(t.name)
            if t.key.startswith("custom:"):
                custom.add(t.name.lower())
    for e in resume.experience:
        for t in e.technologies:
            add(t.name)
            if t.key.startswith("custom:"):
                custom.add(t.name.lower())
        add(e.organization)
    for r in job.requirements.values():
        if not r.domain_context_only:
            add(r.name)
    for s in resume.all_skills():                                # surface aliases the candidate may say aloud (not the ambiguous one-word ones)
        c = K.get(s.key)
        if c:
            for al in c.aliases:
                if al not in K._AMBIGUOUS_ALIASES and len(al) >= 3:
                    add(al)
    for s in resume.all_skills():
        if s.key.startswith("custom:"):
            add(s.name)
    add(company or job.company)
    add(role or job.title)
    add(resume.name)
    for ed in resume.education:
        add(ed.institution)
    for c in resume.certifications:
        add(c.name)
    # a half-captured custom technology ("Weights" from "Weights & Biases") must not pollute the recogniser prompt
    full = [t for t in terms if t.lower() not in custom]
    terms = [t for t in terms if t.lower() not in custom or not any(norm(u).startswith(norm(t)) and len(u) > len(t) for u in full)]
    return terms[:160]


def _skill_rows(resume, bank: SkillBank) -> List[dict]:
    rows = []
    for s in resume.all_skills():
        if s.key.startswith("bank:"):
            src = "nuextract_new" if s.key[5:] in bank.new_this_run else "bank_learned"
            bank.touch(s.key)
        elif s.key.startswith("custom:"):
            src = "unresolved"
        else:
            src = "static_bank"
        c = K.get(s.key)
        rows.append({"key": s.key, "name": s.name, "category": s.category, "source": src, "listed": s.listed, "in_projects": s.in_projects,
                     "aliases": (c.aliases if c else [])[:8]})
    return rows


def build_profile(resume_path: str, jd_path: str, company: str = "", role_key: str = "", out_path: Optional[str] = None, nuextract: str = "auto",
                  bank_path: Optional[str] = None, runner: Optional[NuExtractRunner] = None, project_crosscheck: bool = True, log: Callable = print) -> ProfileBundle:
    """nuextract: 'auto' (use it when something needs it and it can start) | 'off' | 'on' (try even if nothing is unresolved).  `runner` can be injected (tests)."""
    bank = SkillBank.load(bank_path)
    xlog = SkillExtractionLog()
    resume = extract_resume(resume_path, filename=resume_path)
    job = extract_jd(jd_path, False, company, role_key)
    _drop_non_skills(resume, xlog)
    xlog.static_or_bank_hits = sum(1 for s in resume.all_skills() if not s.key.startswith("custom:"))
    r_cand = unresolved_resume_terms(resume)
    jd_lines = list(job.unresolved_lines)
    need_projects = (project_crosscheck and (len(resume.projects) < 2 or os.environ.get("SMARTHIRE_NUEXTRACT_PROJECTS") == "1")) or not resume.projects
    wants = nuextract != "off" and (bool(r_cand) or bool(jd_lines) or need_projects or nuextract == "on")
    server = None
    if wants and runner is None:
        server = NuExtractServer()
        xlog.nuextract_available = server.start()
        xlog.nuextract_reason = server.reason
        runner = NuExtractRunner(server.url) if xlog.nuextract_available else None
        if not xlog.nuextract_available:
            log(f"  NuExtract: not used ({server.reason}); rule-based + skill-bank extraction only")
    elif runner is not None:
        xlog.nuextract_available, xlog.nuextract_reason = True, "injected"
    elif nuextract == "off":
        xlog.nuextract_available, xlog.nuextract_reason = False, "disabled (--no-nuextract)"
    try:
        added = 0
        if runner is not None:
            added += learn_skills(runner, bank, resume.raw_text, r_cand, xlog, mode="resume")
            if jd_lines:
                added += learn_skills(runner, bank, job.raw_text, [], xlog, JD_TEMPLATE, JD_EXAMPLE, mode="jd", chunks=_chunk_lines(jd_lines))
            if added:
                log(f"  skill bank: learned {added} new skill(s): {', '.join(a['name'] for a in xlog.accepted)}")
                resume = extract_resume(resume_path, filename=resume_path)         # re-extract: the new skills are now ordinary concepts
                _drop_non_skills(resume, xlog)
                job = extract_jd(jd_path, False, company, role_key)
            if need_projects:
                for p in nuextract_projects(runner, resume, xlog):
                    resume.projects.append(p)
                    resume.warnings.append(f"project '{p.short_name}' was added by NuExtract (not found by the rule-based extractor); please check it")
                if xlog.projects_added:
                    log(f"  NuExtract added project(s): {', '.join(xlog.projects_added)}")
            xlog.errors += runner.errors[-3:]
        else:
            xlog.unresolved += [c["name"] for c in r_cand]
    finally:
        if server is not None:
            server.stop()                                                          # free the RAM before Qwen starts
    try:
        if bank.new_this_run or bank.data["rejected"] or any(s.key.startswith("bank:") for s in resume.all_skills()):
            bank.save()
    except OSError as e:
        xlog.errors.append(f"skill bank not saved: {e}")
    profile = _profile_dict(resume, job, bank, xlog, resume_path, jd_path, company, role_key)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(profile, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log(f"  candidate profile written: {out_path}")
    return ProfileBundle(resume, job, profile, bank, xlog)


def _drop_non_skills(resume, xlog: SkillExtractionLog) -> None:
    """Skills-list items the rule-based parser kept as plain text but that are soft skills / levels / sentences ('Excellent communication')."""
    from skill_bank_v60 import validate_candidate
    for cat, lst in resume.skills.items():
        keep = []
        for sk in lst:
            if sk.key.startswith("custom:") and not validate_candidate(sk.name)[0]:
                xlog.dropped.append({"name": sk.name, "reason": "not a technical skill: " + validate_candidate(sk.name)[1]})
                continue
            keep.append(sk)
        resume.skills[cat] = keep


def _chunk_lines(lines: List[str], max_chunk: int = 1400) -> List[str]:
    out, cur = [], ""
    for ln in lines:
        if len(cur) + len(ln) + 1 > max_chunk and cur:
            out.append(cur)
            cur = ""
        cur += ln + "\n"
    if cur.strip():
        out.append(cur)
    return out[:3]


def _profile_dict(resume, job, bank: SkillBank, xlog: SkillExtractionLog, rpath: str, jpath: str, company: str, role: str) -> dict:
    from relevance_v60 import resume_chunks
    skills = _skill_rows(resume, bank)
    prof = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {"resume": {"file": os.path.basename(rpath), "sha256_16": _file_hash(rpath)}, "jd": {"file": os.path.basename(jpath), "sha256_16": _file_hash(jpath)}},
        "candidate": {"name": resume.name, "first_name": resume.first_name, "headline": resume.headline},
        "resume": resume.to_dict(),
        "job": job.to_dict(),
        "skills": skills,
        "projects": [{"name": p.short_name, "technologies": [t.name for t in p.technologies], "methods": p.methods, "ownership": p.ownership} for p in resume.projects],
        "job_requirements": [{"key": r.key, "name": r.name, "required": r.required, "importance": r.importance, "depth": r.expected_depth} for r in job.requirements.values()],
        "vocabulary": {"terms": _vocabulary(resume, job, company, role), "used_by": ["whisper initial_prompt / hotwords", "transcript post-correction"]},
        "relevance_chunks": resume_chunks(resume),
        "extraction_log": {**xlog.to_dict(), "skill_bank_file": str(bank.path), "skill_bank_size": len(bank), "skills_by_source": _count(skills)},
        "notes": ["Skills marked 'unresolved' were not in the skill bank and could not be confirmed by NuExtract; they are kept as plain text.",
                  "Everything here is extracted from the documents; nothing is inferred about the person."],
    }
    return prof


def _count(rows: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[r["source"]] = out.get(r["source"], 0) + 1
    return out
