"""schema_v58.py — normalised data model for résumés and job descriptions (plain dataclasses, JSON-serialisable)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

SKILL_CATEGORIES = ["programming_languages", "frameworks", "libraries", "databases", "AI_ML", "tools", "other"]


@dataclass
class Evidence:
    section: str = ""
    text: str = ""             # source text the item was extracted from
    lines: List[int] = field(default_factory=list)


@dataclass
class TechRef:
    key: str                   # concept key, or "custom:<name>"
    name: str
    where: str = "text"        # title | bullet | skills
    snippet: str = ""


@dataclass
class SkillEntry:
    key: str
    name: str
    category: str
    listed: bool = True        # appears in a skills section
    in_projects: List[str] = field(default_factory=list)   # project names where it is actually used
    source: str = "skills_section"


@dataclass
class Project:
    title: str
    short_name: str
    subtitle: str = ""
    dates: str = ""
    description: str = ""
    bullets: List[str] = field(default_factory=list)
    technologies: List[TechRef] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    ownership: str = "unclear"       # individual | team | unclear
    ownership_verbs: List[str] = field(default_factory=list)
    results: List[str] = field(default_factory=list)
    implied: Dict[str, float] = field(default_factory=dict)   # concept key -> weight (e.g. YOLO => Computer Vision)
    evidence: Evidence = field(default_factory=Evidence)

    def tech_keys(self) -> List[str]:
        return [t.key for t in self.technologies]


@dataclass
class Education:
    degree: str = ""
    field_of_study: str = ""
    institution: str = ""
    dates: str = ""
    gpa: str = ""
    status: str = ""                 # in_progress | completed | ""
    evidence: Evidence = field(default_factory=Evidence)


@dataclass
class Experience:
    title: str = ""
    organization: str = ""
    dates: str = ""
    kind: str = "job"                # job | internship | virtual_internship | freelance | volunteer
    bullets: List[str] = field(default_factory=list)
    technologies: List[TechRef] = field(default_factory=list)
    evidence: Evidence = field(default_factory=Evidence)


@dataclass
class Certification:
    name: str
    issuer: str = ""
    date: str = ""
    kind: str = "certificate"        # certificate | course | virtual_internship | training
    evidence: Evidence = field(default_factory=Evidence)


@dataclass
class Contact:
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)


@dataclass
class Resume:
    name: str = ""
    first_name: str = ""
    headline: str = ""
    contact: Contact = field(default_factory=Contact)
    summary: str = ""
    education: List[Education] = field(default_factory=list)
    experience: List[Experience] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    skills: Dict[str, List[SkillEntry]] = field(default_factory=lambda: {c: [] for c in SKILL_CATEGORIES})
    certifications: List[Certification] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    publications: List[str] = field(default_factory=list)
    coursework: List[str] = field(default_factory=list)
    extracurricular: List[str] = field(default_factory=list)
    spoken_languages: List[Dict[str, str]] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    sections_found: List[str] = field(default_factory=list)
    layout: str = "single"
    parse_method: str = ""
    warnings: List[str] = field(default_factory=list)
    raw_text: str = ""

    # convenience -------------------------------------------------------------------------
    def all_skills(self) -> List[SkillEntry]:
        return [s for c in SKILL_CATEGORIES for s in self.skills.get(c, [])]

    def skill_keys(self) -> List[str]:
        return [s.key for s in self.all_skills()]

    def project_by_name(self, name: str) -> Optional[Project]:
        n = (name or "").lower()
        for p in self.projects:
            if p.short_name.lower() == n or p.title.lower() == n:
                return p
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_text", None)
        return d

    def summary_line(self) -> str:
        return (f"name={self.name or '?'} | education={len(self.education)} | experience={len(self.experience)} | projects={len(self.projects)} | "
                f"skills={len(self.all_skills())} ({sum(1 for s in self.all_skills() if s.listed)} listed) | certs={len(self.certifications)} | "
                f"layout={self.layout} | sections={self.sections_found}")


# ── job description ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Requirement:
    key: str                         # concept key or "custom:<slug>" / "soft:<name>"
    name: str
    canonical_name: str
    category: str
    required: bool = True
    importance: int = 2              # 1 nice-to-have, 2 expected, 3 core
    expected_depth: str = "implementation"   # awareness | implementation | depth
    evidence_expected: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    source_text: str = ""
    source_section: str = ""
    kind: str = "technology"         # technology | concept | practice | domain | behavioural
    domain_context_only: bool = False


@dataclass
class JobProfile:
    title: str = ""
    company: str = ""
    seniority: str = "junior"
    responsibilities: List[str] = field(default_factory=list)
    requirements: Dict[str, Requirement] = field(default_factory=dict)
    behavioural: Dict[str, Requirement] = field(default_factory=dict)
    education_required: str = ""
    ignored_sections: List[str] = field(default_factory=list)
    company_context: List[str] = field(default_factory=list)    # domain terms seen only in company/boilerplate text (NOT requirements)
    warnings: List[str] = field(default_factory=list)
    unresolved_lines: List[str] = field(default_factory=list)
    raw_text: str = ""

    def required_list(self) -> List[Requirement]:
        return sorted([r for r in self.requirements.values() if r.required and not r.domain_context_only], key=lambda r: -r.importance)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_text", None)
        return d

    def summary_line(self) -> str:
        req = [r for r in self.requirements.values() if r.required and not r.domain_context_only]
        pref = [r for r in self.requirements.values() if not r.required and not r.domain_context_only]
        dom = [r for r in self.requirements.values() if r.domain_context_only]
        return (f"title={self.title!r} | seniority={self.seniority} | required={len(req)} preferred={len(pref)} domain-context={len(dom)} | "
                f"behavioural={sorted(self.behavioural)} | ignored sections={self.ignored_sections}")
