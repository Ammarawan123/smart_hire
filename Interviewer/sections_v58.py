"""
sections_v58.py — semantic section detection for résumés / JDs.

Two independent signals are combined (neither alone is trusted):
  SHAPE     is this line *a heading*?   letter-spacing, all-caps, bold, larger font, short, no sentence punctuation,
            vertical gap above.  (typography, not vocabulary)
  MEANING   which section is it?        heading vocabulary on the *squashed* text (so "I N T E RE S T S",
            "TECHNICALSKILLS", "Skills & Tools", "What I know" all resolve), and — when the heading is unknown
            or the document has no headings at all — the *content* of the block (dates+degrees => education,
            title+action bullets => projects/experience, concept-dense lists => skills, ...).
A word like "English" can never open a section on its own: a heading must pass the SHAPE test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import concepts_v58 as K
from doc_reader_v58 import Line
from textkit_v58 import is_bullet, strip_bullet, word_count

SECTIONS = ["summary", "skills", "projects", "experience", "education", "certifications", "achievements", "extracurricular",
            "languages", "interests", "coursework", "publications", "references", "contact", "other"]

# (section, squashed-stem regexes) — evaluated in order; first hit wins
_STEMS: List[Tuple[str, List[str]]] = [
    ("contact", ["contact", "personaldetail", "personalinfo", "personaldata"]),
    ("summary", ["summary", "profile", "objective", "aboutme", "careergoal", "overview", "introduction", "professionalstatement"]),
    ("coursework", ["coursework", "relevantcourses", "relevantmodules"]),
    ("publications", ["publication", "papers", "research"]),
    ("extracurricular", ["extracurricular", "cocurricular", "activities", "volunteer", "leadership", "clubs", "societies", "communityservice"]),
    ("certifications", ["certif", "licens", "courses", "training", "onlinecourse"]),
    ("achievements", ["achiev", "award", "honor", "honour", "accomplish", "scholarship", "recognition"]),
    ("education", ["educat", "academicbackground", "qualification", "academics", "schooling"]),
    ("projects", ["project", "portfolio", "selectedwork", "opensource", "casestud", "whatibuilt", "whativebuilt", "thingsibuilt", "thingsivebuilt", "whativemade", "whatimade", "workdone"]),
    ("experience", ["experienc", "employment", "workhistory", "internship", "careerhistory", "professionalbackground", "positionsheld"]),
    ("skills", ["skill", "competenc", "expertise", "proficienc", "technologies", "techstack", "tech", "toolsand", "technicalknowledge", "programminglanguages", "whatiknow", "toolkit", "strengths"]),
    ("languages", ["languages", "languageproficiency", "spokenlanguages"]),
    ("interests", ["interest", "hobbies", "hobby"]),
    ("references", ["reference"]),
]
_VOCAB = sorted(set("""education educational background academic academics qualification qualifications technical professional core key skills skill competencies competency expertise proficiency
 technologies technology tools tool tech stack projects project personal selected notable featured major experience work employment history internship internships career relevant certifications
 certification certificates certificate courses course training licenses license awards award achievements achievement honors honours accomplishments publications research summary profile objective about me
 interests hobbies languages language references contact details information extracurricular activities volunteer volunteering leadership coursework and of other additional links portfolio open source
 contributions programming computer software development online learning highlights overview introduction statement clubs societies community service positions held what i know built""".split()), key=len, reverse=True)


def squash(s: str) -> str:
    return re.sub(r"[^a-z]", "", (s or "").lower())


def _vocab_residue(sq: str) -> int:
    """Characters of `sq` not explained by heading vocabulary (greedy). 0-2 => the line is a pure heading phrase."""
    r = sq
    for w in _VOCAB:
        r = r.replace(w, "")
    return len(r)


def stem_section(text: str) -> Optional[str]:
    sq = squash(text)
    if len(sq) < 3 or len(sq) > 48:
        return None
    if _vocab_residue(sq) > 2:
        return None
    for sec, stems in _STEMS:
        if any(st in sq for st in stems):
            if sec == "languages" and any(x in sq for x in ("programming", "computer", "technical")):
                return "skills"
            return sec
    return None


@dataclass
class Section:
    name: str
    heading: str = ""
    lines: List[Line] = field(default_factory=list)
    how: str = "heading"           # heading | content | inline | header-block
    confidence: float = 1.0


def _heading_shape_score(ln: Line, body_size: float, nxt: Optional[Line]) -> float:
    t = ln.text.strip()
    if not t or is_bullet(t) or len(t) > 70:
        return 0.0
    letters = re.sub(r"[^A-Za-z]", "", t)
    if len(letters) < 3:
        return 0.0
    if re.search(r"[.!?]$", t) or word_count(t) > 7:
        return 0.0
    s = 0.0
    if ln.letterspaced:
        s += 3.0
    if letters.isupper() and len(letters) >= 4:
        s += 2.0
    if ln.bold:
        s += 1.5
    if body_size and ln.size and ln.size >= 1.15 * body_size:
        s += 2.0
    if t.endswith(":"):
        s += 1.0
    if word_count(t) <= 3:
        s += 0.8
    if ln.gap_before >= 0.8:
        s += 0.5
    if "|" in t or t.count(",") >= 2 or "@" in t:
        s -= 3.0
    return s


def detect_heading(ln: Line, body_size: float, nxt: Optional[Line], first_content: bool = False) -> Optional[Tuple[str, float]]:
    shape = _heading_shape_score(ln, body_size, nxt)
    sec = stem_section(ln.text.rstrip(": "))
    t0 = ln.text.strip().rstrip(":").strip()
    if sec and shape > 0 and word_count(t0) <= 3 and (t0.istitle() or t0.isupper() or ln.letterspaced):
        shape += 2.0          # a pure heading phrase ("Technical Projects", "Selected Work") in heading case
    if sec and shape >= 2.5:
        return sec, min(1.0, 0.5 + shape / 8)
    if not sec and shape >= 5.0 and not first_content:
        return "other", 0.5               # unknown but unmistakably a heading — content classifier decides
    return None


_INLINE_RX = re.compile(r"^([A-Za-z&/ ]{3,40}):\s+(.+)$")


def detect_inline_heading(text: str) -> Optional[Tuple[str, str]]:
    """'Skills: Python, SQL' as a single line (label with content)."""
    m = _INLINE_RX.match(text.strip())
    if not m:
        return None
    sec = stem_section(m.group(1))
    return (sec, m.group(2)) if sec in ("skills", "languages", "interests", "summary", "coursework") else None


# ── content-based block classification ─────────────────────────────────────────────────────────
_DEGREE = re.compile(r"\b(b\.?\s?s\.?c?|m\.?\s?s\.?c?|bachelor'?s?|master'?s?|ph\.?d|mba|bba|b\.?e\b|b\.?tech|m\.?tech|intermediate|matric|fsc|hssc|ssc|a-levels?|o-levels?|diploma|cgpa|gpa|university|college|institute|school|board)\b", re.I)
_DATE_RX = re.compile(r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?(?:19|20)\d{2}\s*(?:[-–—]|to)\s*(?:(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+)?(?:19|20)\d{2}|present|current|ongoing)", re.I)
_YEAR_RX = re.compile(r"\b(?:19|20)\d{2}\b")
_ACTION = re.compile(r"\b(built|developed|designed|implemented|created|trained|engineered|wrote|automated|integrated|deployed|optimi[sz]ed|led|managed|improved|applied|wrapped|cloned|analy[sz]ed|launched|maintained|migrated|tested|configured|achieved|reduced|increased|collaborated|delivered|handled|prepared)\b", re.I)
_ROLE = re.compile(r"\b(intern|engineer|developer|analyst|trainee|associate|consultant|manager|lead|officer|researcher|instructor|tutor|designer|freelanc\w*|assistant|coordinator|executive|specialist|administrator|architect)\b", re.I)
_CERT = re.compile(r"\b(certificat\w*|certified|course|specializ\w*|coursera|udemy|edx|linkedin learning|google cloud|aws|microsoft|forage|nptel|bootcamp|training|workshop|license)\b", re.I)


def classify_block(lines: List[str]) -> Tuple[str, float]:
    """Classify an unlabeled block of lines by what it *contains*."""
    txt = [l for l in lines if l.strip()]
    if not txt:
        return "other", 0.0
    n = len(txt)
    joined = " ".join(txt)
    bullets = sum(1 for l in txt if is_bullet(l))
    verbs = sum(1 for l in txt if _ACTION.search(l))
    dates = sum(1 for l in txt if _DATE_RX.search(l))
    years = sum(1 for l in txt if _YEAR_RX.search(l))
    degs = sum(1 for l in txt if _DEGREE.search(l))
    certs = sum(1 for l in txt if _CERT.search(l))
    roles = sum(1 for l in txt if _ROLE.search(l) and word_count(l) <= 14)
    listy = sum(1 for l in txt if (l.count(",") + l.count("|") + l.count(";")) >= 2 or re.match(r"^[A-Za-z &/]{3,28}:\s", l))
    cm = len(K.find_mentions(joined))
    density = cm / max(1, word_count(joined)) * 10
    scores = {
        "education": 1.5 * degs / n + (0.4 if dates and degs else 0),
        "skills": 0.8 * listy / n + min(density, 3.0) * 0.18 - 0.5 * verbs / n,
        "projects": 0.5 * bullets / n + 0.7 * verbs / n + (0.25 if cm >= 2 else 0) - 0.4 * (roles > 0),
        "experience": 0.3 * bullets / n + 0.6 * verbs / n + 0.5 * roles / n + 0.4 * (dates > 0),
        "certifications": 1.0 * certs / n,
        "summary": 0.6 * (n <= 6 and word_count(joined) > 25 and bullets == 0) + 0.15 * len(re.findall(r"\b(passionate|motivated|seeking|student|experienced|aspiring)\b", joined, re.I)),
    }
    sec = max(scores, key=lambda k: scores[k])
    return (sec, min(1.0, scores[sec])) if scores[sec] >= 0.28 else ("other", scores[sec])


def segment(lines: List[Line]) -> Tuple[List[Section], List[str]]:
    """Split ordered lines into sections. Returns (sections, warnings)."""
    warns: List[str] = []
    sizes = [l.size for l in lines if l.size]
    body = sorted(sizes)[len(sizes) // 2] if sizes else 0.0
    secs: List[Section] = []
    cur = Section("header", "", [], "header-block")
    secs.append(cur)
    saw_content = False
    for i, ln in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        h = detect_heading(ln, body, nxt, first_content=not saw_content and i == 0)
        if h:
            sec, conf = h
            cur = Section(sec, ln.text.strip(), [], "heading", conf)
            secs.append(cur)
            continue
        inl = detect_inline_heading(ln.text) if not is_bullet(ln.text) else None
        if inl and (ln.letterspaced or ln.bold or inl[0] in ("skills", "languages", "interests", "coursework")) and cur.name not in ("skills",):
            cur = Section(inl[0], ln.text.split(":")[0].strip(), [], "inline", 0.8)
            secs.append(cur)
            cur.lines.append(Line(inl[1], ln.x0, ln.size, ln.bold, False, ln.column, ln.page, ln.row, 0))
            continue
        cur.lines.append(ln)
        saw_content = True
    named = [s for s in secs if s.name not in ("header",)]
    # unknown-heading sections: decide by content
    for s in named:
        if s.name == "other" and s.lines:
            sec, conf = classify_block([l.text for l in s.lines])
            s.name, s.how, s.confidence = sec, "content", conf
    # no (or almost no) headings: segment the whole document by blocks
    if len([s for s in named if s.lines]) < 2:
        warns.append("few/no section headings found; sections inferred from content blocks")
        secs = _blocks_from_content(lines)
    # merge consecutive sections with the same name (e.g. heading repeated across columns)
    merged: List[Section] = []
    for s in secs:
        same = [m for m in merged if m.name == s.name and s.name not in ("header", "other")]
        if same:
            same[0].lines.extend(s.lines)
        else:
            merged.append(s)
    return [s for s in merged if s.lines or s.name != "header"], warns


def _blocks_from_content(lines: List[Line]) -> List[Section]:
    blocks: List[List[Line]] = []
    cur: List[Line] = []
    for ln in lines:
        if cur and ln.gap_before >= 0.9 and not is_bullet(ln.text):
            blocks.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)
    out = [Section("header", "", blocks[0][:3] if blocks else [], "header-block")]
    if blocks:
        blocks[0] = blocks[0][3:]
    for b in blocks:
        if not b:
            continue
        sec, conf = classify_block([l.text for l in b])
        out.append(Section(sec, "", b, "content", conf))
    return out
