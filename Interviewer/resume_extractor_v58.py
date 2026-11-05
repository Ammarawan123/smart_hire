"""
resume_extractor_v58.py — structured résumé extraction: section -> item -> evidence.

Design rule (the V51 root-cause fix): an item exists only if the *structure* supports it.
  * a PROJECT needs a title line plus a description/bullets, and lives in a Projects-like section
    (or a content-classified project block when the résumé has no headings). A technology name,
    a spoken language or an interest can never become a project.
  * SKILLS are items of a skills section ("Label: a, b | c"), resolved to the concept registry and bucketed
    into categories; "English"/"Urdu" are spoken languages, not skills.
  * every item keeps the source text it came from (Evidence).
No model download is needed: layout analysis + lexicon + typography run on CPU in milliseconds. An optional
sentence-embedding / GLiNER pass can be bolted on (see research_v58.md) but is not required for correctness.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import concepts_v58 as K
import doc_reader_v58 as D
import sections_v58 as S
from schema_v58 import (SKILL_CATEGORIES, Certification, Contact, Education, Evidence, Experience, Project, Resume, SkillEntry, TechRef)
from textkit_v58 import clip, is_bullet, strip_bullet, word_count

NATURAL_LANGS = {"english", "urdu", "hindi", "arabic", "punjabi", "pashto", "sindhi", "balochi", "french", "german", "spanish", "chinese", "mandarin", "turkish", "persian", "farsi",
                 "bengali", "bangla", "tamil", "japanese", "korean", "russian", "italian", "portuguese", "dutch", "swahili"}
_DATE_TAIL = re.compile(r"""(?:\(\s*)?((?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?(?:19|20)\d{2}\s*(?:[-–—]|to)\s*(?:(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?(?:19|20)\d{2}|present|current|ongoing|expected)|
    (?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+)?(?:19|20)\d{2})(?:\s*\))?\s*$""", re.I | re.X)
_GPA_RX = re.compile(r"\b(?:cgpa|gpa)\s*[:=]?\s*(\d(?:\.\d{1,2})?)(?:\s*/\s*\d(?:\.\d)?)?", re.I)
_PRESENT = re.compile(r"present|current|ongoing|expected|in progress", re.I)
_DEGREE_START = re.compile(r"^\s*(bs(?:ai|cs|se|it|ds|ee|me|cy|ba)\b|b\.?\s?s\.?c?\.?|m\.?\s?s\.?c?\.?|bachelor(?:'s|s)?(?:\s+of)?|master(?:'s|s)?(?:\s+of)?|ph\.?d\.?|mba|bba|b\.?e\.?|b\.?tech|m\.?tech|intermediate|matric(?:ulation)?|f\.?sc\.?|h\.?s\.?s\.?c\.?|s\.?s\.?c\.?|a-levels?|o-levels?|diploma(?:\s+of)?|associate(?:'s)?(?:\s+degree)?)\b\.?\s*(?:in\s+|of\s+|[-–—:,(]\s*)?(.*)$", re.I)
_DEGREE_CODES = {"bsai": ("B.S.", "Artificial Intelligence"), "bscs": ("B.S.", "Computer Science"), "bsse": ("B.S.", "Software Engineering"), "bsit": ("B.S.", "Information Technology"),
                 "bsds": ("B.S.", "Data Science"), "bsee": ("B.S.", "Electrical Engineering"), "bsme": ("B.S.", "Mechanical Engineering"), "bscy": ("B.S.", "Cyber Security")}
_INST_RX = re.compile(r"\b(university|institute|college|academy|school|board|polytechnic|duet|nust|fast|iba|lums|uet|seecs|nu)\b", re.I)
_ACTION_VERBS = ("built", "developed", "designed", "implemented", "created", "trained", "engineered", "wrote", "automated", "integrated", "deployed", "optimized", "optimised", "led", "managed",
                 "improved", "applied", "wrapped", "cloned", "analyzed", "analysed", "launched", "maintained", "migrated", "tested", "configured", "achieved", "reduced", "increased", "delivered", "handled",
                 "prepared", "demonstrated", "performed", "constructed", "programmed", "fine-tuned", "evaluated", "annotated", "collected")
_TEAM_CUES = re.compile(r"\b(team of|team|collaborat\w+|group|together|members|contributed to|worked with|alongside|pair)\b", re.I)
_METRIC = re.compile(r"(\d+(?:\.\d+)?\s*%|\bmap\b|\baccuracy\b|\bf1\b|\bprecision\b|\brecall\b|\bimproved?\b|\breduced?\b|\bincreased?\b|\bachiev\w+|\b\d+x\b|\bspeed[- ]?up\b)", re.I)
_TECHNIQUES = ["data augmentation", "batch normalization", "dropout", "regularization", "logistic regression", "linear regression", "transfer learning", "fine-tuning", "fine tuning", "annotation",
               "hyperparameter", "cross-validation", "cross validation", "terrain generation", "collision detection", "sprite animation", "real-time", "end-to-end pipeline", "data labeling",
               "model training", "evaluation", "inference", "deployment", "feature engineering", "clustering", "classification", "segmentation", "object detection", "speech recognition", "tts"]
_LABEL_HINT = [(r"framework", "frameworks"), (r"librar", "libraries"), (r"database|\bdb\b|storage", "databases"), (r"\bml\b|\bdl\b|\bai\b|nlp|vision|machine|deep|learning|data science|\bcv\b", "AI_ML"),
               (r"tool|dev|ide|platform|cloud|devops|version|software", "tools"), (r"language|programming|coding", "programming_languages")]


# ── low-level helpers ───────────────────────────────────────────────────────────────────────────
def find_date(text: str) -> Tuple[str, str]:
    """Return (text_without_trailing_date, date_string)."""
    m = _DATE_TAIL.search(text.strip())
    if not m:
        return text.strip(), ""
    return text[: m.start()].rstrip(" ,;|–—-(").strip(), m.group(1).strip()


def split_title(raw: str) -> Tuple[str, str, str]:
    """'Sign Language Detection — YOLOv8'  ->  (name, subtitle, dates)."""
    parts = [p.strip() for p in re.split(r"\s{2,}", raw.strip()) if p.strip()]
    dates = ""
    if len(parts) > 1:
        rest, d = find_date(parts[-1])
        if d and not rest:
            dates, parts = d, parts[:-1]
    t = "  ".join(parts)
    t, d2 = find_date(t)
    dates = dates or d2
    m = re.split(r"\s[—–|]\s|\s-\s|:\s", t, maxsplit=1)
    name = m[0].strip(" -–—|:")
    sub = m[1].strip() if len(m) > 1 else ""
    return name, sub, dates


def _looks_like_sentence(t: str) -> bool:
    return word_count(t) >= 8 or t.rstrip().endswith((".", "!", "?")) and word_count(t) >= 4


def _starts_with_action(t: str) -> bool:
    w = re.match(r"^\W*([A-Za-z\-]+)", t)
    return bool(w) and w.group(1).lower() in _ACTION_VERBS


def _title_like(t: str) -> bool:
    t = t.strip()
    if not t or is_bullet(t) or word_count(t) > 16 or t.endswith((".", ",", ";")) and not re.search(r"\b(19|20)\d{2}\b", t):
        return False
    if _starts_with_action(t) or not (t[0].isupper() or t[0].isdigit()):
        return False
    return True


@dataclass
class Entry:
    title: str = ""
    meta: List[str] = field(default_factory=list)
    bullets: List[str] = field(default_factory=list)
    desc: List[str] = field(default_factory=list)
    start: int = 0


def parse_entries(lines: List[str]) -> List[Entry]:
    """Group section lines into entries: title [+ meta] + bullets/description, re-joining wrapped bullets."""
    entries: List[Entry] = []
    cur: Optional[Entry] = None
    last = None
    for i, raw in enumerate(lines):
        t = raw.strip()
        if not t:
            continue
        if is_bullet(t):
            if cur is None:
                cur = Entry(start=i)
                entries.append(cur)
            cur.bullets.append(strip_bullet(t))
            last = "bullet"
            continue
        if cur is not None and last in ("bullet", "cont") and _starts_with_action(t) and cur.bullets[-1].rstrip().endswith((".", "!")):
            cur.bullets.append(t)
            last = "bullet"
            continue
        if (cur is None or (last in ("bullet", "cont") and not cur.title)) and (_starts_with_action(t) or word_count(t) >= 9 and t.endswith(".")):
            if cur is None:
                cur = Entry(start=i)
                entries.append(cur)
            cur.bullets.append(t)
            last = "bullet"
            continue
        if cur is not None and last in ("bullet", "cont"):
            prev = cur.bullets[-1]
            has_date = bool(find_date(t)[1])
            is_cont = (not prev.rstrip().endswith((".", "!", "?", ":")) or t[0].islower()) and not (has_date and word_count(t) <= 10 and _title_like(t))
            if t[0].islower() or (is_cont and not (_title_like(t) and word_count(t) <= 9 and not t.endswith("."))):
                cur.bullets[-1] = (prev + " " + t).strip()
                last = "cont"
                continue
            if t.endswith(".") and not _title_like(t):
                cur.bullets[-1] = (prev + " " + t).strip()
                last = "cont"
                continue
        if cur is not None and last in ("title", "desc"):
            if _looks_like_sentence(t) and not (find_date(t)[1] and word_count(t) <= 10):
                cur.desc.append(t)
                last = "desc"
                continue
            if last == "title" and len(cur.meta) < 2 and (find_date(t)[1] or (not cur.bullets and word_count(t) <= 9 and ("|" in t or "," in t or t[0].islower() or t.endswith(":")))):
                cur.meta.append(t)
                continue
        cur = Entry(title=t, start=i)
        entries.append(cur)
        last = "title"
    return entries


def nominal_title(sentence: str) -> str:
    """'Developed an inventory app in C# with SQL Server' -> 'Inventory App' (used only for prose-style projects without titles)."""
    t = re.sub(r"^\W*[A-Za-z\-]+\s+(?:an?|the|my|our|a\s+small|a\s+simple)?\s*", "", sentence.strip(), flags=re.I)
    t = re.split(r"\b(?:using|with|in|for|that|which|where|to|and|by|on|via|from|as)\b|[,.;]", t, maxsplit=1)[0].strip()
    t = " ".join(t.split()[:6])
    return t.title() if t else clip(sentence, 40)


# ── technologies / methods ──────────────────────────────────────────────────────────────────────
_STOPCAP = {"The", "A", "An", "And", "With", "Using", "In", "On", "For", "To", "Of", "Real", "Built", "Trained", "Developed", "Implemented", "Applied", "Demonstrated", "Engineered", "Automated"}


def tech_refs(text: str, where: str) -> List[TechRef]:
    out, seen = [], set()
    for key, a, b, alias in K.find_mentions(text):
        c = K.get(key)
        if not c or key in seen:
            continue
        seen.add(key)
        out.append(TechRef(key, text[a:b].strip() or c.name, where, clip(text[max(0, a - 30): b + 30], 80)))
    for m in re.finditer(r"\b(?:using|with|via|through|in|on)\s+((?:[A-Z][A-Za-z0-9]*[A-Z0-9a-z]*(?:\.[a-z]+)?)(?:\s+[A-Z][A-Za-z0-9]+)?)", text):
        nm = m.group(1).strip()
        if nm in _STOPCAP or len(nm) < 3 or K.find_mentions(nm) or nm.lower() in NATURAL_LANGS:
            continue
        if re.search(r"[A-Z].*[A-Z]|\d|\.", nm) or nm[0].isupper() and len(nm.split()) == 1 and not nm.isupper():
            k = "custom:" + re.sub(r"\W+", "_", nm.lower())
            if k not in seen:
                seen.add(k)
                out.append(TechRef(k, nm, where, clip(text, 80)))
    return out


def build_project(e: Entry, section_lines: List[str]) -> Optional[Project]:
    name, sub, dates = split_title(e.title)
    if not name:
        return None
    if not dates:
        for m in e.meta:
            _, d = find_date(m)
            if d:
                dates = d
                break
    bullets = list(e.bullets)
    desc = " ".join(e.desc + [m for m in e.meta if not find_date(m)[1] or word_count(m) > 6])
    body = " ".join(bullets) + " " + desc
    p = Project(title=e.title.strip(), short_name=name, subtitle=sub, dates=dates, description=(desc + " " + " ".join(bullets)).strip(), bullets=bullets)
    techs: Dict[str, TechRef] = {}
    for t in tech_refs(name + " " + sub, "title"):
        techs.setdefault(t.key, t)
    for b in bullets + ([desc] if desc else []):
        for t in tech_refs(b, "bullet"):
            if t.key in techs:
                if t.name.lower() not in techs[t.key].name.lower():
                    techs[t.key].name += " / " + t.name
            else:
                techs[t.key] = t
    for k, t in list(techs.items()):
        c = K.get(k)
        if c and c.kind != "technology":                 # concepts (ML, CV, GenAI…) are methods, not tools
            p.methods.append(c.name)
            del techs[k]
    p.technologies = list(techs.values())
    low = body.lower()
    for tech in _TECHNIQUES:
        if tech in low and tech.title() not in p.methods:
            p.methods.append(tech)
    verbs = [m.group(1).lower() for b in bullets for m in [re.match(r"^\W*([A-Za-z\-]+)", b)] if m and m.group(1).lower() in _ACTION_VERBS]
    p.ownership_verbs = verbs
    if _TEAM_CUES.search(body):
        p.ownership = "team"
    elif verbs:
        p.ownership = "individual"
    p.results = [clip(x, 140) for x in bullets if _METRIC.search(x)][:3]
    imp: Dict[str, float] = {}
    for t in p.technologies:
        if t.key.startswith("custom:"):
            continue
        for k2, w in K.closure(t.key).items():
            imp[k2] = max(imp.get(k2, 0.0), round(w, 2))
    for mname in p.methods:
        for k2, _a, _b, _al in K.find_mentions(mname):
            imp.setdefault(k2, 0.9)
    p.implied = {k: w for k, w in imp.items() if k not in {t.key for t in p.technologies}}
    p.evidence = Evidence("projects", clip(e.title + " | " + body, 400), [e.start])
    return p


def valid_project(p: Project) -> Tuple[bool, str]:
    nm = p.short_name.strip()
    if len(nm) < 3 or not re.search(r"[A-Za-z]{3}", nm):
        return False, "no usable title"
    if nm.lower() in NATURAL_LANGS or S.stem_section(nm) in ("languages", "interests", "extracurricular", "references", "contact"):
        return False, "title is a section/language word"
    if not p.bullets and len(p.description.split()) < 6:
        return False, "no description or bullets"
    ms = K.find_mentions(nm)
    if ms and len(ms) == 1 and ms[0][1] <= 1 and ms[0][2] >= len(nm) - 1 and len(p.description.split()) < 12:
        return False, "title is just a technology name"
    return True, ""


# ── skills ──────────────────────────────────────────────────────────────────────────────────────
def _split_items(s: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if ch in ",;|•·" and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip(" .\t") for x in out if x.strip(" .\t")]


def _cat_for(key: str, hint: str = "") -> str:
    c = K.get(key)
    if c:
        cat = c.category
        return cat if cat in SKILL_CATEGORIES else "other"
    return hint if hint in SKILL_CATEGORIES else "other"


def parse_skill_lines(lines: List[str], skills: Dict[str, List[SkillEntry]], spoken: List[Dict[str, str]]) -> None:
    seen = {(s.key, s.name.lower()) for lst in skills.values() for s in lst}

    def add(key: str, name: str, cat: str):
        if (key, name.lower()) in seen or (not key.startswith("custom:") and any(s.key == key for lst in skills.values() for s in lst)):
            return
        seen.add((key, name.lower()))
        skills[cat].append(SkillEntry(key=key, name=name, category=cat, listed=True))
    for ln in lines:
        t = strip_bullet(ln)
        label, body = "", t
        m = re.match(r"^([A-Za-z0-9 &/+\-]{2,40}):\s*(.+)$", t)
        if m:
            label, body = m.group(1), m.group(2)
        hint = next((c for rx, c in _LABEL_HINT if re.search(rx, label.lower())), "")
        if "&" in label and "data" in label.lower():
            hint = ""
        for item in _split_items(body):
            base = re.sub(r"\(.*?\)", "", item).strip()
            if not base:
                continue
            if base.lower() in NATURAL_LANGS:
                if not any(x["language"].lower() == base.lower() for x in spoken):
                    spoken.append({"language": base.title(), "level": ""})
                continue
            key = K.lookup_item(item)
            if key:
                c = K.get(key)
                add(key, base if len(base) <= 40 else c.name, _cat_for(key, hint))
                continue
            if "/" in base:
                parts = [x.strip() for x in base.split("/") if x.strip()]
                keys = [K.lookup_item(x) for x in parts]
                if keys and all(keys):
                    for x, k in zip(parts, keys):
                        add(k, x, _cat_for(k, hint))
                    continue
            ms = K.find_mentions(base)
            covered = sum(b - a for _k, a, b, _al in ms)
            if ms and covered >= 0.6 * len(re.sub(r"\s", "", base)):
                for n_, (k, a, b, al) in enumerate(ms):
                    add(k, base if n_ == 0 else base[a:b], _cat_for(k, hint))
                continue
            if word_count(base) <= 4 and not re.search(r"[.!?]$", base) and len(base) <= 40:
                add("custom:" + re.sub(r"\W+", "_", base.lower()), base, hint or "other")


# ── education / certifications / experience ────────────────────────────────────────────────────
def parse_education(lines: List[str]) -> List[Education]:
    out: List[Education] = []
    cur: Optional[Education] = None
    pending: List[str] = []
    for raw in lines:
        t = strip_bullet(raw).strip()
        if not t:
            continue
        g = _GPA_RX.search(t)
        parts = [p.strip() for p in re.split(r"\s{2,}|\s\|\s", t) if p.strip()]
        dm = None
        rest_parts = []
        for p in parts:
            pp = _GPA_RX.sub("", p).strip(" |,;-–")
            r2, d = find_date(pp)
            if d and not r2:
                dm = dm or d
            else:
                if d:
                    dm = dm or d
                    pp = r2
                if pp:
                    rest_parts.append(pp)
        ds = _DEGREE_START.match(rest_parts[0]) if rest_parts else None
        if ds:
            cur = Education(degree=re.sub(r"\s+", " ", rest_parts[0][: ds.start(2)].strip(" -–—:,(")).strip() or rest_parts[0],
                            field_of_study=re.sub(r"\s*\bstudent\b", "", ds.group(2) or "", flags=re.I).strip(" ,-()"), evidence=Evidence("education", clip(t, 160)))
            if ds.group(2) and "(" in ds.group(2):
                cur.field_of_study = re.sub(r"[()]", "", ds.group(2)).strip(" ,-")
            code = _DEGREE_CODES.get(re.sub(r"[^a-z]", "", rest_parts[0].split(",")[0].lower()))
            if code:
                cur.degree = code[0]
                if not (cur.field_of_study and not _INST_RX.search(cur.field_of_study) and "," not in cur.field_of_study):
                    pass
                inst_part = ds.group(2) or ""
                cur.field_of_study = code[1]
                if inst_part and not cur.institution:
                    cur.institution = inst_part.strip(" ,")
            if "," in cur.field_of_study and not cur.institution and not code:
                left, right = [x.strip() for x in cur.field_of_study.split(",", 1)]
                if right:
                    cur.field_of_study, cur.institution = left, right
            if cur.field_of_study and _INST_RX.search(cur.field_of_study) and not cur.institution:
                inst = cur.field_of_study
                cur.field_of_study = code[1] if code else ""
                cur.institution = inst.strip(" ,")
            if pending:
                cur.institution = pending.pop(0)
            for extra in rest_parts[1:]:
                if not cur.institution:
                    cur.institution = extra
            out.append(cur)
        elif cur is not None and rest_parts:
            if not cur.institution:
                cur.institution = rest_parts[0]
                for extra in rest_parts[1:]:
                    pass
            else:
                pending.append(rest_parts[0])
        elif rest_parts:
            pending.append(rest_parts[0])
        if cur is not None:
            if g and not cur.gpa:
                cur.gpa = g.group(1)
            if dm and not cur.dates:
                cur.dates = dm
    for e in out:
        e.status = "in_progress" if _PRESENT.search(e.dates or "") else ("completed" if e.dates else "")
        e.degree = re.sub(r"\s*\bstudent\b", "", e.degree, flags=re.I).strip()
    return out


def parse_certifications(lines: List[str]) -> List[Certification]:
    out: List[Certification] = []
    for raw in lines:
        t = strip_bullet(raw).strip()
        if not t:
            continue
        t, d = find_date(t)
        t = t.strip(" ,;-–—")
        issuer, name = "", t
        m = re.match(r"^([A-Z][\w &.]{2,30}):\s*(.+)$", t)
        if m:
            issuer, name = m.group(1), m.group(2)
        else:
            parts = re.split(r"\s[–—-]\s|\s(?:from|by|at)\s", t, maxsplit=1)
            if len(parts) == 2:
                name, issuer = parts[0].strip(), parts[1].strip()
        names = [n.strip() for n in name.split("|")] if "|" in name else [name]
        for n in names:
            if not n:
                continue
            kind = "certificate"
            if re.search(r"virtual|forage|job simulation", t, re.I) and re.search(r"intern", t, re.I):
                kind = "virtual_internship"
            elif re.search(r"course|training|bootcamp|workshop", t, re.I):
                kind = "course"
            out.append(Certification(n, issuer, d, kind, Evidence("certifications", clip(t, 160))))
    return out


def parse_experience(lines: List[str]) -> List[Experience]:
    out: List[Experience] = []
    for e in parse_entries(lines):
        if not e.title:
            continue
        head = "  ".join([e.title] + e.meta)
        head, dates = find_date(head)
        if not dates:
            for m in e.meta:
                dates = find_date(m)[1] or dates
        parts = [p.strip() for p in re.split(r"\s{2,}|\s\|\s|\s[—–]\s|\s(?:at|@)\s|,\s", head) if p.strip()]
        role = next((p for p in parts if re.search(S._ROLE.pattern, p, re.I)), parts[0] if parts else "")
        org = next((p for p in parts if p != role and not find_date(p)[1]), "")
        kind = "job"
        low = head.lower()
        if "virtual" in low or "forage" in low:
            kind = "virtual_internship"
        elif "intern" in low:
            kind = "internship"
        elif "freelanc" in low:
            kind = "freelance"
        elif re.search(r"volunteer|member", low):
            kind = "volunteer"
        techs = {}
        for b in e.bullets + e.desc:
            for t in tech_refs(b, "bullet"):
                techs.setdefault(t.key, t)
        out.append(Experience(role, org, dates, kind, e.bullets + e.desc, list(techs.values()), Evidence("experience", clip(head + " " + " ".join(e.bullets), 300))))
    return out


# ── name / headline ────────────────────────────────────────────────────────────────────────────
def _split_squashed(sq: str, contact: Contact) -> str:
    low = sq.lower()
    hints = []
    for l in contact.links:
        hints.append(l.rstrip("/").split("/")[-1])
    for e in contact.emails:
        hints.append(e.split("@")[0])
    for h in hints:
        parts = [p for p in re.split(r"[-_.]", h) if p.isalpha()]
        if len(parts) >= 2 and "".join(parts).lower() == low:
            return " ".join(p.title() for p in parts)
    for e in contact.emails:
        loc = re.sub(r"[^a-z]", "", e.split("@")[0].lower())
        for n in range(len(loc), 2, -1):
            seg = loc[-n:] if loc[-n:] and low.startswith(loc[-n:]) else ""
            if seg and len(low) > len(seg) + 2 and low.startswith(seg):
                return seg.title() + " " + low[len(seg):].title()
    return sq.title()


def find_name(header_lines: List[D.Line], contact: Contact, filename: str) -> Tuple[str, str]:
    for ln in header_lines[:5]:
        t = ln.text.strip()
        letters = re.sub(r"[^A-Za-z ]", "", t).strip()
        if not letters or re.search(r"\d|@|\||/", t) or S.stem_section(t) or K.find_mentions(t) or len(t) > 45:
            continue
        toks = letters.split()
        if ln.letterspaced and len(toks) == 1:
            return _split_squashed(toks[0], contact), t
        if 2 <= len(toks) <= 4 and all(x[0].isalpha() for x in toks):
            return " ".join(w.title() if w.isupper() else w for w in toks), t
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    stem = re.sub(r"\[.*?\]|\(.*?\)|\d+|[_\-.]+", " ", stem)
    stem = re.sub(r"(?i)\b(resume|cv|curriculum|vitae|final|updated|new|ai|ml)\b", " ", stem)
    toks = [p for p in stem.split() if p.isalpha()]
    if toks:
        return " ".join(p.capitalize() for p in toks[:3]), ""
    for e in contact.emails:
        loc = re.sub(r"[^a-z]", "", e.split("@")[0].lower())
        if len(loc) >= 3:
            return loc.title(), ""
    return "", ""


# ── main ───────────────────────────────────────────────────────────────────────────────────────
def extract_resume(path_or_text: str, is_text: bool = False, filename: str = "") -> Resume:
    rd = D.read_document(path_or_text, is_text)
    r = Resume(layout=rd.layout, parse_method=rd.method, warnings=list(rd.warnings), raw_text=rd.raw_text)
    r.contact = Contact(rd.contact.emails, rd.contact.phones, rd.contact.links, rd.contact.locations)
    if not rd.lines:
        r.warnings.append("empty document")
        return r
    secs, w = S.segment(rd.lines)
    r.warnings += w
    r.sections_found = [s.name for s in secs if s.name != "header" and s.lines]
    by: Dict[str, List[S.Section]] = {}
    for s in secs:
        by.setdefault(s.name, []).append(s)
    texts = lambda name: [l.text for s in by.get(name, []) for l in s.lines]      # noqa: E731
    header = [l for s in by.get("header", []) for l in s.lines]
    r.name, name_line = find_name(header, r.contact, filename or (path_or_text if not is_text else ""))
    r.first_name = r.name.split()[0] if r.name else ""
    for l in header:
        if l.text.strip() != name_line.strip() and ("|" in l.text or l.text.isupper()) and not S.stem_section(l.text):
            r.headline = l.text.strip()
            break
    # summary
    summ = texts("summary")
    if not summ:
        summ = [l.text for l in header if word_count(l.text) >= 14]
    r.summary = " ".join(summ).strip()
    # skills
    spoken: List[Dict[str, str]] = []
    skills = {c: [] for c in SKILL_CATEGORIES}
    parse_skill_lines(texts("skills"), skills, spoken)
    r.skills = skills
    # languages (spoken)
    for ln in texts("languages"):
        t = strip_bullet(ln)
        for item in _split_items(t):
            m = re.match(r"^([A-Za-z]+)\s*(?:[—–:\-(]\s*(.+?)\)?)?$", item.strip())
            if m and m.group(1).lower() in NATURAL_LANGS and not any(x["language"].lower() == m.group(1).lower() for x in spoken):
                spoken.append({"language": m.group(1).title(), "level": (m.group(2) or "").strip()})
            elif m and m.group(1).lower() in NATURAL_LANGS:
                for x in spoken:
                    if x["language"].lower() == m.group(1).lower() and not x["level"]:
                        x["level"] = (m.group(2) or "").strip()
    r.spoken_languages = spoken
    # education
    r.education = parse_education(texts("education"))
    # projects
    plines = texts("projects")
    cand = [build_project(e, plines) for e in parse_entries(plines) if e.title or e.bullets]
    headless = [e for e in parse_entries(plines) if not e.title and e.bullets]
    if headless and not any(e.title for e in parse_entries(plines)):
        cand = []
        for e in headless:
            for b in e.bullets:
                nm = re.split(r"[:–—-]\s", b, maxsplit=1)[0]
                nm = nm if word_count(nm) <= 8 and len(nm) < len(b) else nominal_title(b)
                cand.append(build_project(Entry(title=nm, bullets=[b]), plines))
    for p in cand:
        if p is None:
            continue
        ok, why = valid_project(p)
        if ok:
            r.projects.append(p)
        else:
            r.warnings.append(f"rejected project-like entry {p.short_name!r}: {why}")
    # experience (+ anything in 'experience' that is really a project is left alone; no guessing)
    r.experience = parse_experience(texts("experience"))
    # certifications
    r.certifications = parse_certifications(texts("certifications"))
    for c in r.certifications:
        if c.kind == "virtual_internship":
            r.warnings.append(f"'{c.name}' is listed under certifications and is a virtual programme, not employment")
    r.achievements = [strip_bullet(x) for x in texts("achievements") if strip_bullet(x)]
    r.publications = [strip_bullet(x) for x in texts("publications") if strip_bullet(x)]
    r.coursework = [it for x in texts("coursework") for it in _split_items(strip_bullet(x))]
    r.extracurricular = [strip_bullet(x) for x in texts("extracurricular") if strip_bullet(x)]
    r.interests = [it for x in texts("interests") for it in _split_items(strip_bullet(x))]
    # skill provenance: which listed skills are actually used in a project / experience
    for sk in r.all_skills():
        for p in r.projects:
            if sk.key in p.tech_keys() or p.implied.get(sk.key, 0) >= 0.6 or sk.key in [K.lookup_item(m) for m in p.methods]:
                sk.in_projects.append(p.short_name)
        for ex in r.experience:
            if sk.key in [t.key for t in ex.technologies]:
                sk.in_projects.append(ex.title or ex.organization)
    # no skills section: infer from evidence-bearing sections, marked as not "listed"
    if not r.all_skills():
        r.warnings.append("no skills section found; skills inferred from projects/experience only (marked listed=False)")
        for p in r.projects:
            for t in p.technologies:
                if not t.key.startswith("custom:") and t.key not in r.skill_keys():
                    r.skills[_cat_for(t.key)].append(SkillEntry(t.key, t.name, _cat_for(t.key), False, [p.short_name], "project"))
    if not r.projects and not r.experience:
        r.warnings.append("no project or experience entries could be extracted")
    return r
