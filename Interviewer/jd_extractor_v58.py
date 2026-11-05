"""
jd_extractor_v58.py — semantic job-description extraction.

V51 failure it fixes: company boilerplate ("started as an R&D & Accounting back-office", "Jonas Club ERP") produced
*required* competencies "Accounting" and "ERP". V58 first decides what each JD section IS (requirements, preferred,
responsibilities, role context, company/legal boilerplate), and only mines requirement-bearing sections. Mentions found in
role-context sections become `domain_context_only` requirements (never validated as skills); company text is ignored.

Per requirement it records: name, canonical_name, category, required/preferred, importance (1–3), expected_depth
(awareness | implementation | depth, from wording such as "exposure", "good knowledge", "strong"), evidence_expected
(the facets that count as real evidence, from the concept registry), synonyms and the source text.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import concepts_v58 as K
import doc_reader_v58 as D
from schema_v58 import JobProfile, Requirement
from textkit_v58 import clip, split_sentences, word_count

_SEC_RULES: List[Tuple[str, str]] = [
    ("preferred", r"preferred|nice to have|bonus|good to have|desirable|plus\b|advantage"),
    ("requirements", r"required|requirement|qualification|what you.?ll need|what you bring|must have|skills|experience you|who you are|you have|essential|minimum"),
    ("competencies", r"competenc|behaviou?r|attributes|personal qualities"),
    ("responsibilities", r"responsibilit|duties|what you.?ll do|you will|day to day|key tasks|the role"),
    ("role_context", r"the position|about the role|position summary|job summary|overview of the role|role overview|about the position|the opportunity|job description|the job"),
    ("ignore", r"about (us|the company)|the division|company|who we are|our (story|culture|values)|benefits|perks|what we offer|disclaimer|equal opportunity|diversity|inclusion|accommodation|how to apply|why join|our mission"),
]
_BEHAV: List[Tuple[str, str, str]] = [
    ("analytical_thinking", "Analytical thinking", r"analytical|analysis skills|problem analysis"),
    ("problem_solving", "Problem solving", r"problem[- ]solving|solve problems|troubleshoot"),
    ("communication", "Communication", r"communication|communicat"),
    ("teamwork", "Teamwork", r"team ?work|team player|collaborat|work(?:ing)? (?:well )?in a team"),
    ("learning_ability", "Learning ability", r"quick learner|fast learner|eager to learn|willingness to learn|learn(?:s)? quickly|learning"),
    ("attention_to_detail", "Attention to detail", r"attention to detail|detail[- ]oriented"),
    ("planning", "Planning & meeting deadlines", r"deadline|planning and organi[sz]ing|time management|organi[sz]ation"),
    ("independence", "Working independently", r"independent|self[- ]motivated|self[- ]starter|initiative"),
    ("decision_making", "Decision making", r"decision[- ]making"),
    ("adaptability", "Adaptability", r"diverse range|adaptab|flexib|passionate to work on|new technolog|tech stacks"),
    ("drive", "Drive & attitude", r"can[- ]do|energetic|passionate|motivated|commitment"),
]
_PHRASES: List[Tuple[str, str]] = [        # phrases the alias table cannot see
    (r"data collection|management,? and analysis|data analysis", "data_analysis"),
    (r"software development (?:life ?cycle|lifecycle)|\bsdlc\b", "sdlc"),
    (r"object[- ]oriented", "oop"),
    (r"application types|architecture", "architecture"),
    (r"professional software engineering|engineering practices", "se_practices"),
    (r"restful?\s+api|rest api", "rest_api"),
    (r"hands-on ai|ai experience", "ai_projects"),
    (r"ai/ml libraries|ai-powered application|ai powered application", None),
]
_DEPTH_STRONG = re.compile(r"\b(strong|expert|expertise|proficient|proficiency|advanced|in-depth|deep|solid|extensive|must have|must)\b", re.I)
_DEPTH_AWARE = re.compile(r"\b(exposure|familiar(?:ity)?|aware(?:ness)?|basic|understanding of|acquainted|introductory|some)\b", re.I)
_DEPTH_IMPL = re.compile(r"\b(good|working|hands-on|practical|knowledge of|experience (?:with|in)|proven|competent|comfortable)\b", re.I)
_PREF_CUE = re.compile(r"\b(preferred|is a plus|nice to have|bonus|desirable|advantage|an asset)\b", re.I)


def _classify_heading(h: str) -> Optional[str]:
    t = h.lower().strip(" :-–")
    if len(t) > 60 or word_count(t) > 8 or t.endswith("."):
        return None
    for name, rx in _SEC_RULES:
        if re.search(rx, t):
            return name
    return None


def _split(text: str) -> List[Tuple[str, str, List[str]]]:
    """-> [(section_kind, heading, lines)]"""
    out: List[Tuple[str, str, List[str]]] = []
    cur = ("title", "", [])
    out.append(cur)
    for raw in text.replace("\r", "").split("\n"):
        t = raw.strip()
        if not t:
            continue
        t2 = re.sub(r"^[•\-–*▪●\d.)\s]+", "", t).strip()
        mi = re.match(r"^(disclaimer|equal opportunity(?: employer)?|about us|benefits|notice|eeo)\s*:\s*(.*)$", t2, re.I)
        if mi:
            cur = ("ignore", mi.group(1), [mi.group(2)] if mi.group(2) else [])
            out.append(cur)
            continue
        kind = _classify_heading(t2) if (len(t2) < 60 and not t2.endswith(".") and word_count(t2) <= 8) else None
        looks_heading = kind and (t2[:1].isupper() or t2.isupper()) and (t.endswith(":") or not re.search(r"[.,;]$", t2))
        if looks_heading and not re.match(r"^[•\-–*▪●]", t):
            cur = (kind, t2, [])
            out.append(cur)
        else:
            cur[2].append(t2)
    return out


def _depth_and_importance(line: str, section: str) -> Tuple[str, int, bool]:
    required = section != "preferred" and not _PREF_CUE.search(line)
    if _DEPTH_STRONG.search(line):
        depth, imp = "depth", 3
    elif _DEPTH_AWARE.search(line):
        depth, imp = "awareness", 2
    elif _DEPTH_IMPL.search(line):
        depth, imp = "implementation", 2
    else:
        depth, imp = "implementation", 2
    if not required:
        imp = max(1, imp - 1)
    return depth, imp, required


def _mk(key: str, source: str, section: str, depth: str, imp: int, required: bool, domain: bool = False) -> Requirement:
    c = K.get(key)
    return Requirement(key=key, name=c.name if c else key, canonical_name=c.name if c else key, category=c.category if c else "other", required=required, importance=imp,
                       expected_depth=depth, evidence_expected=c.expected_evidence() if c else [], synonyms=[a for a in (c.aliases if c else []) if a != c.name.lower()][:8],
                       source_text=clip(source, 200), source_section=section, kind=c.kind if c else "technology", domain_context_only=domain)


def detect_seniority(title: str, text: str) -> str:
    t = (title + " " + text[:600]).lower()
    if re.search(r"intern|trainee|graduate|entry|junior|fresh|apprentice|new/recent", t):
        return "junior"
    if re.search(r"senior|lead|principal|staff|architect|head of|manager", t):
        return "senior"
    return "mid"


def extract_jd(path_or_text: str, is_text: bool = False, company: str = "", role_key: str = "") -> JobProfile:
    if is_text:
        text = path_or_text
    else:
        ext = os.path.splitext(path_or_text)[1].lower()
        if ext == ".pdf" or ext == ".docx":
            rd = D.read_document(path_or_text)
            text = "\n".join(l.text for l in rd.lines)
        else:
            text = open(path_or_text, encoding="utf-8", errors="replace").read()
    jp = JobProfile(company=company, raw_text=text)
    secs = _split(text)
    jp.ignored_sections = [h for k, h, _ in secs if k == "ignore"]
    m = re.search(r"(?:hire|seeking|looking for|recruit(?:ing)?)\s+(?:an?\s+|the\s+)?([A-Z][A-Za-z/&+\- ]{3,50}?)(?:\s+to\b|\s+who\b|\s+for\b|[.,]|\s+with\b)", text)
    jp.title = role_key.strip() or (m.group(1).strip() if m else "")
    jp.seniority = detect_seniority(jp.title, text)
    seen_req: Dict[str, Requirement] = {}

    def add(req: Requirement):
        old = seen_req.get(req.key)
        if old is None:
            seen_req[req.key] = req
            return
        if old.domain_context_only and not req.domain_context_only:
            seen_req[req.key] = req
            return
        old.importance = max(old.importance, req.importance)
        old.required = old.required or req.required
        old.expected_depth = max([old.expected_depth, req.expected_depth], key=["awareness", "implementation", "depth"].index)
        if req.source_text not in old.source_text:
            old.source_text = clip(old.source_text + " | " + req.source_text, 260)

    for kind, head, lines in secs:
        if kind in ("ignore", "title"):
            for ln in lines:
                for key, a, b, al in K.find_mentions(ln):
                    c = K.get(key)
                    if c and c.kind == "domain" and c.name not in jp.company_context:
                        jp.company_context.append(c.name)
            continue
        for ln in lines:
            for sent in (split_sentences(ln) or [ln]):
                if kind == "role_context":
                    for key, a, b, al in K.find_mentions(sent):
                        c = K.get(key)
                        if c and c.kind == "domain":
                            add(_mk(key, sent, kind, "awareness", 1, False, domain=True))
                    if re.search(r"progress|train|apprentice|work(ing)? (as|on|with)|member of", sent, re.I) and len(jp.responsibilities) < 6:
                        jp.responsibilities.append(clip(sent, 220))
                    continue
                if kind == "responsibilities":
                    jp.responsibilities.append(clip(sent, 220))
                if re.search(r"\b(degree|graduate|graduates|bachelor|university)\b", sent, re.I) and not K.find_mentions(sent, False):
                    jp.education_required = jp.education_required or clip(sent, 200)
                    continue
                section_for_depth = "preferred" if kind == "preferred" else kind
                depth, imp, required = _depth_and_importance(sent, section_for_depth)
                found = {}          # ordered set (insertion order = order of mention): iteration must not depend on hash seeds
                for key, a, b, al in K.find_mentions(sent):
                    c = K.get(key)
                    if not c:
                        continue
                    if c.kind == "domain":
                        add(_mk(key, sent, kind, "awareness", 1, False, domain=True))
                        continue
                    found[key] = True
                for rx, key in _PHRASES:
                    if key and re.search(rx, sent, re.I):
                        found[key] = True
                if "data_analysis" in found:
                    found.pop("data_prep", None)           # 'data collection, management and analysis' is one analytical competency
                if re.search(r"ai/ml libraries|ml libraries|libraries and frameworks", sent, re.I):
                    found["ml_libs"] = True
                # AI-powered application development is evidence of GenAI exposure only when the sentence names it
                strong_here = bool(re.search(r"\bstrong\s+python\b", sent, re.I))
                for key in found:
                    d, i = depth, imp
                    # depth cue applies to the nearest named skill: "Strong Python ... familiarity with libraries"
                    if key == "python" and strong_here:
                        d, i = "depth", 3
                    elif key in ("ml_libs",) and strong_here:
                        d, i = "awareness", 2
                    elif key in ("oop",) and re.search(r"good programming knowledge", sent, re.I):
                        d, i = "implementation", 3
                    if kind == "competencies":
                        i = 1                       # generic "Key Competencies" bullets are weak evidence of a technical requirement
                    add(_mk(key, sent, kind, d, i, required))
                if kind in ("requirements", "preferred", "competencies"):
                    for bkey, bname, rx in _BEHAV:
                        if re.search(rx, sent, re.I):
                            jp.behavioural.setdefault(bkey, Requirement(key="soft:" + bkey, name=bname, canonical_name=bname, category="behavioural", required=True, importance=2,
                                                                         expected_depth="implementation", evidence_expected=["situation", "action", "result"], source_text=clip(sent, 160),
                                                                         source_section=kind, kind="behavioural"))
                    if not found and not any(re.search(rx, sent, re.I) for _k, _n, rx in _BEHAV) and word_count(sent) >= 4:
                        jp.unresolved_lines.append(clip(sent, 160))
    jp.requirements = seen_req
    if not jp.requirements:
        jp.warnings.append("no technical requirements recognised in requirement sections; check the JD text")
    if not jp.title:
        jp.warnings.append("job title not found; pass --role-key")
    return jp
