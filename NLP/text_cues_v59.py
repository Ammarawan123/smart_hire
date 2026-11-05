"""
text_cues_v59.py — transcript-derived cues for the NLP side of SmartHire (content channel + coaching metrics).

EVIDENCE POSITION (see RESEARCH_V59.md)
  * detail richness / specificity / consistency  -> CONTENT cues. The verbal-cue literature (Caso et al. 2023; Vrij et al. 2022) finds these are
    mostly cues to TRUTHFULNESS; they feed the fusion 'content' group.
  * fillers, hedges, pace, first-person ownership   -> COACHING metrics (Student mode). They are NOT used as deception evidence:
    Sporer & Schwandt found no reliable deception link; Velutharambath & Klinger (ACL 2025) found text-style deception detectors near chance.
  * All counters are transparent regex/lexicon heuristics.  They need human-coded validation (agreement with 2 raters on ~50 answers) before you report them.

Optional NLI contradiction check: cross-encoder/nli-deberta-v3-xsmall (Apache-2.0, 87.77% MNLI-mm; label order contradiction, entailment, neutral).
Without the model the module still runs (numeric/quantity conflicts only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_WORD = re.compile(r"[A-Za-z0-9'+#./-]+")
FILLERS = re.compile(r"\b(?:um+|uh+|erm?|er|hmm+|ah+)\b", re.I)
FILLER_PHRASES = re.compile(r"\b(?:you know|i mean|kind of like|sort of like)\b", re.I)
HEDGES = re.compile(r"\b(?:maybe|perhaps|probably|i guess|i suppose|i think|sort of|kind of|something like|more or less|not sure|i believe|if i remember|roughly|around)\b", re.I)
ACTION_VERBS = re.compile(r"\b(?:built|implemented|designed|deployed|debugged|trained|wrote|developed|optimi[sz]ed|migrated|refactored|configured|integrated|tested|evaluated|fixed|created|launched|automated|measured|profiled|benchmarked)\b", re.I)
COMPLICATION = re.compile(r"\b(?:bug|failed|failure|crash(?:ed)?|timeout|leak|regression|issue|problem|unexpected(?:ly)?|surpris(?:e|ed|ing)|had to|turned out|root cause|workaround|trade-?off|didn't work|did not work|broke|slow(?:ed|er)?)\b", re.I)
CAUSAL = re.compile(r"\b(?:because|so that|therefore|which caused|as a result|due to|that's why|in order to)\b", re.I)
TEMPORAL = re.compile(r"\b(?:\d{4}|last (?:year|month|week|semester|summer)|(?:two|three|four|five|six|\d+) (?:weeks?|months?|years?|days?|hours?|sprints?)|during my (?:internship|final year|semester)|at that time|first (?:week|month))\b", re.I)
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:%|ms|s|sec|seconds?|minutes?|hours?|gb|mb|kb|k|x|users?|requests?|rows?|samples?|images?|classes|epochs?|layers?|fps|accuracy|f1|auc)?\b", re.I)
FIRST_SG = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|my|me|myself)\b", re.I)
FIRST_PL = re.compile(r"\b(?:we|we're|we've|our|us|the team|my team|teammates?)\b", re.I)

_UNIT_M = {"year": 12.0, "years": 12.0, "yr": 12.0, "yrs": 12.0, "month": 1.0, "months": 1.0, "week": 0.23, "weeks": 0.23}
_NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a": 1, "half": 0.5}


def has_term(text: str, term: str) -> bool:
    """Whole-term match (so the language 'C' or 'R' does not match every sentence containing that letter)."""
    if not term:
        return False
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9+#])", text or "", re.I) is not None


def words(text: str) -> List[str]:
    return _WORD.findall(text or "")


def sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if len(s.strip()) > 2]


@dataclass
class TextCues:
    words: int = 0
    filler_rate_per_100w: float = 0.0
    hedge_rate_per_100w: float = 0.0
    first_person_singular_share: float = 0.0      # I/my vs we/our: ownership signal for coaching
    detail_items: Dict[str, List[str]] = field(default_factory=dict)
    detail_count: int = 0
    detail_density: float = 0.0                   # distinct detail items per 100 words (fusion cue)
    explanation_markers: int = 0
    complication_mentions: int = 0
    usable: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def text_cues(text: str, tech_mentions: Optional[Sequence[str]] = None, min_words: int = 12) -> TextCues:
    """`tech_mentions`: concept names already found in the answer (pass V58's K.find_mentions aliases) so technologies count as details."""
    t = text or ""
    ws = words(t)
    n = len(ws)
    c = TextCues(words=n)
    if n == 0:
        return c
    per100 = 100.0 / n
    c.filler_rate_per_100w = (len(FILLERS.findall(t)) + len(FILLER_PHRASES.findall(t))) * per100
    c.hedge_rate_per_100w = len(HEDGES.findall(t)) * per100
    sg, pl = len(FIRST_SG.findall(t)), len(FIRST_PL.findall(t))
    c.first_person_singular_share = sg / (sg + pl) if (sg + pl) else 0.0
    items = {
        "numbers": sorted({m.group(0).strip() for m in NUMBER.finditer(t) if re.search(r"\d", m.group(0))}),
        "technologies": sorted({x.lower() for x in (tech_mentions or [])}),
        "time_anchors": sorted({m.group(0).lower() for m in TEMPORAL.finditer(t)}),
        "actions": sorted({m.group(0).lower() for m in ACTION_VERBS.finditer(t)}),
        "complications": sorted({m.group(0).lower() for m in COMPLICATION.finditer(t)}),
        "reasons": sorted({m.group(0).lower() for m in CAUSAL.finditer(t)}),
    }
    c.detail_items = {k: v for k, v in items.items() if v}
    c.detail_count = sum(len(v) for v in items.values())
    c.detail_density = c.detail_count * per100
    c.explanation_markers = len(items["reasons"])
    c.complication_mentions = len(items["complications"])
    c.usable = n >= min_words
    return c


def claim_detail_gap(claimed_strength: float, cues: TextCues) -> Optional[float]:
    """Gap between how deep the candidate/resume CLAIMS to be (0..1, e.g. V58 Claim.strength) and how much detail was actually given.
    Positive = claimed more than demonstrated.  Detail is mapped to 0..1 with a saturating curve (8 details / 100 words ~ 0.8)."""
    if not cues.usable:
        return None
    shown = 1.0 - 1.0 / (1.0 + cues.detail_density / 4.0)
    return float(claimed_strength - shown)


# ───────────────────────── quantity conflicts (rule-based, no model) ─────────────────────────
_TOK = r"[A-Za-z][A-Za-z0-9+#.\-]*"
_QTY = re.compile(
    r"(?:(?P<n1>\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|half a|a)\s*\+?\s*(?P<u1>years?|yrs?|months?|weeks?)\s+(?:of\s+)?(?:experience\s+(?:with|in)\s+)?(?P<s1>" + _TOK + r"(?:\s+" + _TOK + r"){0,2}))"
    r"|(?:(?<![A-Za-z0-9])(?P<s2>" + _TOK + r")\s+(?:for|over)\s+(?:about\s+|around\s+|almost\s+)?(?P<n2>\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|a)\s+(?P<u2>years?|yrs?|months?|weeks?))", re.I)
_STOP = {"and", "but", "in", "at", "for", "on", "with", "experience", "to", "using", "of", "so", "then", "because"}


def quantity_claims(text: str) -> List[Tuple[str, float]]:
    """[(skill_phrase_lower, months)] e.g. 'two years of PyTorch' -> ('pytorch', 24)."""
    out = []
    for m in _QTY.finditer(text or ""):
        n = m.group("n1") or m.group("n2")
        u = (m.group("u1") or m.group("u2") or "").lower()
        s = (m.group("s1") or m.group("s2") or "").strip().lower()
        toks = []
        for tk in re.split(r"[\s,;]+", s):
            if tk.lower() in _STOP:
                break
            toks.append(tk.strip(".,;"))
        s = " ".join(toks).strip()
        if not n or not s:
            continue
        try:
            val = float(_NUMW.get(n.lower(), n))
        except ValueError:
            continue
        out.append((s, val * _UNIT_M.get(u, 1.0)))
    return out


def quantity_conflict(resume_text: str, answer_text: str, ratio: float = 2.0, min_gap_months: float = 6.0) -> Tuple[float, List[str]]:
    """Return (activation 0..1, reasons). A conflict = same skill, duration differs by > `ratio` AND > `min_gap_months`."""
    rq, aq = quantity_claims(resume_text), quantity_claims(answer_text)
    reasons = []
    for rs, rm in rq:
        for as_, am in aq:
            if rs == as_ or rs in as_ or as_ in rs:
                hi, lo = max(rm, am), max(min(rm, am), 0.25)
                if hi / lo > ratio and abs(rm - am) > min_gap_months:
                    reasons.append(f"{rs}: resume/earlier ~{rm:.0f} mo vs now ~{am:.0f} mo")
    return (1.0 if reasons else 0.0), reasons


# ───────────────────────── NLI contradiction (optional model) ─────────────────────────
class NLIChecker:
    """Wrapper for cross-encoder/nli-deberta-v3-xsmall (sentence-transformers CrossEncoder).  Output order of that model: contradiction, entailment, neutral.
    Construct with `predict=callable(pairs)->array[n,3]` to inject a fake in tests or another NLI model."""

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall", predict: Optional[Callable] = None):
        self.model_name = model_name
        self._predict = predict
        self.available = predict is not None
        if predict is None:
            try:
                from sentence_transformers import CrossEncoder
                self._ce = CrossEncoder(model_name)
                self._predict = lambda pairs: self._ce.predict(pairs, apply_softmax=True)
                self.available = True
            except Exception as e:                       # model not installed / offline
                self.error = f"{type(e).__name__}: {e}"

    def contradiction_prob(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        if not self.available or not pairs:
            return [0.0] * len(pairs)
        import numpy as np
        both = list(pairs) + [(b, a) for a, b in pairs]                  # check both directions
        probs = np.asarray(self._predict(both), float)
        n = len(pairs)
        return [float(max(probs[i][0], probs[i + n][0])) for i in range(n)]


def candidate_pairs(prior_sentences: Sequence[str], answer_sentences: Sequence[str], anchors: Sequence[str], max_pairs: int = 8) -> List[Tuple[str, str]]:
    """Only sentences about the same topic are compared (keeps NLI calls to a CPU-friendly handful).  `anchors` = concept names / skills in play."""
    low = [a for a in anchors if a]
    pairs = []
    for ps in prior_sentences:
        for as_ in answer_sentences:
            shared = [a for a in low if has_term(ps, a) and has_term(as_, a)]
            if shared and len(words(as_)) >= 4:
                pairs.append((ps, as_))
    return pairs[:max_pairs]


def contradiction_evidence(prior_sentences: Sequence[str], answer_text: str, anchors: Sequence[str], nli: Optional[NLIChecker]) -> Tuple[float, List[str]]:
    """(max contradiction probability among same-topic pairs, human-readable pairs above 0.5)."""
    if nli is None or not nli.available:
        return 0.0, []
    pairs = candidate_pairs(prior_sentences, sentences(answer_text), anchors)
    probs = nli.contradiction_prob(pairs)
    shown = [f"“{a[:90]}” vs “{b[:90]}” (p={p:.2f})" for (a, b), p in zip(pairs, probs) if p >= 0.5]
    return (max(probs) if probs else 0.0), shown
