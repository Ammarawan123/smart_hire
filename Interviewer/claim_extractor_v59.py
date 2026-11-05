"""
claim_extractor_v59.py — turns a résumé (and later the candidate's answers) into CHECKABLE CLAIMS for the content channel.

Three interchangeable back-ends, all returning the same `Claim59` records, all GROUNDED (a claim survives only if its `quote` occurs in the source text,
which removes the model's invented claims):
  1. rule_claims()          deterministic regex (quantities, metrics, ownership/deployment statements). Always available; the fallback.
  2. NuExtractClient        NuMind NuExtract via llama-server.  1.x models use the '<|input|> ### Template ... <|output|>' prompt on /completion;
                            NuExtract 2.0 uses a '# Template:' user message on /v1/chat/completions (formats read from the model cards).
                            LICENCE NOTE: NuExtract-2.0-4B is under the Qwen Research License (non-commercial); 2B and 8B are MIT.
  3. qwen_claims(llm, ...)  Qwen2.5 through the V58 LocalLLM.json() interface (schema-constrained JSON).
NOT TESTED against a real NuExtract/Qwen model in this package (no model files or network here); the HTTP/parse/grounding logic is unit-tested with a stub server.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from text_cues_v59 import has_term, quantity_claims, sentences

CLAIM_TEMPLATE = {
    "claims": [{"skill_or_topic": "", "quote": "", "duration": "", "metric": "", "ownership": ["individual", "team", "unknown"], "deployed_to_production": ["yes", "no", "unknown"]}]
}


@dataclass
class Claim59:
    topic: str
    quote: str
    source: str = "resume"                    # resume | answer
    months: Optional[float] = None
    metric: str = ""
    ownership: str = "unknown"
    production: str = "unknown"
    strength: float = 0.5                      # how deep the claim sounds (0..1) -> used by claim_detail_gap
    extractor: str = "rule"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


_METRIC = re.compile(r"\b\d+(?:\.\d+)?\s?(?:%|x|ms|k\b|users?|requests?|fps|accuracy|f1|auc)", re.I)
_OWN = re.compile(r"\b(?:led|lead|owned|architected|designed|built|developed|implemented|deployed|launched|created|trained)\b", re.I)
_PROD = re.compile(r"\b(?:production|deployed|live users?|serving|in prod|real[- ]time)\b", re.I)
_TEAM = re.compile(r"\b(?:team of \d+|collaborated|worked with|contributed|assisted|helped|supported)\b", re.I)


def _strength(sentence: str, months: Optional[float], metric: str) -> float:
    s = 0.3
    if months:
        s += min(0.3, months / 60.0)
    if metric:
        s += 0.15
    if _OWN.search(sentence):
        s += 0.15
    if _PROD.search(sentence):
        s += 0.1
    if _TEAM.search(sentence):
        s -= 0.1
    return float(max(0.05, min(1.0, s)))


def rule_claims(text: str, topics: Optional[List[str]] = None, source: str = "resume") -> List[Claim59]:
    """`topics`: known concept names (e.g. V58 resume skills) used to label the claim; otherwise the quantity phrase is used."""
    out: List[Claim59] = []
    low_topics = [t.lower() for t in (topics or [])]
    for sent in sentences(text):
        qs = quantity_claims(sent)
        metric = (_METRIC.search(sent).group(0) if _METRIC.search(sent) else "")
        hit = [t for t in low_topics if has_term(sent, t)]
        if not (qs or metric or (hit and (_OWN.search(sent) or _PROD.search(sent)))):
            continue
        months = qs[0][1] if qs else None
        topic = (qs[0][0] if qs else (hit[0] if hit else "")) or "experience"
        own = "team" if _TEAM.search(sent) and not _OWN.search(sent) else ("individual" if _OWN.search(sent) else "unknown")
        out.append(Claim59(topic, sent[:240], source, months, metric, own, "yes" if _PROD.search(sent) else "unknown", _strength(sent, months, metric), "rule"))
    return out


def grounded(claims: List[Claim59], source_text: str) -> List[Claim59]:
    """Keep only claims whose quote really appears in the source (whitespace/case-insensitive)."""
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    src = norm(source_text)
    return [c for c in claims if c.quote and norm(c.quote) in src]


def _from_json(obj: dict, source: str, extractor: str) -> List[Claim59]:
    out = []
    for c in (obj or {}).get("claims", []) or []:
        quote = str(c.get("quote", "")).strip()
        topic = str(c.get("skill_or_topic", "")).strip().lower()
        if not quote or not topic:
            continue
        qs = quantity_claims(str(c.get("duration", "")) + " of " + topic) if c.get("duration") else []
        own = c.get("ownership") if c.get("ownership") in ("individual", "team") else "unknown"
        prod = c.get("deployed_to_production") if c.get("deployed_to_production") in ("yes", "no") else "unknown"
        months = qs[0][1] if qs else None
        out.append(Claim59(topic, quote, source, months, str(c.get("metric", "")), own, prod, _strength(quote, months, str(c.get("metric", ""))), extractor))
    return out


class NuExtractClient:
    """style='1.5' -> POST /completion with the '<|input|>' prompt; style='2.0' -> POST /v1/chat/completions with '# Template:'."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", style: str = "1.5", timeout: float = 60.0, post: Optional[Callable] = None):
        self.base, self.style, self.timeout = base_url.rstrip("/"), style, timeout
        self._post = post or self._http_post

    def _http_post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def extract(self, text: str, source: str = "resume", template: Optional[dict] = None) -> List[Claim59]:
        tpl = json.dumps(template or CLAIM_TEMPLATE, indent=4)
        try:
            if self.style == "2.0":
                resp = self._post("/v1/chat/completions", {"messages": [{"role": "user", "content": f"# Template:\n{tpl}\n{text}"}], "temperature": 0.0, "max_tokens": 700})
                raw = resp["choices"][0]["message"]["content"]
            else:
                prompt = f"<|input|>\n### Template:\n{tpl}\n### Text:\n{text}\n\n<|output|>"
                resp = self._post("/completion", {"prompt": prompt, "temperature": 0.0, "n_predict": 700, "stop": ["<|end-output|>", "<|input|>"]})
                raw = resp.get("content", "")
            m = re.search(r"\{.*\}", raw, re.S)
            obj = json.loads(m.group(0)) if m else {}
        except Exception:
            return []
        return grounded(_from_json(obj, source, f"nuextract-{self.style}"), text)


def qwen_claims(llm, text: str, source: str = "resume", max_tokens: int = 500) -> List[Claim59]:
    """llm: the V58 `LocalLLM` (or StubLLM). Uses its schema-constrained `.json()` call."""
    system = "Extract factual claims a candidate makes about their own experience. Copy the exact supporting sentence into `quote`. Do not invent anything."
    schema = {"type": "object", "properties": {"claims": {"type": "array", "items": {"type": "object", "properties": {
        "skill_or_topic": {"type": "string"}, "quote": {"type": "string"}, "duration": {"type": "string"}, "metric": {"type": "string"},
        "ownership": {"type": "string"}, "deployed_to_production": {"type": "string"}}, "required": ["skill_or_topic", "quote"]}}}, "required": ["claims"]}
    try:
        obj = llm.json(system, text[:6000], "claim_extraction", schema=schema, max_tokens=max_tokens)
    except Exception:
        obj = None
    return grounded(_from_json(obj or {}, source, "qwen"), text)


def extract_claims(text: str, topics: Optional[List[str]] = None, source: str = "resume", nuextract: Optional[NuExtractClient] = None, llm=None) -> List[Claim59]:
    """Order: NuExtract -> Qwen -> rules; the rules always run and fill gaps (de-duplicated by quote)."""
    got: List[Claim59] = []
    if nuextract is not None:
        got = nuextract.extract(text, source)
    if not got and llm is not None:
        got = qwen_claims(llm, text, source)
    seen = {re.sub(r"\W+", "", c.quote.lower())[:80] for c in got}
    for c in rule_claims(text, topics, source):
        k = re.sub(r"\W+", "", c.quote.lower())[:80]
        if k not in seen:
            got.append(c); seen.add(k)
    return got
