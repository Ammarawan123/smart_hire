"""
transcript_fix_v60.py — use candidate_profile.json to get technical words right in the Whisper transcript.

TWO LEVELS (both audited, neither silent)
  1. BEFORE decoding: `whisper_prompt()` / `hotwords()` give faster-whisper an `initial_prompt` / `hotwords` string made of the candidate's own terms
     (skills, project names, tools, company, role).  This biases the decoder toward "OpenCV" instead of "open CV".
  2. AFTER decoding: `correct()` repairs what is still wrong.  A word (or 2-3 word sequence) is replaced by a profile term only if
        - it is not a common English word,
        - its normalised form is very close to the term (difflib ratio >= threshold; short terms need an exact normalised match),
        - and, for fuzzy (non-exact) matches, the consonant skeleton also agrees (so 'tension flow' -> 'TensorFlow' but 'bad' never becomes 'Pandas').
     Every replacement is returned as a change record {from, to, score, kind}; the runner stores them in the report, so a wrong "correction"
     is visible and the original transcript is never lost.
Not a language model; deliberately conservative: a missed correction is cheaper than an invented word in a hiring transcript.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_WORD = re.compile(r"[A-Za-z0-9#+.'’\-]+")
_COMMON = set("""a about above after again all also am an and any are as at be because been before being below between both but by can could did do does doing down during each few
for from further had has have having he her here hers him his how i if in into is it its just like me more most my no nor not now of off on once only or other our out over own same she
should so some such than that the their them then there these they this those through to too under until up very was we were what when where which while who why will with would you your
yes yeah okay ok well right know think mean thing things way want need use used using work worked working make made build built project projects data model models code system app
application web server client user users file files test tests time first second one two three four five six seven eight nine ten good great bad new old big small high low long short
name names team teams day days year years month months week weeks job role company learn learned learning train trained training run running real life people many much very really
actually basically something someone anything everything nothing different important simple hard easy go going goes get got see say said take come came look put give show try ask call rust swift spring express unity oracle""".split())

# acoustic confusions that string similarity alone cannot see; applied only when the target term is in the candidate's own vocabulary
_PHONETIC_ALIASES = {
    "sequel": "SQL", "my sequel": "MySQL", "post gress": "PostgreSQL", "postgress": "PostgreSQL", "pie torch": "PyTorch", "pie thon": "Python", "pi torch": "PyTorch",
    "tensor flow": "TensorFlow", "tension flow": "TensorFlow", "open cv": "OpenCV", "open c v": "OpenCV", "scikit learn": "scikit-learn", "sigh kit learn": "scikit-learn",
    "fast api": "FastAPI", "mongo db": "MongoDB", "java script": "JavaScript", "type script": "TypeScript", "git hub": "GitHub", "numb pie": "NumPy", "num pie": "NumPy",
    "media pipe": "MediaPipe", "hugging face": "Hugging Face", "lang chain": "LangChain", "yolo v8": "YOLOv8", "yolo v5": "YOLOv5", "rest api": "REST API",
    "dot net": ".NET", "asp dot net": "ASP.NET", "c sharp": "C#", "see sharp": "C#", "docker file": "Dockerfile", "jupiter notebook": "Jupyter Notebook", "jupiter": "Jupyter",
    "r a g": "RAG", "l l m": "LLM", "n l p": "NLP", "c n n": "CNN",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9#+.]+", "", (s or "").lower())


def _skeleton(s: str) -> str:
    """Consonant skeleton with a few sound-alike merges (k/c/q, s/z, f/ph, j/g...), vowels dropped, repeats collapsed."""
    t = _norm(s)
    t = t.replace("ph", "f").replace("ck", "k").replace("qu", "k")
    t = t.translate(str.maketrans({"c": "k", "q": "k", "z": "s", "j": "g", "x": "ks", "y": "i", "w": "v"}))
    t = re.sub(r"[aeiou]+", "", t)
    return re.sub(r"(.)\1+", r"\1", t)


class TranscriptCorrector:
    def __init__(self, terms: Iterable[str], min_ratio: float = 0.86, short_term_len: int = 5):
        seen, self.terms = set(), []
        for t in terms:
            t = (t or "").strip()
            if len(t) >= 2 and _norm(t) and _norm(t) not in seen:
                seen.add(_norm(t))
                self.terms.append(t)
        self.min_ratio, self.short = min_ratio, short_term_len
        self._by_norm: Dict[str, str] = {_norm(t): t for t in self.terms}
        self._max_words = max([len(t.split()) for t in self.terms] + [1])
        low = {t.lower(): t for t in self.terms}
        self._aliases = {a: tgt for a, tgt in _PHONETIC_ALIASES.items() if tgt.lower() in low or _norm(tgt) in self._by_norm}
        self.changes_total = 0

    @classmethod
    def from_profile(cls, profile: dict, **kw) -> "TranscriptCorrector":
        return cls((profile or {}).get("vocabulary", {}).get("terms", []), **kw)

    # ---- before decoding ----
    def whisper_prompt(self, max_chars: int = 230, context: str = "A technical job interview.") -> str:
        out, n = [], len(context) + 8
        for t in self.terms:
            if n + len(t) + 2 > max_chars:
                break
            out.append(t)
            n += len(t) + 2
        return f"{context} Terms: " + ", ".join(out) + "." if out else context

    def hotwords(self, limit: int = 40) -> str:
        return " ".join(self.terms[:limit])

    # ---- after decoding ----
    def _match_span(self, words: Sequence[str]) -> Optional[Tuple[str, float, str]]:
        """Best profile term for this word span, or None.  Returns (term, score, kind)."""
        raw = " ".join(words)
        low = raw.lower().strip(".,;:!?")
        if low in self._aliases:
            return self._aliases[low], 1.0, "alias"
        n = _norm(raw)
        if not n:
            return None
        if len(words) == 1 and low in _COMMON:
            return None                                  # 'go', 'r', 'c' ... as ordinary English words are never touched
        if n in self._by_norm:
            tgt = self._by_norm[n]
            return (tgt, 1.0, "exact") if raw.strip(".,;:!?") != tgt else None
        if len(n) < self.short:
            return None
        best, best_r = None, 0.0
        for tn, term in self._by_norm.items():
            if abs(len(tn) - len(n)) > max(2, len(tn) // 4) or len(tn) < self.short:
                continue
            r = SequenceMatcher(None, n, tn).ratio()
            if r > best_r:
                best, best_r = term, r
        if best is None or best_r < self.min_ratio:
            return None
        if _skeleton(raw) != _skeleton(best) and SequenceMatcher(None, _skeleton(raw), _skeleton(best)).ratio() < 0.8:
            return None
        return best, best_r, "fuzzy"

    def correct(self, text: str) -> Tuple[str, List[dict]]:
        if not text or not self.terms:
            return text, []
        toks = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(text)]
        out, changes, i, last = [], [], 0, 0
        while i < len(toks):
            hit = None
            for span in range(min(self._max_words + 1, 4, len(toks) - i), 0, -1):
                words = [t[0] for t in toks[i:i + span]]
                core = [w.strip(".,;:!?") for w in words]
                m = self._match_span(core)
                if m:
                    hit = (span, m)
                    break
            if hit:
                span, (term, score, kind) = hit
                s, e = toks[i][1], toks[i + span - 1][2]
                orig = text[s:e]
                core_orig = orig.rstrip(".,;:!?")
                punct = orig[len(core_orig):]
                out.append(text[last:s])
                out.append(term + punct)
                last = e
                changes.append({"from": core_orig, "to": term, "score": round(float(score), 3), "kind": kind, "pos": s})
                i += span
            else:
                i += 1
        out.append(text[last:])
        self.changes_total += len(changes)
        return "".join(out), changes
