"""textkit_v58.py — small, dependency-free text helpers shared by every V58 module."""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Iterable, List, Set

BULLET_GLYPHS = "•●▪■◦○·‣►▶➢➤❖✓✔*-–—⁃∙●"
_STOP = set("""a an the and or of to in on for with at by from as is are was were be been being it its this that these those i me my we our you your
he she they them their his her do does did have has had will would can could should may might not no so if then than too very just also about into over
out up down off more most some any each other such only own same s t don now what which who whom how when where why all both few""".split())

_CONTRACTIONS = [
    (r"\bdont\b", "don't"), (r"\bdidnt\b", "didn't"), (r"\bdoesnt\b", "doesn't"), (r"\bcant\b", "can't"), (r"\bcannot\b", "can't"),
    (r"\bwont\b", "won't"), (r"\bhavent\b", "haven't"), (r"\bhasnt\b", "hasn't"), (r"\bhadnt\b", "hadn't"), (r"\bisnt\b", "isn't"),
    (r"\barent\b", "aren't"), (r"\bwasnt\b", "wasn't"), (r"\bwerent\b", "weren't"), (r"\bshouldnt\b", "shouldn't"),
    (r"\bi m\b", "i'm"), (r"\bim\b", "i'm"), (r"\bive\b", "i've"), (r"\bid\b(?=\s+(?:use|used|like|build|built))", "i'd"),
    (r"\bi do not\b", "i don't"), (r"\bi did not\b", "i didn't"), (r"\bi have not\b", "i haven't"), (r"\bi can not\b", "i can't"),
    (r"\bdo not\b", "don't"), (r"\bdid not\b", "didn't"), (r"\bhave not\b", "haven't"), (r"\bhas not\b", "hasn't"),
]


def nfkc(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    return s


def norm_answer(s: str) -> str:
    """Lower-case + contraction repair for typed / ASR answers (broken English tolerant)."""
    t = nfkc(s).lower().replace("’", "'").replace("`", "'")
    for pat, rep in _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\b(uh+|um+|umm+|hmm+|er+m?|ah+|like,?(?=\s+(?:i|we|the|if|so|you)\b))\b[,.]?", " ", t)
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t)                  # stutters: "i i used" -> "i used"
    return re.sub(r"\s+", " ", t).strip()


def tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9#+.]+", (s or "").lower())


def content_tokens(s: str) -> Set[str]:
    out = set()
    for t in tokens(s):
        t = t.strip(".")
        if t and t not in _STOP and len(t) > 1:
            out.add(_stem(t))
    return out


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a and b else 0.0


def cosine_sets(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


def split_sentences(s: str) -> List[str]:
    s = re.sub(r"\s+", " ", s or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\s*\n+\s*", s)
    return [p.strip() for p in parts if p and p.strip()]


def word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9#+']+", s or ""))


def fuzzy_eq(a: str, b: str, cutoff: float = 0.88) -> bool:
    if a == b:
        return True
    if len(a) < 5 or len(b) < 5 or abs(len(a) - len(b)) > 2:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= cutoff


def is_bullet(line: str) -> bool:
    s = (line or "").lstrip()
    return bool(s) and (s[0] in "•●▪■◦○·‣►▶➢➤❖✓✔*▪" or re.match(r"^[-–—]\s+\S", s) is not None or re.match(r"^\d{1,2}[.)]\s+\S", s) is not None)


def strip_bullet(line: str) -> str:
    s = (line or "").lstrip()
    s = re.sub(r"^(?:[•●▪■◦○·‣►▶➢➤❖✓✔*▪]|[-–—]|\d{1,2}[.)])\s*", "", s)
    return s.strip()


def clip(s: str, n: int = 140) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def ensure_period(s: str) -> str:
    s = s.strip()
    return s if not s or s[-1] in ".?!" else s + "."
