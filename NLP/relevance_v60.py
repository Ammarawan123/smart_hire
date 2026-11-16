"""
relevance_v60.py — SBERT-based relevance, driven by candidate_profile.json.

  question_relevance(question, answer)        cosine(question, answer)                                  -> did the answer address the question?
  resume_support(answer)                      max cosine(answer sentence, résumé chunk)                  -> is what is said consistent with the résumé?
  unlisted_tech(answer)                       technologies named in the answer that appear nowhere in the résumé/profile (information, not an accusation)
  sentence_quality(question, sentences, ...)  per-sentence 0..1 score used by the cheating-pattern detector (relevance + technical substance - hedging)

Embeddings: sentence-transformers all-MiniLM-L6-v2 from the LOCAL folder (config.SBERT_DIR; offline).  If sentence-transformers or the folder is missing, a
dependency-free lexical fallback (TF-weighted token cosine) is used and `method` says so, so a score is never silently of the other kind.
Résumé chunks come from the profile (projects, experience bullets, skills lines, summary), i.e. the SAME extraction that drives the interview.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_TOKEN = re.compile(r"[a-z0-9#+.]{2,}")
_STOP = set("the a an and or of to in on for with is are was were be been it this that i we you my our your they he she as at by from not no yes so do did does have has had can could would should will just very really also".split())
_HEDGE = re.compile(r"\b(not sure|don'?t know|do not know|i think|i guess|maybe|perhaps|probably|kind of|sort of|something like|no wait|actually no|i forgot|let me think|um+|uh+|er+|hmm+)\b", re.I)


def _tokens(s: str) -> List[str]:
    return [t for t in _TOKEN.findall((s or "").lower()) if t not in _STOP]


class _Lexical:
    method = "lexical"

    def __init__(self, corpus: Sequence[str]):
        df = Counter(t for d in corpus for t in set(_tokens(d)))
        self.n = max(1, len(corpus))
        self.idf = {t: math.log(1 + self.n / (1 + c)) for t, c in df.items()}

    def embed(self, texts: Sequence[str]) -> List[Dict[str, float]]:
        out = []
        for s in texts:
            tf = Counter(_tokens(s))
            v = {t: c * self.idf.get(t, math.log(1 + self.n)) for t, c in tf.items()}
            nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            out.append({t: x / nrm for t, x in v.items()})
        return out

    @staticmethod
    def cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return float(sum(x * b.get(t, 0.0) for t, x in a.items()))


class _Sbert:
    method = "sbert"

    def __init__(self, model):
        self.model = model

    def embed(self, texts: Sequence[str]):
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)

    @staticmethod
    def cos(a, b) -> float:
        return float(np.dot(a, b))


def load_sbert(model_dir: Optional[str] = None):
    """The local SBERT model or None (never downloads: config.apply_runtime_env sets the offline flags)."""
    try:
        from sentence_transformers import SentenceTransformer
        if model_dir is None:
            import config as C
            model_dir = C.local_or(C.SBERT_DIR, "all-MiniLM-L6-v2")
        return SentenceTransformer(str(model_dir))
    except Exception:
        return None


def resume_chunks(profile_or_resume) -> List[str]:
    """Résumé text units from a profile dict (preferred) or a Resume object."""
    r = profile_or_resume
    d = r.get("resume", r) if isinstance(r, dict) else r.to_dict()
    chunks: List[str] = []
    if d.get("summary"):
        chunks.append(d["summary"])
    for p in d.get("projects", []):
        techs = ", ".join(t.get("name", "") for t in p.get("technologies", []))
        chunks.append(f"{p.get('short_name', '')}. {p.get('description', '')} Technologies: {techs}".strip())
    for e in d.get("experience", []):
        chunks.append(f"{e.get('title', '')} at {e.get('organization', '')}. " + " ".join(e.get("bullets", [])))
    skills = [s.get("name", "") for lst in (d.get("skills") or {}).values() for s in lst]
    if skills:
        chunks.append("Skills: " + ", ".join(skills))
    for c in d.get("certifications", []):
        chunks.append(c.get("name", ""))
    return [c.strip() for c in chunks if c and len(c.strip()) > 8]


class ResumeRelevance:
    def __init__(self, chunks: Sequence[str], known_tech_keys: Iterable[str] = (), model=None, use_sbert: bool = True):
        self.chunks = list(chunks)
        self.known = set(known_tech_keys)
        sb = model if model is not None else (load_sbert() if use_sbert else None)
        self.backend = _Sbert(sb) if sb is not None else _Lexical(self.chunks + ["technical interview answer"])
        self.method = self.backend.method
        self._chunk_emb = self.backend.embed(self.chunks) if self.chunks else []

    @classmethod
    def from_profile(cls, profile: dict, **kw) -> "ResumeRelevance":
        keys = [s.get("key", "") for s in profile.get("skills", [])]
        return cls(resume_chunks(profile), keys, **kw)

    # ---- scores ----
    def _cos_many(self, q_emb, embs) -> List[float]:
        return [self.backend.cos(q_emb, e) for e in embs]

    def question_relevance(self, question: str, answer: str) -> float:
        if not (question and answer):
            return 0.0
        a, q = self.backend.embed([answer, question])
        return float(max(0.0, self.backend.cos(a, q)))

    def resume_support(self, answer_sentences: Sequence[str]) -> Tuple[float, str]:
        """(best cosine of any answer sentence with any résumé chunk, that chunk)."""
        if not self.chunks or not answer_sentences:
            return 0.0, ""
        embs = self.backend.embed(list(answer_sentences))
        best, bc = 0.0, ""
        for e in embs:
            sims = self._cos_many(e, self._chunk_emb)
            j = int(np.argmax(sims))
            if sims[j] > best:
                best, bc = float(sims[j]), self.chunks[j]
        return best, bc

    def unlisted_tech(self, answer: str) -> List[str]:
        """Technologies named in the answer that the résumé/profile never mentions (as concept names)."""
        import concepts_v58 as K
        out = []
        for key, a, b, al in K.find_mentions(answer):
            c = K.get(key)
            if c and c.kind == "technology" and key not in self.known and c.name not in out:
                out.append(c.name)
        return out

    def score(self, question: str, answer: str, answer_sentences: Optional[Sequence[str]] = None) -> Dict:
        from text_cues_v59 import sentences
        sents = list(answer_sentences) if answer_sentences else sentences(answer)
        support, chunk = self.resume_support(sents)
        return {"method": self.method, "question_relevance": round(self.question_relevance(question, answer), 3), "resume_support": round(support, 3),
                "best_resume_chunk": chunk[:160], "unlisted_tech": self.unlisted_tech(answer)}

    def sentence_quality(self, question: str, sents: Sequence[str], target_keys: Sequence[str] = ()) -> List[float]:
        """Per-sentence 0..1: 0.45 relevance to the question + 0.40 technical substance (facet hits for the question's targets, named technologies)
        + 0.15 absence of hedging/filler.  Heuristic by design; only used to find large, sudden changes INSIDE one answer."""
        import concepts_v58 as K
        if not sents:
            return []
        q_emb = self.backend.embed([question or ""])[0]
        embs = self.backend.embed(list(sents))
        out = []
        for s, e in zip(sents, embs):
            rel = self.backend.cos(q_emb, e)
            rel_n = float(np.clip((rel - 0.10) / 0.45, 0.0, 1.0)) if self.method == "sbert" else float(np.clip(rel / 0.35, 0.0, 1.0))
            low = re.sub(r"[^a-z0-9 ]", " ", s.lower())
            facets = set()
            for k in target_keys:
                facets.update(K.facets_hit(k, low))
            techs = {k for k, *_ in K.find_mentions(s)}
            tech_n = float(min(1.0, (len(facets) + 0.5 * len(techs)) / 2.0))
            hedge = min(1.0, len(_HEDGE.findall(s)) / 2.0)
            out.append(float(np.clip(0.45 * rel_n + 0.40 * tech_n + 0.15 * (1.0 - hedge), 0.0, 1.0)))
        return out
