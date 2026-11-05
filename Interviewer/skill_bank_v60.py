"""
skill_bank_v60.py — persistent skill bank + NuExtract fallback for résumé / JD skill and project extraction.

FLOW (what happens when a résumé or JD is read)
  1. The static concept registry (concepts_v58.REG, 700-term-style ontology) PLUS every skill learned earlier (skill_bank.json, loaded into the
     same registry as concepts with key "bank:<slug>") is used first.  A match is free, instant and deterministic.
  2. Terms that the rule-based extractor could NOT resolve (skills-list items it kept as "custom:<name>", technologies mentioned in a project as
     "using X", unresolved JD requirement lines) are sent, together with a few lines of context, to NuExtract (a small extraction LLM served by
     llama-server).  The prompt is a JSON template with descriptive field names plus a worked example (the NuExtract 1.x format), temperature 0.
  3. UNCERTAINTY GATING: every value NuExtract returns must (a) occur verbatim in the text it was shown (kills invented skills),
     (b) look like a skill name (2-40 chars, <= 4 words, not a sentence / spoken language / soft skill / level word / date / contact detail).
     Anything uncertain is DROPPED, never guessed (false grounding would propagate into interview questions).
  4. Accepted skills are written to skill_bank.json (name, category, aliases, source, counts) and registered in the live registry, so the
     résumé/JD is re-extracted once and the skill now behaves like any built-in concept, in this run and in every later run.
  5. Candidates NuExtract did not confirm are remembered as "rejected" (after 2 rejections they are no longer sent again).

NuExtract output is only ever used to ADD grounded skills / projects.  If the model, llama-server or the network port is unavailable the pipeline
silently keeps its rule-based result (and says so in the extraction log) — it never fails the interview.
NOT TESTED against a real NuExtract model in this package (no weights here); prompt building, JSON recovery, grounding, gating, persistence and
registry integration are unit-tested with a stub server (tests/test_v60.py).
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import concepts_v58 as K
from schema_v58 import SKILL_CATEGORIES

# ───────────────────────────── paths ─────────────────────────────
def default_bank_path() -> Path:
    env = os.environ.get("SMARTHIRE_SKILL_BANK")
    if env:
        return Path(env)
    try:
        import config as C
        return Path(C.SKILL_BANK)
    except Exception:
        return Path(__file__).resolve().parents[1] / "data" / "skill_bank.json"


# ───────────────────────────── text helpers ─────────────────────────────
def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9#+.]+", "_", (name or "").lower()).strip("_")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9#+.]+", "", (s or "").lower())


def grounded_in(name: str, text: str) -> bool:
    """True if `name` occurs in `text` (case-, space- and hyphen-insensitive, so 'scikit learn' matches 'scikit-learn')."""
    n = norm(name)
    return bool(n) and n in norm(text)


_SOFT_OR_NOISE = {
    "skills", "skill", "technical skills", "soft skills", "communication", "communication skills", "teamwork", "team work", "team player", "leadership", "problem solving",
    "problem-solving", "critical thinking", "time management", "adaptability", "creativity", "management", "hard working", "hardworking", "fast learner", "quick learner",
    "punctual", "responsible", "motivated", "interpersonal skills", "presentation", "presentations", "public speaking", "negotiation", "etc", "and more", "others", "other",
    "basic", "advanced", "intermediate", "beginner", "good", "proficient", "familiar", "excellent", "strong", "knowledge", "experience", "internet", "computer", "computers",
    "hobbies", "interests", "references", "available on request", "n/a", "none", "yes", "no", "tools", "technologies", "languages", "frameworks", "libraries", "databases",
    "programming", "coding", "development", "software", "hardware", "projects", "education", "summary", "profile", "objective",
}
_SOFT_WORDS = {"communication", "teamwork", "leadership", "skills", "excellent", "strong", "good", "proficient", "basic", "advanced", "fluent", "responsible", "hardworking",
               "motivated", "punctual", "creative", "passionate", "dedicated", "willing", "ability", "knowledge", "experience", "familiarity"}
_NATURAL = {"english", "urdu", "hindi", "arabic", "punjabi", "pashto", "sindhi", "balochi", "french", "german", "spanish", "chinese", "mandarin", "turkish", "persian", "farsi",
            "bengali", "bangla", "tamil", "japanese", "korean", "russian", "italian", "portuguese", "dutch", "swahili"}
_MONTHS = re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s*(19|20)?\d{0,2}$", re.I)


def validate_candidate(name: str, source_text: str = "") -> Tuple[bool, str]:
    """Uncertainty gate for a skill name proposed by a model.  (ok, reason)."""
    n = (name or "").strip(" \t\r\n.,;:|-–—•·()[]\"'")
    if not (2 <= len(n) <= 40):
        return False, "length"
    if not re.search(r"[A-Za-z]", n):
        return False, "no letters"
    if len(n.split()) > 4:
        return False, "too many words (sentence-like)"
    low = n.lower()
    if low in _SOFT_OR_NOISE or any(w in _SOFT_WORDS for w in re.findall(r"[a-z]+", low)):
        return False, "generic / soft-skill word"
    if low in _NATURAL:
        return False, "spoken language"
    if re.search(r"@|https?://|www\.|\.com\b|^\+?\d[\d\s\-()]{6,}$", n):
        return False, "contact detail"
    if _MONTHS.match(n) or re.match(r"^(19|20)\d{2}$", n):
        return False, "date"
    if n.endswith((".", "!", "?")) or re.search(r"\b(i|my|we|our)\b", low):
        return False, "sentence-like"
    if source_text and not grounded_in(n, source_text):
        return False, "not found verbatim in the source text"
    return True, ""


_FIELD_TO_CATEGORY = {
    "programming_languages": "programming_languages", "frameworks": "frameworks", "libraries": "libraries", "databases": "databases",
    "ai_ml_methods_and_models": "AI_ML", "tools_and_platforms": "tools", "required_skills": "other", "preferred_skills": "other",
    "tools_and_technologies": "tools", "technologies_used": "other",
}


# ───────────────────────────── registry integration ─────────────────────────────
def register_concept(key: str, name: str, category: str, aliases: Iterable[str]) -> bool:
    """Add one learned skill to the live concept registry (idempotent).  Aliases that already resolve to another concept are skipped so a
    learned term can never hijack a built-in one."""
    if key in K.REG:
        return False
    cat = category if category in SKILL_CATEGORIES else "other"
    keep = []
    for al in aliases:
        al = (al or "").strip()
        if len(al) < 2:
            continue
        existing = K.lookup_item(al)
        if existing and existing != key:
            continue
        keep.append(al)
    c = K._c(key, name, cat, "|".join(a for a in keep if "|" not in a), library_like=True)
    for al in sorted(set(c.aliases), key=len, reverse=True):
        if len(al) < 2:
            continue
        K._ALIAS_RX.append((K._alias_regex(al), key, al))
        K._ITEM_MAP.setdefault(re.sub(r"[^a-z0-9#+.]", "", al.lower()), key)
    K._ALIAS_RX.sort(key=lambda t: -len(t[2]))
    return True


# ───────────────────────────── the bank ─────────────────────────────
class SkillBank:
    """JSON-persisted learned skills.  Never contains anything that did not pass `validate_candidate`."""

    REJECT_AFTER = 2

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else default_bank_path()
        self.data: Dict = {"version": 1, "skills": {}, "rejected": {}}
        self.new_this_run: List[str] = []

    @classmethod
    def load(cls, path: Optional[Path] = None, register: bool = True) -> "SkillBank":
        b = cls(path)
        try:
            if b.path.is_file():
                d = json.loads(b.path.read_text(encoding="utf-8"))
                if isinstance(d, dict) and isinstance(d.get("skills"), dict):
                    b.data = {"version": d.get("version", 1), "skills": d["skills"], "rejected": d.get("rejected", {}) or {}}
        except Exception:
            b.data = {"version": 1, "skills": {}, "rejected": {}}      # a corrupt bank must never stop an interview; it is rebuilt from scratch
        if register:
            b.register()
        return b

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def register(self) -> int:
        n = 0
        for s, e in self.data["skills"].items():
            if register_concept("bank:" + s, e.get("name", s), e.get("category", "other"), e.get("aliases", [])):
                n += 1
        return n

    # ---- queries ----
    def __len__(self) -> int:
        return len(self.data["skills"])

    def content_hash(self) -> str:
        return hashlib.sha1(json.dumps(self.data["skills"], sort_keys=True).encode()).hexdigest()[:10]

    def knows(self, name: str) -> Optional[str]:
        """Concept key (static or learned) that `name` already resolves to, else None."""
        return K.lookup_item(name)

    def is_rejected(self, name: str) -> bool:
        return self.data["rejected"].get(slug(name), {}).get("count", 0) >= self.REJECT_AFTER

    # ---- updates ----
    def add(self, name: str, category: str = "other", aliases: Sequence[str] = (), source: str = "nuextract") -> Optional[str]:
        """Add a validated skill; returns its concept key (existing key if already known)."""
        ok, _why = validate_candidate(name)
        if not ok:
            return None
        known = self.knows(name)
        if known:
            return known
        s = slug(name)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        al = sorted({name.strip().lower(), *(a.strip().lower() for a in aliases if a and validate_candidate(a)[0])})
        cat = category if category in SKILL_CATEGORIES else "other"
        self.data["skills"][s] = {"name": name.strip(), "category": cat, "aliases": al, "source": source, "count": 1, "first_seen": now, "last_seen": now}
        self.data["rejected"].pop(s, None)
        register_concept("bank:" + s, name.strip(), cat, al)
        self.new_this_run.append(s)
        return "bank:" + s

    def touch(self, key: str) -> None:
        if key.startswith("bank:") and key[5:] in self.data["skills"]:
            e = self.data["skills"][key[5:]]
            e["count"] = e.get("count", 0) + 1
            e["last_seen"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def reject(self, name: str, reason: str = "") -> None:
        e = self.data["rejected"].setdefault(slug(name), {"name": name, "count": 0, "reason": reason})
        e["count"] += 1
        e["reason"] = reason or e.get("reason", "")


# ───────────────────────────── NuExtract access ─────────────────────────────
SKILL_TEMPLATE = {"programming_languages": [""], "frameworks": [""], "libraries": [""], "databases": [""], "ai_ml_methods_and_models": [""], "tools_and_platforms": [""]}
SKILL_EXAMPLE = {"programming_languages": ["Python", "C++"], "frameworks": ["Flask", "Angular"], "libraries": ["OpenCV", "pandas"], "databases": ["MySQL"],
                 "ai_ml_methods_and_models": ["YOLOv8", "CNN"], "tools_and_platforms": ["Git", "Docker", "Roboflow"]}
JD_TEMPLATE = {"required_skills": [""], "preferred_skills": [""], "tools_and_technologies": [""]}
JD_EXAMPLE = {"required_skills": ["Python", "SQL", "REST APIs"], "preferred_skills": ["Docker"], "tools_and_technologies": ["Git", "Postman"]}
PROJECT_TEMPLATE = {"projects": [{"name": "", "technologies_used": [""], "what_it_does": "", "candidate_role": ""}]}
PROJECT_EXAMPLE = {"projects": [{"name": "Sign Language Detection", "technologies_used": ["YOLOv8", "OpenCV"],
                                 "what_it_does": "detects hand signs from webcam frames", "candidate_role": "trained and evaluated the model"}]}


def _loads(raw: str) -> Optional[dict]:
    """Recover a JSON object from model output (code fences, trailing commas, single quotes, truncated tail)."""
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    cand = m.group(0) if m else raw
    for attempt in (cand, re.sub(r",\s*([}\]])", r"\1", cand), re.sub(r",\s*([}\]])", r"\1", cand).replace("'", '"')):
        try:
            v = json.loads(attempt)
            return v if isinstance(v, dict) else None
        except Exception:
            pass
    # truncated output: close open brackets
    s = re.sub(r",\s*$", "", cand.rstrip())
    s += "]" * max(0, s.count("[") - s.count("]")) + "}" * max(0, s.count("{") - s.count("}"))
    try:
        v = json.loads(re.sub(r",\s*([}\]])", r"\1", s))
        return v if isinstance(v, dict) else None
    except Exception:
        return None


class NuExtractRunner:
    """style '1.5' (default): POST /completion with the '<|input|> ### Template / ### Example / ### Text <|output|>' prompt.
    style '2.0': POST /v1/chat/completions with '# Template:'.  `post(path, payload)->dict` can be injected (tests)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8082", style: str = "1.5", timeout: float = 90.0, post: Optional[Callable] = None):
        self.base, self.style, self.timeout = base_url.rstrip("/"), style, timeout
        self._post = post or self._http_post
        self._injected = post is not None
        self.calls = 0
        self.errors: List[str] = []

    def _http_post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def available(self) -> bool:
        if self._injected:
            return True
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def build_prompt(self, text: str, template: dict, example: Optional[dict] = None) -> str:
        tpl = json.dumps(template, indent=4)
        if self.style == "2.0":
            ex = f"\n# Examples:\n{json.dumps(example, indent=4)}" if example else ""
            return f"# Template:\n{tpl}{ex}\n# Context:\n{text}"
        ex = f"### Example:\n{json.dumps(example, indent=4)}\n" if example else ""
        return f"<|input|>\n### Template:\n{tpl}\n{ex}### Text:\n{text}\n\n<|output|>"

    def run(self, text: str, template: dict, example: Optional[dict] = None, max_tokens: int = 700) -> Optional[dict]:
        self.calls += 1
        try:
            if self.style == "2.0":
                resp = self._post("/v1/chat/completions", {"messages": [{"role": "user", "content": self.build_prompt(text, template, example)}], "temperature": 0.0, "max_tokens": max_tokens})
                raw = resp["choices"][0]["message"]["content"]
            else:
                resp = self._post("/completion", {"prompt": self.build_prompt(text, template, example), "temperature": 0.0, "n_predict": max_tokens,
                                                  "stop": ["<|end-output|>", "<|input|>"], "cache_prompt": False})
                raw = resp.get("content", "")
        except Exception as e:
            self.errors.append(f"{type(e).__name__}: {e}")
            return None
        obj = _loads(raw)
        if obj is None:
            self.errors.append("unparseable model output: " + (raw or "")[:120].replace("\n", " "))
        return obj


class NuExtractServer:
    """Starts llama-server with the NuExtract GGUF on its own port (the interviewer's Qwen uses another one) and stops it again, so the RAM is free
    before Qwen starts.  If SMARTHIRE_NUEXTRACT_URL is set, that server is used and never stopped."""

    def __init__(self, gguf: Optional[Path] = None, exe: Optional[Path] = None, port: int = 8082, ctx: int = 4096, threads: int = 0, log_path: Optional[Path] = None):
        try:
            import config as C
            gguf = gguf or C.NUEXTRACT_GGUF
            exe = exe or C.LLAMA_SERVER
            threads = threads or C.LLM_THREADS
            log_path = log_path or (C.OUTPUTS_DIR / "nuextract_server.log")
        except Exception:
            pass
        self.gguf, self.exe, self.port, self.ctx, self.threads = Path(gguf) if gguf else None, Path(exe) if exe else None, port, ctx, threads
        self.log_path = Path(log_path) if log_path else Path("nuextract_server.log")
        self.proc: Optional[subprocess.Popen] = None
        self.reason = ""
        self.external = bool(os.environ.get("SMARTHIRE_NUEXTRACT_URL"))
        self.url = os.environ.get("SMARTHIRE_NUEXTRACT_URL") or f"http://127.0.0.1:{port}"

    def start(self, timeout_s: float = 120.0) -> bool:
        probe = NuExtractRunner(self.url)
        if probe.available():
            self.reason = "reused running server" if not self.external else "external server"
            return True
        if self.external:
            self.reason = f"SMARTHIRE_NUEXTRACT_URL={self.url} not reachable"
            return False
        if not (self.gguf and self.gguf.is_file()):
            self.reason = f"NuExtract GGUF not found ({self.gguf})"
            return False
        if not (self.exe and self.exe.is_file()):
            self.reason = f"llama-server not found ({self.exe})"
            return False
        cmd = [str(self.exe), "-m", str(self.gguf), "-c", str(self.ctx), "--host", "127.0.0.1", "--port", str(self.port), "-np", "1", "--no-webui"]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = open(self.log_path, "w", encoding="utf-8", errors="replace")
            self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, creationflags=0x08000000 if os.name == "nt" else 0)
        except OSError as e:
            self.reason = f"launch failed: {e}"
            return False
        atexit.register(self.stop)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                self.reason = f"llama-server exited early (code {self.proc.returncode}); see {self.log_path}"
                return False
            if probe.available():
                self.reason = f"started in {time.time() - t0:.1f}s"
                return True
            time.sleep(0.4)
        self.stop()
        self.reason = f"not ready after {timeout_s:.0f}s; see {self.log_path}"
        return False

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except Exception:
                self.proc.kill()
        self.proc = None


# ───────────────────────────── context windows ─────────────────────────────
def context_for_terms(raw_text: str, terms: Sequence[str], radius: int = 2, max_chunk: int = 1400, max_total: int = 4200) -> List[str]:
    """Lines of `raw_text` that mention a term (+/- `radius` lines), merged and split into chunks small enough for a 4k-context model."""
    lines = [l.rstrip() for l in (raw_text or "").splitlines()]
    keep = set()
    for i, ln in enumerate(lines):
        if any(grounded_in(t, ln) for t in terms):
            keep.update(range(max(0, i - radius), min(len(lines), i + radius + 1)))
    chunks, cur, total = [], "", 0
    for i in sorted(keep):
        ln = lines[i].strip()
        if not ln:
            continue
        if len(cur) + len(ln) + 1 > max_chunk and cur:
            chunks.append(cur)
            total += len(cur)
            cur = ""
        if total + len(cur) > max_total:
            break
        cur += ln + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


# ───────────────────────────── unresolved-term discovery ─────────────────────────────
def unresolved_resume_terms(resume) -> List[Dict[str, str]]:
    """Skills / technologies the rule-based extractor kept as 'custom:' (it could not resolve them to a known concept)."""
    out, seen = [], set()

    def add(name, where, hint="other"):
        n = norm(name)
        if n and n not in seen and validate_candidate(name)[0]:
            seen.add(n)
            out.append({"name": name.strip(), "where": where, "hint": hint})
    for s in resume.all_skills():
        if s.key.startswith("custom:"):
            add(s.name, "skills", s.category)
    for p in resume.projects:
        for t in p.technologies:
            if t.key.startswith("custom:"):
                add(t.name, f"project:{p.short_name}")
    for e in resume.experience:
        for t in e.technologies:
            if t.key.startswith("custom:"):
                add(t.name, f"experience:{e.title}")
    return out


# ───────────────────────────── the extraction passes ─────────────────────────────
class SkillExtractionLog:
    def __init__(self):
        self.nuextract_available: Optional[bool] = None
        self.nuextract_reason = ""
        self.static_or_bank_hits = 0
        self.unresolved: List[str] = []
        self.accepted: List[Dict[str, str]] = []
        self.dropped: List[Dict[str, str]] = []
        self.projects_added: List[str] = []
        self.errors: List[str] = []

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _collect_skill_values(obj: Optional[dict]) -> List[Tuple[str, str]]:
    """[(name, category)] from a NuExtract result (any of the templates above)."""
    out = []
    for field, vals in (obj or {}).items():
        if not isinstance(vals, list) or field == "projects":
            continue
        cat = _FIELD_TO_CATEGORY.get(field, "other")
        for v in vals:
            if isinstance(v, str) and v.strip():
                out.append((v.strip(), cat))
    return out


def learn_skills(runner: Optional[NuExtractRunner], bank: SkillBank, raw_text: str, candidates: Sequence[Dict[str, str]], log: SkillExtractionLog,
                 template: dict = SKILL_TEMPLATE, example: dict = SKILL_EXAMPLE, mode: str = "resume", chunks: Optional[List[str]] = None) -> int:
    """Ask NuExtract about the context of `candidates` (or about explicit `chunks`, e.g. unresolved JD lines); add validated, grounded skills to the
    bank.  Returns the number of NEW bank skills."""
    todo = [c for c in candidates if not bank.is_rejected(c["name"]) and not bank.knows(c["name"])]
    log.unresolved += [c["name"] for c in candidates]
    if not todo and not chunks:
        return 0
    if runner is None or log.nuextract_available is False:
        return 0
    if chunks is None:
        chunks = context_for_terms(raw_text, [c["name"] for c in todo])
    if not chunks:
        return 0
    seen_ok: Dict[str, str] = {}
    answered = False
    for ch in chunks:
        obj = runner.run(ch, template, example)
        if obj is None:
            continue
        answered = True
        for name, cat in _collect_skill_values(obj):
            ok, why = validate_candidate(name, ch)
            if not ok:
                log.dropped.append({"name": name, "reason": why})
                continue
            if bank.knows(name):
                continue
            seen_ok.setdefault(norm(name), (name, cat))
    if not answered:
        log.errors += runner.errors[-2:]
        return 0
    added = 0
    cand_norms = {norm(c["name"]): c for c in todo}
    for nn, (name, cat) in seen_ok.items():
        cand = cand_norms.get(nn)
        if cat == "other" and cand and cand.get("hint") in SKILL_CATEGORIES:
            cat = cand["hint"]
        key = bank.add(name, cat, aliases=[cand["name"]] if cand and cand["name"].lower() != name.lower() else [], source=f"nuextract-{mode}")
        if key:
            added += 1
            log.accepted.append({"name": name, "category": cat, "key": key})
    for c in todo:                                   # asked about, not confirmed -> remembered so the same noise is not sent again and again
        if norm(c["name"]) not in seen_ok:
            bank.reject(c["name"], "not confirmed by NuExtract")
            log.dropped.append({"name": c["name"], "reason": "not confirmed by NuExtract"})
    return added


def nuextract_projects(runner: Optional[NuExtractRunner], resume, log: SkillExtractionLog) -> List:
    """Projects NuExtract finds that the rule-based extractor missed (grounded + validated like every rule-based project).  Returns Project objects."""
    if runner is None or log.nuextract_available is False:
        return []
    import resume_extractor_v58 as RX
    text = resume.raw_text or ""
    if not text.strip():
        return []
    have = {norm(p.short_name) for p in resume.projects} | {norm(p.title) for p in resume.projects}
    found = []
    chunks = [text[i:i + 1500] for i in range(0, min(len(text), 4500), 1500)]
    for ch in chunks:
        obj = runner.run(ch, PROJECT_TEMPLATE, PROJECT_EXAMPLE, max_tokens=800)
        if not obj:
            continue
        for pj in obj.get("projects", []) or []:
            if not isinstance(pj, dict):
                continue
            name = str(pj.get("name", "")).strip()
            if not name or norm(name) in have or not grounded_in(name, ch) or len(name.split()) > 10:
                continue
            what = str(pj.get("what_it_does", "")).strip()
            techs = [t for t in (pj.get("technologies_used") or []) if isinstance(t, str) and grounded_in(t, ch)]
            bullets = [b for b in (what, ("Built with " + ", ".join(techs)) if techs else "") if b]
            e = RX.Entry(title=name, bullets=bullets)
            p = RX.build_project(e, [])
            if p is None:
                continue
            ok, why = RX.valid_project(p)
            if not ok:
                log.dropped.append({"name": name, "reason": "project: " + why})
                continue
            have.add(norm(name))
            found.append(p)
            log.projects_added.append(name)
    return found
