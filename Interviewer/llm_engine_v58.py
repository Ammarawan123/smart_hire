"""
llm_engine_v58.py — local Qwen2.5-1.5B via llama-server, used sparingly and always validated.

What changed vs V51 (latency was the V51 pain: 28 s start, 20–25 s phrasing calls, 18 s adjudications):
  * the LLM is NOT used for section detection, question deduplication, coverage, status transitions or reactions (all deterministic)
  * it is used for (a) ambiguous answer interpretation and (b) follow-up phrasing when the candidate's own words must be woven in
  * both call types request STRICT JSON through llama.cpp's `json_schema` constrained decoding (grammar-based; the model cannot emit
    anything else), tiny max_tokens, low temperature, `cache_prompt` for the shared system prefix
  * a server that is already running is re-used and is left running on exit (persistent server; pass --stop-server to stop it)
  * warm-up generation runs in a background thread while the interviewer's greeting is spoken
  * hard per-interview call caps (cfg.LLM_MAX_*); every failure returns None and the caller uses its deterministic path
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import v58_config as cfg


@dataclass
class CallRecord:
    purpose: str
    ok: bool
    latency_s: float
    reason: str = ""


@dataclass
class StartupResult:
    ok: bool
    reason: str = ""
    seconds: float = 0.0
    exe: str = ""
    model: str = ""
    detail: str = ""
    gen_latency_s: float = 0.0
    reused: bool = False


def _candidate_dirs() -> List[Path]:
    h = cfg.HERE
    import config as _root_cfg
    d = [_root_cfg.LLAMA_SERVER.parent, _root_cfg.QWEN_GGUF.parent, Path.cwd(), h, h.parent, h / "llama", h / "llama.cpp", h / "bin", h / "models", h.parent / "llama.cpp", h.parent / "models", Path.home() / "llama.cpp"]
    if os.name == "nt":
        d += [Path(p) for p in (r"C:\llama.cpp", r"C:\llama", r"D:\llama.cpp")]
    out, seen = [], set()
    for x in d:
        if str(x) not in seen:
            seen.add(str(x))
            out.append(x)
    return out


def find_llama_server(hint: str = "") -> Optional[str]:
    names = ["llama-server.exe", "llama-server"] if os.name == "nt" else ["llama-server"]
    for h in (hint, os.environ.get("SMARTHIRE_LLAMA_EXE", "")):
        if h and Path(h).is_file():
            return str(h)
        if h and Path(h).is_dir():
            for n in names:
                if (Path(h) / n).is_file():
                    return str(Path(h) / n)
    for n in names:
        w = shutil.which(n)
        if w:
            return w
    for d in _candidate_dirs():
        for n in names:
            if (d / n).is_file():
                return str(d / n)
    return None


def find_model(hint: str = "") -> Optional[str]:
    for h in (hint, os.environ.get("SMARTHIRE_LLAMA_MODEL", "")):
        if h and Path(h).is_file():
            return str(h)
        if h and Path(h).is_dir():
            g = sorted(Path(h).glob("*.gguf"))
            if g:
                return str(g[0])
    best = []
    for d in _candidate_dirs():
        try:
            best += list(d.glob("*.gguf"))
        except OSError:
            pass
    best.sort(key=lambda p: (0 if "qwen" in p.name.lower() else 1, p.stat().st_size))
    return str(best[0]) if best else None


def parse_json_loose(text: str) -> Optional[dict]:
    """Extract a JSON object from model output (fences, trailing prose). Returns None if unrecoverable."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.M).strip()
    i = t.find("{")
    if i < 0:
        return None
    t = t[i:]
    depth, in_s, esc, end = 0, False, False, -1
    for idx, ch in enumerate(t):
        if in_s:
            esc = (ch == "\\" and not esc)
            if ch == '"' and not esc:
                in_s = False
            continue
        if ch == '"':
            in_s = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    cand = t[: end + 1] if end > 0 else t
    for c in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
        try:
            v = json.loads(c)
            return v if isinstance(v, dict) else None
        except ValueError:
            continue
    return None


def _check_schema(obj, schema: dict) -> bool:
    """Tiny validator for the subset of JSON-schema V58 uses (type/enum/required/properties/items/maxItems)."""
    t = schema.get("type")
    if "enum" in schema and obj not in schema["enum"]:
        return False
    if t == "object":
        if not isinstance(obj, dict):
            return False
        for r in schema.get("required", []):
            if r not in obj:
                return False
        for k, sub in schema.get("properties", {}).items():
            if k in obj and not _check_schema(obj[k], sub):
                return False
        return True
    if t == "array":
        if not isinstance(obj, list) or len(obj) > schema.get("maxItems", 99):
            return False
        return all(_check_schema(x, schema.get("items", {})) for x in obj)
    if t == "string":
        return isinstance(obj, str)
    return True


class LocalLLM:
    def __init__(self, exe_hint: str = "", model_hint: str = "", host: str = cfg.LLAMA_HOST, port: int = cfg.LLAMA_PORT, ctx: int = cfg.LLAMA_CTX,
                 threads: int = cfg.LLAMA_THREADS, enabled: bool = True, keep_running: bool = True):
        self.exe_hint, self.model_hint, self.host, self.port, self.ctx, self.threads = exe_hint, model_hint, host, port, ctx, threads
        self.enabled, self.keep_running = enabled, keep_running
        self.base = f"http://{host}:{port}"
        self.proc: Optional[subprocess.Popen] = None
        self.records: List[CallRecord] = []
        self.startup: Optional[StartupResult] = None
        self.log_path = str(Path(cfg.CACHE_DIR) / "llama_server_v58.log")
        self.ready = False
        self._schema_ok = True
        self._warm: Optional[threading.Thread] = None
        self.model_name = ""

    def _http(self, path: str, payload: Optional[dict], timeout: float):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")

    def health(self) -> bool:
        try:
            return self._http("/health", None, 2)[0] == 200
        except Exception:
            return False

    def start(self, verbose=print) -> StartupResult:
        t0 = time.time()
        if not self.enabled:
            self.startup = StartupResult(False, "disabled", detail="LLM disabled by flag (--no-llm)")
            return self.startup
        if self.health():
            self.ready = True
            self.startup = StartupResult(True, "reused_running_server", time.time() - t0, detail=self.base, reused=True)
            return self.startup
        exe, model = find_llama_server(self.exe_hint), find_model(self.model_hint)
        if not exe:
            self.startup = StartupResult(False, "exe_not_found", detail="llama-server not found. Pass --llama-server <path> or set SMARTHIRE_LLAMA_EXE.")
            return self.startup
        if not model:
            self.startup = StartupResult(False, "model_not_found", exe=exe, detail="No .gguf found. Pass --model <path> or set SMARTHIRE_LLAMA_MODEL.")
            return self.startup
        self.model_name = Path(model).name
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [exe, "-m", model, "-c", str(self.ctx), "--host", self.host, "--port", str(self.port), "-np", "1", "--no-webui"]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        verbose(f"  LLM: launching {Path(exe).name} with {self.model_name} (log: {self.log_path})")
        try:
            logf = open(self.log_path, "w", encoding="utf-8", errors="replace")
            flags = 0x08000000 if os.name == "nt" else 0
            if os.name == "nt" and self.keep_running:
                flags |= 0x00000200                          # CREATE_NEW_PROCESS_GROUP: survives Ctrl-C of the runner
            self.proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, creationflags=flags)
        except OSError as e:
            self.startup = StartupResult(False, "launch_failed", time.time() - t0, exe, model, str(e))
            return self.startup
        deadline = t0 + cfg.LLAMA_STARTUP_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.startup = StartupResult(False, "exited_early", time.time() - t0, exe, model, f"exit code {self.proc.returncode}. Log tail:\n{self._log_tail()}")
                return self.startup
            if self.health():
                self.ready = True
                self.startup = StartupResult(True, "started", time.time() - t0, exe, model)
                return self.startup
            time.sleep(0.4)
        self.stop(force=True)
        self.startup = StartupResult(False, "timeout", time.time() - t0, exe, model, f"not ready after {cfg.LLAMA_STARTUP_TIMEOUT_S}s. Log tail:\n{self._log_tail()}")
        return self.startup

    def warm_up(self):
        """Prime the prompt cache in the background (first generation is the slow one)."""
        def _w():
            t = time.time()
            self.json("Reply with JSON.", "Return {\"ok\": true}", purpose="warmup", schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}, max_tokens=8, timeout=90)
            self.startup.gen_latency_s = round(time.time() - t, 1) if self.startup else 0
        if self.ready:
            self._warm = threading.Thread(target=_w, daemon=True)
            self._warm.start()

    def _log_tail(self, n=12) -> str:
        try:
            return "\n".join(Path(self.log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return "(no log)"

    def stop(self, force: bool = False):
        if self.proc and self.proc.poll() is None and (force or not self.keep_running):
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except Exception:
                self.proc.kill()
        if force or not self.keep_running:
            self.proc = None
            self.ready = False

    def _rec(self, purpose, ok, t0, reason=""):
        self.records.append(CallRecord(purpose, ok, round(time.time() - t0, 2), reason))

    def json(self, system: str, user: str, purpose: str, schema: Optional[dict] = None, max_tokens: int = 128, timeout: float = 30, retries: int = cfg.LLM_JSON_RETRIES) -> Optional[dict]:
        """Strict-JSON call. Returns a schema-validated dict or None (caller falls back to deterministic logic)."""
        for attempt in range(retries + 1):
            t0 = time.time()
            if not self.enabled or not self.ready:
                self._rec(purpose, False, t0, "unavailable")
                return None
            # Qwen2.5 ChatML prompt; /completion + json_schema => grammar-constrained output
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": cfg.LLM_TEMPERATURE if attempt == 0 else 0.0, "cache_prompt": True, "stream": False, "stop": ["<|im_end|>"]}
            if schema is not None and self._schema_ok:
                payload["json_schema"] = schema
            try:
                _, body = self._http("/completion", payload, timeout)
                txt = json.loads(body).get("content", "")
            except urllib.error.HTTPError as e:
                if e.code == 400 and "json_schema" in payload:
                    self._schema_ok = False                  # older build: fall back to prompt-only JSON + our own validator
                    continue
                self._rec(purpose, False, t0, f"http_{e.code}")
                return None
            except (TimeoutError, urllib.error.URLError, ConnectionError, OSError) as e:
                self._rec(purpose, False, t0, "timeout" if "timed out" in str(e).lower() else "server_unavailable")
                return None
            except ValueError:
                self._rec(purpose, False, t0, "invalid_response")
                return None
            obj = parse_json_loose(txt)
            if obj is not None and (schema is None or _check_schema(obj, schema)):
                self._rec(purpose, True, t0)
                return obj
            self._rec(purpose, False, t0, "malformed_json" if obj is None else "schema_violation")
            user = user + "\nReturn ONLY one valid JSON object that follows the schema."
        return None

    def summary(self) -> Dict:
        by: Dict[str, List[float]] = {}
        for r in self.records:
            if r.purpose != "warmup":
                by.setdefault(r.purpose, []).append(r.latency_s)
        fails: Dict[str, int] = {}
        for r in self.records:
            if not r.ok:
                fails[r.reason] = fails.get(r.reason, 0) + 1
        return {"calls": len([r for r in self.records if r.purpose != "warmup"]), "failures": fails, "avg_latency_s": {k: round(sum(v) / len(v), 2) for k, v in by.items()}}


class StubLLM:
    """Deterministic stand-in used by the automated tests to exercise the LLM code paths (validation, fallback, grounding)."""
    ready = True
    enabled = True

    def __init__(self, responder=None):
        self.responder = responder or (lambda purpose, user: None)
        self.calls: List[tuple] = []
        self.records: List[CallRecord] = []

    def json(self, system, user, purpose, schema=None, max_tokens=128, timeout=30, retries=0):
        self.calls.append((purpose, user))
        out = self.responder(purpose, user)
        ok = isinstance(out, dict) and (schema is None or _check_schema(out, schema))
        self.records.append(CallRecord(purpose, ok, 0.0, "" if ok else "schema_violation"))
        return out if ok else None

    def summary(self):
        return {"calls": len(self.calls), "failures": {}, "avg_latency_s": {}}
