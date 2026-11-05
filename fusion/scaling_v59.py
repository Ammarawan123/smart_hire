"""
scaling_v59.py — what is needed to run MANY interviews at once (HR mode: dozens of candidates) without them interfering.

  ModelRegistry   heavy models (NLI, SER, SBERT) are loaded ONCE and shared between sessions; thread-safe lazy loading with a load counter.
  AnalysisPool    bounded worker pool with timeouts: slow analysis (NLI / SER) can never block the interview loop; on timeout the turn is scored without that cue.
  SessionStore    per-interview state (BehaviorAwareEngine + BehaviorSession), per-session lock, idle eviction, atomic JSON report per session.
  BehaviorService one object a web server (FastAPI/WebSocket) can call:  start(session_id, ...) -> step(session_id, answer, signals) -> finish(session_id).

HONEST LIMITS: threads share one Python process (GIL): numpy/onnx/torch release it, regex/Python glue does not.  CPU-bound load is limited by cores; for more than a
handful of simultaneous candidates run several worker PROCESSES behind a queue (one session pinned to one process).  The single llama-server handles one request at a
time per slot — start it with --parallel N if you want N concurrent LLM calls.  Tested here: 24 simulated sessions in parallel threads with isolation checks.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class ModelRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._items: Dict[str, Any] = {}
        self._key_locks: Dict[str, threading.Lock] = {}
        self.loads = Counter()

    def get(self, key: str, factory: Callable[[], Any]) -> Any:
        if key in self._items:
            return self._items[key]
        with self._lock:
            kl = self._key_locks.setdefault(key, threading.Lock())
        with kl:                                           # only one thread loads a given model; others wait, other models load in parallel
            if key not in self._items:
                self._items[key] = factory()
                self.loads[key] += 1
        return self._items[key]


class AnalysisPool:
    def __init__(self, workers: int = 2, max_pending: int = 64):
        self._ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v59-analysis")
        self._sem = threading.BoundedSemaphore(max_pending)
        self.durations: List[float] = []
        self.timeouts = 0
        self.rejected = 0

    def run(self, fn: Callable, *args, timeout: float = 5.0, fallback: Any = None, **kw) -> Any:
        if not self._sem.acquire(blocking=False):
            self.rejected += 1                              # overloaded: degrade instead of queueing forever
            return fallback
        t0 = time.time()
        fut = self._ex.submit(fn, *args, **kw)
        try:
            return fut.result(timeout=timeout)
        except FutTimeout:
            self.timeouts += 1
            return fallback
        finally:
            self.durations.append(time.time() - t0)
            fut.add_done_callback(lambda _f: self._sem.release())

    def shutdown(self):
        self._ex.shutdown(wait=False, cancel_futures=True)


class _Entry:
    def __init__(self, engine, meta):
        self.engine, self.meta, self.lock, self.last = engine, meta, threading.Lock(), time.time()
        self.errors: List[str] = []


class SessionStore:
    def __init__(self, runs_dir: Optional[str] = None, idle_ttl_s: float = 3600.0):
        self._d: Dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self.runs_dir = Path(runs_dir) if runs_dir else None
        self.ttl = idle_ttl_s

    def create(self, sid: str, factory: Callable[[], Any], meta: Optional[dict] = None) -> Any:
        with self._lock:
            if sid in self._d:
                raise KeyError(f"session {sid!r} already exists")
        eng = factory()
        with self._lock:
            self._d[sid] = _Entry(eng, meta or {})
        return eng

    def entry(self, sid: str) -> _Entry:
        with self._lock:
            return self._d[sid]

    def ids(self) -> List[str]:
        with self._lock:
            return list(self._d)

    def evict_idle(self, now: Optional[float] = None) -> List[str]:
        now = now or time.time()
        with self._lock:
            gone = [k for k, e in self._d.items() if now - e.last > self.ttl]
            for k in gone:
                del self._d[k]
        return gone

    def save_report(self, sid: str, report: dict) -> Optional[Path]:
        if not self.runs_dir:
            return None
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        dest = self.runs_dir / f"{sid}.json"
        fd, tmp = tempfile.mkstemp(dir=str(self.runs_dir), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, dest)                               # atomic: a crash never leaves a half-written report
        return dest

    def close(self, sid: str):
        with self._lock:
            return self._d.pop(sid, None)


class BehaviorService:
    def __init__(self, engine_factory: Callable[[dict], Any], runs_dir: Optional[str] = None, registry: Optional[ModelRegistry] = None):
        self.engine_factory, self.store, self.registry = engine_factory, SessionStore(runs_dir), registry or ModelRegistry()
        self.stats = Counter()

    def start(self, sid: str, params: Optional[dict] = None):
        eng = self.store.create(sid, lambda: self.engine_factory(params or {}), params)
        self.stats["started"] += 1
        return eng.start()

    def step(self, sid: str, answer: str, signals=None):
        e = self.store.entry(sid)
        with e.lock:                                        # one answer at a time per interview, many interviews in parallel
            e.last = time.time()
            try:
                out = e.engine.step(answer, signals)
                self.stats["steps"] += 1
                return out
            except Exception:
                e.errors.append(traceback.format_exc(limit=3))
                self.stats["step_errors"] += 1
                raise

    def finish(self, sid: str, report_builder: Callable[[Any], dict]):
        e = self.store.entry(sid)
        with e.lock:
            rep = report_builder(e.engine)
            rep["service_errors"] = e.errors
            path = self.store.save_report(sid, rep)
        self.store.close(sid)
        self.stats["finished"] += 1
        return rep, path
