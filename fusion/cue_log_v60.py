"""
cue_log_v60.py — plain-text storage of every cue, with timestamps.

One folder per interview run (outputs/cues/<YYYYmmdd_HHMMSS>/):
    vision_cues.txt      one line per vision window (gaze off-screen share, reading-like sweeps, face lost, blink/adaptor rates, ...)
    voice_cues.txt       one line per voice window + one summary line per answer (pitch SD, rate, pauses, disfluency, SER arousal, latency, drift)
    text_cues.txt        one line per sentence (start time, per-sentence quality, hedging) + per-answer text cues (detail density, relevance, unlisted tech)
    integrity_flags.txt  every combined / single-modality integrity event with start-end times and the evidence behind it
    cue_log.jsonl        the same rows as machine-readable JSON (one object per line)
Times are printed as mm:ss.s from the start of the interview and as +s inside the answer, so a reviewer can jump to the exact place in the recording.
Writes are flushed immediately: a crash never loses what was already measured.  Nothing here changes any score.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def mmss(t: Optional[float]) -> str:
    if t is None:
        return "--:--.-"
    t = max(0.0, float(t))
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def _fmt(v: Any) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt(x) for x in v) + "]"
    return str(v)


class CueLog:
    FILES = {"vision": "vision_cues.txt", "voice": "voice_cues.txt", "text": "text_cues.txt", "flag": "integrity_flags.txt"}

    def __init__(self, out_dir: Optional[Path] = None, session_t0: Optional[float] = None, enabled: bool = True):
        self.enabled = enabled
        self.t0 = time.monotonic() if session_t0 is None else session_t0
        self.dir: Optional[Path] = None
        self._fh: Dict[str, Any] = {}
        self.rows = 0
        if not enabled:
            return
        if out_dir is None:
            try:
                import config as C
                out_dir = C.OUTPUTS_DIR / "cues"
            except Exception:
                out_dir = Path("outputs") / "cues"
        self.dir = Path(out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            for k, f in self.FILES.items():
                self._fh[k] = open(self.dir / f, "w", encoding="utf-8")
            self._fh["jsonl"] = open(self.dir / "cue_log.jsonl", "w", encoding="utf-8")
            hdr = {"vision": "# vision cues  (interview time | answer | window start-end in answer, s | cue=value ...)",
                   "voice": "# voice cues   (interview time | answer | window start-end in answer, s | cue=value ...)",
                   "text": "# text cues    (interview time | answer | sentence start in answer, s | cue=value ...)",
                   "flag": "# integrity events  (REVIEW indicators, never proof of anything; see the evidence on each line)"}
            for k, h in hdr.items():
                self._fh[k].write(h + "\n")
        except OSError:
            self.enabled = False                       # a log that cannot be written must never stop the interview

    def rel(self, mono: Optional[float]) -> Optional[float]:
        return None if mono is None else mono - self.t0

    def write(self, kind: str, turn: int, t_answer0_rel: Optional[float], t0: Optional[float], t1: Optional[float], **fields) -> None:
        """kind: vision | voice | text | flag.  t0/t1 are seconds INSIDE the answer; t_answer0_rel = answer start in interview seconds."""
        if not self.enabled or kind not in self._fh:
            return
        abs_t = (t_answer0_rel + t0) if (t_answer0_rel is not None and t0 is not None) else None
        span = f"+{t0:.1f}s..+{t1:.1f}s" if (t0 is not None and t1 is not None) else (f"+{t0:.1f}s" if t0 is not None else "")
        line = f"[{mmss(abs_t)}] q{turn} {span} | " + " ".join(f"{k}={_fmt(v)}" for k, v in fields.items())
        try:
            self._fh[kind].write(line + "\n")
            self._fh[kind].flush()
            self._fh["jsonl"].write(json.dumps({"kind": kind, "turn": turn, "interview_s": abs_t, "answer_s": [t0, t1], **{k: v for k, v in fields.items()}}, default=str, ensure_ascii=False) + "\n")
            self._fh["jsonl"].flush()
            self.rows += 1
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        for f in self._fh.values():
            try:
                f.close()
            except Exception:
                pass
        self._fh = {}

    def summary(self) -> dict:
        return {"folder": str(self.dir) if self.dir else None, "rows": self.rows, "files": list(self.FILES.values()) + ["cue_log.jsonl"]}
