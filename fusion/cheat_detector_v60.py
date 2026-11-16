"""
cheat_detector_v60.py — time-aligned detector for a "possible external assistance" PATTERN inside one answer.

THE PATTERN (what the interviewer actually wants to notice)
    gaze drifts away from the camera (a slight turn, a glance down/aside, a sweep that looks like reading)
 +  within a few seconds the VOICE changes character (flatter pitch, steadier/faster rate, fewer pauses and fillers  -> reads like reading aloud)
 +  the CONTENT jumps: vague / hedged / off-topic sentences, then suddenly fluent, on-topic, technically rich ones.
No single channel is evidence.  An event is raised only when >= 2 channels are active in the same short interval (gaze and text may PRECEDE the voice
change by up to LAG_S seconds), and the report says "review", never "cheating".  Every number below is a [prior] set by hand, NOT a validated threshold:
calibrate on recordings of people reading vs speaking freely before relying on it (research/ETHICS_AND_LIMITATIONS.md).

CHANNELS (all computed per 3 s window, hop 1.5 s, in seconds from the start of the answer audio)
  vision  gaze_off  share of valid frames with |gaze_dx| or |gaze_dy| above the pipeline's own 'looking at the screen' tolerance (units of the person's
                    calibrated tolerance), compared with the candidate's own introduction answer; reading_score from vision_extras (saw-tooth sweeps)
  voice   f0_sd (semitones), articulation rate (words / speaking second), disfluency rate (fillers, repeats), pause share, optional SER arousal (per chunk);
          each as a robust z-score against the candidate's own baseline (introduction answer) AND against the earlier part of the same answer
  text    per-sentence quality (relevance to the question + technical substance - hedging); a jump from <= TEXT_LOW to >= TEXT_HIGH is a 'wrong -> right' event
Fusion: strength s_m in [0,1] per channel; window score = 1 - prod(1 - 0.7 s_m) over channels with s_m >= ACTIVE; severity 'strong' needs a high score
and either all three channels or gaze+voice with a reading-like gaze trace.
Honest limits: a second monitor used for the interview itself, notes the candidate is allowed to use, thinking glances, a nervous start that settles,
an accent/mic change, or a topic the candidate simply knows well produce the same pattern.  The detector therefore also asks for a verification probe
(the interviewer asks the candidate to explain the idea in their own words) instead of making a verdict.
"""
from __future__ import annotations

import math
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import vision_extras_v59 as VX
import voice_cues_v59 as VC
from text_cues_v59 import sentences as split_sentences


# ───────────────────────────── tunables ([prior] = hand-set, not evidence) ─────────────────────────────
@dataclass
class Params:
    win_s: float = 3.0
    hop_s: float = 1.5
    min_answer_s: float = 8.0           # [guard] shorter answers are not analysed
    gaze_away: float = 0.8              # [prior] |gaze_dx| (units of the candidate's tolerance; 1.0 = the pipeline's own limit) counted as 'away' (slight turn)
    gaze_margin: float = 0.15           # [prior] extra off-screen share above the candidate's baseline before strength > 0
    gaze_full: float = 0.60             # [prior] extra share at which strength = 1
    min_face_share: float = 0.5         # [guard] below this the face is mostly not visible: reported as 'face_lost', not as looking away
    min_valid_frames: int = 8           # [guard]
    read_window_s: float = 8.0          # reading_score needs >= 6 s of gaze trace
    read_lo: float = 0.40               # [prior] reading_score mapped 0 -> 1 between read_lo and read_hi
    read_hi: float = 0.80
    min_speech_win_s: float = 1.2       # [guard] voice features only where this much speech exists in the window
    voice_start: float = 0.6            # [prior] mean oriented z at which strength starts
    voice_span: float = 1.4             # [prior] ... and the additional z for strength = 1
    text_low: float = 0.30              # [prior]
    text_high: float = 0.50             # [prior]
    text_delta: float = 0.35            # [prior]
    text_min_words: int = 6
    text_before_s: float = 1.0          # a text jump marks [t - before, t + after]
    text_after_s: float = 4.0
    active: float = 0.40                # [prior] channel counts as 'active' at this strength
    lag_s: float = 3.0                  # gaze / text may precede the voice change by this much
    weight: float = 0.70                # [prior] per-channel weight in the noisy-OR window score
    strong_score: float = 0.75          # [prior]
    min_baseline_windows: int = 4       # [guard] below this the voice reference is the early part of the same answer
    self_ref_windows: int = 3
    ser_chunk_min_s: float = 2.0


SCALE_FLOOR = {"f0_sd": 0.35, "rate": 0.35, "disfl": 0.03, "pause": 0.06, "ser": 0.05}
FILLERS = {"um", "umm", "uh", "uhh", "er", "erm", "ah", "hmm", "mm", "mmm"}
ALTERNATIVE_EXPLANATIONS = ("a second monitor or notes the candidate was allowed to use", "ordinary thinking glances", "a nervous start that settles as the candidate relaxes",
                            "a microphone/room change", "a topic the candidate simply knows well")


# ───────────────────────────── containers ─────────────────────────────
@dataclass
class Window:
    t0: float
    t1: float
    gaze_off: Optional[float] = None
    face_lost: Optional[float] = None
    reading: Optional[float] = None
    f0_sd: Optional[float] = None
    rate: Optional[float] = None
    disfl: Optional[float] = None
    pause: Optional[float] = None
    ser: Optional[float] = None
    z: Dict[str, float] = field(default_factory=dict)
    s_v: float = 0.0
    s_a: float = 0.0
    s_t: float = 0.0
    score: float = 0.0
    active: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["z"] = {k: round(v, 2) for k, v in self.z.items()}
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass
class Event:
    turn: int
    t0: float                       # seconds from the start of the answer
    t1: float
    interview_t0: Optional[float]   # seconds from the start of the interview
    interview_t1: Optional[float]
    channels: List[str]
    score: float
    severity: str                   # 'review' | 'strong'
    evidence: List[str]
    note: str = "possible external-assistance pattern; needs verification (never proof)"

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["alternative_explanations"] = list(ALTERNATIVE_EXPLANATIONS)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass
class AnswerAnalysis:
    turn: int
    analysed: bool = False
    reason: str = ""
    reference: str = "none"         # baseline | self | none
    windows: List[Window] = field(default_factory=list)
    sentences: List[dict] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    text_jumps: List[dict] = field(default_factory=list)

    @property
    def max_score(self) -> float:
        return max([e.score for e in self.events] or [0.0])

    @property
    def severity(self) -> str:
        return "strong" if any(e.severity == "strong" for e in self.events) else ("review" if self.events else "none")

    def to_dict(self) -> dict:
        return {"turn": self.turn, "analysed": self.analysed, "reason": self.reason, "reference": self.reference, "severity": self.severity,
                "events": [e.to_dict() for e in self.events], "text_jumps": self.text_jumps, "windows": [w.to_dict() for w in self.windows], "sentences": self.sentences}


# ───────────────────────────── helpers ─────────────────────────────
def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _grid(duration: float, win: float, hop: float) -> List[Tuple[float, float]]:
    out, t = [], 0.0
    while t + 0.6 * win <= duration + 1e-9:
        out.append((t, min(t + win, duration)))
        t += hop
    return out


def sentence_times(answer_text: str, segments: Optional[Sequence[Tuple[float, float, str]]], speech: Sequence[Tuple[float, float]], duration: float) -> List[Tuple[float, float, str]]:
    """[(start_s, end_s, sentence)] inside the answer.  Whisper segments give real times (sentences inside a segment share it by length);
    without them the sentences are spread over the speech span by character count."""
    out: List[Tuple[float, float, str]] = []
    if segments:
        for s, e, txt in segments:
            sents = split_sentences(txt) or ([txt] if txt.strip() else [])
            tot = sum(len(x) for x in sents) or 1
            cur = float(s)
            for x in sents:
                d = (float(e) - float(s)) * len(x) / tot
                out.append((cur, cur + d, x))
                cur += d
        return out
    sents = split_sentences(answer_text)
    if not sents:
        return out
    a = speech[0][0] if speech else 0.0
    b = speech[-1][1] if speech else duration
    tot = sum(len(x) for x in sents) or 1
    cur = a
    for x in sents:
        d = (b - a) * len(x) / tot
        out.append((cur, cur + d, x))
        cur += d
    return out


# ───────────────────────────── baseline ─────────────────────────────
class Baseline:
    """Robust (median, 1.4826 MAD) reference of window features from the candidate's introduction answer (and later unflagged windows)."""

    def __init__(self):
        self.vals: Dict[str, List[float]] = {k: [] for k in ("f0_sd", "rate", "disfl", "pause", "ser", "gaze_off")}

    def update(self, windows: Sequence[Window]) -> None:
        for w in windows:
            for k, v in (("f0_sd", w.f0_sd), ("rate", w.rate), ("disfl", w.disfl), ("pause", w.pause), ("ser", w.ser), ("gaze_off", w.gaze_off)):
                if v is not None and np.isfinite(v):
                    self.vals[k].append(float(v))

    def n(self, k: str = "f0_sd") -> int:
        return len(self.vals[k])

    def stat(self, k: str) -> Optional[Tuple[float, float]]:
        v = np.asarray(self.vals[k], float)
        if len(v) < 3:
            return None
        med = float(np.median(v))
        mad = float(1.4826 * np.median(np.abs(v - med)))
        return med, max(mad, SCALE_FLOOR.get(k, 0.05))


def _z(x: Optional[float], ref: Optional[Tuple[float, float]]) -> Optional[float]:
    if x is None or ref is None or not np.isfinite(x):
        return None
    return float(np.clip((x - ref[0]) / ref[1], -4.0, 4.0))


# ───────────────────────────── per-window features ─────────────────────────────
def vision_windows(wins: List[Window], samples: Sequence, offset: float, capture_t0: float, p: Params) -> bool:
    """Fill gaze_off / face_lost / reading.  `offset`: monotonic = sample.t + offset.  Returns True if any vision data was usable."""
    if not samples:
        return False
    t = np.array([getattr(s, "t", np.nan) + offset - capture_t0 for s in samples], float)
    face = np.array([bool(getattr(s, "face_ok", False)) for s in samples])
    gv = np.array([bool(getattr(s, "gaze_valid", True)) for s in samples])
    dx = np.array([getattr(s, "gaze_dx", None) if getattr(s, "gaze_dx", None) is not None else np.nan for s in samples], float)
    dy = np.array([getattr(s, "gaze_dy", None) if getattr(s, "gaze_dy", None) is not None else np.nan for s in samples], float)
    any_ok = False
    for w in wins:
        m = (t >= w.t0) & (t < w.t1)
        if m.sum() < p.min_valid_frames:
            continue
        w.face_lost = float(1.0 - face[m].mean())
        ok = m & face & gv & np.isfinite(dx)
        if w.face_lost > p.min_face_share or ok.sum() < p.min_valid_frames:
            continue
        away = (np.abs(dx[ok]) > p.gaze_away) | (np.nan_to_num(np.abs(dy[ok]), nan=0.0) > p.gaze_away)
        w.gaze_off = float(away.mean())
        any_ok = True
        mr = (t >= w.t1 - p.read_window_s) & (t < w.t1) & face & gv & np.isfinite(dx)
        if mr.sum() >= 40 and (t[mr][-1] - t[mr][0]) >= 6.0:
            try:
                w.reading = VX._reading_score(t[mr], dx[mr], dy[mr])
            except Exception:
                w.reading = None
    return any_ok


def voice_windows(wins: List[Window], audio: np.ndarray, sr: int, speech: Sequence[Tuple[float, float]], words: Optional[Sequence[Tuple[float, float, str]]],
                  ser_chunks: Optional[Sequence[Tuple[float, float, Dict]]], p: Params) -> bool:
    if audio is None or len(audio) < sr * 2:
        return False
    f0 = VC.f0_track(np.asarray(audio, np.float32), sr)
    hop = 0.01
    for w in wins:
        a, b = int(w.t0 / hop), int(w.t1 / hop)
        seg = f0[a:b] if a < len(f0) else np.array([])
        st = VC.semitones(seg)
        w.f0_sd = float(np.std(st)) if len(st) >= 15 else None
        sp = sum(max(0.0, min(e, w.t1) - max(s, w.t0)) for s, e in speech)
        w.pause = float(_clip01(1.0 - sp / max(w.t1 - w.t0, 1e-6)))
        if words:
            ww = [x for x in words if w.t0 <= 0.5 * (x[0] + x[1]) < w.t1]
            toks = [re.sub(r"[^a-z']", "", x[2].lower()) for x in ww]
            toks = [x for x in toks if x]
            if sp >= p.min_speech_win_s and len(toks) >= 3:
                w.rate = float(len(toks) / sp)
            if len(toks) >= 4:
                fill = sum(1 for x in toks if x in FILLERS)
                rep = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1] and len(toks[i]) > 1)
                w.disfl = float((fill + rep) / len(toks))
        if ser_chunks:
            vals = []
            for c0, c1, res in ser_chunks:
                if min(c1, w.t1) - max(c0, w.t0) >= 0.5 * min(c1 - c0, w.t1 - w.t0):
                    m1 = (res or {}).get("model1_dimensional") or {}
                    if "arousal" in m1:
                        vals.append(float(m1["arousal"]))
            if vals:
                w.ser = float(np.mean(vals))
    return True


# ───────────────────────────── the monitor ─────────────────────────────
class CheatMonitor:
    def __init__(self, params: Optional[Params] = None, cuelog=None, relevance=None, session_t0: Optional[float] = None):
        self.p = params or Params()
        self.cuelog = cuelog
        self.relevance = relevance
        self.baseline = Baseline()
        self.session_t0 = session_t0
        self.analyses: List[AnswerAnalysis] = []
        self.errors: List[str] = []

    # ---- main entry ----
    def analyze_answer(self, turn: int, bucket: str, question: str, answer_text: str, *, audio=None, sr: int = 16000, capture_t0: Optional[float] = None,
                       vision_samples: Optional[Sequence] = None, vision_offset: float = 0.0, words=None, segments=None, ser_chunks=None,
                       speech: Optional[Sequence[Tuple[float, float]]] = None, target_keys: Sequence[str] = ()) -> AnswerAnalysis:
        res = AnswerAnalysis(turn)
        try:
            self._analyze(res, bucket, question, answer_text, audio, sr, capture_t0, vision_samples, vision_offset, words, segments, ser_chunks, speech, target_keys)
        except Exception:
            self.errors.append(f"cheat_detector q{turn}: " + traceback.format_exc(limit=3))
            res.analysed, res.reason = False, "internal error (logged)"
        self.analyses.append(res)
        return res

    def _analyze(self, res, bucket, question, answer_text, audio, sr, capture_t0, vision_samples, vision_offset, words, segments, ser_chunks, speech, target_keys):
        p = self.p
        has_audio = audio is not None and len(audio) >= sr * 2 and capture_t0 is not None
        has_vision = bool(vision_samples) and capture_t0 is not None
        if not has_audio:
            res.reason = "no audio for this answer (typed or too short): timing channels need the microphone"
            return
        duration = len(audio) / sr
        if duration < p.min_answer_s:
            res.reason = f"answer shorter than {p.min_answer_s:.0f}s"
            return
        speech = list(speech) if speech is not None else VC.energy_vad(np.asarray(audio, np.float32), sr)
        wins = [Window(a, b) for a, b in _grid(duration, p.win_s, p.hop_s)]
        v_ok = vision_windows(wins, vision_samples, vision_offset, capture_t0, p) if has_vision else False
        a_ok = voice_windows(wins, audio, sr, speech, words, ser_chunks, p)
        # ---- text: per-sentence quality with times ----
        st = sentence_times(answer_text, segments, speech, duration)
        q = self._sentence_quality(question, [x[2] for x in st], target_keys)
        res.sentences = [{"t0": round(a, 2), "t1": round(b, 2), "quality": round(qq, 3), "text": s[:140]} for (a, b, s), qq in zip(st, q)]
        jumps = self._text_jumps(st, q)
        res.text_jumps = jumps
        # ---- references ----
        use_base = self.baseline.n("f0_sd") >= p.min_baseline_windows
        if bucket == "intro" or not use_base:
            pass
        res.reference = "baseline" if use_base else ("self" if len([w for w in wins if w.f0_sd is not None]) >= 2 * p.self_ref_windows else "none")
        refs = {k: self.baseline.stat(k) for k in ("f0_sd", "rate", "disfl", "pause", "ser")} if use_base else {}
        base_off = (self.baseline.stat("gaze_off") or (0.0, 0.1))[0] if self.baseline.n("gaze_off") >= 3 else 0.0
        if res.reference == "self":                      # early part of the same answer is the only reference available
            head = [w for w in wins[: max(p.self_ref_windows, 2)] if w.f0_sd is not None]
            tmp = Baseline()
            tmp.update(head)
            refs = {k: (tmp.stat(k) if tmp.n(k) >= 3 else (np.median(tmp.vals[k]).item(), SCALE_FLOOR[k]) if tmp.vals[k] else None) for k in ("f0_sd", "rate", "disfl", "pause", "ser")}
        orient = {"f0_sd": -1, "rate": +1, "disfl": -1, "pause": -1, "ser": -1}
        # ---- per-window strengths ----
        for i, w in enumerate(wins):
            # vision
            sv = 0.0
            if w.gaze_off is not None:
                sv = _clip01((w.gaze_off - base_off - p.gaze_margin) / max(p.gaze_full - p.gaze_margin, 1e-6))
            if w.reading is not None:
                sv = max(sv, _clip01((w.reading - p.read_lo) / max(p.read_hi - p.read_lo, 1e-6)))
            w.s_v = sv
            # voice: z against the baseline/self reference and against the previous windows of this answer
            zs, zl = [], []
            prev = [x for x in wins[max(0, i - 3):i] if x.f0_sd is not None or x.rate is not None]
            for k, attr in (("f0_sd", "f0_sd"), ("rate", "rate"), ("disfl", "disfl"), ("pause", "pause"), ("ser", "ser")):
                x = getattr(w, attr)
                zr = _z(x, refs.get(k))
                if zr is not None:
                    w.z[k] = zr
                    zs.append(orient[k] * zr)
                pv = [getattr(y, attr) for y in prev if getattr(y, attr) is not None]
                if x is not None and len(pv) >= 2:
                    scale = (refs.get(k) or (0.0, SCALE_FLOOR[k]))[1]
                    zl.append(orient[k] * float(np.clip((x - float(np.median(pv))) / scale, -4, 4)))
            def strength(zv):
                return 0.0 if len(zv) < 2 else _clip01((float(np.mean(zv)) - p.voice_start) / p.voice_span)
            sa = strength(zs) if res.reference != "none" else 0.0
            sl = strength(zl) * 0.9
            w.s_a = max(sa, sl) if (w.f0_sd is not None or w.rate is not None) else 0.0
        # text strength spread around each jump
        for j in jumps:
            for w in wins:
                if w.t1 >= j["t"] - p.text_before_s and w.t0 <= j["t"] + p.text_after_s:
                    w.s_t = max(w.s_t, j["strength"])
        # ---- fusion with lag ----
        L = int(math.ceil(p.lag_s / p.hop_s))
        for i, w in enumerate(wins):
            lag = wins[max(0, i - L): i + 1]
            sv, st_ = max(x.s_v for x in lag), max(x.s_t for x in lag)
            act = []
            if sv >= p.active:
                act.append("vision")
            if w.s_a >= p.active:
                act.append("voice")
            if st_ >= p.active:
                act.append("text")
            w.active = act
            w.score = 1.0 - float(np.prod([1.0 - p.weight * s for s in (sv, w.s_a, st_) if s >= p.active] or [1.0]))
        res.windows, res.analysed = wins, True
        res.events = self._events(res, wins, capture_t0)
        # ---- baseline update: introduction always; later answers only if nothing was flagged in them ----
        if bucket == "intro" or not res.events:
            self.baseline.update([w for w in wins if not w.active or bucket == "intro"])
        self._log(res, bucket, capture_t0, v_ok, a_ok)

    # ---- text ----
    def _sentence_quality(self, question, sents, target_keys) -> List[float]:
        if not sents:
            return []
        if self.relevance is None:
            from relevance_v60 import ResumeRelevance
            self.relevance = ResumeRelevance([], use_sbert=False)
        return self.relevance.sentence_quality(question, sents, target_keys)

    def _text_jumps(self, st, q) -> List[dict]:
        p, out = self.p, []
        n = len(q)
        for i in range(1, n):
            before = q[max(0, i - 2):i]
            after = q[i:i + 2]
            words_i = len(st[i][2].split())
            if words_i < p.text_min_words or not before:
                continue
            lo = max(before) <= p.text_low or (float(np.mean(before)) <= p.text_low + 0.05 and len(before) >= 2)
            hi = float(np.mean(after)) >= p.text_high
            delta = float(np.mean(after)) - float(np.mean(before))
            if lo and hi and delta >= p.text_delta:
                out.append({"t": round(st[i][0], 2), "before": round(float(np.mean(before)), 3), "after": round(float(np.mean(after)), 3),
                            "strength": round(_clip01(delta / 0.6), 3), "sentence": st[i][2][:140], "previous": st[i - 1][2][:140]})
        return out

    # ---- events ----
    def _events(self, res: AnswerAnalysis, wins: List[Window], capture_t0: Optional[float]) -> List[Event]:
        p = self.p
        flagged = [i for i, w in enumerate(wins) if len(w.active) >= 2]
        groups: List[List[int]] = []
        for i in flagged:
            if groups and i - groups[-1][-1] <= 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        events = []
        base_t = None if (self.session_t0 is None or capture_t0 is None) else capture_t0 - self.session_t0
        for g in groups:
            ws = [wins[i] for i in g]
            t0, t1 = ws[0].t0, ws[-1].t1
            chans = sorted({c for w in ws for c in w.active})
            score = max(w.score for w in ws)
            ev: List[str] = []
            gv = [w for w in wins[max(0, g[0] - int(math.ceil(p.lag_s / p.hop_s))): g[-1] + 1] if w.gaze_off is not None]
            if "vision" in chans and gv:
                m = max(gv, key=lambda w: w.s_v)
                rd = max([w.reading for w in gv if w.reading is not None] or [0.0])
                ev.append(f"gaze off-screen {m.gaze_off:.0%} of valid frames around +{m.t0:.1f}s" + (f", reading-like sweeps {rd:.2f}" if rd >= p.read_lo else ""))
            if "voice" in chans:
                m = max(ws, key=lambda w: w.s_a)
                zz = ", ".join(f"{k} {v:+.1f}σ" for k, v in m.z.items())
                ev.append(f"voice character shifted at +{m.t0:.1f}s ({res.reference} reference): {zz}")
            if "text" in chans:
                js = [j for j in res.text_jumps if t0 - p.text_after_s - 0.1 <= j["t"] <= t1 + p.text_before_s + 0.1]
                for j in js[:2]:
                    ev.append(f"content jumped at +{j['t']:.1f}s: quality {j['before']:.2f} → {j['after']:.2f}; before: “{j['previous'][:70]}” → after: “{j['sentence'][:70]}”")
            reading_like = any((w.reading or 0) >= 0.6 for w in wins[max(0, g[0] - 5): g[-1] + 1])
            strong = score >= p.strong_score and (len(chans) == 3 or ({"vision", "voice"} <= set(chans) and reading_like))
            events.append(Event(res.turn, round(t0, 2), round(t1, 2), None if base_t is None else round(base_t + t0, 2), None if base_t is None else round(base_t + t1, 2),
                                chans, float(score), "strong" if strong else "review", ev))
        return events

    # ---- logging ----
    def _log(self, res: AnswerAnalysis, bucket: str, capture_t0: Optional[float], v_ok: bool, a_ok: bool) -> None:
        cl = self.cuelog
        if cl is None or not getattr(cl, "enabled", False):
            return
        a0 = None if (capture_t0 is None) else cl.rel(capture_t0)
        for w in res.windows:
            cl.write("vision", res.turn, a0, w.t0, w.t1, gaze_off=w.gaze_off, face_lost=w.face_lost, reading=w.reading, strength=w.s_v)
            cl.write("voice", res.turn, a0, w.t0, w.t1, f0_sd_st=w.f0_sd, rate_wps=w.rate, disfl=w.disfl, pause_share=w.pause, ser_arousal=w.ser, z=w.z, strength=w.s_a)
        for s in res.sentences:
            cl.write("text", res.turn, a0, s["t0"], s["t1"], quality=s["quality"], text=s["text"])
        for e in res.events:
            cl.write("flag", res.turn, a0, e.t0, e.t1, severity=e.severity, score=e.score, channels=e.channels, evidence=" | ".join(e.evidence), note=e.note)

    # ---- report ----
    def report(self) -> dict:
        ev = [e.to_dict() for a in self.analyses for e in a.events]
        return {"schema": "smarthire-assist-pattern/1", "interpretation": "possible external-assistance PATTERNS (gaze + voice + content, aligned in time). Review indicators, not proof; "
                "uncalibrated thresholds (see cheat_detector_v60.py header).", "events": ev,
                "answers": [{"turn": a.turn, "analysed": a.analysed, "reason": a.reason, "reference": a.reference, "severity": a.severity, "max_score": round(a.max_score, 3),
                             "text_jumps": a.text_jumps} for a in self.analyses],
                "summary": {"answers_analysed": sum(1 for a in self.analyses if a.analysed), "events": len(ev), "strong": sum(1 for e in ev if e["severity"] == "strong")},
                "errors": self.errors}
