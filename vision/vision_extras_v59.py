"""
vision_extras_v59.py — cues the Vision v3 pipeline did not compute, derived from the per-frame samples it already produces.

Input: a list of frame samples (the pipeline's `FrameSample`, or anything with the attributes t, face_ok, gaze_dx, gaze_dy, gaze_valid,
yaw_dev, pitch_dev).  No OpenCV / MediaPipe import here, so it is testable offline.

CUES
  head_nod_rate_per_min / head_shake_rate_per_min : oscillation counting on pitch_dev / yaw_dev.      Conversation-management cue (coaching);
                                                    not used as deception evidence.
  reading_score        (0..1) : how much the horizontal gaze trace looks like reading text - slow drift one way then a fast return
                                (saw-tooth) with little vertical change.                         INTEGRITY, EXPERIMENTAL: no validation study was
                                found; calibrate on your own recordings of people reading vs speaking freely before using any threshold.
  offscreen_fixation_s : longest run where gaze sits away from the screen centre in ONE direction with low dispersion (second monitor / notes).
                         Natural 'thinking' glances are shorter, so only long runs matter.             INTEGRITY, EXPERIMENTAL.
  eye_contact_ratio    : share of valid time looking at the screen.                                    COACHING (Student mode).
Gaze aversion itself is NOT diagnostic of deception (Luke et al. 2023, expert survey) — these are about integrity and coaching.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class VisionExtras:
    usable: bool = False
    n_valid: int = 0
    duration_s: float = 0.0
    head_nod_rate_per_min: Optional[float] = None
    head_shake_rate_per_min: Optional[float] = None
    reading_score: Optional[float] = None
    offscreen_fixation_s: Optional[float] = None
    eye_contact_ratio: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _g(s, name, default=None):
    return getattr(s, name, s.get(name, default) if isinstance(s, dict) else default)


def _resample(t: np.ndarray, x: np.ndarray, fs: float = 20.0):
    ok = np.isfinite(x)
    if ok.sum() < 8:
        return None, None
    tt = np.arange(t[ok][0], t[ok][-1], 1.0 / fs)
    return tt, np.interp(tt, t[ok], x[ok])


def _oscillations(sig: np.ndarray, fs: float, min_p2p: float, min_period: float = 0.25, max_period: float = 1.6) -> int:
    """Count head-gesture cycles: detrended signal, count full swings (peak-to-peak >= min_p2p) whose half-period is within bounds."""
    if len(sig) < int(fs * 1.0):
        return 0
    k = max(3, int(fs * 2.0))
    trend = np.convolve(sig, np.ones(k) / k, mode="same")
    x = sig - trend
    # local extrema
    d = np.diff(x)
    ext = [i for i in range(1, len(d)) if d[i - 1] * d[i] < 0]
    cycles, i = 0, 0
    while i + 2 < len(ext):
        a, b, c = ext[i], ext[i + 1], ext[i + 2]
        p2p1, p2p2 = abs(x[b] - x[a]), abs(x[c] - x[b])
        half1, half2 = (b - a) / fs, (c - b) / fs
        if min(p2p1, p2p2) >= min_p2p and min_period / 2 <= half1 <= max_period / 2 and min_period / 2 <= half2 <= max_period / 2:
            cycles += 1
            i += 2
        else:
            i += 1
    return cycles


def _count_sweeps(x: np.ndarray, fs: float) -> int:
    """Slow rise (>= 0.6 s) followed by a fast fall (<= 0.3 s) of comparable size.  `x` is scaled so a typical large gaze offset ~ 1."""
    win = max(2, int(0.3 * fs))
    n, i, count = len(x), 0, 0
    while i < n - win:
        jump = x[i] - x[i + win]
        if jump >= 0.9:                                           # fast fall candidate
            j = i + int(np.argmin(x[i:i + win + 1]))
            lo_i = max(0, i - int(4.0 * fs))
            seg = x[lo_i:i + 1]
            k = lo_i + int(np.argmin(seg))                        # start of the slow rise
            rise = x[i] - x[k]
            up = x[k + win:i + 1] - x[k:i + 1 - win] if i + 1 - k > win else np.array([0.0])
            gradual = (up.size == 0) or float(np.max(up)) <= 0.5 * rise          # a glance rises abruptly; reading drifts
            if (i - k) / fs >= 0.6 and rise >= 0.6 * jump and rise >= 0.7 and gradual:
                count += 1
                i = j + 1
                continue
        i += 1
    return count


def _reading_score(t: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> Optional[float]:
    """Saw-tooth detector on horizontal gaze, in BOTH directions (English reads left->right, Urdu right->left), weighted by low vertical
    dispersion.  Returns 0..1 (0 = no reading-like sweeps).  EXPERIMENTAL / uncalibrated."""
    fs = 20.0
    tt, x = _resample(t, dx, fs)
    if tt is None or tt[-1] - tt[0] < 6.0:
        return None
    k = max(1, int(0.1 * fs))
    x = np.convolve(x - np.median(x), np.ones(k) / k, mode="same")
    x = x / (np.percentile(np.abs(x), 90) + 1e-6)
    sweeps = max(_count_sweeps(x, fs), _count_sweeps(-x, fs))
    rate = sweeps / (tt[-1] - tt[0])                       # sweeps per second; line-by-line reading gives roughly 0.2-0.5 /s
    if dy is not None and np.isfinite(dy).sum() > 8:
        vy = float(np.nanstd(dy)) / (np.nanstd(dx) + 1e-6)
        vert = 1.0 / (1.0 + max(0.0, vy - 0.6) * 2.0)
    else:
        vert = 0.7
    return float(min(1.0, rate / 0.25) * vert)


def _offscreen_run(t: np.ndarray, dx: np.ndarray, dy: np.ndarray, valid: np.ndarray, away_thr: float = 1.0, disp_thr: float = 0.5) -> float:
    """Longest run (s) where gaze is AWAY in one consistent direction and barely moves.  gaze_dx from Vision v3 is in units of the person's own
    tolerance (|dx| < 1 = looking at the screen, see classify_gaze_direction), so away_thr = 1.0 matches the pipeline's own definition."""
    best, start = 0.0, None
    for i in range(len(t)):
        away = valid[i] and np.isfinite(dx[i]) and abs(dx[i]) > away_thr
        if away and start is None:
            start = i
        if (not away or i == len(t) - 1) and start is not None:
            end = i if not away else i + 1
            seg = dx[start:end]
            if end - start >= 3 and np.sign(np.median(seg)) != 0 and np.all(np.sign(seg) == np.sign(np.median(seg))) and np.nanstd(seg) < disp_thr:
                best = max(best, float(t[end - 1] - t[start]))
            start = None
    return best


def vision_extras(samples: Sequence, head_amp_thr: float = 2.0) -> VisionExtras:
    """head_amp_thr: peak-to-peak amplitude (same unit as the pipeline's pitch_dev/yaw_dev, degrees in v3) for a gesture to count."""
    r = VisionExtras()
    if not samples or len(samples) < 20:
        r.notes.append("too few frames")
        return r
    t = np.array([_g(s, "t", np.nan) for s in samples], float)
    face = np.array([bool(_g(s, "face_ok", False)) for s in samples])
    gv = np.array([bool(_g(s, "gaze_valid", True)) for s in samples])
    dx = np.array([_g(s, "gaze_dx", np.nan) if _g(s, "gaze_dx", None) is not None else np.nan for s in samples], float)
    dy = np.array([_g(s, "gaze_dy", np.nan) if _g(s, "gaze_dy", None) is not None else np.nan for s in samples], float)
    # Vision v3 stores yaw_dev as an ABSOLUTE deviation, which folds a left-right shake into one-sided bumps. Prefer a signed yaw if the sample has one.
    signed_yaw = any(_g(s, "yaw_signed", None) is not None for s in samples)
    yname = "yaw_signed" if signed_yaw else "yaw_dev"
    yaw = np.array([_g(s, yname, np.nan) if _g(s, yname, None) is not None else np.nan for s in samples], float)
    pit = np.array([_g(s, "pitch_dev", np.nan) if _g(s, "pitch_dev", None) is not None else np.nan for s in samples], float)
    r.duration_s = float(t[-1] - t[0])
    ok = face & gv
    r.n_valid = int(ok.sum())
    if r.duration_s < 5 or ok.mean() < 0.6:
        r.notes.append("face/gaze valid for < 60% of the segment")
        return r
    r.usable = True
    mins = r.duration_s / 60.0
    dxv, dyv = dx.copy(), dy.copy()
    dxv[~ok], dyv[~ok] = np.nan, np.nan
    r.eye_contact_ratio = float(np.mean([bool(_g(s, "looking", False)) for s, o in zip(samples, ok) if o])) if any(_g(s, "looking", None) is not None for s in samples) else None
    tt, ps = _resample(t, pit, 20.0)
    r.head_nod_rate_per_min = float(_oscillations(ps, 20.0, head_amp_thr) / mins) if tt is not None else None
    tt, ys = _resample(t, yaw, 20.0)
    if tt is not None:
        folded = (not signed_yaw) and bool(np.all(ys >= 0))
        cycles = _oscillations(ys, 20.0, head_amp_thr)
        r.head_shake_rate_per_min = float((cycles / 2.0 if folded else cycles) / mins)      # |yaw| shows 2 bumps per left-right cycle
        if folded:
            r.notes.append("yaw_dev is absolute in Vision v3: shake rate is approximate; add a signed yaw (yaw_signed) to FrameSample for exact counts")
    r.reading_score = _reading_score(t, dxv, dyv)
    r.offscreen_fixation_s = _offscreen_run(t, dxv, dyv, ok)
    return r
