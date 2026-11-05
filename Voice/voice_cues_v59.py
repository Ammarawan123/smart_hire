"""
voice_cues_v59.py — answer-locked voice cues that fix what the V3 Voice module could not know.

WHAT IS NEW vs Voice/acoustic_extractor.py + deception_engine.py
  1. RESPONSE LATENCY FROM THE INTERVIEWER'S OWN CLOCK: latency = first speech onset - TTS-playback end (exact; the old code inferred it from
     3-second VAD windows).  Barge-in (candidate speaks before the question ends) is detected and reported, not scored.
  2. Pause statistics inside the answer (rate, long pauses), articulation rate (words per second of actual speech) and WPM from the transcript.
  3. Pitch in SEMITONES (log scale) so per-person baselines are comparable; pYIN via librosa when installed, otherwise a documented
     autocorrelation fallback.  Loudness variability.
  4. Quality gating (speech duration, SNR proxy) - a cue is None when unreliable instead of a misleading number.
  5. EXPERIMENTAL integrity cue: 'second-voice drift' = change of the long-term spectral envelope inside one answer (a coarse hint of another
     speaker/prompting).  It is NOT speaker verification; use ECAPA-TDNN/pyannote for that if you need it.  Reported separately; never fused.
Jitter/shimmer/HNR stay available through the old module but carry weight 0 in the fusion (no deception-specific validation was found).
Pure numpy/scipy: works without torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ───────────────────────── voice activity (fallback energy VAD) ─────────────────────────
def energy_vad(audio: np.ndarray, sr: int, frame_s: float = 0.03, merge_gap_s: float = 0.15, min_len_s: float = 0.12) -> List[Tuple[float, float]]:
    """Return speech segments [(start_s, end_s)].  Use Silero (Voice module) when available; this is the dependency-free fallback."""
    n = int(sr * frame_s)
    if len(audio) < n * 3:
        return []
    m = len(audio) // n
    fr = audio[: m * n].reshape(m, n)
    e = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    floor = np.percentile(e, 10)
    thr = max(0.008, min(floor * 3.0, 0.25 * np.percentile(e, 95)))   # capped by the loud end so little-silence recordings still segment
    act = e > thr
    segs, start = [], None
    for i, a in enumerate(act):
        if a and start is None:
            start = i
        if (not a) and start is not None:
            segs.append([start * frame_s, i * frame_s]); start = None
    if start is not None:
        segs.append([start * frame_s, m * frame_s])
    merged: List[List[float]] = []
    for s, t in segs:
        if merged and s - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = t
        else:
            merged.append([s, t])
    return [(s, t) for s, t in merged if t - s >= min_len_s]


# ───────────────────────── pitch ─────────────────────────
def f0_track(audio: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0, hop_s: float = 0.01) -> np.ndarray:
    """F0 in Hz per frame, NaN where unvoiced.  librosa.pyin if installed (Mauch & Dixon 2014), else normalised-autocorrelation fallback."""
    try:
        import librosa
        f0, _, _ = librosa.pyin(audio.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=int(sr * hop_s))
        return np.asarray(f0, float)
    except Exception:
        pass
    win, hop = int(sr * 0.04), int(sr * hop_s)
    lo, hi = int(sr / fmax), int(sr / fmin)
    out = []
    gate = max(0.01, 0.1 * float(np.sqrt(np.mean(audio ** 2) + 1e-12)))
    for i in range(0, len(audio) - win, hop):
        x = audio[i:i + win] - np.mean(audio[i:i + win])
        if np.sqrt(np.mean(x ** 2)) < gate:
            out.append(np.nan); continue
        spec = np.fft.rfft(x, 2 * win)
        ac = np.fft.irfft(spec * np.conj(spec))[:win]
        if ac[0] <= 0:
            out.append(np.nan); continue
        ac = ac / ac[0]
        seg = ac[lo:hi]
        k = int(np.argmax(seg))
        out.append(sr / (lo + k) if seg[k] > 0.45 else np.nan)
    return np.asarray(out, float)


def semitones(f0_hz: np.ndarray, ref_hz: float = 100.0) -> np.ndarray:
    v = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
    return 12.0 * np.log2(v / ref_hz)


# ───────────────────────── speaker-drift (experimental) ─────────────────────────
def _mel_envelope(audio: np.ndarray, sr: int, n_mels: int = 24) -> np.ndarray:
    from scipy.fft import dct
    win, hop = int(0.025 * sr), int(0.010 * sr)
    if len(audio) < win * 3:
        return np.zeros(n_mels)
    idx = np.arange(0, len(audio) - win, hop)
    frames = np.stack([audio[i:i + win] * np.hanning(win) for i in idx])
    keep = np.sqrt(np.mean(frames ** 2, axis=1)) > 0.5 * np.median(np.sqrt(np.mean(frames ** 2, axis=1)))
    frames = frames[keep] if keep.sum() > 3 else frames
    mag = np.abs(np.fft.rfft(frames, 512, axis=1)) ** 2
    f = np.fft.rfftfreq(512, 1.0 / sr)
    mel = lambda h: 2595 * np.log10(1 + h / 700.0)
    pts = np.linspace(mel(60), mel(min(7600, sr / 2 - 1)), n_mels + 2)
    hz = 700 * (10 ** (pts / 2595) - 1)
    fb = np.zeros((n_mels, len(f)))
    for m in range(n_mels):
        l, c, r = hz[m], hz[m + 1], hz[m + 2]
        fb[m] = np.clip(np.minimum((f - l) / (c - l + 1e-9), (r - f) / (r - c + 1e-9)), 0, None)
    logmel = np.log(mag @ fb.T + 1e-10)
    return dct(logmel, type=2, norm="ortho", axis=1)[:, 1:13].mean(axis=0)


def second_voice_drift(audio: np.ndarray, sr: int, segs: Sequence[Tuple[float, float]], chunk_s: float = 3.0) -> Optional[float]:
    """Max cosine distance between the answer's first-chunk spectral envelope and any later chunk (0 = same, 1 = unrelated).
    Returns None when the answer is too short.  EXPERIMENTAL coarse hint, not speaker verification."""
    speech = np.concatenate([audio[int(s * sr):int(t * sr)] for s, t in segs]) if segs else np.array([])
    if len(speech) < sr * chunk_s * 2:
        return None
    n = int(chunk_s * sr)
    chunks = [speech[i:i + n] for i in range(0, len(speech) - n + 1, n)]
    ref = _mel_envelope(chunks[0], sr)
    worst = 0.0
    for c in chunks[1:]:
        v = _mel_envelope(c, sr)
        cos = float(np.dot(ref, v) / (np.linalg.norm(ref) * np.linalg.norm(v) + 1e-9))
        worst = max(worst, 1.0 - cos)
    return float(worst)


# ───────────────────────── the main extractor ─────────────────────────
@dataclass
class VoiceCues:
    usable: bool = False
    quality_reasons: List[str] = field(default_factory=list)
    response_latency_s: Optional[float] = None
    barge_in: bool = False
    speaking_s: float = 0.0
    window_s: float = 0.0
    n_pauses: int = 0
    pause_rate_per_min: Optional[float] = None
    long_pause_count: Optional[int] = None
    mean_pause_s: Optional[float] = None
    articulation_rate_wps: Optional[float] = None
    wpm: Optional[float] = None
    f0_mean_st: Optional[float] = None
    f0_sd_st: Optional[float] = None
    voiced_ratio: Optional[float] = None
    loudness_sd_db: Optional[float] = None
    snr_db: Optional[float] = None
    second_voice_drift: Optional[float] = None       # integrity, experimental

    def fusion_cues(self) -> Dict[str, Optional[float]]:
        """Only the cues the fusion model understands (names from behavior_config_v59.CUES). None = unreliable."""
        ok = self.usable
        return {
            "response_latency_s": None if (self.barge_in or self.response_latency_s is None) else self.response_latency_s,
            "pause_rate_per_min": self.pause_rate_per_min if ok else None,
            "long_pause_count": float(self.long_pause_count) if (ok and self.long_pause_count is not None) else None,
            "articulation_rate_wps": self.articulation_rate_wps if ok else None,
            "f0_mean_st": self.f0_mean_st if ok else None,
            "f0_sd_st": self.f0_sd_st if ok else None,
        }

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def voice_cues(audio: np.ndarray, sr: int, capture_t0: float, tts_end: Optional[float], n_words: int = 0,
               segments: Optional[Sequence[Tuple[float, float]]] = None, pause_min_s: float = 0.25, long_pause_s: float = 1.0,
               min_speech_s: float = 3.0, min_snr_db: float = 10.0) -> VoiceCues:
    """audio: float32 mono; `capture_t0` and `tts_end` must come from the SAME monotonic clock (time.monotonic());
    `segments` = speech segments in seconds relative to audio start (Silero or energy_vad)."""
    v = VoiceCues()
    audio = np.asarray(audio, np.float32)
    segs = list(segments) if segments is not None else energy_vad(audio, sr)
    if not segs:
        v.quality_reasons.append("no speech detected")
        return v
    first, last = segs[0][0], segs[-1][1]
    v.window_s = float(last - first)
    v.speaking_s = float(sum(t - s for s, t in segs))
    # latency from the interviewer clock
    if tts_end is not None:
        lat = (capture_t0 + first) - tts_end
        if lat < -0.2:
            v.barge_in = True
            v.response_latency_s = 0.0
        else:
            v.response_latency_s = float(max(0.0, lat))
    # pauses inside the answer (leading/trailing silence excluded)
    gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
    pauses = [g for g in gaps if g >= pause_min_s]
    v.n_pauses = len(pauses)
    v.mean_pause_s = float(np.mean(pauses)) if pauses else 0.0
    # SNR proxy
    sp = np.concatenate([audio[int(s * sr):int(t * sr)] for s, t in segs])
    mask = np.ones(len(audio), bool)
    for s, t in segs:
        mask[int(s * sr):int(t * sr)] = False
    noise = audio[mask]
    rms_s = float(np.sqrt(np.mean(sp ** 2) + 1e-12))
    rms_n = float(np.sqrt(np.mean(noise ** 2) + 1e-12)) if len(noise) > sr * 0.2 else None
    v.snr_db = float(20 * math.log10(rms_s / rms_n)) if rms_n else None
    if v.speaking_s < min_speech_s:
        v.quality_reasons.append(f"only {v.speaking_s:.1f}s of speech")
    if v.snr_db is not None and v.snr_db < min_snr_db:
        v.quality_reasons.append(f"low SNR ({v.snr_db:.0f} dB)")
    v.usable = not v.quality_reasons
    mins = max(v.window_s / 60.0, 1e-6)
    v.pause_rate_per_min = float(len(pauses) / mins)
    v.long_pause_count = int(sum(1 for g in pauses if g >= long_pause_s))
    if n_words > 0 and v.speaking_s > 0:
        v.articulation_rate_wps = float(n_words / v.speaking_s)
        v.wpm = float(n_words / mins)
    # pitch + loudness
    f0 = f0_track(sp, sr)
    st = semitones(f0)
    v.voiced_ratio = float(np.mean(np.isfinite(f0))) if len(f0) else None
    if v.voiced_ratio is not None and v.voiced_ratio < 0.15:               # noise / non-speech sound that fooled the energy VAD
        v.quality_reasons.append(f"little voiced speech ({v.voiced_ratio:.0%} of frames)")
        v.usable = False
    if len(st) >= 20:
        v.f0_mean_st, v.f0_sd_st = float(np.mean(st)), float(np.std(st))
    hop = int(0.03 * sr)
    if len(sp) > hop * 5:
        e = np.sqrt(np.mean(sp[: (len(sp) // hop) * hop].reshape(-1, hop) ** 2, axis=1) + 1e-12)
        db = 20 * np.log10(e[e > 0.3 * np.median(e)] + 1e-9)
        v.loudness_sd_db = float(np.std(db)) if len(db) > 5 else None
    v.second_voice_drift = second_voice_drift(audio, sr, segs)
    return v


# ───────────────────────── adapters for the existing Voice module ─────────────────────────
def ser_to_cues(arena_results: Dict[str, dict]) -> Dict[str, Optional[float]]:
    """Map Voice/models_arena.py output (model1_dimensional / model2_categorical / model3_hubert) to fusion cue names."""
    out: Dict[str, Optional[float]] = {"ser_arousal": None, "ser_negative_mass": None}
    m1 = arena_results.get("model1_dimensional")
    if m1 and "arousal" in m1:
        out["ser_arousal"] = float(m1["arousal"])
    neg = []
    if arena_results.get("model2_categorical") and "fear_cluster_score" in arena_results["model2_categorical"]:
        neg.append(float(arena_results["model2_categorical"]["fear_cluster_score"]))
    if arena_results.get("model3_hubert") and "stress_score" in arena_results["model3_hubert"]:
        neg.append(float(arena_results["model3_hubert"]["stress_score"]))
    if neg:
        out["ser_negative_mass"] = float(np.mean(neg))
    return out
