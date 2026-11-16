"""
deception_engine.py — smartHIRE Ai
========================================
Turns raw acoustic + model-arena outputs into three interpretable
deception/stress indicators (0-100%) plus a weighted composite score:

  Type A — Cognitive Load & Hesitation   (Silero VAD pauses/latency)
  Type B — Vocal Tension & Micro-Tremors (Parselmouth jitter/shimmer/HNR/F0)
  Type C — Emotional Nervousness/Arousal (deep learning audio models)

Type A and Type B are normalized against the CANDIDATE'S OWN calibrated
baseline (the first ~8s of their speech), not an absolute population
threshold — voice-stress indicators vary a lot between individuals, so an
anomaly-from-self-baseline approach is far more defensible than a fixed
cutoff. This is documented as a design decision, not a hidden detail.
"""

from __future__ import annotations

import numpy as np

import voice_config as config


def zscore(value: float, mean: float, std: float) -> float:
    return (value - mean) / std if std > 0 else 0.0


# Baseline ke std ka minimum (physical scale ke hisaab se). Ye na ho to baseline mein
# agar sab values barabar hon (e.g. pauses hamesha 0) to std ~0 ho jata hai aur ek
# choti si tabdeeli bhi z-score ko behad bada kar ke score ko foran 100% kar deti hai.
# Ye floors mere andaze hain - apne data ke hisaab se tune karo.
MIN_STD = {
    "num_pauses": 1.0,
    "response_latency_sec": 1.0,
    "speech_ratio": 0.10,
    "jitter_local_pct": 0.20,
    "shimmer_local_pct": 1.0,
    "hnr_db": 2.0,
    "pitch_std_hz": 5.0,
}


def sigmoid_100(x: float) -> float:
    """Squashes a z-score-like value into a 0-100 scale (z ko [-4, 4] tak clip karke)."""
    x = float(np.clip(x, -4.0, 4.0))
    return float(100.0 / (1.0 + np.exp(-x)))


class Baseline:
    """
    Collects the candidate's first BASELINE_SECONDS of speech to establish
    their personal 'calm' reference for pause frequency, pitch variability,
    jitter, shimmer, and HNR.
    """

    def __init__(self):
        self.ready = False
        self.seconds_collected = 0.0
        self._raw: dict[str, list[float]] = {
            "num_pauses": [], "response_latency_sec": [], "speech_ratio": [],
            "jitter_local_pct": [], "shimmer_local_pct": [],
            "hnr_db": [], "pitch_std_hz": [],
        }
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def add(self, acoustic: dict, temporal: dict, window_seconds: float):
        self.seconds_collected += window_seconds
        self._raw["num_pauses"].append(temporal["num_pauses"])
        self._raw["response_latency_sec"].append(temporal["response_latency_sec"])
        self._raw["speech_ratio"].append(temporal["speech_ratio"])
        self._raw["jitter_local_pct"].append(acoustic["jitter_local_pct"])
        self._raw["shimmer_local_pct"].append(acoustic["shimmer_local_pct"])
        self._raw["hnr_db"].append(acoustic["hnr_db"])
        self._raw["pitch_std_hz"].append(acoustic["pitch_std_hz"])

        if self.seconds_collected >= config.BASELINE_SECONDS and not self.ready:
            self._finalize()

    def _finalize(self):
        for name, vals in self._raw.items():
            arr = np.array(vals) if vals else np.array([0.0])
            self.mean[name] = float(arr.mean())
            self.std[name] = max(float(arr.std()), MIN_STD.get(name, 1e-3))
        self.ready = True


# --------------------------------------------------------------------------
# Type A — Cognitive Load & Hesitation
# --------------------------------------------------------------------------

def compute_type_a_cognitive_load(temporal: dict, baseline: Baseline) -> dict:
    if not baseline.ready:
        return {"score": 0.0, "ready": False}

    pause_z = zscore(temporal["num_pauses"], baseline.mean["num_pauses"], baseline.std["num_pauses"])
    latency_z = zscore(
        temporal["response_latency_sec"],
        baseline.mean["response_latency_sec"],
        baseline.std["response_latency_sec"],
    )
    # speech-to-pause ratio: baseline ke muqable mein jitni kam speech, utna zyada cognitive load
    ratio_z = -zscore(temporal["speech_ratio"], baseline.mean["speech_ratio"], baseline.std["speech_ratio"])
    combined_z = (pause_z + latency_z + ratio_z) / 3.0
    return {
        "score": round(sigmoid_100(combined_z), 1),
        "ready": True,
        "z_scores": {"pause_z": pause_z, "latency_z": latency_z, "ratio_z": ratio_z},
    }


# --------------------------------------------------------------------------
# Type B — Vocal Tension & Micro-Tremors
# --------------------------------------------------------------------------

def compute_type_b_vocal_tension(acoustic: dict, baseline: Baseline) -> dict:
    if not baseline.ready:
        return {"score": 0.0, "ready": False, "z_score_composite": 0.0}

    jitter_z = zscore(acoustic["jitter_local_pct"], baseline.mean["jitter_local_pct"], baseline.std["jitter_local_pct"])
    shimmer_z = zscore(acoustic["shimmer_local_pct"], baseline.mean["shimmer_local_pct"], baseline.std["shimmer_local_pct"])
    pitch_var_z = zscore(acoustic["pitch_std_hz"], baseline.mean["pitch_std_hz"], baseline.std["pitch_std_hz"])
    hnr_drop_z = -zscore(acoustic["hnr_db"], baseline.mean["hnr_db"], baseline.std["hnr_db"])  # lower HNR = more strain

    composite_z = (jitter_z + shimmer_z + pitch_var_z + hnr_drop_z) / 4.0
    return {
        "score": round(sigmoid_100(composite_z), 1),
        "ready": True,
        "z_score_composite": composite_z,
        "z_scores": {
            "jitter_z": jitter_z, "shimmer_z": shimmer_z,
            "pitch_var_z": pitch_var_z, "hnr_drop_z": hnr_drop_z,
        },
    }


# --------------------------------------------------------------------------
# Type C — Emotional Nervousness / Arousal (deep learning models)
# --------------------------------------------------------------------------

def compute_type_c_emotional_stress(arena_results: dict) -> dict:
    """
    Averages the stress-relevant score from every active deep-learning
    model this cycle (model1 dimensional stress, model2 fear-cluster mass,
    model3 hubert stress mass). If model4_hybrid ran, it's reported
    separately since it already blends in acoustic signal (Type B territory)
    and would double-count if folded into Type C.
    """
    scores = []
    if "model1_dimensional" in arena_results:
        scores.append(arena_results["model1_dimensional"]["stress_score"])
    if "model2_categorical" in arena_results:
        scores.append(arena_results["model2_categorical"]["fear_cluster_score"])
    if "model3_hubert" in arena_results:
        scores.append(arena_results["model3_hubert"]["stress_score"])

    if not scores:
        return {"score": 0.0, "ready": False}

    mean_score = float(np.mean(scores))  # already 0-1
    return {"score": round(mean_score * 100, 1), "ready": True, "contributing_models": len(scores)}


# --------------------------------------------------------------------------
# Final Composite Deception Risk Score
# --------------------------------------------------------------------------

def compute_composite(type_a: dict, type_b: dict, type_c: dict) -> dict:
    w = config.TYPE_WEIGHTS
    if not (type_a["ready"] and type_b["ready"] and type_c["ready"]):
        return {"score": None, "ready": False, "reason": "baseline still calibrating"}

    composite = (
        w["cognitive_load"] * type_a["score"]
        + w["vocal_tension"] * type_b["score"]
        + w["emotional_stress"] * type_c["score"]
    )
    return {"score": round(composite, 1), "ready": True}
