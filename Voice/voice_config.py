"""
config.py — smartHIRE Ai Vocal Stress & Deception Module
=============================================================
Single place to toggle which pretrained models run, and to tune the
weights used when fusing signals into Type A/B/C and the final
composite deception score. No training happens anywhere in this
module — every model below is a pretrained checkpoint pulled from
HuggingFace on first run and cached locally.
"""

# ---------------------------------------------------------------------
# Model selection — toggle for benchmarking
# ---------------------------------------------------------------------
# True  -> every cycle runs ALL models side-by-side (for comparison/benchmarking)
# False -> only the models listed in ACTIVE_MODELS run (for production/low-latency)

from pathlib import Path

_BASE = Path(__file__).resolve().parent / "voice_models"

MODEL_IDS = {
    "model1_dimensional": str(_BASE / "audeering-dim"),
    "model2_categorical": str(_BASE / "ehcalabres-xlsr"),
    "model3_hubert":      str(_BASE / "hubert-superb-er"),
    # "model4_hybrid" HuggingFace checkpoint nahi hai, models_arena.py mein
    # model1 + model2 + acoustic ka weighted fusion hai.
}

TORCH_HOME = str(_BASE / "torch_home")   # Silero VAD cache yahan se uthayega

# True  -> har cycle mein ALL models side-by-side chalte hain (benchmarking)
# False -> sirf ACTIVE_MODELS wale chalte hain (production / low latency)
RUN_ALL_FOR_COMPARISON = False

ACTIVE_MODELS = ["model1_dimensional", "model3_hubert", "model4_hybrid"]

# Bade models har cycle chalane ki bajaye har N-wein cycle par chalao; beech mein
# pichla result reuse hota hai (dashboard par latency "(cache)" dikhti hai).
# 1 = har cycle. Model 3 ko 2 par rakhne se average cycle time kaafi kam hota hai.
MODEL_RUN_EVERY = {
    "model1_dimensional": 1,
    "model2_categorical": 2,
    "model3_hubert": 2,
}

ALL_MODEL_KEYS = ["model1_dimensional", "model2_categorical", "model3_hubert", "model4_hybrid"]

# Which of ehcalabres' 8 categorical labels count as "nervous/fear cluster"
# (used by Type C and by the Model 4 hybrid combo)
FEAR_CLUSTER_LABELS = {"fearful", "angry", "sad"}

# ---------------------------------------------------------------------
# Audio capture / windowing
# ---------------------------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
WINDOW_SECONDS = 3.0     # analysis window fed to models each cycle
HOP_SECONDS = 0.0        # 0 = cycle khatam hote hi agla shuru (CPU par sabse zyada coverage)
BASELINE_SECONDS = 8.0   # seconds of speech used to calibrate personal baseline
VAD_THRESHOLD = 0.5
MIN_SPEECH_SECONDS = 1.0  # window mein itni speech na ho to cycle skip (noise par measure na ho)
MIN_PAUSE_SEC = 0.15     # ignore sub-150ms gaps as not perceptible hesitation
LONG_SILENCE_RESET_SEC = 20.0  # after this much silence, reset "response latency" clock

# ---------------------------------------------------------------------
# Model 1 (dimensional) -> stress mapping
# High arousal + low dominance correlates with vocal stress (audeering docs)
# ---------------------------------------------------------------------
AROUSAL_STRESS_WEIGHT = 0.6
LOW_DOMINANCE_STRESS_WEIGHT = 0.4

# ---------------------------------------------------------------------
# Model 4 hybrid combo internal weights
# ---------------------------------------------------------------------
HYBRID_WEIGHTS = {
    "dimensional_arousal": 0.40,   # Model 1
    "categorical_fear": 0.35,      # Model 2 fear-cluster mass
    "acoustic": 0.25,              # jitter/shimmer/HNR/pitch z-score composite
}

# ---------------------------------------------------------------------
# Deception Type A/B/C -> Composite fusion weights
# ---------------------------------------------------------------------
TYPE_WEIGHTS = {
    "cognitive_load": 0.30,    # Type A — Silero VAD pauses / hesitation / latency
    "vocal_tension": 0.35,     # Type B — Parselmouth jitter/shimmer/HNR/F0
    "emotional_stress": 0.35,  # Type C — deep learning audio models
}

# ---------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------
REFRESH_EVERY_CYCLE = True   # clear terminal each cycle for a live dashboard feel
