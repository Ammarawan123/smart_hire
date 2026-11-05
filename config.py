"""Shared SmartHire configuration. Every value can be overridden with an environment variable.

Nothing here downloads anything. Runtime is offline-by-default: `apply_runtime_env()` sets the
Hugging Face offline flags so a missing model raises an error instead of silently downloading.
"""
from __future__ import annotations

import os
from pathlib import Path

from shared import paths as P


def _p(env: str, default: Path) -> Path:
    return Path(os.environ.get(env, str(default)))


# ── model locations: defaults match the author's folder layout; every one can be overridden by env var ──
IM, NM, VM, XM = P.INTERVIEWER_MODELS, P.NLP_MODELS, P.VOICE_MODELS, P.VISION_MODELS

QWEN_GGUF = _p("SMARTHIRE_LLAMA_MODEL", IM / "Qwen2.5" / "qwen2.5-1.5b-instruct-q4_k_m.gguf")
LLAMA_SERVER = _p("SMARTHIRE_LLAMA_EXE", IM / "llama-server" / "llama-server.exe")
PIPER_EXE = _p("SMARTHIRE_PIPER_EXE", IM / "Piper" / "piper" / "piper.exe")
PIPER_VOICE = _p("SMARTHIRE_PIPER_MODEL", IM / "Piper" / "en_US-lessac-medium.onnx")
NUEXTRACT_GGUF = _p("SMARTHIRE_NUEXTRACT_GGUF", IM / "NuExtract-tiny-v1.5.Q4_K_M.gguf")
INTERVIEWER_SBERT = _p("SMARTHIRE_INTERVIEWER_SBERT", IM / "all-MiniLM-L6-v2")   # present on disk; no Interviewer code loads it yet
KOKORO_DIR = _p("SMARTHIRE_KOKORO_DIR", IM / "kokoro")

SBERT_DIR = _p("SMARTHIRE_SBERT_DIR", NM / "all-MiniLM-L6-v2")
WHISPER_BASE_DIR = _p("SMARTHIRE_WHISPER_BASE_DIR", NM / "faster-whisper-base.en")
WHISPER_DISTIL_DIR = _p("SMARTHIRE_WHISPER_DISTIL_DIR", NM / "faster-distil-whisper-large-v3")
GRAMMAR_T5_DIR = _p("SMARTHIRE_GRAMMAR_DIR", NM / "t5-base-grammar-correction")
HF_CACHE = _p("SMARTHIRE_HF_CACHE", NM / "hf")      # HF_HOME only
SER_CACHE = _p("SMARTHIRE_SER_CACHE", VM)             # Hugging Face hub layout (models--org--name) for the speech-emotion models

SER_AUDEERING_DIR = _p("SMARTHIRE_SER_AUDEERING", VM / "audeering-dim")
SER_EHCALABRES_DIR = _p("SMARTHIRE_SER_EHCALABRES", VM / "ehcalabres-xlsr")
SER_HUBERT_DIR = _p("SMARTHIRE_SER_HUBERT", VM / "hubert-superb-er")
TORCH_HOME = _p("TORCH_HOME", VM / "torch_home")
SILERO_DIR = TORCH_HOME

FACE_LANDMARKER = _p("SMARTHIRE_FACE_TASK", XM / "face_landmarker.task")
GAZE_ONNX = _p("SMARTHIRE_GAZE_MODEL", XM / "mobileone_s0_gaze.onnx")
EMOTION_ONNX = _p("SMARTHIRE_EMOTION_MODEL", XM / "resemotenet.onnx")
PHONE_ONNX = _p("SMARTHIRE_PHONE_MODEL", XM / "yolov8n.onnx")
NLI_DIR = _p("SMARTHIRE_NLI_DIR", NM / "nli-deberta-v3-xsmall")          # optional contradiction checker (cross-encoder/nli-deberta-v3-xsmall)
SKILL_BANK = _p("SMARTHIRE_SKILL_BANK", P.DATA_DIR / "skill_bank.json")   # learned skills (Interviewer/skill_bank_v60.py)

# local-folder lookup used by the model loaders (SER keys follow Voice/voice_config.MODEL_IDS)
SER_LOCAL = {"model1_dimensional": SER_AUDEERING_DIR, "model2_categorical": SER_EHCALABRES_DIR, "model3_hubert": SER_HUBERT_DIR}


def local_or(local: Path, fallback: str) -> str:
    """Return the local folder/file path if it exists, else the original hub id / name (which then fails offline)."""
    return str(local) if Path(local).exists() else fallback


# ── outputs ───────────────────────────────────────────────────────────────────────────────────
OUTPUTS_DIR = _p("SMARTHIRE_OUTPUTS", P.OUTPUTS_DIR)
CACHE_DIR = _p("SMARTHIRE_CACHE", P.CACHE_DIR)
PROFILE_JSON = _p("SMARTHIRE_PROFILE_JSON", OUTPUTS_DIR / "candidate_profile.json")

# ── hardware profile (target machine; used only for documentation and defaults) ───────────────
TARGET_HARDWARE = {"cpu": "Intel i5-3470S", "logical_cores": 4, "ram_gb": 4, "gpu": None, "avx2": "not assumed"}
LLM_THREADS = int(os.environ.get("SMARTHIRE_LLAMA_THREADS", "3"))   # leave a core for audio/vision
LLM_CTX = int(os.environ.get("SMARTHIRE_LLAMA_CTX", "2048"))


def apply_runtime_env(offline: bool = True) -> None:
    """Point every HF/torch cache at the canonical folders and (by default) forbid network use."""
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(SER_CACHE))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(HF_CACHE / "sbert"))
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    # names the migrated modules already read
    os.environ.setdefault("SMARTHIRE_LLAMA_EXE", str(LLAMA_SERVER))
    os.environ.setdefault("SMARTHIRE_LLAMA_MODEL", str(QWEN_GGUF))
    os.environ.setdefault("SMARTHIRE_PIPER_EXE", str(PIPER_EXE))
    os.environ.setdefault("SMARTHIRE_PIPER_MODEL", str(PIPER_VOICE))
    os.environ.setdefault("SMARTHIRE_FACE_TASK", str(FACE_LANDMARKER))
    os.environ.setdefault("SMARTHIRE_EMOTION_MODEL", str(EMOTION_ONNX))
    os.environ.setdefault("SMARTHIRE_PHONE_MODEL", str(PHONE_ONNX))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
