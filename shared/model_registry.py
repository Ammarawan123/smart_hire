"""Single registry of every model/asset SmartHire can use, who owns it, where it lives, and how to get it.

Design rules (see MODEL_DOWNLOAD_GUIDE.md):
  * one canonical location per asset, no duplicate weights (SBERT/Whisper live once in NLP/nlp_models);
  * nothing is downloaded at import or application start; downloads are explicit (scripts/download_models.py);
  * `verified` means "official source page opened and checked by a person/tool in THIS repo's records";
    it is False until verification is recorded in research/MODEL_INVENTORY.csv. Sizes/licences below were
    carried over from the V59 table or the model cards named there; treat them as unverified until then.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import config as C

# Backwards-compatible name used by older scripts. "Models dir" is now per-module; this is only the repo root.
MODELS_DIR = C.P.ROOT


@dataclass
class Asset:
    key: str
    owner: str            # Interviewer | Voice | NLP | vision
    group: str            # core | nli | ser | vad | tts-piper | tts-kokoro | llm | nuextract | vision
    kind: str             # hf | hf_file | url | faster_whisper | pip | manual
    source: str
    location: Path        # canonical on-disk location (file or folder)
    size: str = ""
    licence: str = ""
    purpose: str = ""
    required: bool = False        # needed for the core interview path
    files: Optional[List[str]] = None
    note: str = ""
    verified: bool = False        # see module docstring
    fmt: str = ""


def _hf_dir(cache: Path, repo: str) -> Path:
    return cache / "hub" / ("models--" + repo.replace("/", "--"))


ASSETS: List[Asset] = [
    # ── Interviewer ────────────────────────────────────────────────────────────────────────
    Asset("qwen2.5-1.5b-q4km", "Interviewer", "llm", "hf_file", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", C.QWEN_GGUF,
          "~1 GB", "Apache-2.0", "primary local LLM (interpretation + question wording)", True,
          files=["qwen2.5-1.5b-instruct-q4_k_m.gguf"], fmt="GGUF Q4_K_M"),
    Asset("llama-server", "Interviewer", "llm", "manual", "https://github.com/ggml-org/llama.cpp/releases", C.LLAMA_SERVER,
          "", "MIT", "local runtime for the GGUF", True, note="Windows CPU build; non-AVX2 build if the CPU lacks AVX2", fmt="exe"),
    Asset("piper-exe", "Interviewer", "tts-piper", "manual", "https://github.com/rhasspy/piper/releases", C.PIPER_EXE,
          "", "MIT", "Piper executable (fallback when the piper python package is absent)", False, fmt="exe"),
    Asset("piper-voice-onnx", "Interviewer", "tts-piper", "url",
          "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
          C.PIPER_VOICE, "~60 MB", "see voice card", "spoken questions (Piper TTS)", False, fmt="ONNX"),
    Asset("piper-voice-json", "Interviewer", "tts-piper", "url",
          "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
          Path(str(C.PIPER_VOICE) + ".json"), "5 KB", "", "Piper voice config", False, fmt="JSON"),
    Asset("nuextract-tiny", "Interviewer", "nuextract", "manual", "numind NuExtract-tiny-v1.5 GGUF (Q4_K_M)", C.NUEXTRACT_GGUF,
          "", "see model card", "optional resume claim extraction (needs its own llama-server; see RUN notes)", False, fmt="GGUF"),
    Asset("interviewer-sbert", "Interviewer", "optional", "manual", "sentence-transformers/all-MiniLM-L6-v2 (second copy)", C.INTERVIEWER_SBERT,
          "~90 MB", "Apache-2.0 (check card)", "present on disk; no Interviewer code loads it in this repo", False, fmt="folder",
          note="duplicate of NLP copy; delete or point SMARTHIRE_INTERVIEWER_SBERT to the NLP folder to save disk"),
    # ── NLP ────────────────────────────────────────────────────────────────────────────────
    Asset("sbert", "NLP", "core", "hf_dir", "sentence-transformers/all-MiniLM-L6-v2", C.SBERT_DIR,
          "~90 MB", "Apache-2.0 (check card)", "answer relevance / semantic matching", False, fmt="folder"),
    Asset("whisper-base.en", "NLP", "core", "hf_dir", "Systran/faster-whisper-base.en", C.WHISPER_BASE_DIR,
          "~145 MB", "MIT", "local speech-to-text (CPU)", False, fmt="CTranslate2 folder"),
    Asset("whisper-distil-large-v3", "NLP", "whisper-big", "hf_dir", "Systran/faster-distil-whisper-large-v3", C.WHISPER_DISTIL_DIR,
          "~1.5 GB", "MIT", "higher-accuracy STT (heavy for 4 GB; do not co-load with Qwen)", False, fmt="CTranslate2 folder"),
    Asset("grammar-t5", "NLP", "grammar", "hf_dir", "vennify/t5-base-grammar-correction", C.GRAMMAR_T5_DIR,
          "~900 MB", "see card", "post-session transcript grammar pass (NLP/stt_realtime_v5.py)", False, fmt="folder"),
    Asset("nli-deberta-xsmall", "NLP", "nli", "hf_dir", "cross-encoder/nli-deberta-v3-xsmall", C.NLI_DIR,
          "~90 MB", "Apache-2.0 (check card)", "optional: contradiction check between earlier and current statements (main.py interview --nli)", False, fmt="folder"),
    # ── Voice ──────────────────────────────────────────────────────────────────────────────
    Asset("silero-vad", "Voice", "vad", "pip", "pip install silero-vad (or torch.hub cache in torch_home)", C.TORCH_HOME, "~2 MB", "MIT", "voice-activity detection", False,
          note="Voice/acoustic_extractor.py uses torch.hub with TORCH_HOME=torch_home; it needs the silero-vad repo cached there, else network once", fmt="cache"),
    Asset("ser-audeering", "Voice", "ser", "hf", "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", _hf_dir(C.SER_CACHE, "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"),
          "~1.2 GB", "CC BY-NC-SA 4.0 (non-commercial)", "optional dimensional speech emotion", False, fmt="folder", note="large; non-commercial licence"),
    Asset("ser-ehcalabres", "Voice", "ser", "hf", "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", _hf_dir(C.SER_CACHE, "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"),
          "~1.2 GB", "Apache-2.0", "optional categorical speech emotion", False, fmt="folder"),
    Asset("ser-hubert", "Voice", "ser", "hf", "superb/hubert-large-superb-er", _hf_dir(C.SER_CACHE, "superb/hubert-large-superb-er"),
          "~1.2 GB", "see card", "optional 4-class speech emotion", False, fmt="folder"),
    # ── Vision ─────────────────────────────────────────────────────────────────────────────
    Asset("face-landmarker", "vision", "vision", "url", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", C.FACE_LANDMARKER,
          "~4 MB", "Apache-2.0 (MediaPipe)", "face landmarks (only used with SMARTHIRE_FACE_BACKEND=tasks)", False, fmt="task"),
    Asset("gaze-onnx", "vision", "vision", "url", "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0_gaze.onnx", C.GAZE_ONNX,
          "~5 MB", "see repo", "EXPERIMENTAL learned gaze (only with SMARTHIRE_USE_GAZE=1)", False, fmt="ONNX"),
    Asset("resemotenet", "vision", "vision", "manual", "team model (hsemotion-derived, renamed)", C.EMOTION_ONNX,
          "~16 MB", "check hsemotion licence", "main facial-expression model (loaded by VisionPipeline; required for camera mode)", False, fmt="ONNX"),
    Asset("yolov8n", "vision", "vision", "manual", "yolov8n.onnx", C.PHONE_ONNX,
          "~12 MB", "AGPL-3.0 (Ultralytics)", "optional phone detector", False, fmt="ONNX"),
]


def get(key: str) -> Asset:
    for a in ASSETS:
        if a.key == key:
            return a
    raise KeyError(key)


def by_group(groups: List[str]) -> List[Asset]:
    return [a for a in ASSETS if a.group in groups]


def by_owner(owner: str) -> List[Asset]:
    return [a for a in ASSETS if a.owner == owner]


def path_of(asset: Asset, models_dir=None) -> Path:   # models_dir kept only for old call signatures
    return Path(asset.location)


def is_present(a: Asset) -> bool:
    """True only if the asset looks complete (files exist and are non-empty; model folders contain config.json)."""
    p = Path(a.location)
    if a.kind == "pip":
        mod = {"silero-vad": "silero_vad"}.get(a.key, a.key)
        try:
            import importlib.util
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False
    if a.kind in ("hf", "hf_dir"):
        return p.is_dir() and any(p.rglob("config.json"))
    if p.is_dir():
        return any(p.iterdir())
    return p.is_file() and p.stat().st_size > 0


def verify_all() -> List[Dict]:
    """Presence check only (no loading, no network). Returns one dict per asset."""
    rows = []
    for a in ASSETS:
        d = asdict(a)
        d["location"] = str(a.location)
        d["present"] = is_present(a)
        rows.append(d)
    return rows


def apply_env(models_dir=None, offline: bool = True) -> Path:
    C.apply_runtime_env(offline=offline)
    return MODELS_DIR
