"""Repository paths and import bootstrap.

The migrated modules (Interviewer, Voice, NLP, vision, fusion) use flat imports such as
`import concepts_v58`. Rather than rewriting hundreds of import lines, every entry point calls
`bootstrap()` once, which puts the module folders on sys.path. This is a deliberate, documented
migration decision (see research/MIGRATION_MAP.md); a later refactor can convert to packages.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INTERVIEWER_DIR = ROOT / "Interviewer"
VOICE_DIR = ROOT / "Voice"
NLP_DIR = ROOT / "NLP"
VISION_DIR = ROOT / "vision"
FUSION_DIR = ROOT / "fusion"

MODULE_DIRS = [INTERVIEWER_DIR, FUSION_DIR, VOICE_DIR, NLP_DIR, VISION_DIR]

# Model folders (see MODEL_DOWNLOAD_GUIDE.md and shared/model_registry.py)
VISION_MODELS = VISION_DIR / "vision_models"
VOICE_MODELS = VOICE_DIR / "voice_models"
NLP_MODELS = NLP_DIR / "nlp_models"
INTERVIEWER_MODELS = INTERVIEWER_DIR / "Interviewer_Models"

OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache"
DATA_DIR = ROOT / "data"


def bootstrap() -> Path:
    """Idempotently add the repo root and module folders to sys.path. Returns ROOT."""
    for p in [ROOT, *MODULE_DIRS]:
        s = str(p)
        if s not in sys.path:
            sys.path.append(s)
    return ROOT
