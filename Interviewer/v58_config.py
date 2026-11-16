"""
v58_config.py — central tunables for SmartHire V58 (clean rebuild; imports nothing from V51).

Every number is a tunable, not a research claim. Where a value encodes a design decision the
comment says why.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = "V58"
PARSER_VERSION = "58.0"          # bump to invalidate cached résumé / JD profiles
CACHE_DIR = os.environ.get("SMARTHIRE_CACHE_V58", str(HERE.parent / "outputs" / "cache" / "interviewer"))   # private: git-ignored

# ── Interview length (virtual clock: speaking + answering time, NOT LLM latency) ─────────────
TARGET_MINUTES = 10.0            # centre of the 8–12 min target
MIN_SECONDS = 8 * 60
MAX_SECONDS = 12 * 60
MAX_QUESTIONS = 20               # hard rail (follow-ups included); the virtual clock normally ends the interview first
MIN_QUESTIONS = 7
SPEAK_WPS = 2.6                  # interviewer speech, words / second (for the virtual clock)
ANSWER_WPS = 2.0                 # candidate speech, words / second
THINK_S = 5.0                    # thinking pause per answer

# Time allocation by phase (share of the budget); the planner re-balances as it goes.
PHASE_SHARE = {"intro": 0.09, "resume": 0.27, "technical": 0.46, "behavioural": 0.14, "closing": 0.04}

# ── Conversation control ─────────────────────────────────────────────────────────────────────
MAX_FOLLOWUPS_PER_TOPIC = 2      # bounded follow-up rounds (bounded follow-up rounds with explicit exit conditions, cf. PolyInterview arXiv:2607.10310)
MAX_CLARIFICATIONS = 1
MAX_CONSECUTIVE_FOLLOWUPS = 3
DEDUP_THRESHOLD = 0.62           # token-set cosine above which two questions count as the same ask
DEDUP_WINDOW = 99                # compare against the whole interview

# ── Evidence thresholds ──────────────────────────────────────────────────────────────────────
VALIDATED_CONFIDENCE = 0.60      # a requirement is "validated" only above this AND at PRACTICAL+/DEMONSTRATED
DEMONSTRATED_FACETS = 2          # distinct evidence facets (or depth signals) needed for DEMONSTRATED

# ── Local LLM (llama-server) ─────────────────────────────────────────────────────────────────
LLAMA_HOST = os.environ.get("SMARTHIRE_LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.environ.get("SMARTHIRE_LLAMA_PORT", "8081"))
LLAMA_CTX = int(os.environ.get("SMARTHIRE_LLAMA_CTX", "2048"))
LLAMA_THREADS = int(os.environ.get("SMARTHIRE_LLAMA_THREADS", "0"))
LLAMA_STARTUP_TIMEOUT_S = 150
LLM_TEMPERATURE = 0.3
<<<<<<< HEAD
LLM_INTERPRET_MAX_TOKENS = 320
=======
LLM_INTERPRET_MAX_TOKENS = 160
>>>>>>> ca3f767841167c86d532ea2212bbb3e4357a23f9
LLM_QUESTION_MAX_TOKENS = 60
LLM_INTERPRET_TIMEOUT_S = 30
LLM_QUESTION_TIMEOUT_S = 25
LLM_MAX_INTERPRET_CALLS = 4      # hard cap per interview
LLM_MAX_QUESTION_CALLS = 5       # hard cap per interview
LLM_JSON_RETRIES = 1

# ── Voice ────────────────────────────────────────────────────────────────────────────────────
PIPER_VOICE_HINT = os.environ.get("SMARTHIRE_PIPER_MODEL", "")
STT_SILENCE_S = 1.6
STT_MAX_UTTERANCE_S = 90
STT_MIN_UTTERANCE_S = 0.6
STT_MIN_CHARS = 2

INTERVIEWER_NAME = "Smarty"
