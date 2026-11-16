"""models_v58.py — shared dataclasses passed between answer-understanding, evidence, planner and question engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# evidence statuses (the full lattice the spec asks for)
UNKNOWN = "UNKNOWN"
CLAIM_ONLY = "CLAIM_ONLY"
THEORETICAL = "THEORETICAL"
LEARNING_ONLY = "LEARNING_ONLY"
PRACTICAL = "PRACTICAL"
DEMONSTRATED = "DEMONSTRATED"
EXPLICIT_NO_EXPERIENCE = "EXPLICIT_NO_EXPERIENCE"
CONTRADICTED = "CONTRADICTED"
PARTIALLY_VALIDATED = "PARTIALLY_VALIDATED"

STATUS_RANK = {UNKNOWN: 0, EXPLICIT_NO_EXPERIENCE: 1, CLAIM_ONLY: 1, LEARNING_ONLY: 2, THEORETICAL: 3, PRACTICAL: 4, PARTIALLY_VALIDATED: 5, DEMONSTRATED: 6, CONTRADICTED: 0}


@dataclass
class QuestionContext:
    kind: str = "open"               # intro | project_walkthrough | project_probe | knowledge | scenario | cluster_gate | behavioural | clarify | contradiction | closing | followup
    targets: List[str] = field(default_factory=list)       # concept keys the question is meant to address
    anchor_project: str = ""
    aspect: str = ""                 # ownership | dataset | labeling | training | evaluation | deployment | challenge | result | ...
    level: str = "basic"             # difficulty ladder level
    scenario: bool = False           # hypothetical / design question (answers are THEORETICAL by construction)
    text: str = ""
    turn: int = 0
    soft: str = ""                   # behavioural competency key


@dataclass
class Claim:
    key: str
    stance: str                      # no_experience | unknown | learning | theoretical | practical | claim
    implied: bool = False
    ownership: str = "unknown"       # strong | some | none | unknown
    facets: List[str] = field(default_factory=list)
    quote: str = ""
    strength: float = 0.5            # how concrete (0..1)
    note: str = ""


@dataclass
class Interpretation:
    text: str = ""
    intent: str = "substantive"      # substantive | explicit_unknown | no_experience | skip | quit | help | repeat | clarify_request | dispute | affirmation | negation | empty
    direct_answer: str = ""          # yes | no | ""
    claims: List[Claim] = field(default_factory=list)
    projects_mentioned: List[str] = field(default_factory=list)
    specificity: str = "none"        # none | low | medium | high
    ownership: str = "unknown"
    facts: List[Dict[str, str]] = field(default_factory=list)       # [{aspect, phrase, quote}] — what the candidate actually told us
    uncertainty: bool = False
    contradictions: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    question_answered: bool = False
    quality: str = "none"            # none | vague | partial | good | strong
    words: int = 0
    confidence: float = 0.7
    notes: List[str] = field(default_factory=list)
    used_llm: bool = False
    anchor_is_library: bool = False  # candidate said the anchor is a library/tool they use, not something they authored
    soft: Dict[str, float] = field(default_factory=dict)

    def claim_for(self, key: str) -> Optional[Claim]:
        for c in self.claims:
            if c.key == key:
                return c
        return None
