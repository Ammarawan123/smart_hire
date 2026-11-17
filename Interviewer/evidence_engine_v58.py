"""
evidence_engine_v58.py — the evidence graph.

Per competency we keep résumé evidence and interview evidence *separately*; the blueprint is never evidence.
  status      UNKNOWN | CLAIM_ONLY | THEORETICAL | LEARNING_ONLY | PRACTICAL | DEMONSTRATED | EXPLICIT_NO_EXPERIENCE | CONTRADICTED | PARTIALLY_VALIDATED
  addressed   a candidate-facing question was actually asked about it
  resolved    the interview produced a determinate outcome (anything except UNKNOWN / unresolved CLAIM_ONLY)
  validated   DEMONSTRATED, PARTIALLY_VALIDATED, or PRACTICAL with confidence >= cfg.VALIDATED_CONFIDENCE — from *direct* interview evidence only
Updates are scoped: an answer only changes the competencies it actually speaks about (Interpretation.claims). Implied evidence
(YOLO => Computer Vision) is stored with an `implied` flag, capped at PRACTICAL, never counts as validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import concepts_v58 as K
import v58_config as cfg
from models_v58 import *            # noqa: F401,F403
from schema_v58 import JobProfile, Resume


@dataclass
class EvItem:
    turn: int
    stance: str
    implied: bool
    facets: List[str]
    quote: str
    weight: float
    ownership: str = "unknown"


@dataclass
class Competency:
    key: str
    name: str
    category: str
    importance: int = 2
    required: bool = True
    expected_depth: str = "implementation"
    kind: str = "technology"
    # résumé side
    resume_level: str = "none"                    # none | listed | used_in_project | strong
    resume_sources: List[str] = field(default_factory=list)
    # interview side
    interview: List[EvItem] = field(default_factory=list)
    candidate_claim: bool = False                 # candidate affirmed it verbally
    candidate_demonstration: bool = False
    questions_asked: List[int] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    answer_quality: List[str] = field(default_factory=list)
    status: str = UNKNOWN
    confidence: float = 0.0
    followup_count: int = 0
    depth: int = 0                                # ladder index reached (0 basic .. 4 debug)
    facets_covered: Set[str] = field(default_factory=set)
    scenario_offered: bool = False
    clarified: bool = False
    contradicts_resume: bool = False
    notes: List[str] = field(default_factory=list)
    declined: bool = False                        # candidate said skip / pass
    said_unknown: bool = False                    # candidate said they don't know / no idea (a determinate negative answer)

    @property
    def addressed(self) -> bool:
        return bool(self.questions_asked)

    def direct_items(self) -> List[EvItem]:
        return [i for i in self.interview if not i.implied]


@dataclass
class ProjectState:
    name: str
    aspects: Dict[str, str] = field(default_factory=dict)    # aspect -> short fact the candidate gave
    asked: int = 0
    ownership: str = "unknown"
    discussed: bool = False


@dataclass
class SoftState:
    key: str
    name: str
    strength: float = 0.0
    facets: Set[str] = field(default_factory=set)             # situation | action | result
    quotes: List[str] = field(default_factory=list)
    asked: int = 0
    importance: int = 2


class EvidenceGraph:
    def __init__(self, resume: Resume, job: JobProfile):
        self.resume, self.job = resume, job
        self.comp: Dict[str, Competency] = {}
        self.projects: Dict[str, ProjectState] = {p.short_name: ProjectState(p.short_name) for p in resume.projects}
        self.soft: Dict[str, SoftState] = {}
        self.log: List[str] = []
        for r in job.requirements.values():
            if r.domain_context_only:
                continue
            self._ensure(r.key, r.name, r.category, r.importance, r.required, r.expected_depth, r.kind)
        for k, r in job.behavioural.items():
            self.soft[k] = SoftState(k, r.name, importance=r.importance)
        self._seed_resume()

    # ── setup ──────────────────────────────────────────────────────────────────────────────────
    def _ensure(self, key, name=None, cat="other", imp=1, required=False, depth="implementation", kind="technology") -> Competency:
        if key not in self.comp:
            c = K.get(key)
            self.comp[key] = Competency(key, name or (c.name if c else key), cat or (c.category if c else "other"), imp, required, depth, kind)
        return self.comp[key]

    def _seed_resume(self):
        """Résumé evidence per competency: listed < used in a project < strong (project + method/ownership)."""
        for sk in self.resume.all_skills():
            if sk.key.startswith("custom:"):
                continue
            self._resume_mark(sk.key, "used_in_project" if sk.in_projects else "listed", ("project: " + ", ".join(sk.in_projects)) if sk.in_projects else "skills list")
        for p in self.resume.projects:
            for t in p.technologies:
                if not t.key.startswith("custom:"):
                    self._resume_mark(t.key, "used_in_project", "project: " + p.short_name)
            for k, w in p.implied.items():
                self._resume_mark(k, "used_in_project" if w >= 0.6 else "listed", f"project (implied): {p.short_name}", only_if_tracked=True)
            for m in p.methods:
                for k, *_ in K.find_mentions(m):
                    self._resume_mark(k, "used_in_project", "project method: " + p.short_name, only_if_tracked=True)
        for c in self.comp.values():
            n_proj = len([s for s in c.resume_sources if s.startswith("project")])
            if c.resume_level == "used_in_project" and n_proj >= 2:
                c.resume_level = "strong"
            if c.resume_level != "none":
                c.status = CLAIM_ONLY if c.status == UNKNOWN and False else c.status     # résumé alone never changes the interview status

    def _resume_mark(self, key: str, level: str, src: str, only_if_tracked: bool = False):
        if only_if_tracked and key not in self.comp:
            return
        c = self._ensure(key, imp=1, required=False)
        order = ["none", "listed", "used_in_project", "strong"]
        if order.index(level) > order.index(c.resume_level):
            c.resume_level = level
        if src not in c.resume_sources:
            c.resume_sources.append(src)
        # a family JD requirement (SQL Server) is supported at lower strength by the family skill (SQL)
        fam = [k for k, parent in K.FAMILY.items() if parent == key]
        for f in fam:
            if f in self.comp and order.index(level) > order.index(self.comp[f].resume_level) and order.index(level) <= 1:
                self.comp[f].resume_level = "listed"
                self.comp[f].resume_sources.append(f"family skill: {K.name_of(key)}")

    # ── queries ────────────────────────────────────────────────────────────────────────────────
    def get(self, key: str) -> Optional[Competency]:
        return self.comp.get(key)

    def required_keys(self) -> List[str]:
        return [k for k, c in self.comp.items() if c.required and k in self.job.requirements and not self.job.requirements[k].domain_context_only]

    def resolved(self, c: Competency) -> bool:
        if (c.declined and not c.interview) or c.said_unknown:
            return True
        # implied-only evidence (e.g. RAG answer implying LLM use) does not make an un-asked requirement "resolved"
        if c.status not in (UNKNOWN, CLAIM_ONLY):
            return bool(c.addressed or c.direct_items())
        return False

    def validation_basis(self, c: Competency) -> str:
        """'hands_on' | 'conceptual' | ''.  Honest about WHAT was validated:
        hands-on  = the candidate described doing it (DEMONSTRATED, or PRACTICAL with enough confidence);
        conceptual = a design/knowledge answer that touched >=2 evidence areas (PARTIALLY_VALIDATED) — counts as validation ONLY for
                     requirements the JD words as awareness-level ('exposure to', 'familiar with'); never for implementation/depth requirements."""
        if c.status == DEMONSTRATED and c.confidence >= 0.4:
            return "hands_on"
        if c.status == PRACTICAL and c.confidence >= cfg.VALIDATED_CONFIDENCE and c.direct_items():
            return "hands_on"
        if c.status == PARTIALLY_VALIDATED and c.confidence >= 0.4 and c.expected_depth == "awareness":
            return "conceptual"
        return ""

    def validated(self, c: Competency) -> bool:
        return bool(self.validation_basis(c))

    def coverage(self) -> Dict[str, object]:
        keys = self.required_keys()
        tot = sum(self.comp[k].importance for k in keys) or 1
        w = lambda pred: round(100 * sum(self.comp[k].importance for k in keys if pred(self.comp[k])) / tot)      # noqa: E731
        return {
            "required": len(keys),
            "addressed": sum(1 for k in keys if self.comp[k].addressed),
            "resolved": sum(1 for k in keys if self.resolved(self.comp[k])),
            "validated": sum(1 for k in keys if self.validated(self.comp[k])),
            "weighted_addressed_pct": w(lambda c: c.addressed),
            "weighted_resolved_pct": w(self.resolved),
            "weighted_validated_pct": w(self.validated),
            "soft_with_evidence": sum(1 for s in self.soft.values() if s.strength >= 0.35),
            "soft_total": len(self.soft),
        }

    # ── updates ────────────────────────────────────────────────────────────────────────────────
    def note_question(self, ctx: QuestionContext, turn: int):
        for k in ctx.targets:
            c = self._ensure(k, imp=1, required=False)
            if turn not in c.questions_asked:
                c.questions_asked.append(turn)
            if ctx.scenario:
                c.scenario_offered = True
        if ctx.anchor_project and ctx.anchor_project in self.projects:
            self.projects[ctx.anchor_project].asked += 1
            self.projects[ctx.anchor_project].discussed = True
        if ctx.soft and ctx.soft in self.soft:
            self.soft[ctx.soft].asked += 1

    def apply(self, ctx: QuestionContext, it: Interpretation, turn: int) -> List[str]:
        """Fold an interpreted answer into the graph. Returns human-readable change log lines."""
        changes: List[str] = []
        if it.intent in ("quit", "help", "repeat", "clarify_request", "dispute", "empty"):
            return changes
        # Closing answers are conversational only and must not affect competency evidence.
        if ctx.kind == "closing":
            return changes
        if it.intent == "skip":
            for k in ctx.targets:
                c = self._ensure(k, imp=1, required=False)
                c.declined = True
                c.answers.append("(skipped)")
                c.answer_quality.append("none")
                changes.append(f"{c.name}: candidate skipped (no evidence)")
            return changes
        # book-keeping for the targets of the question
        for k in ctx.targets:
            c = self._ensure(k, imp=1, required=False)
            c.answers.append(it.text)
            c.answer_quality.append(it.quality)
            if it.quality in ("good", "strong", "partial") and ctx.level in K.LEVELS:
                c.depth = max(c.depth, K.LEVELS.index(ctx.level))
        # direct yes/no on the targets of the question ("Have you worked with RAG?" -> no)
        claims = list(it.claims)
        have = {c.key for c in claims}
        if it.intent in ("explicit_unknown", "no_experience", "negation", "affirmation") and ctx.kind in ("knowledge", "cluster_gate", "scenario", "project_probe", "followup") and not claims:
            stance = {"explicit_unknown": "unknown", "no_experience": "no_experience", "negation": "no_experience", "affirmation": "claim"}[it.intent]
            if ctx.kind == "scenario" and stance == "no_experience":
                stance = "unknown"
            # an intent that attaches to the question's own targets (never to anything else)
            if not (it.anchor_is_library and ctx.anchor_project):
                for k in ctx.targets:
                    if k not in have:
                        claims.append(Claim(k, stance, quote=it.text, strength=0.3))
        for cl in claims:
            ch = self._fold(cl, it, turn, ctx)
            if ch:
                changes.append(ch)
        # project aspects
        if ctx.anchor_project and ctx.anchor_project in self.projects:
            ps = self.projects[ctx.anchor_project]
            if it.ownership in ("strong", "some", "none"):
                ps.ownership = it.ownership
            for f in it.facts:
                if f.get("aspect") and f["aspect"] not in ps.aspects:
                    ps.aspects[f["aspect"]] = f.get("phrase", "")
                    changes.append(f"project {ps.name}: {f['aspect']} covered")
        for p in it.projects_mentioned:
            if p in self.projects:
                self.projects[p].discussed = True
        # behavioural
        for sk, val in it.soft.items():
            s = self.soft.get(sk)
            if s:
                s.strength = max(s.strength, min(1.0, val))
                changes.append(f"soft:{s.name} strength {s.strength:.2f}")
        return changes

    def _fold(self, cl: Claim, it: Interpretation, turn: int, ctx: QuestionContext) -> Optional[str]:
        c = self._ensure(cl.key, imp=1, required=False)
        before = c.status
        # tracked implied evidence is stored but low-weight and capped at PRACTICAL
        if cl.implied:
            wgt = 0.25 + 0.2 * cl.strength
            c.interview.append(EvItem(turn, "practical" if cl.stance in ("practical", "claim") else cl.stance, True, cl.facets, cl.quote, wgt, cl.ownership))
            self._recompute(c)
            return f"{c.name}: {before}->{c.status} (implied)" if c.status != before else None
        weights = {"no_experience": 0.8, "unknown": 0.6, "claim": 0.3, "learning": 0.55, "theoretical": 0.5, "practical": 0.55}
        wgt = weights.get(cl.stance, 0.4) + 0.3 * cl.strength
        c.interview.append(EvItem(turn, cl.stance, False, list(cl.facets), cl.quote, min(wgt, 0.95), cl.ownership))
        c.facets_covered.update(cl.facets)
        if cl.stance == "unknown":
            c.said_unknown = True
        if cl.stance in ("claim", "practical", "learning", "theoretical"):
            c.candidate_claim = True
        if cl.stance == "practical" and cl.strength >= 0.5:
            c.candidate_demonstration = True
        self._recompute(c)
        if cl.stance == "no_experience" and c.resume_level in ("used_in_project", "strong") and c.direct_items() and any(i.stance == "practical" for i in c.direct_items()):
            c.status, c.contradicts_resume = CONTRADICTED, True
        return f"{c.name}: {before}->{c.status} (conf {c.confidence:.2f}, facets {sorted(c.facets_covered)})" if (c.status != before or cl.facets) else None

    def _recompute(self, c: Competency):
        direct = c.direct_items()
        stances = [i.stance for i in direct]
        def cl(items):
            """Confidence from independent *turns*, not from the number of clauses: best single item + small gain per extra turn."""
            by_turn = {}
            for i in items:
                by_turn[i.turn] = max(by_turn.get(i.turn, 0.0), min(i.weight, 0.95))
            ws = sorted(by_turn.values(), reverse=True)
            return min(0.95, ws[0] + 0.08 * (len(ws) - 1)) if ws else 0.0
        if not direct:
            imp = [i for i in c.interview if i.implied]
            if imp:
                c.status = PRACTICAL if c.status in (UNKNOWN, CLAIM_ONLY) else c.status
                c.confidence = round(min(0.45, cl(imp)), 2)
                if c.status == PRACTICAL and c.status != UNKNOWN:
                    c.notes = [n for n in c.notes if n != "implied only"] + ["implied only"]
            return
        pos = [i for i in direct if i.stance in ("practical", "theoretical", "learning", "claim")]
        neg = [i for i in direct if i.stance in ("no_experience", "unknown")]
        last = direct[-1]
        # contradiction between the candidate's own statements
        if any(i.stance == "practical" for i in direct) and any(i.stance == "no_experience" and i.turn != _first(direct, "practical").turn for i in direct):
            c.status, c.confidence, c.contradicts_resume = CONTRADICTED, 0.5, False
            return
        if not pos:
            if any(i.stance == "no_experience" for i in neg):
                c.status = EXPLICIT_NO_EXPERIENCE
                if c.resume_level in ("listed", "used_in_project", "strong"):
                    c.contradicts_resume = True
                    if c.resume_level != "listed":
                        c.status = CONTRADICTED
            else:
                c.status = UNKNOWN
            c.confidence = round(cl(neg), 2)
            return
        best = max(pos, key=lambda i: {"claim": 0, "learning": 1, "theoretical": 2, "practical": 3}[i.stance])
        nf = len(c.facets_covered)
        has_prac = any(i.stance == "practical" for i in pos)
        strong_own = any(i.ownership == "strong" for i in pos)
        if has_prac:
            if nf >= cfg.DEMONSTRATED_FACETS or (nf >= 1 and strong_own and len([i for i in pos if i.stance == "practical"]) >= 2):
                c.status = DEMONSTRATED
            else:
                c.status = PRACTICAL
        elif best.stance == "theoretical":
            c.status = PARTIALLY_VALIDATED if nf >= 2 else THEORETICAL
        elif best.stance == "learning":
            c.status = LEARNING_ONLY
        else:
            c.status = CLAIM_ONLY
        cap = 0.95 if c.status == DEMONSTRATED else 0.85 if c.status in (PRACTICAL, PARTIALLY_VALIDATED) else 0.7
        c.confidence = round(min(cap, cl(pos) + 0.04 * min(nf, 4)), 2)
        if neg and c.status != CONTRADICTED:
            c.confidence = round(max(0.2, c.confidence - 0.1), 2)

    # ── reporting helpers ──────────────────────────────────────────────────────────────────────
    def unresolved_required(self) -> List[Competency]:
        return [self.comp[k] for k in self.required_keys() if not self.resolved(self.comp[k])]


def _prod(it):
    p = 1.0
    for x in it:
        p *= x
    return p


def _first(items: List[EvItem], stance: str) -> EvItem:
    return next(i for i in items if i.stance == stance)