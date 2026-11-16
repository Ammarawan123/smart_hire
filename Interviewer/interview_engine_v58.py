"""interview_engine_v58.py — orchestrates one interview: understand -> update evidence -> plan -> react -> speak."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import concepts_v58 as K
import v58_config as cfg
from answer_understanding_v58 import understand
from evidence_engine_v58 import EvidenceGraph
from interview_planner_v58 import Decision, Planner
from models_v58 import Interpretation, QuestionContext
from question_engine_v58 import QuestionEngine
from reaction_engine_v58 import ReactionEngine, compose
from schema_v58 import JobProfile, Resume
from textkit_v58 import clip


@dataclass
class TurnOutput:
    greeting: str = ""
    text: str = ""                 # what the interviewer says (ack + transition + question) — ONE conversational unit
    ack: str = ""
    question: str = ""
    finished: bool = False
    farewell: str = ""
    debug: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    decision: Optional[Decision] = None


@dataclass
class TranscriptEntry:
    turn: int
    interviewer: str
    question: str
    phase: str
    kind: str
    targets: List[str]
    why: Dict[str, str]
    source: str
    answer: str = ""
    interpretation: Dict = field(default_factory=dict)
    evidence_changes: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)


class InterviewEngine:
    def __init__(self, resume: Resume, job: JobProfile, llm=None, target_minutes: float = cfg.TARGET_MINUTES, max_questions: int = cfg.MAX_QUESTIONS):
        self.resume, self.job, self.llm = resume, job, llm
        self.graph = EvidenceGraph(resume, job)
        self.qe = QuestionEngine(resume, job, self.graph, llm)
        self.planner = Planner(resume, job, self.graph, self.qe, target_minutes, max_questions)
        self.react = ReactionEngine()
        self.transcript: List[TranscriptEntry] = []
        self.ctx: Optional[QuestionContext] = None
        self.llm_budget = [cfg.LLM_MAX_INTERPRET_CALLS]
        self.last_decision: Optional[Decision] = None
        self.finished = False
        self.timings: List[Dict[str, float]] = []

    def greeting(self) -> str:
        who = self.resume.first_name or "there"
        role = self.job.title or "this role"
        return (f"Hi {who}, I'm {cfg.INTERVIEWER_NAME}, and I'll be interviewing you for the {role} role"
                f"{(' at ' + self.job.company) if self.job.company else ''} today. I've read your résumé and the role description, so I'll ask about your own work "
                f"rather than generic questions. This should take around ten minutes.")

    def start(self) -> TurnOutput:
        d = self.planner.first()
        self._register(d)
        return TurnOutput(self.greeting(), d.text, "", d.text, False, "", "", {}, d)

    def _register(self, d: Decision, ack_text: str = ""):
        self.last_decision = d
        self.ctx = d.spec.ctx(d.text, self.planner.turn)
        self.transcript.append(TranscriptEntry(self.planner.turn, (ack_text + " " + d.text).strip(), d.text, d.phase, d.spec.kind, list(d.spec.targets), d.why(), d.source, rejected=list(d.rejected)))

    def step(self, answer: str) -> TurnOutput:
        t0 = time.time()
        ctx = self.ctx
        turn = self.planner.turn
        t_u = time.time()
        it = understand(answer, ctx, self.resume, self.graph, self.llm, self.llm_budget)
        if ctx.kind == "intro" and it.words >= 20:
            it.soft.setdefault("communication", 0.3)
        understand_ms = (time.time() - t_u) * 1000
        self.planner.advance_clock(ctx.text, answer)
        changes = self.graph.apply(ctx, it, turn)
        te = self.transcript[-1]
        te.answer, te.evidence_changes = answer, changes
        te.interpretation = _it_dict(it)
        t_p = time.time()
        d = self.planner.next(ctx, it)
        plan_ms = (time.time() - t_p) * 1000
        if d.close:
            farewell = f"Thanks so much for your time today{(', ' + self.resume.first_name) if self.resume.first_name else ''}. That wraps up our conversation, and we'll be in touch about next steps. Have a great day!"
            ack = self.react.acknowledge(it, ctx, "close", turn)
            self.finished = True
            out = TurnOutput("", "", ack, "", True, farewell, self._debug(it, changes, d, ctx), {"understand_ms": understand_ms, "plan_ms": plan_ms}, d)
            out.text = ack
            self.last_decision = d
            return out
        topic = K.name_of(ctx.targets[0]) if ctx.targets else ctx.anchor_project
        ack = self.react.acknowledge(it, ctx, d.spec.followup or d.spec.kind, turn, topic, will_hook=bool(d.spec.hooks), switching_topic=d.switching_topic, same_topic_next=bool(set(d.spec.targets) & {c.key for c in it.claims if c.stance in ("no_experience", "unknown") and not c.implied} or (it.intent in ("explicit_unknown", "no_experience", "negation") and set(d.spec.targets) & set(ctx.targets))) and d.spec.followup in ("scenario_simplify", "learning_to_impl", "evidence_probe", "clarification"))
        trans = ""
        if d.new_phase == "technical" and self.planner.phase_q.get("technical", 0) > 1:
            trans = "Let me go back to a couple of technical areas before we finish."
        elif d.new_phase in ("technical", "behavioural") and self.planner.phase_q.get(d.new_phase, 0) <= 1:
            trans = self.react.transition("", turn, d.new_phase)
        if d.spec.kind == "closing":
            ack = ack or ""
        text = compose(ack, trans, d.text)
        self._register(d, ack_text=(ack + " " + trans).strip())
        self.transcript[-1].interviewer = text
        out = TurnOutput("", text, ack, d.text, False, "", self._debug(it, changes, d, ctx),
                         {"understand_ms": understand_ms, "plan_ms": plan_ms, "total_ms": (time.time() - t0) * 1000}, d)
        self.timings.append(out.metrics)
        return out

    def _debug(self, it: Interpretation, changes: List[str], d: Decision, ctx: QuestionContext) -> str:
        L = ["ANSWER UNDERSTANDING",
             f"  intent={it.intent} quality={it.quality} specificity={it.specificity} ownership={it.ownership} words={it.words} answered={it.question_answered}"
             + (" [LLM-assisted]" if it.used_llm else "")]
        for c in it.claims:
            L.append(f"  · {K.name_of(c.key):28} {c.stance.upper():14} {'implied' if c.implied else 'direct '} facets={c.facets} {('· ' + c.note) if c.note else ''}")
        for f in it.facts:
            L.append(f"  fact[{f['aspect']}]: {f['phrase']}")
        if it.contradictions:
            L.append(f"  contradiction: {[K.name_of(k) for k in it.contradictions]}")
        for n in it.notes:
            L.append(f"  note: {n}")
        L.append("EVIDENCE CHANGES" if changes else "EVIDENCE CHANGES: none")
        L += [f"  {c}" for c in changes]
        L.append("QUESTION DECISION")
        s = d.spec
        L.append(f"  phase={d.phase} | kind={s.kind} | followup={s.followup or '-'} | level={s.level} | aspect={s.aspect or '-'} | targets={[K.name_of(k) for k in s.targets]} | project={s.project or '-'} | source={d.source}")
        L.append(f"  why_this_question: {s.reason}")
        L.append(f"  what_new_evidence_it_seeks: {s.new_evidence}")
        L.append(f"  why_previous_answer_was_insufficient: {s.why_insufficient or '-'}")
        if d.rejected:
            L.append(f"  rejected/considered: {d.rejected[:5]}")
        L.append(f"  clock={self.planner.clock / 60:.1f} min (virtual) | coverage {self.graph.coverage()['weighted_addressed_pct']}% addressed / {self.graph.coverage()['weighted_validated_pct']}% validated")
        return "\n".join(L)


def _it_dict(it: Interpretation) -> Dict:
    return {"intent": it.intent, "quality": it.quality, "specificity": it.specificity, "ownership": it.ownership, "question_answered": it.question_answered,
            "claims": [{"concept": c.key, "stance": c.stance, "implied": c.implied, "facets": c.facets, "quote": c.quote} for c in it.claims],
            "facts": it.facts, "contradictions": it.contradictions, "missing_information": it.missing_information, "uncertainty": it.uncertainty,
            "soft": it.soft, "notes": it.notes, "used_llm": it.used_llm}
