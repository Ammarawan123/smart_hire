"""
interview_planner_v58.py — evidence-driven planning, follow-up policy, time control and de-duplication.

Every turn the planner answers, in order:
  1. Did the candidate issue a command (quit / skip / repeat / help / clarify / dispute)?  -> handle it as a dialogue act
  2. Is a follow-up justified by the previous answer?   (policy below)  -> otherwise move on
  3. Which unresolved requirement is the most valuable to probe next?  (coverage matrix x importance x evidence gap)
  4. Does the realised question pass validation (is a question, grounded, relevant, not a duplicate, asks for new evidence)?

Follow-up policy (adaptive-interview literature: PolyInterview / Zara-style evidence loops — see research_v58.md):
    unknown / no experience  -> at most ONE simpler scenario variant for a required competency, then move on
    learning only            -> "how would you implement it" once
    claim without detail     -> evidence probe once
    vague                    -> clarification / ownership probe once
    partial                  -> probe the highest-value MISSING evidence facet
    good + important         -> step up the difficulty ladder (basic -> implementation -> depth -> trade-offs -> debugging)
    good / strong            -> move on (no repetition)
    real contradiction       -> one neutral clarification
Follow-ups per topic are bounded (cfg.MAX_FOLLOWUPS_PER_TOPIC).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import concepts_v58 as K
import v58_config as cfg
from evidence_engine_v58 import EvidenceGraph
from models_v58 import *            # noqa: F401,F403
from question_engine_v58 import ASPECT_CHAIN, FACET_Q, QuestionEngine, Spec, _name
from textkit_v58 import cosine_sets, content_tokens, word_count

CAPS = {"intro": 1, "resume": 4, "technical": 11, "behavioural": 3, "closing": 1}
PHASE_ORDER = ["intro", "resume", "technical", "behavioural", "closing"]
LADDER_CAP = {"awareness": 1, "implementation": 2, "depth": 3}
BEHAVIOURAL_PRIORITY = ["teamwork", "problem_solving", "learning_ability", "communication", "adaptability", "planning", "attention_to_detail", "analytical_thinking", "independence", "decision_making"]


@dataclass
class Decision:
    spec: Spec
    text: str = ""
    followup: bool = False
    phase: str = ""
    switching_topic: bool = False
    new_phase: str = ""
    rejected: List[str] = field(default_factory=list)
    source: str = "deterministic"
    close: bool = False
    label: str = ""

    def why(self) -> Dict[str, str]:
        return {"why_this_question": self.spec.reason, "what_new_evidence_it_seeks": self.spec.new_evidence, "why_previous_answer_was_insufficient": self.spec.why_insufficient}


class Planner:
    def __init__(self, resume, job, graph: EvidenceGraph, qe: QuestionEngine, target_minutes: float = cfg.TARGET_MINUTES, max_questions: int = cfg.MAX_QUESTIONS):
        self.resume, self.job, self.g, self.qe = resume, job, graph, qe
        self.target_s = target_minutes * 60
        self.max_q = max_questions
        self.turn = 0
        self.clock = 0.0
        self.phase = "intro"
        self.phase_q = {p: 0 for p in PHASE_ORDER}
        self.asked: List[str] = []
        self.asked_specs: List[Spec] = []
        self.sigs: set = set()
        self.topic_key: str = ""
        self.topic_follow = 0
        self.consec_follow = 0
        self.clarifications = 0
        self.gate_done: set = set()
        self.hooks: Dict[str, str] = {}
        self.projects_done: List[str] = []
        self.last_spec: Optional[Spec] = None
        self.rank = self._rank_projects()
        self.finished = False
        self.contradiction_asked: set = set()

    # ── time ────────────────────────────────────────────────────────────────────────────────────
    def advance_clock(self, question_text: str, answer_text: str):
        self.clock += word_count(question_text) / cfg.SPEAK_WPS + word_count(answer_text) / cfg.ANSWER_WPS + cfg.THINK_S

    def time_frac(self) -> float:
        return self.clock / self.target_s

    # ── project ranking by actual JD relevance ─────────────────────────────────────────────────────
    def _rank_projects(self) -> List[Tuple[float, str, List[str]]]:
        out = []
        for p in self.resume.projects:
            keys: Dict[str, float] = {t.key: 1.0 for t in p.technologies if not t.key.startswith("custom:")}
            for k, w in p.implied.items():
                keys[k] = max(keys.get(k, 0), w)
            for m in p.methods:
                for k, *_ in K.find_mentions(m):
                    keys[k] = max(keys.get(k, 0), 0.9)
            val, cov = 0.0, []
            for k, w in keys.items():
                r = self.job.requirements.get(k)
                if r and not r.domain_context_only and w >= 0.4:
                    val += r.importance * w * (1.0 if r.required else 0.5)
                    cov.append(r.name)
            ai_like = any(k in keys for k in ("ml", "dl", "cv", "od", "genai", "llm", "nlp"))
            if ai_like and "ai_projects" in self.job.requirements:
                val += 0.6 * self.job.requirements["ai_projects"].importance + 1.0       # the JD asks for hands-on AI project experience
                cov.append("Hands-on AI project experience")
            val += 0.6 if p.ownership == "individual" else 0.0
            val += 0.2 * min(len(p.bullets), 4)
            out.append((round(val, 2), p.short_name, cov))
        return sorted(out, reverse=True)

    # ── competency values ─────────────────────────────────────────────────────────────────────────
    def _need(self, c) -> float:
        """How much interview evidence is still missing for this competency (0 = nothing more to learn by asking)."""
        if c.said_unknown or c.declined or c.status == EXPLICIT_NO_EXPERIENCE:
            return 0.0
        direct = c.direct_items()
        if c.addressed and direct and c.status in (DEMONSTRATED, PARTIALLY_VALIDATED, PRACTICAL, THEORETICAL, LEARNING_ONLY):
            if c.status == PRACTICAL and c.confidence < cfg.VALIDATED_CONFIDENCE:
                return 0.45
            return 0.0 if c.importance < 3 else 0.35
        if c.addressed and c.status in (UNKNOWN, CLAIM_ONLY) and len(c.questions_asked) >= 2:
            return 0.0
        if c.interview and not direct:                  # implied only (e.g. 'used SQL' => SQL Server): still unverified
            return 0.8
        return {UNKNOWN: 1.0, CLAIM_ONLY: 0.75}.get(c.status, 0.5)

    def _fit(self, c) -> float:
        return {"none": 1.25, "listed": 1.0, "used_in_project": 0.85, "strong": 0.75}.get(c.resume_level, 1.0)

    def values(self) -> List[Tuple[float, str]]:
        """Value of asking about each unresolved required competency.
        value = importance x evidence-need x résumé-gap, + a bundle bonus when one question can cover several related unverified items;
        ties are broken by the order the JD lists them (earlier = more prominent)."""
        keys = self.g.required_keys()
        order = {k: i for i, k in enumerate(self.job.requirements)}
        out = []
        for k in keys:
            c = self.g.comp[k]
            v = c.importance * self._need(c) * self._fit(c)
            if v <= 0:
                continue
            cd = K.get(k)
            if cd and cd.cluster and cd.cluster not in self.gate_done:
                mem = [m for m in keys if K.get(m) and K.get(m).cluster == cd.cluster and self._need(self.g.comp[m]) > 0]
                v += 0.35 * max(0, len(mem) - 1)
            if cd and cd.bridge_q and self._has_bridge(cd):
                v += 0.6          # a question grounded in the candidate's own project gets concrete evidence in fewer turns
            v -= 0.01 * order.get(k, 0)
            out.append((round(v, 3), k))
        return sorted(out, reverse=True)

    def _has_bridge(self, cd) -> bool:
        return any(any(t.key in cd.bridge_from for t in p.technologies) or any(p.implied.get(b, 0) >= 0.6 for b in cd.bridge_from) for p in self.resume.projects)

    # ── signatures / dedup ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _sig(spec: Spec) -> tuple:
        return (spec.kind if spec.kind not in ("followup", "project_probe", "knowledge") else "q", tuple(spec.targets), spec.level, spec.aspect, spec.project, spec.followup, spec.soft)

    def _try(self, spec: Spec, answer_text: str = "", interp=None, rejected: Optional[List[str]] = None, allow_dup: bool = False) -> Optional[Decision]:
        sig = self._sig(spec)
        if not allow_dup and sig in self.sigs:
            if rejected is not None:
                rejected.append(f"{spec.followup or spec.kind}:{(spec.targets or [spec.project])[:1]}: same question signature already asked")
            return None
        text = self.qe.realize(spec)
        source = "deterministic"
        if (spec.followup in ("implementation", "why_how", "debugging", "tradeoff", "evidence_probe") and spec.hook_quote and not spec.hooks and self.qe.llm is not None):
            alt = self.qe.llm_followup(spec, interp, text)
            if alt:
                ok, why = self.qe.validate(alt, spec, self.asked, answer_text, self.asked_specs)
                if ok:
                    text, source = alt, "llm"
                else:
                    if rejected is not None:
                        rejected.append(f"llm question rejected: {why}")
        ok, why = (True, "") if allow_dup else self.qe.validate(text, spec, self.asked, answer_text, self.asked_specs)
        if not ok:
            if rejected is not None:
                rejected.append(f"{spec.followup or spec.kind}:{(spec.targets or [spec.project])[:1]}: {why}")
            return None
        return Decision(spec, text, spec.followup not in ("", "knowledge_validation"), self.phase, source=source)

    # ── first question ───────────────────────────────────────────────────────────────────────────
    def first(self) -> Decision:
        spec = Spec("intro", reason="open with the candidate's own background and motivation", new_evidence="motivation, background, communication style", soft="communication")
        d = self._try(spec, allow_dup=True)
        return self._commit(d)

    def _commit(self, d: Decision) -> Decision:
        self.turn += 1
        self.asked.append(d.text)
        self.asked_specs.append(d.spec)
        self.sigs.add(self._sig(d.spec))
        self.phase_q[self.phase] = self.phase_q.get(self.phase, 0) + 1
        self.last_spec = d.spec
        d.phase = self.phase
        if d.followup:
            self.topic_follow += 1
            self.consec_follow += 1
        else:
            self.topic_follow = 0
            self.consec_follow = 0
            self.topic_key = (d.spec.targets or [d.spec.project or d.spec.soft or d.spec.kind])[0]
        self.g.note_question(d.spec.ctx(d.text, self.turn), self.turn)
        return d

    # ── main step ────────────────────────────────────────────────────────────────────────────────
    def next(self, ctx: QuestionContext, it: Interpretation) -> Decision:
        rejected: List[str] = []
        self._collect_hooks(it, ctx)
        # 1. dialogue acts
        if it.intent == "quit":
            return self._close("candidate asked to end the interview", rejected)
        if it.intent == "help":
            return self._reask(ctx, "candidate asked for help; repeating the question", rejected, same=True)
        if it.intent == "repeat":
            return self._reask(ctx, "candidate asked for a repeat", rejected, same=True)
        if it.intent == "clarify_request" and self.clarifications < 3:
            self.clarifications += 1
            return self._reask(ctx, "candidate asked what the question meant; rephrasing once", rejected, same=False)
        if ctx.kind == "closing":
            return self._close("closing answered", rejected)
        # 2. time / budget rails
        if self.turn >= self.max_q - 1 or self.clock >= cfg.MAX_SECONDS:
            return self._close("question or time budget reached", rejected)
        # 3. follow-up
        if it.intent not in ("skip", "dispute") or (it.intent == "dispute" and False):
            fu = self._followup_specs(ctx, it)
            for spec in fu:
                if self.topic_follow >= cfg.MAX_FOLLOWUPS_PER_TOPIC and spec.followup not in ("contradiction", "scenario_simplify", "clarification"):
                    rejected.append(f"{spec.followup}: follow-up budget for this topic spent")
                    continue
                if self.consec_follow >= cfg.MAX_CONSECUTIVE_FOLLOWUPS + 1 and spec.followup != "contradiction":
                    rejected.append(f"{spec.followup}: too many consecutive follow-ups")
                    continue
                if self.phase_q.get(self.phase, 0) >= CAPS.get(self.phase, 9) + (2 if spec.followup in ("contradiction", "scenario_simplify") else 0):
                    rejected.append(f"{spec.followup}: phase budget spent")
                    continue
                d = self._try(spec, it.text, it, rejected)
                if d:
                    d.rejected = rejected
                    return self._commit(d)
        # 4. next topic
        return self._next_topic(ctx, it, rejected)

    # ── dialogue helpers ─────────────────────────────────────────────────────────────────────────
    def _reask(self, ctx: QuestionContext, reason: str, rejected: List[str], same: bool) -> Decision:
        prev = self.asked[-1] if self.asked else ctx.text
        if same:
            text = prev
        else:
            alt = None
            if ctx.targets:
                c = K.get(ctx.targets[0])
                alt = (c.q.get("scenario") if c else None)
                if alt and alt.strip() != prev.strip():
                    from question_engine_v58 import _strip_ack
                    text = "In simpler terms: " + _strip_ack(alt)
                else:
                    text = "Put simply: " + prev
            else:
                text = "Put simply: " + prev
        spec = Spec(ctx.kind if ctx.kind != "clarify" else "knowledge", list(ctx.targets), ctx.level, ctx.aspect, ctx.anchor_project, "clarification", reason=reason,
                    new_evidence="the answer to the question that was not understood", why_insufficient="candidate did not understand or hear the question", soft=ctx.soft)
        d = Decision(spec, text, True, self.phase)
        d.rejected = rejected
        self.turn += 1
        self.asked.append(text)
        self.asked_specs.append(spec)
        self.last_spec = spec
        return d

    def _close(self, reason: str, rejected: List[str]) -> Decision:
        spec = Spec("closing", reason=reason, new_evidence="final remarks", why_insufficient="")
        if self.phase == "closing" and self.phase_q.get("closing", 0) >= 1 or reason.startswith("candidate asked to end"):
            d = Decision(spec, "", False, "closing", close=True)
            d.rejected = rejected
            self.finished = True
            return d
        self.phase = "closing"
        spec.reason = reason
        d = self._try(spec, allow_dup=True)
        d.new_phase = "closing"
        d.rejected = rejected
        return self._commit(d)

    # ── candidate hooks for contextual behavioural questions ─────────────────────────────────────
    def _collect_hooks(self, it: Interpretation, ctx: QuestionContext):
        import re
        n = it.text.lower()
        if "teamwork" in it.soft and re.search(r"divid\w+|share[ds]? the work|each (?:one|person)|different part", n) and "teamwork" not in self.hooks:
            self.hooks["teamwork"] = "dividing the work between group members"
        for c in it.claims:
            if c.stance == "learning" and not c.implied and "learning_ability" not in self.hooks:
                self.hooks["learning_ability"] = K.name_of(c.key)

    # ── follow-up policy ─────────────────────────────────────────────────────────────────────────
    def _followup_specs(self, ctx: QuestionContext, it: Interpretation) -> List[Spec]:
        out: List[Spec] = []
        # a) a real contradiction (only once per competency)
        for k in dict.fromkeys(it.contradictions):
            if k not in self.contradiction_asked and k in self.g.comp:
                self.contradiction_asked.add(k)
                p = next((pj.short_name for pj in self.resume.projects if k in pj.tech_keys() or pj.implied.get(k, 0) >= 0.6), "")
                out.append(Spec("contradiction", [k], "basic", "", p, "contradiction", reason=f"the answer conflicts with earlier evidence about {_name(k)}",
                                new_evidence="which statement is accurate", why_insufficient="two statements about the same competency disagree"))
        if ctx.kind in ("intro", "closing", "contradiction", "clarify"):
            return out
        if ctx.kind == "behavioural":
            return out + self._behavioural_followup(ctx, it)
        if ctx.anchor_project and ctx.anchor_project in self.g.projects and (ctx.kind == "project_walkthrough" or (ctx.kind in ("project_probe", "followup") and ctx.aspect)):
            return out + self._project_followup(ctx, it)
        if ctx.targets:
            return out + self._concept_followup(ctx, it)
        return out

    def _concept_followup(self, ctx: QuestionContext, it: Interpretation) -> List[Spec]:
        specs: List[Spec] = []
        key = ctx.targets[0]
        c = self.g.comp.get(key)
        if not c:
            return specs
        cdef = K.get(key)
        time_ok = self.time_frac() < 0.85
        # cluster gate: dig into the members the candidate said they have touched, else give ONE scenario for the most important member
        if ctx.kind == "cluster_gate":
            pos = [(self.g.comp[cl.key].importance, cl.key) for cl in it.claims if not cl.implied and cl.stance in ("claim", "practical", "learning", "theoretical") and cl.key in self.g.comp and cl.key in self.job.requirements]
            if pos:
                k = sorted(pos, reverse=True)[0][1]
                specs.append(Spec("knowledge", [k], "basic", "", "", "evidence_probe", reason=f"candidate says they have touched {_name(k)} in the stack question",
                                  new_evidence=f"concrete evidence of {_name(k)}", why_insufficient="the gate question only establishes familiarity, not what was done"))
            elif time_ok:
                members = [m for m in ctx.targets if K.get(m) and K.get(m).q.get("scenario")]
                best = sorted(members, key=lambda m: -self.g.comp[m].importance)
                pick = next((m for m in best if m in ("sqlserver", "sql")), best[0] if best else None)
                if pick and not self.g.comp[pick].scenario_offered:
                    specs.append(Spec("scenario", [pick], "scenario", "", "", "scenario_simplify", scenario=True,
                                      reason=f"candidate has not used the {_join_names(ctx.targets)} stack; test transferable understanding through {_name(pick)}",
                                      new_evidence=f"conceptual understanding of {_name(pick)} without hands-on use", why_insufficient="no hands-on experience reported"))
            return specs
        said_no = it.intent in ("explicit_unknown", "no_experience", "negation") or c.said_unknown or c.status == EXPLICIT_NO_EXPERIENCE
        scen = bool(cdef and cdef.q.get("scenario"))
        if said_no:
            if ctx.kind == "knowledge" and not c.scenario_offered and scen and c.importance >= 2 and time_ok:
                specs.append(Spec("scenario", [key], "scenario", "", "", "scenario_simplify", scenario=True,
                                  reason=f"{_name(key)} is a required competency and the candidate hasn't used it; check design-level understanding instead",
                                  new_evidence=f"conceptual understanding of {_name(key)}", why_insufficient="candidate has no hands-on experience, so practical evidence is unavailable"))
            return specs
        if it.intent == "skip":
            return specs
        cl = it.claim_for(key)
        stance = cl.stance if cl else ""
        missing = [m.split(":", 1)[1] for m in it.missing_information if m.startswith(key + ":")]
        covered = sorted({f for f in c.facets_covered})
        if stance == "learning":
            specs.append(Spec("followup", [key], "impl", "", "", "learning_to_impl", reason=f"candidate learned {_name(key)} but has not built with it",
                              new_evidence="whether they can turn study into an implementation plan", why_insufficient="learning-only: no implementation evidence"))
        elif stance == "claim" or (it.intent in ("affirmation",) and not cl):
            specs.append(Spec("followup", [key], "basic", "", "", "evidence_probe", hook_quote=it.text, reason=f"candidate claims {_name(key)} without detail",
                              new_evidence=f"one concrete example of using {_name(key)}", why_insufficient="a bare 'yes' gives no evidence of what was done"))
        elif it.quality in ("vague", "none") and not it.question_answered:
            specs.append(Spec("followup", [key], ctx.level, "", "", "evidence_probe", hook_quote=it.text, reason=f"the answer did not address {_name(key)}",
                              new_evidence=f"an on-topic answer about {_name(key)}", why_insufficient="the answer was off-topic or too vague to assess"))
        elif it.quality in ("partial", "vague") and [m for m in missing if (key, m) in FACET_Q]:
            missing = [m for m in missing if (key, m) in FACET_Q]
            f = missing[0]
            specs.append(Spec("followup", [key], ctx.level, "", "", "facet_probe", missing_facets=[f], covered_facets=covered, hook_quote=it.text,
                              reason=f"{_name(key)} answer covered {len(covered)} of {len(cdef.facets) if cdef else '?'} evidence areas; '{f}' is missing",
                              new_evidence=f"how they handle {f}", why_insufficient=f"answer did not mention {f}"))
        elif it.quality in ("good", "strong") and c.importance >= 3 or (it.quality == "strong" and c.expected_depth == "depth"):
            idx = K.LEVELS.index(ctx.level) if ctx.level in K.LEVELS else 0
            cap = LADDER_CAP.get(c.expected_depth, 2)
            vals = self.values()
            breadth_first = bool(vals) and vals[0][0] >= 1.6 and self.phase == "technical"     # unasked high-value requirements outrank going deeper
            if idx < cap and time_ok and not breadth_first and cdef and any(lv in cdef.q for lv in K.LEVELS[idx + 1:cap + 1]):
                nxt = next(lv for lv in K.LEVELS[idx + 1:cap + 1] if lv in cdef.q)
                specs.append(Spec("followup", [key], nxt, "", "", "technical_depth", reason=f"strong answer on {_name(key)}; {c.importance=} so step up the difficulty ladder to '{nxt}'",
                                  new_evidence=f"{nxt}-level understanding of {_name(key)}", why_insufficient="basics are established; depth is not yet shown"))
        return specs

    def _project_followup(self, ctx: QuestionContext, it: Interpretation) -> List[Spec]:
        pj = ctx.anchor_project
        ps = self.g.projects[pj]
        p = self.resume.project_by_name(pj)
        if it.intent in ("explicit_unknown", "no_experience", "negation") and it.ownership == "none":
            return []
        if it.ownership in ("strong", "some", "none"):
            ps.ownership = it.ownership
        done = set(ps.aspects)
        if ps.ownership != "unknown":
            done.add("ownership")
        rank_idx = next((i for i, (_v, n, _c) in enumerate(self.rank) if n == pj), 0)
        data_centric = bool(p and any(k in p.tech_keys() or p.implied.get(k, 0) >= 0.6 for k in ("yolo", "tensorflow", "keras", "pytorch", "sklearn", "od", "dl", "cv")) and not any(
            k in p.tech_keys() for k in ("llm_api", "llm")))
        chain = ["ownership", "dataset", "labeling", "training", "evaluation", "challenge"] if data_centric else ["ownership", "deployment", "challenge", "evaluation"]
        # follow-up budget: the first (highest-value) project gets up to 3 probes, later ones only fill genuine gaps
        budget = 3 if rank_idx == 0 and data_centric else 2 if rank_idx == 0 else 1
        if ps.asked > budget:
            return []
        if ps.ownership != "unknown" and it.quality in ("good", "strong", "partial") and not data_centric and rank_idx > 0:
            return []
        if it.quality in ("good", "strong") and len(done) >= 4:
            return []
        last_aspects = [f["aspect"] for f in it.facts]
        order = [a for a in chain if a != "ownership"]
        nxt = None
        if "ownership" not in done:
            nxt = "ownership"
        else:
            idxs = [order.index(a) for a in last_aspects if a in order]
            start = max(idxs) + 1 if idxs else 0          # follow the thread: ask about the step AFTER the one just described
            for a in order[start:] + order[:start]:
                if a not in done and a not in last_aspects:
                    nxt = a
                    break
        if nxt is None:
            return []
        hook = ""
        for f in it.facts:
            if f["aspect"] in ("dataset", "labeling", "training", "evaluation", "deployment", "challenge", "function") and f["aspect"] != nxt:
                hook = f["phrase"]
                break
        followtype = {"ownership": "ownership_probe", "dataset": "implementation", "labeling": "implementation", "training": "technical_depth", "evaluation": "result_eval",
                      "challenge": "debugging", "deployment": "implementation"}.get(nxt, "implementation")
        tech = ""
        if p:
            for t in p.technologies:
                if t.key in ("yolo", "tensorflow", "keras", "pytorch", "sklearn"):
                    tech = t.name.split(" / ")[0]
                    break
        why_ins = {"ownership": "the answer did not make clear which parts the candidate personally built", "dataset": "the data side is not yet described",
                   "labeling": "the candidate described where the data came from but not how it was labelled", "training": "training choices have not been described",
                   "evaluation": "no evidence yet of how the result was measured", "challenge": "no evidence yet of problem solving or debugging", "deployment": "how it runs end-to-end is unclear"}.get(nxt, "needs more detail")
        if it.quality == "vague" and nxt != "ownership":
            why_ins = "the answer was too general to assess; asking for the next concrete step"
        return [Spec("project_probe", [], "impl", nxt, pj, followtype, hooks=[hook] if hook else [], hook_quote=it.text if not hook else "", tech=tech,
                     reason=f"project follow-up on {pj}: next evidence area is '{nxt}'", new_evidence=f"{nxt} evidence for {pj}", why_insufficient=why_ins)]

    def _behavioural_followup(self, ctx: QuestionContext, it: Interpretation) -> List[Spec]:
        sk = ctx.soft
        if not sk or sk not in self.g.soft:
            return []
        s = self.g.soft[sk]
        if s.strength >= 0.5 or s.asked >= 2 or it.intent in ("explicit_unknown", "no_experience"):
            return []
        return [Spec("behavioural", [], "basic", "", ctx.anchor_project, "behavioural_context", soft=sk, hooks=[], reason=f"the {s.name} answer had no clear personal action/result",
                     new_evidence="what the candidate personally did and what happened", why_insufficient="no concrete action or outcome described",
                     extra={"text": ""})]

    # ── next topic ───────────────────────────────────────────────────────────────────────────────
    def _advance_phase(self):
        i = PHASE_ORDER.index(self.phase)
        self.phase = PHASE_ORDER[min(i + 1, len(PHASE_ORDER) - 1)]

    def _phase_done(self) -> bool:
        ph = self.phase
        if ph == "intro":
            return self.phase_q["intro"] >= 1
        if ph == "resume":
            if not self.resume.projects:
                return True
            return self.phase_q["resume"] >= CAPS["resume"] or (len(self.projects_done) >= 2) or self.time_frac() > 0.40 or (self.phase_q["resume"] >= 2 and not self._more_projects())
        if ph == "technical":
            nothing = not self.values() or self.values()[0][0] < 0.9
            return self.phase_q["technical"] >= CAPS["technical"] or self.time_frac() > 0.80 or nothing
        if ph == "behavioural":
            return self.phase_q["behavioural"] >= CAPS["behavioural"] or self.time_frac() > 0.95
        return False

    def _more_projects(self) -> bool:
        return any(n not in self.projects_done for _, n, _ in self.rank)

    def _next_topic(self, ctx: QuestionContext, it: Interpretation, rejected: List[str]) -> Decision:
        new_phase = ""
        if self.phase == "behavioural" and self._phase_done() and self.clock < 0.85 * cfg.MIN_SECONDS and self.turn < self.max_q - 1 and self.values() and self.values()[0][0] >= 0.7 and not getattr(self, "_reopened", False):
            self._reopened = True                      # finishing at 6 minutes would leave high-value requirements unasked: reopen the technical phase once
            self.phase = "technical"
            CAPS["technical"] = self.phase_q["technical"] + 3
            new_phase = "technical"
        for _ in range(6):
            if self._phase_done():
                before = self.phase
                self._advance_phase()
                new_phase = self.phase if self.phase != before else new_phase
                if self.phase == "closing":
                    return self._close("planned interview complete" if self.time_frac() < 0.95 else "time budget reached", rejected)
            d = self._topic_in_phase(it, rejected)
            if d:
                d.new_phase = new_phase if (new_phase and not d.followup) else ""
                d.switching_topic = True
                d.rejected = rejected
                return self._commit(d)
            # nothing valid in this phase: advance
            self.phase_q[self.phase] = 99
        return self._close("no further valuable question", rejected)

    def _topic_in_phase(self, it: Interpretation, rejected: List[str]) -> Optional[Decision]:
        if self.phase == "resume":
            for val, name, cov in self.rank:
                if name in self.projects_done:
                    continue
                if self.projects_done and not cov:
                    rejected.append(f"project:{name}: covers no JD requirement")
                    continue
                if self.projects_done:
                    new = [r for r in cov if not any(r in x[2] for x in self.rank if x[1] in self.projects_done)]
                    if not new:
                        rejected.append(f"project:{name}: adds no new JD coverage ({', '.join(cov[:3])})")
                        continue
                pobj = self.resume.project_by_name(name)
                named = [t.key for t in (pobj.technologies if pobj else []) if not t.key.startswith("custom:")][:3]
                tgt = [k for k in named if k in self.g.comp]
                spec = Spec("project_walkthrough", tgt, "basic", "ownership", name, "", reason=f"highest-value unexplored résumé project (JD value {val}; covers {', '.join(cov[:4]) or 'general engineering'})",
                            new_evidence=f"what {name} does and which parts the candidate personally built", why_insufficient="no previous question about this project")
                d = self._try(spec, rejected=rejected)
                if d:
                    self.projects_done.append(name)
                    return d
            return None
        if self.phase == "technical":
            vals = self.values()
            rejected.extend([f"{_name(k)}: value {v}" for v, k in vals[3:6]])
            for v, k in vals[:6]:
                spec = self._spec_for(k)
                if spec is None:
                    continue
                spec.reason = spec.reason or f"{_name(k)} is a required JD competency (importance {self.g.comp[k].importance}) with no interview evidence yet (value {v})"
                spec.new_evidence = spec.new_evidence or f"{self._expect(k)}"
                spec.why_insufficient = spec.why_insufficient or ("no résumé support: the requirement is unverified" if self.g.comp[k].resume_level == "none" else f"résumé evidence is '{self.g.comp[k].resume_level}', not interview proof")
                d = self._try(spec, rejected=rejected)
                if d:
                    return d
            return None
        if self.phase == "behavioural":
            order = sorted(self.g.soft, key=lambda k: (self.g.soft[k].strength >= 0.45, BEHAVIOURAL_PRIORITY.index(k) if k in BEHAVIOURAL_PRIORITY else 99))
            for k in order[:5]:
                s = self.g.soft[k]
                if s.strength >= 0.45 or s.asked >= 1 or k not in BEHAVIOURAL_PRIORITY:
                    continue
                hooks = []
                if k == "teamwork":
                    hk = self.hooks.get("teamwork")
                    if hk:
                        hooks = [hk]
                    elif self.resume.extracurricular:
                        hooks = []                              # the generic wording below mentions a group project or the society; nothing is claimed
                if k == "learning_ability" and self.hooks.get("learning_ability"):
                    hooks = [self.hooks["learning_ability"]]
                pj = self.projects_done[0] if self.projects_done else (self.resume.projects[0].short_name if self.resume.projects else "")
                spec = Spec("behavioural", [], "basic", "", pj if k in ("problem_solving", "analytical_thinking") else "", "", hooks=hooks, soft=k,
                            reason=f"{s.name} is a JD behavioural competency with no evidence yet" + (" — grounded in what the candidate said" if hooks else " — grounded in their résumé"),
                            new_evidence="a concrete situation, the candidate's own action and the result", why_insufficient="behavioural area not yet covered")
                d = self._try(spec, rejected=rejected)
                if d:
                    return d
            return None
        return None

    def _expect(self, k: str) -> str:
        c = K.get(k)
        return "evidence of " + (", ".join(list(c.facets)[:4]) if c and c.facets else _name(k))

    def _spec_for(self, k: str) -> Optional[Spec]:
        c = self.g.comp[k]
        cdef = K.get(k)
        # a cluster gate when several related requirements are unresolved and none has project-level résumé support
        if cdef and cdef.cluster and cdef.cluster not in self.gate_done:
            mem = [m for m in self.g.required_keys() if K.get(m) and K.get(m).cluster == cdef.cluster and self._need(self.g.comp[m]) > 0 and self.g.comp[m].resume_level in ("none", "listed")]
            if len(mem) >= 3:
                mem = sorted(mem, key=lambda m: (-self.g.comp[m].importance, m))
                self.gate_done.add(cdef.cluster)
                names = [self.g.comp[m].name for m in mem]
                return Spec("cluster_gate", mem, "basic", "", "", "knowledge_validation", cluster_names=names,
                            reason=f"{len(mem)} related required stack items ({', '.join(names)}) are unverified; one gate question covers them efficiently",
                            new_evidence="which of these the candidate has actually used", why_insufficient="no résumé project and no interview evidence for these items")
        idx = K.LEVELS.index(self._level_for(c))
        level = K.LEVELS[idx]
        proj = ""
        targets = [k]
        if cdef and cdef.bridge_q:
            for p in self.resume.projects:
                if any(t.key in cdef.bridge_from for t in p.technologies) or any(p.implied.get(b, 0) >= 0.6 for b in cdef.bridge_from):
                    proj = p.short_name
                    tk = next((t for t in p.technologies if t.key in cdef.bridge_from), None)
                    if tk:
                        for k2, w in K.closure(tk.key).items():
                            if w >= 0.7 and k2 in self.g.comp and self.g.comp[k2].required and k2 not in targets:
                                targets.append(k2)
                    break
        kind = "project_probe" if proj and level == "basic" else "knowledge"
        sp = Spec(kind, targets[:3], level, "", proj if kind == "project_probe" else "", "knowledge_validation")
        return sp

    def _level_for(self, c) -> str:
        if not c.addressed:
            return "basic"
        idx = min(c.depth + 1, LADDER_CAP.get(c.expected_depth, 2))
        return K.LEVELS[idx]


def _join_names(keys: List[str]) -> str:
    return ", ".join(_name(k) for k in keys)


def _article_member(ex: str) -> str:
    import re
    m = re.match(r"^\s*member\s*[–—-]\s*(.+)$", ex, re.I)
    return f"a member of {m.group(1).strip()}" if m else ex
