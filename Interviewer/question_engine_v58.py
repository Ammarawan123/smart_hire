"""
question_engine_v58.py — grounded, validated question generation.

Order of preference for every question:
  1. Deterministic semantic builder  (concept question bank + candidate's own facts as hooks + résumé/JD grounding)
  2. Optional Qwen rewrite           only for follow-ups whose hook is weak (quote-based) — strict JSON, then validated
  3. Deterministic fallback           never a random generic template: the builder always knows the target, level and aspect
Every candidate question must pass `validate_question` (is a question, relevant to its target, grounded — no technology that is not
in the résumé / JD / candidate's answers / concept bank —, not a near-duplicate of anything already asked, asks for NEW evidence).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import concepts_v58 as K
import v58_config as cfg
from models_v58 import QuestionContext
from textkit_v58 import clip, content_tokens, cosine_sets, word_count

FACET_LABEL = {
    "ingestion": "document ingestion", "chunking": "chunking", "embeddings": "embeddings", "retrieval": "retrieval", "grounding": "grounding the answer in the retrieved text",
    "evaluation": "evaluation", "structure": "prompt structure", "iteration": "iterating on prompts", "control": "controlling the output", "join": "joins", "aggregate": "aggregation",
    "filter_sort": "filtering and sorting", "design": "schema design", "methods": "HTTP methods", "status": "status codes", "format": "data formats", "request_flow": "the request flow",
    "components": "components", "binding": "data binding", "class_object": "classes and objects", "pillars": "the OOP pillars", "phases": "the lifecycle phases", "testing": "testing",
    "vcs": "version control", "types": "application types", "layers": "layers", "tradeoffs": "trade-offs", "sourcing": "data sourcing", "labeling": "labelling", "training": "training",
    "calling": "calling the API", "prompting": "prompting", "limits": "limitations", "models": "the models", "usage": "how it's used", "data_split": "data splitting", "overfit": "overfitting",
    "metrics": "metrics", "setup": "model setup", "data": "the data", "model": "the model", "eval": "evaluation",
}

FACET_Q = {
    ("rag", "chunking"): "How would you split the documents into chunks for retrieval, and what changes if the chunks are too large or too small?",
    ("rag", "embeddings"): "How would you turn the chunks into something a computer can search by meaning, and how is that different from keyword search?",
    ("rag", "retrieval"): "Once the question arrives, how would you find the right chunks, and how many would you pass to the model?",
    ("rag", "grounding"): "How would you make sure the model answers only from the retrieved text and says so when the answer isn't there?",
    ("rag", "evaluation"): "How would you check that the system retrieves the right passages and doesn't make things up?",
    ("rag", "ingestion"): "How would the documents get from files, such as PDFs, into the system in the first place?",
    ("prompt_eng", "structure"): "What did one of your prompts actually look like: did you give the model a role, rules or an example of the answer you wanted?",
    ("prompt_eng", "iteration"): "When a prompt gave a bad answer, how did you work out what to change, and how did you check the new version was better?",
    ("prompt_eng", "control"): "How did you control the length and format of the answers, for example a word limit or a fixed JSON shape?",
    ("llm_api", "errors"): "What happened in your code when the API call failed or returned something unexpected?",
    ("sqlserver", "join"): "How would you combine rows from two tables, and what is the difference between an inner join and a left join?",
    ("rest_api", "status"): "Which HTTP status codes would you return when a record is created, when it is missing, and when the request is invalid?",
    ("rest_api", "methods"): "Which HTTP methods would you use to read, create, update and delete a record, and why?",
    ("oop", "pillars"): "Can you give me a small example of inheritance or polymorphism from code you've written?",
    ("oop", "class_object"): "What is the difference between a class and an object? Use an example from your own code.",
    ("ml", "overfit"): "What did you do to stop your model from memorising the training data?",
    ("ml_libs", "evaluation"): "How did you measure whether the model was good, and on which data?",
    ("ml_libs", "training"): "What settings did you choose when training, and how did you decide on them?",
    ("sdlc", "phases"): "In which order would you do requirements, design, coding and testing, and what would you produce at each step?",
    ("architecture", "layers"): "In a web app, which parts would run in the browser, which on the server, and where would the data be stored?",
    ("se_practices", "testing"): "How do you check that your code works before you share it: do you write tests, and what do they look like?",
    ("se_practices", "vcs"): "How do you use Git on a project, and what do you do when two people change the same file?",
}

ASPECT_CHAIN = ["ownership", "dataset", "labeling", "training", "evaluation", "challenge", "result"]
FOLLOW_TYPES = ["clarification", "evidence_probe", "ownership_probe", "technical_depth", "why_how", "implementation", "result_eval", "tradeoff", "debugging",
                "contradiction", "knowledge_validation", "scenario_simplify", "learning_to_impl", "facet_probe", "behavioural_context"]


_IMPERATIVE = re.compile(r"\b(?:please\s+)?(?:walk me through|tell me|explain|describe|talk me through|pick|give me|show me|take me through|say|think of|imagine|suppose|picture|let's talk about|let's look at)\b", re.I)


def is_interview_prompt(t: str) -> bool:
    """A real ask: ends with '?' or contains an imperative interview request (\"Walk me through ...\"). Statements are rejected."""
    t = t.strip()
    if "?" in t:
        return True
    return bool(_IMPERATIVE.search(t)) and t.endswith((".", "!")) and word_count(t) >= 8


@dataclass
class Spec:
    kind: str                                  # QuestionContext.kind
    targets: List[str] = field(default_factory=list)
    level: str = "basic"
    aspect: str = ""
    project: str = ""
    followup: str = ""                         # one of FOLLOW_TYPES for follow-ups
    hooks: List[str] = field(default_factory=list)       # phrases the candidate said ("downloaded existing images and ...")
    hook_quote: str = ""
    soft: str = ""
    scenario: bool = False
    missing_facets: List[str] = field(default_factory=list)
    covered_facets: List[str] = field(default_factory=list)
    tech: str = ""
    reason: str = ""
    new_evidence: str = ""
    why_insufficient: str = ""
    topic_label: str = ""
    cluster_names: List[str] = field(default_factory=list)
    extra: Dict[str, str] = field(default_factory=dict)

    def ctx(self, text: str, turn: int) -> QuestionContext:
        return QuestionContext(self.kind, list(self.targets), self.project, self.aspect, self.level, self.scenario or self.kind == "scenario", text, turn, self.soft)


def _name(k: str) -> str:
    return K.name_of(k)


def _join(items: List[str], conj: str = "and") -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {conj} " + items[-1]


def _strip_ack(q: str) -> str:
    return re.sub(r"^(?:No problem|That's fine|No worries)[.,!]\s*", "", q).strip()


class QuestionEngine:
    def __init__(self, resume, job, graph, llm=None):
        self.resume, self.job, self.graph, self.llm = resume, job, graph, llm
        self.llm_calls = 0
        self.rejections: List[str] = []
        self._allowed_cache: Optional[set] = None
        voc = set()
        for c in K.REG.values():
            voc |= content_tokens(c.name)
            for a in c.aliases:
                voc |= content_tokens(a)
        for p in resume.projects:
            voc |= content_tokens(p.short_name)
        for s in resume.all_skills():
            voc |= content_tokens(s.name)
        self.topic_vocab = voc

    # ── grounding vocabulary ────────────────────────────────────────────────────────────────────
    def allowed_tech(self, spec: Spec, answer_text: str = "") -> set:
        s = set(self.resume.skill_keys()) | set(self.job.requirements) | set(spec.targets)
        for p in self.resume.projects:
            s |= set(p.tech_keys()) | set(p.implied)
        for k in spec.targets:
            c = K.get(k)
            if c:
                s |= set(c.implies) | {k}
        for k in spec.targets:                      # alternatives named inside the concept's own question text ("Angular, React, or plain JavaScript")
            c = K.get(k)
            if c:
                for txt in c.q.values():
                    s |= {m[0] for m in K.find_mentions(txt)}
                s |= {m[0] for m in K.find_mentions(c.bridge_q or "")}
        s |= {m[0] for m in K.find_mentions(answer_text)}
        s |= {m[0] for m in K.find_mentions(spec.hook_quote)}
        for h in spec.hooks:
            s |= {m[0] for m in K.find_mentions(h)}
        return s

    # ── validation ─────────────────────────────────────────────────────────────────────────────
    def validate(self, text: str, spec: Spec, asked: List[str], answer_text: str = "", prev_specs: Optional[List[Spec]] = None) -> Tuple[bool, str]:
        t = text.strip()
        if not is_interview_prompt(t):
            return False, "not_a_question"
        if t.count("?") > 2:
            return False, "multiple_questions"
        wc = word_count(t)
        if wc < 6 or wc > 62:
            return False, "length"
        if re.search(r"\b(as an ai|language model|i cannot|json|certainly!|sure!)\b", t, re.I):
            return False, "meta_text"
        if re.search(r"[\w.+-]+@[\w-]+\.\w+|https?://", t):
            return False, "contact_info_in_question"
        allowed = self.allowed_tech(spec, answer_text)
        for key, a, b, al in K.find_mentions(t):
            if key not in allowed and K.get(key) and K.get(key).kind == "technology":
                return False, f"ungrounded_technology:{al}"
        # duplicate = same *ask*. Topic words (target / project names) are removed first, so "Have you used C#?" and
        # "Pick one concrete thing you did with C#" are different asks about the same topic, while two ownership questions are not.
        topic = self.topic_vocab
        qt = content_tokens(t) - topic
        qfull = content_tokens(t)
        for pi, prev in enumerate(asked):
            ps = prev_specs[pi] if prev_specs and pi < len(prev_specs) else None
            if ps is not None:
                same_topic = bool(set(ps.targets) & set(spec.targets)) or bool(ps.project and ps.project == spec.project) or (not ps.targets and not spec.targets and not ps.project and not spec.project and ps.soft == spec.soft)
                if not same_topic:
                    continue            # the same template aimed at a different topic/project is a different question
            pt_full = content_tokens(prev)
            pt = pt_full - topic
            sim = cosine_sets(qt, pt) if len(qt) >= 3 and len(pt) >= 3 else 0.0
            if sim >= cfg.DEDUP_THRESHOLD or cosine_sets(qfull, pt_full) >= 0.85:
                return False, "duplicate_of_previous"
        if spec.kind not in ("behavioural", "intro", "closing", "clarify", "contradiction") and spec.targets:
            names = [content_tokens(_name(k)) for k in spec.targets] + [content_tokens(a) for k in spec.targets if K.get(k) for a in K.get(k).aliases]
            proj = content_tokens(spec.project) if spec.project else set()
            hooks = content_tokens(" ".join(spec.hooks))
            if not any(n and n & qfull for n in names) and not (proj & qfull) and not (hooks & qfull) and spec.kind != "scenario":
                # concept-bank questions describe the concept without naming it ("two tables, customers and orders") — accept when they come from the bank
                if spec.extra.get("source") != "bank":
                    return False, "irrelevant_to_target"
        return True, ""

    # ── deterministic builders ──────────────────────────────────────────────────────────────────
    def _project_techs(self, project: str, limit: int = 3) -> List[str]:
        p = self.resume.project_by_name(project)
        if not p:
            return []
        names = [t.name for t in p.technologies if not t.key.startswith("custom:")]
        return names[:limit]

    def realize(self, spec: Spec) -> str:
        k = spec.kind
        if k == "intro":
            return "Tell me a bit about your background and what draws you to the role."
        if k == "closing":
            return "Before we wrap up, is there anything about your experience you would like me to know, or any question for me?"
        if k == "project_walkthrough":
            techs = _join(self._project_techs(spec.project))
            p = self.resume.project_by_name(spec.project)
            sub = f" ({p.subtitle})" if p and p.subtitle and len(p.subtitle) < 50 else ""
            if techs:
                return f"Let's talk about {spec.project}{sub}. What problem does it solve, and which parts did you build yourself — for example the parts using {techs}?"
            return f"Let's talk about {spec.project}{sub}. What problem does it solve, and which parts did you build yourself?"
        if k == "contradiction":
            nm = _name(spec.targets[0]) if spec.targets else "that"
            where = f" in {spec.project}" if spec.project else ""
            return f"I want to record this correctly. Your résumé mentions {nm}{where}, and a moment ago you said you haven't used it. Which is closer to the truth?"
        if k == "clarify":
            return spec.extra.get("text") or "Could you say that again with one concrete example from your own work?"
        if k == "behavioural":
            return self._behavioural(spec)
        if k in ("project_probe", "followup") and spec.project and spec.followup in ("", "ownership_probe", "why_how", "implementation", "result_eval", "tradeoff", "debugging", "facet_probe", "technical_depth", "evidence_probe") \
                and not (spec.targets and spec.followup in ("technical_depth", "facet_probe") and not spec.aspect):
            return self._project_followup(spec)
        return self._concept_question(spec)

    # concept (knowledge) questions ---------------------------------------------------------------
    def _concept_question(self, spec: Spec) -> str:
        key = spec.targets[0] if spec.targets else ""
        c = K.get(key)
        nm = _name(key) if key else "this topic"
        res_lvl = self.graph.get(key).resume_level if self.graph.get(key) else "none"
        # a specific facet probe: "You covered X and Y. How would you handle Z?"
        if spec.followup == "facet_probe" and spec.missing_facets and c:
            f = spec.missing_facets[0]
            fq = FACET_Q.get((key, f))
            if fq:
                return fq
            lab = FACET_LABEL.get(f, f.replace("_", " "))
            have = _join([FACET_LABEL[x] for x in spec.covered_facets if x in FACET_LABEL and x not in ("usage", "models")][:3])
            lead = f"You covered {have}. " if have else ""
            return f"{lead}How would you handle {lab} in {nm}?"
        if spec.followup == "learning_to_impl":
            return f"You said you've learned about {nm} but haven't built with it. If you had to implement it in a small project, what would be the first three things you'd do?"
        if spec.followup == "evidence_probe":
            n_prev = sum(1 for s in getattr(self, "_probe_log", []))
            self._probe_log = getattr(self, "_probe_log", []) + [key]
            variants = [f"You said you've worked with {nm}. Pick one concrete thing you did with it and walk me through how it worked.",
                        f"What is the most substantial thing you have done with {nm}, and what was your own part in it?",
                        f"Give me one real example where you used {nm}: what was the task, and what did you do?"]
            return variants[n_prev % len(variants)]
        if spec.kind == "cluster_gate":
            names = spec.cluster_names or [_name(t) for t in spec.targets]
            return f"This role uses {_join(names)}. Have you used any of these, even in a small project? Tell me which one is closest to something you've built."
        if spec.scenario or spec.level == "scenario" or spec.followup == "scenario_simplify":
            q = (c.q.get("scenario") if c else "") or f"Suppose you had to use {nm} in a small project. How would you approach it?"
            return _strip_ack(q)
        if c:
            if res_lvl == "listed" and spec.level == "basic" and spec.followup == "" and c.kind in ("technology",) and not c.q.get("basic", "").startswith("Your"):
                return f"Your résumé lists {nm}. Tell me about one thing you actually did with it."
            if spec.followup == "knowledge_validation" or spec.level == "basic" or spec.level not in c.q:
                bridge = self._bridge(c, spec)
                if bridge:
                    return bridge
            q = c.q.get(spec.level) or c.q.get("impl") or c.q.get("basic")
            if q:
                spec.extra["source"] = "bank"
                return _strip_ack(q)
        generic = {
            "basic": f"How would you explain {nm} to a teammate, and where have you used it?",
            "impl": f"Walk me through how you would implement {nm} in a small project.",
            "depth": f"What are the common mistakes or limitations to watch for with {nm}?",
            "tradeoff": f"When would you choose {nm}, and when would you pick an alternative?",
            "debug": f"Something goes wrong with {nm} in production. Where would you start looking, and why?",
        }
        return generic.get(spec.level, generic["basic"])

    def _bridge(self, c, spec: Spec) -> str:
        """Ground a concept question in something the candidate actually built (résumé project + technology)."""
        if not c.bridge_q or not c.bridge_from:
            return ""
        for bk in c.bridge_from:                                   # earlier keys in bridge_from are the better fit
            for p in self.resume.projects:
                techs = [t for t in p.technologies if t.key == bk]
                if not techs and p.implied.get(bk, 0) < 0.6:
                    continue
                names = [t.name.split(" / ")[0] for t in p.technologies if t.key in c.bridge_from][:2] if bk not in ("pygame", "ursina", "tkinter") else [techs[0].name]
                tname = _join(names) if names else _name(bk)
                spec.project, spec.tech = p.short_name, tname
                spec.extra["source"] = "bridge"
                return c.bridge_q.format(project=p.short_name, tech=tname)
        return ""

    # project follow-ups -------------------------------------------------------------------------
    def _project_followup(self, spec: Spec) -> str:
        pj = spec.project
        hook = spec.hooks[0] if spec.hooks else ""
        lead = f"You mentioned that you {hook}. " if hook else ""
        asp, ft = spec.aspect, spec.followup
        tech = spec.tech or (self._project_techs(pj, 1)[0] if self._project_techs(pj, 1) else "")
        if ft == "ownership_probe" or asp == "ownership":
            return f"In {pj}, which parts did you write or train yourself, and which came from libraries, pretrained models or tutorials?"
        if ft == "evidence_probe" and not asp:
            return f"{lead}Walk me through one concrete step you took in {pj}, from the input to the result."
        table = {
            "dataset": f"{lead}Where did the data for {pj} come from, how many examples did you end up with, and how did you decide it was enough?",
            "labeling": f"{lead}How did you label and organise those images before training{(' ' + tech) if tech else ''}, and how did you keep the labels consistent?",
            "training": f"{lead}What training settings did you choose{(' for ' + tech) if tech else ''} — for example epochs, image size or a pretrained starting point — and why?",
            "evaluation": f"{lead}How did you check that {pj} actually worked well? Which metrics or tests did you use, and what did you get?",
            "challenge": f"{lead}What was the hardest problem you hit while building {pj}, and how did you find the cause and fix it?",
            "result": f"{lead}What would you change about {pj} if you rebuilt it today, and why?",
            "deployment": f"{lead}How does {pj} run when someone actually uses it — what happens between the input and the output you show?",
        }
        if ft == "tradeoff":
            return f"{lead}Why did you choose {tech or 'that approach'} for {pj} rather than another option, and what did you give up?"
        if ft == "debugging":
            return table["challenge"]
        if ft == "result_eval":
            return table["evaluation"]
        if ft == "why_how":
            return f"{lead}Why did you build it that way rather than another way?"
        if asp in table:
            return table[asp]
        return f"{lead}Walk me through the steps you took to build {pj}, from the first idea to the working version."

    # behavioural ------------------------------------------------------------------------------
    def _behavioural(self, spec: Spec) -> str:
        soft = spec.soft
        hook = spec.hooks[0] if spec.hooks else ""
        pj = spec.project
        if soft == "teamwork":
            if hook:
                return f"You mentioned {hook}. Tell me about a time when someone else's part affected your work. What did you do?"
            return "Tell me about a time you worked with other people, for example in a group project or a society activity. When someone else's part affected yours, what did you do?"
        if soft in ("problem_solving", "analytical_thinking"):
            if pj:
                return f"In {pj}, what was the hardest thing to get working, and how did you work out the cause?"
            return "Tell me about a technical problem that took you a while to solve. How did you break it down?"
        if soft == "learning_ability":
            if hook:
                return f"You said you learned {hook}. How did you go about learning it, and how long before you could use it on your own?"
            return "Tell me about a tool or technology you had to learn quickly. How did you go about it, and what was the result?"
        if soft == "communication":
            return "Explain one of your projects to me as if I were a non-technical manager. What would you say in two or three sentences?"
        if soft == "planning":
            return "Tell me about a deadline you had to meet, for example for a university project. How did you plan the work?"
        if soft == "attention_to_detail":
            return "Describe a time a small detail caused a bug or a lost mark. How did you catch it, and what do you do now to avoid it?"
        if soft == "adaptability":
            return "Tell me about a time you had to switch to a tool or language you didn't know. How did you handle it?"
        if soft == "independence":
            return "Tell me about something you built or solved without anyone guiding you. How did you decide what to do next?"
        if soft == "decision_making":
            return "Tell me about a technical decision you made where you had to choose between two options. What did you pick, and why?"
        return "Tell me about a time you faced a difficulty in a project, and how you handled it."

    # ── LLM follow-up (only when the deterministic hook is weak) ──────────────────────────────────
    def llm_followup(self, spec: Spec, interp, base_text: str) -> Optional[str]:
        if self.llm is None or self.llm_calls >= cfg.LLM_MAX_QUESTION_CALLS or not getattr(self.llm, "ready", False):
            return None
        fact = clip(spec.hook_quote or (interp.text if interp else ""), 160)
        if not fact:
            return None
        schema = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
        sysm = ("You are a friendly technical interviewer. Write ONE short follow-up question (max 35 words) that builds on what the candidate just said "
                "and asks for the missing detail. Do not praise. Do not mention any technology the candidate did not mention. Output JSON {\"question\": \"...\"}.")
        user = f"Candidate said: \"{fact}\"\nMissing detail to ask about: {spec.new_evidence or spec.aspect or spec.followup}\nTopic: {spec.project or _name(spec.targets[0]) if spec.targets else spec.project}"
        self.llm_calls += 1
        out = self.llm.json(sysm, user, purpose="followup", schema=schema, max_tokens=cfg.LLM_QUESTION_MAX_TOKENS, timeout=cfg.LLM_QUESTION_TIMEOUT_S)
        if isinstance(out, dict) and isinstance(out.get("question"), str):
            return out["question"].strip()
        return None
