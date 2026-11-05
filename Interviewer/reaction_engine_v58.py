"""
reaction_engine_v58.py — contextual acknowledgements.

Rules (from the interview-practice literature on active listening and from the V51 failure — "Okay./Right./Alright." after everything):
  * a reaction must refer to what was actually said (a fact phrase, a covered facet, the topic) or be a plain dialogue-act reply
  * no praise unless the answer earned it (quality good/strong with >=2 evidence facets or concrete specifics)
  * never claim evidence that was not given; never say "great"/"excellent" to a vague or empty answer
  * never use bare filler (Okay / Right / Alright / Got it) as a reaction; variants rotate and are not reused within a window
  * the reaction and the next question are ONE utterance: ack + (optional transition) + question
"""
from __future__ import annotations

from typing import Dict, List, Optional

import concepts_v58 as K
from models_v58 import Interpretation, QuestionContext
from question_engine_v58 import FACET_LABEL, _join

BANNED_FILLER = {"okay.", "ok.", "right.", "alright.", "got it.", "sure.", "i see.", "fine."}


class ReactionEngine:
    def __init__(self):
        self.recent: List[str] = []
        self.count_by_kind: Dict[str, int] = {}
        self.turn_of_last_praise = -99

    def _pick(self, kind: str, options: List[str], turn: int) -> str:
        fresh = [o for o in options if o not in self.recent[-8:]] or options
        choice = fresh[(turn + self.count_by_kind.get(kind, 0)) % len(fresh)]
        self.count_by_kind[kind] = self.count_by_kind.get(kind, 0) + 1
        self.recent.append(choice)
        return choice

    def acknowledge(self, it: Interpretation, ctx: QuestionContext, decision_kind: str, turn: int, topic_name: str = "", will_hook: bool = False,
                    switching_topic: bool = False, next_label: str = "", same_topic_next: bool = False) -> str:
        """Return the lead-in text (may be empty). `will_hook` means the next question itself starts with 'You mentioned that you ...'."""
        topic = topic_name or (K.name_of(ctx.targets[0]) if ctx.targets else ctx.anchor_project or "that")
        i = it.intent
        if i == "quit":
            return ""
        if i == "skip":
            return self._pick("skip", ["Sure, we'll skip that one.", "No problem, let's leave that and move on.", "That's fine, we'll skip it."], turn)
        if i == "dispute":
            return self._pick("dispute", ["You're right, I may have misread that. I'll go by what you've told me.", "Fair point, sorry about that. I'll go with what you actually said."], turn)
        if i == "repeat":
            return self._pick("repeat", ["Of course, here it is again.", "Happy to repeat it."], turn)
        if i == "clarify_request":
            return self._pick("clarify", ["Of course, let me put it another way.", "Sure, let me rephrase that."], turn)
        if i == "help":
            return "You can answer in your own words. Say 'skip' to pass on a question, 'repeat' to hear it again, or 'quit' to end the interview."
        if i == "explicit_unknown":
            if same_topic_next:
                return self._pick("unknown_same", ["Understood. Let me make it more concrete, since I'm interested in how you would reason about it.",
                                                   "No worries. Let me put it as a concrete situation instead."], turn)
            return self._pick("unknown", ["That's fine, thank you for being straightforward.", "No worries, thank you for telling me.", "That's okay, being clear about what you haven't covered is helpful."], turn)
        if i in ("no_experience", "negation") or any(c.stance == "no_experience" and not c.implied for c in it.claims):
            names = [K.name_of(c.key) for c in it.claims if c.stance == "no_experience" and not c.implied]
            nm = _join(names[:3]) if names else topic
            if same_topic_next:
                return self._pick("noexp_same", [f"That's completely fine, {nm} isn't something every candidate has worked with yet. Let me ask about it from a different angle.",
                                                 f"Thanks for being clear about {nm}. I'm more interested in how you would think about it, so let me give you a concrete situation."], turn)
            return self._pick("noexp", [f"That's completely fine, {nm} isn't something every candidate has worked with yet.",
                                        f"Thanks for being clear about {nm}. We'll treat {'those as things' if ',' in nm or ' and ' in nm else 'it as something'} you haven't worked with yet."], turn)
        if it.anchor_is_library:
            nm = next((n.split(":")[0] for n in it.notes if "library" in n), topic)
            return f"Understood, {nm} is a library you use rather than something you wrote, so that's a fair point."
        if any(c.stance == "learning" and not c.implied for c in it.claims) and not any(c.stance == "practical" and not c.implied for c in it.claims):
            nm = next(K.name_of(c.key) for c in it.claims if c.stance == "learning" and not c.implied)
            return self._pick("learn", [f"Good, having studied {nm} is a real starting point.", f"Thanks, so {nm} is something you've learned about rather than built with so far."], turn)
        if ctx.kind == "intro":
            return self._pick("intro", ["Thanks for the introduction.", "Thanks, that gives me a good starting point."], turn)
        if ctx.kind == "closing":
            return ""
        tgt = [c for c in it.claims if not c.implied and c.key in ctx.targets and c.stance in ("practical", "theoretical")]
        facets = list(dict.fromkeys(FACET_LABEL.get(f, f.replace("_", " ")) for c in tgt for f in c.facets if f in FACET_LABEL))
        aspects = list(dict.fromkeys(f["aspect"] for f in it.facts))
        earned = it.quality in ("good", "strong") and (len(facets) >= 2 or len(aspects) >= 2)
        if earned and turn - self.turn_of_last_praise >= 3:
            self.turn_of_last_praise = turn
            if len(facets) >= 2:
                return self._pick("praise_f", [f"That's useful. You covered {_join(facets[:3])} clearly.", f"That makes sense. You touched on {_join(facets[:3])}."], turn)
            names = {"dataset": "the data", "labeling": "the labelling", "training": "the training", "evaluation": "the evaluation", "deployment": "how it runs", "challenge": "the problems you hit", "function": "what it does"}
            return self._pick("praise_a", [f"That's useful. You covered {_join([names.get(a, a) for a in aspects[:3]])} clearly."], turn)
        if it.quality in ("good", "strong", "partial") and not will_hook:
            what = ctx.anchor_project or topic
            if ctx.kind == "behavioural":
                return self._pick("neutral_b", ["Thanks for the example.", "Thank you, that is a helpful example."], turn)
            return self._pick("neutral", [f"Thanks, that gives me a clearer picture of {what}.", f"Thank you, that helps me understand {what}.", f"Thanks, that's concrete enough for {what}.", f"Good, that tells me how you approached {what}."], turn)
        if it.quality in ("good", "strong", "partial"):
            return ""
        if it.quality in ("vague", "none"):
            return self._pick("vague", ["I understand the general idea, and I'd like to make it more concrete.", "I get the outline of it. Let me ask for a little more detail."], turn)
        return ""

    def transition(self, label: str, turn: int, new_phase: str = "") -> str:
        if new_phase == "behavioural":
            return self._pick("tr_beh", ["Let's switch to how you work with other people.", "Now a couple of questions about how you work in a team."], turn)
        if new_phase == "technical":
            return self._pick("tr_tech", ["Let's move to the technical side of the role.", "Now I'd like to look at the technical areas this role uses."], turn)
        if label:
            return self._pick("tr_topic", [f"Let's look at a different area: {label}.", f"Now let's talk about {label}.", f"Moving to {label}."], turn)
        return ""


def compose(ack: str, transition: str, question: str) -> str:
    parts = [p.strip() for p in (ack, transition, question) if p and p.strip()]
    text = " ".join(parts)
    return text.replace("  ", " ")
