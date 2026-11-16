"""
interviewer_bridge_v59.py — connects the V58 interviewer to the Vision / Voice / NLP cue modules and the V59 fusion.

  V58 InterviewEngine  --(question/ctx, answer text, Interpretation)-->  BehaviorSession  --> FusionResult / probe request
  Vision v3 / Voice / STT  --(Signals per answer)------------------------^
  BehaviorAwareEngine.step(answer, signals)  ==  InterviewEngine.step(answer) + behaviour analysis + (optional) verification probe

DESIGN RULES
  * The interviewer's own state machine is untouched; the bridge wraps `planner.next`.  If the bridge fails, the interview continues
    (exceptions are logged to `session.errors`, never swallowed silently, and never stop the interview).
  * Every answer is analysed and logged.  A verification probe is only an EXTRA QUESTION (cheap, symmetric, never an accusation): at most one per
    topic, at most MAX_PROBES_PER_INTERVIEW per interview, never in the first turns, never after a skip / 'I don't know'.
  * A flag changes status only through the probe outcome: verified / unresolved / inconsistent.  The report says 'needs verification', never 'lying'.
  * Face/voice cues alone cannot raise a flag (bounded influence in fusion_v59.FusionModel).
"""
from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import behavior_config_v59 as C
import concepts_v58 as K
from claim_extractor_v59 import Claim59, extract_claims
from fusion_v59 import BaselineBank, FusionModel, FusionResult
from interview_engine_v58 import InterviewEngine
from interview_planner_v58 import Decision
from question_engine_v58 import Spec
from text_cues_v59 import has_term, NLIChecker, TextCues, claim_detail_gap, contradiction_evidence, quantity_conflict, sentences, text_cues
from vision_extras_v59 import VisionExtras, vision_extras
from voice_cues_v59 import VoiceCues, ser_to_cues, voice_cues

BUCKETS = {"intro": "intro", "project_walkthrough": "project", "project_probe": "project", "followup": "project", "knowledge": "technical",
           "scenario": "technical", "cluster_gate": "technical", "behavioural": "behavioural"}
NO_PROBE_INTENTS = {"skip", "quit", "help", "repeat", "clarify_request", "explicit_unknown", "no_experience", "negation", "empty", "affirmation", "dispute"}


@dataclass
class Signals:
    """Everything the capture layer measured during ONE answer.  All fields optional: missing modality = that modality abstains."""
    audio: Optional[np.ndarray] = None
    sr: int = 16000
    capture_t0: Optional[float] = None          # time.monotonic() of the first audio sample
    tts_end: Optional[float] = None             # time.monotonic() when the question finished playing (TimedTTS.last_end)
    speech_segments: Optional[list] = None      # [(start_s, end_s)] from Silero VAD; energy VAD is used if None
    ser_results: Optional[dict] = None          # Voice/models_arena output
    vision_record: Optional[dict] = None        # BehaviorFeatures.to_fusion_record()
    vision_samples: Optional[list] = None       # FrameSample list of the segment
    multi_face: bool = False
    phone_seen: bool = False
    # ---- V60: time-aligned inputs for the assistance-pattern detector (all optional) ----
    words: Optional[list] = None                # [(start_s, end_s, word)] relative to the start of `audio`
    segments: Optional[list] = None             # [(start_s, end_s, text)] Whisper segments, text already transcript-corrected
    ser_chunks: Optional[list] = None           # [(start_s, end_s, {model: output})] speech-emotion results per chunk of the answer
    vision_offset: float = 0.0                  # time.monotonic() = FrameSample.t + vision_offset
    corrections: Optional[list] = None          # transcript corrections applied to this answer (audit trail)


@dataclass
class TurnBehavior:
    turn: int
    segment_id: str
    bucket: str
    question: str = ""
    words: int = 0
    cues: Dict[str, Optional[float]] = field(default_factory=dict)
    z: Dict[str, Optional[float]] = field(default_factory=dict)
    hard: Dict[str, float] = field(default_factory=dict)
    fusion: Optional[FusionResult] = None
    coaching: Dict[str, Optional[float]] = field(default_factory=dict)
    integrity: Dict[str, Any] = field(default_factory=dict)
    content_notes: List[str] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)
    probe_requested: Optional[dict] = None
    probe_outcome: Optional[dict] = None
    used_as_baseline: bool = False

    def to_report(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "fusion"}
        d["fusion"] = self.fusion.to_dict() if self.fusion else None
        return d


# ───────────────────────── probe wording (grounded, neutral, one question) ─────────────────────────
PROBES = {
    "duration_check": "Your résumé lists {topic}. Roughly how long did you work with it, and what kind of task was it for?",
    "specifics": "Pick one concrete moment from your work with {topic}. What was the situation, what did you do first, and how did it turn out?",
    "sideways": "On {topic}, what was the least expected problem you ran into, and how did you track down the cause?",
    "ownership": "On {topic}, which part did you personally write or decide, as opposed to the rest of the team?",
    "own_words": "Let us go one step further on {topic}: explain that idea again in your own words, using a small example from something you have actually built.",
}


class BehaviorSession:
    def __init__(self, resume_text: str = "", topics: Optional[List[str]] = None, nli: Optional[NLIChecker] = None, model: Optional[FusionModel] = None,
                 bank: Optional[BaselineBank] = None, nuextract=None, llm=None, relevance=None, monitor=None, cuelog=None):
        self.resume_text = resume_text
        self.nli = nli
        self.relevance = relevance                  # NLP/relevance_v60.ResumeRelevance (SBERT, profile-driven) or None
        self.monitor = monitor                      # fusion/cheat_detector_v60.CheatMonitor or None
        self.cuelog = cuelog                        # fusion/cue_log_v60.CueLog or None
        self.model = model or FusionModel()
        self.bank = bank or BaselineBank()
        self.claims: List[Claim59] = extract_claims(resume_text, topics, "resume", nuextract=nuextract, llm=llm) if resume_text else []
        self.prior_sentences: List[str] = [c.quote for c in self.claims]
        self.turns: List[TurnBehavior] = []
        self.probes_used = 0
        self.probed_topics: set = set()
        self.pending_probe: Optional[dict] = None
        self.probe_log: List[dict] = []
        self.errors: List[str] = []

    # ---------- per-answer analysis ----------
    def analyze(self, turn: int, ctx, it, answer: str, sig: Optional[Signals]) -> TurnBehavior:
        sig = sig or Signals()
        bucket = BUCKETS.get(ctx.kind, "other")
        seg_id = f"q{turn}"
        tb = TurnBehavior(turn, seg_id, bucket, ctx.text, it.words)
        topic_keys = list(ctx.targets) + [c.key for c in it.claims]
        anchors = [K.name_of(k) for k in dict.fromkeys(topic_keys)] + ([ctx.anchor_project] if ctx.anchor_project else [])
        tech_names = [K.name_of(k) for k in dict.fromkeys(c.key for c in it.claims)]
        # --- text / content ---
        tc: TextCues = text_cues(answer, tech_names)
        strength = max([c.strength for c in it.claims if c.stance in ("practical", "claim")] or [0.0])
        if strength == 0.0:
            rc = [c.strength for c in self.claims if any(has_term(c.quote, a) for a in anchors)]
            strength = max(rc) if rc else 0.0
        cues: Dict[str, Optional[float]] = {"detail_density": tc.detail_density if tc.usable else None,
                                            "claim_detail_gap": claim_detail_gap(strength, tc) if (strength > 0 and tc.usable) else None}
        hard: Dict[str, float] = {}
        p, shown = (0.0, [])
        try:
            p, shown = contradiction_evidence(self.prior_sentences, answer, anchors, self.nli)
        except Exception:
            self.errors.append("nli: " + traceback.format_exc(limit=2))
        if p > 0.5:
            hard["nli_contradiction"] = min(1.0, (p - 0.5) / (C.HARD_CONTRADICTION_P - 0.5))
            tb.content_notes += [f"possible contradiction: {s}" for s in shown]
        qa, qr = quantity_conflict(" ".join(c.quote for c in self.claims) + " " + " ".join(self.prior_sentences), answer)
        if qa:
            hard["quantity_conflict"] = qa
            tb.content_notes += qr
        # --- profile-driven relevance (SBERT over the résumé chunks; informational, never fused) ---
        rel: Dict[str, Any] = {}
        if self.relevance is not None and answer.strip():
            try:
                rel = self.relevance.score(ctx.text, answer)
                if rel.get("unlisted_tech"):
                    tb.content_notes.append("technologies named that the résumé does not list: " + ", ".join(rel["unlisted_tech"][:5]))
            except Exception:
                self.errors.append("relevance: " + traceback.format_exc(limit=2))
        # --- voice ---
        vc: Optional[VoiceCues] = None
        if sig.audio is not None and sig.capture_t0 is not None:
            try:
                vc = voice_cues(sig.audio, sig.sr, sig.capture_t0, sig.tts_end, n_words=tc.words, segments=sig.speech_segments)
                cues.update(vc.fusion_cues())
            except Exception:
                self.errors.append("voice: " + traceback.format_exc(limit=2))
        if sig.ser_results:
            cues.update(ser_to_cues(sig.ser_results))
        # --- vision ---
        vx: Optional[VisionExtras] = None
        vrec = sig.vision_record or {}
        vq = vrec.get("quality", {})
        v_ok = bool(vrec) and vq.get("valid_ratio", 0) >= 0.6
        if v_ok:
            f = vrec.get("features", {})
            for name in ("gaze_shift_rate_per_min", "self_adaptor_rate_per_min", "face_activity", "blink_rate_per_min"):
                cues[name] = f.get(name)
        if sig.vision_samples:
            try:
                vx = vision_extras(sig.vision_samples)
            except Exception:
                self.errors.append("vision_extras: " + traceback.format_exc(limit=2))
        # --- coaching + integrity (never fused) ---
        tb.coaching = {"question_relevance": rel.get("question_relevance"), "resume_support": rel.get("resume_support"), "relevance_method": rel.get("method"),
                       "filler_rate_per_100w": tc.filler_rate_per_100w, "hedge_rate_per_100w": tc.hedge_rate_per_100w, "wpm": vc.wpm if vc else None,
                       "first_person_singular_share": tc.first_person_singular_share, "eye_contact_ratio": vx.eye_contact_ratio if vx else None,
                       "head_nod_rate_per_min": vx.head_nod_rate_per_min if vx else None, "detail_count": tc.detail_count}
        integ: Dict[str, Any] = {"flags": []}
        if sig.multi_face:
            integ["flags"].append("more than one face seen")
        if sig.phone_seen:
            integ["flags"].append("phone-like object seen")
        if vx and vx.usable:
            integ["reading_score"], integ["offscreen_fixation_s"] = vx.reading_score, vx.offscreen_fixation_s
            if vx.reading_score is not None and vx.reading_score >= C.READING_SCORE_FLAG:
                integ["flags"].append("gaze looks like reading from a screen (experimental)")
            if vx.offscreen_fixation_s >= C.OFFSCREEN_FIXATION_FLAG_S:
                integ["flags"].append(f"gaze held off-screen for {vx.offscreen_fixation_s:.1f}s (experimental)")
        if vc and vc.second_voice_drift is not None:
            integ["second_voice_drift"] = vc.second_voice_drift
            if vc.second_voice_drift >= C.SECOND_VOICE_DRIFT_FLAG:
                integ["flags"].append("voice character changed within the answer (experimental, uncalibrated)")
        if vc and vc.barge_in:
            integ["barge_in"] = True
        # --- time-aligned assistance-pattern detector (gaze x voice x content inside the answer) ---
        if self.monitor is not None:
            try:
                an = self.monitor.analyze_answer(turn, bucket, ctx.text, answer, audio=sig.audio, sr=sig.sr, capture_t0=sig.capture_t0, vision_samples=sig.vision_samples,
                                                 vision_offset=sig.vision_offset, words=sig.words, segments=sig.segments, ser_chunks=sig.ser_chunks,
                                                 speech=sig.speech_segments, target_keys=list(ctx.targets))
                integ["assist_pattern"] = {"analysed": an.analysed, "reason": an.reason, "reference": an.reference, "severity": an.severity, "max_score": round(an.max_score, 3),
                                           "events": [e.to_dict() for e in an.events], "text_jumps": an.text_jumps}
                for e in an.events:
                    integ["flags"].append(f"possible external-assistance pattern ({'+'.join(e.channels)}) at +{e.t0:.0f}s..+{e.t1:.0f}s of this answer"
                                          f"{'' if e.interview_t0 is None else f' (interview {int(e.interview_t0 // 60):02d}:{e.interview_t0 % 60:04.1f})'}: needs verification")
            except Exception:
                self.errors.append("assist_pattern: " + traceback.format_exc(limit=3))
        if sig.corrections:
            integ["transcript_corrections"] = sig.corrections
        tb.integrity = integ
        tb.quality = {"voice_usable": bool(vc and vc.usable), "voice_reasons": vc.quality_reasons if vc else ["no audio"], "vision_usable": v_ok or bool(vx and vx.usable),
                      "text_usable": tc.usable}
        tb.cues = cues
        # --- z-scores against the candidate's own baseline (cues without any baseline yet are held back, except content cues) ---
        usable_cues = {k: v for k, v in cues.items() if v is not None and (self.bank.n_baseline(k) >= 1 or C.CUES.get(k, {}).get("group") == "content")}
        tb.z = self.bank.zscores(usable_cues, bucket)
        tb.hard = hard
        modalities = {"text": tc.usable, "voice": bool(vc and vc.usable), "vision": v_ok}
        tb.fusion = self.model.score(tb.z, hard, modalities, seg_id)
        if self.bank.n_baseline("detail_density") == 0 and not any(self.bank.n_baseline(k) for k in cues):
            tb.fusion.abstain_reasons.append("no personal baseline yet (first answer)")
        # --- baseline update: the introduction always; later answers only if they did not look unsupported ---
        if bucket == "intro" or (tb.fusion.flag == "none" and not hard):
            self.bank.update(cues, bucket)
            tb.used_as_baseline = True
        # remember what was said for later consistency checks
        self.prior_sentences += [s for s in sentences(answer) if len(s.split()) >= 5][:6]
        self._close_probe(tb, it, tc)
        if self.cuelog is not None and getattr(self.cuelog, "enabled", False):
            try:
                self._log_answer_cues(tb, tc, vc, vx, vrec if v_ok else {}, rel, sig)
            except Exception:
                self.errors.append("cuelog: " + traceback.format_exc(limit=2))
        self.turns.append(tb)
        return tb

    def _log_answer_cues(self, tb: "TurnBehavior", tc, vc, vx, vrec: dict, rel: dict, sig: "Signals") -> None:
        cl = self.cuelog
        a0 = cl.rel(sig.capture_t0) if sig.capture_t0 is not None else None
        f = (vrec or {}).get("features", {})
        cl.write("vision", tb.turn, a0, 0.0, vx.duration_s if vx else None, scope="answer", eye_contact=vx.eye_contact_ratio if vx else None,
                 reading_score=vx.reading_score if vx else None, offscreen_fixation_s=vx.offscreen_fixation_s if vx else None, gaze_shift_rate=f.get("gaze_shift_rate_per_min"),
                 blink_rate=f.get("blink_rate_per_min"), self_adaptor_rate=f.get("self_adaptor_rate_per_min"), face_activity=f.get("face_activity"),
                 head_nod_rate=vx.head_nod_rate_per_min if vx else None, usable=tb.quality.get("vision_usable") if tb.quality else None)
        cl.write("voice", tb.turn, a0, 0.0, vc.window_s if vc else None, scope="answer", latency_s=vc.response_latency_s if vc else None, wpm=vc.wpm if vc else None,
                 articulation_wps=vc.articulation_rate_wps if vc else None, pause_rate=vc.pause_rate_per_min if vc else None, long_pauses=vc.long_pause_count if vc else None,
                 f0_mean_st=vc.f0_mean_st if vc else None, f0_sd_st=vc.f0_sd_st if vc else None, second_voice_drift=vc.second_voice_drift if vc else None,
                 ser_arousal=tb.cues.get("ser_arousal"), ser_negative=tb.cues.get("ser_negative_mass"), usable=vc.usable if vc else None)
        cl.write("text", tb.turn, a0, 0.0, None, scope="answer", words=tc.words, detail_density=tc.detail_density, claim_detail_gap=tb.cues.get("claim_detail_gap"),
                 filler_per_100w=tc.filler_rate_per_100w, hedge_per_100w=tc.hedge_rate_per_100w, question_relevance=rel.get("question_relevance"),
                 resume_support=rel.get("resume_support"), unlisted_tech=rel.get("unlisted_tech"), corrections=len(sig.corrections or []))

    # ---------- probe outcome ----------
    def _close_probe(self, tb: TurnBehavior, it, tc: TextCues):
        pp = self.pending_probe
        if not pp or pp["turn"] != tb.turn:
            return
        z_detail = (tb.z or {}).get("detail_density")
        if tb.hard.get("nli_contradiction", 0) >= 0.99 or tb.hard.get("quantity_conflict", 0) >= 0.5:      # NLI must reach the hard threshold, not just 'possible'
            status = "inconsistent"
        elif tc.usable and tc.words >= C.MIN_WORDS_PROBE_ANSWER and (z_detail is None or z_detail >= C.PROBE_VERIFIED_DETAIL_Z):
            status = "verified"
        else:
            status = "unresolved"
        tb.probe_outcome = {"topic": pp["topic"], "type": pp["type"], "status": status, "detail_z": z_detail, "words": tc.words}
        self.probe_log.append(tb.probe_outcome)
        self.pending_probe = None

    # ---------- probe request ----------
    def want_probe(self, tb: TurnBehavior, ctx, it) -> Optional[dict]:
        assist = (tb.integrity or {}).get("assist_pattern") or {}
        if tb.fusion is None or (tb.fusion.flag != "verify" and assist.get("severity") != "strong"):
            return None
        if self.probes_used >= C.MAX_PROBES_PER_INTERVIEW or ctx.turn < C.MIN_TURN_FOR_PROBE or it.words < C.MIN_WORDS_FOR_PROBE:
            return None
        if it.intent in NO_PROBE_INTENTS or ctx.kind in ("intro", "closing", "clarify", "contradiction"):
            return None
        if self.pending_probe is not None:
            return None
        key = (ctx.targets[0] if ctx.targets else ctx.anchor_project or ctx.kind)
        if key in self.probed_topics:
            return None
        topic = K.name_of(ctx.targets[0]) if ctx.targets else (ctx.anchor_project or "that work")
        if assist.get("severity") == "strong":
            ptype = "own_words"                         # a strong time-aligned pattern: check understanding, do not accuse
        elif tb.hard.get("quantity_conflict", 0) >= 0.5:
            ptype = "duration_check"
        elif tb.z.get("claim_detail_gap", 0) and tb.z["claim_detail_gap"] > 0.5 or (tb.z.get("detail_density") or 0) < -0.5:
            ptype = "specifics"
        elif tb.hard:
            ptype = "ownership"
        else:
            ptype = "sideways"
        reasons = (tb.fusion.explanation[:3] if tb.fusion else []) + tb.content_notes[:2] + ([f"assist pattern: {assist['events'][0]['evidence'][0]}"] if assist.get("events") else [])
        req = {"type": ptype, "topic": topic, "key": key, "text": PROBES[ptype].format(topic=topic), "reasons": reasons}
        tb.probe_requested = req
        return req

    def commit_probe(self, req: dict, turn: int):
        self.probes_used += 1
        self.probed_topics.add(req["key"])
        self.pending_probe = {"turn": turn, "topic": req["topic"], "type": req["type"]}


# ───────────────────────── the wrapper engine ─────────────────────────
class BehaviorAwareEngine(InterviewEngine):
    def __init__(self, resume, job, llm=None, target_minutes: float = None, max_questions: int = None, *, session: Optional[BehaviorSession] = None,
                 resume_text: str = "", nuextract=None, use_llm_claims: bool = False, nli=None, relevance=None, monitor=None, cuelog=None):
        import v58_config as _cfg
        super().__init__(resume, job, llm, _cfg.TARGET_MINUTES if target_minutes is None else target_minutes, _cfg.MAX_QUESTIONS if max_questions is None else max_questions)
        topics = [K.name_of(k) for k in resume.skill_keys()] if hasattr(resume, "skill_keys") else None
        self.behavior = session or BehaviorSession(resume_text or getattr(resume, "raw_text", ""), topics, llm=llm if use_llm_claims else None, nuextract=nuextract,
                                                   nli=nli, relevance=relevance, monitor=monitor, cuelog=cuelog)
        self._signals: Optional[Signals] = None
        self._last_tb: Optional[TurnBehavior] = None
        orig_next = self.planner.next

        def next_with_behaviour(ctx, it):
            tb = None
            try:
                tb = self.behavior.analyze(self.planner.turn, ctx, it, it.text or "", self._signals)
                self._last_tb = tb
                req = self.behavior.want_probe(tb, ctx, it)
                if req:
                    d = self._probe_decision(req, ctx, it)
                    if d is not None:
                        self.behavior.commit_probe(req, self.planner.turn + 1)    # outcome is recorded on the NEXT answer
                        return self.planner._commit(d)
                    tb.probe_requested = {**req, "blocked": "question validation failed"}
            except Exception:
                self.behavior.errors.append("bridge: " + traceback.format_exc(limit=3))
            return orig_next(ctx, it)

        self.planner.next = next_with_behaviour

    def _probe_decision(self, req: dict, ctx, it) -> Optional[Decision]:
        spec = Spec("followup", targets=list(ctx.targets), project=ctx.anchor_project, followup="evidence_probe", aspect=ctx.aspect, level=ctx.level,
                    reason=f"verification probe ({req['type']}): behaviour/content evidence suggested checking this claim", new_evidence="concrete, checkable detail about the claim",
                    why_insufficient="; ".join(req["reasons"])[:200] or "claim needs checkable detail", extra={"probe_type": req["type"]})
        text = req["text"]
        ok, why = self.planner.qe.validate(text, spec, self.planner.asked, it.text, self.planner.asked_specs)
        if not ok:
            return None
        return Decision(spec, text, True, self.planner.phase, source="behavior_probe")

    def step(self, answer: str, signals: Optional[Signals] = None):
        self._signals, self._last_tb = signals, None
        out = super().step(answer)
        out.behavior = self._last_tb
        try:                                            # attach to the entry that was just answered (latest entry with an answer and no behaviour yet)
            if self._last_tb is not None:
                for te in reversed(self.transcript):
                    interp = getattr(te, "interpretation", None)
                    if getattr(te, "answer", "") and isinstance(interp, dict) and "behavior" not in interp:
                        interp["behavior"] = self._last_tb.to_report()
                        break
        except Exception:
            self.behavior.errors.append("attach: " + traceback.format_exc(limit=2))
        return out


# ───────────────────────── report ─────────────────────────
DISCLAIMER = ("Behavioural cues are weak, person-dependent signals (Luke et al. 2023; Caso et al. 2023). 'needs verification' means a claim lacked checkable detail or "
              "conflicted with earlier statements; it does not mean the candidate lied. Voice/face evidence alone never raises a flag. Integrity items are experimental.")


def behavior_report(engine: BehaviorAwareEngine) -> dict:
    s = engine.behavior
    turns = [t.to_report() for t in s.turns]
    claims_status = []
    for c in s.claims:
        st = "not_checked"
        for pl in s.probe_log:
            if pl["topic"].lower() in c.topic.lower() or c.topic.lower() in pl["topic"].lower():
                st = pl["status"]
        claims_status.append({"claim": c.quote, "topic": c.topic, "status": st})
    return {"schema": "smarthire-behavior/1", "disclaimer": DISCLAIMER, "model": {"group_effect_sizes": s.model.d, "calibrated": s.model.platt is not None,
            "fitted_n": s.model.fitted_n}, "turns": turns, "probes": s.probe_log, "resume_claims": claims_status, "errors": s.errors,
            "assist_pattern": s.monitor.report() if s.monitor is not None else None, "cue_log": s.cuelog.summary() if s.cuelog is not None else None,
            "summary": {"answers": len(turns), "flagged": sum(1 for t in s.turns if t.fusion and t.fusion.flag == "verify"), "probes": s.probes_used,
                        "verified": sum(1 for p in s.probe_log if p["status"] == "verified"), "unresolved": sum(1 for p in s.probe_log if p["status"] == "unresolved"),
                        "inconsistent": sum(1 for p in s.probe_log if p["status"] == "inconsistent")}}
