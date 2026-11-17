"""
test_v58.py — automated validation for SmartHire V58.   Run:  python -m unittest test_v58 -v     (or: python test_v58.py)

Covers: résumé extraction (the real V51 failure résumé + other layouts), JD normalisation, answer understanding -> evidence state,
question planning (dedup / coverage / follow-up / move-on), reactions, question validation + LLM fallback, the strict-JSON LLM client
(against a fake llama-server), and an end-to-end simulated interview.
"""
import json
import os
import re
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.paths import bootstrap  # noqa: E402
bootstrap()
sys.path.insert(0, str(HERE))

import answer_understanding_v58 as A       # noqa: E402
import concepts_v58 as K                   # noqa: E402
import doc_reader_v58 as D                 # noqa: E402
import evidence_engine_v58 as E            # noqa: E402
import interview_engine_v58 as IE          # noqa: E402
import jd_extractor_v58 as J               # noqa: E402
import llm_engine_v58 as L                 # noqa: E402
import resume_extractor_v58 as R           # noqa: E402
import sim_candidate_v58 as SIM            # noqa: E402
from models_v58 import *                   # noqa: E402,F401,F403
from question_engine_v58 import Spec       # noqa: E402
from reaction_engine_v58 import BANNED_FILLER   # noqa: E402
from textkit_v58 import content_tokens, cosine_sets   # noqa: E402

FIX = HERE / "fixtures"
RESUME_TXT = (FIX / "faaiz_resume_layout.txt").read_text(encoding="utf-8")
JD_TXT = (FIX / "contour_jd.txt").read_text(encoding="utf-8")
GARBAGE = ("english", "interests", "extracurricular", "opencv", "languages", "education", "contact", "skills")


def resume():
    return R.extract_resume(RESUME_TXT, is_text=True, filename="Faaiz_Resume.pdf")


def job():
    return J.extract_jd(JD_TXT, True, "Contour Software", "Trainee Software Developer")


def fresh():
    r, j = resume(), job()
    return r, j, E.EvidenceGraph(r, j)


def judge(text, ctx, graph=None, r=None):
    r = r or resume()
    return A.understand(text, ctx, r, graph)


# ───────────────────────────────────────────────────────────────────────────────────────────────
class TestResumeExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = resume()

    def test_projects_are_real_projects(self):
        names = [p.short_name for p in self.r.projects]
        self.assertEqual(set(names), {"Sign Language Detection", "Cat vs. Dog Classifier", "Fraday", "Minecraft Clone & Flappy Bird Clone"}, names)

    def test_no_garbage_projects(self):
        for p in self.r.projects:
            low = (p.short_name + " " + p.title).lower()
            for g in GARBAGE:
                self.assertNotEqual(p.short_name.lower(), g)
            self.assertNotIn("i n t e r", low)
            self.assertNotIn("a c t i v", low)
            self.assertGreaterEqual(len(p.bullets) + len(p.description.split()) // 8, 1)

    def test_no_email_or_phone_contamination(self):
        blob = json.dumps(self.r.to_dict(), default=str)
        for p in self.r.projects:
            for field in (p.title, p.description, " ".join(p.bullets)):
                self.assertNotRegex(field, r"@|\d{4}-\d{7}|github\.com|linkedin\.com", field)
        self.assertTrue(any("@" in e for e in self.r.contact.emails))
        self.assertTrue(self.r.contact.phones)
        self.assertTrue(any("Karachi" in l for l in self.r.contact.locations))

    def test_sections_recognised_including_letter_spaced(self):
        for s in ("summary", "skills", "education", "projects", "certifications", "extracurricular", "languages", "interests"):
            self.assertIn(s, self.r.sections_found)

    def test_skills_categorised(self):
        by = {c: {s.key for s in lst} for c, lst in self.r.skills.items()}
        self.assertIn("python", by["programming_languages"])
        self.assertIn("csharp", by["programming_languages"])
        self.assertIn("pandas", by["libraries"])
        self.assertIn("sklearn", by["libraries"])
        self.assertIn("yolo", by["libraries"])
        self.assertTrue({"sql"} & by["databases"])
        self.assertTrue(len(self.r.all_skills()) >= 20)
        names = {s.name.lower() for s in self.r.all_skills()}
        self.assertNotIn("english", names)
        self.assertNotIn("urdu", names)

    def test_spoken_languages_separate(self):
        langs = {x["language"] for x in self.r.spoken_languages}
        self.assertTrue({"English", "Urdu"} <= langs)

    def test_education(self):
        self.assertEqual(len(self.r.education), 2)
        e0 = self.r.education[0]
        self.assertEqual(e0.degree, "B.S.")
        self.assertIn("Artificial Intelligence", e0.field_of_study)
        self.assertEqual(e0.gpa, "3.42")
        self.assertEqual(e0.status, "in_progress")
        self.assertNotIn("(", e0.institution)

    def test_name_and_contact(self):
        self.assertEqual(self.r.first_name, "Faaiz")
        self.assertIn("Ahmed", self.r.name)

    def test_project_structure(self):
        sl = self.r.project_by_name("Sign Language Detection")
        self.assertEqual(sl.subtitle, "YOLOv8")
        self.assertIn("yolo", sl.tech_keys())
        self.assertEqual(sl.ownership, "individual")
        self.assertIn("cv", sl.implied)
        self.assertTrue(sl.evidence.text)
        fr = self.r.project_by_name("Fraday")
        self.assertIn("llm_api", fr.tech_keys())

    def test_certifications_and_extracurricular(self):
        self.assertGreaterEqual(len(self.r.certifications), 3)
        self.assertTrue(any(c.kind == "virtual_internship" for c in self.r.certifications))
        self.assertEqual(self.r.experience, [])
        self.assertTrue(self.r.extracurricular)

    def test_real_pdf_geometry_path(self):
        pdf = FIX / "real_layout_render.pdf"
        if not pdf.exists():
            self.skipTest("fixture PDF missing")
        try:
            r = R.extract_resume(str(pdf))
        except Exception as e:                                           # pragma: no cover
            self.skipTest(f"no PDF backend: {e}")
        self.assertEqual({p.short_name for p in r.projects}, {p.short_name for p in self.r.projects})
        self.assertEqual(r.layout, "two-column")

    def test_other_layouts_generalise(self):
        # single column, unusual heading words, inline skills, no 'Projects' word
        t1 = """Ayesha Raza
ayesha@mail.com | +92 300 1234567 | Lahore, Pakistan

WHAT I'VE BUILT
Hospital Queue Manager (2024)
- Built a Flask + PostgreSQL web app that lets patients take a token online
- Wrote the REST endpoints and deployed it on Render
Expense Tracker - Android
- Developed an Android app in Kotlin with Room database; a team of 3, I did the UI

Tech: Python, Flask, PostgreSQL, Kotlin, Git
EDUCATION
BS Software Engineering, NUST, 2021 - 2025, CGPA 3.6

HOBBIES
Cricket, Reading
LANGUAGES
English (fluent), Urdu (native)
"""
        r = R.extract_resume(t1, is_text=True)
        self.assertEqual({p.short_name for p in r.projects}, {"Hospital Queue Manager", "Expense Tracker"}, [p.short_name for p in r.projects])
        self.assertIn("flask", {s.key for s in r.all_skills()})
        self.assertNotIn("english", {p.short_name.lower() for p in r.projects})
        self.assertEqual(len(r.education), 1)
        # all-caps letter-spaced headings, experience section, no skills heading
        t2 = """B I L A L  K H A N
bilal.khan@gmail.com
W O R K  E X P E R I E N C E
Software Intern - Systems Ltd   Jun 2024 - Aug 2024
• Developed REST APIs in ASP.NET Core with SQL Server for an invoicing module
• Wrote unit tests with xUnit
P R O J E C T S
Chat Server | Java
• Implemented a multi-threaded TCP chat server in Java
E D U C A T I O N
BSCS - FAST NUCES 2020 - 2024
"""
        r2 = R.extract_resume(t2, is_text=True)
        self.assertEqual(len(r2.experience), 1, r2.experience)
        self.assertEqual(r2.experience[0].kind, "internship")
        self.assertIn("aspnet", {t.key for t in r2.experience[0].technologies})
        self.assertEqual([p.short_name for p in r2.projects], ["Chat Server"])

    def test_v51_fixtures_have_no_garbage(self):
        p = FIX / "v51"
        if not p.exists():
            self.skipTest("v51 fixtures not bundled")
        for f in sorted(p.glob("fixture_resume*.txt")):
            r = R.extract_resume(f.read_text(encoding="utf-8"), is_text=True, filename=str(f))
            for pr in r.projects:
                self.assertNotIn(pr.short_name.lower(), GARBAGE + ("python",), f"{f.name}: {pr.short_name}")
            self.assertGreaterEqual(len(r.projects), 1, f.name)

    def test_technology_names_never_become_projects(self):
        t = """Sam Lee
sam@x.com
SKILLS
Python, OpenCV, English
INTERESTS
OpenCV
English
PROJECTS
OpenCV
English
Weather App
- Built a weather dashboard using Python and OpenWeather API
"""
        r = R.extract_resume(t, is_text=True)
        self.assertEqual([p.short_name for p in r.projects], ["Weather App"])


class TestDocReader(unittest.TestCase):
    def test_glued_contact_split_out(self):
        rd = D.read_document("Sign Language Detection real-time0320-2797793 more text\nlivegithub.com/faaiz-ahmed and Karachi, Pakistan", is_text=True)
        txt = " ".join(l.text for l in rd.lines)
        self.assertNotIn("0320", txt)
        self.assertNotIn("github", txt)
        self.assertIn("0320-2797793", rd.contact.phones)

    def test_letter_spaced_heading_collapsed(self):
        rd = D.read_document("N A M E\nP R O J E C T S\nI N T E RE S T S\nline", is_text=True)
        texts = [l.text for l in rd.lines]
        self.assertIn("PROJECTS", texts)
        self.assertIn("INTERESTS", texts)


# ───────────────────────────────────────────────────────────────────────────────────────────────
class TestJDExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.j = job()

    def test_requirements_normalised(self):
        req = self.j.requirements
        for k in ("python", "rag", "prompt_eng", "genai", "llm", "aspnet", "angular", "sqlserver", "csharp", "rest_api", "oop", "sdlc", "architecture", "se_practices", "ml_libs"):
            self.assertIn(k, req, k)
        self.assertEqual(req["python"].importance, 3)
        self.assertEqual(req["python"].expected_depth, "depth")
        self.assertEqual(req["rag"].canonical_name, "RAG")
        self.assertIn("chunking", req["rag"].evidence_expected)
        self.assertIn("retrieval", req["rag"].evidence_expected)
        self.assertEqual(req["rag"].category, "AI_ML")
        self.assertEqual(req["rag"].expected_depth, "awareness")
        self.assertTrue(req["rag"].source_text)
        self.assertFalse(req["ai_projects"].required)

    def test_company_boilerplate_is_not_a_requirement(self):
        self.assertNotIn("erp", self.j.requirements)
        self.assertNotIn("accounting", self.j.requirements)
        self.assertIn("Accounting", self.j.company_context)

    def test_behavioural_and_title(self):
        self.assertEqual(self.j.title, "Trainee Software Developer")
        self.assertEqual(self.j.seniority, "junior")
        for b in ("teamwork", "communication", "problem_solving", "learning_ability"):
            self.assertIn(b, self.j.behavioural)


# ───────────────────────────────────────────────────────────────────────────────────────────────
class TestAnswerUnderstanding(unittest.TestCase):
    def setUp(self):
        self.r, self.j, self.g = fresh()

    def stance(self, text, ctx, key):
        it = judge(text, ctx, self.g, self.r)
        cl = it.claim_for(key)
        return it, (cl.stance if cl else None)

    def test_dont_know(self):
        it, _ = self.stance("I don't know", QuestionContext("knowledge", ["rag"]), "rag")
        self.assertEqual(it.intent, "explicit_unknown")
        self.assertEqual(it.quality, "none")
        ctx = QuestionContext("knowledge", ["rag"])
        self.g.apply(ctx, it, 1)
        self.assertTrue(self.g.get("rag").said_unknown)
        self.assertNotEqual(self.g.get("rag").status, DEMONSTRATED)

    def test_havent_used_rag(self):
        ctx = QuestionContext("knowledge", ["rag"])
        it, st = self.stance("I haven't used RAG", ctx, "rag")
        self.assertEqual(st, "no_experience")
        self.g.apply(ctx, it, 1)
        self.assertEqual(self.g.get("rag").status, EXPLICIT_NO_EXPERIENCE)

    def test_learned_rag(self):
        ctx = QuestionContext("knowledge", ["rag"])
        it, st = self.stance("I learned RAG but haven't implemented it", ctx, "rag")
        self.assertEqual(st, "learning")
        self.g.apply(ctx, it, 1)
        self.assertEqual(self.g.get("rag").status, LEARNING_ONLY)

    def test_implemented_rag(self):
        ctx = QuestionContext("knowledge", ["rag"])
        it, st = self.stance("I used it in my project, I created the retrieval pipeline and connected the vector database. I split the pdf into chunks and used embeddings.", ctx, "rag")
        self.assertEqual(st, "practical")
        self.g.apply(ctx, it, 1)
        self.assertIn(self.g.get("rag").status, (PRACTICAL, DEMONSTRATED))
        self.assertGreaterEqual(len(self.g.get("rag").facets_covered), 2)
        self.assertEqual(self.g.get("rag").status, DEMONSTRATED)

    def test_rag_design_after_explicit_nonimplementation_is_theoretical(self):
        ctx = QuestionContext("knowledge", ["rag"])
        answer = "I understand the concept of RAG, although I haven't implemented a complete RAG pipeline independently yet. The process starts by loading documents, extracting their text, and splitting them into smaller chunks. Then we generate embeddings and store them in a vector database. When a user asks a question, we convert it into an embedding, retrieve the most relevant chunks, and pass those chunks along with the question to an LLM. The model then generates an answer grounded in the retrieved information. I would also include source references to make the answer easier to verify."
        it = judge(answer, ctx, self.g, self.r)
        claim = it.claim_for("rag")
        self.assertTrue(claim and claim.stance == "theoretical", [(c.key, c.stance) for c in it.claims])
        self.g.apply(ctx, it, 1)
        self.assertNotEqual(self.g.get("rag").status, DEMONSTRATED)
        self.assertFalse(self.g.get("rag").candidate_demonstration)

    def test_used_python(self):
        ctx = QuestionContext("project_probe", ["python"], "Sign Language Detection")
        it, st = self.stance("I used Python for my sign language detection project. I trained the YOLO model and wrote the real-time detection code.", ctx, "python")
        self.assertEqual(st, "practical")
        self.assertEqual(it.ownership, "strong")
        self.assertIn("Sign Language Detection", it.projects_mentioned)
        self.assertEqual(it.claim_for("yolo").stance, "practical")
        cv = it.claim_for("cv")
        self.assertTrue(cv and cv.implied and cv.stance == "practical")
        self.assertIn(it.quality, ("good", "strong"))

    def test_dont_build_opencv_changes_nothing_else(self):
        ctx = QuestionContext("project_probe", ["opencv"], "OpenCV")
        before = {k: (c.status, c.confidence) for k, c in self.g.comp.items()}
        for text in ("I don't build OpenCV", "OpenCV is a prebuilt library i cant built it", "I dont built it"):
            it = judge(text, ctx, self.g, self.r)
            self.g.apply(ctx, it, 1)
            for k in ("python", "ml", "ml_libs", "cv"):
                c = self.g.get(k)
                if c:
                    self.assertNotEqual(c.status, EXPLICIT_NO_EXPERIENCE, (text, k))
                    self.assertEqual(c.status, before[k][0], (text, k))
            self.assertNotEqual(self.g.get("opencv").status if self.g.get("opencv") else UNKNOWN, EXPLICIT_NO_EXPERIENCE, text)
            self.assertTrue(it.anchor_is_library, text)
            self.assertEqual(it.contradictions, [])

    def test_broken_english_dataset_answer_is_understood(self):
        ctx = QuestionContext("project_probe", [], "Sign Language Detection", "dataset", text="How did you collect the dataset?")
        it = judge("I collect it myself like i download the dataset of some images and if a class which i want i cant find then i take my image of making sign gestures", ctx, self.g, self.r)
        ph = [f["phrase"] for f in it.facts if f["aspect"] == "dataset"]
        self.assertTrue(ph and "downloaded" in ph[0] and "captured your own" in ph[0], ph)
        self.assertEqual(it.ownership, "strong")
        self.assertIn(it.quality, ("partial", "good", "strong"))
        self.assertTrue(it.question_answered)
        self.assertFalse(any(c.stance == "no_experience" for c in it.claims))      # 'cant find' is not 'no experience'
        self.g.apply(ctx, it, 1)
        for k in ("python", "ml"):
            c = self.g.get(k)
            if c:
                self.assertNotEqual(c.status, EXPLICIT_NO_EXPERIENCE)

    def test_commands(self):
        for text, intent in (("skip", "skip"), ("next question", "skip"), ("quit", "quit"), ("exit", "quit"), ("help", "help"), ("repeat", "repeat"), ("not sure", "explicit_unknown"),
                             ("no idea", "explicit_unknown"), ("I have no experience", "no_experience"), ("When did i say i have not used it", "dispute"),
                             ("what do you mean?", "clarify_request"), ("I don't know much about it", "explicit_unknown")):
            it = judge(text, QuestionContext("knowledge", ["rag"]), self.g, self.r)
            self.assertEqual(it.intent, intent, text)
            self.assertEqual(it.claims, [], text)

    def test_hypothetical_design_answer_is_theoretical_not_practical(self):
        ctx = QuestionContext("scenario", ["rag"], scenario=True)
        it = judge("I think we put the documents in a vector database, then search the similar parts for the question and give these parts to the LLM in the prompt so it answer from them.", ctx, self.g, self.r)
        st = it.claim_for("rag")
        self.assertTrue(st and st.stance == "theoretical", [(c.key, c.stance) for c in it.claims])
        self.g.apply(ctx, it, 1)
        self.assertIn(self.g.get("rag").status, (THEORETICAL, PARTIALLY_VALIDATED))
        self.assertFalse(self.g.get("rag").candidate_demonstration)

    def test_stance_is_not_grammar_dependent(self):
        ctx = QuestionContext("project_probe", ["pandas"], "Fraday")
        it = judge("I use pandas for clean the data, i remove missing value and join two file", ctx, self.g, self.r)
        self.assertEqual(it.claim_for("pandas").stance, "practical")

    def test_negation_scope_is_per_clause(self):
        ctx = QuestionContext("cluster_gate", ["csharp", "aspnet", "angular", "sqlserver"])
        it = judge("I know basic C# but not ASP.NET or Angular. I used SQL in university.", ctx, self.g, self.r)
        self.assertEqual(it.claim_for("csharp").stance, "claim")
        self.assertEqual(it.claim_for("aspnet").stance, "no_experience")
        self.assertEqual(it.claim_for("angular").stance, "no_experience")
        self.assertEqual(it.claim_for("sql").stance, "practical")

    def test_hedging_flag_and_future_learning_is_not_learning(self):
        it = judge("I am not very sure, I think I need to learn more about this.", QuestionContext("knowledge", ["angular"]), self.g, self.r)
        self.assertEqual(it.intent, "explicit_unknown")
        it2 = judge("I need to learn more about RAG", QuestionContext("knowledge", ["rag"]), self.g, self.r)
        self.assertFalse(it2.claim_for("rag") and it2.claim_for("rag").stance == "learning")

    def test_implied_evidence_never_validates(self):
        ctx = QuestionContext("project_probe", ["yolo"], "Sign Language Detection")
        it = judge("I trained yolov8 myself on my own images for 50 epochs and checked mAP", ctx, self.g, self.r)
        self.g.apply(ctx, it, 1)
        c = self.g.get("ml_libs")
        if c:
            self.assertTrue(c.interview and not c.direct_items())
            self.assertFalse(self.g.validated(c))

    def test_llm_refinement_is_validated_and_add_only(self):
        def responder(purpose, user):
            return {"claims": [{"concept": "rag", "stance": "practical", "evidence": "completely invented sentence about databases"}, {"concept": "angular", "stance": "practical", "evidence": "small pages were made"}], "ownership": "strong"}
        stub = L.StubLLM(responder)
        ctx = QuestionContext("knowledge", ["rag", "angular"], text="Have you used these?")
        it = A.understand("honestly well small pages were made for the university once, nothing else really", ctx, self.r, self.g, stub, [2])
        keys = {c.key for c in it.claims}
        self.assertNotIn("rag", keys)               # evidence not found in the answer -> rejected
        self.assertIn("angular", keys)              # grounded evidence -> accepted
        self.assertEqual(it.claim_for("angular").stance, "theoretical")   # no first-person build verb with the claim -> downgraded from LLM's 'practical'


class TestEvidenceGraph(unittest.TestCase):
    def test_resume_and_blueprint_are_not_interview_evidence(self):
        r, j, g = fresh()
        cov = g.coverage()
        self.assertEqual((cov["addressed"], cov["resolved"], cov["validated"]), (0, 0, 0))
        self.assertEqual(cov["weighted_validated_pct"], 0)
        for k in g.required_keys():
            self.assertEqual(g.comp[k].status, UNKNOWN)
        self.assertIn(g.get("python").resume_level, ("used_in_project", "strong"))
        self.assertEqual(g.get("rag").resume_level, "none")

    def test_addressed_requires_a_question_and_validated_requires_evidence(self):
        r, j, g = fresh()
        ctx = QuestionContext("knowledge", ["rag"], level="basic", text="RAG?")
        g.note_question(ctx, 1)
        self.assertTrue(g.get("rag").addressed)
        self.assertFalse(g.validated(g.get("rag")))
        it = judge("yes I know RAG", ctx, g, r)
        g.apply(ctx, it, 1)
        self.assertEqual(g.get("rag").status, CLAIM_ONLY)
        self.assertFalse(g.validated(g.get("rag")))

    def test_real_contradiction_detected_only_when_real(self):
        r, j, g = fresh()
        ctx = QuestionContext("project_probe", ["python"], "Fraday")
        it = judge("I never used Python", ctx, g, r)
        self.assertIn("python", it.contradictions)                   # résumé shows Python in projects
        it2 = judge("I never used Angular", QuestionContext("knowledge", ["angular"]), g, r)
        self.assertEqual(it2.contradictions, [])                     # résumé does not claim Angular

    def test_conceptual_answer_validates_only_awareness_level_requirements(self):
        r, j, g = fresh()
        ctx = QuestionContext("scenario", ["rag"], scenario=True)
        it = judge("First I would split the documents into chunks, create embeddings, store them in a vector database and retrieve the top results to put in the prompt", ctx, g, r)
        g.apply(ctx, it, 1)
        c = g.get("rag")
        self.assertEqual(c.expected_depth, "awareness")
        self.assertEqual(g.validation_basis(c), "conceptual")          # JD asks only for exposure
        self.assertFalse(c.candidate_demonstration)
        sc = g.get("sqlserver")                                         # JD asks for working knowledge: conceptual answer is not enough
        sc.expected_depth = "implementation"
        ctx2 = QuestionContext("scenario", ["sqlserver"], scenario=True)
        it2 = judge("I would join the two tables and group by customer and sum the order value", ctx2, g, r)
        g.apply(ctx2, it2, 2)
        self.assertNotEqual(g.validation_basis(sc), "hands_on")
        self.assertNotEqual(g.validation_basis(sc), "conceptual")

    def test_skip_gives_no_evidence(self):
        r, j, g = fresh()
        ctx = QuestionContext("knowledge", ["rag"])
        g.apply(ctx, judge("skip", ctx, g, r), 1)
        self.assertEqual(g.get("rag").status, UNKNOWN)
        self.assertEqual(g.get("rag").interview, [])


# ───────────────────────────────────────────────────────────────────────────────────────────────
def run_sim(llm=None, answers=None, max_turns=40):
    r, j = resume(), job()
    e = IE.InterviewEngine(r, j, llm)
    o = e.start()
    n = 0
    while not o.finished and n < max_turns:
        a = answers(e, o) if answers else SIM.answer(e.ctx, o.text)
        o = e.step(a)
        n += 1
    return e


class TestPlanning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = run_sim()

    def test_finishes_in_budget_without_exceptions(self):
        self.assertTrue(self.e.finished)
        self.assertLessEqual(self.e.planner.turn, 20)
        self.assertGreaterEqual(self.e.planner.turn, 8)

    def test_no_duplicate_questions(self):
        qs = [t.question for t in self.e.transcript if t.question]
        self.assertEqual(len(qs), len(set(qs)))
        for i, a in enumerate(qs):
            for b in qs[i + 1:]:
                ta, tb = content_tokens(a) - self.e.qe.topic_vocab, content_tokens(b) - self.e.qe.topic_vocab
                if len(ta) >= 4 and len(tb) >= 4:
                    self.assertLess(cosine_sets(ta, tb), 0.9, (a, b))

    def test_every_followup_has_a_reason_and_new_evidence(self):
        for t in self.e.transcript:
            self.assertTrue(t.why["why_this_question"], t.question)
            self.assertTrue(t.why["what_new_evidence_it_seeks"], t.question)

    def test_important_unresolved_requirements_are_asked(self):
        g = self.e.graph
        asked = {k for k in g.required_keys() if g.comp[k].addressed}
        for k in ("rag", "prompt_eng", "rest_api", "oop", "sdlc", "architecture", "aspnet", "angular", "sqlserver", "csharp"):
            self.assertIn(k, asked, f"{k} never asked")

    def test_questions_are_real_technical_questions(self):
        qs = " ".join(t.question for t in self.e.transcript)
        self.assertRegex(qs, r"Retrieval-Augmented Generation")
        self.assertRegex(qs, r"retrieval")
        self.assertRegex(qs, r"REST API|HTTP|endpoint")
        self.assertRegex(qs, r"customers and orders|SQL")
        self.assertNotRegex(qs, r"(?i)tell me about AI\b")

    def test_no_english_opencv_interests_questions(self):
        for t in self.e.transcript:
            self.assertNotRegex(t.question, r"(?i)let's talk about (english|opencv|interests|extracurricular)", t.question)
            self.assertNotRegex(t.question, r"\w+@\w+\.\w+")
            self.assertNotRegex(t.question, r"I N T E|E X T R A")

    def test_scenario_fallback_after_no_experience(self):
        kinds = [(t.kind, t.targets) for t in self.e.transcript]
        self.assertIn(("scenario", ["rag"]), kinds)

    def test_one_followup_only_for_unknown(self):
        # after 'I don't know' the planner offers one simpler scenario, then moves on (no interrogation)
        for key in ("architecture",):
            n = sum(1 for t in self.e.transcript if t.targets == [key])
            self.assertLessEqual(n, 2)

    def test_followup_when_insufficient_and_move_on_when_sufficient(self):
        # vague answer to a project walkthrough => project follow-up on the SAME project
        r, j = resume(), job()
        e = IE.InterviewEngine(r, j)
        o = e.start()
        o = e.step("I am a student and I like software")
        pj = e.last_decision.spec.project
        self.assertTrue(pj)
        o = e.step("it is a good project")
        self.assertEqual(e.last_decision.spec.project, pj)
        self.assertTrue(e.last_decision.followup)
        # strong, complete answer on a low-importance topic => move on to a different topic
        e2 = IE.InterviewEngine(r, j)
        o = e2.start()
        o = e2.step("I am a student and I like software")
        o = e2.step(SIM.P["fraday"])
        o = e2.step(SIM.P["fraday"] + " I handled errors with try except and retry.")
        self.assertNotEqual((e2.last_decision.spec.project, e2.last_decision.spec.aspect), (pj, "ownership"))

    def test_clock_stays_in_target_window_or_candidate_is_brief(self):
        self.assertLessEqual(self.e.planner.clock, 12.5 * 60)

    def test_command_handling_in_flow(self):
        r, j = resume(), job()
        e = IE.InterviewEngine(r, j)
        e.start()
        o = e.step("I am a student")
        first_q = o.question
        o = e.step("repeat")
        self.assertIn(first_q.split("?")[0][:40], o.text)                  # same question again
        n = e.planner.turn
        o = e.step("skip")
        self.assertFalse(o.finished)
        o2 = e.step("quit")
        self.assertTrue(o2.finished)


class TestReactions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = run_sim()

    def test_no_bare_filler_and_not_repetitive(self):
        acks = [t.interviewer.replace(t.question, "").strip() for t in self.e.transcript[1:]]
        for a in acks:
            self.assertNotIn(a.lower().strip(), BANNED_FILLER)
        firsts = [re.split(r"(?<=[.!?])\s", a)[0] for a in acks if a]
        from collections import Counter
        top = Counter(firsts).most_common(1)[0][1]
        self.assertLessEqual(top, 3, Counter(firsts).most_common(3))

    def test_no_praise_for_vague_or_empty_answers(self):
        # the interviewer text of turn N+1 is the reaction to the answer of turn N
        for prev, t in zip(self.e.transcript, self.e.transcript[1:]):
            q = prev.interpretation.get("quality")
            if q in ("none", "vague"):
                self.assertNotRegex(t.interviewer.replace(t.question, ""), r"(?i)\b(great|excellent|impressive|perfect|clearly|well explained|useful)\b", t.interviewer)

    def test_reaction_references_what_was_said(self):
        r, j = resume(), job()
        e = IE.InterviewEngine(r, j)
        e.start()
        e.step("I am a student")
        o = e.step(SIM.P["fraday"])
        o = e.step(SIM.P["dataset"])      # answered in the context of the project follow-up
        txt = o.text
        self.assertTrue(re.search(r"(?i)(downloaded|captured|your own|you mentioned|Fraday|picture|concrete)", txt), txt)

    def test_noexp_reaction_names_the_concept_and_does_not_blame(self):
        r, j = resume(), job()
        e = IE.InterviewEngine(r, j)
        e.ctx = QuestionContext("knowledge", ["rag"], text="Have you worked with RAG?")
        e.transcript.append(IE.TranscriptEntry(1, "", "q", "technical", "knowledge", ["rag"], {}, "deterministic"))
        e.planner.turn = 1
        o = e.step("I haven't used RAG")
        self.assertIn("RAG", o.text)
        self.assertNotRegex(o.text, r"(?i)contradict|but your r[ée]sum[ée]")

    def test_dispute_is_acknowledged_without_recontradiction(self):
        r, j = resume(), job()
        e = IE.InterviewEngine(r, j)
        e.start()
        o = e.step("When did i say i have not used it")
        self.assertRegex(o.text, r"(?i)you're right|sorry|fair point")
        self.assertNotRegex(o.text, r"(?i)you said you haven't")


class TestQuestionValidation(unittest.TestCase):
    def setUp(self):
        self.r, self.j, self.g = fresh()
        self.qe = IE.QuestionEngine(self.r, self.j, self.g)

    def test_not_a_question_and_ungrounded_and_duplicate(self):
        sp = Spec("knowledge", ["rag"])
        self.assertEqual(self.qe.validate("RAG is a retrieval method for LLMs.", sp, [])[1], "not_a_question")
        ok, why = self.qe.validate("Have you built RAG pipelines with Kubernetes and Terraform for production at scale?", sp, [])
        self.assertFalse(ok)
        self.assertTrue(why.startswith("ungrounded_technology"), why)
        q = "Walk me through how you would design the retrieval part of a RAG system."
        self.assertTrue(self.qe.validate(q, sp, [])[0])
        self.assertEqual(self.qe.validate(q, sp, [q])[1], "duplicate_of_previous")
        self.assertEqual(self.qe.validate("Tell me about your favourite food in detail please today?", sp, [])[1], "irrelevant_to_target")

    def test_bad_llm_question_falls_back_to_deterministic(self):
        bad = L.StubLLM(lambda p, u: {"question": "Great job! Now tell me about Kubernetes."})
        r, j = resume(), job()
        e = IE.InterviewEngine(r, j, bad)
        o = e.start()
        for _ in range(12):
            o = e.step(SIM.answer(e.ctx, o.text))
            if o.finished:
                break
        for t in e.transcript:
            self.assertNotIn("Kubernetes", t.question)
        self.assertTrue(all(t.source != "llm" for t in e.transcript))

    def test_good_llm_question_is_used_only_when_valid(self):
        good = L.StubLLM(lambda p, u: {"question": "How did you decide which images were good enough to keep in the training set for Sign Language Detection?"} if p == "followup" else None)
        sp = Spec("project_probe", [], project="Sign Language Detection", followup="implementation", hook_quote="I download images and take some myself", aspect="labeling")
        qe = IE.QuestionEngine(self.r, self.j, self.g, good)
        alt = qe.llm_followup(sp, type("I", (), {"text": "I download images and take some myself"})(), "")
        self.assertIsNotNone(alt)
        self.assertTrue(qe.validate(alt, sp, [])[0])


# ───────────────────────────────────────────────────────────────────────────────────────────────
class _FakeLlama(BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if _FakeLlama.mode == "no_schema" and "json_schema" in body:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"json_schema not supported"}'); return
        out = {"ok": '{"question": "What did you do?"}', "no_schema": '{"question": "What did you do?"}', "fenced": '```json\n{"question": "What did you do?"}\n```',
               "garbage": "I think the answer is: sure!", "violation": '{"q": 1}'}[_FakeLlama.mode]
        self.send_response(200); self.end_headers(); self.wfile.write(json.dumps({"content": out}).encode())


class TestLLMClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), _FakeLlama)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def llm(self):
        m = L.LocalLLM(port=self.port)
        st = m.start(verbose=lambda *_: None)
        self.assertTrue(st.ok and st.reused)                  # an already-running server is re-used (persistent server)
        return m

    SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}

    def test_json_ok_fenced_and_schema_fallback(self):
        for mode in ("ok", "fenced", "no_schema"):
            _FakeLlama.mode = mode
            out = self.llm().json("s", "u", "followup", self.SCHEMA)
            self.assertEqual(out, {"question": "What did you do?"}, mode)

    def test_malformed_never_corrupts(self):
        for mode in ("garbage", "violation"):
            _FakeLlama.mode = mode
            m = self.llm()
            self.assertIsNone(m.json("s", "u", "interpret", self.SCHEMA), mode)
            self.assertTrue(m.records and not m.records[-1].ok)

    def test_unavailable_server_returns_none_quickly(self):
        m = L.LocalLLM(port=1, enabled=True)
        m.ready = True
        self.assertIsNone(m.json("s", "u", "x", self.SCHEMA, timeout=1))

    def test_parse_json_loose(self):
        self.assertEqual(L.parse_json_loose('noise {"a": [1,2,], "b": "x"} tail'), {"a": [1, 2], "b": "x"})
        self.assertIsNone(L.parse_json_loose("no json here"))


class TestDeterminism(unittest.TestCase):
    def test_same_transcript_for_different_hash_seeds(self):
        import subprocess
        code = ("import sys; sys.path.insert(0, %r); import test_v58 as T; e = T.run_sim(); "
                "print('\\n'.join(t.interviewer + ' || ' + t.answer for t in e.transcript))") % str(HERE)
        outs = []
        for seed in ("1", "2", "3"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outs.append(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(HERE)).stdout)
        self.assertTrue(outs[0].strip())
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], outs[2])


class TestEndToEnd(unittest.TestCase):
    def test_report_is_honest(self):
        from report_engine_v58 import build_report, render_text
        e = run_sim()
        rep = build_report(e)
        txt = render_text(rep)
        g = e.graph
        for c in rep["competencies"]:
            comp = g.comp[c["key"]]
            if c["validated"]:
                self.assertTrue(comp.direct_items(), c["name"])
                self.assertIn(comp.status, (DEMONSTRATED, PARTIALLY_VALIDATED, PRACTICAL))
            if not c["addressed"]:
                self.assertFalse(c["validated"] and not comp.direct_items())
        cov = rep["interview"]["coverage"]
        self.assertLessEqual(cov["validated"], cov["resolved"] + cov["validated"])
        self.assertLessEqual(cov["weighted_validated_pct"], cov["weighted_resolved_pct"] + 100)
        self.assertNotIn("English =", txt)
        self.assertNotRegex(txt, r"(?i)RAG\s+.*DEMONSTRATED") if g.get("rag").status != DEMONSTRATED else None
        json.dumps(rep, default=str)

    def test_runs_without_llm_and_without_tts(self):
        e = run_sim(llm=None)
        self.assertTrue(e.finished)

if __name__ == "__main__":
    unittest.main(verbosity=2)