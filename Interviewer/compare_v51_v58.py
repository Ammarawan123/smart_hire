"""compare_v51_v58.py — replay the REAL V51 failure case against V58 and print an acceptance table.

Inputs: the résumé/JD text captured from V51's own cache (fixtures/), the V51 transcript facts (hard-coded below, copied from the V51 run), and a
simulated candidate (sim_candidate_v58.py) that answers what is *asked* in the V51 candidate's voice. Also replays the literal V51 answers in the
V51 question contexts through V58's answer understanding.   Run: python compare_v51_v58.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.paths import bootstrap  # noqa: E402
bootstrap()
sys.path.insert(0, str(HERE))
import interview_engine_v58 as IE
import jd_extractor_v58 as J
import resume_extractor_v58 as R
import sim_candidate_v58 as SIM
import answer_understanding_v58 as A
import evidence_engine_v58 as E
from models_v58 import QuestionContext as Q
from report_engine_v58 import build_report

R_TXT = (HERE / "fixtures/faaiz_resume_layout.txt").read_text(encoding="utf-8")
J_TXT = (HERE / "fixtures/contour_jd.txt").read_text(encoding="utf-8")

V51 = {
    "projects": ["Sign Language Detection", "OpenCV", "English", "I N T E RE S T S", "E X T R A C U R R I C U L AR A C T I V I T I E S"],
    "skills": "31 mapped / 0 listed",
    "q9": "Can you explain what Sign Language Detection using YOLOv8 20252005faaiz@gmail.com does, and how you built it?",
    "questions": 10, "minutes": 22.7, "weighted_required_addressed": 38, "weighted_required_validated": 4,
    "asked_jd_topics": [],  # none of RAG / Prompt Engineering / REST / SQL Server / ASP.NET / Angular / architecture was asked
    "reaction_openers": ["Okay.", "Right that gives me a picture of OpenCV.", "Right.", "Alright.", "Alright.", "Got it, so that's English.", "Okay.", "Right.", "Okay."],
}

ok_all = True


def check(label, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))


print("=" * 78 + "\nA. EXTRACTION  (same résumé/JD text V51 parsed)\n" + "=" * 78)
r = R.extract_resume(R_TXT, is_text=True, filename="Faaiz_Resume.pdf")
j = J.extract_jd(J_TXT, True, "Contour Software", "Trainee Software Developer")
print(f"  V51 projects : {V51['projects']}\n  V58 projects : {[p.short_name for p in r.projects]}")
print(f"  V51 skills   : {V51['skills']}\n  V58 skills   : {len(r.all_skills())} extracted, {sum(1 for s in r.all_skills() if s.listed)} listed; categories: "
      + ", ".join(f"{c}={len(l)}" for c, l in r.skills.items() if l))
names = [p.short_name.lower() for p in r.projects]
for bad in ("english", "opencv", "interests", "extracurricular activities"):
    check(f"'{bad}' is not a project", not any(bad in n for n in names))
check("no spaced-letter headings in projects", not any(re.search(r"\b[A-Z] [A-Z] [A-Z]\b", p.title) for p in r.projects))
check("no e-mail/phone/URL inside project text", not any(re.search(r"@|\d{4}-\d{7}|github\.com|linkedin\.com", p.title + p.description) for p in r.projects))
check("contact captured separately", bool(r.contact.emails and r.contact.phones and r.contact.locations), f"{r.contact.emails} {r.contact.phones} {r.contact.locations}")
check("education parsed", len(r.education) == 2 and r.education[0].field_of_study == "Artificial Intelligence", f"{[(e.degree, e.field_of_study, e.institution, e.dates, e.gpa) for e in r.education]}")
check("spoken languages not skills", {"English", "Urdu"} <= {x["language"] for x in r.spoken_languages} and "english" not in {s.name.lower() for s in r.all_skills()})
check("JD: Accounting/ERP are company context, not requirements", "accounting" not in j.requirements and "erp" not in j.requirements, f"company_context={j.company_context}")
check("JD: RAG, Prompt Eng., REST, SQL Server, ASP.NET, Angular, architecture are required", all(k in j.requirements for k in ("rag", "prompt_eng", "rest_api", "sqlserver", "aspnet", "angular", "architecture")))

print("\n" + "=" * 78 + "\nB. ANSWER UNDERSTANDING  (V51's literal answers, in V51's question contexts)\n" + "=" * 78)
g = E.EvidenceGraph(r, j)
before = {k: c.status for k, c in g.comp.items()}
turn = 1
for text, ctx in [
    ("OpenCV is a prebuilt library i cant built it", Q("project_probe", ["opencv"], "OpenCV", text="What part of OpenCV did you personally build?")),
    ("I dont built it", Q("project_probe", ["opencv"], "OpenCV", text="What part of OpenCV did you personally build?")),
    ("When did i say i have not used it", Q("clarify", [], "OpenCV")),
    ("I have not write english", Q("project_probe", ["genai"], "English")),
    ("I collect it myself like i download the dataset of some images and if a class which i want i cant find then i take my image of making sign gestures",
     Q("project_probe", [], "Sign Language Detection", "dataset", text="How did you collect the dataset?")),
]:
    it = A.understand(text, ctx, r, g)
    g.apply(ctx, it, turn)
    turn += 1
    print(f"  > {text[:70]!r}\n      intent={it.intent} lib={it.anchor_is_library} claims={[(c.key, c.stance) for c in it.claims]} facts={[f['phrase'] for f in it.facts]} contradictions={it.contradictions}")
check("Machine Learning NOT set to NO_EXPERIENCE by the OpenCV answers", g.get("ml") is None or g.get("ml").status != "EXPLICIT_NO_EXPERIENCE")
check("Python NOT set to NO_EXPERIENCE by the OpenCV answers", g.get("python").status == before["python"])
check("no contradiction manufactured", all(not A.understand(t, c, r, g).contradictions for t, c in [("I dont built it", Q("project_probe", ["opencv"], "OpenCV"))]))
check("broken-English dataset answer understood (download + own capture for missing classes)", any("downloaded" in f["phrase"] and "captured" in f["phrase"] for f in A.understand(
    "I collect it myself like i download the dataset of some images and if a class which i want i cant find then i take my image of making sign gestures",
    Q("project_probe", [], "Sign Language Detection", "dataset"), r, g).facts))

print("\n" + "=" * 78 + "\nC. FULL SIMULATED INTERVIEW  (V58, same résumé/JD/role; candidate persona from V51 answers)\n" + "=" * 78)
e = IE.InterviewEngine(r, j)
o = e.start()
print(f"Smarty : {o.greeting} {o.text}")
while not o.finished:
    a = SIM.answer(e.ctx, o.text)
    print(f"You    : {a}")
    o = e.step(a)
    print(f"Smarty : {o.farewell if o.finished else o.text}")
rep = build_report(e)
cov = rep["interview"]["coverage"]
txt_all = " ".join(t["interviewer"] for t in rep["transcript"])
print("\n" + "=" * 78 + "\nD. ACCEPTANCE TABLE\n" + "=" * 78)
check("No question about English / OpenCV / Interests / Extracurricular as a project", not re.search(r"(?i)talk about (english|opencv|i n t|interests|extracurricular)", txt_all))
check("No e-mail text inside any question", not re.search(r"\w+@\w+\.\w+", txt_all))
asked = {c["key"] for c in rep["competencies"] if c["addressed"]}
for k, nm in [("rag", "RAG"), ("prompt_eng", "Prompt Engineering"), ("rest_api", "REST API"), ("sqlserver", "SQL Server"), ("aspnet", "ASP.NET"), ("angular", "Angular"), ("architecture", "architecture")]:
    check(f"{nm} was actually asked", k in asked)
check("RAG question is a real RAG question", bool(re.search(r"Retrieval-Augmented Generation.*retrieval pipeline", txt_all)))
check("no-experience on RAG led to a concrete scenario, not a dead end", "documents and needed an LLM to answer questions using only those documents" in txt_all)
qs = [t["interviewer"] for t in rep["transcript"]]
check("no bare 'Okay./Right./Alright.' reactions", not any(re.match(r"^(Okay|Right|Alright|Got it)[.,]", q) for q in qs))
check("no duplicate questions", len({t.question for t in e.transcript}) == len(e.transcript))
check("every question has why / new-evidence / insufficiency logged", all(t["why"]["why_this_question"] and t["why"]["what_new_evidence_it_seeks"] for t in rep["transcript"]))
rag = next(c for c in rep["competencies"] if c["key"] == "rag")
check("RAG is not claimed as hands-on from a scenario-only answer", rag["validation_basis"] != "hands_on", f"status={rag['interview_status']} basis={rag['validation_basis'] or 'none'} (JD wording: {rag['expected_depth']})")
check("any hands-on validation rests on first-person evidence", all(e.graph.comp[c['key']].direct_items() for c in rep["competencies"] if c["validation_basis"] == "hands_on"))
check("report keeps résumé evidence apart from interview status", all("resume_evidence" in c and "interview_status" in c for c in rep["competencies"]))
print(f"\n  V51 : {V51['questions']} questions | {V51['minutes']} min wall | weighted required asked {V51['weighted_required_addressed']}% | validated {V51['weighted_required_validated']}% | JD topics asked: none of the 7 above")
print(f"  V58 : {rep['interview']['questions']} questions | {rep['interview']['minutes_virtual']} min (speaking+answering estimate) | weighted required asked {cov['weighted_addressed_pct']}% | "
      f"resolved {cov['weighted_resolved_pct']}% | validated {cov['weighted_validated_pct']}% (hands-on or JD-awareness-level conceptual)")
print(f"  V58 distinct reaction openers: {len({re.split(r'(?<=[.!?])', q)[0] for q in qs[1:]})}/{len(qs) - 1}")
print("\nRESULT:", "ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
