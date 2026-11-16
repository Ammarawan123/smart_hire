# SmartHire V51 — internal diagnosis (written before V58 was implemented)

Evidence used: the V51 source in the ZIP, the V51 console output/report you pasted, and the raw résumé/JD text recovered from V51's own cache
(`.smarthire_cache_v51/*.pkl`). The original PDF was not in the ZIP; V58 was tested on that recovered text *and* on a PDF rendered from it
(see "Limitations" in `research_v58.md`). I could not run V51 itself end-to-end (no llama-server/model/Piper in my sandbox), so the interview-level
findings come from your transcript, not from a re-run.

## Root causes (ordered by how much damage they did)

| # | Symptom in the V51 output | Root cause (architectural, not cosmetic) |
|---|---|---|
| 1 | `OpenCV`, `English`, `I N T E RE S T S`, `E X T R A C U R R I C U L AR…` became *projects*; `skills=31 mapped / 0 listed`; "No Projects section detected"; email glued into a project title | **Extraction ran on position-ordered text of a two-column PDF.** Sidebar rows (EDUCATION, LANGUAGES, INTERESTS) were interleaved with the project column; letter-spaced headings were never collapsed so no heading was recognised; the contact column was glued to sentences (`real-time0320-2797793`). With no sections, the parser fell back to `fallback_blocks` and promoted *any* capitalised block (or technology name) to a project. There was no rule that a project needs a title **and** a body in a projects-type section. |
| 2 | Q2 "tell me about OpenCV… what did you build yourself?" and Q5 "Let's talk about English…" | Consequence of #1: the planner ranked those fake "projects" by JD keyword value (OpenCV 2.8, English 0.9). Project relevance was keyword overlap, never *structure*. |
| 3 | "I dont built it" (answer about OpenCV) → `Machine Learning NO_EXPERIENCE`, `Python NO_EXPERIENCE`, then a *manufactured* "your résumé lists Machine Learning but you said you haven't used it" | **Answer understanding attached a bare negation to every competency "implied" by the question (`elliptical` attribution)** instead of to the one thing the clause talks about. It also treated "I didn't build OpenCV" (a library nobody expects you to build) as "no experience". The contradiction detector then compared the manufactured evidence with the résumé. |
| 4 | The interviewer repeats ownership questions ("what did you build" ×3) | Follow-ups were chosen by *answer-quality label* (`vague → probe`) with a small template pool, not by *which evidence is still missing*. No semantic de-duplication; no reason recorded per question. |
| 5 | `source=template (fallback: validation:not_a_question)` ×5, each after a 12–25 s LLM call | The LLM was used to **phrase** every question, its output was validated only at the end, and a failed validation threw the work away and fell back to a generic template. Latency paid, no benefit. |
| 6 | RAG, Prompt Engineering, ASP.NET, Angular, SQL Server, REST API, architecture were never asked; report said "addressed 38%" | Coverage was driven by the (broken) project list; the blueprint listed gaps but the planner had no "highest-value unresolved requirement" loop. "Addressed/resolved" mixed résumé evidence, blueprint and interview. |
| 7 | `ERP` and `Accounting` listed as core required competencies | JD extraction scanned the whole text. Both words occur only in company boilerplate (division description). |
| 8 | Reactions: "Okay." "Right." "Alright." "Got it, so that's English." | Reaction was a function of the *decision type*, not of the answer content; a fixed pool of fillers; one reaction even confirmed a nonsensical fact ("that's English"). |
| 9 | 10 questions in 22.7 min; planning 20–37 s per turn | Two LLM calls per turn on a 1.5B CPU model (phrasing ≈ 21 s, adjudication ≈ 19 s), plus ~28 s start-up, plus an `analysis` step that itself called the LLM once for 22 s. |
| 10 | "Data preparation → THEORETICAL/LEARNING_ONLY" for a candidate who described downloading and capturing their own images | Evidence depth was decided by keyword cues; broken English and the sentence structure "if a class … I can't find, then I take my image" were mis-parsed (the negation "can't find" was scoped over the whole clause). |

## Design consequences for V58
1. Read documents **geometrically** (words with coordinates → gutter → bands/columns → lines) and strip contact data *inside tokens* before anything else.
2. Detect headings by **shape**, classify them by **meaning** (squashed-text stems + content classification), never by exact spelling; a word can't open a section unless it looks like a heading.
3. A **project is an item with structure** (title + body, in a projects-type section); a skill is an item of a skills section; spoken languages are separate.
4. JD: classify **sections first**; only requirement-bearing sections create requirements; company text is kept as context.
5. A real **evidence graph**; updates are scoped to the competencies a clause actually names; implied evidence is stored but never validates.
6. Answer understanding = dialogue acts → clause analysis with **scoped negation** → entity linking → facets → (optional, validated) LLM.
7. Planner = **coverage matrix × importance × evidence gap**, follow-up **policy** driven by *missing evidence*, difficulty ladder, semantic de-duplication, reasons logged.
8. Reaction and question are **one utterance**, composed from what the candidate actually said.
9. LLM only for ambiguity and weak-hook follow-ups, strict JSON via grammar-constrained decoding, hard call caps, deterministic fallback everywhere.
