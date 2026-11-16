# SmartHire V58 — research notes

Scope and honesty: every external claim below was checked in this session against the source named next to it. Where I did **not** verify something
or could **not** measure it, the text says so. My sandbox had no access to Hugging Face (blocked), no llama-server/GGUF and no Piper, so **no neural model
was downloaded or benchmarked here**. Latency/RAM numbers for neural options are therefore *published* figures, not measurements on your i5/4 GB machine.
The only numbers I measured myself are for the V58 hybrid parser (bottom of §2).

## 1. How strong AI-interview systems actually work (design principles, not marketing)

| Source (verified) | What it says that is usable offline | Where it shaped V58 |
|---|---|---|
| Yazdani, Mahajan, Ansari, *Zara: An LLM-based Candidate Interview Feedback System*, arXiv:2507.02869 (cs.HC, Apr 2025) | A GPT-4o-based recruiter that generates role/skill-tailored practice interviews, runs conversational AI-led interviews and produces structured feedback. It is a **case study of a hosted-LLM product**; it does not publish a local-model question-selection algorithm. | Used only as the product benchmark you named (role+skills → tailored AI-led interview → structured report). I did **not** copy an implementation (none is described). |
| *Better Together: Quantifying the Benefits of AI-Assisted Recruitment*, arXiv:2507.08029 (micro1 AI Recruiter, field experiment) | The platform derives skills from the JD, builds an **interview blueprint**, poses an **initial question per targeted skill**, then generates follow-ups probing **depth of understanding, practical application and problem-solving approach**; the report rates each skill. Internals are proprietary. | The *skill → initial question → evidence-seeking follow-up → per-skill rating* loop (V58 coverage matrix + follow-up policy). The blueprint is **not** evidence (V58 keeps résumé evidence and interview evidence apart). |
| *PolyInterview*, arXiv:2607.10310 | JD+CV-personalised questions; **answer-aware follow-ups using diagnostic scoring, bounded follow-up rounds and explicit exit conditions**; proceeds when a response is "sufficiently resolved" or a time/follow-up limit is reached. | `MAX_FOLLOWUPS_PER_TOPIC`, explicit exit rules (sufficient evidence → move on), time budget per phase. |
| *Requirements Elicitation Follow-Up Question Generation*, arXiv:2507.02858 | Studies LLM follow-up generation from preceding context and **common interviewer mistakes** to avoid. (I read the approach section only.) | Motivation for the validator (no leading/ungrounded/duplicate questions) and for logging *why* a question is asked. |
| Kim et al., *LLM-as-an-Interviewer*, arXiv:2412.10424 | A multi-turn framework where the interviewer **probes deeper after correct responses and gives targeted feedback after incorrect ones**. (It evaluates LLMs, not people; I use only the control idea.) | Difficulty ladder: strong answer → step up; weak → simplify once, then move on. |

What I could **not** find: a peer-reviewed, reproducible algorithm for *local small-model* interview planning, nor empirical work on "conversational reactions" in AI interviews specific enough to cite.
The V58 reaction rules (§5) are therefore **my own design**, based on the failure analysis, not on a cited study.

## 2. Résumé extraction — what I investigated and what I chose

| Option | Facts I verified | Verdict for CPU-only, ~4 GB, offline, Windows |
|---|---|---|
| **Layout-aware parsing + efficient LLM** (arXiv:2510.09722, Zhu et al.) | A fine-tuned *layout parser* normalises heterogeneous résumé layouts, then a compact LLM extracts fields; they report a fine-tuned **0.6B** LLM reaching top-tier accuracy with lower latency. | Confirms the architecture "layout first, extraction second". Their fine-tuned models/data are not something I can ship or train here. **Adopted the principle**, not the models. |
| **XY-Cut / XY-Cut++** (arXiv:2504.10258, Liu et al.) | Classic XY-Cut recursively splits by projections and struggles with complex (e.g. L-shaped) layouts; XY-Cut++ adds pre-masking of cross-layout elements, multi-granularity splitting and light semantic cues. | V58 uses a **simplified, résumé-specific** variant (full-width bands vs. column bands, gutter by x-occupancy, straddlers assigned by centre). It is *not* an implementation of XY-Cut++. |
| **GLiNER** (arXiv:2311.08526; model cards on HF) | Compact bidirectional encoder for zero-shot NER. Paper: small ≈50M, medium ≈90M; HF cards list v2.x as **166M / 209M / 459M** parameters (the two sources differ in how they count); v2/v2.1 are Apache-2.0, v0/v1 are CC-BY-NC. | Plausible optional upgrade for *free-text* skill/entity spotting (CPU-capable per its authors). **Not measured** by me (no model access), needs `torch`+`gliner`, adds hundreds of MB of RAM on a 4 GB machine, and gives *entities*, not *document structure* — which was V51's actual failure. Left as an optional future add-on, not a dependency. |
| LayoutLM-family, DeBERTa/MiniLM/DistilBERT classifiers, spaCy NER | **Not researched/benchmarked in this session** — I will not quote sizes or accuracy from memory. LayoutLM-type models need page images/OCR boxes and fine-tuning data I do not have; a MiniLM/DeBERTa section classifier needs labelled résumés. | Not adopted. The semantic section classifier in V58 is lexicon-stem + content features; its limits are listed in §7. |

**Selected: hybrid, deterministic, zero-download.** `doc_reader_v58` (PyMuPDF word boxes → pdfplumber fallback → pseudo-geometry for text/DOCX) →
`sections_v58` (shape × meaning heading detection; content classifier for unnamed blocks) → `resume_extractor_v58` (item grammar: entry = title + meta + bullets;
skills lists; education/cert/experience parsers) → concept registry (`concepts_v58`, ~85 concepts, 7 résumé skill categories).
*Why:* the V51 failure was structural (reading order, headings, item boundaries). A model that tags entities cannot repair interleaved columns; geometry can.
It costs nothing to ship, runs on any CPU, and every decision is inspectable for the viva.

*Measured on this sandbox (Python 3.12):* résumé from text ≈ 106 ms, from PDF incl. PyMuPDF ≈ 200 ms, JD ≈ 37 ms, whole process peak RSS ≈ 53 MB. Your hardware will differ, but there is no model to load.

## 3. JD understanding
Section classification first (requirements / preferred / competencies / responsibilities / role context / ignore), then concept mining only in requirement-bearing sections;
depth cue words ("strong", "good knowledge", "exposure", "familiar") set `expected_depth` and importance; `evidence_expected` comes from the concept's evidence facets.
This is a rule design; I found no published JD-parsing benchmark that I could reuse, so there is **no accuracy figure** for it beyond the tests in `test_v58.py`.

## 4. Answer understanding
Pipeline: dialogue acts (skip/quit/help/repeat/unknown/no-experience/dispute) → clause split → **scoped negation** (a negator counts only when it governs a use/build/know verb) →
entity linking (explicit mention, or the question's single target for "it/this") → stance (practical / theoretical / learning / no-experience / unknown / claim) →
**evidence facets** (what counts as evidence for *that* concept) → ownership → quality. Grammar and word order are not inputs to quality.
Optional Qwen pass only when the deterministic reading is ambiguous: strict JSON, enum-restricted concepts, quoted evidence must be found in the answer, add-only.
*Research basis:* none specific beyond the interviewing principles above; this is an engineering design validated by tests, not by a labelled corpus.

## 5. Reactions
My rules (not from a cited study): refer to something actually said; praise only when the answer earned it (≥2 evidence facets/aspects *and* good quality), at most every third turn;
never bare filler; acknowledge "no experience" without blame and move to a scenario; a dispute ("when did I say that?") is accepted, not argued;
reaction + question are composed as one utterance; wording rotates and is never reused inside a window.

## 6. Question generation
Concept question bank (basic → implementation → depth → trade-off → debug + a *scenario* variant for no-experience), grounded bridges ("In *Fraday* you used *Cohere*. How did you write the prompts…"),
candidate-fact hooks ("You mentioned that you downloaded existing images and captured your own for the classes you couldn't find…"). Validator: is an ask, ≤2 questions, grounded
technologies only, relevant to target, not a near-duplicate *on the same topic*, asks something new. Qwen is used for follow-up phrasing only when the candidate's own wording must be woven in; invalid output is discarded.

**Structured output:** llama.cpp's server accepts a `json_schema` field on `/completion` (grammar-based sampling) and `cache_prompt` to reuse the KV prefix (llama.cpp server README). V58 uses both.
Caveat found while researching: a llama.cpp issue (#19051) reports that if the schema→grammar step fails the server can **continue unconstrained with HTTP 200** — so V58 *always* parses and schema-validates the reply itself.

## 7. Limitations (please read before the viva)
* **No real Qwen run.** The LLM client was tested against a fake HTTP llama-server and a stub; I could not measure real latency or Qwen's JSON compliance. Your first real run is the real test. `--no-llm` runs the full interview deterministically.
* **No Piper/STT run here.** TTS/STT modules are your V51 files renamed (config import changed); typed mode was run; speech paths were not exercised in my sandbox.
* **Résumé evidence base is thin:** your résumé (as text recovered from V51's cache and as a PDF *rendered from that text*, not your original PDF), two synthetic layouts, V51's three fixtures. No precision/recall over a corpus. Real PDFs may expose word-gap/spacing cases I haven't seen. Scanned/image-only PDFs: unsupported (no OCR).
* **Lexicon-bound:** ~85 concepts. Unknown technologies become `custom:` skills (listed, but no tailored question). Non-English text and Urdu script are not handled.
* **Heuristic answer understanding:** negation/stance rules can be wrong on unusual phrasing (e.g. "I can't find the time to use Angular" is read as use). Facet matching can be gamed by keyword stuffing.
* **The simulated candidate is mine.** The V58 transcript shows behaviour on a scripted persona built from your V51 answers; it is not evidence about real candidates.
* "Validated" has two flavours (hands-on vs. conceptual-for-awareness-level JD wording) and is a **policy choice** documented in `evidence_engine_v58.py`.
* The 8–12 min target uses a **virtual clock** (speaking + typing/answer time estimates), not wall-clock, so LLM latency doesn't end the interview early.
* Deception/behavioural multimodal analysis is out of scope of this runner and was not touched.
