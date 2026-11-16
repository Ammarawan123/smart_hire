# SmartHire V58 (clean rebuild; V51 untouched)

Run (same inputs as your V51 run):
```
python run_interview_v58_interactive.py --resume $RESUME --jd $JD --company "Contour Software" --role-key "Trainee Software Developer" `
   --llama-server $LLAMA --model $GGUF --voice $VOICE --debug
```
Useful flags: `--no-llm` (fully deterministic), `--no-tts`, `--simulate` (scripted persona, no input), `--answers-file f.txt`, `--dump-profile` (print parsed résumé+JD as JSON), `--stop-server`, `--test-tts`.
Typed commands: `skip`, `next`, `repeat`, `help`, `quit`, `I don't know`, `I haven't used it`.

Tests: `python -m unittest test_v58 -v` (63 tests, hash-seed independent) · V51-vs-V58 replay: `python compare_v51_v58.py`.

| File | Role |
|---|---|
| `doc_reader_v58.py` | PDF/DOCX/TXT → ordered lines (geometry, columns, contact stripping, letter-spacing repair) |
| `sections_v58.py` | heading shape × meaning; content-based block classification |
| `resume_extractor_v58.py`, `jd_extractor_v58.py`, `schema_v58.py`, `concepts_v58.py` | structured résumé / JD, concept registry |
| `answer_understanding_v58.py` | intent, scoped stance, facets, ownership, optional validated LLM refinement |
| `evidence_engine_v58.py` | evidence graph, statuses, coverage (addressed / resolved / validated) |
| `interview_planner_v58.py`, `question_engine_v58.py`, `reaction_engine_v58.py` | what to ask next, how to phrase it, how to react |
| `interview_engine_v58.py`, `report_engine_v58.py`, `run_interview_v58_interactive.py` | orchestration, honest report, runner |
| `llm_engine_v58.py` | persistent llama-server client, strict JSON, call caps |
| `tts_engine_v58.py`, `stt_engine_v58.py` | your V51 voice modules (renamed; import `v58_config`) |
| `DIAGNOSIS_V58.md`, `research_v58.md` | what was wrong, what the research supports, limitations |
