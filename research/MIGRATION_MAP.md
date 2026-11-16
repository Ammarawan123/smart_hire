# Migration map
Original ZIPs are preserved unmodified in `backups/original_source_backup/` (git-ignored). Test status = result in this session.

| Original | New location | Decision / changes | Tests |
|---|---|---|---|
| SmartHire_V58 `*_v58.py` (engine, planner, extractors, evidence, TTS/STT/LLM adapters) | `Interviewer/` | kept as-is; `llm_engine`/`tts_engine` search lists now start from `config.py` paths and no longer contain `F:\` | 64 original tests pass (67 with live-capture tests) |
| `test_v58.py`, `fixtures/` | `Interviewer/tests/` | import bootstrap only. `faaiz_resume_layout.txt` and `real_layout_render.pdf` are a real personal resume: git-ignored | pass |
| `.smarthire_cache_v58/*.pkl`, `interview_report_v58.json` | NOT migrated | contain the author's resume data (private); remain only in the backup ZIP | n/a |
| V59 `interviewer_bridge`, `claim_extractor`, `tts_timed`, `live_session`, `run_interview_v59` | `Interviewer/` | runner now calls `config.apply_runtime_env()` (offline flags) instead of `model_paths_v59` | `test_live_v59` pass |
| V59 `fusion_v59`, `behavior_config_v59`, `scaling_v59`, `eval_harness_v59`, `sim_study_v59`, `test_v59` | `fusion/` | unchanged | 34 tests, 2 skipped |
| V59 `text_cues_v59` | `NLP/` | unchanged | via fusion tests |
| V59 `voice_cues_v59` | `Voice/` | unchanged | via fusion tests |
| V59 `vision_extras_v59` | `vision/` | unchanged | via fusion tests |
| V59 `model_paths_v59`, `download_models_v59` | `shared/model_registry.py`, `scripts/download_models.py` | rewritten around per-module canonical folders; no `F:\Faaiz_work` default | integration tests |
| V59 `selftest_v59.py` | `scripts/selftest_v59.py` | now imports the registry; NOT re-run in full (needs packages/devices) | baseline: 28 pass / 7 fail (missing packages) |
| Voice `config.py` | `Voice/voice_config.py` | renamed to avoid clashing with root `config.py`; 5 imports updated | parse tests |
| Voice `acoustic_extractor, deception_engine, models_arena, benchmark, live_mic_runner` | `Voice/` | unchanged apart from import rename | NOT executed (needs librosa/torch/audio) |
| nlp `main.py`, `v5.py`, `sbert_analyzer.py` | `NLP/nlp_cli.py`, `NLP/stt_realtime_v5.py`, `NLP/sbert_analyzer.py` | renamed to avoid `main` clash | NOT executed (needs sentence-transformers/RealtimeSTT) |
| Vision `vision_pipeline_v3.py` | `vision/` | hard-coded `D:\` / `F:\` model paths replaced by `vision/vision_models/` defaults (env overrides kept) | parse + path tests |
| Vision `vision_pipeline.py` (v2) | `vision/legacy/vision_pipeline_v2_legacy.py` | kept for reference, still contains old paths | not tested |
| Vision `models/*` | `vision/vision_models/` | copied (3 files, ~24 MB; git-ignored) | presence verified |
| Vision PDFs | `vision/docs/` | kept | n/a |
| original requirements files | `*/docs/requirements_*_original.txt`, `research/v59_original_setup/` | superseded by root `requirements*.txt` | n/a |
| `.bat` installers | `research/v59_original_setup/` | superseded by `scripts/*.ps1`; not run | n/a |
