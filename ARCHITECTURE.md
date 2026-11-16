# Architecture
```
main.py (orchestrator) -> shared/ (paths, config registry) -> Interviewer/ (V58 engine + V59 bridge)
  Interviewer --asks--> candidate (typed | mic+Whisper) ; Piper/Kokoro speak
  mic/cam -> Voice/ (voice_cues, acoustic, SER) , vision/ (pipeline v3 + extras) , NLP/ (text_cues, SBERT, STT)
  per-answer Signals -> fusion/ (baselines, LLR fusion, abstention, probes) -> report
```
Import mechanism: modules use flat imports; `shared.paths.bootstrap()` adds the module folders to `sys.path` (documented compromise in research/MIGRATION_MAP.md). Root `config.py` holds model paths and offline flags; `shared/model_registry.py` is the single model table. Missing modality = skipped, flagged in `quality`; fusion abstains when no cue is usable. Missing components: frontend, database, auth, billing.
