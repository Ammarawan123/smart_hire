# Methodology decisions
Label: [lit] supported by a verified source; [eng] engineering heuristic, unvalidated; [prior] uncalibrated prior (see `fusion/behavior_config_v59.py`, which tags every constant).
1. Competency assessment is driven by answer evidence, JD requirements and resume evidence with explicit status levels (V58 evidence graph) [eng]. Resume claims are kept separate from interview evidence.
2. Non-verbal cues (gaze, face, voice) are coaching/context signals. They cannot raise a flag alone [eng, motivated by Luke et al. 2023 (verified) and the absence of person-independent validation].
3. Fusion = per-person baseline z-scores -> bounded log-likelihood-ratio per cue group -> missing modalities are skipped, never imputed; abstains when quality gates fail [eng]; effect sizes are priors, uncalibrated.
4. "Deception" is not output as a probability. The report states "needs verification" items and probe outcomes (verified / unresolved / inconsistent).
5. LLM: Qwen2.5-1.5B-Instruct Q4_K_M via llama.cpp, optional; deterministic fallback is the tested default path.
6. Offline by default: `config.apply_runtime_env()` sets HF/Transformers offline flags; downloads only through `scripts/download_models.py`.
7. One canonical location per model; SER models (~1.2 GB each) are optional and not co-loaded with Qwen on 4 GB.
Pipeline (as implemented): capture (typed or mic/cam) -> preprocessing/quality gates (voice_cues, vision_extras) -> feature extraction (text_cues, voice_cues, vision) -> temporal alignment to the question clock (`tts_timed_v59`, `SignalCollector`) -> per-person normalisation -> group assessment -> fusion with abstention -> report.
