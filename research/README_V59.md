# SmartHire V59 — behaviour layer for the V58 interviewer (add-on package)

**Nothing in V58 was edited.** These are new files that sit next to the V58 files (`SmartHire_V58_032005/`). Copy all of them into that folder (or set
`SMARTHIRE_V58_DIR` to it), then:

```
python -m unittest test_v59 test_live_v59 -v        # 41 tests of the new layer (V58's own 64 tests still pass)
python selftest_v59.py                              # checks packages, model files, every cue (see SmartHire_V59_Guide.pdf)
python run_interview_v59.py --resume <cv> --jd <jd> --simulate --no-llm     # full simulated interview, report has a 'behavior' section
python run_interview_v59.py --resume <cv> --jd <jd> --stt whisper --voice <piper.onnx> --debug     # real run (see "Untested" below)
```

## What each file does

| File | Replaces / extends | Purpose |
|---|---|---|
| `behavior_config_v59.py` | – | every number, tagged `[lit] / [lit?] / [prior] / [guard]` so no hand-set value can pass as a result |
| `fusion_v59.py` | your hand-set Type A/B/C weights and the planned SVM+LSTM | Bayesian per-person baselines, correlation-corrected group statistics, bounded log-likelihood-ratio fusion, conjugate learning of effect sizes, Platt + conformal abstention |
| `text_cues_v59.py` | `nlp/` (SBERT cosine only) | detail density, quantity conflicts, NLI contradiction (optional model), fillers/hedges (coaching) |
| `voice_cues_v59.py` | `Voice/` | latency from the interviewer's TTS clock, pauses, articulation rate, pitch in semitones, quality gating, experimental second-voice drift, adapter for your SER arena |
| `vision_extras_v59.py` | `Vision/vision_pipeline_v3.py` (adds, does not modify) | head nod/shake, reading-from-screen score (left-to-right and right-to-left), off-screen fixation, eye-contact ratio |
| `claim_extractor_v59.py` | – | résumé claims via rules, **NuExtract** (1.x and 2.0 formats) or **Qwen2.5** (`LocalLLM.json`); every claim must be quoted from the text |
| `tts_timed_v59.py` | – | `TimedTTS` (exact question-end time) and a **Kokoro** engine with the same interface as your Piper engine |
| `interviewer_bridge_v59.py` | – | `BehaviorAwareEngine(InterviewEngine)`: per-answer analysis, verification probes, probe outcomes, report |
| `live_session_v59.py`, `run_interview_v59.py` | – | audio capture with a monotonic clock, camera adapter, V58 runner reused unchanged |
| `model_paths_v59.py`, `download_models_v59.py` | – | one table of every model (source, size, licence) and a downloader into `F:\Faaiz_work\FYP\Models` |
| `selftest_v59.py` | – | stages A–G: environment, model files, every cue on synthetic data, unit tests, real models, microphone, camera |
| `scaling_v59.py` | – | shared model registry, bounded analysis pool, session store, `BehaviorService` for many simultaneous interviews |
| `INSTALL_V59.bat`, `setup_env_v59.bat`, `requirements_all_v59.txt` | – | install and environment (not run by the author: no Windows/network there) |
| `eval_harness_v59.py`, `sim_study_v59.py` | – | subject-grouped CV, cluster bootstrap CI, ablations, conformal coverage, fairness table; synthetic generator for machinery checks |

## How the interviewer is connected

```
V58 InterviewEngine ── question ctx (type, difficulty, targets) ──▶ BehaviorSession ◀── Signals per answer ── mic (audio+clock) / TimedTTS (question end) / Vision v3 segment
        ▲                                                                │
        └── verification probe (extra question, once per topic, ≤3) ◀── flag = content evidence AND (hard conflict OR posterior ≥ 0.40 with ≥2 groups)
```
Probe wording is a fixed, neutral template filled with the topic name and checked by V58's own question validator (no ungrounded technology names, no
duplicates). A flag changes status only through the probe outcome (`verified / unresolved / inconsistent`); the report never says "lying".

## A deliberate change from my earlier review
The earlier review allowed two *non-verbal* groups to trigger a probe. I removed that: **face/voice evidence alone can no longer raise a flag**
(`fusion_v59.py`, test `test_nonverbal_alone_never_flags`). The literature gives too little support for it (see RESEARCH_V59.md) and it is the main
source of unfair false flags.

## Untested here (be honest in the viva)
* Real microphone / camera / Whisper / Piper / Kokoro / NuExtract / Qwen runs. Logic around them is tested with fakes and a stub HTTP server; the models were not available.
* The NLI model (`cross-encoder/nli-deberta-v3-xsmall`) — tested through an injected stand-in, not the real weights.
* `[prior]` numbers in `behavior_config_v59.py` (reading threshold, drift threshold, baseline priors, hard-evidence weights) are uncalibrated.
* Nothing here has been evaluated on real candidates. The synthetic study checks the maths only.

## First things to do on your machine
1. `python -m unittest test_v59 test_live_v59 -v`, then V58's `python -m unittest test_v58`.
2. Run one real interview with `--stt whisper --debug`; check the JSON `behavior.turns[*].cues.response_latency_s` is plausible (0.3–4 s) and `quality` flags make sense.
3. Record 10 people reading text aloud and 10 speaking freely (for `reading_score` and `offscreen_fixation_s`), and replace the `[prior]` thresholds.
4. Run the planted-claim study in RESEARCH_V59.md §6; feed rows to `eval_harness_v59.evaluate/ablation/calibrated_gate`.
