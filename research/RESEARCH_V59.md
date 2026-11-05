# RESEARCH_V59 — evidence, fusion rationale, evaluation plan

Evidence labels: **[V]** = source opened and read (or its abstract/model card) while preparing this package. **[R]** = recalled or taken from your own
Vision report; verify before citing. Textbook statistics (Stouffer, conjugate Bayes, Platt, Welch) are cited by name only.

## 1. What "proof that it is correct" can mean

| Layer | What can be proven | Status |
|---|---|---|
| **Mathematical properties of the fusion** | evidence is not double-counted, is bounded, is additive/explainable; baselines are honest with few observations; abstention has a coverage guarantee (under stated assumptions) | proven by construction and **tested** (`test_v59.py`) |
| **Plumbing** | the interviewer, STT clock, TTS clock, cue modules and report agree on time and identity; a failure never stops the interview | **tested** with fakes (`test_v59.py`, `test_live_v59.py`) |
| **Validity: do these cues predict an unsupported claim?** | only data can show this | **not proven**. The literature says nonverbal cues are weak (below). You prove it with the planted-claim study in §6 |

Anyone who tells you a webcam/microphone system is "proven" to detect lies without a labelled study of the target setting is overclaiming; the best cross-domain
audio-visual systems sit near 60% accuracy [V: Guo et al. 2024, arXiv 2405.06995, 59.13% multi-to-single; SVC 2025, arXiv 2508.04129 / 2508.12842, 60.43% acc, 56.99% F1].

## 2. Evidence per cue and how V59 uses it

| Cue | Evidence | Use in V59 |
|---|---|---|
| Gaze aversion | >80% of 50 experts: not diagnostic [V: Luke et al. 2023] | context only (`gaze_shift_rate`, load group, weight 0.4); integrity/reading features separate |
| Vocal pitch | higher when lying, d = 0.268 [V as reported from Sporer & Schwandt 2006]; DePaulo et al. 2003 also report higher pitch [R] | arousal group, in semitones, baseline-relative |
| Response latency | d = 0.177 [V as reported]; mixed elsewhere | load group, measured from the TTS-end clock, condition on question type |
| Pauses, speech rate | inconsistent [V] | load group, low weights |
| Fillers, hedges, style words | not reliable deception cues [V: Sporer & Schwandt as reported]; text-style detectors near chance on belief-based data, LLM truth bias [V: Velutharambath & Klinger, ACL 2025] | **coaching only** (Student mode), never fused |
| Detail richness, verifiability, complications, consistency | verbal cues are mostly cues to truthfulness [V: Caso et al. 2023; Vrij et al. 2022]; liars give fewer details and less consistent accounts [V via secondary report: DePaulo 2003] | **content group** = strongest channel, plus hard evidence (NLI contradiction, quantity conflict) |
| Emotion / SER arousal | emotion is not deception; transformer SER fair across gender but not across individual speakers [V: Wagner et al., arXiv 2203.07378] | arousal group, per-speaker baselines |
| Jitter, shimmer, HNR | no deception-specific validation found in this search | kept in your Voice module, **weight 0** in fusion |
| Self-adaptors (hand-to-face) | single small study [R] | adaptor group, weight prior 0.10 |
| Micro-expressions, saccades, pupil | need high-rate sensors [R] | not claimed |
| Reading from a screen / off-screen fixation / second voice | no validation study found | integrity block, **experimental**, never fused, thresholds are `[prior]` |
| Job-interview context | applicants' deceptive impression management is measurable in video interviews (self-report labels, N = 121, face + head movements) [V: Suen et al., IEEE TCSS 2024]; faking intentions differ for AI interviewers [V: Bhattacharya et al., IJOA — verify authors] | motivates secondary label source (§6) |

## 3. How other deception systems work, and what to take from each

| System | Technique | What SmartHire takes | What it does not |
|---|---|---|---|
| DOLOS + PECL (Guo et al., ICCV 2023) [V] | big pretrained audio/visual encoders adapted with a small number of trainable parameters; gameshow data | the *principle*: freeze strong backbones, learn little. V59 learns 4 effect sizes. Stage 2 (≥ ~100 labelled subjects): logistic probe on frozen wav2vec2/HuBERT utterance embeddings, as an ablation against the hand-built cues | I did not re-read PECL's architecture; read it before describing it |
| Cross-domain benchmark (Guo et al. 2024) [V] | trains on several datasets, tests on another; Attention-Mixer fusion | evaluation habit: leave-session/device/accent-group out, not only random subjects. Expect the drop | its fusion network (needs far more data than you will have) |
| SVC 2025 challenge (Lin et al.) [V] | cross-domain challenge, 21 teams; top-2 entry ≈ 60% | an honest accuracy ceiling to quote | method details not read |
| Joshi et al. 2025 (your slide) [R] | late score-level fusion with RF/SVM over several sensors | late fusion with missing-modality tolerance (V59 does this); an RF stacker as an ablation (`scikit-learn`) | physiological sensors |
| Suen et al. 2024 [V] | deep model of face + head movement on interview video, self-report labels | head-movement features, self-report impression-management label as a secondary target | in-sample variance as an accuracy claim |
| Cognitive lie detection (Vrij, Fisher & Blank 2017, meta-analysis 71% vs 56%) [V via abstract] | impose load, ask unexpected and specific questions | **verification probes** by the interviewer (the main V59 idea) | the result is for human interviewers in lab studies; whether an automated interviewer reproduces it is your hypothesis H1 |
| Commercial AI interview vendors | reportedly moved away from facial scoring [R: your Vision report] | no face-based scoring of people | – |

## 4. Fusion: design and the properties that are actually proven

1. **Baseline (Normal–scaled-inverse-χ² conjugate update).** For *n* answers the person's mean and variance posterior is closed-form; z-scores use the posterior-predictive
   Student-t, mapped to a standard-normal quantile. With 1–2 answers the population prior dominates, so z-scores stay small instead of exaggerating noise.
   Questions are bucketed (intro / project / technical / behavioural); a bucket baseline is used after 3 answers, otherwise the global one — so a hard technical question is
   not compared with an easy introduction. Baselines absorb later answers only if they did not look unsupported (no contamination).
2. **Correlation-corrected Stouffer.** Cues of one group are merged as Z = Σ wᵢuᵢ / √(w′Rw). Test: with ρ = 0.5 across 4 cues, naive √k-scaled sums have variance > 1.5, the corrected Z has variance 1 ± 0.06.
3. **Log-likelihood ratio.** Under Z ~ N(0,1) vs N(d,1): LLR = d·Z − d²/2. Group LLRs add to the prior log-odds. Nonverbal total is clipped at ±1.2 (LR ≈ 3.3). The model is linear, so each cue's contribution to the log-odds is **exact** (no SHAP approximation).
4. **Learning.** `fit()` updates each group's *d* with a Normal–Normal posterior (prior sd 0.1–0.3) and refuses a negative effect. Four parameters, not thousands.
5. **Calibration and abstention.** Platt on out-of-fold scores, then class-conditional split conformal (Vovk; Angelopoulos & Bates 2021 [V]): per-class coverage ≥ 1−α under exchangeability. **Caveats [V]:** the guarantee is marginal (not per-person), and it breaks under distribution shift (new camera, accent, room). The gate refuses to claim anything with < 20 calibration segments per class.
6. **Content-anchored flag.** A flag needs hard content evidence, or a clearly positive content group *and* posterior ≥ 0.40 *and* ≥ 2 groups positive. Non-verbal evidence can only add to it.

Tests that back these: `test_stouffer_unit_variance_under_correlation`, `test_llr_formula_and_additive_attribution`, `test_nonverbal_alone_never_flags`, `test_fit_recovers_effect_and_never_flips_sign`,
`test_conformal_coverage_guarantee_on_exchangeable_data`, `test_heldout_conformal_coverage_close_to_target` (subject-grouped, synthetic ≈ 0.90 per class), `test_null_data_is_at_chance` (no effect → AUC CI contains 0.5, i.e. no leakage),
`test_grouped_folds_never_split_a_subject`.

## 5. The interviewer link (what flows where)

* Interviewer → cues: `type/difficulty` (baseline bucket), `targets` (claim topic), exact question-end clock, résumé claims (quotes), prior sentences (consistency).
* Cues → interviewer: flag + reasons → **one** probe from four templates (duration check, specifics, least-expected-problem, ownership), validated by V58's question validator; outcome measured on the next answer
  (`verified` = ≥ 25 words and detail density not below the candidate's own baseline; `inconsistent` = NLI contradiction ≥ 0.85-equivalent or quantity conflict; else `unresolved`).
* Limits: ≤ 3 probes per interview, one per topic, never before turn 3, never after skip / unknown / no-experience.
* LLM roles: Qwen2.5 (via `llama-server`) and/or NuExtract only **extract** claims, and every claim must be a quote found in the text. Neither is asked to judge truthfulness (LLMs show a truth bias [V]).

## 6. The study that would actually prove (or disprove) validity

**Target:** "claim not supported by the candidate's real experience" (fabrication/exaggeration), not general lying.

**Planted-claim protocol.** Each volunteer gets a résumé with some claims that are really theirs and some planted false/exaggerated ones (random assignment, balanced order), and is interviewed by the system.
Label = planted vs real, known by construction. Collect a second label: the volunteer's own post-hoc rating of how much they exaggerated each answer (impression-management self-report, as in Suen et al.).
Debrief everyone; get informed consent for recording (research ethics, not legal advice). Include volunteers across accents and English proficiency; record device/room.

**Sizing (approximate Hanley–McNeil standard errors, 8 segments each, 30% positives, assumed within-person correlation 0.3):**

| True AUC | 20 subjects | 30 | 40 | 60 |
|---|---|---|---|---|
| 0.65 | lower 95% bound 0.48 | 0.51 | 0.53 | 0.55 |
| 0.70 | 0.54 | 0.57 | 0.58 | 0.61 |
| 0.75 | 0.59 | 0.62 | 0.64 | 0.66 |

So 30–40 subjects can show "clearly above chance" if the true AUC is ≥ 0.70; they cannot show 0.80. Those numbers are my calculation, not a measurement.

**Hypotheses (state them before looking at data):** H1 probe-augmented score > passive score (AUC); H2 baseline-relative z > raw features; H3 content group > any non-verbal group; H4 face/voice add nothing beyond content (be ready to report this).
**Metrics:** AUC + cluster-bootstrap 95% CI, PR-AUC, recall at 10% false-positive rate, Brier and ECE, risk–coverage curve, per-group FPR (gender, accent, glasses), and leave-device-out. Code: `eval_harness_v59.py`.
**Acceptance wording for the viva:** "content group lower CI bound above 0.5; calibrated; no group with a false-flag rate more than twice another; non-verbal cues reported as context" — not "AUC > 0.80".
**Per-module checks:** FER on your own webcam frames (domain shift), gaze on target-dot tasks, VAD/pauses on hand-annotated clips, STT word-error rate on Pakistani-accented English, `detail_density` vs two human raters (Cohen's κ), reading detector on people reading aloud vs speaking freely.

## 7. Model notes for your stack [V unless marked]

* **NuExtract:** 1.x models use `<|input|> ### Template: … ### Text: … <|output|>`; 2.0 takes `# Template:` in a chat message. The 2.0-4B GGUF is under a non-commercial Qwen research licence; check the card for 2B/8B terms.
* **Kokoro-82M:** Apache-2.0 weights; `kokoro-onnx` runs on CPU; adapter is a drop-in for your Piper engine (`tts_timed_v59.KokoroTTS`).
* **audeering SER:** CC BY-NC-SA 4.0 (non-commercial). **ehcalabres SER:** trained on RAVDESS acted speech (1,440 clips) — domain shift expected.
* **NLI:** `cross-encoder/nli-deberta-v3-xsmall`, Apache-2.0, 87.77 MNLI-mismatched, label order (contradiction, entailment, neutral).
* **Whisper-family transcripts** tend to omit fillers [R]; measure on your recordings before reporting a filler rate.

## 8. Roadmap to something close to perfect (honestly: it will not be perfect)

1. Run §6 with ≥ 30 subjects; replace every `[prior]` by data (`fit_priors`, `fit`, thresholds).
2. If content wins, invest there: better claim extraction (NuExtract vs Qwen head-to-head on 50 résumés), human-coded detail validation.
3. Add the frozen-embedding probe (DOLOS/PECL idea) only after ≥ 100 subjects.
4. Treat distribution shift as the default: calibrate per device/room with a short baseline reading task.
5. Keep the output a *verification priority with evidence and a probe outcome*. That is the claim the evidence can carry.
