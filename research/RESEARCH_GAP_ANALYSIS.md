# Research gaps and conflicts with the approved methodology
| Methodology item (FYP slides, 30 Apr 2026) | Repository reality | Gap / decision needed |
|---|---|---|
| SVM + LSTM deception classifier trained on FER-2013 + custom labelled data, probability 0-1 per segment | No trained deception model and no labelled interview data. V59 `fusion_v59` is a Bayesian log-likelihood-ratio combiner with per-person baselines and abstention; it outputs a *verification flag*, never a verdict | Do not present a deception probability as scientifically meaningful until person-independent labelled data exist. Agree wording with the supervisor |
| Target Deception AUC-ROC > 0.80 | Unmeasured. Earlier notes cite ~60% accuracy ceilings for cross-domain audio-visual benchmarks (unverified here) | Target may be unrealistic; plan an honest evaluation instead |
| FER accuracy > 75% | ResEmoteNet ONNX present; project memory records 82.5% on RAF-DB with a clean split (historical, not re-measured here); FER-2013 route reported 53.98% in the slide's own literature table | Domain shift to live webcam is unmeasured |
| Deepgram API streaming STT | Contradicts the offline mandate. Repo uses local faster-whisper/RealtimeSTT | Update the methodology text |
| ResNet-50 FER, LSTM vocal stress model | Not present; Voice uses pretrained SER models + rules | Document as future work |
| SHAP / LIME | Not implemented; explanations are the per-cue LLR contributions in the fusion output | Optional |
| React + FastAPI + MongoDB + WebRTC + WebSocket | None exists | Missing; not simulated |
| Monthly HR subscription, free tier of 3 interviews, email links | None exists | Missing |
| "End-to-end latency < 500 ms" | Unmeasured; Qwen on this CPU cannot meet it per turn (V58 uses a deterministic path to avoid per-turn LLM latency) | Reword as "per-answer analysis latency" and measure |
| Validation: 5-fold CV, 20% hold-out, ablation, SUS | `eval_harness_v59` implements subject-grouped CV, ablation, conformal coverage on SYNTHETIC data only | Needs real data collection with consent |
