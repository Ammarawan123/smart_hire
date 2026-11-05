# Ethics and limitations
- Facial expression, gaze, blinking, posture, voice and wording are not proof of deception, honesty, personality, confidence or competence. Unvalidated outputs are experimental indicators.
- Consent: obtain explicit informed consent before recording audio/video; state purpose, retention and who reviews results.
- Retention: keep recordings and resumes out of Git (`outputs/`, `data/private/` are ignored); define a deletion period (the methodology's 90-day TTL is a design idea, not implemented).
- Bias: accents, non-native English, neurodivergence, disability, camera quality and culture affect every cue. No fairness evaluation has been run; `eval_harness_v59` has a fairness-table function tested only on synthetic data.
- Human review is required for any hiring decision. The system must not auto-reject.
- Privacy: the repo's real-resume fixtures are the author's own data; replace with synthetic fixtures before public release.
- EU AI Act/other regulation of emotion recognition in hiring was not researched in this session; check before any HR deployment.
