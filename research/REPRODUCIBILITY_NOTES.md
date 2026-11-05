# Reproducibility
- Run `python main.py test` (unit + integration, no models needed) and `python main.py simulate`.
- Synthetic data are labelled synthetic; they check machinery, not validity.
- Model files are not in Git; record SHA-256 of each downloaded model before reporting results (not done yet).
- Pin versions: numpy<2, mediapipe==0.10.21, opencv-python==4.10.0.84 (from the vision team's notes; conflicts with the sandbox's newer mediapipe).
- No benchmark on the target hardware has been run; all timings quoted in older notes are historical.
